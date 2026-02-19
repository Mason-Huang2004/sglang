# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import bisect
import gc
import inspect
import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    graph_capture,
    set_pdmux_status,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.input_buffers import GraphInputBuffers
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, get_stream_groups
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    get_bool_env_var,
    is_hip,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

# Green Context support (CUDA 12.4+) for SM partitioning
try:
    from cuda.bindings import driver as cuda_drv

    GREEN_CONTEXT_AVAILABLE = True
except ImportError:
    try:
        from cuda import cuda as cuda_drv

        GREEN_CONTEXT_AVAILABLE = True
    except ImportError:
        cuda_drv = None
        GREEN_CONTEXT_AVAILABLE = False

_is_hip = is_hip()

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

# Detect whether the current forward pass is in capture mode
is_capture_mode = False


def get_is_capture_mode():
    return is_capture_mode


@contextmanager
def model_capture_mode():
    global is_capture_mode
    is_capture_mode = True

    yield

    is_capture_mode = False


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                ),
                dynamic=_is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

    monkey_patch_torch_compile()


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if max(capture_bs) > model_runner.req_to_token_pool.size:
        # In some cases (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [model_runner.req_to_token_pool.size]

    mul_base = 1

    if server_args.enable_two_batch_overlap:
        mul_base *= 2

    if require_gathered_buffer(server_args):
        mul_base *= get_attention_tp_size()

    capture_bs = [bs for bs in capture_bs if bs % mul_base == 0]

    capture_bs = [bs for bs in capture_bs if bs <= model_runner.req_to_token_pool.size]
    capture_bs = list(sorted(set(capture_bs)))
    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class GreenContextManager:
    """
    Manages CUDA Green Contexts for SM (Streaming Multiprocessor) partitioning.

    Green Contexts (CUDA 12.4+) split GPU SMs into isolated partitions,
    enabling concurrent CUDA graph execution on different SM groups.
    This allows, e.g., splitting bs=128 into 2x bs=64 requests each
    running on half the SMs concurrently.

    Requirements:
        - CUDA 12.4+ driver
        - cuda-python >= 12.4 (pip install cuda-python)

    Usage:
        manager = GreenContextManager(device_id=0, num_partitions=2)
        with manager.use_partition(0):
            # GPU ops here run on first half of SMs
            ...
        manager.destroy()
    """

    def __init__(self, device_id: int, num_partitions: int = 2):
        if not GREEN_CONTEXT_AVAILABLE:
            raise RuntimeError(
                "Green Context requires cuda-python package (>= 12.4). "
                "Install with: pip install cuda-python"
            )

        self.device_id = device_id
        self.num_partitions = num_partitions
        self.green_ctxs = []
        self.cuda_ctxs = []
        self.driver_streams = []
        self.torch_streams = []
        self.total_sms = 0
        self.sms_per_partition = 0
        self._initialized = False

        self._setup()

    @staticmethod
    def _check(result):
        """Check CUDA driver API result tuple and return value(s)."""
        if not isinstance(result, tuple):
            return result
        err = result[0]
        err_val = err.value if hasattr(err, "value") else int(err)
        if err_val != 0:
            raise RuntimeError(f"CUDA driver API error: {err} (code={err_val})")
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]

    # def _setup(self):
    #     """Initialize green contexts with SM partitioning."""
    #     drv = cuda_drv

    #     # Initialize CUDA driver (idempotent)
    #     self._check(drv.cuInit(0))

    #     # Get device handle
    #     device = self._check(drv.cuDeviceGet(self.device_id))
    #     self._device = device

    #     # Query total SM count
    #     sm_count = self._check(
    #         drv.cuDeviceGetAttribute(
    #             drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    #             device,
    #         )
    #     )
    #     self.total_sms = sm_count
    #     self.sms_per_partition = sm_count // self.num_partitions

    #     if self.sms_per_partition < 1:
    #         raise RuntimeError(
    #             f"Not enough SMs ({sm_count}) for {self.num_partitions} partitions"
    #         )

    #     logger.info(
    #         f"GreenContext: device={self.device_id}, total_sms={self.total_sms}, "
    #         f"partitions={self.num_partitions}, sms_per_partition={self.sms_per_partition}"
    #     )

    #     # Get device SM resource
    #     dev_resource = self._check(
    #         drv.cuDeviceGetDevResource(
    #             device, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    #         )
    #     )

    #     # Split SMs into equal partitions using cuDevSmResourceSplitByCount
    #     # API: cuDevSmResourceSplitByCount(nbGroups, input, flags, minCount)
    #     # Returns: (err, result_array, actual_count, remainder)
    #     #
    #     # On Compute 8.x (e.g. ADA 6000, 142 SMs), SM coscheduled alignment = 2.
    #     # min_count=71 (odd) would be rounded up to 72, 72*2=144>142 → only 1 group.
    #     # Solution: use discovery to detect, then try IGNORE_SM_COSCHEDULING flag
    #     # (flag=1) which removes alignment constraints (safe for compute < 9.0).
    #     min_count = self.sms_per_partition

    #     # Discovery call: pass nbGroups=0 to query how many groups are possible
    #     disc_result = self._check(
    #         drv.cuDevSmResourceSplitByCount(0, dev_resource, 0, min_count)
    #     )
    #     _, disc_nb_groups, _ = disc_result
    #     disc_nb_groups = (
    #         int(disc_nb_groups)
    #         if not isinstance(disc_nb_groups, int)
    #         else disc_nb_groups
    #     )

    #     if disc_nb_groups >= self.num_partitions:
    #         # Default alignment is fine
    #         split_flags = 0
    #     else:
    #         # Alignment prevents creating enough groups.
    #         # Use CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING (=1)
    #         # to treat each SM independently, bypassing coscheduled alignment.
    #         logger.info(
    #             f"GreenContext: default alignment gives {disc_nb_groups} groups, "
    #             f"need {self.num_partitions}. Using IGNORE_SM_COSCHEDULING flag."
    #         )
    #         split_flags = 1  # CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING

    #         # Re-discover with the flag to verify
    #         disc2 = self._check(
    #             drv.cuDevSmResourceSplitByCount(0, dev_resource, split_flags, min_count)
    #         )
    #         _, disc_nb_groups2, _ = disc2
    #         disc_nb_groups2 = (
    #             int(disc_nb_groups2)
    #             if not isinstance(disc_nb_groups2, int)
    #             else disc_nb_groups2
    #         )
    #         if disc_nb_groups2 < self.num_partitions:
    #             raise RuntimeError(
    #                 f"Cannot create {self.num_partitions} SM partitions even with "
    #                 f"IGNORE_SM_COSCHEDULING (max={disc_nb_groups2}, total_sms={self.total_sms})"
    #             )

    #     # Actual split
    #     split_result = self._check(
    #         drv.cuDevSmResourceSplitByCount(
    #             self.num_partitions, dev_resource, split_flags, min_count
    #         )
    #     )
    #     split_resources, nb_groups, _remainder = split_result

    #     actual_partitions = min(
    #         nb_groups if isinstance(nb_groups, int) else int(nb_groups),
    #         self.num_partitions,
    #     )

    #     # Create a green context for each partition
    #     for i in range(actual_partitions):
    #         resource = (
    #             split_resources[i]
    #             if isinstance(split_resources, (list, tuple))
    #             else split_resources
    #         )

    #         # Generate resource descriptor from the split resource
    #         desc = self._check(drv.cuDevResourceGenerateDesc([resource], 1))

    #         # Create green context with default stream support
    #         green_ctx = self._check(
    #             drv.cuGreenCtxCreate(
    #                 desc,
    #                 device,
    #                 drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
    #             )
    #         )
    #         self.green_ctxs.append(green_ctx)

    #         # Get a usable CUDA context from the green context
    #         cuda_ctx = self._check(drv.cuCtxFromGreenCtx(green_ctx))
    #         self.cuda_ctxs.append(cuda_ctx)

    #         # Create a stream within the green context
    #         # cuGreenCtxStreamCreate requires CU_STREAM_NON_BLOCKING (=1)
    #         stream = self._check(
    #             drv.cuGreenCtxStreamCreate(
    #                 green_ctx, 1, 0
    #             )  # flags=NON_BLOCKING, priority=0
    #         )
    #         self.driver_streams.append(stream)

    #         # Wrap as PyTorch stream for interop
    #         torch_stream = torch.cuda.ExternalStream(
    #             stream_ptr=int(stream),
    #             device=torch.device(f"cuda:{self.device_id}"),
    #         )
    #         self.torch_streams.append(torch_stream)

    #     self.num_partitions = actual_partitions
    #     self._initialized = True
    #     logger.info(
    #         f"GreenContext: successfully created {actual_partitions} partitions "
    #         f"({self.sms_per_partition} SMs each)"
    #     )

    def _setup(self):
        """Initialize green contexts with SM partitioning."""
        drv = cuda_drv

        # Initialize CUDA driver (idempotent)
        self._check(drv.cuInit(0))

        # Get device handle
        device = self._check(drv.cuDeviceGet(self.device_id))
        self._device = device

        # Query total SM count
        sm_count = self._check(
            drv.cuDeviceGetAttribute(
                drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device,
            )
        )
        self.total_sms = sm_count

        # 核心修改：遵循官方文档，Compute 8.x 必须是 2 的倍数
        # 强制向下对齐，避免底层自动向上取整导致总 SM 数量溢出 (例如 142/2=71 -> 72*2=144>142)
        alignment = 2
        raw_sms_per_partition = sm_count // self.num_partitions
        self.sms_per_partition = (raw_sms_per_partition // alignment) * alignment

        if self.sms_per_partition < alignment:
            raise RuntimeError(
                f"Not enough SMs ({sm_count}) for {self.num_partitions} partitions after alignment"
            )

        logger.info(
            f"GreenContext: device={self.device_id}, total_sms={self.total_sms}, "
            f"partitions={self.num_partitions}, sms_per_partition={self.sms_per_partition} (Aligned)"
        )

        # Get device SM resource
        dev_resource = self._check(
            drv.cuDeviceGetDevResource(
                device, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
        )

        min_count = self.sms_per_partition
        split_flags = 0  # 恢复默认 flag，因为我们已经手动对齐

        # Actual split
        split_result = self._check(
            drv.cuDevSmResourceSplitByCount(
                self.num_partitions, dev_resource, split_flags, min_count
            )
        )
        split_resources, nb_groups, _remainder = split_result

        actual_partitions = min(
            nb_groups if isinstance(nb_groups, int) else int(nb_groups),
            self.num_partitions,
        )

        if actual_partitions < self.num_partitions:
            raise RuntimeError(
                f"Hardware only allowed {actual_partitions} partitions, requested {self.num_partitions}."
            )

        # Create a green context for each partition
        for i in range(actual_partitions):
            resource = (
                split_resources[i]
                if isinstance(split_resources, (list, tuple))
                else split_resources
            )

            # Generate resource descriptor from the split resource
            desc = self._check(drv.cuDevResourceGenerateDesc([resource], 1))

            # Create green context with default stream support
            green_ctx = self._check(
                drv.cuGreenCtxCreate(
                    desc,
                    device,
                    drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
                )
            )
            self.green_ctxs.append(green_ctx)

            # Get a usable CUDA context from the green context
            cuda_ctx = self._check(drv.cuCtxFromGreenCtx(green_ctx))
            self.cuda_ctxs.append(cuda_ctx)

            # Create a stream within the green context
            stream = self._check(
                drv.cuGreenCtxStreamCreate(
                    green_ctx, 1, 0
                )  # flags=NON_BLOCKING, priority=0
            )
            self.driver_streams.append(stream)

            # Wrap as PyTorch stream for interop
            torch_stream = torch.cuda.ExternalStream(
                stream_ptr=int(stream),
                device=torch.device(f"cuda:{self.device_id}"),
            )
            self.torch_streams.append(torch_stream)

        self.num_partitions = actual_partitions
        self._initialized = True
        logger.info(
            f"GreenContext: successfully created {actual_partitions} partitions "
            f"({self.sms_per_partition} SMs each)"
        )

    @contextmanager
    def use_partition(self, partition_idx: int):
        """
        Context manager to run GPU operations within a specific SM partition.

        Args:
            partition_idx: Index of the partition (0 to num_partitions-1)

        Yields:
            torch.cuda.Stream: The stream bound to this partition's SMs
        """
        assert (
            0 <= partition_idx < len(self.green_ctxs)
        ), f"partition_idx {partition_idx} out of range [0, {len(self.green_ctxs)})"

        drv = cuda_drv
        self._check(drv.cuCtxPushCurrent(self.cuda_ctxs[partition_idx]))
        try:
            with torch.cuda.stream(self.torch_streams[partition_idx]):
                yield self.torch_streams[partition_idx]
        finally:
            drv.cuCtxPopCurrent()  # don't raise in finally

    def get_torch_stream(self, partition_idx: int) -> torch.cuda.Stream:
        """Get the PyTorch stream for a specific partition."""
        return self.torch_streams[partition_idx]

    def synchronize_all(self):
        """Synchronize all partition streams."""
        for stream in self.torch_streams:
            stream.synchronize()

    def synchronize_partition(self, partition_idx: int):
        """Synchronize a specific partition's stream."""
        self.torch_streams[partition_idx].synchronize()

    def destroy(self):
        """Clean up all green contexts and streams."""
        if not self._initialized:
            return

        drv = cuda_drv
        for stream in self.driver_streams:
            try:
                drv.cuStreamDestroy(stream)
            except Exception:
                pass
        for green_ctx in self.green_ctxs:
            try:
                drv.cuGreenCtxDestroy(green_ctx)
            except Exception:
                pass

        self.green_ctxs.clear()
        self.cuda_ctxs.clear()
        self.driver_streams.clear()
        self.torch_streams.clear()
        self._initialized = False
        logger.info("GreenContext: destroyed all partitions")

    def __del__(self):
        self.destroy()


class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.enable_pdmux = model_runner.server_args.enable_pdmux

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm = self.dllm_config is not None

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        log_info_on_rank0(logger, f"Capture cuda graph bs {self.capture_bs}")
        if KTRANSFORMERS_AVAILABLE:
            KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if (
            model_runner.spec_algorithm.is_eagle()
            or model_runner.spec_algorithm.is_standalone()
            or model_runner.spec_algorithm.is_ngram()
        ):
            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                self.capture_forward_mode = ForwardMode.TARGET_VERIFY
                self.num_tokens_per_bs = (
                    self.model_runner.server_args.speculative_num_draft_tokens
                )
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND
            self.num_tokens_per_bs = self.dllm_config.block_size

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )

        # Init PDMux if needed
        self.maybe_init_pdmux()
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
            if self.dllm_config is None
            else self.dllm_config.block_size
        )

        self.encoder_len_fill_value = 0

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.server_args.enable_lora:
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_bs=self.num_tokens_per_bs,
            )

        enable_mamba_track = (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and self.model_runner.spec_algorithm.is_none()
        )

        if self.require_gathered_buffer:
            assert self.require_mlp_tp_gather or self.require_attn_tp_gather
        self.buffers: GraphInputBuffers = GraphInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            hidden_size=self.model_runner.model_config.hidden_size,
            vocab_size=self.model_runner.model_config.vocab_size,
            dtype=self.model_runner.model_config.dtype,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            is_encoder_decoder=self.is_encoder_decoder,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            seq_len_fill_value=self.seq_len_fill_value,
            encoder_len_fill_value=self.encoder_len_fill_value,
            num_tokens_per_bs=self.num_tokens_per_bs,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
        )

        self.tbo_plugin = TboCudaGraphRunnerPlugin()

        # Green Context support for SM partitioning
        self.green_ctx_manager: Optional[GreenContextManager] = None
        self.green_ctx_graphs: dict = {}  # {partition_idx: {bs: graph}}
        self.green_ctx_output_buffers: dict = {}  # {partition_idx: {bs: output}}
        self.green_ctx_buffers: dict = {}  # {partition_idx: GraphInputBuffers}

        # Speculative_inference
        if model_runner.spec_algorithm.is_eagle3():
            self.model_runner.model.set_eagle3_layers_to_capture()

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def maybe_init_pdmux(self):
        if self.enable_pdmux:
            self.stream_groups = get_stream_groups()
            for attn_backend in self.model_runner.decode_attn_backend_group:
                attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

    def _cache_loc_dtype(self):
        return torch.int64

    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size

        graph_key = cuda_graph_bs
        if self.enable_pdmux:
            graph_key = f"{get_current_stream_idx()}_{cuda_graph_bs}"

        is_bs_supported = (
            graph_key in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.num_tokens_per_bs
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and capture_hidden_mode_matches
            and is_ngram_supported
        )

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        torch.cuda.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.cuda.memory._dump_snapshot(f"cuda_graph_runner_memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by CUDA Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=10
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
            + "\n\nMemory Usage is saved to cuda_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)

    def capture(self) -> None:
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()

        def _capture_one_stream(stream_idx: Optional[int] = None):
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            for i, bs in enumerate(capture_range):
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                    )

                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward, stream_idx)
                    # For pd_multiplexing, we need to save the graph and output buffers
                    key = bs if stream_idx is None else f"{stream_idx}_{bs}"
                    self.graphs[key] = graph
                    self.output_buffers[key] = output_buffers

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with graph_capture() as graph_capture_context, profile_context as prof:
                    self.stream = graph_capture_context.stream
                    _capture_one_stream()
            else:
                set_pdmux_status(False)
                for i, sg in enumerate(self.stream_groups):
                    with graph_capture(
                        stream=sg[1]
                    ) as graph_capture_context, profile_context as prof:
                        self.stream = graph_capture_context.stream
                        _capture_one_stream(i)

        if self.enable_profile_cuda_graph:
            self._post_process_after_profile(prof)

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.model_runner.server_args.enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        graph_fn = (
            partial(memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)
            if memory_saver_adapter.enabled
            else self.device_module.graph
        )
        with graph_fn(cuda_graph=graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _create_device_graph(self):
        return torch.cuda.CUDAGraph()

    def capture_one_batch_size(
        self, bs: int, forward: Callable, stream_idx: Optional[int] = None
    ):
        buffers: GraphInputBuffers = self.buffers
        graph = self._create_device_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        if self.is_encoder_decoder:
            encoder_lens = buffers.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]
        buffers.num_token_non_padded[...] = num_tokens

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        # mamba state tracking
        mamba_track_indices = (
            buffers.mamba_track_indices[:bs]
            if buffers.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            buffers.mamba_track_mask[:bs]
            if buffers.mamba_track_mask is not None
            else None
        )

        if stream_idx is None:
            attn_backend = self.model_runner.attn_backend
        else:
            assert self.enable_pdmux
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=None,  # Prefill only
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
        )
        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        # Attention backend
        attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )

            logits_output_or_pp_proxy_tensors = forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        self.deepep_adapter.capture(is_extend_in_batch=False)

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        return graph, out

    def recapture_if_needed(self, forward_batch: ForwardBatch):

        # If the required capture_hidden_mode changes, we need to recapture the graph

        # These are the different factors that can influence the capture_hidden_mode
        capture_hidden_mode_required_by_forward_batch = (
            forward_batch.capture_hidden_mode
        )
        capture_hidden_mode_required_by_spec_info = (
            getattr(forward_batch.spec_info, "capture_hidden_mode", None)
            or CaptureHiddenMode.NULL
        )
        capture_hidden_mode_required_for_returning_hidden_states = (
            CaptureHiddenMode.FULL
            if self.model_runner.server_args.enable_return_hidden_states
            else CaptureHiddenMode.NULL
        )

        # Determine the highest capture_hidden_mode required
        # (If we have FULL, we can emulate LAST or NULL)
        # (If we have LAST, we can emulate NULL)
        required_capture_hidden_mode = max(
            capture_hidden_mode_required_by_forward_batch,
            capture_hidden_mode_required_by_spec_info,
            capture_hidden_mode_required_for_returning_hidden_states,
        )

        # If the current hidden mode is no longer aligned with the required hidden mode, we need to set it to what is required and re-capture
        if self.capture_hidden_mode != required_capture_hidden_mode:
            self.capture_hidden_mode = required_capture_hidden_mode
            self.capture()

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        buffers = self.buffers
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens / self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]

        seq_lens_cpu = buffers.populate_from_forward_batch(
            forward_batch=forward_batch,
            raw_bs=raw_bs,
            raw_num_token=raw_num_token,
            bs=bs,
            seq_len_fill_value=self.seq_len_fill_value,
            require_gathered_buffer=self.require_gathered_buffer,
            num_tokens_per_bs=self.num_tokens_per_bs,
            nsa_enable_prefill_cp=self.nsa_enable_prefill_cp,
            enable_num_token_non_padded_flag=enable_num_token_non_padded(
                self.model_runner.server_args
            ),
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.enable_two_batch_overlap:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if forward_batch.forward_mode.is_idle() and forward_batch.spec_info is not None:
            forward_batch.spec_info.custom_mask = buffers.custom_mask
        # Attention backend
        if self.enable_pdmux:
            stream_idx = get_current_stream_idx()
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]
        else:
            attn_backend = self.model_runner.attn_backend
        attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            buffers.req_pool_indices[:bs],
            buffers.seq_lens[:bs],
            forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
            buffers.encoder_lens[:bs] if self.is_encoder_decoder else None,
            self.capture_forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        self.deepep_adapter.replay()

        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        if self.enable_pdmux:
            graph_key = f"{get_current_stream_idx()}_{self.bs}"
        else:
            graph_key = self.bs
        self.graphs[graph_key].replay()
        output = self.output_buffers[graph_key]

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = output.full_logits[: self.raw_num_token]
            else:
                full_logits = None
                next_token_logits = output.next_token_logits[: self.raw_num_token]

            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.buffers.custom_mask,
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )

        elif self.model_runner.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                tree_mask=self.buffers.custom_mask,
                positions=None,
                retrive_index=None,
                retrive_next_token=None,
                retrive_next_sibling=None,
                draft_token_num=self.num_tokens_per_bs,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info

    # ── Green Context Methods ───────────────────────────────────────────

    def init_green_context(self, num_partitions: int = 2):
        """
        Initialize green contexts for SM partitioning and capture graphs
        in each partition.

        After calling this, use `replay_green_context()` to run a batch on
        a specific SM partition, or `replay_concurrent_green_context()` to
        split a batch across partitions and run concurrently.

        Args:
            num_partitions: Number of SM partitions (default 2 for 50/50 split)
        """
        self.green_ctx_manager = GreenContextManager(
            device_id=self.model_runner.gpu_id,
            num_partitions=num_partitions,
        )

        # Determine batch sizes for each partition
        partition_max_bs = self.max_bs // num_partitions
        partition_max_num_token = partition_max_bs * self.num_tokens_per_bs
        partition_capture_bs = [bs for bs in self.capture_bs if bs <= partition_max_bs]
        if not partition_capture_bs:
            partition_capture_bs = [partition_max_bs]

        enable_mamba_track = (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and self.model_runner.spec_algorithm.is_none()
        )

        # Create separate buffers and graphs for each partition
        for part_idx in range(num_partitions):
            # Each partition needs its own buffers for concurrent execution
            part_buffers = GraphInputBuffers.create(
                device=self.device,
                max_bs=partition_max_bs,
                max_num_token=partition_max_num_token,
                hidden_size=self.model_runner.model_config.hidden_size,
                vocab_size=self.model_runner.model_config.vocab_size,
                dtype=self.model_runner.model_config.dtype,
                dp_size=self.dp_size,
                pp_size=self.pp_size,
                is_encoder_decoder=self.is_encoder_decoder,
                require_mlp_tp_gather=self.require_mlp_tp_gather,
                seq_len_fill_value=self.seq_len_fill_value,
                encoder_len_fill_value=self.encoder_len_fill_value,
                num_tokens_per_bs=self.num_tokens_per_bs,
                cache_loc_dtype=self._cache_loc_dtype(),
                enable_mamba_track=enable_mamba_track,
            )
            self.green_ctx_buffers[part_idx] = part_buffers
            self.green_ctx_graphs[part_idx] = {}
            self.green_ctx_output_buffers[part_idx] = {}

        # Capture CUDA graphs in each green context partition
        self._capture_green_context_graphs(partition_capture_bs)

        logger.info(
            f"GreenContext: captured graphs for bs={partition_capture_bs} "
            f"in {num_partitions} partitions"
        )

    def _capture_green_context_graphs(self, partition_capture_bs: list):
        """
        Capture CUDA graphs within each green context partition.
        Each partition gets its own set of graphs using its own buffers.
        """
        orig_buffers = self.buffers
        orig_max_bs = self.max_bs
        orig_max_num_token = self.max_num_token

        for part_idx in range(self.green_ctx_manager.num_partitions):
            # 核心修复 1：重置全局图内存池！
            # 避免子上下文盗用主上下文内存池导致非法寻址（Illegal Memory Access）
            set_global_graph_memory_pool(None)

            part_buffers = self.green_ctx_buffers[part_idx]
            part_max_bs = max(partition_capture_bs)
            part_max_num_token = part_max_bs * self.num_tokens_per_bs

            # Temporarily swap buffers so capture_one_batch_size uses
            # partition-specific buffers
            self.buffers = part_buffers
            self.max_bs = part_max_bs
            self.max_num_token = part_max_num_token

            # Init attention backend state for partition size
            self.model_runner.attn_backend.init_cuda_graph_state(
                part_max_bs, part_max_num_token
            )

            with self.green_ctx_manager.use_partition(part_idx) as gc_stream:
                with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
                    with graph_capture(stream=gc_stream) as graph_capture_context:
                        self.stream = graph_capture_context.stream
                        for bs in reversed(partition_capture_bs):
                            with patch_model(
                                self.model_runner.model,
                                bs in self.compile_bs,
                                num_tokens=bs * self.num_tokens_per_bs,
                                tp_group=self.model_runner.tp_group,
                            ) as forward:
                                graph, output = self.capture_one_batch_size(bs, forward)
                                self.green_ctx_graphs[part_idx][bs] = graph
                                self.green_ctx_output_buffers[part_idx][bs] = output

            # 核心修复 2：捕获完毕后清空 PyTorch 缓存碎片，防止上下文中毒
            torch.cuda.empty_cache()

        # 核心修复 3：离开 Green Context，再次重置池子给 Primary Context 用
        set_global_graph_memory_pool(None)

        # Restore original state
        self.buffers = orig_buffers
        self.max_bs = orig_max_bs
        self.max_num_token = orig_max_num_token
        self.model_runner.attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )

    def replay_green_context(
        self,
        partition_idx: int,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """
        Replay a CUDA graph on a specific green context SM partition.

        Args:
            partition_idx: Which SM partition to run on
            forward_batch: The forward batch (should be sized for the partition)

        Returns:
            LogitsProcessorOutput with results
        """
        assert self.green_ctx_manager is not None, "Call init_green_context() first"

        part_buffers = self.green_ctx_buffers[partition_idx]
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Find the right batch size bucket
        part_bs_list = sorted(self.green_ctx_graphs[partition_idx].keys())
        index = bisect.bisect_left(part_bs_list, raw_bs)
        if index >= len(part_bs_list):
            raise RuntimeError(
                f"Batch size {raw_bs} too large for partition {partition_idx} "
                f"(max={part_bs_list[-1] if part_bs_list else 'none'})"
            )
        bs = part_bs_list[index]
        num_tokens = bs * self.num_tokens_per_bs

        # Populate partition buffers from forward_batch
        part_buffers.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        part_buffers.positions[:raw_num_token].copy_(forward_batch.positions)
        part_buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        part_buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        part_buffers.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        if raw_bs < bs:
            part_buffers.seq_lens[raw_bs:bs].fill_(self.seq_len_fill_value)

        # Init attention metadata for this partition's replay
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            part_buffers.req_pool_indices[:bs],
            part_buffers.seq_lens[:bs],
            forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
            (part_buffers.encoder_lens[:bs] if self.is_encoder_decoder else None),
            self.capture_forward_mode,
            forward_batch.spec_info,
        )

        # Replay within the green context's SM partition
        with self.green_ctx_manager.use_partition(partition_idx):
            self.green_ctx_graphs[partition_idx][bs].replay()

        output = self.green_ctx_output_buffers[partition_idx][bs]
        if isinstance(output, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                next_token_logits=output.next_token_logits[:raw_num_token],
                full_logits=None,
                hidden_states=(
                    output.hidden_states[:raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        return output

    def replay_concurrent_green_context(
        self,
        forward_batches: list,
    ) -> list:
        """
        Replay CUDA graphs concurrently on different SM partitions.

        Each forward_batch in the list is assigned to a partition and
        executed concurrently using different SM groups.

        Example: Split bs=128 into 2x bs=64:
            sub_batch_0 = ...  # first 64 requests
            sub_batch_1 = ...  # last 64 requests
            results = runner.replay_concurrent_green_context(
                [sub_batch_0, sub_batch_1]
            )

        Args:
            forward_batches: List of ForwardBatch, one per partition

        Returns:
            List of LogitsProcessorOutput, one per partition
        """
        assert self.green_ctx_manager is not None, "Call init_green_context() first"
        assert len(forward_batches) <= self.green_ctx_manager.num_partitions, (
            f"Too many batches ({len(forward_batches)}) for "
            f"{self.green_ctx_manager.num_partitions} partitions"
        )

        outputs = [None] * len(forward_batches)

        # Prepare all partitions (populate buffers, init attn metadata)
        prepared_bs = []
        for part_idx, fb in enumerate(forward_batches):
            part_buffers = self.green_ctx_buffers[part_idx]
            raw_bs = fb.batch_size
            raw_num_token = raw_bs * self.num_tokens_per_bs

            part_bs_list = sorted(self.green_ctx_graphs[part_idx].keys())
            index = bisect.bisect_left(part_bs_list, raw_bs)
            bs = part_bs_list[index]

            part_buffers.input_ids[:raw_num_token].copy_(fb.input_ids)
            part_buffers.positions[:raw_num_token].copy_(fb.positions)
            part_buffers.req_pool_indices[:raw_bs].copy_(fb.req_pool_indices)
            part_buffers.seq_lens[:raw_bs].copy_(fb.seq_lens)
            part_buffers.out_cache_loc[:raw_num_token].copy_(fb.out_cache_loc)
            if raw_bs < bs:
                part_buffers.seq_lens[raw_bs:bs].fill_(self.seq_len_fill_value)

            prepared_bs.append((bs, raw_bs, raw_num_token))

        # Launch all graph replays concurrently (each on its own SM partition)
        for part_idx, fb in enumerate(forward_batches):
            bs, raw_bs, raw_num_token = prepared_bs[part_idx]
            with self.green_ctx_manager.use_partition(part_idx):
                self.green_ctx_graphs[part_idx][bs].replay()

        # Synchronize all partitions
        self.green_ctx_manager.synchronize_all()

        # Collect outputs
        for part_idx in range(len(forward_batches)):
            bs, raw_bs, raw_num_token = prepared_bs[part_idx]
            output = self.green_ctx_output_buffers[part_idx][bs]
            if isinstance(output, LogitsProcessorOutput):
                outputs[part_idx] = LogitsProcessorOutput(
                    next_token_logits=(
                        output.next_token_logits[:raw_num_token].clone()
                    ),
                    full_logits=None,
                    hidden_states=(
                        output.hidden_states[:raw_num_token].clone()
                        if output.hidden_states is not None
                        else None
                    ),
                )
            else:
                outputs[part_idx] = output

        return outputs


CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs to a smaller value (e.g., 16)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)


class DeepEPCudaGraphRunnerAdapter:
    def __init__(self):
        # Record DeepEP mode used during capture to ensure replay consistency
        self._captured_deepep_mode = None

    def capture(self, is_extend_in_batch: bool):
        if not get_moe_a2a_backend().is_deepep():
            return
        self._captured_deepep_mode = get_deepep_mode().resolve(
            is_extend_in_batch=is_extend_in_batch
        )
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)

    def replay(self):
        if not get_moe_a2a_backend().is_deepep():
            return
        assert self._captured_deepep_mode is not None
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)
