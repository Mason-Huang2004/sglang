"""
Benchmark: Green Context SM partitioning decode latency.
Tests: Baseline (bs=128 full GPU) vs GreenCtx (2x bs=64 half SM each)
"""

import csv, math, os, sys

# ══════════════════════════════════════════════════════════════════════
# 关键修复：增大 FlashInfer workspace buffer
# 默认 2MB 在 bs=128 * decode_tokens=32 = 4096 tokens 时会 OOM
# 设为 512MB，足够覆盖所有配置
# 必须在 import sglang 之前设置！
# ══════════════════════════════════════════════════════════════════════
os.environ["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = str(512 * 1024 * 1024)

SGLANG_PATH = "/home/hhuang/code/sglang/python/"
MODEL = "/home/hhuang/.cache/huggingface/LLM-Research/llama_3_1"
TP_SIZE = 1
TOTAL_BS = 128
NUM_PARTITIONS = 2
DECODE_TOKENS = [1, 4, 8, 16, 32]
SEQ_LEN = 512
# SEQ_LEN = 1024
WARMUP_ITERS = 5
BENCH_ITERS = 5
MEM_FRACTION = 0.67
OUTPUT_CSV = "green_ctx_benchmark_results.csv"

if SGLANG_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PATH)

import torch


def create_mock_fb(model_runner, cgr, bs, seq_len, ndt, device):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )

    nt = bs * ndt
    fb = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        input_ids=torch.randint(0, 32000, (nt,), device=device, dtype=torch.int64),
        req_pool_indices=torch.arange(
            0,
            min(bs, model_runner.req_to_token_pool.size),
            device=device,
            dtype=torch.int32,
        ),
        seq_lens=torch.full((bs,), seq_len, device=device, dtype=torch.int32),
        seq_lens_cpu=torch.full((bs,), seq_len, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.zeros(nt, device=device, dtype=torch.int64),
        seq_lens_sum=bs * seq_len,
        positions=torch.arange(
            seq_len, seq_len + ndt, device=device, dtype=torch.int64
        ).repeat(bs),
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=model_runner.attn_backend,
        return_logprob=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        spec_algorithm=model_runner.spec_algorithm,
        spec_info=cgr.get_spec_info(nt),
    )
    return fb


def bench_baseline(mr, cgr, bs, seq_len, ndt, warmup, iters, dev):
    fb = create_mock_fb(mr, cgr, bs, seq_len, ndt, dev)
    can = cgr.can_run(fb)
    for _ in range(warmup):
        cgr.replay(fb) if can else mr.forward(fb)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        cgr.replay(fb) if can else mr.forward(fb)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_green(mr, cgr, total_bs, npart, seq_len, ndt, warmup, iters, dev):
    sub_bs = total_bs // npart
    pool = mr.req_to_token_pool.size
    subs = []
    for p in range(npart):
        fb = create_mock_fb(mr, cgr, sub_bs, seq_len, ndt, dev)
        off = (p * sub_bs) % pool
        fb.req_pool_indices = (
            torch.arange(sub_bs, device=dev, dtype=torch.int32) + off
        ) % pool
        subs.append(fb)
    for _ in range(warmup):
        cgr.replay_concurrent_green_context(subs)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        cgr.replay_concurrent_green_context(subs)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    print("=" * 60)
    print(" Green Context SM Partition Benchmark")
    print(
        f" All tests: Baseline bs={TOTAL_BS} vs GreenCtx 2x bs={TOTAL_BS//NUM_PARTITIONS}"
    )
    print("=" * 60)

    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # cuda_graph_bs = [1, 2, 4, 8, 16, 32, 64, 128]
    cuda_graph_bs = [64, 128]

    sa = ServerArgs(
        model_path=MODEL,
        tp_size=TP_SIZE,
        mem_fraction_static=MEM_FRACTION,
        trust_remote_code=True,
        cuda_graph_bs=cuda_graph_bs,
    )
    _set_envs_and_config(sa)
    pa = PortArgs.init_new(sa)
    mc = ModelConfig.from_server_args(sa)
    mr = ModelRunner(
        model_config=mc,
        mem_fraction_static=sa.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=sa.tp_size,
        moe_ep_rank=0,
        moe_ep_size=sa.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=pa.nccl_port,
        server_args=sa,
    )
    dev = mr.device

    results = []
    for ndt in DECODE_TOKENS:
        print(f"\n--- Decode Tokens = {ndt} ---")
        tbs = TOTAL_BS
        sbs = tbs // NUM_PARTITIONS
        print(f"  total_bs={tbs}, sub_bs={sbs}, total_tokens={tbs*ndt}")

        sa.cuda_graph_bs = cuda_graph_bs
        sa.speculative_algorithm = "ngram"
        sa.speculative_num_draft_tokens = ndt
        mr.attn_backend.num_draft_tokens = ndt
        if hasattr(SpeculativeAlgorithm, "from_string"):
            mr.spec_algorithm = SpeculativeAlgorithm.from_string("ngram")
        else:
            mr.spec_algorithm = SpeculativeAlgorithm(path=None, value="ngram")

        print(f"  [Capturing CUDA Graph...]")
        cgr = CudaGraphRunner(mr)
        mr.graph_runner = cgr

        # Baseline: bs=128, full GPU
        try:
            bms = bench_baseline(
                mr, cgr, tbs, SEQ_LEN, ndt, WARMUP_ITERS, BENCH_ITERS, dev
            )
            print(f"  Baseline (bs={tbs}):   {bms:.3f} ms")
        except Exception as e:
            print(f"  Baseline FAILED: {e}")
            bms = float("nan")

        # Green Context: 2x bs=64, half SM each
        gms = float("nan")
        try:
            cgr.init_green_context(num_partitions=NUM_PARTITIONS)
            gms = bench_green(
                mr,
                cgr,
                tbs,
                NUM_PARTITIONS,
                SEQ_LEN,
                ndt,
                WARMUP_ITERS,
                BENCH_ITERS,
                dev,
            )
            print(f"  GreenCtx (2x bs={sbs}): {gms:.3f} ms")
            if cgr.green_ctx_manager:
                cgr.green_ctx_manager.destroy()
                cgr.green_ctx_manager = None
                cgr.green_ctx_buffers.clear()
        except Exception as e:
            print(f"  GreenCtx FAILED: {e}")
            import traceback

            traceback.print_exc()

        spd = (
            bms / gms
            if not (math.isnan(bms) or math.isnan(gms) or gms <= 0)
            else float("nan")
        )
        results.append(
            {
                "decode_tokens": ndt,
                "total_bs": tbs,
                "sub_bs": sbs,
                "baseline_ms": bms,
                "green_ctx_ms": gms,
                "speedup": spd,
            }
        )
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(
        f"{'Tokens':>7} | {'TotalBS':>8} | {'SubBS':>6} | "
        f"{'Baseline':>10} | {'GreenCtx':>10} | {'Speedup':>8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['decode_tokens']:>7} | {r['total_bs']:>8} | {r['sub_bs']:>6} | "
            f"{r['baseline_ms']:>10.3f} | {r['green_ctx_ms']:>10.3f} | {r['speedup']:>8.3f}"
        )

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
