import torch
import json


def analyze_logits_and_dump(logits, batch_ids, model_type, step_offset=0):
    """
    计算 Entropy 和 Prob，并以 JSON 格式打印，方便后续 Python 读取。
    """
    with torch.no_grad():
        # Softmax 转概率
        probs = torch.softmax(logits, dim=-1)
        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        # Max Prob
        max_probs, _ = torch.max(probs, dim=-1)

        # 转为 CPU 列表
        entropy_list = entropy.tolist()
        probs_list = max_probs.tolist()
        batch_ids_list = (
            batch_ids.tolist() if isinstance(batch_ids, torch.Tensor) else batch_ids
        )

        # 打印日志 (带特殊前缀 DATA_LOG)
        # 格式: DATA_LOG | model_type | batch_id | token_index | entropy | prob
        for i, bid in enumerate(batch_ids_list):
            log_entry = {
                "type": model_type,
                "bid": bid,  # Batch ID，用于对齐 Draft 和 Target
                "idx": i + step_offset,  # Token 在当前 step 的位置
                "ent": round(entropy_list[i], 4),
                "prob": round(probs_list[i], 4),
            }
            print(f"DATA_LOG_V1:{json.dumps(log_entry)}", flush=True)
