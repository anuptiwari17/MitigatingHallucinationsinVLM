from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalConfig:
    """Central experiment configuration. Change 1 field to switch model, method, or hardware."""
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    benchmark: str = "hallusionbench"
    method: str = "greedy"                 # "greedy" | "vcd" (Phase 1) | "selective_gate" (Phase 1)
    dtype: str = "auto"                    # "auto" (fp16 on T4/P100, bf16 on L4/A100) | "fp16" | "bf16"
    attn_implementation: str = "sdpa"      # "sdpa" (fast baseline) | "eager" (required when log_attn=True)
    max_pixels: int = 512 * 28 * 28        # Safe cap for Kaggle 15GB T4 on 5K chart PNGs
    min_pixels: int = 256 * 28 * 28
    max_new_tokens: int = 64
    log_attn: bool = False                 # Set True in Phase 1 probing to record L0-L5 visual attention
    limit: Optional[int] = None            # Set e.g. 20 for quick smoke test, None for full 1,129 rows
    output_path: str = "results/qwen3_vl_4b_greedy.jsonl"
    prompt_suffix: str = (
        " Answer with 'Yes' or 'No' first, followed by a brief one-sentence explanation."
    )
