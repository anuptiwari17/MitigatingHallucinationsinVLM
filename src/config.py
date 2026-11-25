from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalConfig:
    """Central experiment configuration. Change 1 field to switch model, method, or hardware."""
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    benchmark: str = "hallusionbench"      # "hallusionbench" | "pope"
    method: str = "greedy"                 # "greedy" | "vcd" | "selective_gate"
    dtype: str = "auto"                    # "auto" (fp16 on T4/P100, bf16 on L4/A100) | "fp16" | "bf16"
    attn_implementation: str = "sdpa"      # "sdpa" (fast) | "eager" (required when log_attn=True)
    max_pixels: int = 512 * 28 * 28        # Safe cap for Kaggle 15GB T4 on 5K chart PNGs
    min_pixels: int = 256 * 28 * 28
    max_new_tokens: int = 64
    log_attn: bool = False                 # Set True to record L0-L5 visual attention
    limit: Optional[int] = None            # Set e.g. 50 for quick smoke test, None for full dataset
    output_path: str = "results/qwen3_vl_4b_greedy.jsonl"
    prompt_suffix: str = (
        " Answer with 'Yes' or 'No' first, followed by a brief one-sentence explanation."
    )

    # Sharding for 2-GPU parallel execution (e.g., 2x T4 for 4B model)
    shard_idx: int = 0
    num_shards: int = 1

    # VCD & Selective Gate hyperparameters
    cd_alpha: float = 1.0                  # Contrastive subtraction strength
    cd_beta: float = 0.1                   # Adaptive plausibility cutoff (VCD CVPR 2024)
    vcd_noise_step: int = 500              # Diffusion noise step (0..999) for noisy-image VCD
    contrast_steps: int = 4                # Decision-window tokens steered by contrastive logits
    contrast_source: str = "text"          # For selective_gate: "text" (blind prior) | "noisy" (VCD image)

    # Selective Gate thresholds (calibrated per model on a non-HallusionBench split)
    tau_ent: float = 0.15                  # Alarm 1: image-pass first-token entropy threshold
    tau_blind_ent: float = 0.15            # Alarm 2a: blind text-only pass is confident
    tau_kl: float = 0.10                   # Alarm 2b: image barely changed blind text distribution
    tau_attn: Optional[float] = None       # Alarm 3 (optional, requires --log-attn): L0-L5 visual attn floor
    alpha_prior: float = 1.2               # Contrastive strength when prior-dominance alarm fires
