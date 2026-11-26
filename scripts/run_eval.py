import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import EvalConfig
from src.datasets_loader import load_benchmark
from src.model_wrapper import QwenVLWrapper
from src.decoding import run_decoding


def load_completed_uids(jsonl_path: str) -> set:
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:
                    pass
    return done


def parse_args():
    p = argparse.ArgumentParser(description="Modular, resume-safe VLM hallucination benchmark runner.")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="Hugging Face model ID")
    p.add_argument("--benchmark", default="hallusionbench", help="Benchmark name (hallusionbench | pope_adv | pope_popular | pope_random)")
    p.add_argument("--method", default="greedy", choices=["greedy", "vcd", "selective_gate", "selective_zoom"], help="Decoding method")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"], help="Model precision")
    p.add_argument("--max-pixels", type=int, default=512 * 28 * 28, help="Base max image pixels cap")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max generated tokens per question")
    p.add_argument("--log-attn", action="store_true", help="Log L0-L5 visual attention (uses eager attn)")
    p.add_argument("--limit", type=int, default=None, help="Optional row cap for quick smoke/dev test")
    p.add_argument("--output", required=True, help="Output JSONL path (auto-resumes if file exists)")

    # 2-GPU parallel sharding
    p.add_argument("--shard-idx", type=int, default=0, help="Shard index (0..num_shards-1)")
    p.add_argument("--num-shards", type=int, default=1, help="Total parallel shards")

    # VCD & Selective Gate v2/v3 knobs
    p.add_argument("--cd-alpha", type=float, default=1.0, help="VCD / v2 contrastive weight alpha")
    p.add_argument("--cd-beta", type=float, default=0.1, help="Adaptive plausibility cutoff beta")
    p.add_argument("--vcd-noise-step", type=int, default=500, help="Gaussian diffusion noise step (0..999)")
    p.add_argument("--contrast-steps", type=int, default=4, help="Decision-window tokens steered")
    p.add_argument("--contrast-source", default="text", choices=["text", "noisy"], help="v2 contrast source")
    p.add_argument("--tau-ent", type=float, default=0.15, help="Image entropy gate threshold")
    p.add_argument("--tau-blind-ent", type=float, default=0.15, help="(Unused in v2/v3; kept for CLI compatibility)")
    p.add_argument("--tau-kl", type=float, default=0.10, help="(Unused in v2/v3; kept for CLI compatibility)")
    p.add_argument("--tau-attn", type=float, default=None, help="Optional L0-L5 visual attn threshold")
    p.add_argument("--alpha-prior", type=float, default=1.2, help="(Kept for CLI compatibility)")
    p.add_argument("--zoom-max-pixels", type=int, default=1280 * 28 * 28, help="v3 high-res visual zoom cap")
    p.add_argument("--zoom-alpha", type=float, default=0.30, help="v3 normalized log-prob prior calibration weight")

    args = p.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"Invalid shard settings: shard_idx={args.shard_idx}, num_shards={args.num_shards}")

    cfg = EvalConfig(
        model_id=args.model,
        benchmark=args.benchmark,
        method=args.method,
        dtype=args.dtype,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        log_attn=args.log_attn,
        limit=args.limit,
        output_path=args.output,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        vcd_noise_step=args.vcd_noise_step,
        contrast_steps=args.contrast_steps,
        contrast_source=args.contrast_source,
        tau_ent=args.tau_ent,
        tau_blind_ent=args.tau_blind_ent,
        tau_kl=args.tau_kl,
        tau_attn=args.tau_attn,
        alpha_prior=args.alpha_prior,
    )
    return cfg, args.zoom_max_pixels, args.zoom_alpha


def main():
    cfg, zoom_max_pixels, zoom_alpha = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)

    examples = load_benchmark(cfg.benchmark, limit=cfg.limit)
    if cfg.num_shards > 1:
        examples = [ex for idx, ex in enumerate(examples) if idx % cfg.num_shards == cfg.shard_idx]

    done_uids = load_completed_uids(cfg.output_path)
    remaining = [ex for ex in examples if ex.uid not in done_uids]

    shard_tag = f" [shard {cfg.shard_idx}/{cfg.num_shards}]" if cfg.num_shards > 1 else ""
    print(f"Benchmark: {cfg.benchmark}{shard_tag} | Method: {cfg.method} | Total: {len(examples)} | Already done: {len(done_uids)} | Remaining: {len(remaining)}")
    if not remaining:
        print("All examples already completed! Run scripts/score_results.py to view metrics.")
        return

    wrapper = QwenVLWrapper(
        model_id=cfg.model_id,
        dtype=cfg.dtype,
        attn_implementation=cfg.attn_implementation,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
        log_attn=cfg.log_attn,
    )

    decoding_kwargs = dict(
        cd_alpha=cfg.cd_alpha,
        cd_beta=cfg.cd_beta,
        vcd_noise_step=cfg.vcd_noise_step,
        contrast_steps=cfg.contrast_steps,
        contrast_source=cfg.contrast_source,
        tau_ent=cfg.tau_ent,
        tau_attn=cfg.tau_attn,
        zoom_max_pixels=zoom_max_pixels,
        zoom_alpha=zoom_alpha,
    )

    with open(cfg.output_path, "a", encoding="utf-8") as fout:
        for ex in tqdm(remaining, desc=f"{cfg.model_id.split('/')[-1]} [{cfg.method}]{shard_tag}"):
            full_prompt = ex.question + cfg.prompt_suffix
            out = run_decoding(
                method=cfg.method,
                wrapper=wrapper,
                question=full_prompt,
                image=ex.image,
                max_new_tokens=cfg.max_new_tokens,
                log_attn=cfg.log_attn,
                **decoding_kwargs,
            )
            record = {
                "uid": ex.uid,
                "model_id": cfg.model_id,
                "method": cfg.method,
                "benchmark": ex.benchmark,
                "split": ex.split,
                "category": ex.category,
                "subcategory": ex.subcategory,
                "visual_input": ex.visual_input,
                "set_id": ex.set_id,
                "figure_id": ex.figure_id,
                "question_id": ex.question_id,
                "question": ex.question,
                "gt_answer": ex.gt_answer,
                **out,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Done! Saved predictions to {cfg.output_path}")


if __name__ == "__main__":
    main()
