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


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Modular, resume-safe VLM hallucination benchmark runner.")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="Hugging Face model ID")
    p.add_argument("--benchmark", default="hallusionbench", help="Benchmark name (hallusionbench)")
    p.add_argument("--method", default="greedy", help="Decoding method: greedy | vcd | selective_gate")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"], help="Model precision")
    p.add_argument("--max-pixels", type=int, default=512 * 28 * 28, help="Max image pixels cap (prevents T4 OOM)")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max generated tokens per question")
    p.add_argument("--log-attn", action="store_true", help="Log L0-L5 visual attention (uses eager attn)")
    p.add_argument("--limit", type=int, default=None, help="Optional row cap for quick smoke test (e.g., 20)")
    p.add_argument("--output", required=True, help="Output JSONL path (auto-resumes if file exists)")
    args = p.parse_args()

    return EvalConfig(
        model_id=args.model,
        benchmark=args.benchmark,
        method=args.method,
        dtype=args.dtype,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        log_attn=args.log_attn,
        limit=args.limit,
        output_path=args.output,
    )


def main():
    cfg = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)

    examples = load_benchmark(cfg.benchmark, limit=cfg.limit)
    done_uids = load_completed_uids(cfg.output_path)
    remaining = [ex for ex in examples if ex.uid not in done_uids]

    print(f"Benchmark: {cfg.benchmark} | Total: {len(examples)} | Already done: {len(done_uids)} | Remaining: {len(remaining)}")
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

    with open(cfg.output_path, "a", encoding="utf-8") as fout:
        for ex in tqdm(remaining, desc=f"{cfg.model_id.split('/')[-1]} [{cfg.method}]"):
            full_prompt = ex.question + cfg.prompt_suffix
            out = run_decoding(
                method=cfg.method,
                wrapper=wrapper,
                question=full_prompt,
                image=ex.image,
                max_new_tokens=cfg.max_new_tokens,
                log_attn=cfg.log_attn,
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
