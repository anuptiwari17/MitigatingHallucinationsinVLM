# Hallusion_Project — Modular VLM Hallucination Evaluation & Mitigation

**Project:** Mitigating Visually Ungrounded Hallucinations in Vision-Language Models  
**Author:** Anup Tiwari (`anupt.it.24@nitj.ac.in`)  
**Supervisor:** Dr. Bharat Krishan Mahaur · Dept. of IT, Dr. B. R. Ambedkar NIT Jalandhar  

## Why This Repo Is Structured This Way
- **Portable across GPUs:** Auto-selects `fp16` on Kaggle T4/P100 and `bf16` on Colab L4/A100 or college GPUs.
- **Model-agnostic:** Swap `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, or `Qwen/Qwen3-VL-8B-Instruct` via `--model`.
- **Pluggable decoding:** Add Phase 1 `vcd` and `selective_gate` directly inside `src/decoding.py` without changing the runner or scorer.
- **Resume-safe:** Every prediction is flushed immediately to JSONL; re-running skips completed rows.

## Quick Commands
```bash
# 1. Verify parser and metric math (CPU, 1 second)
python tests/test_core.py

# 2. Quick 20-sample smoke test on GPU (~1 min)
python scripts/run_eval.py --model Qwen/Qwen3-VL-4B-Instruct --method greedy --limit 20 --output results/smoke.jsonl
python scripts/score_results.py --input results/smoke.jsonl

# 3. Full 1,129-question HallusionBench baseline (~20 min on Kaggle T4)
python scripts/run_eval.py --model Qwen/Qwen3-VL-4B-Instruct --method greedy --output results/qwen3_vl_4b_greedy.jsonl
python scripts/score_results.py --input results/qwen3_vl_4b_greedy.jsonl

# 4. Run a second model (zero code changes!)
python scripts/run_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --method greedy --output results/qwen25_vl_7b_greedy.jsonl
python scripts/score_results.py --input results/qwen25_vl_7b_greedy.jsonl
```
