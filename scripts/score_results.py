import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import compute_hallusionbench_metrics


def main():
    p = argparse.ArgumentParser(description="Score saved HallusionBench JSONL predictions (CPU only).")
    p.add_argument("--input", required=True, help="Path to predictions .jsonl file")
    p.add_argument("--save-summary", default=None, help="Optional path to save JSON summary")
    args = p.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    summary = compute_hallusionbench_metrics(records)
    print("\n" + "=" * 64)
    print(f"HALLUSIONBENCH REPORT: {os.path.basename(args.input)}")
    print("=" * 64)
    print(f"  Total Questions       : {summary['total_questions']} (Unclear: {summary['unclear_predictions']})")
    print(f"  aAcc (Overall)        : {summary['aAcc_overall']:.2f}%")
    print(f"  fAcc (Figure Consist.): {summary['fAcc_figure_consistency']:.2f}% ({summary['num_figures']} figures)")
    print(f"  qAcc (Question Pair)  : {summary['qAcc_question_pair']:.2f}% ({summary['num_question_pairs']} pairs)")
    print(f"  Avg Latency           : {summary['avg_latency_sec']} sec/question")
    print(f"  Avg Entropy (Correct) : {summary['avg_entropy_correct']} | (Wrong): {summary['avg_entropy_wrong']}")
    print("\n  By Category:")
    for k, v in summary["by_category"].items():
        print(f"    {k:10s}: {v['acc']:6.2f}% (n={v['n']})")
    print("\n  By Difficulty:")
    for k, v in summary["by_difficulty"].items():
        print(f"    {k:10s}: {v['acc']:6.2f}% (n={v['n']})")
    print("\n  By Subcategory:")
    for k, v in summary["by_subcategory"].items():
        print(f"    {k:22s}: {v['acc']:6.2f}% (n={v['n']})")
    print("=" * 64 + "\n")

    out_json = args.save_summary or args.input.replace(".jsonl", "_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to: {out_json}")


if __name__ == "__main__":
    main()
