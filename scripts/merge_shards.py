"""
Merges parallel shard JSONL files (e.g., from 2x T4 GPU runs) into one clean deduplicated JSONL.

Usage:
    python scripts/merge_shards.py \
        --inputs /kaggle/working/qwen3_4b_vcd_s0.jsonl /kaggle/working/qwen3_4b_vcd_s1.jsonl \
        --output /kaggle/working/qwen3_vl_4b_vcd.jsonl
"""

import argparse
import json
import os


def main():
    p = argparse.ArgumentParser(description="Merge sharded JSONL prediction files cleanly.")
    p.add_argument("--inputs", nargs="+", required=True, help="List of shard .jsonl paths")
    p.add_argument("--output", required=True, help="Merged output .jsonl path")
    args = p.parse_args()

    by_uid = {}
    for path in args.inputs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing shard file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    by_uid[rec["uid"]] = rec

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for uid in sorted(by_uid.keys()):
            fout.write(json.dumps(by_uid[uid], ensure_ascii=False) + "\n")

    print(f"Merged {len(args.inputs)} shards -> {len(by_uid)} unique rows saved to {args.output}")


if __name__ == "__main__":
    main()
