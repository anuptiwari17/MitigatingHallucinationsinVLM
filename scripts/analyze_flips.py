"""
Local Flip Audit Script (0 GPU needed): Compares Greedy vs VCD vs Selective Gate on saved JSONL files.

Usage (from D:\\work\\mywork\\Hallusion_Project):
    python scripts/analyze_flips.py
  or explicitly:
    python scripts/analyze_flips.py \
        --greedy results/qwen3_vl_4b_greedy.jsonl \
        --vcd results/qwen3_vl_4b_vcd.jsonl \
        --gate results/qwen3_vl_4b_selective_gate_v1.jsonl
"""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List


def load_by_uid(path: str) -> Dict[str, Dict[str, Any]]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                data[r["uid"]] = r
    return data


def audit_pair(name: str, base: Dict[str, Dict[str, Any]], cand: Dict[str, Dict[str, Any]]):
    common = sorted(set(base.keys()) & set(cand.keys()))
    fixed, broken, same_right, same_wrong = [], [], 0, 0
    by_reason = defaultdict(lambda: {"triggered": 0, "flipped": 0, "fixed": 0, "broken": 0})
    by_subcat = defaultdict(lambda: {"fixed": 0, "broken": 0})

    for uid in common:
        b = base[uid]
        c = cand[uid]
        gt = int(b["gt_answer"])
        b_ok = int(b["pred_binary"]) == gt
        c_ok = int(c["pred_binary"]) == gt
        reason = str(c.get("gate_reason", "none"))
        subcat = f"{b['category']}/{b['subcategory']}"

        if c.get("gate_triggered"):
            by_reason[reason]["triggered"] += 1
        if int(b["pred_binary"]) != int(c["pred_binary"]):
            by_reason[reason]["flipped"] += 1

        if (not b_ok) and c_ok:
            fixed.append((b, c))
            by_reason[reason]["fixed"] += 1
            by_subcat[subcat]["fixed"] += 1
        elif b_ok and (not c_ok):
            broken.append((b, c))
            by_reason[reason]["broken"] += 1
            by_subcat[subcat]["broken"] += 1
        elif b_ok and c_ok:
            same_right += 1
        else:
            same_wrong += 1

    total_flips = len(fixed) + len(broken)
    win_rate = (100.0 * len(fixed) / total_flips) if total_flips else 0.0
    net = len(fixed) - len(broken)

    print("=" * 78)
    print(f"FLIP AUDIT: {name} vs. Greedy ({len(common)} common questions)")
    print("=" * 78)
    print(f"  Total Flips : {total_flips} | Fixed (Wrong->Right): {len(fixed)} | Broken (Right->Wrong): {len(broken)}")
    print(f"  Net Gain    : {net:+d} questions ({100.0 * net / len(common):+.2f}% aAcc) | Flip Win Rate: {win_rate:.1f}%")

    if by_reason:
        print("\n  Breakdown by Gate Alarm (gate_reason):")
        print(f"    {'Alarm':<26} {'Triggered':<11} {'Flipped':<9} {'Fixed':<7} {'Broken':<8} {'Net':<6} {'Win Rate'}")
        for r_name in sorted(by_reason.keys()):
            st = by_reason[r_name]
            fl = st["fixed"] + st["broken"]
            wr = (100.0 * st["fixed"] / fl) if fl else 0.0
            print(f"    {r_name:<26} {st['triggered']:<11} {st['flipped']:<9} {st['fixed']:<7} {st['broken']:<8} {st['fixed']-st['broken']:+d}    {wr:5.1f}%")

    print("\n  Net Change by Subcategory (Fixed - Broken):")
    for sc in sorted(by_subcat.keys()):
        st = by_subcat[sc]
        print(f"    {sc:<16}: Fixed={st['fixed']:2d}, Broken={st['broken']:2d}, Net={st['fixed']-st['broken']:+d}")

    if fixed:
        print("\n  Sample FIXED cases (Wrong -> Right):")
        for idx, (b, c) in enumerate(fixed[:5], 1):
            gt_s = "Yes" if int(b["gt_answer"]) == 1 else "No"
            print(f"    {idx}. [{b['category']}/{b['subcategory']}] reason={c.get('gate_reason')} | "
                  f"H_img={c.get('first_token_entropy')} H_blind={c.get('blind_entropy')} KL={c.get('kl_img_vs_text')} | GT={gt_s}")
            print(f"       Q: {b['question'].replace(chr(10), ' ')[:95]}...")

    if broken:
        print("\n  Sample BROKEN cases (Right -> Wrong):")
        for idx, (b, c) in enumerate(broken[:5], 1):
            gt_s = "Yes" if int(b["gt_answer"]) == 1 else "No"
            print(f"    {idx}. [{b['category']}/{b['subcategory']}] reason={c.get('gate_reason')} | "
                  f"H_img={c.get('first_token_entropy')} H_blind={c.get('blind_entropy')} KL={c.get('kl_img_vs_text')} | GT={gt_s}")
            print(f"       Q: {b['question'].replace(chr(10), ' ')[:95]}...")
    print()


def main():
    p = argparse.ArgumentParser(description="Audit exact Wrong->Right vs Right->Wrong flips across methods.")
    p.add_argument("--greedy", default="results/qwen3_vl_4b_greedy.jsonl")
    p.add_argument("--vcd", default="results/qwen3_vl_4b_vcd.jsonl")
    p.add_argument("--gate", default="results/qwen3_vl_4b_selective_gate_v1.jsonl")
    args = p.parse_args()

    greedy = load_by_uid(args.greedy)
    if args.vcd:
        audit_pair("Always-On VCD", greedy, load_by_uid(args.vcd))
    if args.gate:
        audit_pair("Selective Gate v1", greedy, load_by_uid(args.gate))


if __name__ == "__main__":
    main()