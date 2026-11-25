"""
Step 1 Diagnostic Script: Analyze saved HallusionBench greedy baselines locally (0 GPU needed).

Usage (from your project root D:\\work\\mywork\\Hallusion_Project):
    python scripts/analyze_baselines.py
  or explicitly:
    python scripts/analyze_baselines.py results/qwen3_vl_4b_greedy.jsonl results/qwen25_vl_7b_greedy.jsonl results/qwen3_vl_8b_greedy.jsonl

Uses only Python's built-in standard library (no extra pip packages required).
"""

import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """
    Exact Mann-Whitney U AUROC (higher score = more likely label==1, i.e., error).
    Handles ties cleanly using average ranks.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    n = len(paired)
    while i < n:
        j = i
        while j + 1 < n and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed average rank
        pos_in_tie = sum(paired[k][1] for k in range(i, j + 1))
        rank_sum_pos += pos_in_tie * avg_rank
        i = j + 1

    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return u / (n_pos * n_neg)


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(max(int(round(q * (len(sorted_vals) - 1))), 0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def analyze_single_model(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = [1 if int(r["pred_binary"]) == int(r["gt_answer"]) else 0 for r in rows]
    errors = [1 - c for c in correct]

    # Bias breakdown: gt==1 is Yes, gt==0 is No
    tp = sum(1 for r in rows if int(r["gt_answer"]) == 1 and int(r["pred_binary"]) == 1)
    tn = sum(1 for r in rows if int(r["gt_answer"]) == 0 and int(r["pred_binary"]) == 0)
    fp = sum(1 for r in rows if int(r["gt_answer"]) == 0 and int(r["pred_binary"]) == 1)
    fn = sum(1 for r in rows if int(r["gt_answer"]) == 1 and int(r["pred_binary"]) == 0)
    gt_yes_pct = 100.0 * (tp + fn) / total if total else 0.0
    pred_yes_pct = 100.0 * (tp + fp) / total if total else 0.0

    # Entropy AUROC overall, image-only, VD, VS
    def _auroc_subset(subset_rows):
        sc = [float(r["first_token_entropy"]) for r in subset_rows if r.get("first_token_entropy") is not None]
        lb = [1 if int(r["pred_binary"]) != int(r["gt_answer"]) else 0 for r in subset_rows if r.get("first_token_entropy") is not None]
        return compute_auroc(sc, lb)

    auroc_all = _auroc_subset(rows)
    img_rows = [r for r in rows if str(r.get("visual_input", "1")) != "0"]
    vd_rows = [r for r in rows if str(r.get("category")) == "VD"]
    vs_rows = [r for r in rows if str(r.get("category")) == "VS"]
    auroc_img = _auroc_subset(img_rows)
    auroc_vd = _auroc_subset(vd_rows)
    auroc_vs = _auroc_subset(vs_rows)

    # Percentile sweep on image-only questions (since VCD / visual gate only applies when an image exists)
    img_with_ent = [
        (float(r["first_token_entropy"]), 1 if int(r["pred_binary"]) != int(r["gt_answer"]) else 0)
        for r in img_rows
        if r.get("first_token_entropy") is not None
    ]
    ent_sorted = sorted(x[0] for x in img_with_ent)
    total_img_errors = sum(x[1] for x in img_with_ent)

    sweep = []
    for top_pct in [10, 20, 30, 40, 50]:
        thresh = quantile(ent_sorted, 1.0 - top_pct / 100.0)
        flagged = [x for x in img_with_ent if x[0] >= thresh]
        flagged_errors = sum(x[1] for x in flagged)
        recall = (100.0 * flagged_errors / total_img_errors) if total_img_errors else 0.0
        precision = (100.0 * flagged_errors / len(flagged)) if flagged else 0.0
        sweep.append({
            "top_pct": top_pct,
            "threshold": thresh,
            "flagged_n": len(flagged),
            "error_recall_pct": recall,
            "error_precision_pct": precision,
        })

    # Confident-wrong count (bottom 50% entropy yet still wrong -> where L0-L5 visual attention alarm helps!)
    median_ent = quantile(ent_sorted, 0.50)
    confident_wrong = [
        r for r in img_rows
        if r.get("first_token_entropy") is not None
        and float(r["first_token_entropy"]) < median_ent
        and int(r["pred_binary"]) != int(r["gt_answer"])
    ]

    return {
        "name": name,
        "total": total,
        "acc": 100.0 * sum(correct) / total if total else 0.0,
        "num_errors": sum(errors),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "gt_yes_pct": gt_yes_pct,
        "pred_yes_pct": pred_yes_pct,
        "auroc_all": auroc_all,
        "auroc_img": auroc_img,
        "auroc_vd": auroc_vd,
        "auroc_vs": auroc_vs,
        "median_ent_img": median_ent,
        "confident_wrong_n": len(confident_wrong),
        "confident_wrong_pct_of_img_errors": (100.0 * len(confident_wrong) / total_img_errors) if total_img_errors else 0.0,
        "total_img_errors": total_img_errors,
        "sweep": sweep,
        "confident_wrong_examples": sorted(confident_wrong, key=lambda r: float(r["first_token_entropy"]))[:5],
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob(os.path.join("results", "*_greedy.jsonl")))
    if not paths:
        print("No *_greedy.jsonl files found in results/. Pass file paths as arguments.")
        sys.exit(1)

    loaded: List[Tuple[str, List[Dict[str, Any]]]] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARNING] File not found, skipping: {p}")
            continue
        name = os.path.basename(p).replace("_greedy.jsonl", "").replace(".jsonl", "")
        loaded.append((name, load_jsonl(p)))

    if not loaded:
        print("No valid JSONL files loaded.")
        sys.exit(1)

    print("=" * 78)
    print("STEP 1 BASELINE DIAGNOSTIC REPORT (ENTROPY, BIAS & ERROR CONCENTRATION)")
    print("=" * 78)

    summaries = []
    for name, rows in loaded:
        s = analyze_single_model(name, rows)
        summaries.append(s)
        print(f"\n--- {s['name']} (n={s['total']}, aAcc={s['acc']:.2f}%, Errors={s['num_errors']}) ---")
        print(f"  Yes/No Bias  : GT Yes={s['gt_yes_pct']:.1f}% | Model Pred Yes={s['pred_yes_pct']:.1f}%")
        print(f"  Error Split  : False Positives (said Yes, true No)={s['fp']} | False Negatives (said No, true Yes)={s['fn']}")
        print(f"  Entropy AUROC: All={s['auroc_all']:.4f} | Image-Only={s['auroc_img']:.4f} | VD={s['auroc_vd']:.4f} | VS={s['auroc_vs']:.4f}")
        print(f"  Confident-Wrong (image errors with entropy < median {s['median_ent_img']:.4f}): "
              f"{s['confident_wrong_n']} / {s['total_img_errors']} ({s['confident_wrong_pct_of_img_errors']:.1f}% of image errors)")
        print("  Entropy-Only Gate Sweep (Image-Only rows):")
        print(f"    {'Top % Flagged':<15} {'Threshold':<12} {'Flagged N':<11} {'Errors Caught':<16} {'Error Rate in Flagged'}")
        for sw in s["sweep"]:
            print(f"    Top {sw['top_pct']:<11} >= {sw['threshold']:<9.4f} {sw['flagged_n']:<11} "
                  f"{sw['error_recall_pct']:5.1f}%{'':<10} {sw['error_precision_pct']:5.1f}%")

    # Cross-model agreement if 2+ models loaded
    if len(loaded) >= 2:
        by_uid: Dict[str, Dict[str, int]] = defaultdict(dict)
        meta_by_uid: Dict[str, Dict[str, Any]] = {}
        for name, rows in loaded:
            for r in rows:
                uid = r["uid"]
                by_uid[uid][name] = 1 if int(r["pred_binary"]) == int(r["gt_answer"]) else 0
                meta_by_uid[uid] = r

        common_uids = [u for u, m in by_uid.items() if len(m) == len(loaded)]
        all_right = [u for u in common_uids if all(v == 1 for v in by_uid[u].values())]
        all_wrong = [u for u in common_uids if all(v == 0 for v in by_uid[u].values())]
        mixed = [u for u in common_uids if u not in all_right and u not in all_wrong]

        print("\n" + "=" * 78)
        print(f"CROSS-MODEL AGREEMENT ({len(loaded)} models, {len(common_uids)} common questions)")
        print("=" * 78)
        print(f"  All models CORRECT : {len(all_right):4d} ({100.0 * len(all_right) / len(common_uids):.1f}%)")
        print(f"  Mixed (disagree)   : {len(mixed):4d} ({100.0 * len(mixed) / len(common_uids):.1f}%)")
        print(f"  All models WRONG   : {len(all_wrong):4d} ({100.0 * len(all_wrong) / len(common_uids):.1f}%)")

        subcat_wrong = defaultdict(int)
        subcat_total = defaultdict(int)
        for u in common_uids:
            sc = f"{meta_by_uid[u]['category']}/{meta_by_uid[u]['subcategory']}"
            subcat_total[sc] += 1
            if u in all_wrong:
                subcat_wrong[sc] += 1

        print("\n  Subcategories with highest ALL-MODELS-WRONG rate:")
        ranked = sorted(subcat_total.keys(), key=lambda k: subcat_wrong[k] / subcat_total[k], reverse=True)
        for k in ranked:
            rate = 100.0 * subcat_wrong[k] / subcat_total[k]
            print(f"    {k:<18}: {subcat_wrong[k]:3d} / {subcat_total[k]:3d} ({rate:5.1f}%)")

    # Show 5 most confident-wrong examples from the strongest model (last loaded or highest acc)
    best = max(summaries, key=lambda x: x["acc"])
    print("\n" + "=" * 78)
    print(f"TOP 5 MOST CONFIDENT-WRONG IMAGE ERRORS ({best['name']})")
    print("(These are low-entropy visual-prior hallucinations where L0-L5 attention gating matters!)")
    print("=" * 78)
    for idx, r in enumerate(best["confident_wrong_examples"], 1):
        gt_str = "Yes" if int(r["gt_answer"]) == 1 else "No"
        pred_str = "Yes" if int(r["pred_binary"]) == 1 else "No"
        q_short = r["question"].replace("\n", " ")[:95]
        print(f"  {idx}. [{r['category']}/{r['subcategory']}] ent={r['first_token_entropy']:.4f} | GT={gt_str}, Pred={pred_str}")
        print(f"     Q: {q_short}...")


if __name__ == "__main__":
    main()
