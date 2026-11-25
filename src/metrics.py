from collections import defaultdict
from typing import List, Dict, Any


def compute_hallusionbench_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes official HallusionBench metrics directly from saved JSONL prediction records:
      - aAcc: Overall per-question accuracy
      - fAcc: Per-figure consistency accuracy (ALL questions on a figure must be correct)
      - qAcc: Per-question-pair accuracy (same question across control figures must ALL be correct)
      - composite_avg: (aAcc + fAcc + qAcc) / 3
      - VD vs VS breakdown, Easy vs Hard breakdown, Subcategory breakdown, Latency, Entropy & Gate stats.
    """
    total = len(records)
    if total == 0:
        return {"error": "No records found."}

    correct_flags = []
    unclear_count = 0

    fig_groups = defaultdict(list)       # (category, subcategory, set_id, figure_id)
    qpair_groups = defaultdict(list)     # (category, subcategory, set_id, question_id)
    cat_groups = defaultdict(list)       # VD / VS
    subcat_groups = defaultdict(list)    # chart, illusion, map, math, etc.
    diff_groups = defaultdict(list)      # easy vs hard (based on visual_input / figure_id)

    latencies = []
    ent_correct = []
    ent_wrong = []
    gate_triggered_count = 0
    flipped_count = 0
    gate_reasons = defaultdict(int)

    for r in records:
        gt = int(r["gt_answer"])
        pred = int(r["pred_binary"])
        if pred == -1:
            unclear_count += 1
        is_correct = 1 if (pred == gt) else 0
        correct_flags.append(is_correct)

        cat = str(r["category"])
        subcat = str(r["subcategory"])
        sid = str(r["set_id"])
        fid = str(r["figure_id"])
        qid = str(r["question_id"])
        vis = str(r["visual_input"])

        fig_groups[(cat, subcat, sid, fid)].append(is_correct)
        qpair_groups[(cat, subcat, sid, qid)].append(is_correct)
        cat_groups[cat].append(is_correct)
        subcat_groups[f"{cat}/{subcat}"].append(is_correct)

        if cat == "VD":
            diff = "Easy" if fid == "0" else "Hard"
        else:
            diff = "Easy" if vis == "0" else "Hard"
        diff_groups[diff].append(is_correct)

        if r.get("latency_sec") is not None:
            latencies.append(float(r["latency_sec"]))
        if r.get("first_token_entropy") is not None:
            if is_correct:
                ent_correct.append(float(r["first_token_entropy"]))
            else:
                ent_wrong.append(float(r["first_token_entropy"]))
        if r.get("gate_triggered"):
            gate_triggered_count += 1
            gate_reasons[str(r.get("gate_reason", "triggered"))] += 1
        if r.get("flipped_from_greedy"):
            flipped_count += 1

    a_acc = 100.0 * sum(correct_flags) / total
    f_acc = 100.0 * sum(1 for v in fig_groups.values() if all(v)) / len(fig_groups)
    q_acc = 100.0 * sum(1 for v in qpair_groups.values() if all(v)) / len(qpair_groups)
    composite_avg = (a_acc + f_acc + q_acc) / 3.0

    def _pct(vals):
        return round(100.0 * sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "total_questions": total,
        "unclear_predictions": unclear_count,
        "aAcc_overall": round(a_acc, 2),
        "fAcc_figure_consistency": round(f_acc, 2),
        "qAcc_question_pair": round(q_acc, 2),
        "composite_avg": round(composite_avg, 2),
        "num_figures": len(fig_groups),
        "num_question_pairs": len(qpair_groups),
        "by_category": {k: {"n": len(v), "acc": _pct(v)} for k, v in sorted(cat_groups.items())},
        "by_difficulty": {k: {"n": len(v), "acc": _pct(v)} for k, v in sorted(diff_groups.items())},
        "by_subcategory": {k: {"n": len(v), "acc": _pct(v)} for k, v in sorted(subcat_groups.items())},
        "avg_latency_sec": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "avg_entropy_correct": round(sum(ent_correct) / len(ent_correct), 4) if ent_correct else None,
        "avg_entropy_wrong": round(sum(ent_wrong) / len(ent_wrong), 4) if ent_wrong else None,
        "gate_trigger_rate_pct": round(100.0 * gate_triggered_count / total, 2),
        "gate_triggered_count": gate_triggered_count,
        "flipped_from_greedy_count": flipped_count,
        "gate_reason_counts": dict(sorted(gate_reasons.items())),
    }
