import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parser import parse_yes_no
from src.metrics import compute_hallusionbench_metrics


def test_yes_no_parser():
    assert parse_yes_no("Yes, the chart shows 2019 is highest.") == 1
    assert parse_yes_no("**No.** There are six fingers.") == 0
    assert parse_yes_no("no") == 0
    assert parse_yes_no("The statement is incorrect.") == 0
    assert parse_yes_no("That is correct.") == 1
    assert parse_yes_no("") == -1


def test_hallusionbench_metrics():
    dummy = [
        # Figure (VD, illusion, s0, f0): 2 questions, both correct -> figure correct
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "0", "question_id": "0", "gt_answer": "1", "pred_binary": 1, "latency_sec": 1.0, "first_token_entropy": 0.2},
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "0", "question_id": "1", "gt_answer": "0", "pred_binary": 0, "latency_sec": 1.2, "first_token_entropy": 0.3},
        # Figure (VD, illusion, s0, f1): 2 questions, 1 right + 1 wrong -> figure wrong, pair q0 right, pair q1 wrong
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "1", "question_id": "0", "gt_answer": "0", "pred_binary": 0, "latency_sec": 1.0, "first_token_entropy": 0.2},
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "1", "question_id": "1", "gt_answer": "1", "pred_binary": 0, "latency_sec": 1.4, "first_token_entropy": 0.8},
    ]
    m = compute_hallusionbench_metrics(dummy)
    assert m["aAcc_overall"] == 75.0
    assert m["fAcc_figure_consistency"] == 50.0
    assert m["qAcc_question_pair"] == 50.0
    assert m["unclear_predictions"] == 0


if __name__ == "__main__":
    test_yes_no_parser()
    test_hallusionbench_metrics()
    print("ALL CORE UNIT TESTS PASSED!")
