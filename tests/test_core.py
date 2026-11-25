import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image

from src.parser import parse_yes_no
from src.metrics import compute_hallusionbench_metrics
from src.decoding import add_diffusion_noise, apply_contrastive_logits, _kl_divergence, _token_entropy


def test_yes_no_parser():
    assert parse_yes_no("Yes, the chart shows 2019 is highest.") == 1
    assert parse_yes_no("**No.** There are six fingers.") == 0
    assert parse_yes_no("no") == 0
    assert parse_yes_no("The statement is incorrect.") == 0
    assert parse_yes_no("That is correct.") == 1
    assert parse_yes_no("") == -1


def test_hallusionbench_metrics():
    dummy = [
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "0", "question_id": "0", "gt_answer": "1", "pred_binary": 1, "latency_sec": 1.0, "first_token_entropy": 0.2, "gate_triggered": False},
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "0", "question_id": "1", "gt_answer": "0", "pred_binary": 0, "latency_sec": 1.2, "first_token_entropy": 0.3, "gate_triggered": True, "gate_reason": "entropy", "flipped_from_greedy": True},
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "1", "question_id": "0", "gt_answer": "0", "pred_binary": 0, "latency_sec": 1.0, "first_token_entropy": 0.2, "gate_triggered": False},
        {"category": "VD", "subcategory": "illusion", "visual_input": "1", "set_id": "0", "figure_id": "1", "question_id": "1", "gt_answer": "1", "pred_binary": 0, "latency_sec": 1.4, "first_token_entropy": 0.8, "gate_triggered": True, "gate_reason": "prior_dominance", "flipped_from_greedy": False},
    ]
    m = compute_hallusionbench_metrics(dummy)
    assert m["aAcc_overall"] == 75.0
    assert m["fAcc_figure_consistency"] == 50.0
    assert m["qAcc_question_pair"] == 50.0
    assert m["composite_avg"] == 58.33
    assert m["gate_trigger_rate_pct"] == 50.0
    assert m["flipped_from_greedy_count"] == 1
    assert m["unclear_predictions"] == 0


def test_vcd_and_gate_math():
    # 1. Diffusion noise returns valid RGB image of same size
    img = Image.new("RGB", (64, 64), color=(120, 180, 200))
    noisy = add_diffusion_noise(img, noise_step=500, seed=42)
    assert noisy.size == img.size and noisy.mode == "RGB"

    # 2. Contrastive logits boost visually-supported token and suppress prior-only token
    # Suppose token 0 ("No") is strongly preferred by blind prior, token 1 ("Yes") is close in clean logits
    clean = torch.tensor([[3.0, 2.8, -5.0]])
    blind = torch.tensor([[6.0, 0.0, -5.0]])
    cd = apply_contrastive_logits(clean, blind, alpha=1.0, beta=0.1)
    assert int(cd.argmax(dim=-1).item()) == 1  # Should flip from token 0 ("No") to token 1 ("Yes")!
    assert torch.isinf(cd[0, 2])               # Implausible token 2 should be masked to -inf

    # 3. KL divergence is ~0 when distributions match, >0 when they differ
    same_kl = _kl_divergence(clean[0], clean[0])
    diff_kl = _kl_divergence(clean[0], blind[0])
    assert same_kl < 1e-5
    assert diff_kl > 0.1
    assert _token_entropy(clean[0]) > 0.0


if __name__ == "__main__":
    test_yes_no_parser()
    test_hallusionbench_metrics()
    test_vcd_and_gate_math()
    print("ALL CORE UNIT TESTS PASSED!")
