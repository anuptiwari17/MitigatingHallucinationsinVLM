import time
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from src.model_wrapper import QwenVLWrapper
from src.parser import parse_yes_no


def _token_entropy(logits: torch.Tensor) -> float:
    """Computes Shannon entropy (nats) of next-token probability distribution."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return float(-(probs * log_probs).sum(dim=-1).item())


@torch.inference_mode()
def run_greedy(
    wrapper: QwenVLWrapper,
    question: str,
    image: Optional[Any],
    max_new_tokens: int = 64,
    log_attn: bool = False,
) -> Dict[str, Any]:
    """
    Standard greedy baseline. Also logs first-token entropy (confidence), latency,
    and optionally L0-L5 visual-token attention mass for Phase 1 probing.
    """
    t0 = time.perf_counter()
    inputs = wrapper.build_inputs(question=question, image=image)
    prompt_len = inputs["input_ids"].shape[1]

    gen_out = wrapper.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=log_attn,
    )
    latency_sec = round(time.perf_counter() - t0, 4)

    new_tokens = gen_out.sequences[0][prompt_len:]
    raw_output = wrapper.processor.decode(new_tokens, skip_special_tokens=True).strip()
    pred_binary = parse_yes_no(raw_output)

    first_token_entropy = None
    if gen_out.scores and len(gen_out.scores) > 0:
        first_token_entropy = round(_token_entropy(gen_out.scores[0][0]), 4)

    visual_attn_l0_5 = None
    if log_attn and image is not None and gen_out.attentions:
        span = wrapper.find_vision_token_span(inputs["input_ids"])
        if span is not None:
            s, e = span
            # Average visual attention mass across layers 0..5 at first generated token
            step0_attns = gen_out.attentions[0]
            n_layers = min(6, len(step0_attns))
            vals = []
            for layer_idx in range(n_layers):
                # Shape: [batch=1, heads, q_len, kv_len] -> take last query token
                attn_last = step0_attns[layer_idx][0, :, -1, s:e]
                vals.append(float(attn_last.sum(dim=-1).mean().item()))
            visual_attn_l0_5 = round(sum(vals) / len(vals), 5)

    return {
        "raw_output": raw_output,
        "pred_binary": pred_binary,
        "first_token_entropy": first_token_entropy,
        "latency_sec": latency_sec,
        "visual_attn_l0_5": visual_attn_l0_5,
        "gate_triggered": False,
    }


def run_decoding(
    method: str,
    wrapper: QwenVLWrapper,
    question: str,
    image: Optional[Any],
    max_new_tokens: int = 64,
    log_attn: bool = False,
) -> Dict[str, Any]:
    """
    Pluggable decoding dispatcher.
    Add 'vcd' and 'selective_gate' right here in Phase 1 without touching run_eval.py!
    """
    m = method.lower().strip()
    if m == "greedy":
        return run_greedy(wrapper, question, image, max_new_tokens=max_new_tokens, log_attn=log_attn)
    if m == "vcd":
        raise NotImplementedError("Phase 1 Step 4: Implement always-on VCD decoding here.")
    if m == "selective_gate":
        raise NotImplementedError("Phase 1 Step 5: Implement Conflict & Collapse-Gated Steering here.")
    raise ValueError(f"Unknown decoding method: {method}")
