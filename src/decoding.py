import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from src.model_wrapper import QwenVLWrapper
from src.parser import parse_yes_no


def _token_entropy(logits: torch.Tensor) -> float:
    """Computes Shannon entropy (nats) of next-token distribution, safely ignoring -inf masked entries."""
    logits_f = logits.float()
    valid = torch.isfinite(logits_f)
    if not valid.any():
        return 0.0
    logits_v = logits_f[valid]
    probs = F.softmax(logits_v, dim=-1)
    log_probs = F.log_softmax(logits_v, dim=-1)
    return float(-(probs * log_probs).sum(dim=-1).item())


def _kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Computes D_KL(P || Q) in nats between two 1D next-token logit vectors."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1).item()
    return float(max(kl, 0.0))


def add_diffusion_noise(image: Image.Image, noise_step: int = 500, seed: int = 42) -> Image.Image:
    """
    Standard VCD (Leng et al., CVPR 2024) forward Gaussian diffusion noise on a PIL image.
    Uses linear beta schedule [1e-4, 0.02] over 1000 steps.
    """
    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [-1, 1]

    num_steps = 1000
    step = max(0, min(int(noise_step), num_steps - 1))
    betas = np.linspace(1e-4, 0.02, num_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alpha_bar = float(alphas_cumprod[step])

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(arr.shape, dtype=np.float32)
    noisy = math.sqrt(alpha_bar) * arr + math.sqrt(1.0 - alpha_bar) * noise
    noisy = np.clip((noisy + 1.0) * 0.5 * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def apply_contrastive_logits(
    logits_clean: torch.Tensor,
    logits_contrast: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    normalized_logprob: bool = False,
) -> torch.Tensor:
    """
    Contrastive logit adjustment with Adaptive Plausibility Constraint (APC):
      - Raw VCD mode (normalized_logprob=False): (1 + alpha) * l_clean - alpha * l_contrast
      - Normalized Log-Prob mode (normalized_logprob=True): log_p_clean - alpha * log_p_contrast
        (prevents sharp blind-text logit magnitudes from blindly overpowering visual evidence).
    """
    clean_f = logits_clean.float()
    contrast_f = logits_contrast.float()

    log_probs_clean = F.log_softmax(clean_f, dim=-1)
    max_log_prob = log_probs_clean.max(dim=-1, keepdim=True).values
    cutoff = math.log(max(beta, 1e-8)) + max_log_prob

    if normalized_logprob:
        log_probs_contrast = F.log_softmax(contrast_f, dim=-1)
        cd_logits = log_probs_clean - alpha * log_probs_contrast
    else:
        cd_logits = (1.0 + alpha) * clean_f - alpha * contrast_f

    masked = cd_logits.masked_fill(log_probs_clean < cutoff, float("-inf"))
    return masked.to(logits_clean.dtype)


class ContrastiveDecisionLogitsProcessor(LogitsProcessor):
    """
    Version-safe LogitsProcessor for Qwen2.5-VL & Qwen3-VL.
    Lets Hugging Face's native `model.generate()` manage M-RoPE position_ids and KV-cache
    for the primary image stream, while injecting contrastive logits from a secondary
    contrast stream over the first `contrast_steps` decision tokens.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        contrast_inputs: Dict[str, torch.Tensor],
        prompt_len: int,
        alpha: float = 1.0,
        beta: float = 0.1,
        contrast_steps: int = 4,
        cached_step0_logits: Optional[torch.Tensor] = None,
        normalized_logprob: bool = False,
    ):
        self.model = model
        self.contrast_inputs = contrast_inputs
        self.prompt_len = prompt_len
        self.alpha = alpha
        self.beta = beta
        self.contrast_steps = max(1, int(contrast_steps))
        self.cached_step0_logits = cached_step0_logits
        self.normalized_logprob = normalized_logprob
        self.step_idx = 0

    @torch.inference_mode()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.step_idx >= self.contrast_steps:
            self.step_idx += 1
            return scores

        if self.step_idx == 0 and self.cached_step0_logits is not None:
            c_logits = self.cached_step0_logits.to(scores.device)
        else:
            gen_so_far = input_ids[:, self.prompt_len:]
            if gen_so_far.shape[1] > 0:
                c_in = dict(self.contrast_inputs)
                c_dev = self.contrast_inputs["input_ids"].device
                gen_dev = gen_so_far.to(c_dev)
                c_in["input_ids"] = torch.cat([self.contrast_inputs["input_ids"], gen_dev], dim=1)
                if "attention_mask" in self.contrast_inputs and self.contrast_inputs["attention_mask"] is not None:
                    ones = torch.ones(
                        (gen_dev.shape[0], gen_dev.shape[1]),
                        dtype=self.contrast_inputs["attention_mask"].dtype,
                        device=self.contrast_inputs["attention_mask"].device,
                    )
                    c_in["attention_mask"] = torch.cat([self.contrast_inputs["attention_mask"], ones], dim=1)
                for type_key in ("mm_token_type_ids", "token_type_ids"):
                    if type_key in self.contrast_inputs and self.contrast_inputs[type_key] is not None:
                        zeros = torch.zeros(
                            (gen_dev.shape[0], gen_dev.shape[1]),
                            dtype=self.contrast_inputs[type_key].dtype,
                            device=self.contrast_inputs[type_key].device,
                        )
                        c_in[type_key] = torch.cat([self.contrast_inputs[type_key], zeros], dim=1)
            else:
                c_in = self.contrast_inputs

            c_out = self.model(**c_in, use_cache=False)
            c_logits = c_out.logits[:, -1, :].to(scores.device)

        self.step_idx += 1
        return apply_contrastive_logits(
            scores,
            c_logits,
            alpha=self.alpha,
            beta=self.beta,
            normalized_logprob=self.normalized_logprob,
        )


def _extract_l0_5_visual_attn(
    wrapper: QwenVLWrapper,
    input_ids: torch.Tensor,
    gen_attentions: Optional[Tuple],
) -> Optional[float]:
    if not gen_attentions:
        return None
    span = wrapper.find_vision_token_span(input_ids)
    if span is None:
        return None
    s, e = span
    step0_attns = gen_attentions[0]
    n_layers = min(6, len(step0_attns))
    vals = []
    for layer_idx in range(n_layers):
        attn_last = step0_attns[layer_idx][0, :, -1, s:e]
        vals.append(float(attn_last.sum(dim=-1).mean().item()))
    return round(sum(vals) / len(vals), 5) if vals else None


@torch.inference_mode()
def run_greedy(
    wrapper: QwenVLWrapper,
    question: str,
    image: Optional[Any],
    max_new_tokens: int = 64,
    log_attn: bool = False,
    return_step0_logits: bool = False,
    max_pixels_override: Optional[int] = None,
    upscale_to_cap: bool = False,
) -> Dict[str, Any]:
    """Standard greedy baseline. Logs first-token entropy, latency, and optional L0-L5 visual attn."""
    t0 = time.perf_counter()
    inputs = wrapper.build_inputs(
        question=question,
        image=image,
        max_pixels_override=max_pixels_override,
        upscale_to_cap=upscale_to_cap,
    )
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

    step0_logits = gen_out.scores[0] if (gen_out.scores and len(gen_out.scores) > 0) else None
    first_token_entropy = round(_token_entropy(step0_logits[0]), 4) if step0_logits is not None else None
    visual_attn_l0_5 = _extract_l0_5_visual_attn(wrapper, inputs["input_ids"], gen_out.attentions) if (log_attn and image is not None) else None

    res = {
        "raw_output": raw_output,
        "pred_binary": pred_binary,
        "first_token_entropy": first_token_entropy,
        "latency_sec": latency_sec,
        "visual_attn_l0_5": visual_attn_l0_5,
        "gate_triggered": False,
        "gate_reason": "none",
    }
    if return_step0_logits:
        res["_step0_logits"] = step0_logits
        res["_inputs"] = inputs
    return res


@torch.inference_mode()
def run_vcd(
    wrapper: QwenVLWrapper,
    question: str,
    image: Optional[Any],
    max_new_tokens: int = 64,
    log_attn: bool = False,
    cd_alpha: float = 1.0,
    cd_beta: float = 0.1,
    vcd_noise_step: int = 500,
    contrast_steps: int = 4,
) -> Dict[str, Any]:
    """Always-On Visual Contrastive Decoding baseline (Leng et al., CVPR 2024)."""
    if image is None:
        return run_greedy(wrapper, question, image=None, max_new_tokens=max_new_tokens, log_attn=False)

    t0 = time.perf_counter()
    inputs_clean = wrapper.build_inputs(question=question, image=image)
    noisy_img = add_diffusion_noise(image, noise_step=vcd_noise_step, seed=42)
    inputs_noisy = wrapper.build_inputs(question=question, image=noisy_img)
    prompt_len = inputs_clean["input_ids"].shape[1]

    proc = ContrastiveDecisionLogitsProcessor(
        model=wrapper.model,
        contrast_inputs=inputs_noisy,
        prompt_len=prompt_len,
        alpha=cd_alpha,
        beta=cd_beta,
        contrast_steps=contrast_steps,
    )

    gen_out = wrapper.model.generate(
        **inputs_clean,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        logits_processor=LogitsProcessorList([proc]),
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=log_attn,
    )
    latency_sec = round(time.perf_counter() - t0, 4)

    new_tokens = gen_out.sequences[0][prompt_len:]
    raw_output = wrapper.processor.decode(new_tokens, skip_special_tokens=True).strip()
    pred_binary = parse_yes_no(raw_output)
    first_token_entropy = round(_token_entropy(gen_out.scores[0][0]), 4) if gen_out.scores else None
    visual_attn_l0_5 = _extract_l0_5_visual_attn(wrapper, inputs_clean["input_ids"], gen_out.attentions) if log_attn else None

    return {
        "raw_output": raw_output,
        "pred_binary": pred_binary,
        "first_token_entropy": first_token_entropy,
        "latency_sec": latency_sec,
        "visual_attn_l0_5": visual_attn_l0_5,
        "gate_triggered": True,
        "gate_reason": "always_on_vcd",
    }


@torch.inference_mode()
def run_selective_gate(
    wrapper: QwenVLWrapper,
    question: str,
    image: Optional[Any],
    max_new_tokens: int = 64,
    log_attn: bool = False,
    cd_alpha: float = 1.0,
    cd_beta: float = 0.1,
    vcd_noise_step: int = 500,
    contrast_steps: int = 4,
    contrast_source: str = "text",
    tau_ent: float = 0.15,
    tau_attn: Optional[float] = None,
) -> Dict[str, Any]:
    """Selective Gate v2: Zero-Waste Prior-Agreement Selective Steering."""
    if image is None:
        return run_greedy(wrapper, question, image=None, max_new_tokens=max_new_tokens, log_attn=False)

    t0 = time.perf_counter()
    greedy_res = run_greedy(
        wrapper,
        question=question,
        image=image,
        max_new_tokens=max_new_tokens,
        log_attn=log_attn,
        return_step0_logits=True,
    )
    step0_img_logits = greedy_res.pop("_step0_logits")
    inputs_clean = greedy_res.pop("_inputs")
    h_img = greedy_res["first_token_entropy"]
    v_attn = greedy_res["visual_attn_l0_5"]

    runner_up_plausible = False
    if step0_img_logits is not None:
        top2 = F.softmax(step0_img_logits[0].float(), dim=-1).topk(2).values
        runner_up_plausible = bool(top2[1].item() >= cd_beta * top2[0].item())

    attn_alarm = (tau_attn is not None and v_attn is not None and v_attn <= tau_attn)
    ent_alarm = (h_img is not None and h_img >= tau_ent)

    if (not runner_up_plausible) or (not (ent_alarm or attn_alarm)):
        greedy_res["latency_sec"] = round(time.perf_counter() - t0, 4)
        greedy_res["blind_entropy"] = None
        greedy_res["kl_img_vs_text"] = None
        greedy_res["top1_agrees_with_blind"] = None
        greedy_res["gate_triggered"] = False
        greedy_res["gate_reason"] = "none"
        greedy_res["flipped_from_greedy"] = False
        return greedy_res

    inputs_blind = wrapper.build_inputs(question=question, image=None)
    blind_out = wrapper.model(**inputs_blind, use_cache=False)
    step0_blind_logits = blind_out.logits[:, -1, :]
    h_blind = round(_token_entropy(step0_blind_logits[0]), 4)
    kl_img_vs_text = round(_kl_divergence(step0_img_logits[0], step0_blind_logits[0]), 4)

    top1_img = int(step0_img_logits[0].argmax(dim=-1).item())
    top1_blind = int(step0_blind_logits[0].argmax(dim=-1).item())
    if top1_img != top1_blind:
        greedy_res["latency_sec"] = round(time.perf_counter() - t0, 4)
        greedy_res["blind_entropy"] = h_blind
        greedy_res["kl_img_vs_text"] = kl_img_vs_text
        greedy_res["top1_agrees_with_blind"] = False
        greedy_res["gate_triggered"] = False
        greedy_res["gate_reason"] = "skipped_prior_disagree"
        greedy_res["flipped_from_greedy"] = False
        return greedy_res

    reasons = []
    if ent_alarm:
        reasons.append("entropy+prior_agree")
    if attn_alarm:
        reasons.append("attn_collapse")

    if contrast_source == "noisy":
        noisy_img = add_diffusion_noise(image, noise_step=vcd_noise_step, seed=42)
        c_inputs = wrapper.build_inputs(question=question, image=noisy_img)
        cached_c0 = None
    else:
        c_inputs = inputs_blind
        cached_c0 = step0_blind_logits

    prompt_len = inputs_clean["input_ids"].shape[1]
    proc = ContrastiveDecisionLogitsProcessor(
        model=wrapper.model,
        contrast_inputs=c_inputs,
        prompt_len=prompt_len,
        alpha=cd_alpha,
        beta=cd_beta,
        contrast_steps=contrast_steps,
        cached_step0_logits=cached_c0,
    )

    gen_out = wrapper.model.generate(
        **inputs_clean,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        logits_processor=LogitsProcessorList([proc]),
        return_dict_in_generate=True,
        output_scores=True,
    )
    latency_sec = round(time.perf_counter() - t0, 4)

    new_tokens = gen_out.sequences[0][prompt_len:]
    raw_output = wrapper.processor.decode(new_tokens, skip_special_tokens=True).strip()
    pred_binary = parse_yes_no(raw_output)

    return {
        "raw_output": raw_output,
        "pred_binary": pred_binary,
        "greedy_pred_binary": greedy_res["pred_binary"],
        "flipped_from_greedy": (pred_binary != greedy_res["pred_binary"]),
        "first_token_entropy": h_img,
        "blind_entropy": h_blind,
        "kl_img_vs_text": kl_img_vs_text,
        "top1_agrees_with_blind": True,
        "latency_sec": latency_sec,
        "visual_attn_l0_5": v_attn,
        "gate_triggered": True,
        "gate_reason": "+".join(reasons),
    }


@torch.inference_mode()
def run_selective_zoom(
    wrapper: QwenVLWrapper,
    question: str,
    image: Optional[Any],
    max_new_tokens: int = 64,
    log_attn: bool = False,
    zoom_max_pixels: int = 1280 * 28 * 28,
    zoom_alpha: float = 0.30,
    cd_beta: float = 0.1,
    contrast_steps: int = 4,
    tau_ent: float = 0.15,
    tau_attn: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Proposed Method (Selective Gate v3): Entropy-Gated Adaptive Visual Zoom + Normalized Prior Calibration.

    1. Fast Pass 1 (standard resolution): measures image entropy H_img and top-2 plausibility.
    2. If confident (H_img < tau_ent or runner-up implausible): stop immediately (~0 overhead!).
    3. If genuinely uncertain (~3%-15% of questions):
       a) Adaptive High-Res Visual Zoom: re-encodes the image at 2.5x visual token budget
          (`zoom_max_pixels = 1280*28*28`, gentle LANCZOS upscale up to 2x for smaller images)
          so the ViT actually gets sharper visual evidence for tiny objects/charts/maps.
       b) Gentle Normalized Prior Calibration: runs ~30ms blindfold pass; ONLY if the zoomed
          view still agrees with the blind prior AND remains somewhat uncertain, applies mild
          log-probability subtraction (`log_p_zoom - zoom_alpha * log_p_blind`, default alpha=0.30)
          so high-res vision stays in charge while weak blind biases are removed.
    """
    if image is None:
        return run_greedy(wrapper, question, image=None, max_new_tokens=max_new_tokens, log_attn=False)

    t0 = time.perf_counter()
    greedy_res = run_greedy(
        wrapper,
        question=question,
        image=image,
        max_new_tokens=max_new_tokens,
        log_attn=log_attn,
        return_step0_logits=True,
    )
    step0_img_logits = greedy_res.pop("_step0_logits")
    greedy_res.pop("_inputs")
    h_img = greedy_res["first_token_entropy"]
    v_attn = greedy_res["visual_attn_l0_5"]

    runner_up_plausible = False
    if step0_img_logits is not None:
        top2 = F.softmax(step0_img_logits[0].float(), dim=-1).topk(2).values
        runner_up_plausible = bool(top2[1].item() >= cd_beta * top2[0].item())

    attn_alarm = (tau_attn is not None and v_attn is not None and v_attn <= tau_attn)
    ent_alarm = (h_img is not None and h_img >= tau_ent)

    if (not runner_up_plausible) or (not (ent_alarm or attn_alarm)):
        greedy_res["latency_sec"] = round(time.perf_counter() - t0, 4)
        greedy_res["zoom_entropy"] = None
        greedy_res["blind_entropy"] = None
        greedy_res["gate_triggered"] = False
        greedy_res["gate_reason"] = "none"
        greedy_res["flipped_from_greedy"] = False
        return greedy_res

    # Step 3a: Adaptive High-Res Visual Zoom
    inputs_zoom = wrapper.build_inputs(
        question=question,
        image=image,
        max_pixels_override=zoom_max_pixels,
        upscale_to_cap=True,
    )
    zoom_prefill = wrapper.model(**inputs_zoom, use_cache=False)
    step0_zoom_logits = zoom_prefill.logits[:, -1, :]
    h_zoom = round(_token_entropy(step0_zoom_logits[0]), 4)

    # Step 3b: Fast ~30ms blindfold prefill
    inputs_blind = wrapper.build_inputs(question=question, image=None)
    blind_out = wrapper.model(**inputs_blind, use_cache=False)
    step0_blind_logits = blind_out.logits[:, -1, :]
    h_blind = round(_token_entropy(step0_blind_logits[0]), 4)
    kl_zoom_vs_text = round(_kl_divergence(step0_zoom_logits[0], step0_blind_logits[0]), 4)

    top1_zoom = int(step0_zoom_logits[0].argmax(dim=-1).item())
    top1_blind = int(step0_blind_logits[0].argmax(dim=-1).item())
    agrees_with_blind = (top1_zoom == top1_blind)

    prompt_len = inputs_zoom["input_ids"].shape[1]

    # Only apply gentle normalized prior calibration if the zoomed view STILL agrees with blind prior
    if agrees_with_blind and zoom_alpha > 0.0:
        proc = ContrastiveDecisionLogitsProcessor(
            model=wrapper.model,
            contrast_inputs=inputs_blind,
            prompt_len=prompt_len,
            alpha=zoom_alpha,
            beta=cd_beta,
            contrast_steps=contrast_steps,
            cached_step0_logits=step0_blind_logits,
            normalized_logprob=True,
        )
        gen_out = wrapper.model.generate(
            **inputs_zoom,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([proc]),
            return_dict_in_generate=True,
            output_scores=True,
        )
        reason = "zoom+norm_prior"
    else:
        gen_out = wrapper.model.generate(
            **inputs_zoom,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        reason = "zoom_only"

    latency_sec = round(time.perf_counter() - t0, 4)
    new_tokens = gen_out.sequences[0][prompt_len:]
    raw_output = wrapper.processor.decode(new_tokens, skip_special_tokens=True).strip()
    pred_binary = parse_yes_no(raw_output)

    return {
        "raw_output": raw_output,
        "pred_binary": pred_binary,
        "greedy_pred_binary": greedy_res["pred_binary"],
        "flipped_from_greedy": (pred_binary != greedy_res["pred_binary"]),
        "first_token_entropy": h_img,
        "zoom_entropy": h_zoom,
        "blind_entropy": h_blind,
        "kl_img_vs_text": kl_zoom_vs_text,
        "top1_agrees_with_blind": agrees_with_blind,
        "latency_sec": latency_sec,
        "visual_attn_l0_5": v_attn,
        "gate_triggered": True,
        "gate_reason": reason,
    }


def run_decoding(
    method: str,
    wrapper: QwenVLWrapper,
    question: str,
    image: Optional[Any],
    max_new_tokens: int = 64,
    log_attn: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Pluggable decoding dispatcher: greedy | vcd | selective_gate | selective_zoom."""
    m = method.lower().strip()
    if m == "greedy":
        return run_greedy(wrapper, question, image, max_new_tokens=max_new_tokens, log_attn=log_attn)
    if m == "vcd":
        return run_vcd(
            wrapper,
            question,
            image,
            max_new_tokens=max_new_tokens,
            log_attn=log_attn,
            cd_alpha=kwargs.get("cd_alpha", 1.0),
            cd_beta=kwargs.get("cd_beta", 0.1),
            vcd_noise_step=kwargs.get("vcd_noise_step", 500),
            contrast_steps=kwargs.get("contrast_steps", 4),
        )
    if m == "selective_gate":
        return run_selective_gate(
            wrapper,
            question,
            image,
            max_new_tokens=max_new_tokens,
            log_attn=log_attn,
            cd_alpha=kwargs.get("cd_alpha", 1.0),
            cd_beta=kwargs.get("cd_beta", 0.1),
            vcd_noise_step=kwargs.get("vcd_noise_step", 500),
            contrast_steps=kwargs.get("contrast_steps", 4),
            contrast_source=kwargs.get("contrast_source", "text"),
            tau_ent=kwargs.get("tau_ent", 0.15),
            tau_attn=kwargs.get("tau_attn", None),
        )
    if m == "selective_zoom":
        return run_selective_zoom(
            wrapper,
            question,
            image,
            max_new_tokens=max_new_tokens,
            log_attn=log_attn,
            zoom_max_pixels=kwargs.get("zoom_max_pixels", 1280 * 28 * 28),
            zoom_alpha=kwargs.get("zoom_alpha", 0.30),
            cd_beta=kwargs.get("cd_beta", 0.1),
            contrast_steps=kwargs.get("contrast_steps", 4),
            tau_ent=kwargs.get("tau_ent", 0.15),
            tau_attn=kwargs.get("tau_attn", None),
        )
    raise ValueError(f"Unknown decoding method: {method}")
