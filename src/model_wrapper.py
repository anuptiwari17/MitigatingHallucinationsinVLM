from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Automatically picks fp16 on Kaggle T4/P100 and bf16 on L4/A100/RTX 3090+."""
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


class QwenVLWrapper:
    """
    Hardware-portable wrapper for Qwen3-VL and Qwen2.5-VL models.
    Works across Kaggle (2x T4), Google Colab (T4/L4/A100), and single college GPUs.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        log_attn: bool = False,
    ):
        self.model_id = model_id
        self.torch_dtype = resolve_dtype(dtype)
        # Eager attention is required to read raw attention matrices for L0-L5 probing
        self.attn_impl = "eager" if log_attn else attn_implementation
        self.log_attn = log_attn

        print(f"Loading {model_id} | dtype={self.torch_dtype} | attn={self.attn_impl}...")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            attn_implementation=self.attn_impl,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def build_inputs(self, question: str, image: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """Builds model inputs cleanly for both image+text and text-only control rows."""
        if image is not None:
            if image.mode != "RGB":
                image = image.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }
            ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def find_vision_token_span(self, input_ids: torch.Tensor) -> Optional[Tuple[int, int]]:
        """
        Locates [start, end) indices of visual tokens in Qwen2.5-VL / Qwen3-VL prompts.
        Used in Phase 1 Attention-Collapse Probing & Selective Gate.
        """
        tok = self.processor.tokenizer
        start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        ids = input_ids[0].tolist()
        if start_id in ids and end_id in ids:
            s = ids.index(start_id) + 1
            e = ids.index(end_id)
            if e > s:
                return (s, e)
        return None
