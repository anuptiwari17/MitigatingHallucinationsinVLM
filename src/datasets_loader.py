from dataclasses import dataclass
from typing import List, Optional, Any
from datasets import load_dataset


@dataclass
class BenchmarkExample:
    """Unified sample schema so HallusionBench, MathVista, MMBench, and POPE share one runner."""
    uid: str
    benchmark: str
    split: str
    category: str
    subcategory: str
    visual_input: str          # "1" or "2" = image provided; "0" = text-only control
    set_id: str
    figure_id: str
    question_id: str
    question: str
    gt_answer: str             # "1" (Yes) or "0" (No)
    image: Optional[Any] = None  # PIL.Image or None


def load_hallusionbench(limit: Optional[int] = None) -> List[BenchmarkExample]:
    """
    Loads official HallusionBench from Hugging Face (lmms-lab-encoder/HallusionBench).
    Total: 1,129 rows = 951 image rows + 178 non_image text-only control rows.
    """
    ds = load_dataset("lmms-lab-encoder/HallusionBench")
    examples: List[BenchmarkExample] = []

    for split_name in ["image", "non_image"]:
        if split_name not in ds:
            continue
        split_data = ds[split_name]
        for idx, row in enumerate(split_data):
            uid = (
                f"{split_name}_{row['category']}_{row['subcategory']}_"
                f"s{row['set_id']}_f{row['figure_id']}_q{row['question_id']}_{idx}"
            )
            img = row.get("image", None) if split_name == "image" else None
            examples.append(
                BenchmarkExample(
                    uid=uid,
                    benchmark="hallusionbench",
                    split=split_name,
                    category=str(row["category"]),
                    subcategory=str(row["subcategory"]),
                    visual_input=str(row["visual_input"]),
                    set_id=str(row["set_id"]),
                    figure_id=str(row["figure_id"]),
                    question_id=str(row["question_id"]),
                    question=str(row["question"]),
                    gt_answer=str(row["gt_answer"]),
                    image=img,
                )
            )

    if limit is not None:
        # For small smoke tests, sample evenly from image and non_image if possible
        img_rows = [e for e in examples if e.split == "image"]
        txt_rows = [e for e in examples if e.split == "non_image"]
        n_txt = min(len(txt_rows), max(1, limit // 5))
        n_img = min(len(img_rows), limit - n_txt)
        return img_rows[:n_img] + txt_rows[:n_txt]

    return examples


def load_benchmark(name: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
    """Factory function to add MathVista, MMBench, or POPE cleanly later."""
    key = name.lower().strip()
    if key == "hallusionbench":
        return load_hallusionbench(limit=limit)
    raise ValueError(
        f"Benchmark '{name}' is not implemented yet. Currently supported: ['hallusionbench']. "
        "Add new loaders (e.g., mathvista, mmbench, pope) in src/datasets_loader.py."
    )
