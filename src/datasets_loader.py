from dataclasses import dataclass
from typing import List, Optional, Any
from datasets import load_dataset


@dataclass
class BenchmarkExample:
    """Unified sample schema so HallusionBench, POPE, MathVista, and MMBench share one runner."""
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
        img_rows = [e for e in examples if e.split == "image"]
        txt_rows = [e for e in examples if e.split == "non_image"]
        n_txt = min(len(txt_rows), max(1, limit // 5))
        n_img = min(len(img_rows), limit - n_txt)
        return img_rows[:n_img] + txt_rows[:n_txt]

    return examples


def load_pope(category_filter: str = "adversarial", limit: Optional[int] = None) -> List[BenchmarkExample]:
    """
    Loads official POPE from Hugging Face (lmms-lab/POPE, test split: adversarial/popular/random).
    Used as a leak-free dev/calibration benchmark AND secondary object-hallucination check.
    When `limit` is set, returns an exact 50/50 Yes/No balanced slice.
    """
    ds = load_dataset("lmms-lab/POPE", split="test")
    yes_rows: List[BenchmarkExample] = []
    no_rows: List[BenchmarkExample] = []
    target_cat = category_filter.lower().strip()

    for idx, row in enumerate(ds):
        cat = str(row.get("category", "adversarial")).lower().strip()
        if target_cat != "all" and cat != target_cat:
            continue
        ans_raw = str(row.get("answer", "")).strip().lower()
        gt = "1" if ans_raw in ("yes", "1", "true") else "0"
        qid = str(row.get("question_id", idx))
        img_src = str(row.get("image_source", qid))
        ex = BenchmarkExample(
            uid=f"pope_{cat}_{qid}_{idx}",
            benchmark=f"pope_{target_cat}",
            split="test",
            category="POPE",
            subcategory=cat,
            visual_input="1",
            set_id=img_src,
            figure_id=img_src,
            question_id=qid,
            question=str(row["question"]),
            gt_answer=gt,
            image=row.get("image", None),
        )
        if gt == "1":
            yes_rows.append(ex)
        else:
            no_rows.append(ex)

        # Early stop once we have enough balanced rows for `limit`
        if limit is not None:
            half = limit // 2
            if len(yes_rows) >= half and len(no_rows) >= (limit - half):
                break

    if limit is not None:
        half = limit // 2
        return yes_rows[:half] + no_rows[: (limit - half)]

    return yes_rows + no_rows


def load_benchmark(name: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
    """Factory function for HallusionBench and POPE (adversarial/popular/random/all)."""
    key = name.lower().strip()
    if key == "hallusionbench":
        return load_hallusionbench(limit=limit)
    if key in ("pope", "pope_adv", "pope_adversarial"):
        return load_pope(category_filter="adversarial", limit=limit)
    if key == "pope_popular":
        return load_pope(category_filter="popular", limit=limit)
    if key == "pope_random":
        return load_pope(category_filter="random", limit=limit)
    if key == "pope_all":
        return load_pope(category_filter="all", limit=limit)
    raise ValueError(
        f"Benchmark '{name}' is not supported. Supported: "
        "['hallusionbench', 'pope_adv', 'pope_popular', 'pope_random', 'pope_all']."
    )