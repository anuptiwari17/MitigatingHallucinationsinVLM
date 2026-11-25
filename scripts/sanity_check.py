import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets_loader import load_hallusionbench


def main():
    samples = load_hallusionbench(limit=10)
    print(f"Loaded {len(samples)} HallusionBench samples cleanly.\n" + "-" * 70)
    for i, ex in enumerate(samples, 1):
        img_info = f"PIL {ex.image.size}" if ex.image is not None else "None (text-only control)"
        gt = "Yes (1)" if ex.gt_answer == "1" else "No (0)"
        print(f"[{i}] {ex.uid}")
        print(f"    Split: {ex.split} | Image: {img_info} | GT: {gt}")
        print(f"    Q: {ex.question}\n")


if __name__ == "__main__":
    main()
