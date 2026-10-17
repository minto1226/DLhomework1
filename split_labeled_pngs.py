"""Split labeled digit PNG folders into separate training and validation folders.

Example:
    python3 split_labeled_pngs.py

The original source folder is left untouched. The output has train/0..9 and
val/0..9, which can be passed to phase2_train.py with --train-dir/--val-dir.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


def class_pngs(source_dir):
    """Return sorted PNG paths for each class, rejecting missing classes."""
    if not source_dir.is_dir():
        raise ValueError(f"找不到有標籤的來源資料夾: {source_dir}")
    extra_labels = [
        path.name for path in source_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and int(path.name) not in range(10)
    ]
    if extra_labels:
        raise ValueError(f"來源有超出 0 到 9 的類別資料夾: {extra_labels}")

    by_class = []
    for label in range(10):
        class_dir = source_dir / str(label)
        if not class_dir.is_dir():
            raise ValueError(f"缺少類別資料夾: {class_dir}")
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        if len(paths) < 2:
            raise ValueError(f"{class_dir} 至少需要 2 張 PNG，才能同時保留訓練與驗證樣本")
        by_class.append(paths)
    return by_class


def validation_counts(by_class, fraction):
    """Allocate an exact total fraction while keeping every class in both sets."""
    counts = [len(paths) for paths in by_class]
    target = round(sum(counts) * fraction)
    if not 10 <= target <= sum(counts) - 10:
        raise ValueError("此比例無法讓 0 到 9 每類同時有訓練及驗證圖片")

    ideals = [count * fraction for count in counts]
    quotas = [max(1, min(count - 1, int(ideal))) for count, ideal in zip(counts, ideals)]
    while sum(quotas) < target:
        eligible = [i for i in range(10) if quotas[i] < counts[i] - 1]
        chosen = max(eligible, key=lambda i: (ideals[i] - quotas[i], -i))
        quotas[chosen] += 1
    while sum(quotas) > target:
        eligible = [i for i in range(10) if quotas[i] > 1]
        chosen = max(eligible, key=lambda i: (quotas[i] - ideals[i], -i))
        quotas[chosen] -= 1
    return quotas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/mnist_png/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mnist_png_split"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction 必須介於 0 與 1 之間")
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error("輸出資料夾不可位於來源資料夾內")
    if output.exists():
        parser.error(f"輸出資料夾已存在，為避免覆蓋資料請改用其他路徑: {output}")

    try:
        by_class = class_pngs(source)
        quotas = validation_counts(by_class, args.val_fraction)
    except ValueError as error:
        parser.error(str(error))

    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        parser.error(f"暫存資料夾已存在，請先檢查後再執行: {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    rng = np.random.default_rng(args.seed)
    try:
        for label, paths in enumerate(by_class):
            train_dir = staging / "train" / str(label)
            val_dir = staging / "val" / str(label)
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            val_indices = set(rng.permutation(len(paths))[:quotas[label]])
            for index, source_path in enumerate(paths):
                target_dir = val_dir if index in val_indices else train_dir
                shutil.copy2(source_path, target_dir / source_path.name)
            print(f"類別 {label}: train={len(paths) - quotas[label]}, val={quotas[label]}", flush=True)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise

    print(f"完成: train={sum(map(len, by_class)) - sum(quotas)}, val={sum(quotas)}")
    print(f"訓練資料夾: {output / 'train'}")
    print(f"驗證資料夾: {output / 'val'}")


if __name__ == "__main__":
    main()
