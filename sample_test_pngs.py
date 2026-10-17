"""Create a small, flat PNG test folder for local practice.

The MNIST test labels are used only to sample each digit equally and to write
a separate local answer key. The network reads only the PNG test folder.
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from prepare_png_mnist import read_idx_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/mnist_png/test"))
    parser.add_argument("--labels-file", type=Path, default=Path("data/mnist/t10k-labels-idx1-ubyte.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mnist_png_split/test"))
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.per_class <= 0:
        parser.error("--per-class 必須大於 0")
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    answer_key = output.parent / f"{output.name}_ground_truth.csv"
    if not source.is_dir():
        parser.error(f"找不到測試圖片資料夾: {source}")
    if output == source or source in output.parents:
        parser.error("輸出資料夾不可位於來源資料夾內")
    if output.exists() or answer_key.exists():
        parser.error(f"輸出已存在，為避免覆蓋資料請改用其他 --output-dir: {output}")

    try:
        labels = read_idx_labels(args.labels_file)
    except (OSError, ValueError) as error:
        parser.error(f"無法讀取本機測試標籤: {error}")

    rng = np.random.default_rng(args.seed)
    chosen = []
    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        if len(indices) < args.per_class:
            parser.error(f"數字 {digit} 只有 {len(indices)} 張，無法抽出 {args.per_class} 張")
        chosen.extend(int(index) for index in rng.choice(indices, size=args.per_class, replace=False))
    chosen.sort()

    source_paths = [source / f"{index:04d}.png" for index in chosen]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        parser.error(f"來源圖片缺失，第一個檔案是: {missing[0]}")

    staging = output.with_name(output.name + ".partial")
    key_staging = answer_key.with_name(answer_key.name + ".partial")
    if staging.exists() or key_staging.exists():
        parser.error("暫存輸出已存在，請先檢查後再執行")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        for source_path in source_paths:
            shutil.copy2(source_path, staging / source_path.name)
        with key_staging.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename_stem", "label"])
            for index in chosen:
                writer.writerow([f"{index:04d}", int(labels[index])])
        staging.rename(output)
        key_staging.rename(answer_key)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        key_staging.unlink(missing_ok=True)
        raise

    print(f"已建立 {len(chosen)} 張無標籤測試 PNG（每類 {args.per_class} 張）: {output}")
    print(f"本機比對用答案（不在測試資料夾內）: {answer_key}")


if __name__ == "__main__":
    main()
