"""Prepare four labeled PNG layouts and two unlabeled test folders for local checks.

Run from any directory: python3 prepare_four_layout_cases.py
The source is the existing, independent MNIST PNG train/test split under data/.
Outputs are created directly under data/ and existing outputs are never replaced.
"""

import csv
import shutil
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_TRAIN = DATA_DIR / "mnist_png_split" / "train"
SOURCE_TEST = DATA_DIR / "mnist_png_split" / "test"
SOURCE_ANSWERS = DATA_DIR / "mnist_png_split" / "test_ground_truth.csv"
ODD_DIGITS = {1, 3, 5, 7, 9}
TRAIN_FOLDERS = (
    "case1_digit_folders",
    "case2_odd_digit_folders",
    "case3_flat_filenames",
    "case4_odd_flat_filenames",
)
TEST_FOLDERS = ("test_all_digits", "test_odd_digits")
ANSWER_FILES = ("test_all_digits_ground_truth.csv", "test_odd_digits_ground_truth.csv")
OUTPUT_NAMES = (*TRAIN_FOLDERS, *TEST_FOLDERS, *ANSWER_FILES)


def collect_training_images():
    """Keep every class in the full cases; only odd classes in cases 2 and 4."""
    if not SOURCE_TRAIN.is_dir():
        raise ValueError(f"找不到來源訓練資料夾: {SOURCE_TRAIN}")
    by_digit = {}
    for digit in range(10):
        folder = SOURCE_TRAIN / str(digit)
        if not folder.is_dir():
            raise ValueError(f"缺少訓練類別資料夾: {folder}")
        paths = sorted(path for path in folder.iterdir()
                       if path.is_file() and path.suffix.lower() == ".png")
        if not paths:
            raise ValueError(f"訓練類別沒有 PNG: {folder}")
        by_digit[digit] = paths
    return by_digit


def collect_test_images():
    """Use the answer key only for selecting odd test images, never in test dirs."""
    if not SOURCE_TEST.is_dir() or not SOURCE_ANSWERS.is_file():
        raise ValueError("找不到既有的 MNIST 測試圖片或答案表")
    with SOURCE_ANSWERS.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["filename_stem", "label"]:
            raise ValueError(f"答案表欄位不符: {SOURCE_ANSWERS}")
        rows = [(row["filename_stem"], int(row["label"])) for row in reader]

    if len({stem for stem, _ in rows}) != len(rows):
        raise ValueError("測試答案表有重複檔名")
    paths = sorted(path for path in SOURCE_TEST.iterdir()
                   if path.is_file() and path.suffix.lower() == ".png")
    if {path.stem for path in paths} != {stem for stem, _ in rows}:
        raise ValueError("測試圖片與答案表的檔名不一致")
    if set(label for _, label in rows) != set(range(10)):
        raise ValueError("來源測試集必須包含 0 至 9 的圖片")
    answer_by_stem = dict(rows)
    return [(path, answer_by_stem[path.stem]) for path in paths]


def flat_name(digit, number):
    """Match phase2_train.py's accepted 1.png and 1 (2).png forms."""
    return f"{digit}.png" if number == 1 else f"{digit} ({number}).png"


def write_answer_key(path, test_items):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename_stem", "label"])
        writer.writerows((image.stem, digit) for image, digit in test_items)


def main():
    by_digit = collect_training_images()
    test_items = collect_test_images()
    existing = [DATA_DIR / name for name in OUTPUT_NAMES if (DATA_DIR / name).exists()]
    if existing:
        raise ValueError(f"輸出已存在，不會覆蓋: {existing[0]}")

    staging = DATA_DIR / ".four_layout_cases.partial"
    if staging.exists():
        raise ValueError(f"暫存資料夾已存在，請先檢查: {staging}")
    staging.mkdir()
    try:
        for name in (*TRAIN_FOLDERS, *TEST_FOLDERS):
            (staging / name).mkdir()

        for digit, paths in by_digit.items():
            folder1 = staging / TRAIN_FOLDERS[0] / str(digit)
            folder1.mkdir()
            folder2 = staging / TRAIN_FOLDERS[1] / str(digit) if digit in ODD_DIGITS else None
            folder4 = staging / TRAIN_FOLDERS[3] if digit in ODD_DIGITS else None
            if folder2 is not None:
                folder2.mkdir()

            for number, source in enumerate(paths, start=1):
                shutil.copy2(source, folder1 / source.name)
                shutil.copy2(source, staging / TRAIN_FOLDERS[2] / flat_name(digit, number))
                if folder2 is not None:
                    shutil.copy2(source, folder2 / source.name)
                    shutil.copy2(source, folder4 / flat_name(digit, number))
            print(f"類別 {digit}: {len(paths)} 張完成", flush=True)

        for source, digit in test_items:
            shutil.copy2(source, staging / TEST_FOLDERS[0] / source.name)
            if digit in ODD_DIGITS:
                shutil.copy2(source, staging / TEST_FOLDERS[1] / source.name)
        write_answer_key(staging / ANSWER_FILES[0], test_items)
        odd_test_items = [(source, digit) for source, digit in test_items if digit in ODD_DIGITS]
        write_answer_key(staging / ANSWER_FILES[1], odd_test_items)

        for name in OUTPUT_NAMES:
            (staging / name).rename(DATA_DIR / name)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    full_count = sum(len(paths) for paths in by_digit.values())
    odd_count = sum(len(paths) for digit, paths in by_digit.items() if digit in ODD_DIGITS)
    print(f"訓練圖片：情況 1/3 各 {full_count} 張；情況 2/4 各 {odd_count} 張")
    print(f"測試圖片：全數字 {len(test_items)} 張；奇數 {len(odd_test_items)} 張")
    print(f"每個測試數字張數：{dict(sorted(Counter(digit for _, digit in test_items).items()))}")
    print(f"輸出位置：{DATA_DIR}")


if __name__ == "__main__":
    main()
