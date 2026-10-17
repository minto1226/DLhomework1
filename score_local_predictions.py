"""Check a local prediction text file against the separate MNIST answer key."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("data/mnist_png_split/LOCAL_TEST.txt"))
    parser.add_argument("--answer-key", type=Path, default=Path("data/mnist_png_split/test_ground_truth.csv"))
    args = parser.parse_args()

    try:
        with args.answer_key.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["filename_stem", "label"]:
                raise ValueError("答案檔欄位應為 filename_stem,label")
            answers = {row["filename_stem"]: int(row["label"]) for row in reader}
        predictions = {}
        with args.predictions.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                columns = line.split()
                if len(columns) != 2 or columns[1] not in "0123456789":
                    raise ValueError(f"預測檔第 {line_number} 行不是「檔名 數字」兩欄")
                stem, digit = columns
                if stem in predictions:
                    raise ValueError(f"預測檔重複檔名: {stem}")
                predictions[stem] = int(digit)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if not answers or set(predictions) != set(answers):
        missing = len(set(answers) - set(predictions))
        extra = len(set(predictions) - set(answers))
        parser.error(f"預測檔與答案檔的檔名不一致：缺少 {missing}、多出 {extra}")
    correct = sum(predictions[name] == label for name, label in answers.items())
    print(f"本機測試: {correct}/{len(answers)} 正確，accuracy={correct / len(answers):.2%}")


if __name__ == "__main__":
    main()
