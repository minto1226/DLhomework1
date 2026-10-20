"""Train and predict PNG digits with a NumPy multilayer network.

Training_data can have digit subfolders or flat PNGs named like 1.png and
1 (2).png. Testing_data is a flat PNG folder. When test data arrives later,
keep this process open at its prompt:
    python3 phase2_train.py --train-dir ./Training_data --student-id YOUR_ID

If both folders are already available:
    python3 phase2_train.py --train-dir ./Training_data --test-dir ./Testing_data --student-id YOUR_ID
"""

import argparse
from pathlib import Path
import re
import numpy as np
from PIL import Image


def load_training_pngs(train_dir):
    """讀取簡報示例的 Training_data/0/*.png 至 Training_data/9/*.png。"""
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        raise ValueError(f"訓練資料夾不存在: {train_dir}")
    data = []
    target = []

    for label_path in sorted(train_dir.iterdir()):
        if not label_path.is_dir() or not label_path.name.isdigit():
            continue
        label_int = int(label_path.name)
        if not 0 <= label_int <= 9:
            raise ValueError(f"類別資料夾須為 0 到 9: {label_path}")
        for img_path in sorted(label_path.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() == ".png":
                with Image.open(img_path) as img:
                    data.append(np.asarray(img.convert("L").resize((28, 28)), dtype=np.uint8).reshape(784))
                target.append(label_int)

    if len(data) == 0:
        raise ValueError(f"在 {train_dir} 中找不到任何子資料夾與 .png 圖片！請確認路徑層級。")
    return np.array(data, dtype=np.uint8), np.array(target, dtype=np.uint8)


def load_training_pngs_from_filenames(train_dir):
    """讀取同一層的 0.png、1 (2).png 等圖片，從檔名取得 0 至 9 標籤。"""
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        raise ValueError(f"訓練資料夾不存在: {train_dir}")
    paths = sorted(
        (path for path in train_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError(f"在 {train_dir} 中找不到任何 .png 訓練圖片！")

    data = []
    target = []
    for img_path in paths:
        match = re.fullmatch(r"([0-9])(?: \([0-9]+\))?", img_path.stem)
        if match is None:
            raise ValueError(f"無法從檔名取得 0 到 9 標籤: {img_path}；請使用 1.png 或 1 (2).png 形式")
        with Image.open(img_path) as img:
            data.append(np.asarray(img.convert("L").resize((28, 28)), dtype=np.uint8).reshape(784))
        target.append(int(match.group(1)))

    return np.array(data, dtype=np.uint8), np.array(target, dtype=np.uint8)


def detect_training_layout(train_dir):
    """用圖片位置判斷格式；一個數字子資料夾仍算子資料夾格式。"""
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        raise ValueError(f"訓練資料夾不存在: {train_dir}")

    entries = list(train_dir.iterdir())
    root_pngs = [path for path in entries if path.is_file() and path.suffix.lower() == ".png"]
    digit_dirs = [path for path in entries if path.is_dir() and path.name.isdigit()]
    other_dirs = [path for path in entries if path.is_dir() and not path.name.isdigit() and not path.name.startswith(".")]

    if root_pngs and digit_dirs:
        raise ValueError(f"{train_dir} 同時有根目錄 PNG 與數字子資料夾，請分開存放訓練圖片")
    if other_dirs:
        raise ValueError(f"{train_dir} 有無法辨識的子資料夾: {other_dirs[0].name}；請將 --train-dir 指向圖片或數字子資料夾所在的那一層")
    if root_pngs:
        return "filenames"
    if digit_dirs:
        return "digit_subfolders"
    raise ValueError(f"在 {train_dir} 找不到根目錄 PNG 或 0 到 9 數字子資料夾；請確認 --train-dir 路徑")


def load_training_pngs_auto(train_dir):
    """保留兩種獨立載入函式，依資料排列選擇其一。"""
    layout = detect_training_layout(train_dir)
    if layout == "digit_subfolders":
        return load_training_pngs(train_dir)
    return load_training_pngs_from_filenames(train_dir)


def load_testing_pngs(test_dir):
    """讀取扁平 Testing_data 資料夾的 PNG，保留不含副檔名的檔名。"""
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise ValueError(f"測試資料夾不存在: {test_dir}")
    paths = sorted(
        (path for path in test_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name,
    )

    data = []
    txt_names = []
    for img_path in paths:
        with Image.open(img_path) as img:
            data.append(np.asarray(img.convert("L").resize((28, 28)), dtype=np.uint8).reshape(784))
        txt_names.append(img_path.stem)

    if len(data) == 0:
        raise ValueError(f"在 {test_dir} 中找不到任何 .png 測試圖片！")
    return np.array(data, dtype=np.uint8), txt_names

def initialize_parameters(hidden_layers, rng):
    if any(width <= 0 for width in hidden_layers):
        raise ValueError("hidden-layer widths must be positive")
    layer_sizes = (784, *hidden_layers, 10)
    weights = []
    biases = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
        biases.append(np.zeros((1, fan_out)))
    return weights, biases


def forward(x, weights, biases):
    z_values = []
    activations = [x]
    for layer in range(len(weights)):
        z = activations[-1] @ weights[layer] + biases[layer]
        z_values.append(z)
        if layer == len(weights) - 1:
            z_stable = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_stable)
            a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            a = np.maximum(0.0, z)
        activations.append(a)
    return z_values, activations


def make_one_hot(labels):
    y = np.zeros((len(labels), 10))
    for row, label in enumerate(labels):
        y[row, int(label)] = 1.0
    return y


def batch_gradients(x, y, weights, biases):
    z_values, activations = forward(x, weights, biases)
    p = activations[-1]
    batch_size = len(x)
    loss = -np.sum(y * np.log(p + 1e-12)) / batch_size

    # Cross-entropy + Softmax: average the output delta over B examples.
    delta = (p - y) / batch_size
    d_weights = [None] * len(weights)
    d_biases = [None] * len(biases)
    for layer in range(len(weights) - 1, -1, -1):
        # (fan_in, B) @ (B, fan_out) -> (fan_in, fan_out).
        d_weights[layer] = activations[layer].T @ delta
        d_biases[layer] = np.sum(delta, axis=0, keepdims=True)
        if layer > 0:
            delta = (delta @ weights[layer].T) * (z_values[layer - 1] > 0.0)
    return loss, p, d_weights, d_biases


def train_epoch(images, labels, weights, biases, batch_size, learning_rate, rng):
    order = rng.permutation(len(labels))
    total_loss = 0.0
    correct = 0
    for start in range(0, len(labels), batch_size):
        indices = order[start : start + batch_size] 
        x = images[indices].astype(np.float64) / 255.0
        batch_labels = labels[indices]
        y = make_one_hot(batch_labels)

        loss, p, d_weights, d_biases = batch_gradients(x, y, weights, biases)
        total_loss += loss * len(indices)
        correct += np.count_nonzero(np.argmax(p, axis=1) == batch_labels)

        # All gradients above used the same pre-update parameter values.
        for layer in range(len(weights)):
            weights[layer] -= learning_rate * d_weights[layer]
            biases[layer] -= learning_rate * d_biases[layer]

    return total_loss / len(labels), correct / len(labels)


def evaluate(images, labels, weights, biases, batch_size):
    loss_total = 0.0
    correct = 0
    for start in range(0, len(labels), batch_size):
        x = images[start : start + batch_size].astype(np.float64) / 255.0
        batch_labels = labels[start : start + batch_size]
        y = make_one_hot(batch_labels)
        p = forward(x, weights, biases)[1][-1]
        loss_total += -np.sum(y * np.log(p + 1e-12))
        correct += np.count_nonzero(np.argmax(p, axis=1) == batch_labels)
    return loss_total / len(labels), correct / len(labels)


def predict_all(images, weights, biases, batch_size=256):
    preds = []
    for start in range(0, len(images), batch_size):
        x = images[start : start + batch_size].astype(np.float64) / 255.0
        _, activations = forward(x, weights, biases)
        preds.append(np.argmax(activations[-1], axis=1))
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True, help="訓練資料夾：數字子資料夾或檔名標籤 PNG")
    parser.add_argument("--val-dir", type=Path, default=None, help="可選的獨立驗證資料夾，支援同樣兩種格式")
    parser.add_argument("--test-dir", type=Path, default=None, help="助教 USB 給的測試資料夾路徑")
    parser.add_argument("--student-id", type=str, default=None, help="你的學號 (將輸出 學號.txt)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--anneal-factor", type=float, default=0.93)
    parser.add_argument("--hidden-layers", type=int, nargs="*", default=[256, 128])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("--epochs、--batch-size 與 --learning-rate 都必須大於 0")
    if not 0 < args.anneal_factor <= 1:
        parser.error("--anneal-factor 必須介於 0 與 1 之間（含 1）")
    if any(width <= 0 for width in args.hidden_layers):
        parser.error("--hidden-layers 的每層寬度都必須大於 0")
    if args.test_dir is not None and args.student_id is None:
        parser.error("提供 --test-dir 時也必須提供 --student-id")
    if args.student_id is not None and (not args.student_id or Path(args.student_id).name != args.student_id or args.student_id in (".", "..")):
        parser.error("--student-id 必須是不含路徑的學號")

    if args.test_dir is not None and not args.test_dir.is_dir():
        parser.error(f"測試資料夾不存在: {args.test_dir}")
    if args.test_dir is not None and not any(
        path.is_file() and path.suffix.lower() == ".png" for path in args.test_dir.iterdir()
    ):
        parser.error(f"測試資料夾沒有 PNG 圖片: {args.test_dir}")

    try:
        all_images, all_labels = load_training_pngs_auto(args.train_dir)
    except (ValueError, OSError) as error:
        parser.error(f"訓練資料載入失敗: {error}")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_labels))
    all_images, all_labels = all_images[perm], all_labels[perm]

    if args.val_dir is None:
        if len(all_labels) < 2:
            parser.error("自動切分 validation 至少需要 2 張訓練圖片")
        val_count = max(1, round(len(all_labels) * 0.1))
        val_images, val_labels = all_images[:val_count], all_labels[:val_count]
        train_images, train_labels = all_images[val_count:], all_labels[val_count:]
    else:
        try:
            val_images, val_labels = load_training_pngs_auto(args.val_dir)
        except (ValueError, OSError) as error:
            parser.error(f"驗證資料載入失敗: {error}")
        train_images, train_labels = all_images, all_labels

    weights, biases = initialize_parameters(tuple(args.hidden_layers), rng)
    print(f"載入訓練圖片: {len(train_labels)}, 驗證圖片: {len(val_labels)}, 網路結構: {(784, *args.hidden_layers, 10)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        learning_rate = args.learning_rate * args.anneal_factor ** (epoch - 1)
        train_epoch(
            train_images, train_labels, weights, biases,
            args.batch_size, learning_rate, rng,
        )
        train_loss, train_acc = evaluate(train_images, train_labels, weights, biases, args.batch_size)
        _, val_acc = evaluate(val_images, val_labels, weights, biases, args.batch_size)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} - lr={learning_rate:.6f}, train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, val_acc={val_acc:.2%}", flush=True)

    test_dir = args.test_dir
    wrote_result = False
    while args.student_id is not None:
        if test_dir is None:
            prompt = ("若有下一輪測試，請輸入測試資料夾路徑（直接 Enter 結束）：" if wrote_result
                      else "訓練完成。請輸入助教提供的 Testing_data 資料夾路徑：")
            try:
                typed_path = input(prompt).strip()
            except EOFError:
                if not wrote_result:
                    parser.exit(2, "標準輸入已關閉，未產生預測檔；請重新執行並提供 --test-dir。\n")
                break
            if not typed_path:
                if wrote_result:
                    break
                print("尚未產生預測檔；請輸入測試資料夾路徑。")
                continue
            test_dir = Path(typed_path).expanduser()

        try:
            test_images, txt_names = load_testing_pngs(test_dir)
        except ValueError as error:
            if args.test_dir is not None:
                parser.error(str(error))
            print(f"{error} 請重新輸入。")
            test_dir = None
            continue

        predictions = predict_all(test_images, weights, biases, args.batch_size)
        result_path = Path(f"{args.student_id}.txt")
        with result_path.open("w", encoding="utf-8") as f:
            for name, ans in zip(txt_names, predictions):
                f.write(f"{name} {int(ans)}\n")
        print(f"\n[完成] 已讀取 {len(txt_names)} 張測試圖，並輸出至: {result_path.resolve()}")
        wrote_result = True
        if args.test_dir is not None:
            break
        test_dir = None

if __name__ == "__main__":
    main()
