"""Phase 2: train a NumPy multilayer network on local MNIST IDX files.

Example:
    python3 phase2_train.py --hidden-layers 256 128 --epochs 5

The four original MNIST .gz files must already be in data/mnist. This script
does not download data or use a neural-network framework/autodiff.
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image


def load_training_pngs(train_dir):
    """完全對齊助教簡報 p.41-42 的資料夾結構: Training_data/0/xxxx.png"""
    train_dir = Path(train_dir)
    data = []
    target = []

    # 依序讀取 0, 1, 2, ... 子資料夾
    for label_name in sorted(os.listdir(train_dir)):
        label_path = train_dir / label_name
        if not label_path.is_dir() or not label_name.isdigit():
            continue
        label_int = int(label_name)
        for filename in sorted(os.listdir(label_path)):
            if filename.lower().endswith(".png"):
                img_path = label_path / filename
                img = Image.open(img_path).convert("L").resize((28, 28))
                data.append(np.array(img, dtype=np.uint8).reshape(784))
                target.append(label_int)

    if len(data) == 0:
        raise ValueError(f"在 {train_dir} 中找不到任何子資料夾與 .png 圖片！請確認路徑層級。")
    return np.array(data, dtype=np.uint8), np.array(target, dtype=np.uint8)


def load_testing_pngs(test_dir):
    """完全對齊助教簡報 p.53: 讀取 Testing_data 內的 0000.png, 0001.png..."""
    test_dir = Path(test_dir)
    filenames = [f for f in os.listdir(test_dir) if f.lower().endswith(".png")]
    # 依照檔名數字大小排序 (確保 0000, 0001, 0002... 順序絕對正確)
    filenames.sort(key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else f)

    data = []
    txt_names = []
    for filename in filenames:
        img_path = test_dir / filename
        img = Image.open(img_path).convert("L").resize((28, 28))
        data.append(np.array(img, dtype=np.uint8).reshape(784))
        txt_names.append(os.path.splitext(filename)[0])  # 去除 .png 副檔名

    if len(data) == 0:
        raise ValueError(f"在 {test_dir} 中找不到任何 .png 測試圖片！")
    return np.array(data, dtype=np.uint8), txt_names

def initialize_parameters(hidden_layers, rng):
    """Create 784 -> hidden_layers -> 10 weights and bias rows."""
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
    """Save each z and activation for the backward loop."""
    z_values = []
    activations = [x]
    for layer in range(len(weights)):
        z = activations[-1] @ weights[layer] + biases[layer]
        z_values.append(z)
        if layer == len(weights) - 1:
            z_stable = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z)
            a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            a = np.maximum(0.0, z)
        activations.append(a)
    return z_values, activations


def make_one_hot(labels):
    """Keep the Phase 1 one-hot idea explicit for a whole mini-batch."""
    y = np.zeros((len(labels), 10))
    for row, label in enumerate(labels):
        y[row, int(label)] = 1.0
    return y


def batch_gradients(x, y, weights, biases):
    """Return mean loss, predictions, and mean gradients for one batch."""
    z_values, activations = forward(x, weights, biases)
    p = activations[-1]
    batch_size = len(x)
    loss = -np.sum(y * np.log(p)) / batch_size

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
    """Shuffle, visit every example once, and update after each mini-batch."""
    order = rng.permutation(len(labels))
    total_loss = 0.0
    correct = 0
    for start in range(0, len(labels), batch_size):
        indices = order[start : start + batch_size]  # Includes the short last batch.
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

def predict_all(images, weights, biases, batch_size=256):
    """Predict class labels (0-9) for an array of images."""
    preds = []
    for start in range(0, len(images), batch_size):
        x = images[start : start + batch_size].astype(np.float64) / 255.0
        _, activations = forward(x, weights, biases)
        preds.append(np.argmax(activations[-1], axis=1))
    return np.concatenate(preds)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True, help="訓練資料夾路徑 (內含 0~9 子資料夾)")
    parser.add_argument("--test-dir", type=Path, default=None, help="助教 USB 給的測試資料夾路徑")
    parser.add_argument("--student-id", type=str, default="Student_ID", help="你的學號 (將輸出 學號.txt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hidden-layers", type=int, nargs="*", default=[256, 128])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    
    all_images, all_labels = load_training_pngs(args.train_dir)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_labels))
    all_images, all_labels = all_images[perm], all_labels[perm]

    weights, biases = initialize_parameters(tuple(args.hidden_layers), rng)
    print(f"載入訓練圖片總數: {len(all_labels)}, 網路結構: {(784, *args.hidden_layers, 10)}")
    
    for epoch in range(1, args.epochs + 1):
        mean_loss, train_acc = train_epoch(
            all_images, all_labels, weights, biases,
            args.batch_size, args.learning_rate, rng,
        )
        print(f"Epoch {epoch:02d}/{args.epochs:02d} - loss={mean_loss:.4f}, train_acc={train_acc:.2%}")

    if args.test_dir is not None:
        test_images, txt_names = load_testing_pngs(args.test_dir)
        predictions = predict_all(test_images, weights, biases, args.batch_size)
        result_path = f"{args.student_id}.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            for name, ans in zip(txt_names, predictions):
                f.write(f"{name} {int(ans)}\n")
        print(f"\n[完成] 已讀取 {len(txt_names)} 張測試圖，並成功輸出至: {result_path}")

if __name__ == "__main__":
    main()
