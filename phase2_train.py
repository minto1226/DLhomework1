"""Phase 2: train a NumPy multilayer network on local MNIST IDX files.

Example:
    python3 phase2_train.py --hidden-layers 256 128 --epochs 5

The four original MNIST .gz files must already be in data/mnist. This script
does not download data or use a neural-network framework/autodiff.
"""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np


def read_images(path):
    """Read IDX image bytes; normalize one batch at a time during training."""
    with gzip.open(path, "rb") as file:
        data = file.read()
    if len(data) < 16:
        raise ValueError(f"{path}: image header is incomplete")
    magic, count, rows, columns = struct.unpack(">IIII", data[:16])
    if magic != 2051 or (rows, columns) != (28, 28):
        raise ValueError(f"{path}: expected IDX images with shape 28 x 28")
    if count == 0 or len(data) != 16 + count * rows * columns:
        raise ValueError(f"{path}: image count or payload size is invalid")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, 784)


def read_labels(path):
    """Read and validate IDX digit labels."""
    with gzip.open(path, "rb") as file:
        data = file.read()
    if len(data) < 8:
        raise ValueError(f"{path}: label header is incomplete")
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"{path}: expected IDX labels")
    if count == 0 or len(data) != 8 + count:
        raise ValueError(f"{path}: label count or payload size is invalid")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    if np.any(labels > 9):
        raise ValueError(f"{path}: labels must be digits from 0 to 9")
    return labels


def load_mnist(data_dir):
    """Return train images/labels and test images/labels as uint8 arrays."""
    data_dir = Path(data_dir)
    train_images = read_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = read_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    if len(train_images) != len(train_labels):
        raise ValueError("train image and label counts differ")
    if len(test_images) != len(test_labels):
        raise ValueError("test image and label counts differ")
    return train_images, train_labels, test_images, test_labels


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


def accuracy(images, labels, weights, biases, batch_size):
    """Evaluate the current parameters without updating them."""
    correct = 0
    for start in range(0, len(labels), batch_size):
        x = images[start : start + batch_size].astype(np.float64) / 255.0
        _, activations = forward(x, weights, biases)
        predictions = np.argmax(activations[-1], axis=1)
        correct += np.count_nonzero(predictions == labels[start : start + batch_size])
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-layers", type=int, nargs="*", default=[128])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if any(width <= 0 for width in args.hidden_layers):
        parser.error("hidden-layer widths must be positive")

    try:
        train_images, train_labels, test_images, test_labels = load_mnist(args.data_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"MNIST data error: {error}\n")

    rng = np.random.default_rng(args.seed)
    weights, biases = initialize_parameters(tuple(args.hidden_layers), rng)
    layer_sizes = (784, *args.hidden_layers, 10)
    print(f"network: {layer_sizes}")
    print(f"examples: train={len(train_labels)}, test={len(test_labels)}")
    for epoch in range(1, args.epochs + 1):
        mean_loss, train_accuracy = train_epoch(
            train_images, train_labels, weights, biases,
            args.batch_size, args.learning_rate, rng,
        )
        # These train metrics were collected as the weights changed in this epoch.
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train mean loss={mean_loss:.4f}, "
            f"train accuracy during updates={train_accuracy:.2%}"
        )

    test_accuracy = accuracy(test_images, test_labels, weights, biases, args.batch_size)
    print(f"final test accuracy={test_accuracy:.2%}")


if __name__ == "__main__":
    main()
