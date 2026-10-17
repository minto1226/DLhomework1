"""Train a from-scratch MLP on labeled PNG digits and predict a PNG folder.

Expected training layout: train/0/*.png, train/1/*.png, ..., train/9/*.png.
Prediction images are directly inside one folder, for example test/0000.png.
Only NumPy handles network mathematics; Pillow reads the PNG files.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


EPSILON = 1e-12
DEFAULT_TRAIN_DIR = Path(__file__).resolve().parent / "data/mnist_png/train"


def read_png(path):
    """Return one 28 x 28 grayscale PNG as a flattened uint8 row."""
    with Image.open(path) as image:
        if image.size != (28, 28):
            raise ValueError(f"{path}: expected a 28 x 28 PNG, got {image.size}")
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    return pixels.reshape(784)


def load_labeled_pngs(folder):
    """Use the parent digit directory as each PNG's integer label."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"training/validation folder does not exist: {folder}")
    images = []
    labels = []
    for digit in range(10):
        digit_folder = folder / str(digit)
        if not digit_folder.is_dir():
            continue
        paths = sorted(
            path for path in digit_folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        for path in paths:
            images.append(read_png(path))
            labels.append(digit)
    if not images:
        raise ValueError(f"no labeled PNGs found under {folder}/0 ... {folder}/9")
    return np.stack(images), np.asarray(labels, dtype=np.uint8)


def list_prediction_pngs(folder):
    """Read every PNG directly in the folder, in filename order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"prediction folder does not exist: {folder}")
    paths = sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError(f"no PNG files found in {folder}")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError(f"duplicate PNG filename stems in {folder}")
    return paths


def split_validation(images, labels, ratio, rng):
    """Make one fixed validation holdout before the epoch loop."""
    if len(labels) < 2:
        raise ValueError("at least two labeled images are needed for a validation split")
    order = rng.permutation(len(labels))
    validation_count = min(len(labels) - 1, max(1, int(len(labels) * ratio)))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    return (
        images[training_indices], labels[training_indices],
        images[validation_indices], labels[validation_indices],
    )


def initialize_parameters(hidden_layers, rng):
    """He initialization for 784 -> hidden_layers -> 10."""
    if any(width <= 0 for width in hidden_layers):
        raise ValueError("hidden-layer widths must be positive")
    widths = (784, *hidden_layers, 10)
    weights = []
    biases = []
    for fan_in, fan_out in zip(widths[:-1], widths[1:]):
        weights.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
        biases.append(np.zeros((1, fan_out)))
    return weights, biases


def forward(x, weights, biases):
    """Save every z and activation so the backward loop can use them."""
    z_values = []
    activations = [x]
    for layer in range(len(weights)):
        z = activations[-1] @ weights[layer] + biases[layer]
        z_values.append(z)
        if layer == len(weights) - 1:
            shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(shifted)
            a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            a = np.maximum(0.0, z)
        activations.append(a)
    return z_values, activations


def make_one_hot(labels):
    """Keep the class-label matrix visible in the backpropagation formula."""
    y = np.zeros((len(labels), 10))
    for row, digit in enumerate(labels):
        y[row, int(digit)] = 1.0
    return y


def cross_entropy(p, y):
    """Average cross-entropy with 1e-12 added before taking the logarithm."""
    return -np.sum(y * np.log(p + EPSILON)) / len(p)


def batch_gradients(x, labels, weights, biases):
    """Calculate all layer gradients before any parameter is changed."""
    z_values, activations = forward(x, weights, biases)
    p = activations[-1]
    y = make_one_hot(labels)
    loss = cross_entropy(p, y)

    # B examples: (B, 10). This is the standard Softmax-CE gradient.
    # The 1e-12 in the displayed loss only changes its derivative near p = 0.
    # Divide by the size of this actual batch, including a short last batch.
    delta = (p - y) / len(x)
    d_weights = [None] * len(weights)
    d_biases = [None] * len(biases)
    for layer in range(len(weights) - 1, -1, -1):
        d_weights[layer] = activations[layer].T @ delta
        d_biases[layer] = np.sum(delta, axis=0, keepdims=True)
        if layer > 0:
            delta = (delta @ weights[layer].T) * (z_values[layer - 1] > 0.0)
    return loss, d_weights, d_biases


def train_epoch(images, labels, weights, biases, batch_size, learning_rate, rng):
    """Shuffle once, then update after each mini-batch, including the last one."""
    order = rng.permutation(len(labels))
    for start in range(0, len(labels), batch_size):
        indices = order[start:start + batch_size]
        x = images[indices].astype(np.float64) / 255.0
        batch_labels = labels[indices]
        _, d_weights, d_biases = batch_gradients(x, batch_labels, weights, biases)
        for layer in range(len(weights)):
            weights[layer] -= learning_rate * d_weights[layer]
            biases[layer] -= learning_rate * d_biases[layer]


def evaluate(images, labels, weights, biases, batch_size):
    """Measure loss and accuracy with the current parameters, without updates."""
    loss_total = 0.0
    correct = 0
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        x = images[start:stop].astype(np.float64) / 255.0
        batch_labels = labels[start:stop]
        _, activations = forward(x, weights, biases)
        p = activations[-1]
        loss_total += cross_entropy(p, make_one_hot(batch_labels)) * len(batch_labels)
        correct += np.count_nonzero(np.argmax(p, axis=1) == batch_labels)
    return loss_total / len(labels), correct / len(labels)


def write_predictions(folder, student_id, weights, biases, batch_size):
    """Write sorted 'filename<TAB>digit' lines, without a header."""
    paths = list_prediction_pngs(folder)
    lines = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        x = np.stack([read_png(path) for path in batch_paths]).astype(np.float64) / 255.0
        _, activations = forward(x, weights, biases)
        predictions = np.argmax(activations[-1], axis=1)
        for path, digit in zip(batch_paths, predictions):
            lines.append(f"{path.stem}\t{int(digit)}\n")
    output_path = Path(f"{student_id}.txt")
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path, len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir", type=Path, default=DEFAULT_TRAIN_DIR,
        help="labeled PNG folder (default: data/mnist_png/train beside this script)",
    )
    parser.add_argument("--val-dir", type=Path, help="optional labeled validation folder")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--predict-dir", type=Path, help="flat folder of unlabeled PNGs")
    parser.add_argument("--student-id", help="output filename is STUDENT_ID.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--anneal-factor", type=float, default=0.95)
    parser.add_argument("--hidden-layers", type=int, nargs="*", default=[128])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if not 0.0 < args.anneal_factor <= 1.0:
        parser.error("anneal-factor must be in (0, 1]")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("val-ratio must be in (0, 1)")
    if any(width <= 0 for width in args.hidden_layers):
        parser.error("hidden-layer widths must be positive")
    if (args.predict_dir is None) != (args.student_id is None):
        parser.error("predict-dir and student-id must be given together")
    if args.student_id is not None and (
        not args.student_id or
        not all(character.isalnum() or character in "_-" for character in args.student_id)
    ):
        parser.error("student-id may contain only letters, digits, underscores, or hyphens")
    if args.val_dir is not None and args.train_dir.resolve() == args.val_dir.resolve():
        parser.error("train-dir and val-dir must be different folders")
    if args.train_dir == DEFAULT_TRAIN_DIR and not args.train_dir.is_dir():
        parser.exit(
            2,
            f"default training folder not found: {DEFAULT_TRAIN_DIR}\n"
            "Run 'python3 prepare_png_mnist.py' after downloading MNIST, "
            "or provide --train-dir PATH for another dataset.\n",
        )

    try:
        print(f"loading training PNGs from {args.train_dir}...", flush=True)
        if args.predict_dir is None:
            print("training and validation only; use --predict-dir and --student-id together to write predictions", flush=True)
        images, labels = load_labeled_pngs(args.train_dir)
        rng = np.random.default_rng(args.seed)
        if args.val_dir is None:
            train_images, train_labels, val_images, val_labels = split_validation(
                images, labels, args.val_ratio, rng,
            )
        else:
            train_images, train_labels = images, labels
            val_images, val_labels = load_labeled_pngs(args.val_dir)
        weights, biases = initialize_parameters(tuple(args.hidden_layers), rng)
        print(f"network: {(784, *args.hidden_layers, 10)}")
        print(f"examples: train={len(train_labels)}, validation={len(val_labels)}")
        for epoch in range(1, args.epochs + 1):
            learning_rate = args.learning_rate * args.anneal_factor ** (epoch - 1)
            train_epoch(
                train_images, train_labels, weights, biases,
                args.batch_size, learning_rate, rng,
            )
            train_loss, train_accuracy = evaluate(
                train_images, train_labels, weights, biases, args.batch_size,
            )
            _, validation_accuracy = evaluate(
                val_images, val_labels, weights, biases, args.batch_size,
            )
            print(
                f"epoch {epoch}/{args.epochs}: lr={learning_rate:.6f}, "
                f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.2%}, "
                f"validation accuracy={validation_accuracy:.2%}"
            )
        if args.predict_dir is not None:
            output_path, count = write_predictions(
                args.predict_dir, args.student_id, weights, biases, args.batch_size,
            )
            print(f"wrote {count} predictions to {output_path}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"data error: {error}\n")


if __name__ == "__main__":
    main()
