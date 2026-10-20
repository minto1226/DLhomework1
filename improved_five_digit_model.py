"""Train a NumPy neural network for the digit folders in a training directory.

Example:
    OPENBLAS_NUM_THREADS=1 python3 improved_five_digit_model.py \
        --train-dir data/Training_data2 --test-dir data/Testing_data3

The labeled holdout is used only to select the number of epochs. After that,
fresh networks are trained on all labeled images before predicting the test set.
The test folder is never used for training or model selection.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_grayscale_png(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((28, 28)), dtype=np.uint8)
    return pixels.reshape(784)


def load_training_pngs(train_dir):
    """Return normalized images, class indices, and sorted digit labels."""
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        raise ValueError(f"Training directory does not exist: {train_dir}")

    numeric_dirs = sorted(
        (child for child in train_dir.iterdir()
         if child.is_dir() and child.name.isdigit()),
        key=lambda path: int(path.name),
    )
    invalid_dirs = [path.name for path in numeric_dirs
                    if path.name not in {str(digit) for digit in range(10)}]
    if invalid_dirs:
        raise ValueError(f"Class directories must be named 0 through 9: {invalid_dirs}")
    digits = tuple(int(path.name) for path in numeric_dirs)
    if not digits:
        raise ValueError(f"No digit class directories (0 through 9) in {train_dir}")

    images = []
    labels = []
    for class_index, digit in enumerate(digits):
        class_dir = train_dir / str(digit)
        paths = sorted(
            (path for path in class_dir.iterdir()
             if path.is_file() and path.suffix.lower() == ".png"),
            key=lambda path: path.name,
        )
        if not paths:
            raise ValueError(f"No PNG images in {class_dir}")
        for path in paths:
            images.append(read_grayscale_png(path))
            labels.append(class_index)

    x = np.stack(images).astype(np.float32) / 255.0
    y = np.asarray(labels, dtype=np.int64)
    return x, y, digits


def load_testing_pngs(test_dir):
    """Return normalized images and their original filename stems."""
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise ValueError(f"Test directory does not exist: {test_dir}")
    paths = sorted(
        (path for path in test_dir.iterdir()
         if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError(f"No PNG images in {test_dir}")
    names = [path.stem for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Test image stems are not unique; output names would collide")
    x = np.stack([read_grayscale_png(path) for path in paths]).astype(np.float32) / 255.0
    return x, names


def stratified_split(labels, digits, seed=2026, val_per_class=400):
    """Choose exactly val_per_class indices per digit with a fixed RNG seed."""
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for class_index, digit in enumerate(digits):
        class_indices = np.flatnonzero(labels == class_index)
        if len(class_indices) <= val_per_class:
            raise ValueError(
                f"Digit {digit} has {len(class_indices)} images; "
                f"more than {val_per_class} are required"
            )
        shuffled = rng.permutation(class_indices)
        val_parts.append(shuffled[:val_per_class])
        train_parts.append(shuffled[val_per_class:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def initialize_parameters(rng, n_classes):
    weights = []
    biases = []
    layer_sizes = (784, 256, 128, n_classes)
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(
            (rng.standard_normal((fan_in, fan_out)).astype(np.float32)
             * np.float32(np.sqrt(2.0 / fan_in)))
        )
        biases.append(np.zeros((1, fan_out), dtype=np.float32))
    return weights, biases


def forward(x, weights, biases):
    z1 = x @ weights[0] + biases[0]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ weights[1] + biases[1]
    a2 = np.maximum(z2, 0)
    logits = a2 @ weights[2] + biases[2]
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return z1, a1, z2, a2, probabilities


def batch_gradients(x, y, weights, biases, weight_decay):
    z1, a1, z2, a2, probabilities = forward(x, weights, biases)
    rows = np.arange(len(y))
    loss = -np.log(np.maximum(probabilities[rows, y], 1e-12)).mean()

    # Softmax + cross-entropy; divide by the actual batch size, including the last batch.
    delta3 = probabilities.copy()
    delta3[rows, y] -= 1.0
    delta3 /= len(y)
    grad_w3 = a2.T @ delta3 + weight_decay * weights[2]
    grad_b3 = delta3.sum(axis=0, keepdims=True)

    delta2 = (delta3 @ weights[2].T) * (z2 > 0)
    grad_w2 = a1.T @ delta2 + weight_decay * weights[1]
    grad_b2 = delta2.sum(axis=0, keepdims=True)

    delta1 = (delta2 @ weights[1].T) * (z1 > 0)
    grad_w1 = x.T @ delta1 + weight_decay * weights[0]
    grad_b1 = delta1.sum(axis=0, keepdims=True)
    return float(loss), (grad_w1, grad_w2, grad_w3), (grad_b1, grad_b2, grad_b3)


class Adam:
    """First/second moments and bias correction, written directly in NumPy."""

    def __init__(self, weights, biases):
        self.parameters = [*weights, *biases]
        self.first = [np.zeros_like(parameter) for parameter in self.parameters]
        self.second = [np.zeros_like(parameter) for parameter in self.parameters]
        self.step_count = 0

    def step(self, weight_grads, bias_grads, learning_rate):
        self.step_count += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        correction1 = 1.0 - beta1 ** self.step_count
        correction2 = 1.0 - beta2 ** self.step_count
        for parameter, gradient, first, second in zip(
            self.parameters, (*weight_grads, *bias_grads), self.first, self.second
        ):
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * gradient * gradient
            first_unbiased = first / correction1
            second_unbiased = second / correction2
            parameter -= learning_rate * first_unbiased / (np.sqrt(second_unbiased) + epsilon)


def train_epoch(x, y, weights, biases, optimizer, batch_size, learning_rate,
                weight_decay, rng):
    order = rng.permutation(len(y))
    total_loss = 0.0
    for start in range(0, len(y), batch_size):
        indices = order[start:start + batch_size]
        loss, grad_weights, grad_biases = batch_gradients(
            x[indices], y[indices], weights, biases, weight_decay
        )
        optimizer.step(grad_weights, grad_biases, learning_rate)
        total_loss += loss * len(indices)
    return total_loss / len(y)


def evaluate(x, y, weights, biases, batch_size):
    total_loss = 0.0
    correct = 0
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        probabilities = forward(x[start:end], weights, biases)[-1]
        rows = np.arange(end - start)
        total_loss -= np.log(np.maximum(probabilities[rows, y[start:end]], 1e-12)).sum()
        correct += np.count_nonzero(probabilities.argmax(axis=1) == y[start:end])
    return float(total_loss / len(y)), correct / len(y)


def predict_probabilities(x, weights, biases, batch_size):
    probabilities = np.empty((len(x), weights[-1].shape[1]), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        probabilities[start:end] = forward(x[start:end], weights, biases)[-1]
    return probabilities


def choose_epochs(x, y, train_indices, val_indices, max_epochs, batch_size,
                  learning_rate, weight_decay, patience, seed, n_classes):
    rng = np.random.default_rng(seed)
    train_x, train_y = x[train_indices], y[train_indices]
    val_x, val_y = x[val_indices], y[val_indices]
    weights, biases = initialize_parameters(rng, n_classes)
    optimizer = Adam(weights, biases)
    best_epoch = 0
    best_accuracy = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            train_x, train_y, weights, biases, optimizer,
            batch_size, learning_rate, weight_decay, rng
        )
        val_loss, val_accuracy = evaluate(
            val_x, val_y, weights, biases, batch_size
        )
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.2%}",
            flush=True,
        )
        if (val_accuracy > best_accuracy or
                (val_accuracy == best_accuracy and val_loss < best_loss)):
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    return best_epoch, best_accuracy, best_loss


def fit_all(x, y, epochs, batch_size, learning_rate, weight_decay, seed,
            n_classes):
    rng = np.random.default_rng(seed)
    weights, biases = initialize_parameters(rng, n_classes)
    optimizer = Adam(weights, biases)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            x, y, weights, biases, optimizer, batch_size, learning_rate,
            weight_decay, rng,
        )
        print(f"Refit seed={seed}, epoch {epoch:02d}/{epochs:02d}: train_loss={loss:.4f}",
              flush=True)
    return weights, biases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/Training_data2"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/Testing_data3"))
    parser.add_argument("--output", type=Path,
                        default=Path("711580103_improved_training2.txt"))
    parser.add_argument("--epochs", type=int, default=40, help="Maximum validation epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-per-class", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=1,
                        help="Number of independently refitted models to average")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.val_per_class < 1 or
            args.learning_rate <= 0 or args.weight_decay < 0 or
            args.patience < 1 or args.ensemble_size < 1):
        parser.error("Epochs, batch size, validation count, learning rate, patience, "
                     "and ensemble size must be positive; weight decay must be nonnegative")
    if args.output.resolve() == Path("711580103.txt").resolve():
        parser.error("Output must differ from the original 711580103.txt")

    try:
        x, y, digits = load_training_pngs(args.train_dir)
        train_indices, val_indices = stratified_split(
            y, digits, args.seed, args.val_per_class
        )
        test_x, test_names = load_testing_pngs(args.test_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        f"Loaded {len(train_indices)} training, {len(val_indices)} validation, "
        f"and {len(test_x)} test PNGs; classes={digits}", flush=True,
    )

    best_epoch, best_accuracy, best_loss = choose_epochs(
        x, y, train_indices, val_indices, args.epochs, args.batch_size,
        args.learning_rate, args.weight_decay, args.patience, args.seed,
        len(digits),
    )
    print(
        f"Selected epoch {best_epoch}: validation_accuracy={best_accuracy:.2%}, "
        f"validation_loss={best_loss:.4f}", flush=True,
    )

    mean_probabilities = np.zeros((len(test_x), len(digits)), dtype=np.float32)
    for model_number in range(args.ensemble_size):
        weights, biases = fit_all(
            x, y, best_epoch, args.batch_size, args.learning_rate,
            args.weight_decay, args.seed + 1000 + model_number,
            len(digits),
        )
        mean_probabilities += predict_probabilities(
            test_x, weights, biases, args.batch_size
        ) / args.ensemble_size
    predictions = np.asarray(digits, dtype=np.uint8)[mean_probabilities.argmax(axis=1)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as result:
        for stem, prediction in zip(test_names, predictions):
            result.write(f"{stem} {int(prediction)}\n")
    print(f"Wrote {len(predictions)} predictions to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
