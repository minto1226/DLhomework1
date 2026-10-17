"""Convert the downloaded MNIST IDX files into the PNG folder layout.

This is only a data-format helper; the networks are implemented in
phase2_train.py and coursework_mlp.py. Run it after downloading the four IDX
.gz files described in README.md.
"""

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def read_idx_images(path):
    """Read an original MNIST image .gz file as (count, 784) bytes."""
    with gzip.open(path, "rb") as file:
        data = file.read()
    if len(data) < 16:
        raise ValueError(f"{path}: incomplete image header")
    magic, count, rows, columns = struct.unpack(">IIII", data[:16])
    if magic != 2051 or (rows, columns) != (28, 28):
        raise ValueError(f"{path}: expected 28 x 28 IDX images")
    if count == 0 or len(data) != 16 + count * rows * columns:
        raise ValueError(f"{path}: invalid image count or payload size")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, 784)


def read_idx_labels(path):
    """Read an original MNIST label .gz file as digit bytes."""
    with gzip.open(path, "rb") as file:
        data = file.read()
    if len(data) < 8:
        raise ValueError(f"{path}: incomplete label header")
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"{path}: expected IDX labels")
    if count == 0 or len(data) != 8 + count:
        raise ValueError(f"{path}: invalid label count or payload size")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    if np.any(labels > 9):
        raise ValueError(f"{path}: labels must be digits from 0 to 9")
    return labels


def load_mnist(idx_dir):
    """Read and pair the four local IDX files without importing training code."""
    idx_dir = Path(idx_dir)
    train_images = read_idx_images(idx_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(idx_dir / "train-labels-idx1-ubyte.gz")
    test_images = read_idx_images(idx_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(idx_dir / "t10k-labels-idx1-ubyte.gz")
    if len(train_images) != len(train_labels):
        raise ValueError("training image and label counts differ")
    if len(test_images) != len(test_labels):
        raise ValueError("test image and label counts differ")
    return train_images, train_labels, test_images, test_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mnist_png"))
    parser.add_argument("--train-count", type=int, help="optional smaller training sample")
    parser.add_argument("--test-count", type=int, help="optional smaller prediction sample")
    args = parser.parse_args()

    try:
        train_images, train_labels, test_images, _ = load_mnist(args.idx_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"MNIST data error: {error}\n")
    train_count = len(train_images) if args.train_count is None else args.train_count
    test_count = len(test_images) if args.test_count is None else args.test_count
    if not 1 <= train_count <= len(train_images):
        parser.error(f"train-count must be from 1 to {len(train_images)}")
    if not 1 <= test_count <= len(test_images):
        parser.error(f"test-count must be from 1 to {len(test_images)}")

    for index in range(train_count):
        digit_folder = args.output_dir / "train" / str(int(train_labels[index]))
        digit_folder.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(train_images[index].reshape(28, 28))
        image.save(digit_folder / f"{index:05d}.png")
    test_folder = args.output_dir / "test"
    test_folder.mkdir(parents=True, exist_ok=True)
    for index in range(test_count):
        image = Image.fromarray(test_images[index].reshape(28, 28))
        image.save(test_folder / f"{index:04d}.png")
    print(f"wrote {train_count} labeled PNGs and {test_count} test PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
