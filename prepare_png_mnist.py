"""Convert the downloaded MNIST IDX files into the PNG folder layout.

This is only a data-format helper; the network is implemented in
coursework_mlp.py. Run it after downloading the four IDX .gz files described
in README.md.
"""

import argparse
from pathlib import Path

from PIL import Image

from phase2_train import load_mnist


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
