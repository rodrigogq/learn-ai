"""Chapter 9 - export MNIST as flat binary files for the C training program.

C should not have to parse dataset formats, so this script writes the images
and labels as raw bytes: each image is 784 unsigned bytes (brightness 0-255,
row by row), each label one unsigned byte (0-9). The C program derives the
image count from the file size - no header needed.

Run from the repository root (before the first run of the C program):
    .venv/bin/python chapters/09-first-neural-network/python/export_mnist_for_c.py
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import get_datasets_directory, load_mnist_datasets  # noqa: E402


def export_split(dataset, images_file_path, labels_file_path):
    """Write one dataset split as two flat binary files.

    Arguments:
        dataset: a torchvision MNIST dataset (train or test split).
        images_file_path: destination for count*784 raw uint8 pixels.
        labels_file_path: destination for count raw uint8 labels.
    """
    dataset.data.numpy().astype("uint8").tofile(images_file_path)
    dataset.targets.numpy().astype("uint8").tofile(labels_file_path)
    print(f"  wrote {images_file_path.name} ({images_file_path.stat().st_size:,} bytes) "
          f"and {labels_file_path.name}")


def main():
    print("Loading MNIST (downloads ~12 MB on first run)...")
    training_dataset, test_dataset = load_mnist_datasets()
    datasets_directory = get_datasets_directory()

    print("Exporting for the C program:")
    export_split(training_dataset,
                 datasets_directory / "mnist_train_images.bin",
                 datasets_directory / "mnist_train_labels.bin")
    export_split(test_dataset,
                 datasets_directory / "mnist_test_images.bin",
                 datasets_directory / "mnist_test_labels.bin")
    print(f"Done. The C program reads them from {datasets_directory}/")


if __name__ == "__main__":
    main()
