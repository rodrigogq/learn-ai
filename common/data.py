"""Shared dataset download helpers for the course.

All chapters store downloaded data under one repo-level "datasets/" folder
(ignored by git), so switching between chapters never re-downloads anything
and cleaning up is a single folder deletion.
"""

from pathlib import Path

# The datasets folder lives at the repository root regardless of which chapter
# script is running, so we anchor it to this file's location instead of the
# current working directory.
REPOSITORY_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
DATASETS_DIRECTORY = REPOSITORY_ROOT_DIRECTORY / "datasets"


def get_datasets_directory() -> Path:
    """Return the shared datasets directory, creating it on first use."""
    DATASETS_DIRECTORY.mkdir(exist_ok=True)
    return DATASETS_DIRECTORY


def load_mnist_datasets(flatten_images_to_vectors: bool = False):
    """Download (once) and return the MNIST train and test datasets.

    MNIST is the classic dataset of 70,000 handwritten digits (28x28 grayscale
    images, labels 0-9). It is small (~12 MB) and trains in minutes, which is
    why the course uses it for first experiments.

    Arguments:
        flatten_images_to_vectors: when True, each 28x28 image is returned as a
            flat vector of 784 numbers. Plain fully-connected networks
            (chapters 9-11) need vectors; convolutional networks (chapter 13+)
            want the 2D image, so this defaults to False.

    Returns:
        A pair (training_dataset, test_dataset) of torchvision datasets whose
        items are (image_tensor, integer_label).
    """
    from torchvision import datasets, transforms

    transform_steps = [transforms.ToTensor()]
    if flatten_images_to_vectors:
        transform_steps.append(transforms.Lambda(lambda image_tensor: image_tensor.reshape(-1)))
    image_transform = transforms.Compose(transform_steps)

    storage_directory = str(get_datasets_directory())
    training_dataset = datasets.MNIST(storage_directory, train=True, download=True, transform=image_transform)
    test_dataset = datasets.MNIST(storage_directory, train=False, download=True, transform=image_transform)
    return training_dataset, test_dataset


def load_tiny_shakespeare():
    """Download (once) and return the tiny-Shakespeare corpus as a string.

    One megabyte of concatenated Shakespeare plays - the traditional first
    corpus for language models (Chapters 20-23): real English, tiny download,
    and results you can judge by reading them.
    """
    from urllib.request import urlopen

    corpus_path = get_datasets_directory() / "tinyshakespeare.txt"
    if not corpus_path.exists():
        corpus_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading tiny Shakespeare (~1 MB) from {corpus_url} ...")
        corpus_path.write_bytes(urlopen(corpus_url).read())
    return corpus_path.read_text(encoding="utf-8")


def load_cifar10_datasets():
    """Download (once) and return the CIFAR-10 train and test datasets.

    CIFAR-10 contains 60,000 small color photos (32x32 pixels) in 10 classes
    such as airplane, cat, and truck. It is the course's step up from MNIST:
    real photos, but still small enough to train on any machine.

    Returns:
        A pair (training_dataset, test_dataset) of torchvision datasets whose
        items are (image_tensor, integer_label).
    """
    from torchvision import datasets, transforms

    storage_directory = str(get_datasets_directory())
    image_transform = transforms.ToTensor()
    training_dataset = datasets.CIFAR10(storage_directory, train=True, download=True, transform=image_transform)
    test_dataset = datasets.CIFAR10(storage_directory, train=False, download=True, transform=image_transform)
    return training_dataset, test_dataset
