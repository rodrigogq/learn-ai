"""Chapter 12 - data pipelines: splits, augmentation, and honest metrics.

Three demonstrations:
  1. a proper train/validation split, with validation-driven early stopping,
  2. data augmentation rescuing Chapter 11's overfitting setup: the same
     1,000 images, randomly shifted/rotated each epoch, act like more data,
  3. a confusion matrix: WHERE a digit classifier goes wrong, not just how often.

Run from the repository root:
    .venv/bin/python chapters/12-data-pipelines/python/data_pipelines.py
    .venv/bin/python chapters/12-data-pipelines/python/data_pipelines.py --quick
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import get_datasets_directory  # noqa: E402
from common.device import select_best_available_device  # noqa: E402

# Chapter 11's overfitting setup, reused on purpose so the comparisons are fair.
SMALL_TRAINING_SIZE = 1000

FLATTEN_TRANSFORM = transforms.Lambda(lambda image_tensor: image_tensor.reshape(-1))
PLAIN_TRANSFORM = transforms.Compose([transforms.ToTensor(), FLATTEN_TRANSFORM])

# The augmentation policy: small random rotations, shifts, and zooms - changes
# a human would not even notice, but every epoch the network sees a slightly
# different version of each image. Free "new" data, manufactured on the fly.
AUGMENTED_TRANSFORM = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    FLATTEN_TRANSFORM,
])


def build_classifier():
    """Chapter 11's defended network: 784 -> 256 ReLU + dropout -> 10."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10),
    )


def measure_accuracy(model, data_loader, device):
    """Fraction of correct predictions; eval() disables dropout while measuring."""
    model.eval()
    correct_count = total_count = 0
    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            correct_count += (model(image_batch).argmax(dim=1) == label_batch).sum().item()
            total_count += len(label_batch)
    model.train()
    return correct_count / total_count


def train_epochs(model, training_loader, optimizer, device, number_of_epochs,
                 per_epoch_callback=None):
    """The eternal loop for a fixed number of epochs.

    Arguments:
        model, training_loader, optimizer, device: the usual suspects.
        number_of_epochs: how many passes to run.
        per_epoch_callback: optional function called as callback(epoch_number)
            after each epoch - used by the early-stopping demo to peek at
            validation accuracy.
    """
    loss_function = nn.CrossEntropyLoss()
    for epoch_number in range(1, number_of_epochs + 1):
        for image_batch, label_batch in training_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            loss = loss_function(model(image_batch), label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if per_epoch_callback is not None:
            per_epoch_callback(epoch_number)


def demonstrate_validation_split(device, number_of_epochs):
    """Demo 1: split off a validation set and let IT decide when to stop."""
    print("1. Train/validation split with early stopping")
    print(f"   {SMALL_TRAINING_SIZE} labeled images -> 800 train / 200 validation (test set stays untouched)")

    full_dataset = datasets.MNIST(str(get_datasets_directory()), train=True, transform=PLAIN_TRANSFORM)
    training_part = Subset(full_dataset, range(800))
    validation_part = Subset(full_dataset, range(800, SMALL_TRAINING_SIZE))
    test_dataset = datasets.MNIST(str(get_datasets_directory()), train=False, transform=PLAIN_TRANSFORM)

    training_loader = DataLoader(training_part, batch_size=100, shuffle=True)
    validation_loader = DataLoader(validation_part, batch_size=200)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    torch.manual_seed(42)
    model = build_classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_validation_accuracy = 0.0
    best_epoch = 0
    # Keeping a copy of the best weights is what "early stopping" means in
    # practice: train on, but remember the model as it was at its peak.
    best_model_state = None

    def check_validation(epoch_number):
        nonlocal best_validation_accuracy, best_epoch, best_model_state
        validation_accuracy = measure_accuracy(model, validation_loader, device)
        marker = ""
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch_number
            best_model_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            marker = "  <- new best, weights saved"
        if epoch_number in (1, 5, 10, 20, number_of_epochs) or marker:
            print(f"   epoch {epoch_number:>3}: validation {validation_accuracy:6.2%}{marker}")

    train_epochs(model, training_loader, optimizer, device, number_of_epochs, check_validation)

    model.load_state_dict(best_model_state)
    print(f"   -> best validation was epoch {best_epoch}; restoring those weights.")
    print(f"   Final, single measurement on the test set: {measure_accuracy(model, test_loader, device):.2%}")


def demonstrate_augmentation(device, number_of_epochs):
    """Demo 2: augmentation turns 1,000 images into endless variations."""
    print()
    print("2. Data augmentation on Chapter 11's 1,000-image setup")
    print("   Same images, but each epoch every image is randomly rotated (10 deg),")
    print("   shifted (10%), and zoomed (10%) - the network never sees the exact same pixel twice.")

    augmented_training = Subset(
        datasets.MNIST(str(get_datasets_directory()), train=True, transform=AUGMENTED_TRANSFORM),
        range(SMALL_TRAINING_SIZE),
    )
    test_dataset = datasets.MNIST(str(get_datasets_directory()), train=False, transform=PLAIN_TRANSFORM)
    training_loader = DataLoader(augmented_training, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    torch.manual_seed(42)
    model = build_classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    def report(epoch_number):
        if epoch_number in (10, 20, number_of_epochs):
            print(f"   epoch {epoch_number:>3}: test {measure_accuracy(model, test_loader, device):6.2%}")

    train_epochs(model, training_loader, optimizer, device, number_of_epochs, report)
    print("   -> Chapter 11 got 88.1% plain and 88.8% with dropout+decay on these images.")
    print("      Augmentation beats both - manufactured variety acts like more data.")


def demonstrate_confusion_matrix(device, number_of_epochs):
    """Demo 3: the confusion matrix - where the mistakes actually live."""
    print()
    print("3. Confusion matrix (full 60,000-image training, then analyzed on the test set)")

    training_dataset = datasets.MNIST(str(get_datasets_directory()), train=True, transform=PLAIN_TRANSFORM)
    test_dataset = datasets.MNIST(str(get_datasets_directory()), train=False, transform=PLAIN_TRANSFORM)
    training_loader = DataLoader(training_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    torch.manual_seed(42)
    model = build_classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train_epochs(model, training_loader, optimizer, device, number_of_epochs)

    # confusion[true][predicted] counts every (truth, guess) pair - accuracy
    # is just the diagonal, but the OFF-diagonal cells tell you what to fix.
    confusion_counts = torch.zeros(10, 10, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for image_batch, label_batch in test_loader:
            predictions = model(image_batch.to(device)).argmax(dim=1).cpu()
            for true_label, predicted_label in zip(label_batch, predictions):
                confusion_counts[true_label, predicted_label] += 1

    print(f"   overall test accuracy: {confusion_counts.diagonal().sum().item() / confusion_counts.sum().item():.2%}")
    print()
    print("   true\\pred " + " ".join(f"{digit:>5}" for digit in range(10)))
    for true_digit in range(10):
        row_text = " ".join(f"{confusion_counts[true_digit, predicted].item():>5}" for predicted in range(10))
        print(f"        {true_digit}  {row_text}")

    off_diagonal = confusion_counts.clone()
    off_diagonal.fill_diagonal_(0)
    worst_pair = (off_diagonal == off_diagonal.max()).nonzero()[0]
    print()
    print(f"   Most common mistake: true {worst_pair[0].item()} predicted as {worst_pair[1].item()} "
          f"({off_diagonal.max().item()} times) - look at the pair and you will sympathize.")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="fewer epochs, same story")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    long_epochs = 10 if parsed_arguments.quick else 40
    short_epochs = 2 if parsed_arguments.quick else 5

    print()
    demonstrate_validation_split(device, long_epochs)
    demonstrate_augmentation(device, long_epochs)
    demonstrate_confusion_matrix(device, short_epochs)


if __name__ == "__main__":
    main()
