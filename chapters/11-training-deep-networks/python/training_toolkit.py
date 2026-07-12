"""Chapter 11 - the training toolkit: optimizers, overfitting, and the fixes.

Four experiments on MNIST:
  1. optimizer race: plain SGD vs SGD+momentum vs Adam on the same network,
  2. overfitting on purpose: a large network trained on only 1,000 images
     memorizes them (train accuracy ~100%) while test accuracy stalls,
  3. the classic defenses: dropout + weight decay help - modestly,
  4. the real cure: the same defended network on the full 60,000 images.

Run from the repository root:
    .venv/bin/python chapters/11-training-deep-networks/python/training_toolkit.py
    .venv/bin/python chapters/11-training-deep-networks/python/training_toolkit.py --quick
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_mnist_datasets  # noqa: E402
from common.device import select_best_available_device  # noqa: E402


def build_classifier(hidden_size, dropout_probability=0.0):
    """A 784 -> hidden -> 10 network, optionally with dropout after the ReLU.

    Arguments:
        hidden_size: number of hidden neurons.
        dropout_probability: fraction of hidden activations randomly zeroed
            during training (0.0 = no dropout). nn.Dropout switches itself off
            automatically in eval() mode.
    """
    return nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_probability),
        nn.Linear(hidden_size, 10),
    )


def run_training(model, training_loader, test_loader, optimizer, device, number_of_epochs,
                 epochs_to_report):
    """Train a model and report train/test accuracy at chosen epochs.

    Arguments:
        model: the network to train (moved to device here).
        training_loader, test_loader: the data.
        optimizer: any torch.optim optimizer, already bound to model.parameters().
        device: where to compute.
        number_of_epochs: total epochs to run.
        epochs_to_report: epochs at which to print train/test accuracy.

    Returns (final_train_accuracy, final_test_accuracy).
    """
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    train_accuracy = test_accuracy = 0.0
    for epoch_number in range(1, number_of_epochs + 1):
        for image_batch, label_batch in training_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            loss = loss_function(model(image_batch), label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch_number in epochs_to_report:
            train_accuracy = measure_accuracy(model, training_loader, device)
            test_accuracy = measure_accuracy(model, test_loader, device)
            print(f"    epoch {epoch_number:>3}:  train {train_accuracy:6.2%}   test {test_accuracy:6.2%}"
                  f"   gap {train_accuracy - test_accuracy:+6.2%}")
    return train_accuracy, test_accuracy


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


def optimizer_race(training_loader, test_loader, device, number_of_epochs):
    """Experiment 1: same network, same data, three optimizers."""
    print("1. Optimizer race (same network and data, only the optimizer changes)")
    contenders = [
        ("plain SGD, lr 0.1", lambda parameters: torch.optim.SGD(parameters, lr=0.1)),
        ("SGD + momentum 0.9, lr 0.1", lambda parameters: torch.optim.SGD(parameters, lr=0.1, momentum=0.9)),
        ("Adam, lr 0.001", lambda parameters: torch.optim.Adam(parameters, lr=0.001)),
    ]
    for optimizer_name, make_optimizer in contenders:
        # A fresh identically-initialized model per contender keeps the race fair.
        torch.manual_seed(42)
        model = build_classifier(hidden_size=128)
        print(f"  {optimizer_name}:")
        run_training(model, training_loader, test_loader, make_optimizer(model.parameters()),
                     device, number_of_epochs, epochs_to_report={1, number_of_epochs})


def overfitting_demonstration(small_training_loader, test_loader, device, number_of_epochs, epochs_to_report):
    """Experiments 2 and 3: overfit a big network on 1,000 images, then fix it."""
    print()
    print("2. Overfitting on purpose: 1,000 training images, 256 hidden neurons, no defenses")
    torch.manual_seed(42)
    plain_model = build_classifier(hidden_size=256)
    plain_optimizer = torch.optim.Adam(plain_model.parameters(), lr=0.001)
    run_training(plain_model, small_training_loader, test_loader, plain_optimizer,
                 device, number_of_epochs, epochs_to_report)
    print("    -> training accuracy reaches ~100% (memorized) while test accuracy stalls: overfitting.")

    print()
    print("3. Same setup + dropout 0.5 + weight decay 1e-4")
    torch.manual_seed(42)
    defended_model = build_classifier(hidden_size=256, dropout_probability=0.5)
    defended_optimizer = torch.optim.Adam(defended_model.parameters(), lr=0.001, weight_decay=1e-4)
    run_training(defended_model, small_training_loader, test_loader, defended_optimizer,
                 device, number_of_epochs, epochs_to_report)
    print("    -> better test accuracy and a slower-growing gap - real help, but no miracle.")
    print("       With only 1,000 images there is only so much any defense can do.")


def more_data_demonstration(full_training_loader, test_loader, device, number_of_epochs):
    """Experiment 4: the strongest regularizer ever discovered is more data."""
    print()
    print("4. The real cure: the same defended network, but all 60,000 training images")
    torch.manual_seed(42)
    model = build_classifier(hidden_size=256, dropout_probability=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    run_training(model, full_training_loader, test_loader, optimizer,
                 device, number_of_epochs, epochs_to_report={1, number_of_epochs})
    print("    -> the gap nearly vanishes and test accuracy jumps: data beats every trick.")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="fewer epochs, same story")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    print("Loading MNIST...")
    training_dataset, test_dataset = load_mnist_datasets(flatten_images_to_vectors=True)

    race_epochs = 1 if parsed_arguments.quick else 3
    overfit_epochs = 10 if parsed_arguments.quick else 40
    overfit_reports = {1, 5, 10} if parsed_arguments.quick else {1, 10, 20, 40}

    full_training_loader = DataLoader(training_dataset, batch_size=100, shuffle=True)
    # 1,000 images is deliberately far too few for a 256-neuron network:
    # scarcity of data is what makes memorization both possible and tempting.
    small_training_loader = DataLoader(Subset(training_dataset, range(1000)), batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    print()
    optimizer_race(full_training_loader, test_loader, device, race_epochs)
    overfitting_demonstration(small_training_loader, test_loader, device, overfit_epochs, overfit_reports)
    more_data_demonstration(full_training_loader, test_loader, device, number_of_epochs=race_epochs + 2)


if __name__ == "__main__":
    main()
