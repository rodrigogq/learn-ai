"""Chapter 10 - Chapter 9's network, rebuilt in PyTorch.

Same architecture (784 -> 128 ReLU -> 10), same training procedure (mini-batch
SGD, batch 100, learning rate 0.1, 5 epochs), same ~96% accuracy - but the
model definition plus training step is now about 30 lines, runs on whatever
GPU exists, and required zero hand-derived gradients.

Run from the repository root:
    .venv/bin/python chapters/10-intro-to-pytorch/python/train_mnist_pytorch.py
    .venv/bin/python chapters/10-intro-to-pytorch/python/train_mnist_pytorch.py --quick
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_mnist_datasets  # noqa: E402
from common.device import select_best_available_device  # noqa: E402


class DigitClassifier(nn.Module):
    """Chapter 9's 784 -> 128 (ReLU) -> 10 network as a PyTorch module.

    nn.Linear IS the weighted-sum layer we built by hand: it owns a weight
    matrix and a bias vector, and its forward pass is x @ W + b. nn.Module
    tracks every parameter inside, so optimizers and .to(device) find them.
    """

    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(784, 128)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, image_batch):
        """From a batch of flat images to a batch of 10 raw scores.

        Arguments:
            image_batch: tensor of shape (batch_size, 784).

        No softmax here: PyTorch's cross-entropy loss applies it internally
        (fused with the log for numerical stability - the same trick our
        Chapter 9 code did by hand).
        """
        hidden_activation = torch.relu(self.hidden_layer(image_batch))
        return self.output_layer(hidden_activation)


def measure_accuracy(model, data_loader, device):
    """Fraction of correctly classified images over a whole dataset.

    Arguments:
        model: the trained DigitClassifier.
        data_loader: DataLoader yielding (images, labels) batches.
        device: where the model lives; batches are moved there one by one.
    """
    model.eval()
    correct_count = 0
    total_count = 0
    # no_grad tells autograd not to record: evaluation needs no gradients,
    # so skipping the bookkeeping makes it faster and lighter.
    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            class_scores = model(image_batch)
            correct_count += (class_scores.argmax(dim=1) == label_batch).sum().item()
            total_count += len(label_batch)
    model.train()
    return correct_count / total_count


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true",
                                 help="train on a 2,000-image subset for 1 epoch")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)

    print("Loading MNIST (downloads ~12 MB on first run)...")
    training_dataset, test_dataset = load_mnist_datasets(flatten_images_to_vectors=True)
    number_of_epochs = 5
    if parsed_arguments.quick:
        training_dataset = Subset(training_dataset, range(2000))
        test_dataset = Subset(test_dataset, range(1000))
        number_of_epochs = 1
        print("(quick mode: 2,000 training images, 1 epoch)")

    training_loader = DataLoader(training_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = DigitClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.CrossEntropyLoss()

    print()
    print("Training: batch size 100, learning rate 0.1")
    print("  epoch   average loss   test accuracy   seconds")
    for epoch_number in range(1, number_of_epochs + 1):
        epoch_start_time = time.perf_counter()
        loss_sum = 0.0
        batch_count = 0
        for image_batch, label_batch in training_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            # The eternal loop, in PyTorch spelling:
            class_scores = model(image_batch)                  # forward
            loss = loss_function(class_scores, label_batch)    # loss
            optimizer.zero_grad()                              # clear old gradients (they accumulate otherwise - Chapter 8's +=)
            loss.backward()                                    # backward: every gradient, automatically
            optimizer.step()                                   # update all parameters

            loss_sum += loss.item()
            batch_count += 1
        epoch_seconds = time.perf_counter() - epoch_start_time
        test_accuracy = measure_accuracy(model, test_loader, device)
        print(f"  {epoch_number:>5}   {loss_sum / batch_count:>12.4f}   {test_accuracy:>12.2%}   {epoch_seconds:>7.1f}")

    print()
    print(f"Final test accuracy: {measure_accuracy(model, test_loader, device):.2%} - "
          "Chapter 9's result, a fraction of the code.")


if __name__ == "__main__":
    main()
