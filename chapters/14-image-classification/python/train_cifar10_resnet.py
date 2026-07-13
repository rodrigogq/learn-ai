"""Chapter 14 - a residual network (ResNet) trained on CIFAR-10.

Architecture: a small CIFAR-style ResNet - conv stem, three stages of residual
blocks (16 -> 32 -> 64 channels, spatial size 32 -> 16 -> 8), global average
pooling, one linear layer. Trained with augmentation (random crops + flips,
Chapter 12) and SGD with momentum (Chapter 11).

The trained checkpoint is saved to checkpoints/cifar10_resnet.pt so the C
inference program (see export_for_c.py) can use it.

Run from the repository root:
    .venv/bin/python chapters/14-image-classification/python/train_cifar10_resnet.py --quick
    .venv/bin/python chapters/14-image-classification/python/train_cifar10_resnet.py            # ~10 epochs
    .venv/bin/python chapters/14-image-classification/python/train_cifar10_resnet.py --epochs 50  # the full ride
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import get_datasets_directory  # noqa: E402
from common.device import select_best_available_device  # noqa: E402

CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "cifar10_resnet.pt"

CIFAR10_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

# Channel statistics of the CIFAR-10 training set - the standard normalization
# (Chapter 5's standardize-your-inputs rule, per color channel).
CIFAR10_CHANNEL_MEANS = (0.4914, 0.4822, 0.4465)
CIFAR10_CHANNEL_STDS = (0.2470, 0.2435, 0.2616)


class ResidualBlock(nn.Module):
    """Two 3x3 conv layers plus the shortcut: output = relu(F(x) + x).

    The shortcut ("skip connection") is the ResNet idea: the block only has
    to learn the CHANGE it wants to make to x, not to reproduce x itself.
    Gradients also flow backward through the shortcut untouched, which is
    what lets very deep stacks train (the chapter explains why).
    """

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.convolution_1 = nn.Conv2d(input_channels, output_channels, kernel_size=3,
                                       stride=stride, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(output_channels)
        self.convolution_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,
                                       stride=1, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(output_channels)

        # When the block changes resolution (stride 2) or channel count, x and
        # F(x) have different shapes and cannot be added directly; a 1x1
        # convolution on the shortcut reshapes x to match.
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, feature_map):
        block_output = torch.relu(self.batch_norm_1(self.convolution_1(feature_map)))
        block_output = self.batch_norm_2(self.convolution_2(block_output))
        return torch.relu(block_output + self.shortcut(feature_map))


class SmallResNet(nn.Module):
    """A compact CIFAR ResNet: stem + six residual blocks + pooled linear head."""

    def __init__(self, class_count=10):
        super().__init__()
        self.stem_convolution = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.stem_batch_norm = nn.BatchNorm2d(16)
        self.stage_1 = nn.Sequential(ResidualBlock(16, 16), ResidualBlock(16, 16))
        self.stage_2 = nn.Sequential(ResidualBlock(16, 32, stride=2), ResidualBlock(32, 32))
        self.stage_3 = nn.Sequential(ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64))
        self.classifier_head = nn.Linear(64, class_count)

    def forward(self, image_batch):
        feature_map = torch.relu(self.stem_batch_norm(self.stem_convolution(image_batch)))
        feature_map = self.stage_3(self.stage_2(self.stage_1(feature_map)))
        # Global average pooling: each of the 64 channels collapses to its
        # mean - 64 numbers summarizing the whole image, position-free.
        pooled_features = feature_map.mean(dim=(2, 3))
        return self.classifier_head(pooled_features)


def build_data_loaders(quick_mode):
    """CIFAR-10 loaders: augmented training set, plain test set.

    Arguments:
        quick_mode: when True, use small subsets so a smoke test finishes in
            about a minute.
    """
    datasets_root = str(get_datasets_directory())
    augmented_transform = transforms.Compose([
        # Pad-then-random-crop shifts the image up to 4 pixels; flips are safe
        # for photos (a mirrored cat is a cat - unlike Chapter 12's digits).
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_CHANNEL_MEANS, CIFAR10_CHANNEL_STDS),
    ])
    plain_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_CHANNEL_MEANS, CIFAR10_CHANNEL_STDS),
    ])

    training_dataset = datasets.CIFAR10(datasets_root, train=True, download=True,
                                        transform=augmented_transform)
    test_dataset = datasets.CIFAR10(datasets_root, train=False, download=True,
                                    transform=plain_transform)
    if quick_mode:
        training_dataset = Subset(training_dataset, range(5000))
        test_dataset = Subset(test_dataset, range(1000))

    training_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500)
    return training_loader, test_loader


def measure_accuracy(model, data_loader, device):
    """Fraction of correct predictions (eval mode: batch norm uses running stats)."""
    model.eval()
    correct_count = total_count = 0
    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            correct_count += (model(image_batch).argmax(dim=1) == label_batch).sum().item()
            total_count += len(label_batch)
    model.train()
    return correct_count / total_count


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="small subset, 2 epochs, ~1 minute")
    argument_parser.add_argument("--epochs", type=int, default=10,
                                 help="training epochs (10 default; 50 reaches ~91%%)")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    number_of_epochs = 2 if parsed_arguments.quick else parsed_arguments.epochs

    print("Loading CIFAR-10 (downloads ~170 MB on first run)...")
    training_loader, test_loader = build_data_loaders(parsed_arguments.quick)

    model = SmallResNet().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"SmallResNet: {parameter_count:,} parameters")

    # SGD with momentum + cosine schedule is the standard recipe for CIFAR
    # ResNets (Chapter 11 explains momentum; the schedule slowly lowers the
    # learning rate so late training takes fine steps).
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    learning_rate_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=number_of_epochs)
    loss_function = nn.CrossEntropyLoss()

    print()
    print("  epoch   average loss   test accuracy   seconds")
    for epoch_number in range(1, number_of_epochs + 1):
        epoch_start_time = time.perf_counter()
        loss_sum = 0.0
        batch_count = 0
        for image_batch, label_batch in training_loader:
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            loss = loss_function(model(image_batch), label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_count += 1
        learning_rate_schedule.step()
        epoch_seconds = time.perf_counter() - epoch_start_time
        test_accuracy = measure_accuracy(model, test_loader, device)
        print(f"  {epoch_number:>5}   {loss_sum / batch_count:>12.4f}   {test_accuracy:>12.2%}   {epoch_seconds:>7.1f}")

    final_accuracy = measure_accuracy(model, test_loader, device)
    print(f"\nFinal test accuracy: {final_accuracy:.2%}")

    CHECKPOINT_DIRECTORY.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)}")
    print("Next: export it for the C inference program with export_for_c.py")


if __name__ == "__main__":
    main()
