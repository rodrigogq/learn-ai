"""Chapter 17 - video understanding: a task no single frame can solve.

The task: 8-frame clips of an MNIST digit sliding up, down, left, or right;
classify the DIRECTION of motion. A single frame contains zero information
about it - which is the whole point. Three models compete:

  1. single-frame CNN (sees only the middle frame)  -> stuck at chance (~25%),
  2. early fusion (all 8 frames stacked as channels) -> solves it,
  3. 3D convolutions (kernels sliding over space AND time) -> solves it.

Run from the repository root:
    .venv/bin/python chapters/17-video-understanding/python/train_motion_classifier.py --quick
    .venv/bin/python chapters/17-video-understanding/python/train_motion_classifier.py
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_mnist_datasets  # noqa: E402
from common.device import select_best_available_device  # noqa: E402

CANVAS_SIZE = 48
FRAME_COUNT = 8
PIXELS_PER_FRAME = 2
DIRECTION_NAMES = ["up", "down", "left", "right"]
DIRECTION_STEPS = {0: (-PIXELS_PER_FRAME, 0), 1: (PIXELS_PER_FRAME, 0),
                   2: (0, -PIXELS_PER_FRAME), 3: (0, PIXELS_PER_FRAME)}


def build_clip_batch(digit_images, batch_size, random_generator):
    """Generate clips of one digit sliding in a straight line.

    Arguments:
        digit_images: tensor (N, 28, 28) of MNIST digits in 0..1.
        batch_size: clips to build.
        random_generator: torch.Generator for reproducibility.

    Returns (clips, direction_labels): clips (batch, FRAME_COUNT, 48, 48),
    labels (batch,) in 0..3. The starting position is chosen so the digit
    stays fully inside the canvas for the whole clip.
    """
    clips = torch.zeros(batch_size, FRAME_COUNT, CANVAS_SIZE, CANVAS_SIZE)
    direction_labels = torch.randint(0, 4, (batch_size,), generator=random_generator)

    for clip_index in range(batch_size):
        digit = digit_images[int(torch.randint(len(digit_images), (1,), generator=random_generator))]
        row_step, column_step = DIRECTION_STEPS[int(direction_labels[clip_index])]

        # Start inside a margin that guarantees the whole trajectory fits.
        min_row = max(0, -row_step * (FRAME_COUNT - 1))
        max_row = CANVAS_SIZE - 28 - max(0, row_step * (FRAME_COUNT - 1))
        min_column = max(0, -column_step * (FRAME_COUNT - 1))
        max_column = CANVAS_SIZE - 28 - max(0, column_step * (FRAME_COUNT - 1))
        row = int(torch.randint(min_row, max_row + 1, (1,), generator=random_generator))
        column = int(torch.randint(min_column, max_column + 1, (1,), generator=random_generator))

        for frame_index in range(FRAME_COUNT):
            top = row + row_step * frame_index
            left = column + column_step * frame_index
            clips[clip_index, frame_index, top:top + 28, left:left + 28] = digit
    return clips, direction_labels


class SingleFrameCNN(nn.Module):
    """Sees only the MIDDLE frame - the control group of the experiment."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 4),
        )

    def forward(self, clips):
        middle_frame = clips[:, FRAME_COUNT // 2: FRAME_COUNT // 2 + 1]
        return self.network(middle_frame)


class EarlyFusionCNN(nn.Module):
    """All 8 frames stacked as input channels: time becomes 'color'.

    The first convolution's kernels span all frames at each position, so a
    kernel can learn 'bright here in frame 0, bright 2px right in frame 7' -
    a motion detector. Cheap and effective when clips are short.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(FRAME_COUNT, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 4),
        )

    def forward(self, clips):
        return self.network(clips)


class Small3DCNN(nn.Module):
    """3D convolutions: kernels of shape (time, height, width) slide over all
    three axes, producing feature maps that are themselves little videos.
    The general tool for long clips, at a real compute cost."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(32, 4),
        )

    def forward(self, clips):
        # Conv3d wants (batch, channels, time, height, width); our clips are
        # single-channel videos, so insert the channel axis.
        return self.network(clips[:, None])


def train_and_evaluate(model, model_name, digit_images, device, total_steps):
    """Train one contender and report its final accuracy on fresh clips."""
    generator = torch.Generator().manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    model.to(device)

    for _ in range(total_steps):
        clips, labels = build_clip_batch(digit_images, 32, generator)
        loss = loss_function(model(clips.to(device)), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    evaluation_generator = torch.Generator().manual_seed(999)
    clips, labels = build_clip_batch(digit_images, 400, evaluation_generator)
    model.eval()
    with torch.no_grad():
        predictions = model(clips.to(device)).argmax(dim=1).cpu()
    accuracy = (predictions == labels).float().mean().item()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"  {model_name:<28} {parameter_count:>9,} params   accuracy {accuracy:.1%}")
    return accuracy


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="200 steps per model instead of 600")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    total_steps = 200 if parsed_arguments.quick else 600

    print("Loading MNIST digits to animate...")
    training_dataset, _ = load_mnist_datasets()
    digit_images = training_dataset.data.float() / 255.0

    print(f"\nTask: which way is the digit moving? ({FRAME_COUNT} frames, 4 directions, {total_steps} steps each)\n")
    train_and_evaluate(SingleFrameCNN(), "single frame (control)", digit_images, device, total_steps)
    train_and_evaluate(EarlyFusionCNN(), "early fusion (frames=channels)", digit_images, device, total_steps)
    train_and_evaluate(Small3DCNN(), "3D convolutions", digit_images, device, total_steps)
    print()
    print("The single-frame model is stuck at chance (~25%): motion does not exist in a still.")
    print("Both temporal models solve the task - seeing time is everything here.")


if __name__ == "__main__":
    main()
