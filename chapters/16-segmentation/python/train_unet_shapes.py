"""Chapter 16 - semantic segmentation with a U-Net, on synthetic shape scenes.

The task: label EVERY PIXEL of a 64x64 image as background, circle, or
rectangle. Scenes are generated fresh each step (shapes at random positions
and sizes, with noise), so the ground-truth masks are exact and the dataset
is infinite. The model is a small U-Net: an encoder that shrinks the image,
a decoder that grows it back, and skip connections that carry fine detail
across. The metric is per-class IoU - Chapter 15's box ruler, applied to
pixel sets.

Run from the repository root:
    .venv/bin/python chapters/16-segmentation/python/train_unet_shapes.py --quick
    .venv/bin/python chapters/16-segmentation/python/train_unet_shapes.py
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.device import select_best_available_device  # noqa: E402

IMAGE_SIZE = 64
CLASS_NAMES = ["background", "circle", "rectangle"]
CLASS_COUNT = len(CLASS_NAMES)


def build_shape_batch(batch_size, random_generator):
    """Generate scenes of noisy circles and rectangles with exact pixel masks.

    Arguments:
        batch_size: scenes to build.
        random_generator: torch.Generator for reproducibility.

    Returns (images, masks): images (batch, 1, 64, 64) with values in 0..1,
    masks (batch, 64, 64) of integer class ids per pixel.
    """
    images = torch.rand(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE, generator=random_generator) * 0.3
    masks = torch.zeros(batch_size, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)

    row_coordinates = torch.arange(IMAGE_SIZE).view(-1, 1).expand(IMAGE_SIZE, IMAGE_SIZE)
    column_coordinates = torch.arange(IMAGE_SIZE).view(1, -1).expand(IMAGE_SIZE, IMAGE_SIZE)

    for scene_index in range(batch_size):
        for _ in range(int(torch.randint(1, 4, (1,), generator=random_generator))):
            shape_class = int(torch.randint(1, CLASS_COUNT, (1,), generator=random_generator))
            center_row = int(torch.randint(12, IMAGE_SIZE - 12, (1,), generator=random_generator))
            center_column = int(torch.randint(12, IMAGE_SIZE - 12, (1,), generator=random_generator))
            half_size = int(torch.randint(5, 11, (1,), generator=random_generator))

            if shape_class == 1:
                inside = (row_coordinates - center_row) ** 2 + (column_coordinates - center_column) ** 2 <= half_size ** 2
            else:
                inside = ((row_coordinates - center_row).abs() <= half_size) \
                       & ((column_coordinates - center_column).abs() <= half_size)

            brightness = 0.6 + 0.4 * float(torch.rand(1, generator=random_generator))
            images[scene_index, 0][inside] = brightness
            masks[scene_index][inside] = shape_class

    # Pixel noise on top of everything makes the task honest: single-pixel
    # brightness alone cannot decide the class; context must.
    images += torch.randn(images.shape, generator=random_generator) * 0.05
    return images.clamp(0, 1), masks


class MiniUNet(nn.Module):
    """A small U-Net: two downsampling stages, a bottleneck, two upsampling
    stages, with skip connections carrying detail from encoder to decoder."""

    @staticmethod
    def double_convolution(input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def __init__(self):
        super().__init__()
        self.encoder_stage_1 = self.double_convolution(1, 16)     # 64x64
        self.encoder_stage_2 = self.double_convolution(16, 32)    # 32x32
        self.bottleneck = self.double_convolution(32, 64)         # 16x16
        self.downsample = nn.MaxPool2d(2)

        self.upsample_2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_stage_2 = self.double_convolution(64, 32)    # 32 skip + 32 upsampled
        self.upsample_1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder_stage_1 = self.double_convolution(32, 16)    # 16 skip + 16 upsampled
        self.per_pixel_classifier = nn.Conv2d(16, CLASS_COUNT, kernel_size=1)

    def forward(self, image_batch):
        encoder_1_features = self.encoder_stage_1(image_batch)                     # fine detail, 64x64
        encoder_2_features = self.encoder_stage_2(self.downsample(encoder_1_features))  # 32x32
        bottleneck_features = self.bottleneck(self.downsample(encoder_2_features))      # 16x16, most context

        # The U-Net move: after upsampling, CONCATENATE the encoder's features
        # from the same resolution. The decoder gets context (from below) and
        # crisp edges (from the skip) at the same time.
        decoder_2_features = self.decoder_stage_2(
            torch.cat([self.upsample_2(bottleneck_features), encoder_2_features], dim=1))
        decoder_1_features = self.decoder_stage_1(
            torch.cat([self.upsample_1(decoder_2_features), encoder_1_features], dim=1))
        return self.per_pixel_classifier(decoder_1_features)


def compute_per_class_iou(predicted_masks, true_masks):
    """IoU per class over a batch: overlap of pixel sets / union of pixel sets.

    Arguments:
        predicted_masks, true_masks: integer masks of shape (batch, H, W).

    Returns a list of CLASS_COUNT IoU values (nan-free: classes absent from
    both prediction and truth count as perfect 1.0).
    """
    iou_per_class = []
    for class_id in range(CLASS_COUNT):
        predicted_set = predicted_masks == class_id
        true_set = true_masks == class_id
        intersection = (predicted_set & true_set).sum().item()
        union = (predicted_set | true_set).sum().item()
        iou_per_class.append(intersection / union if union > 0 else 1.0)
    return iou_per_class


def render_mask_as_text(mask):
    """A terminal visualization: one character per 2x2 pixel block."""
    symbols = {0: ".", 1: "o", 2: "#"}
    lines = []
    for row in range(0, IMAGE_SIZE, 2):
        line = "".join(symbols[int(mask[row, column])] for column in range(0, IMAGE_SIZE, 2))
        lines.append("   " + line)
    return "\n".join(lines)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="400 steps instead of 2000")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    scene_generator = torch.Generator().manual_seed(42)

    model = MiniUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Per-pixel cross-entropy: the same classifier loss as always, just applied
    # 4,096 times per image. Segmentation IS classification, per pixel.
    loss_function = nn.CrossEntropyLoss()
    total_steps = 400 if parsed_arguments.quick else 2000

    print(f"Training the U-Net for {total_steps} steps (fresh synthetic scenes every step)...")
    print("   step    loss     IoU background / circle / rectangle")
    evaluation_generator = torch.Generator().manual_seed(999)
    evaluation_images, evaluation_masks = build_shape_batch(64, evaluation_generator)
    for step_number in range(1, total_steps + 1):
        images, masks = build_shape_batch(16, scene_generator)
        loss = loss_function(model(images.to(device)), masks.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_number in (1, 100, total_steps // 2, total_steps):
            model.eval()
            with torch.no_grad():
                predicted = model(evaluation_images.to(device)).argmax(dim=1).cpu()
            model.train()
            iou_values = compute_per_class_iou(predicted, evaluation_masks)
            print(f"  {step_number:>5}   {loss.item():.4f}   "
                  + " / ".join(f"{value:.3f}" for value in iou_values))

    print()
    print("One evaluation scene, truth vs prediction ('.'=background 'o'=circle '#'=rectangle):")
    model.eval()
    with torch.no_grad():
        predicted = model(evaluation_images[:1].to(device)).argmax(dim=1).cpu()[0]
    print("  truth:")
    print(render_mask_as_text(evaluation_masks[0]))
    print("  prediction:")
    print(render_mask_as_text(predicted))


if __name__ == "__main__":
    main()
