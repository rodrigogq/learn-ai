"""Chapter 14 - export the trained ResNet for the pure-C inference program.

Two jobs:
  1. FOLD BATCH NORM INTO THE CONVOLUTIONS. At inference time a batch norm
     layer is just a fixed linear transform (its statistics are frozen), so it
     can be merged into the preceding convolution's weights and bias:
         folded_weight = conv_weight * gamma / sqrt(running_var + eps)
         folded_bias   = beta - gamma * running_mean / sqrt(running_var + eps)
     The C program then needs only convolutions - and this is a real
     deployment technique, used by every serious inference engine.
  2. Write the folded weights (float32, fixed order documented below) and the
     first 1,000 raw test images to datasets/, for the C program to read.

Run from the repository root (after training):
    .venv/bin/python chapters/14-image-classification/python/export_for_c.py
"""

import struct
import sys
from pathlib import Path

import numpy
import torch
from torchvision import datasets

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.data import get_datasets_directory  # noqa: E402
from train_cifar10_resnet import CHECKPOINT_PATH, SmallResNet  # noqa: E402

EXPORTED_IMAGE_COUNT = 1000


def fold_batch_norm_into_convolution(convolution, batch_norm):
    """Merge an inference-mode batch norm into the convolution before it.

    Arguments:
        convolution: an nn.Conv2d with bias=False.
        batch_norm: the nn.BatchNorm2d that follows it.

    Returns (folded_weight, folded_bias) as float32 numpy arrays. The math:
    batch norm computes gamma * (y - mean) / sqrt(var + eps) + beta, which is
    a per-channel scale and shift - both absorbable into the conv.
    """
    gamma = batch_norm.weight.detach().numpy()
    beta = batch_norm.bias.detach().numpy()
    running_mean = batch_norm.running_mean.numpy()
    running_std = numpy.sqrt(batch_norm.running_var.numpy() + batch_norm.eps)

    per_channel_scale = gamma / running_std
    folded_weight = convolution.weight.detach().numpy() * per_channel_scale[:, None, None, None]
    folded_bias = beta - running_mean * per_channel_scale
    return folded_weight.astype(numpy.float32), folded_bias.astype(numpy.float32)


def export_weights(model, output_path):
    """Write every folded layer in the fixed order the C program expects.

    The order (each conv as flattened weights then bias, all float32):
      stem; stage1 block1 conv1, conv2; block2 conv1, conv2;
      stage2 block1 conv1, conv2, shortcut-1x1; block2 conv1, conv2;
      stage3 block1 conv1, conv2, shortcut-1x1; block2 conv1, conv2;
      classifier linear weight (10x64) then bias (10).
    """
    folded_layers = [fold_batch_norm_into_convolution(model.stem_convolution, model.stem_batch_norm)]
    for stage in (model.stage_1, model.stage_2, model.stage_3):
        for block_index, block in enumerate(stage):
            folded_layers.append(fold_batch_norm_into_convolution(block.convolution_1, block.batch_norm_1))
            folded_layers.append(fold_batch_norm_into_convolution(block.convolution_2, block.batch_norm_2))
            if not isinstance(block.shortcut, torch.nn.Identity):
                folded_layers.append(fold_batch_norm_into_convolution(block.shortcut[0], block.shortcut[1]))

    with open(output_path, "wb") as weights_file:
        # A tiny header lets the C program sanity-check it got the right file.
        weights_file.write(struct.pack("ii", 0x524E3134, len(folded_layers)))  # magic "RN14"
        for folded_weight, folded_bias in folded_layers:
            weights_file.write(folded_weight.tobytes())
            weights_file.write(folded_bias.tobytes())
        weights_file.write(model.classifier_head.weight.detach().numpy().astype(numpy.float32).tobytes())
        weights_file.write(model.classifier_head.bias.detach().numpy().astype(numpy.float32).tobytes())
    print(f"  wrote {output_path.name} ({output_path.stat().st_size:,} bytes, {len(folded_layers)} folded convs + linear)")


def export_test_images(datasets_directory):
    """Write the first 1,000 CIFAR-10 test images as raw bytes + labels.

    Images go out channels-first (3x32x32 uint8, RGB); the C program applies
    the same per-channel normalization the Python pipeline used.
    """
    test_dataset = datasets.CIFAR10(str(datasets_directory), train=False, download=True)
    images_path = datasets_directory / "cifar10_test_images.bin"
    labels_path = datasets_directory / "cifar10_test_labels.bin"

    # dataset.data is (N, 32, 32, 3) uint8; transpose to channels-first.
    image_array = test_dataset.data[:EXPORTED_IMAGE_COUNT].transpose(0, 3, 1, 2)
    image_array.astype(numpy.uint8).tofile(images_path)
    numpy.array(test_dataset.targets[:EXPORTED_IMAGE_COUNT], dtype=numpy.uint8).tofile(labels_path)
    print(f"  wrote {images_path.name} and {labels_path.name} ({EXPORTED_IMAGE_COUNT} images)")


def main():
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint found - run train_cifar10_resnet.py first.")
        sys.exit(1)

    model = SmallResNet()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    datasets_directory = get_datasets_directory()
    print("Exporting for the C inference program:")
    export_weights(model, datasets_directory / "cifar10_resnet_weights.bin")
    export_test_images(datasets_directory)
    print("Done. Build and run the C program:")
    print("  make -C chapters/14-image-classification/c && ./chapters/14-image-classification/c/build/resnet_inference")


if __name__ == "__main__":
    main()
