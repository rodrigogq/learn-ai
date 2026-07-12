"""Chapter 13 - convolution built from scratch, checked against PyTorch, timed.

Four demonstrations:
  1. the figure's worked example: a vertical-edge kernel on a striped image,
  2. padding and stride: how they change the output size (the formula, live),
  3. our loops agree with torch.nn.functional.conv2d to float precision,
  4. speed: pure-Python loops vs PyTorch on a real workload.

Run from the repository root:
    .venv/bin/python chapters/13-convolutions/python/convolution_from_scratch.py
"""

import sys
import time
from pathlib import Path

import numpy
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))


def convolve_2d(input_image, kernel, padding=0, stride=1):
    """Slide a kernel over an image: the whole algorithm, no libraries.

    Arguments:
        input_image: 2D numpy array (height, width).
        kernel: 2D numpy array (kernel_height, kernel_width) of weights.
        padding: rows/columns of zeros added around the image border, so the
            kernel can center on edge pixels too.
        stride: how many pixels the window jumps per step; stride 2 halves
            the output size.

    Returns the output feature map, of shape given by the chapter's formula:
        output_size = (input_size + 2*padding - kernel_size) // stride + 1
    """
    if padding > 0:
        input_image = numpy.pad(input_image, padding)

    input_height, input_width = input_image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output_map = numpy.zeros((output_height, output_width))
    for output_row in range(output_height):
        for output_column in range(output_width):
            image_patch = input_image[
                output_row * stride: output_row * stride + kernel_height,
                output_column * stride: output_column * stride + kernel_width,
            ]
            # The heart of it: patch times kernel, element by element, summed.
            # One weighted sum (Chapter 0!) per output pixel.
            output_map[output_row, output_column] = (image_patch * kernel).sum()
    return output_map


VERTICAL_EDGE_KERNEL = numpy.array([
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
])


def demonstrate_worked_example():
    """Demo 1: reproduce the chapter figure exactly."""
    striped_image = numpy.zeros((5, 5))
    striped_image[:, 2:4] = 1.0

    print("1. The figure's example: vertical-edge kernel on a 5x5 striped image")
    print("   input rows are [0, 0, 1, 1, 0]; output (no padding):")
    output_map = convolve_2d(striped_image, VERTICAL_EDGE_KERNEL)
    for row in output_map:
        print("   " + "  ".join(f"{value:+.0f}" for value in row))
    print("   +3 where brightness rises, -3 where it falls, exactly as the figure computes.")


def demonstrate_padding_and_stride():
    """Demo 2: the output-size formula, verified live."""
    print()
    print("2. Padding and stride (input 28x28, kernel 3x3)")
    test_image = numpy.random.default_rng(42).random((28, 28))
    for padding, stride in ((0, 1), (1, 1), (1, 2), (0, 2)):
        output_map = convolve_2d(test_image, VERTICAL_EDGE_KERNEL, padding, stride)
        formula_size = (28 + 2 * padding - 3) // stride + 1
        print(f"   padding={padding} stride={stride}: output {output_map.shape[0]}x{output_map.shape[1]}"
              f"   (formula: (28 + {2*padding} - 3)//{stride} + 1 = {formula_size})")
    print("   padding=1 keeps 28x28 ('same' padding); stride=2 halves it - CNNs shrink maps this way.")


def demonstrate_agreement_with_pytorch():
    """Demo 3: our loops and PyTorch produce the same numbers."""
    print()
    print("3. Cross-check against torch.nn.functional.conv2d")
    random_generator = numpy.random.default_rng(123)
    test_image = random_generator.random((28, 28))
    test_kernel = random_generator.random((3, 3))

    our_output = convolve_2d(test_image, test_kernel, padding=1)
    # conv2d wants shapes (batch, channels, height, width) - hence the [None, None].
    torch_output = torch.nn.functional.conv2d(
        torch.tensor(test_image)[None, None],
        torch.tensor(test_kernel)[None, None],
        padding=1,
    )[0, 0].numpy()

    largest_difference = numpy.abs(our_output - torch_output).max()
    print(f"   largest difference over all 784 output values: {largest_difference:.2e}  "
          f"{'(agreement)' if largest_difference < 1e-9 else '(MISMATCH!)'}")


def demonstrate_speed():
    """Demo 4: the loops are correct but slow; the C version and PyTorch race them."""
    print()
    print("4. Speed on a realistic workload: 224x224 image, 3x3 kernel, padding 1")
    large_image = numpy.random.default_rng(7).random((224, 224))

    start_time = time.perf_counter()
    convolve_2d(large_image, VERTICAL_EDGE_KERNEL, padding=1)
    python_seconds = time.perf_counter() - start_time

    image_tensor = torch.tensor(large_image)[None, None]
    kernel_tensor = torch.tensor(VERTICAL_EDGE_KERNEL)[None, None]
    torch.nn.functional.conv2d(image_tensor, kernel_tensor, padding=1)  # warm-up
    start_time = time.perf_counter()
    torch.nn.functional.conv2d(image_tensor, kernel_tensor, padding=1)
    torch_seconds = time.perf_counter() - start_time

    print(f"   Python loops: {python_seconds*1000:8.1f} ms")
    print(f"   PyTorch:      {torch_seconds*1000:8.2f} ms   ({python_seconds/torch_seconds:,.0f}x faster)")
    print("   Compare with the C example's time on this same workload: on one small")
    print("   single-channel image, plain compiled loops match PyTorch. PyTorch's real")
    print("   advantage appears on batched many-channel workloads - and on GPUs.")


def main():
    demonstrate_worked_example()
    demonstrate_padding_and_stride()
    demonstrate_agreement_with_pytorch()
    demonstrate_speed()


if __name__ == "__main__":
    main()
