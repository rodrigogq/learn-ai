"""Chapter 10 - PyTorch's three ideas: tensors, autograd, and devices.

Three demonstrations:
  1. tensors are Chapter 2's matrices with shapes, slicing, and broadcasting,
  2. autograd is Chapter 8's engine, industrial-strength: the same
     L = (a*b + c)^2 example gives the same gradients,
  3. the same code runs on CPU or GPU by changing one word.

Run from the repository root:
    .venv/bin/python chapters/10-intro-to-pytorch/python/pytorch_basics.py
"""

import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.device import select_best_available_device  # noqa: E402


def demonstrate_tensors():
    """Tensors: n-dimensional arrays with a shape - Chapter 2, upgraded."""
    print("1. Tensors")

    weight_matrix = torch.tensor([[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]])
    input_vector = torch.tensor([4.0, 2.0])
    print(f"   a (3, 2) matrix times a (2,) vector: {(weight_matrix @ input_vector).tolist()}"
          "   <- Chapter 2's worked example")

    image_batch = torch.zeros(100, 28, 28)
    print(f"   a batch of 100 MNIST images is one tensor of shape {list(image_batch.shape)}")
    flattened_batch = image_batch.reshape(100, 784)
    print(f"   .reshape(100, 784) gives shape {list(flattened_batch.shape)} - no data is copied, only the shape changes")

    # Broadcasting: operating on tensors of different shapes stretches the
    # smaller one for free. This replaces most explicit loops.
    per_pixel_offset = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"   broadcasting: shape (2,4) + shape (4,) -> {list((torch.ones(2, 4) + per_pixel_offset).shape)}"
          " (the vector was applied to every row)")


def demonstrate_autograd():
    """Chapter 8's engine, built in: same example, same gradients."""
    print()
    print("2. Autograd - Chapter 8's L = (a*b + c)^2, now in one line each")

    input_a = torch.tensor(2.0, requires_grad=True)
    input_b = torch.tensor(3.0, requires_grad=True)
    input_c = torch.tensor(-1.0, requires_grad=True)

    loss = (input_a * input_b + input_c) ** 2
    loss.backward()

    print(f"   L = {loss.item():.0f}")
    print(f"   dL/da = {input_a.grad.item():.0f}   dL/db = {input_b.grad.item():.0f}   "
          f"dL/dc = {input_c.grad.item():.0f}   (Chapter 8's numbers, from PyTorch this time)")
    print("   requires_grad=True marks a leaf as trainable; .backward() runs backpropagation;")
    print("   .grad holds the result. That is our whole engine, renamed.")


def demonstrate_devices():
    """One word moves the same computation to whatever hardware exists."""
    print()
    print("3. Devices")
    chosen_device = select_best_available_device()

    matrix_size = 2048
    first_matrix = torch.rand(matrix_size, matrix_size)
    second_matrix = torch.rand(matrix_size, matrix_size)

    start_time = time.perf_counter()
    first_matrix @ second_matrix
    cpu_seconds = time.perf_counter() - start_time

    if chosen_device.type == "cpu":
        print(f"   {matrix_size}x{matrix_size} matmul on cpu: {cpu_seconds*1000:.1f} ms (no GPU found - everything still works)")
        return

    first_matrix_on_device = first_matrix.to(chosen_device)
    second_matrix_on_device = second_matrix.to(chosen_device)
    # First device operation pays one-time startup costs; run once unmeasured.
    first_matrix_on_device @ second_matrix_on_device
    torch.mps.synchronize() if chosen_device.type == "mps" else torch.cuda.synchronize()

    start_time = time.perf_counter()
    first_matrix_on_device @ second_matrix_on_device
    torch.mps.synchronize() if chosen_device.type == "mps" else torch.cuda.synchronize()
    device_seconds = time.perf_counter() - start_time

    print(f"   {matrix_size}x{matrix_size} matmul: cpu {cpu_seconds*1000:.1f} ms  vs  "
          f"{chosen_device.type} {device_seconds*1000:.1f} ms")
    print("   .to(device) is the only change - the training script uses it on the whole model.")


def main():
    demonstrate_tensors()
    demonstrate_autograd()
    demonstrate_devices()


if __name__ == "__main__":
    main()
