"""Chapter 0 - first Python program of the course.

Verifies that the environment is ready: prints tool versions, reports which
compute devices PyTorch can use, and times a matrix multiplication on each
available device so the reader sees concretely what a GPU buys them.

Run from the repository root:
    .venv/bin/python chapters/00-setup/python/hello_ai.py
"""

import sys
import time


def report_tool_versions():
    """Print the Python, NumPy, and PyTorch versions in use."""
    import numpy
    import torch

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Hello from Python {python_version}!")
    print(f"NumPy version:    {numpy.__version__}")
    print(f"PyTorch version:  {torch.__version__}")


def report_available_devices():
    """Print which compute devices PyTorch can see, and return the fastest one.

    The course prefers CUDA (NVIDIA), then MPS (Apple Silicon), then CPU -
    the same order used by common/device.py in every later chapter.
    """
    import torch

    cuda_is_available = torch.cuda.is_available()
    mps_is_available = torch.backends.mps.is_available()

    print()
    print("Device check:")
    if cuda_is_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  CUDA (NVIDIA GPU):     AVAILABLE ({gpu_name})  <- PyTorch will use this")
    else:
        print("  CUDA (NVIDIA GPU):     not available")
    if mps_is_available:
        marker = "" if cuda_is_available else "  <- PyTorch will use this"
        print(f"  MPS (Apple Silicon):   AVAILABLE{marker}")
    else:
        print("  MPS (Apple Silicon):   not available")
    print("  CPU:                   always available")

    if cuda_is_available:
        return torch.device("cuda")
    if mps_is_available:
        return torch.device("mps")
    return torch.device("cpu")


def time_matrix_multiplication_on_device(device, matrix_size=2048):
    """Time one big matrix multiplication on the given device and return seconds.

    Arguments:
        device: the torch.device to run on ("cpu", "cuda", or "mps").
        matrix_size: width and height of the square matrices to multiply.
            2048x2048 is big enough to show a GPU's advantage but still takes
            well under a second on any modern machine.
    """
    import torch

    first_matrix = torch.rand(matrix_size, matrix_size, device=device)
    second_matrix = torch.rand(matrix_size, matrix_size, device=device)

    # GPUs run asynchronously: the multiply call returns before the math is
    # done. Synchronizing before and after the timer is what makes the
    # measurement honest.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()
    first_matrix @ second_matrix
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - start_time


def main():
    report_tool_versions()
    fastest_device = report_available_devices()

    import torch

    print()
    print("Small speed test: multiplying two 2048x2048 matrices...")
    devices_to_test = [torch.device("cpu")]
    if fastest_device.type != "cpu":
        devices_to_test.append(fastest_device)
    for device in devices_to_test:
        # The first operation on a GPU pays one-time startup costs, so we run
        # once unmeasured before timing.
        time_matrix_multiplication_on_device(device)
        elapsed_seconds = time_matrix_multiplication_on_device(device)
        print(f"  {device.type}: {elapsed_seconds:.3f} seconds")

    print()
    print("Your machine is ready for the course.")


if __name__ == "__main__":
    main()
