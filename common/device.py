"""Shared device selection for every PyTorch example in the course.

Every chapter imports select_best_available_device() instead of hard-coding
"cuda" or "cpu", so the same code runs unchanged on an NVIDIA GPU, an Apple
Silicon Mac, or a plain CPU.
"""

import torch


def select_best_available_device(print_choice: bool = True) -> torch.device:
    """Return the fastest device PyTorch can use on this machine.

    The preference order is CUDA (NVIDIA GPUs), then MPS (Apple Silicon GPUs),
    then CPU. All course code works on any of the three; only speed changes.

    Arguments:
        print_choice: when True, prints one line saying which device was picked
            and why, so readers always know where their code is running.
    """
    if torch.cuda.is_available():
        chosen_device = torch.device("cuda")
        explanation = f"NVIDIA GPU found: {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available():
        chosen_device = torch.device("mps")
        explanation = "Apple Silicon GPU found (Metal Performance Shaders)"
    else:
        chosen_device = torch.device("cpu")
        explanation = "no GPU found, using the CPU (slower, but everything works)"

    if print_choice:
        print(f"Using device '{chosen_device}': {explanation}")
    return chosen_device
