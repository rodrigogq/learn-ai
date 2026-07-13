"""Chapter 24, step 2 - train the mini-LLM, with checkpoint/resume built in.

This is Chapter 23's GPT, scaled up and made to survive interruptions. The
model config is chosen so the DEFAULT run finishes in minutes for a smoke
test, while --size medium/large give real multi-hour/multi-day runs sized for
a 16 GB GPU or a 64 GB Mac. A checkpoint is saved every --checkpoint-minutes;
--resume picks up exactly where the last one stopped. Ctrl+C, reboot, or a
week-long pause change nothing.

Run from the repository root (after prepare_data.py):
    .venv/bin/python chapters/24-train-your-mini-llm/python/train_mini_llm.py --quick
    .venv/bin/python chapters/24-train-your-mini-llm/python/train_mini_llm.py --size small
    .venv/bin/python chapters/24-train-your-mini-llm/python/train_mini_llm.py --size medium --resume
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.data import get_datasets_directory  # noqa: E402
from common.device import select_best_available_device  # noqa: E402
from model import MiniLanguageModel, MODEL_SIZES  # noqa: E402

MINI_LLM_DIRECTORY = get_datasets_directory() / "mini_llm"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"
VOCABULARY_SIZE = 256 + 4096      # bytes + BPE merges, matching prepare_data.py


def load_token_memmap():
    """Memory-map tokens.bin so multi-gigabyte corpora need no RAM to sample."""
    tokens_path = MINI_LLM_DIRECTORY / "tokens.bin"
    if not tokens_path.exists():
        print("No tokens.bin - run prepare_data.py first.")
        sys.exit(1)
    return numpy.memmap(tokens_path, dtype=numpy.uint16, mode="r")


def get_batch(token_data, batch_size, context_length, device, generator):
    """Sample a batch of (context, next-token) windows from the token stream."""
    starts = torch.randint(0, len(token_data) - context_length - 1, (batch_size,), generator=generator)
    inputs = torch.stack([torch.from_numpy(token_data[s:s + context_length].astype(numpy.int64)) for s in starts])
    targets = torch.stack([torch.from_numpy(token_data[s + 1:s + context_length + 1].astype(numpy.int64)) for s in starts])
    return inputs.to(device), targets.to(device)


def save_checkpoint(path, model, optimizer, step_number, config_name):
    """Write everything needed to resume: weights, optimizer state, position.

    A checkpoint is not just the weights - the optimizer's momentum buffers
    (Chapter 11) and the step counter matter too, or resuming would jolt the
    training. Saving all three is what makes --resume seamless.
    """
    temporary_path = path.with_suffix(".tmp")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step_number": step_number,
        "config_name": config_name,
    }, temporary_path)
    # Rename is atomic on every OS, so a crash mid-save never corrupts the
    # existing good checkpoint - a small habit that saves multi-day runs.
    temporary_path.replace(path)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--size", choices=list(MODEL_SIZES), default="tiny",
                                 help="model size (tiny default; see model.py for the table)")
    argument_parser.add_argument("--quick", action="store_true", help="tiny model, 200 steps - smoke test")
    argument_parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    argument_parser.add_argument("--max-steps", type=int, default=None, help="override the size's default step count")
    argument_parser.add_argument("--checkpoint-minutes", type=float, default=5.0,
                                 help="how often to save (default every 5 minutes)")
    parsed_arguments = argument_parser.parse_args()

    config_name = "tiny" if parsed_arguments.quick else parsed_arguments.size
    config = MODEL_SIZES[config_name]
    # --max-steps always wins if given; otherwise --quick means 200 and each
    # size has its own default.
    if parsed_arguments.max_steps is not None:
        max_steps = parsed_arguments.max_steps
    elif parsed_arguments.quick:
        max_steps = 200
    else:
        max_steps = config["default_steps"]

    device = select_best_available_device()
    torch.manual_seed(42)
    CHECKPOINT_DIRECTORY.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIRECTORY / f"mini_llm_{config_name}.pt"

    token_data = load_token_memmap()
    print(f"Corpus: {len(token_data):,} tokens")
    model = MiniLanguageModel(VOCABULARY_SIZE, config).to(device)
    print(f"Model '{config_name}': {sum(p.numel() for p in model.parameters()):,} parameters "
          f"({config['block_count']} blocks, {config['embedding_size']} wide, context {config['context_length']})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=0.1, betas=(0.9, 0.95))

    start_step = 0
    if parsed_arguments.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint["step_number"]
        print(f"Resumed from step {start_step:,} (checkpoint {checkpoint_path.name})")
    elif parsed_arguments.resume:
        print("--resume given but no checkpoint found; starting fresh.")

    loss_function = nn.CrossEntropyLoss()
    batch_generator = torch.Generator().manual_seed(42 + start_step)
    last_checkpoint_time = time.perf_counter()
    training_start = time.perf_counter()

    print(f"\nTraining to step {max_steps:,} (checkpoint every {parsed_arguments.checkpoint_minutes} min)")
    print("Interrupt any time with Ctrl+C - the latest checkpoint is safe.\n")
    print("   step     loss    perplexity   tokens/sec")
    try:
        for step_number in range(start_step + 1, max_steps + 1):
            # Cosine schedule with a short warmup - the standard LLM recipe.
            learning_rate = config["learning_rate"] * min(1.0, step_number / 100) \
                * 0.5 * (1 + math.cos(math.pi * step_number / max_steps))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            batch_start = time.perf_counter()
            inputs, targets = get_batch(token_data, config["batch_size"], config["context_length"],
                                        device, batch_generator)
            logits = model(inputs)
            loss = loss_function(logits.reshape(-1, VOCABULARY_SIZE), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step_number % 20 == 0 or step_number == start_step + 1:
                tokens_per_second = config["batch_size"] * config["context_length"] / (time.perf_counter() - batch_start)
                print(f"  {step_number:>5}   {loss.item():>7.4f}   {math.exp(min(loss.item(), 20)):>9.1f}   "
                      f"{tokens_per_second:>10,.0f}")

            if (time.perf_counter() - last_checkpoint_time) / 60 >= parsed_arguments.checkpoint_minutes:
                save_checkpoint(checkpoint_path, model, optimizer, step_number, config_name)
                print(f"       [checkpoint saved at step {step_number:,}]")
                last_checkpoint_time = time.perf_counter()
        final_step = max_steps
    except KeyboardInterrupt:
        print("\nInterrupted - saving a final checkpoint before exiting...")
        final_step = step_number

    if start_step >= max_steps:
        print(f"Already at step {start_step} >= target {max_steps}; nothing to do. "
              "Raise --max-steps to train further.")
        return

    save_checkpoint(checkpoint_path, model, optimizer, final_step, config_name)
    elapsed_minutes = (time.perf_counter() - training_start) / 60
    print(f"\nStopped at step {final_step}. Checkpoint: {checkpoint_path.relative_to(REPOSITORY_ROOT)}")
    print(f"Trained for {elapsed_minutes:.1f} minutes. Resume with --resume, or sample with sample.py.")


if __name__ == "__main__":
    main()
