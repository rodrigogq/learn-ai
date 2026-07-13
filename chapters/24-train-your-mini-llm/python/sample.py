"""Chapter 24, step 3 - generate text from a trained mini-LLM checkpoint.

Loads a checkpoint, the tokenizer's merges, and prints continuations of a
prompt. Run it at any point during training (even between --resume sessions)
to hear how the model is coming along.

Run from the repository root:
    .venv/bin/python chapters/24-train-your-mini-llm/python/sample.py --size tiny
    .venv/bin/python chapters/24-train-your-mini-llm/python/sample.py --size small --prompt "The night was"
"""

import argparse
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.data import get_datasets_directory  # noqa: E402
from common.device import select_best_available_device  # noqa: E402
from model import MiniLanguageModel, MODEL_SIZES  # noqa: E402

MINI_LLM_DIRECTORY = get_datasets_directory() / "mini_llm"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"
VOCABULARY_SIZE = 256 + 4096


def load_tokenizer():
    """Load merges.txt and rebuild each token's byte string (Chapter 20)."""
    token_bytes = {token_id: bytes([token_id]) for token_id in range(256)}
    merges = []
    for line in (MINI_LLM_DIRECTORY / "merges.txt").read_text().splitlines():
        left_id, right_id, new_id = map(int, line.split())
        merges.append(((left_id, right_id), new_id))
        token_bytes[new_id] = token_bytes[left_id] + token_bytes[right_id]
    return merges, token_bytes


def encode(text, merges):
    """Encode a prompt by replaying merges in order (Chapter 20)."""
    token_ids = list(text.encode("utf-8"))
    for (left_id, right_id), new_id in merges:
        rewritten, position = [], 0
        while position < len(token_ids):
            if position + 1 < len(token_ids) and token_ids[position] == left_id and token_ids[position + 1] == right_id:
                rewritten.append(new_id)
                position += 2
            else:
                rewritten.append(token_ids[position])
                position += 1
        token_ids = rewritten
    return token_ids


def decode(token_ids, token_bytes):
    return b"".join(token_bytes[i] for i in token_ids).decode("utf-8", errors="replace")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--size", choices=list(MODEL_SIZES), default="tiny")
    argument_parser.add_argument("--prompt", default="It was ")
    argument_parser.add_argument("--length", type=int, default=300, help="tokens to generate")
    argument_parser.add_argument("--temperature", type=float, default=0.8)
    argument_parser.add_argument("--top-k", type=int, default=40)
    parsed_arguments = argument_parser.parse_args()

    checkpoint_path = CHECKPOINT_DIRECTORY / f"mini_llm_{parsed_arguments.size}.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint {checkpoint_path.name} - train first with train_mini_llm.py --size {parsed_arguments.size}")
        sys.exit(1)

    device = select_best_available_device()
    merges, token_bytes = load_tokenizer()
    model = MiniLanguageModel(VOCABULARY_SIZE, MODEL_SIZES[parsed_arguments.size]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded {checkpoint_path.name} at step {checkpoint['step_number']:,}\n")

    prompt_ids = torch.tensor([encode(parsed_arguments.prompt, merges)], device=device)
    generated = model.generate(prompt_ids, parsed_arguments.length,
                               temperature=parsed_arguments.temperature, top_k=parsed_arguments.top_k)
    print("-" * 60)
    print(decode(generated[0].tolist(), token_bytes))
    print("-" * 60)


if __name__ == "__main__":
    main()
