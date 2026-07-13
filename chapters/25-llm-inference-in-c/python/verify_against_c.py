"""Chapter 25 - verify the C engine's forward pass matches PyTorch's.

Generation is random (sampling), so matching TEXT is not the right check.
Instead we check the deterministic part: given the same prompt, do the two
implementations compute the same LOGITS? This script runs the PyTorch model on
a fixed prompt, prints its top-5 next tokens with logit values, and tells you
what the C engine should print for the same prompt with a temperature near 0
(greedy), so you can compare by eye. Small float differences are expected
(different operation order); the ranking must match.

Run from the repository root:
    .venv/bin/python chapters/25-llm-inference-in-c/python/verify_against_c.py --size small
"""

import argparse
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "chapters" / "24-train-your-mini-llm" / "python"))

from common.data import get_datasets_directory  # noqa: E402
from model import MiniLanguageModel, MODEL_SIZES  # noqa: E402

MINI_LLM_DIRECTORY = get_datasets_directory() / "mini_llm"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"
VOCABULARY_SIZE = 256 + 4096


def load_tokenizer():
    token_bytes = {i: bytes([i]) for i in range(256)}
    merges = []
    for line in (MINI_LLM_DIRECTORY / "merges.txt").read_text().splitlines():
        left, right, new = map(int, line.split())
        merges.append(((left, right), new))
        token_bytes[new] = token_bytes[left] + token_bytes[right]
    return merges, token_bytes


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    for (left, right), new in merges:
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == left and ids[i + 1] == right:
                out.append(new); i += 2
            else:
                out.append(ids[i]); i += 1
        ids = out
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=list(MODEL_SIZES), default="small")
    parser.add_argument("--prompt", default="The night was ")
    args = parser.parse_args()

    checkpoint_path = CHECKPOINT_DIRECTORY / f"mini_llm_{args.size}.pt"
    model = MiniLanguageModel(VOCABULARY_SIZE, MODEL_SIZES[args.size])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state"])
    model.eval()

    merges, token_bytes = load_tokenizer()
    prompt_ids = torch.tensor([encode(args.prompt, merges)])
    with torch.no_grad():
        logits = model(prompt_ids)[0, -1]

    top_values, top_indices = logits.topk(5)
    print(f"PyTorch, prompt {args.prompt!r} - the 5 most likely next tokens:")
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        piece = token_bytes[index].decode("utf-8", errors="replace")
        print(f"   logit {value:+7.3f}   token {index:>4}  {piece!r}")
    print("\nRun the C engine on the same prompt (its greedy pick is the top token here).")
    print("The float values will differ slightly (operation order); the RANKING must match.")


if __name__ == "__main__":
    main()
