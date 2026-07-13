"""Chapter 25, step 1 - export a trained mini-LLM to a single flat file the
pure-C engine reads.

Writes checkpoints/mini_llm_<size>.bin containing, in a fixed documented order:
  a small header (magic, dimensions),
  the tokenizer's merges,
  every weight tensor as float32.
Optionally quantizes the big matrices to int8 (--quantize) - the same file
format, smaller, and the C engine reads either.

Run from the repository root (after training in Chapter 24):
    .venv/bin/python chapters/25-llm-inference-in-c/python/export_llm_for_c.py --size small
    .venv/bin/python chapters/25-llm-inference-in-c/python/export_llm_for_c.py --size small --quantize
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "chapters" / "24-train-your-mini-llm" / "python"))

from common.data import get_datasets_directory  # noqa: E402
from model import MiniLanguageModel, MODEL_SIZES  # noqa: E402

MINI_LLM_DIRECTORY = get_datasets_directory() / "mini_llm"
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"
VOCABULARY_SIZE = 256 + 4096
EXPORT_MAGIC = 0x4C4C4D31            # "LLM1"


def write_tensor(output_file, tensor, quantize):
    """Write one tensor as float32, or as int8 + a per-tensor scale if asked.

    Arguments:
        output_file: the open binary file.
        tensor: a torch tensor.
        quantize: when True and the tensor is 2-D (a weight matrix), store it
            as signed 8-bit integers plus one float32 scale, cutting its size
            to a quarter. Chapter 25 explains the accuracy trade.

    Returns the number of bytes written (for the summary).
    """
    array = tensor.detach().to(torch.float32).cpu().numpy()
    if quantize and array.ndim == 2:
        # Symmetric per-tensor quantization: map [-max, +max] onto [-127, 127].
        # Simple and transparent; real engines quantize per-row for accuracy.
        scale = numpy.abs(array).max() / 127.0
        quantized = numpy.round(array / scale).astype(numpy.int8)
        output_file.write(struct.pack("f", scale))
        output_file.write(quantized.tobytes())
        return 4 + quantized.nbytes
    output_file.write(array.astype(numpy.float32).tobytes())
    return array.nbytes


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--size", choices=list(MODEL_SIZES), default="small")
    argument_parser.add_argument("--quantize", action="store_true", help="int8 weight matrices (4x smaller)")
    parsed_arguments = argument_parser.parse_args()

    config = MODEL_SIZES[parsed_arguments.size]
    checkpoint_path = CHECKPOINT_DIRECTORY / f"mini_llm_{parsed_arguments.size}.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint {checkpoint_path.name} - train it in Chapter 24 first.")
        sys.exit(1)

    model = MiniLanguageModel(VOCABULARY_SIZE, config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    merges = [tuple(map(int, line.split())) for line in (MINI_LLM_DIRECTORY / "merges.txt").read_text().splitlines()]
    suffix = "_int8" if parsed_arguments.quantize else ""
    output_path = CHECKPOINT_DIRECTORY / f"mini_llm_{parsed_arguments.size}{suffix}.bin"

    with open(output_path, "wb") as output_file:
        # Header: magic + the six numbers the C engine needs to size everything.
        output_file.write(struct.pack("<7i", EXPORT_MAGIC, VOCABULARY_SIZE,
                                      config["context_length"], config["embedding_size"],
                                      config["block_count"], config["head_count"],
                                      1 if parsed_arguments.quantize else 0))

        # Tokenizer merges: count, then (left, right, new) triples.
        output_file.write(struct.pack("<i", len(merges)))
        for left_id, right_id, new_id in merges:
            output_file.write(struct.pack("<3i", left_id, right_id, new_id))

        # Weights, in the exact order model.py defines them - the C engine
        # walks this same order. float32 for norms/embeddings/bias (small and
        # sensitive), quantizable for the big linear matrices.
        total_bytes = 0
        state = model.state_dict()
        total_bytes += write_tensor(output_file, state["token_embedding.weight"], False)
        total_bytes += write_tensor(output_file, state["position_embedding.weight"], False)
        for block_index in range(config["block_count"]):
            prefix = f"blocks.{block_index}."
            for name in ("attention_norm.weight", "attention_norm.bias"):
                total_bytes += write_tensor(output_file, state[prefix + name], False)
            total_bytes += write_tensor(output_file, state[prefix + "attention.query_key_value_projection.weight"], parsed_arguments.quantize)
            total_bytes += write_tensor(output_file, state[prefix + "attention.query_key_value_projection.bias"], False)
            total_bytes += write_tensor(output_file, state[prefix + "attention.output_projection.weight"], parsed_arguments.quantize)
            total_bytes += write_tensor(output_file, state[prefix + "attention.output_projection.bias"], False)
            for name in ("mlp_norm.weight", "mlp_norm.bias"):
                total_bytes += write_tensor(output_file, state[prefix + name], False)
            total_bytes += write_tensor(output_file, state[prefix + "mlp.0.weight"], parsed_arguments.quantize)
            total_bytes += write_tensor(output_file, state[prefix + "mlp.0.bias"], False)
            total_bytes += write_tensor(output_file, state[prefix + "mlp.2.weight"], parsed_arguments.quantize)
            total_bytes += write_tensor(output_file, state[prefix + "mlp.2.bias"], False)
        total_bytes += write_tensor(output_file, state["final_norm.weight"], False)
        total_bytes += write_tensor(output_file, state["final_norm.bias"], False)
        # next_token_head shares the token embedding (weight tying), already written.

    megabytes = output_path.stat().st_size / 1e6
    print(f"Wrote {output_path.relative_to(REPOSITORY_ROOT)} ({megabytes:.1f} MB, "
          f"{'int8-quantized' if parsed_arguments.quantize else 'float32'})")
    print("Build and run the C engine:")
    print(f"  make -C chapters/25-llm-inference-in-c/c")
    print(f"  ./chapters/25-llm-inference-in-c/c/build/llm_inference {output_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
