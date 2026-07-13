"""Chapter 20 - byte-pair encoding (BPE) from scratch: how text becomes tokens.

Language models do not read characters or words - they read TOKENS, chunks
discovered from data by a compression algorithm. This script:
  1. trains a byte-level BPE tokenizer on Shakespeare (merge the most common
     pair, repeat - that is the whole algorithm),
  2. shows the first merges being discovered (they are recognizable English),
  3. encodes and decodes text, verifying the round trip is exact,
  4. writes the learned merges to datasets/bpe_merges.txt - the C encoder and
     Chapter 24's mini-LLM read that same file.

Run from the repository root:
    .venv/bin/python chapters/20-text-and-tokenization/python/bpe_tokenizer.py
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import get_datasets_directory, load_tiny_shakespeare  # noqa: E402

TRAINING_BYTES = 200_000     # enough text to find real statistics, fast to count
MERGE_COUNT = 200            # final vocabulary: 256 raw bytes + 200 merged tokens


def count_adjacent_pairs(token_ids):
    """Count how often each adjacent pair of tokens occurs.

    Arguments:
        token_ids: the corpus as a list of token ids.

    Returns a dict mapping (left_id, right_id) -> count.
    """
    pair_counts = {}
    for left_id, right_id in zip(token_ids, token_ids[1:]):
        pair_counts[(left_id, right_id)] = pair_counts.get((left_id, right_id), 0) + 1
    return pair_counts


def replace_pair_everywhere(token_ids, pair_to_replace, new_token_id):
    """Rewrite the corpus with every occurrence of a pair fused into one token.

    Arguments:
        token_ids: the corpus as a list of token ids.
        pair_to_replace: the (left_id, right_id) being merged.
        new_token_id: the id of the freshly minted merged token.
    """
    rewritten = []
    position = 0
    while position < len(token_ids):
        if (position + 1 < len(token_ids)
                and (token_ids[position], token_ids[position + 1]) == pair_to_replace):
            rewritten.append(new_token_id)
            position += 2
        else:
            rewritten.append(token_ids[position])
            position += 1
    return rewritten


def train_bpe(text, merge_count):
    """The whole BPE training algorithm: merge the most frequent pair, repeat.

    Arguments:
        text: the training corpus.
        merge_count: how many new tokens to mint.

    Returns (merges, vocabulary): merges is the ordered list of
    ((left_id, right_id), new_id); vocabulary maps every id to its bytes.
    Starting from raw bytes (ids 0-255) means ANY text - any language, any
    symbol - is representable; merges only add shortcuts. This is exactly
    GPT-2's design.
    """
    token_ids = list(text.encode("utf-8"))
    vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}
    merges = []

    for merge_index in range(merge_count):
        pair_counts = count_adjacent_pairs(token_ids)
        most_frequent_pair = max(pair_counts, key=pair_counts.get)
        new_token_id = 256 + merge_index
        token_ids = replace_pair_everywhere(token_ids, most_frequent_pair, new_token_id)
        merges.append((most_frequent_pair, new_token_id))
        vocabulary[new_token_id] = vocabulary[most_frequent_pair[0]] + vocabulary[most_frequent_pair[1]]

        if merge_index < 8 or merge_index in (49, 99, 199):
            left_text = vocabulary[most_frequent_pair[0]].decode("utf-8", errors="replace")
            right_text = vocabulary[most_frequent_pair[1]].decode("utf-8", errors="replace")
            merged_text = vocabulary[new_token_id].decode("utf-8", errors="replace")
            print(f"   merge {merge_index + 1:>3}: {left_text!r} + {right_text!r} -> {merged_text!r}"
                  f"   (seen {pair_counts[most_frequent_pair]:,} times)")
    return merges, vocabulary


def encode(text, merges):
    """Turn text into token ids by replaying the learned merges in order.

    Arguments:
        text: any string.
        merges: the ordered merge list from train_bpe.

    Earlier merges were more frequent in training, so they get priority -
    replaying in training order is what makes encoding deterministic.
    """
    token_ids = list(text.encode("utf-8"))
    for pair, new_token_id in merges:
        token_ids = replace_pair_everywhere(token_ids, pair, new_token_id)
    return token_ids


def decode(token_ids, vocabulary):
    """Token ids back to text: concatenate each token's bytes, decode UTF-8."""
    return b"".join(vocabulary[token_id] for token_id in token_ids).decode("utf-8", errors="replace")


def save_merges(merges, output_path):
    """Write merges as 'left right new' lines - the format the C encoder and
    Chapter 24 read back."""
    with open(output_path, "w") as merges_file:
        for (left_id, right_id), new_token_id in merges:
            merges_file.write(f"{left_id} {right_id} {new_token_id}\n")
    print(f"\nSaved {len(merges)} merges to {output_path}")


def main():
    text = load_tiny_shakespeare()
    training_text = text[:TRAINING_BYTES]
    print(f"1. Training BPE on {TRAINING_BYTES:,} bytes of Shakespeare, {MERGE_COUNT} merges")
    print("   watch the first merges - the algorithm discovers English on its own:")
    merges, vocabulary = train_bpe(training_text, MERGE_COUNT)

    print("\n2. Some full tokens the vocabulary ended up with:")
    interesting_tokens = sorted(vocabulary, key=lambda t: len(vocabulary[t]), reverse=True)[:10]
    print("   " + "  ".join(repr(vocabulary[t].decode('utf-8', errors='replace')) for t in interesting_tokens))

    print("\n3. Encoding and decoding")
    sample_sentence = "To be, or not to be: that is the question."
    token_ids = encode(sample_sentence, merges)
    print(f"   text:   {sample_sentence!r}  ({len(sample_sentence.encode('utf-8'))} bytes)")
    print(f"   tokens: {token_ids}  ({len(token_ids)} tokens)")
    print("   pieces: " + " | ".join(vocabulary[t].decode('utf-8', errors='replace') for t in token_ids))
    round_trip = decode(token_ids, vocabulary)
    print(f"   decode(encode(text)) == text: {round_trip == sample_sentence}")

    fresh_text = text[TRAINING_BYTES:TRAINING_BYTES + 50_000]
    compressed_length = len(encode(fresh_text, merges))
    print(f"\n4. Compression on text the tokenizer never saw:")
    print(f"   {len(fresh_text.encode('utf-8')):,} bytes -> {compressed_length:,} tokens "
          f"({len(fresh_text.encode('utf-8')) / compressed_length:.2f} bytes per token)")
    print("   Every token the model reads carries that much more text - context windows")
    print("   and training budgets are measured in tokens, so compression is capacity.")

    save_merges(merges, get_datasets_directory() / "bpe_merges.txt")


if __name__ == "__main__":
    main()
