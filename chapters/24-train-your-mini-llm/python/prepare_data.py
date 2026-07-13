"""Chapter 24, step 1 - build the mini-LLM's dataset and tokenizer.

Four stages, each idempotent (rerunning skips finished work):
  1. download ten public-domain novels from Project Gutenberg (~8 MB),
  2. clean them (strip the Gutenberg license headers/footers),
  3. train a byte-level BPE tokenizer on the corpus (Chapter 20's algorithm,
     plus the word-caching trick that makes encoding 8 MB practical),
  4. encode the whole corpus and write it as a flat binary of uint16 token
     ids - the exact file the training script (and Chapter 25's C engine's
     inspector) will read.

Run from the repository root:
    .venv/bin/python chapters/24-train-your-mini-llm/python/prepare_data.py
"""

import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import get_datasets_directory  # noqa: E402

MINI_LLM_DIRECTORY = get_datasets_directory() / "mini_llm"

# Ten classics: enough variety that the model cannot just memorize one voice.
GUTENBERG_BOOKS = {
    "pride_and_prejudice": 1342,
    "moby_dick": 2701,
    "sherlock_holmes": 1661,
    "frankenstein": 84,
    "dracula": 345,
    "tale_of_two_cities": 98,
    "alice_in_wonderland": 11,
    "tom_sawyer": 74,
    "war_of_the_worlds": 36,
    "treasure_island": 120,
}

MERGE_COUNT = 4096          # vocabulary: 256 bytes + 4096 learned tokens
BPE_TRAINING_BYTES = 2_000_000   # merges are learned from a sample; encoding uses all


def download_and_clean_corpus():
    """Stage 1+2: fetch the books and strip Gutenberg's boilerplate."""
    corpus_path = MINI_LLM_DIRECTORY / "corpus.txt"
    if corpus_path.exists():
        print(f"1-2. corpus.txt already exists ({corpus_path.stat().st_size:,} bytes) - skipping download")
        return corpus_path.read_text(encoding="utf-8")

    MINI_LLM_DIRECTORY.mkdir(exist_ok=True)
    book_texts = []
    for book_name, gutenberg_id in GUTENBERG_BOOKS.items():
        book_url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
        print(f"  downloading {book_name} ...")
        raw_text = urlopen(book_url).read().decode("utf-8", errors="replace")

        # Gutenberg wraps every book in '*** START OF ... ***' / '*** END OF
        # ... ***' markers; the real text lives between them.
        start_match = re.search(r"\*\*\* START OF.*?\*\*\*", raw_text)
        end_match = re.search(r"\*\*\* END OF.*?\*\*\*", raw_text)
        body = raw_text[start_match.end():end_match.start()] if start_match and end_match else raw_text
        book_texts.append(body.strip())

    corpus_text = "\n\n".join(book_texts)
    corpus_path.write_text(corpus_text, encoding="utf-8")
    print(f"1-2. wrote corpus.txt: {corpus_path.stat().st_size:,} bytes from {len(GUTENBERG_BOOKS)} books")
    return corpus_text


def train_tokenizer(corpus_text):
    """Stage 3: BPE trained efficiently, the way GPT-2 does it.

    The naive Chapter 20 loop recounts and rewrites the WHOLE token list every
    merge - fine for a chapter demo on 200 KB, hopeless on megabytes. The
    production trick: English is mostly repeated WORDS, so
      1. split into word-like chunks and count how often each distinct chunk
         occurs (a few tens of thousands of unique words, not millions of
         tokens),
      2. represent each unique word once as a list of token ids with its
         frequency, and count pairs weighted by that frequency,
      3. each merge only rewrites the words that actually contain the pair.
    Same result as the naive algorithm, minutes instead of hours.
    """
    merges_path = MINI_LLM_DIRECTORY / "merges.txt"
    if merges_path.exists():
        print("3.   merges.txt already exists - skipping tokenizer training")
        merges = []
        for line in merges_path.read_text().splitlines():
            left_id, right_id, new_id = map(int, line.split())
            merges.append(((left_id, right_id), new_id))
        return merges

    print(f"3.   training BPE: {MERGE_COUNT} merges on {BPE_TRAINING_BYTES:,} bytes "
          f"(word-frequency method; a progress line every 512 merges)")
    training_start = time.perf_counter()

    # Stage 1: unique words with frequencies (leading whitespace kept on each).
    word_frequencies = Counter(re.findall(r"\s*\S+|\s+", corpus_text[:BPE_TRAINING_BYTES]))
    # Each unique word as a mutable list of token ids, paired with its count.
    word_token_lists = [[list(word.encode("utf-8")), count] for word, count in word_frequencies.items()]
    print(f"     {len(word_token_lists):,} unique words to learn from")

    merges = []
    for merge_index in range(MERGE_COUNT):
        # Stage 2: count adjacent pairs, weighted by each word's frequency.
        pair_counts = Counter()
        for token_list, count in word_token_lists:
            for left_id, right_id in zip(token_list, token_list[1:]):
                pair_counts[(left_id, right_id)] += count
        if not pair_counts:
            break
        most_frequent_pair = pair_counts.most_common(1)[0][0]
        new_token_id = 256 + merge_index
        merges.append((most_frequent_pair, new_token_id))

        # Stage 3: fuse the pair inside only the words that contain it.
        left_target, right_target = most_frequent_pair
        for entry in word_token_lists:
            token_list = entry[0]
            if left_target not in token_list:
                continue
            rewritten, position = [], 0
            while position < len(token_list):
                if (position + 1 < len(token_list)
                        and token_list[position] == left_target and token_list[position + 1] == right_target):
                    rewritten.append(new_token_id)
                    position += 2
                else:
                    rewritten.append(token_list[position])
                    position += 1
            entry[0] = rewritten

        if (merge_index + 1) % 512 == 0:
            elapsed = time.perf_counter() - training_start
            print(f"     merge {merge_index + 1}/{MERGE_COUNT} ({elapsed:.0f} s)")

    with open(merges_path, "w") as merges_file:
        for (left_id, right_id), new_token_id in merges:
            merges_file.write(f"{left_id} {right_id} {new_token_id}\n")
    print(f"     wrote merges.txt ({time.perf_counter() - training_start:.0f} s total)")
    return merges


def encode_corpus(corpus_text, merges):
    """Stage 4: encode all ~8 MB and write uint16 token ids.

    Encoding replays merges per chunk of text. The crucial trick (GPT-2 does
    the same): split the text into word-like chunks and CACHE each distinct
    chunk's encoding - English reuses words, so almost every chunk is a cache
    hit and 8 MB encodes in seconds instead of hours.
    """
    tokens_path = MINI_LLM_DIRECTORY / "tokens.bin"
    if tokens_path.exists():
        token_count = tokens_path.stat().st_size // 2
        print(f"4.   tokens.bin already exists ({token_count:,} tokens) - skipping encoding")
        return

    merge_rank = {pair: (rank, new_id) for rank, (pair, new_id) in enumerate(merges)}

    def encode_chunk(chunk_bytes):
        chunk_ids = list(chunk_bytes)
        while len(chunk_ids) >= 2:
            # Apply the best-ranked (earliest-learned = most frequent) merge
            # present in this chunk, repeatedly, until none remain.
            best = None
            for pair in zip(chunk_ids, chunk_ids[1:]):
                if pair in merge_rank and (best is None or merge_rank[pair][0] < merge_rank[best][0]):
                    best = pair
            if best is None:
                break
            left_target, right_target, new_id = best[0], best[1], merge_rank[best][1]
            rewritten, position = [], 0
            while position < len(chunk_ids):
                if position + 1 < len(chunk_ids) and chunk_ids[position] == left_target and chunk_ids[position + 1] == right_target:
                    rewritten.append(new_id)
                    position += 2
                else:
                    rewritten.append(chunk_ids[position])
                    position += 1
            chunk_ids = rewritten
        return chunk_ids

    print("4.   encoding the full corpus (word-cached)...")
    encoding_start = time.perf_counter()
    chunk_cache = {}
    all_token_ids = []
    # Words keep their leading space ("the" and " the" are different chunks,
    # matching how the merges were learned from running text).
    for chunk in re.findall(r"\s*\S+|\s+", corpus_text):
        chunk_bytes = chunk.encode("utf-8")
        if chunk_bytes not in chunk_cache:
            chunk_cache[chunk_bytes] = encode_chunk(chunk_bytes)
        all_token_ids.extend(chunk_cache[chunk_bytes])

    token_array = numpy.array(all_token_ids, dtype=numpy.uint16)
    token_array.tofile(tokens_path)
    elapsed = time.perf_counter() - encoding_start
    print(f"     wrote tokens.bin: {len(token_array):,} tokens from {len(corpus_text.encode('utf-8')):,} bytes "
          f"({len(corpus_text.encode('utf-8')) / len(token_array):.2f} bytes/token, {elapsed:.0f} s, "
          f"{len(chunk_cache):,} distinct chunks cached)")


def main():
    corpus_text = download_and_clean_corpus()
    merges = train_tokenizer(corpus_text)
    encode_corpus(corpus_text, merges)
    print("\nData ready. Next: train_mini_llm.py")


if __name__ == "__main__":
    main()
