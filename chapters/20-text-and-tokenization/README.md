# Chapter 20 — Text and tokenization

Part V begins the road to your own language model, and it begins where every LLM begins: turning text into numbers. The answer used by GPT and virtually every modern model is **byte-pair encoding (BPE)** — a compression algorithm from 1994 that, given nothing but raw text and a counter, rediscovers letters' habits, then syllables, then whole words. You will build it completely (train in Python, encode in C from the same merges file), and the tokenizer you train here is *literally the one Chapter 24's mini-LLM will use*.

<!-- CONTENTS_START -->
## Contents

- [What you will learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [1. The goldilocks problem](#1-the-goldilocks-problem)
- [2. BPE: the whole algorithm](#2-bpe-the-whole-algorithm)
- [3. Compression is capacity](#3-compression-is-capacity)
- [4. What happens to the ids: embeddings (preview)](#4-what-happens-to-the-ids-embeddings-preview)
- [Code walkthrough](#code-walkthrough)
- [Run it](#run-it)
- [What the C version covers](#what-the-c-version-covers)
- [Exercises](#exercises)
- [Next](#next)

<!-- CONTENTS_END -->

## What you will learn

- Why models read tokens (not characters, not words).
- The BPE algorithm — all of it; it fits in 40 lines.
- Encoding, decoding, and the compression-is-capacity principle.
- Embeddings: what happens to token ids inside the model (preview).

## Prerequisites

- [Chapter 12](../12-data-pipelines/README.md) — data thinking.
- No new math.

## 1. The goldilocks problem

Neural networks eat numbers, so text needs an alphabet of ids. Two obvious choices both fail at scale:

- **Characters** (Chapter 21 will use them for simplicity): tiny vocabulary, but sequences get long — and sequence length is the scarcest resource a language model has (attention, Chapter 22, pays *quadratically* for it). "The" costs 3 steps of model thought instead of 1.
- **Words**: short sequences, but the vocabulary explodes (every name, typo, and new word), and anything unseen at training time becomes an unrepresentable `<UNK>`.

**Tokens** sit between: a vocabulary of a few thousand (here) to ~100,000 (GPT-scale) *learned chunks* — common words whole, rare words in pieces. The trick is where the chunks come from: not from a linguist, but from counting.

## 2. BPE: the whole algorithm

Start from **bytes** (ids 0–255 — so any text in any language is representable from the first moment; this is exactly GPT-2's design). Then:

> Count every adjacent pair of tokens in the corpus. Fuse the most frequent pair into a new token. Repeat.

That is the entire algorithm. Watch it run on Shakespeare (real output):

```
   merge   1: 'e' + ' ' -> 'e '   (seen 5,249 times)
   merge   2: 't' + 'h' -> 'th'   (seen 4,065 times)
   merge   5: 'o' + 'u' -> 'ou'   (seen 2,529 times)
   merge   8: 'e' + 'r' -> 'er'   (seen 2,128 times)
   merge 200: 'm' + 'an' -> 'man' (seen 127 times)
```

![BPE building the token "the " from bytes by successive merges](figures/bpe-merges.svg)

Nobody told it about English. Frequency alone finds `th`, `ou`, `er` — the true statistical joints of the language — then assembles words; by merge 200 the vocabulary contains `' the '`, `'your '`, `'have '`, and (this being Shakespeare) `'MENENIUS:\n'` as single tokens. **A tokenizer is a mirror of its training corpus** — which is why code-heavy models train tokenizers on code, and why oddly-tokenized strings can make LLMs behave strangely.

**Encoding** new text replays the learned merges in training order (earlier = more frequent = higher priority). **Decoding** is trivial: every token knows its bytes; concatenate. The round trip is exact — verified in both languages.

## 3. Compression is capacity

On held-out Shakespeare the 456-token vocabulary packs **1.85 bytes per token** (production tokenizers with 100k vocabularies reach ~4). This number quietly governs LLM economics: context windows, training budgets, and API prices are all counted in tokens, so every extra byte-per-token means the same model reads, remembers, and learns from proportionally more text. When Chapter 24 trains the mini-LLM with this exact tokenizer, its 256-token context will hold ~470 characters of story instead of 256.

## 4. What happens to the ids: embeddings (preview)

A token id is just an index — 315 is not "more" than 314. The model's first layer is an **embedding table**: a learned matrix with one row of, say, 128 numbers per vocabulary entry; reading token 315 means fetching row 315. Those vectors are parameters like any other, trained by the same backpropagation as everything since Chapter 5 — and after training, tokens used similarly end up with similar vectors (the famous king − man + woman ≈ queen arithmetic falls out of this). Chapters 21–24 all start with an embedding table; now you know what it is.

## Code walkthrough

The example is `python/bpe_tokenizer.py`. The whole algorithm is two small helpers and a loop — genuinely simple. No prior programming assumed.

### Step 1 — the two helpers: count pairs, and fuse a pair

```python
def count_adjacent_pairs(token_ids):
    pair_counts = {}
    for left_id, right_id in zip(token_ids, token_ids[1:]):
        pair_counts[(left_id, right_id)] = pair_counts.get((left_id, right_id), 0) + 1
    return pair_counts
```

`count_adjacent_pairs` walks the sequence looking at each neighbor pair (`zip(token_ids, token_ids[1:])` pairs each token with the next) and tallies how often each pair appears — a dictionary of counts. Its partner `replace_pair_everywhere` does the opposite: it rewrites the sequence with every occurrence of one chosen pair fused into a single new token. Count, then fuse — those are the only two operations BPE needs.

### Step 2 — training: merge the most frequent pair, over and over

```python
token_ids = list(text.encode("utf-8"))                # start from raw bytes
vocabulary = {token_id: bytes([token_id]) for token_id in range(256)}
for merge_index in range(merge_count):
    pair_counts = count_adjacent_pairs(token_ids)
    most_frequent_pair = max(pair_counts, key=pair_counts.get)
    new_token_id = 256 + merge_index
    token_ids = replace_pair_everywhere(token_ids, most_frequent_pair, new_token_id)
    merges.append((most_frequent_pair, new_token_id))
```

This is the *entire* BPE training algorithm. Start with the text as raw **bytes** (0–255) — starting from bytes means *any* text in any language is representable, which is exactly GPT-2's design. Then repeat: count all pairs, find the **most frequent** one (`max(..., key=pair_counts.get)`), mint a new token id for it, and fuse every occurrence. Each pass adds one shortcut to the vocabulary. The merges it discovers first are recognizable English — `th`, `ou`, `er` — the algorithm rediscovers spelling from statistics alone.

### Step 3 — encoding: replay the merges, in order

```python
token_ids = list(text.encode("utf-8"))
for pair, new_token_id in merges:
    token_ids = replace_pair_everywhere(token_ids, pair, new_token_id)
```

To tokenize new text, `encode` starts from its bytes and replays the learned merges **in the order they were learned**. That order is priority: earlier merges were more frequent in training, so they apply first, which is what makes tokenization deterministic — the same string always becomes the same tokens.

### Step 4 — decoding: bytes back to text

`decode` is trivial because every token id remembers its bytes in the `vocabulary` dict: concatenate each token's bytes and interpret as UTF-8. The round trip `decode(encode(text)) == text` is exact. Finally `save_merges` writes the merge list to a file — the contract that the C encoder and Chapter 24's mini-LLM both read, so all three share one tokenizer.

The C file `c/bpe_encoder.c` loads that merges file and produces **identical token IDs** — one tokenizer, two languages. Encoding (not training) is what runs at every inference, so it is the half worth owning in C.

### Quick reference

| Function | What it does | What to notice |
|----------|--------------|----------------|
| `count_adjacent_pairs(token_ids)` | Counts how often each adjacent pair occurs. | The "which pair is most frequent?" question BPE keeps asking. |
| `replace_pair_everywhere(token_ids, pair, new_id)` | Fuses a pair into one new token everywhere. | One merge, applied. |
| `train_bpe(text, merge_count)` | Count pairs, fuse the most frequent, repeat — from raw bytes. | Starting from bytes means *any* text is representable (GPT-2's design). |
| `encode(text, merges)` | Replays the learned merges *in order*. | Order = priority; this makes encoding deterministic. |
| `decode(token_ids, vocabulary)` | Concatenates each token's bytes. | Every token knows its bytes; the round trip is exact. |
| `save_merges(merges, path)` | Writes `left right new` lines. | The contract the C encoder and Chapter 24 read back. |

## Run it

```bash
.venv/bin/python chapters/20-text-and-tokenization/python/bpe_tokenizer.py   # trains, ~1 min; writes datasets/bpe_merges.txt
make -C chapters/20-text-and-tokenization/c && ./chapters/20-text-and-tokenization/c/build/bpe_encoder
```

## What the C version covers

The deployment half: load the Python-trained merges file, encode, decode. Training happens once; *encoding happens at every single inference*, which makes it the part worth owning in C — and Chapter 25's pure-C LLM engine reuses this exact code path. The two implementations produce **identical token ids** from one shared merges file: one tokenizer, two languages.

## Exercises

1. By hand: with merges `[t+h→th, th+e→the]`, encode `"then the"` step by step. (Mind the spaces.)
2. Retrain with 1,000 merges instead of 200. How do bytes-per-token and the longest tokens change? Where would you expect diminishing returns?
3. Train the tokenizer on only the *first half* of the alphabet's lines (grep for speakers A–L) and measure compression on the rest. Quantify the "mirror of its corpus" claim.
4. Feed the encoder a string with characters absent from Shakespeare (emoji, accented text). Verify nothing breaks and explain *why* byte-level BPE cannot have an out-of-vocabulary problem.
5. Challenge (C): the C encoder replays all 200 merges over the whole sequence (O(merges × length)). Implement the faster standard approach: repeatedly find the *highest-priority mergeable pair present* using the merge ranks. Verify identical output.

## Next

[Chapter 21 — Recurrent networks](../21-recurrent-networks/README.md)

<!-- NAV_START -->
---

[← Chapter 19: Speech recognition](../19-speech-recognition/README.md) · [↑ Course index](../../README.md) · [Chapter 21: Recurrent networks →](../21-recurrent-networks/README.md)

<!-- NAV_END -->
