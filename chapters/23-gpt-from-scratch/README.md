# Chapter 23 — GPT from scratch

The chapter where the pieces become the thing itself. A **GPT** — *generative pre-trained transformer* — is nothing beyond what you now own: Chapter 20's tokens, Chapter 22's blocks, Chapter 9's cross-entropy, Chapter 21's next-token task. You will write it in ~150 readable lines, train it on Shakespeare in about a minute, and watch the same object go from spraying random glyphs to writing blank verse with named characters. Chapter 24 will then scale exactly this file — nothing conceptually new remains.

<!-- CONTENTS_START -->
## Contents

- [What you will learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [1. The architecture, in full](#1-the-architecture-in-full)
- [2. Training, and the honest yardstick](#2-training-and-the-honest-yardstick)
- [3. Before and after](#3-before-and-after)
- [4. The knob at the end: sampling](#4-the-knob-at-the-end-sampling)
- [Code walkthrough](#code-walkthrough)
- [Run it](#run-it)
- [What the C version covers](#what-the-c-version-covers)
- [Exercises](#exercises)
- [Next](#next)

<!-- CONTENTS_END -->

## What you will learn

- The complete GPT parts list (all of it already yours).
- What "training a language model" actually optimizes, and how generation works.
- Perplexity — the standard yardstick for language models.
- Sampling strategies: greedy, temperature, top-k (the C program measures them).

## Prerequisites

- [Chapter 22](../22-attention-and-transformers/README.md) — attention and the block.
- [Chapter 21](../21-recurrent-networks/README.md) — language modeling and sampling.

## 1. The architecture, in full

![The GPT stack from text input to next-token distribution and the generation loop](figures/gpt-architecture.svg)

That figure is the *entire* architecture of a GPT — including the real ones. From bottom to top: the tokenizer turns text into ids (characters here, 65 of them, for maximum transparency; Chapter 24 swaps in BPE). Each token's embedding is added to a **position embedding** (attention is order-blind; Chapter 22). N transformer blocks alternate *communicate* (causal attention) and *compute* (MLP), with residuals throughout. A final linear layer maps each position's vector to logits over the vocabulary: a prediction of the **next** token at every position simultaneously.

Today's build: 4 blocks, 4 heads, 128-dimensional embeddings, 128 characters of context — **826,433 parameters**. GPT-2 was this times ~180 with the same code shape; GPT-4-class models, this times millions plus engineering. The scaling story is Chapter 24's; the *ideas* end here.

Two code notes worth reading in the source: Q, K, V for all heads come from a single fused linear layer (an efficiency idiom in every real GPT), and the MLP uses GELU — ReLU's smooth cousin, the transformer convention since BERT.

## 2. Training, and the honest yardstick

Nothing you have not done since Chapter 21: random 128-character windows, targets shifted one step, cross-entropy at every position, AdamW. A held-out 10% of the corpus (Chapter 12 discipline) gives the number that matters:

```
   step    train loss   validation loss   seconds
      1       4.2989            4.1720         1
    200       2.5030            2.4968         5
   1000       1.9872            2.0118        21
   3000       1.4800            1.6910        62

Final validation perplexity: 5.4 (untrained would be 65)
```

**Perplexity** = $e^{\text{loss}}$, read as: "the model is as uncertain as a fair choice among this many options." Untrained, 65 (all characters equally likely); after one minute, 5.4 — the model has compressed English spelling, Shakespearean names, and dramatic formatting into effective five-way uncertainty. Note also the train/validation gap opening after step 1000: Chapter 11's overfitting, right on schedule for 0.8M parameters on 1MB of text.

## 3. Before and after

Generation is the purple loop in the figure: predict, sample one token, append, run again. The same weights, same prompt, one minute apart:

```
Before:  ROMEO:i;wENB3xtytKRvPnpHH$iAAClMLK&o
         SEahIET -r$ijdO' ST3DdaueUBp

After:   ROMEO:
         Bnows to appare you affer'd to in man.

         POLIXENES:
         Say he farents, have that would, home reson
         And the master'd to be to there our in the ears
         Of grave and he foult the fear's better--clowed
```

Same critique as Chapter 21 — English-shaped nonsense — but notice what one minute and 0.8M parameters bought over the RNN: cleaner line structure, real character names used consistently (POLIXENES and FLORIZELL are genuine *Winter's Tale* speakers), and phrases with grammatical spines ("Say he ..., have that would"). The recipe from here to a model that *means* things is: more parameters, more data, more compute, better tokens — that is Chapter 24, and then the industry.

## 4. The knob at the end: sampling

Every generated token ends in one decision: given the logits, pick. The C program takes one realistic logit vector ("To be, or not to ___") and runs each strategy 10,000 times:

```
  greedy (T -> 0)        100.0%   0.0%  ...            <- deterministic, gets repetitive
  temperature 0.5         93.2%   4.3%   1.8%  ...     <- confident, safe
  temperature 1.0         67.5%  14.9%   9.0%  ...  0.0%
  temperature 1.5         50.4%  18.4%  13.2%  ...  0.5%  0.2%   <- garbage tokens leak in
  top-k 3 (at T=1)        74.4%  15.9%   9.7%   0%   0%   0%     <- tail amputated
```

The story is in the last columns (the "xylophone" and "%" tokens): plain sampling occasionally picks garbage, high temperature often does, and **top-k never can** — which is why production systems combine moderate temperature with top-k or top-p. When an LLM playground shows you those sliders, this table is what they do.

## Code walkthrough

The example is `python/train_gpt_shakespeare.py`. Three classes stack into a GPT, and each one is a chapter you already finished. No prior programming assumed.

### Step 1 — multi-head causal attention

```python
queries, keys, values = self.query_key_value_projection(token_features).chunk(3, dim=-1)
def split_heads(tensor):
    return tensor.view(batch_size, sequence_length, HEAD_COUNT, head_size).transpose(1, 2)
queries, keys, values = split_heads(queries), split_heads(keys), split_heads(values)
attended = nn.functional.scaled_dot_product_attention(queries, keys, values, is_causal=True)
```

`CausalSelfAttention` is Chapter 22's attention, done efficiently and with several **heads**. One `nn.Linear` computes all three of Q, K, V at once (`.chunk(3, ...)` then splits them apart) — a speed idiom in every real GPT. `split_heads` reshapes so each of the 4 heads works in its own slice of the embedding, letting the model attend several different ways in parallel (one head might track subjects, another quotation marks). The actual attention is `scaled_dot_product_attention` with `is_causal=True` — the exact four-line mechanism you verified in Chapter 22, mask and all.

### Step 2 — the transformer block: communicate, then compute

```python
def forward(self, token_features):
    token_features = token_features + self.attention(self.attention_norm(token_features))
    token_features = token_features + self.mlp(self.mlp_norm(token_features))
    return token_features
```

Every piece here is an old friend. The two `token_features +` are Chapter 14's **residual** shortcuts; `attention_norm`/`mlp_norm` are Chapter 11's **layer normalization**; the `mlp` is Chapter 9's little two-layer network. And the *pattern* is the whole idea of a transformer: **attention mixes information across positions** (tokens look at each other), then the **MLP thinks about each position on its own** — communicate, then compute — with a residual around each so gradients flow and depth is safe.

### Step 3 — the whole GPT: tokens, positions, blocks, head

```python
token_features = self.token_embedding(token_ids) + self.position_embedding(positions)
token_features = self.final_norm(self.blocks(token_features))
return self.next_token_head(token_features)
```

`MiniGPT` assembles it. Each token id becomes a vector two ways and they are **added**: `token_embedding` says *what* the token is, and `position_embedding` says *where* it sits — necessary because attention on its own is order-blind (it sees a bag of tokens, not a sequence). Then `self.blocks` runs the 4 transformer blocks in a row, a final layer norm cleans up, and `next_token_head` (an `nn.Linear`) turns each position's vector into a score for every possible next token. That is a GPT — everything else is scale.

### Step 4 — generating text (autoregressive)

```python
logits = self(visible_context)[:, -1]
probabilities = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probabilities, 1)
token_ids = torch.cat([token_ids, next_token], dim=1)
```

`generate` is the Chapter 21 loop again: take the last `CONTEXT_LENGTH` tokens, predict the next token's probabilities, **sample** one (with `temperature` controlling boldness), append it, and feed the longer sequence back in. Training never calls this — training predicts all positions at once — but this is exactly how ChatGPT produces text, one token at a time. `main` samples before *and* after training so you can watch perplexity fall from ~65 to ~5.4 and the output go from random glyphs to blank verse with character names.

The C file `c/sampling_strategies.c` measures greedy / temperature / top-k over 10,000 draws — the "decision step" of Chapter 25's inference engine, and the meaning of the sliders in every LLM playground.

### Quick reference

| Piece | What it does | What to notice |
|-------|--------------|----------------|
| `class CausalSelfAttention` | Chapter 22's attention, batched over multiple heads. | One projection makes Q, K, V for all heads; calls the fused `scaled_dot_product_attention`. |
| `class TransformerBlock` | `x = x + attention(norm(x))`, then `x = x + mlp(norm(x))`. | Residual + norm + attention + MLP: communicate, then compute. |
| `class MiniGPT` | Token + **position** embedding + N blocks + next-token head. | `token_embedding + position_embedding` injects word order. |
| `.generate(ids, count, temperature)` | Autoregressive sampling: predict, sample, append, repeat. | This loop is generation; training predicts all positions at once. |
| `main()` | Trains, samples **before and after**. | Perplexity (`e^loss`) falls from 65 to ~5.4. |

## Run it

```bash
.venv/bin/python chapters/23-gpt-from-scratch/python/train_gpt_shakespeare.py --quick   # ~20 s
.venv/bin/python chapters/23-gpt-from-scratch/python/train_gpt_shakespeare.py           # ~2 min

make -C chapters/23-gpt-from-scratch/c && ./chapters/23-gpt-from-scratch/c/build/sampling_strategies
```

## What the C version covers

The sampling toolbox — temperature softmax, distribution sampling, top-k filtering — measured over 10,000 draws so the strategies' characters are visible in the counts. These exact functions are the "decision step" of Chapter 25's pure-C inference engine.

## Exercises

1. Compute the perplexity of Chapter 21's RNN from its final loss (1.427) and compare architectures at equal training time. (Careful: the GPT saw 3,000 steps of 64×128 characters too — is the comparison fair? What else differs?)
2. Generate 500 characters at temperature 0.2. Diagnose the failure mode in one word, and explain it with the C program's greedy row.
3. Double the context to 256 and retrain. Loss improves slightly; step time grows more than slightly. Explain both with Chapter 22's $O(n^2)$.
4. Add top-k to `MiniGPT.generate` (five lines: `torch.topk`, zero the rest). Sample at temperature 1.2 with and without k=10 and compare the worst lines of each.
5. Challenge: print the attention weights of one head in the last block for a prompt containing a colon (like `ROMEO:`). Chapter 22's lookup experiment predicts you will find heads that lock onto the newline and colon structure — go find one.

## Next

[Chapter 24 — Train your mini-LLM](../24-train-your-mini-llm/README.md): this file, scaled, with checkpoints, BPE, and your GPU's full attention for a night.

<!-- NAV_START -->
---

[← Chapter 22: Attention and transformers](../22-attention-and-transformers/README.md) · [↑ Course index](../../README.md) · [Chapter 24: Train your mini-LLM →](../24-train-your-mini-llm/README.md)

<!-- NAV_END -->
