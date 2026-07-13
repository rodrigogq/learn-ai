# Chapter 25 — LLM inference in C

The course's biggest payoff, and the reason this book insisted on C the whole way. The language model you trained in Chapter 24 — a real transformer — now **runs in a single pure-C file**: no PyTorch, no framework, nothing but `libm` and about 400 lines of loops. It takes a text prompt and writes English. Along the way you add **int8 quantization**, the trick that lets `llama.cpp` run billion-parameter models on laptops. When you finish this chapter, the last piece of mystery is gone: an LLM is a file of numbers and arithmetic you have personally written.

## What you will learn

- Why training and inference are different problems with different tools.
- Exporting a PyTorch model to a flat, self-describing binary.
- The complete transformer forward pass in C — every line a chapter you finished.
- int8 quantization: 4× smaller, still coherent.
- Verifying a from-scratch engine against the framework.

## Prerequisites

- [Chapter 24](../24-train-your-mini-llm/README.md) — the trained model (and its `model.py`).
- [Chapter 22](../22-attention-and-transformers/README.md) — attention (you will re-implement it).
- [Chapter 20](../20-text-and-tokenization/README.md) — BPE (built into the engine).

## 1. Training and inference are different jobs

![PyTorch trains; one C file runs](figures/inference-pipeline.svg)

Training needs autograd, optimizer state, huge batches, and a GPU — Chapter 24's world, gigabytes of framework. **Inference needs none of it**: no gradients, no optimizer, one sequence at a time. All that survives is the weights and the forward pass. That asymmetry is why production LLMs are *served* by lean, specialized engines utterly unlike the PyTorch that trained them — and why this chapter can fit a working one in a file you read in an afternoon.

## 2. Export: from checkpoint to a flat file

`export_llm_for_c.py` reads Chapter 24's `.pt` checkpoint and writes a self-describing `.bin`: a small header (magic number + the six dimensions the engine needs to size every buffer), the tokenizer's merges, then every weight tensor in a fixed, documented order. The C engine walks that exact order — the two files are a contract. Small, sensitive tensors (layer-norm parameters, biases, embeddings) stay float32; the big matrices can be quantized (Section 4).

## 3. The forward pass in C

`llm_inference.c` is the whole model, and every piece is something you built:

- **`matvec`** — matrix times vector plus bias, dequantizing int8 as it goes. This is Chapter 2, and it is where ~99% of the runtime lives (an LLM is overwhelmingly matrix multiplies).
- **multi-head causal attention** — Chapter 22's Q·K, scale, causal mask, softmax, blend, done per position per head.
- **layer norm** (Chapter 11), **GELU** (Chapter 23), **residual adds** (Chapter 14), **tied output head** (Chapter 24's embedding, reused as the projection).
- the **BPE tokenizer** (Chapter 20) so it accepts real text, and **temperature + top-k sampling** (Chapter 23) so it generates.

Running it on the Chapter 24 `small` model (float32 export), pure C:

```
$ ./llm_inference checkpoints/mini_llm_small.bin "It was a dark night, and " 60
Loaded: float32, 6 blocks, 384 wide, vocab 4352, context 256

It was a dark night, and identified as we saw the old Redruth of the Regent's
Jove. I found that the Renfield's reality of the affair. He was terribly the
old sly, very first, and I was there I had one wary. ...
```

Not deathless prose — a 12 M-parameter model trained for 17 minutes — but unmistakably English, with characters straight out of the training corpus (*Treasure Island*'s Redruth, *Dracula*'s Renfield), generated with **zero PyTorch**. The engine omits the *KV cache* (real engines save each step's keys/values to avoid recomputing attention over the whole context every token) to stay readable; the exercises add it, and it is the single biggest inference speedup there is.

## 4. Quantization: 4× smaller, still fluent

The float32 export is 49.7 MB. **int8 quantization** stores each weight-matrix entry as a single signed byte plus one shared float32 scale: to read a weight, multiply the byte by the scale (`matvec` does this inline). The `--quantize` export drops the file to **17.9 MB** — under 2.8× here (only the big matrices quantize; norms and embeddings stay float32) and toward the full 4× on larger models — and the text stays coherent:

```
$ ./llm_inference checkpoints/mini_llm_small_int8.bin "The night was "
The night was errory, we might be. At times I could hear the palm of a campstern
bolt of ...
```

Squeezing 32 bits into 8 loses precision, but neural networks are famously tolerant of it — a lesson that scales all the way up: this exact idea (with per-row scales and 4-bit variants) is how people run 7-billion-parameter models on a MacBook. You just implemented its heart.

## 5. Trust, but verify

A from-scratch engine that *looks* right can be subtly wrong. Generation is random, so matching text proves nothing; the honest check is deterministic — do the two implementations compute the same **logits**? `verify_against_c.py` prints PyTorch's top-5 next tokens for a fixed prompt:

```
PyTorch, prompt 'It was a dark night, and ' - the 5 most likely next tokens:
   logit  +8.817   token   46  '.'
   logit  +8.650   token   90  'Z'
   logit  +8.471   token 2451  'edd'
   ...
```

Run the C engine on the same prompt and it samples from exactly this ranking. The float values differ in the last digits (operation order differs between PyTorch's kernels and the C loops); the **ranking matches**, which is the real correctness criterion. That agreement is the proof that your C forward pass and PyTorch's are the same computation.

## Code walkthrough

This chapter's real code is the C file; the Python is glue. Here is the map of both.

**`python/export_llm_for_c.py`** — turns the checkpoint into a flat `.bin`:
- `write_tensor(file, tensor, quantize)` — writes a tensor as float32, or (for 2-D weight matrices, if `--quantize`) as **int8 + a scale**. Small/sensitive tensors stay float32.
- `main()` — writes the header (magic + dimensions), the tokenizer merges, then every weight in the exact order the C engine reads them.

**`c/llm_inference.c`** — the whole model, ~400 lines. The functions worth finding:
- `load_model` / `read_matrix` — walk the `.bin` in the documented order, handling float32 or int8.
- `matvec(weight, bias, input, output)` — matrix×vector plus bias, **dequantizing int8 on the fly**. ~99% of the runtime lives here; it is Chapter 2, finally running an LLM.
- `forward(model, tokens, length, logits_out)` — the full pass: embeddings, then per block the multi-head causal attention (Chapter 22, by hand) and the MLP, with `layer_norm`, `gelu`, and residual adds.
- `encode_prompt` / `print_token` — the BPE tokenizer (Chapter 20) built in.
- `sample(logits, ..., temperature, top_k)` — Chapter 23's sampling.

**`python/verify_against_c.py`** prints PyTorch's top-5 next tokens so you can confirm the C engine's ranking matches — the honest correctness check (generation is random, so text can't be the test).

## Run it

```bash
# Needs a Chapter 24 checkpoint (any size). Then:
.venv/bin/python chapters/25-llm-inference-in-c/python/export_llm_for_c.py --size small
.venv/bin/python chapters/25-llm-inference-in-c/python/export_llm_for_c.py --size small --quantize

make -C chapters/25-llm-inference-in-c/c
./chapters/25-llm-inference-in-c/c/build/llm_inference checkpoints/mini_llm_small.bin "The night was "
./chapters/25-llm-inference-in-c/c/build/llm_inference checkpoints/mini_llm_small_int8.bin "The night was "

.venv/bin/python chapters/25-llm-inference-in-c/python/verify_against_c.py --size small
```

## What the C version covers

Everything — this chapter *is* the C chapter. One file loads the model (float32 or int8), tokenizes the prompt, runs the full forward pass, and samples, in ~400 documented lines. It is the culmination of every C example before it: the tensor library (ch. 10), the conv/matvec loops (chs. 2, 13), the attention head (ch. 22), the BPE encoder (ch. 20), and the samplers (ch. 23), assembled into a working language model. Read it once and no LLM will ever be a black box to you again.

## Exercises

1. Time the engine (`time ./llm_inference ...`) on float32 vs int8. Is int8 faster, slower, or the same? Explain via memory bandwidth vs compute (int8 reads a quarter of the bytes but does a multiply per weight either way).
2. Add a `--temperature` and `--top-k` command-line argument. Regenerate at temperature 0.1 and 1.4 and connect the outputs to Chapter 23's sampling table.
3. **KV cache** (the big one): the engine recomputes attention over the whole context every token — O(n²) per token, O(n³) for a passage. Cache each position's keys and values so each new token only attends, never recomputes. Measure the speedup on a 200-token generation.
4. The tokenizer's `token_byte_strings` table caps token length at 64 bytes. Find where, and reason about when a BPE token could exceed it (hint: it cannot, for this vocabulary — why?).
5. Challenge: implement 4-bit quantization (two weights per byte, per-row scales). Compare file size and output quality against int8. You are now within sight of how `llama.cpp` actually works.

## Next

Part V complete — you trained a language model and run it in pure C. [Chapter 26 — Autoencoders and VAEs](../26-autoencoders-and-vaes/README.md) opens Part VI: teaching machines not to classify, but to *create*.
