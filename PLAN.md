# Course Build Plan

This file is the construction roadmap for this repository. It exists so that work can stop and resume at any time, on any computer, without losing context. Anyone (human or AI assistant) can continue the work by reading this file alone.

**How to resume work:** read the [Progress](#progress) table, pick the first unchecked chapter, write it following the [Chapter template](#chapter-template) and [Style rules](#style-rules), test every example, check the box, commit, push.

## Ground rules

1. **Audience:** readers with zero AI and zero advanced math knowledge. Only basic Python or C programming is assumed.
2. **Language:** simple English. Short sentences. No idioms, no jargon without definition. Every math symbol is spelled out in words the first time it appears (and collected in `appendices/A-math-notation`).
3. **Code languages:** Python and pure C only. No C++. C is C11, compiled with `make` (`-Wall -Wextra -O2`, link only `-lm`). Python uses the repo venv (`.venv`), NumPy for "from scratch" chapters, PyTorch from Chapter 10 on.
4. **No personal information** anywhere in the repository content.
5. **Every example must run.** Python examples finish in seconds by default; anything long takes a `--quick` flag for a smoke test and supports checkpoint/resume for the full run. C examples compile warning-clean and print the output shown in the chapter.
6. **Device support:** all PyTorch code selects its device through `common/device.py` (CUDA → MPS → CPU).

## Coding style (mandatory, both languages)

- Long, descriptive variable and function names. Prefer `hidden_layer_weight_matrix` over `W1`, `compute_gradient_of_loss_with_respect_to_weights()` over `grad_w()`. Long lines are fine. When code implements a formula, the chapter text maps each name to its math symbol.
- Every function starts with documentation: what it does and what each argument means (Python docstring; C block comment above the function).
- Comments explain **why** a decision was made or why the flow proceeds as it does — never what an obvious line does. No filler comments.

## Chapter template

Each chapter lives in `chapters/NN-slug/` with this layout:

```
NN-slug/
├── README.md    # the chapter
├── figures/     # SVG diagrams embedded by the README
├── python/      # runnable examples (may have several .py files)
└── c/           # .c files + Makefile
```

`README.md` sections, in order:

1. Title + one-paragraph promise ("In this chapter you will…").
2. **What you will learn** — bullet list.
3. **Prerequisites** — links to earlier chapters/appendices.
4. Body — theory built up in small steps. Every formula introduced symbol by symbol. Diagrams are **good-quality SVG files** in the chapter's `figures/` folder, embedded with markdown image syntax — never ASCII art (project decision). Give every SVG an explicit background color so it reads well on both GitHub light and dark themes. Monospaced text blocks are only for code, file trees, and terminal output. Small numeric examples worked by hand before code.

   Two hard pedagogy rules (project decision, from reader feedback):
   - **Explain before code.** Before any example is shown or run, the chapter explains: what the algorithm is, *why this algorithm was chosen* over obvious alternatives, and what the code does step by step — with a small numeric case worked by hand first.
   - **Never use a concept before it is taught.** If a later concept must be mentioned, give a one-line plain-language description plus an explicit forward reference ("you will meet this properly in Chapter N"). Check every draft against this before marking it done.
5. **Run it** — exact commands for the Python and C examples, plus the expected output.
6. **What the C version covers** — one honest paragraph on how the C example relates to the Python one (full port vs. simplified core).
7. **Exercises** — 3–5, from easy to challenging, with hints.
8. **Next** — link to the next chapter.

## Progress

Status: `[ ]` not started · `[~]` drafted, examples untested · `[x]` done (text + all examples tested).

### Root and infrastructure

- [x] Directory skeleton (all chapters + appendices)
- [x] `README.md` (course index)
- [x] `PLAN.md` (this file)
- [x] `LICENSE` (MIT)
- [x] `.gitignore`
- [x] `requirements.txt`
- [x] `SETUP.md` + `setup.sh` (macOS/Linux) + `setup.ps1` (Windows)
- [x] `.vscode/` (settings, recommended extensions, build/setup tasks)
- [x] `common/device.py`, `common/data.py`
- [x] Appendix A (math notation) — keep growing with each chapter
- [x] Appendix B (Python refresher)
- [x] Appendix C (C refresher)
- [x] Appendix D (glossary) — keep growing with each chapter
- [x] Appendix E (hardware guide)

### Part 0 — Getting ready

- [x] 00-setup — install tools, verify GPU, hello world in Python and C.

### Part I — Foundations

- [x] 01-what-is-ai — AI/ML/DL map; rule-based vs learned classifier in both languages.
- [x] 02-vectors-and-matrices — vectors, dot product, matmul, shapes; matrix ops from scratch in both languages.
- [x] 03-derivatives-and-gradients — slope → derivative → gradient; numerical gradient checker in both languages.
- [x] 04-probability-basics — distributions, expectation, likelihood, cross-entropy; sampling simulations.
- [x] 05-linear-regression — loss, gradient descent, learning rate; full from-scratch fit in both languages.
- [x] 06-logistic-regression — sigmoid, cross-entropy, decision boundary.
- [x] 07-perceptron-and-neurons — activation functions, XOR limitation.
- [x] 08-backpropagation — chain rule; micrograd-style autograd engine in Python and C.
- [x] 09-first-neural-network — MLP on MNIST from scratch (NumPy + full C port). Milestone.

### Part II — The deep learning toolkit

- [x] 10-intro-to-pytorch — tensors, autograd, nn.Module, GPU; Chapter 9 redone in 30 lines. C: minimal tensor library.
- [x] 11-training-deep-networks — SGD→Adam, init, batchnorm, dropout, regularization. C: Adam from scratch.
- [ ] 12-data-pipelines — Dataset/DataLoader, augmentation, splits, metrics. C: binary dataset reader.

### Part III — Computer vision

- [ ] 13-convolutions — conv2d from scratch, padding/stride. C: conv kernel + benchmark vs Python.
- [ ] 14-image-classification — ResNet on CIFAR-10; training curves, confusion matrix. C: inference from exported weights.
- [ ] 15-object-detection — IoU, anchors, NMS; simple single-stage detector. C: IoU + NMS.
- [ ] 16-segmentation — U-Net on a small dataset. C: mask post-processing.
- [ ] 17-video-understanding — frame models / 3D convs for action recognition. C: temporal pooling.

### Part IV — Audio

- [ ] 18-sound-and-spectrograms — sampling, FFT from zero, mel spectrograms, audio classifier. C: WAV reader + FFT.
- [ ] 19-speech-recognition — keyword spotting; CTC intuition. C: greedy CTC decoder.

### Part V — Sequences and language

- [ ] 20-text-and-tokenization — BPE tokenizer from scratch in both languages (C version reused in ch. 25).
- [ ] 21-recurrent-networks — char-level RNN/LSTM language model. C: RNN forward pass.
- [ ] 22-attention-and-transformers — attention with tiny matrices; transformer block. C: one attention head.
- [ ] 23-gpt-from-scratch — nanoGPT-style model trained on a small text corpus in minutes.
- [ ] 24-train-your-mini-llm — **capstone**: data → tokenizer → multi-day training with checkpoint/resume → sampling. Sized for a 16 GB VRAM GPU or a 64 GB unified-memory Mac.
- [ ] 25-llm-inference-in-c — export ch. 24 weights; single-file pure-C inference engine; int8 quantization.

### Part VI — Generative AI

- [ ] 26-autoencoders-and-vaes — latent spaces; VAE on MNIST. C: decoder inference.
- [ ] 27-gans — DCGAN on small images. C: generator inference.
- [ ] 28-diffusion-models — DDPM from scratch; classifier-free guidance. C: sampling loop for a tiny exported model.
- [ ] 29-text-to-image-and-video — text conditioning, latent diffusion; toy video generation. C: tiny sampler.

### Part VII — Beyond prediction

- [ ] 30-reinforcement-learning — Q-learning on a gridworld (both languages), DQN on CartPole, policy gradients.
- [ ] 31-deployment — ONNX/TorchScript export, quantization, local serving, embedding models in C programs; course wrap-up.

## Per-session workflow

1. `git pull`, read this file, pick the first unchecked item.
2. Write the chapter README following the template.
3. Write and test the Python examples: `.venv/bin/python chapters/NN-slug/python/example.py`.
4. Write and test the C examples: `make -C chapters/NN-slug/c && ./chapters/NN-slug/c/<binary>`.
5. Add new symbols to Appendix A and new terms to Appendix D.
6. Update the checkbox here, commit (`Add chapter NN: <title>`), push.

## Environment (what setup scripts must guarantee)

- Python 3.12 venv at `.venv` created with `uv` (or plain `python3 -m venv` if 3.12+ exists), packages from `requirements.txt`.
- C compiler: clang (macOS, via Xcode Command Line Tools), gcc (Linux, via build-essential), MSVC or MinGW-w64 (Windows — or WSL, which is the recommended path).
- VS Code opens the repo root; `.vscode/settings.json` points at `.venv`; tasks exist for "Setup environment" and "Build current chapter C examples".

## Optional future work (not planned unless requested)

- CI that builds every C example and smoke-tests every Python example.
- Jupyter notebook mirrors of chapters.
- Translations.
