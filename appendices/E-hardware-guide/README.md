# Appendix E — Hardware guide

What your computer can and cannot train, and why. Nothing in this course *requires* a GPU — but this page tells you what to expect from what you have.

## The one idea that matters: memory

Training a neural network keeps four things in fast memory at once:

1. the model's **parameters** (the weights),
2. the **gradients** (one number per parameter),
3. the **optimizer state** (Adam keeps two extra numbers per parameter),
4. the **activations** (intermediate results, needed for backpropagation; grows with batch size).

Rule of thumb for standard 32-bit training: budget **16 bytes per parameter** plus activation room. A 50-million-parameter model therefore wants roughly 1 GB before activations — comfortable on almost anything. A 1-billion-parameter model wants ~16 GB before activations — that is where ordinary hardware starts to say no.

## The three kinds of machine

### NVIDIA GPU (CUDA) — e.g. a 16 GB VRAM desktop card

The fastest option per dollar. VRAM (the GPU's own memory) is the hard limit: the model, gradients, optimizer state, and activations must all fit in it.

With 16 GB of VRAM you can comfortably:
- train every model in this course, including the Chapter 24 mini-LLM (10–50M parameters) at good speed;
- fine-tune models in the low billions of parameters using the memory-saving tricks covered in Chapter 24 (mixed precision, gradient accumulation);
- run inference on quantized models far larger than you can train.

### Apple Silicon Mac (MPS) — e.g. 64 GB unified memory

Apple's GPU shares one memory pool with the CPU ("unified memory"), so a 64 GB Mac can *hold* much larger models than a 16 GB GPU — the ceiling is high but the compute is slower than a big NVIDIA card. Expect the same trainings to take roughly 2–5× longer than a fast CUDA card, but almost never to run out of memory in this course.

PyTorch uses the Apple GPU through the **MPS** backend automatically (the course's `common/device.py` picks it up). Two MPS quirks the chapters flag when relevant: a few rare operations fall back to CPU, and float64 is not supported on the GPU.

### CPU only

Everything in Parts 0–II runs fine on CPU. From Part III on, full trainings get slow (hours instead of minutes), which is why **every long script has `--quick`** (tiny run, seconds, verifies correctness) and checkpoint/resume (run overnight, in pieces). The C examples do not care — they are CPU programs by design.

## What each course milestone needs

| Project | CPU only | 16 GB NVIDIA GPU | 64 GB Apple Silicon |
|---------|----------|------------------|---------------------|
| MNIST MLP (ch. 9–10) | minutes | seconds | seconds |
| CIFAR-10 ResNet (ch. 14) | hours | ~10 min | ~30 min |
| Small GPT on tiny corpus (ch. 23) | ~1 hour | minutes | ~15 min |
| Mini-LLM capstone (ch. 24) | days (use --quick) | hours–2 days | 1–4 days |
| DDPM diffusion (ch. 28) | overnight | ~1 hour | few hours |

Numbers are rough orders of magnitude; each chapter gives concrete settings for each machine class.

## Practical tips

- **Watch memory, not utilization.** Out-of-memory is the error you will actually hit. Chapter 24 shows how to size batch and model to your memory.
- **Heat and sleep:** on laptops, plug in and disable sleep for long runs (macOS: `caffeinate -i python train.py ...`).
- **Disk:** datasets in this course are small (MB to a few GB). The Chapter 24 corpus is the largest download at a few GB. Keep ~20 GB free.
- **Cloud is always an option:** any chapter runs on a rented GPU exactly as on your desk; nothing in the course assumes local hardware beyond speed.
