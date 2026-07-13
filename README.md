# Learn AI from Zero

A complete, hands-on course for building your own artificial intelligence systems — from your first line of math to training your own small language model (LLM) on your own computer.

**You need zero prior knowledge of AI or math beyond school level.** The only requirement is basic programming in Python or C. Every math concept is explained from scratch, in simple English, with every symbol spelled out.

Every chapter has:

- A `README.md` — the chapter text: theory, worked examples, and exercises.
- A `python/` folder — runnable Python examples.
- A `c/` folder — the same ideas in pure C, so you can see exactly what happens under the hood, with no libraries hiding the details.

## Contents

- [How to start](#how-to-start)
- [The course](#the-course)
  - [Part 0 — Getting ready](#part-0--getting-ready)
  - [Part I — Foundations](#part-i--foundations-zero-prior-knowledge)
  - [Part II — The deep learning toolkit](#part-ii--the-deep-learning-toolkit)
  - [Part III — Computer vision](#part-iii--computer-vision)
  - [Part IV — Audio](#part-iv--audio)
  - [Part V — Sequences and language](#part-v--sequences-and-language-the-road-to-your-own-llm)
  - [Part VI — Generative AI](#part-vi--generative-ai)
  - [Part VII — Beyond prediction](#part-vii--beyond-prediction)
  - [Appendices](#appendices)
- [Hardware](#hardware)
- [License](#license)

## How to start

1. Open this folder in [VS Code](https://code.visualstudio.com/) (or any editor).
2. Follow [SETUP.md](SETUP.md) to install the tools (one script does it for you).
3. Start with [Chapter 0](chapters/00-setup/README.md) and go in order. Each chapter builds on the previous one.

Long training runs (like the mini-LLM in Chapter 24) support **checkpoint and resume**: you can stop them at any time and continue later, even after a reboot.

## The course

### Part 0 — Getting ready

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 00 | [Setup](chapters/00-setup/README.md) | Install Python, a C compiler, and PyTorch; check your GPU; run your first programs. |

### Part I — Foundations (zero prior knowledge)

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 01 | [What is AI?](chapters/01-what-is-ai/README.md) | The map of AI, machine learning, and deep learning; what "training" really means. |
| 02 | [Vectors and matrices](chapters/02-vectors-and-matrices/README.md) | The linear algebra every AI uses, from zero: vectors, dot products, matrix multiplication. |
| 03 | [Derivatives and gradients](chapters/03-derivatives-and-gradients/README.md) | The calculus of learning: slopes, derivatives, and gradients, explained visually. |
| 04 | [Probability basics](chapters/04-probability-basics/README.md) | Chance, distributions, expectation, and why "cross-entropy" measures surprise. |
| 05 | [Linear regression](chapters/05-linear-regression/README.md) | Your first learning algorithm: fit a line to data using gradient descent. |
| 06 | [Logistic regression](chapters/06-logistic-regression/README.md) | Your first classifier: sigmoid, cross-entropy loss, and decision boundaries. |
| 07 | [Perceptrons and neurons](chapters/07-perceptron-and-neurons/README.md) | The artificial neuron, activation functions, and why one neuron is not enough (XOR). |
| 08 | [Backpropagation](chapters/08-backpropagation/README.md) | How networks learn: the chain rule, step by step, and a tiny autograd engine built from scratch. |
| 09 | [Your first neural network](chapters/09-first-neural-network/README.md) | Train a full network on handwritten digits (MNIST) from scratch — no frameworks. |

### Part II — The deep learning toolkit

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 10 | [Introduction to PyTorch](chapters/10-intro-to-pytorch/README.md) | Tensors, automatic gradients, and the GPU; redo Chapter 9 in 30 lines. |
| 11 | [Training deep networks](chapters/11-training-deep-networks/README.md) | Optimizers (SGD to Adam), initialization, batch norm, dropout, and fighting overfitting. |
| 12 | [Data pipelines](chapters/12-data-pipelines/README.md) | Datasets, loaders, augmentation, train/validation/test splits, and honest metrics. |

### Part III — Computer vision

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 13 | [Convolutions](chapters/13-convolutions/README.md) | The sliding-window operation behind all image AI, built from scratch. |
| 14 | [Image classification](chapters/14-image-classification/README.md) | Build and train a ResNet on CIFAR-10; read training curves like a pro. |
| 15 | [Object detection](chapters/15-object-detection/README.md) | Bounding boxes: IoU, anchors, non-maximum suppression, and a simple YOLO-style detector. |
| 16 | [Segmentation](chapters/16-segmentation/README.md) | Label every pixel: build a U-Net from scratch. |
| 17 | [Video understanding](chapters/17-video-understanding/README.md) | From single images to sequences of frames: action recognition. |

### Part IV — Audio

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 18 | [Sound and spectrograms](chapters/18-sound-and-spectrograms/README.md) | Waveforms, sampling, the FFT from zero, and classifying sounds. |
| 19 | [Speech recognition](chapters/19-speech-recognition/README.md) | Keyword spotting and how full speech-to-text works (CTC). |

### Part V — Sequences and language (the road to your own LLM)

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 20 | [Text and tokenization](chapters/20-text-and-tokenization/README.md) | How text becomes numbers: build a BPE tokenizer from scratch. |
| 21 | [Recurrent networks](chapters/21-recurrent-networks/README.md) | RNNs and LSTMs: a character-level language model that writes text. |
| 22 | [Attention and transformers](chapters/22-attention-and-transformers/README.md) | The attention mechanism, step by step with tiny matrices; the full transformer block. |
| 23 | [GPT from scratch](chapters/23-gpt-from-scratch/README.md) | Build a small GPT and train it in minutes. |
| 24 | [Train your mini-LLM](chapters/24-train-your-mini-llm/README.md) | **Capstone**: the full pipeline — data, tokenizer, multi-day training with checkpoints, and sampling. |
| 25 | [LLM inference in C](chapters/25-llm-inference-in-c/README.md) | Run your trained LLM in pure C — one file, no dependencies — plus int8 quantization. |

### Part VI — Generative AI

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 26 | [Autoencoders and VAEs](chapters/26-autoencoders-and-vaes/README.md) | Compression, latent spaces, and your first generative model. |
| 27 | [GANs](chapters/27-gans/README.md) | Two networks in competition: generate images with a DCGAN. |
| 28 | [Diffusion models](chapters/28-diffusion-models/README.md) | How modern image generators work: noise, denoising, and guidance, from scratch. |
| 29 | [Text-to-image and video](chapters/29-text-to-image-and-video/README.md) | Conditioning generation on text; latent diffusion; toy video generation. |

### Part VII — Beyond prediction

| # | Chapter | What you will learn |
|---|---------|---------------------|
| 30 | [Reinforcement learning](chapters/30-reinforcement-learning/README.md) | Agents, rewards, Q-learning, and deep RL on classic control problems. |
| 31 | [Deployment](chapters/31-deployment/README.md) | Export, quantize, and serve your models; embed them in C programs; where to go next. |

### Appendices

| | Appendix | What it covers |
|---|----------|----------------|
| A | [Math notation](appendices/A-math-notation/README.md) | Every math symbol used in this course, explained in plain English. |
| B | [Python refresher](appendices/B-python-refresher/README.md) | The Python features the course relies on. |
| C | [C refresher](appendices/C-c-refresher/README.md) | The C features the course relies on. |
| D | [Glossary](appendices/D-glossary/README.md) | Every AI term used in this course, in one place. |
| E | [Hardware guide](appendices/E-hardware-guide/README.md) | GPUs, VRAM, Apple Silicon, and what your machine can train. |

## Hardware

Everything runs on an ordinary computer. A GPU (NVIDIA with CUDA, or Apple Silicon) makes the later chapters much faster, but every example has a CPU fallback and a `--quick` mode that finishes in seconds. Reference machines used to size the projects: a desktop GPU with 16 GB of VRAM and a Mac with 64 GB of unified memory. See the [hardware guide](appendices/E-hardware-guide/README.md).

## License

[MIT](LICENSE). Use anything here for anything, including your own courses and projects.
