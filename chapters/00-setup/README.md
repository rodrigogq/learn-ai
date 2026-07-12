# Chapter 0 — Setup

In this chapter you will get your computer ready for the whole course and run your first two programs — one in Python, one in C. At the end you will know exactly what tools you have, whether you have a usable GPU, and how to run any example in any chapter.

## What you will learn

- What each tool in this course is for (Python, NumPy, PyTorch, a C compiler, `make`).
- How to install everything with one script.
- How to check whether PyTorch can see your GPU.
- How every chapter's examples are organized and how to run them.

## Prerequisites

None. This is the beginning.

## 1. Install the tools

Follow [SETUP.md](../../SETUP.md) — the short version is one command from the repository root:

```bash
./setup.sh        # macOS / Linux
.\setup.ps1       # Windows PowerShell
```

While it runs, here is what each tool is and why the course needs it:

| Tool | What it is | Why we need it |
|------|-----------|----------------|
| **Python** | The most popular language for AI. Slow by itself, but it glues together fast libraries. | All the "real" AI work in this course happens in Python. |
| **NumPy** | A Python library for fast math on arrays of numbers. | Chapters 1–9 build everything "from scratch" — NumPy is our calculator. |
| **PyTorch** | The leading deep learning library. Computes gradients automatically and runs on GPUs. | From Chapter 10 on, our models get too big to hand-build. |
| **C compiler** | Turns C source code into machine code. | Every chapter has a pure-C version so you can see the machinery with nothing hidden. |
| **make** | A small tool that runs the compiler with the right options. | So building any C example is just `make`. |
| **uv** | A fast Python installer/manager. | It downloads its own Python 3.12, so your system Python does not matter. |

The script creates a **virtual environment** in the `.venv` folder — a private copy of Python just for this course. Nothing it installs touches the rest of your system. Delete the folder and it is gone.

## 2. Run your first Python program

From the repository root:

```bash
.venv/bin/python chapters/00-setup/python/hello_ai.py
```

(Windows: `.venv\Scripts\python chapters\00-setup\python\hello_ai.py`)

You should see a report like this (numbers will differ):

```
Hello from Python 3.12!
NumPy version:    2.5.1
PyTorch version:  2.13.0

Device check:
  CUDA (NVIDIA GPU):     not available
  MPS (Apple Silicon):   AVAILABLE  <- PyTorch will use this
  CPU:                   always available

Small speed test: multiplying two 2048x2048 matrices...
  cpu: 0.019 seconds
  mps: 0.005 seconds

Your machine is ready for the course.
```

The **device check** matters for later chapters: it tells you which processor PyTorch will train on. Any result is fine — a GPU just makes the big chapters faster. The [hardware guide](../../appendices/E-hardware-guide/README.md) explains what your machine can handle.

## 3. Run your first C program

The C program computes one **weighted sum**, so let us first say plainly what that is — it is the only piece of math in this chapter, and you already know it from school.

A weighted sum combines several numbers into one, giving each number an importance factor (a *weight*). You have used one if a teacher ever graded you like this:

```
final grade = 0.7 * exam + 0.3 * homework
```

The exam "weighs" more than the homework. That is the whole idea: multiply each value by its weight, add the results. Sometimes a fixed extra is added at the end (a bonus point in the grade analogy); we call that the *offset*.

The program computes this weighted sum, by hand, in ten lines:

```
0.5 * 0.8  +  0.3 * (-0.2)  +  0.1  =  0.44
```

Two values (0.5 and 0.3), two weights (0.8 and −0.2 — a *negative* weight means "this value counts against the result"), and an offset of 0.1. Nothing more.

Why show this trivial computation at all? Because — surprisingly — **this is the single operation AI is built from.** Almost everything a modern AI model does, from recognizing a cat to writing a sentence, is enormous numbers of weighted sums. What makes AI interesting is *where the weights come from*: nobody types them in — they are learned from data. Chapter 1 shows the smallest possible example of "learned from data"; in Chapter 7 you will see weighted sums assembled into the units neural networks are made of.

Every chapter's `c/` folder contains a `Makefile`. From the repository root:

```bash
make -C chapters/00-setup/c
./chapters/00-setup/c/build/hello_ai
```

`make -C <folder>` means "run make inside that folder". The compiled programs always go into a `build/` subfolder. Expected output:

```
Hello from C (standard C11)!

This machine uses 8 bytes for a 'double' (the number type used everywhere in this course).

A weighted sum, computed by hand:
  values:  [0.50, 0.30]
  weights: [0.80, -0.20]
  offset:  0.10
  result = 0.5*0.8 + 0.3*(-0.2) + 0.1 = 0.440

Your C toolchain is ready for the course.
```

## 4. How every chapter works

```
chapters/NN-name/
├── README.md    <- the chapter text (you are reading one now)
├── python/      <- runnable Python examples
└── c/           <- the same ideas in pure C, with a Makefile
```

- Read the README first; it tells you when to run each example.
- Python examples: `.venv/bin/python chapters/NN-name/python/<file>.py`
- C examples: `make -C chapters/NN-name/c` then run from `c/build/`.
- **Long trainings** (from Part III on): every slow script has a `--quick` flag that runs a tiny version in seconds, and saves checkpoints so you can stop with Ctrl+C and continue later with `--resume`. Chapter 24 relies on this heavily.

If you use VS Code: open the repository root folder, accept the recommended extensions, and use `Terminal → Run Task…` for "Setup environment" and "Build C examples of current chapter".

## Run it

Already covered above — the two commands to remember:

```bash
.venv/bin/python chapters/00-setup/python/hello_ai.py
make -C chapters/00-setup/c && ./chapters/00-setup/c/build/hello_ai
```

## What the C version covers

A full equivalent of the Python check, minus the GPU part: C programs in this course always run on the CPU (that is the point — you see every instruction). It also computes your first weighted sum.

## Exercises

1. Open `python/hello_ai.py` and change the matrix size from 2048 to 4096. How much longer does the multiplication take? (It should be close to 8× — the work grows with the cube of the size. Chapter 2 explains why.)
2. In `c/hello_ai.c`, change the weights of the weighted sum so the result becomes negative. Recompile with `make -C chapters/00-setup/c` and check. First predict the result on paper, then compare.
3. Run the Python script with the venv's Python and with your system Python (`python3 chapters/00-setup/python/hello_ai.py`). The system one probably fails to import PyTorch — now you know why virtual environments exist.

## Next

[Chapter 1 — What is AI?](../01-what-is-ai/README.md)
