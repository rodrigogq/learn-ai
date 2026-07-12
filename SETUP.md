# Setup

This page gets you from a fresh computer to running every example in the course. You only do this once. It works on macOS, Linux, and Windows.

## The short version

Open a terminal in this folder and run the setup script for your system:

**macOS or Linux:**

```bash
./setup.sh
```

**Windows (PowerShell):**

```powershell
.\setup.ps1
```

The script checks your machine, installs what is missing, and tells you exactly what it did. It is safe to run more than once. When it finishes, it prints a short report like this:

```
[ok] C compiler: clang (Apple)
[ok] uv: 0.9.x
[ok] Python venv: .venv (Python 3.12)
[ok] Packages: numpy, torch, torchvision, matplotlib
[ok] GPU: Apple Silicon (MPS) available
Setup complete. Try it: .venv/bin/python chapters/00-setup/python/hello_ai.py
```

If you use **VS Code**: open this folder (`File → Open Folder…`), accept the recommended extensions when asked, and run the setup through `Terminal → Run Task… → Setup environment`. VS Code will automatically use the Python environment the script creates.

## What the script installs

| Tool | Why the course needs it |
|------|-------------------------|
| **C compiler** (clang or gcc) | Every chapter has pure-C examples so you can see how things work with nothing hidden. |
| **uv** | A fast Python manager. It downloads Python 3.12 by itself, so it does not matter which Python your system has. |
| **Python 3.12 virtual environment** (the `.venv` folder) | A private Python just for this course, so nothing touches your system Python. |
| **Python packages** (`requirements.txt`) | NumPy (math on arrays), PyTorch (deep learning, from Chapter 10 on), Matplotlib (plots). |

## Manual setup (if you prefer, or if the script fails)

### 1. C compiler

- **macOS:** run `xcode-select --install` and accept the dialog. This installs Apple's clang.
- **Linux (Debian/Ubuntu):** `sudo apt install build-essential`. (Fedora: `sudo dnf group install c-development`.)
- **Windows:** the simplest reliable path is [WSL](https://learn.microsoft.com/windows/wsl/install) (`wsl --install`, then follow the Linux steps inside it). Native alternative: install [w64devkit](https://github.com/skeeto/w64devkit) or MinGW-w64 so that `gcc` and `make` work in your terminal.

### 2. Python environment

Install [uv](https://docs.astral.sh/uv/) (one command, shown on their page), then from this folder:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt        # run with the venv active, or set VIRTUAL_ENV=.venv
```

### 3. Check that everything works

```bash
.venv/bin/python chapters/00-setup/python/hello_ai.py      # Windows: .venv\Scripts\python
make -C chapters/00-setup/c && ./chapters/00-setup/c/build/hello_ai
```

Both should print a greeting and a report about your machine (CPU, GPU, memory). Chapter 0 explains what the numbers mean.

## GPUs (optional but nice)

You do not need a GPU: every example runs on CPU, and long trainings have a `--quick` mode. But if you have one:

- **NVIDIA (Windows/Linux):** install a recent [NVIDIA driver](https://www.nvidia.com/drivers). The PyTorch build installed by this setup uses CUDA automatically when available. If `torch.cuda.is_available()` is `False`, follow the selector at [pytorch.org](https://pytorch.org/get-started/locally/) to install the CUDA build for your system.
- **Apple Silicon (M-series Macs):** nothing to install. PyTorch uses the GPU through MPS automatically.

Chapter 0 shows how to verify which device PyTorch found. The [hardware guide](appendices/E-hardware-guide/README.md) explains what model sizes fit your machine.

## Long training runs

Some chapters (especially [Chapter 24, the mini-LLM](chapters/24-train-your-mini-llm/README.md)) train for hours or days. Every long-running script in this course:

- saves a **checkpoint** regularly (default: every few minutes),
- **resumes** from the last checkpoint when you run it again with `--resume`,
- has a `--quick` flag that runs a tiny version in seconds, so you can test that everything works before committing your GPU for the night.

So you can stop a training at any time (Ctrl+C, reboot, laptop sleep) and continue later without losing work. Each chapter's "Run it" section tells you what to expect.
