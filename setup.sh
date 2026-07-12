#!/usr/bin/env bash
# Setup script for macOS and Linux.
# Checks for a C compiler, installs uv (Python manager) if missing, creates the
# .venv virtual environment with Python 3.12, and installs the course packages.
# Safe to run multiple times: every step is skipped when already done.
set -euo pipefail

repository_root_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repository_root_directory"

print_status_ok()   { printf '[ok] %s\n' "$1"; }
print_status_todo() { printf '[..] %s\n' "$1"; }
print_status_fail() { printf '[!!] %s\n' "$1"; }

# ---------------------------------------------------------------- C compiler
# The course needs any C11 compiler. We only check; installing one requires
# admin rights or an OS dialog, so we tell the user the exact command instead
# of trying to run it for them.
if command -v cc >/dev/null 2>&1; then
    c_compiler_description="$(cc --version 2>/dev/null | head -1)"
    print_status_ok "C compiler: ${c_compiler_description}"
else
    operating_system_name="$(uname -s)"
    if [ "$operating_system_name" = "Darwin" ]; then
        print_status_fail "No C compiler found. Run: xcode-select --install  (then run this script again)"
    else
        print_status_fail "No C compiler found. Run: sudo apt install build-essential  (or your distribution's equivalent), then run this script again."
    fi
    exit 1
fi

# ------------------------------------------------------------------------ uv
# uv downloads and manages Python versions by itself, which is why the course
# uses it: it works the same on every machine regardless of the system Python.
if command -v uv >/dev/null 2>&1; then
    print_status_ok "uv: $(uv --version | awk '{print $2}')"
else
    print_status_todo "Installing uv (official installer from astral.sh)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv in ~/.local/bin, which may not be on PATH yet in
    # this same shell session, so we add it explicitly for the steps below.
    export PATH="$HOME/.local/bin:$PATH"
    print_status_ok "uv: $(uv --version | awk '{print $2}')"
fi

# ------------------------------------------------------- Python 3.12 venv
if [ -x ".venv/bin/python" ]; then
    print_status_ok "Python venv: .venv ($(.venv/bin/python --version))"
else
    print_status_todo "Creating .venv with Python 3.12 (uv downloads Python if needed)..."
    uv venv --python 3.12 .venv
    print_status_ok "Python venv: .venv ($(.venv/bin/python --version))"
fi

print_status_todo "Installing course packages from requirements.txt (first run downloads ~2 GB for PyTorch)..."
VIRTUAL_ENV="$repository_root_directory/.venv" uv pip install --quiet -r requirements.txt
print_status_ok "Packages: $(VIRTUAL_ENV="$repository_root_directory/.venv" uv pip list 2>/dev/null | grep -cE '^[a-zA-Z]') installed"

# ----------------------------------------------------------------- GPU check
.venv/bin/python - <<'PYTHON_GPU_CHECK'
import torch

if torch.cuda.is_available():
    print(f"[ok] GPU: NVIDIA CUDA available ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    print("[ok] GPU: Apple Silicon (MPS) available")
else:
    print("[ok] GPU: none found - everything still works on CPU (long trainings have --quick mode)")
PYTHON_GPU_CHECK

echo
echo "Setup complete. Try it: .venv/bin/python chapters/00-setup/python/hello_ai.py"
