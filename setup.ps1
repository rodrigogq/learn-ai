# Setup script for Windows (PowerShell).
# Checks for a C compiler, installs uv (Python manager) if missing, creates the
# .venv virtual environment with Python 3.12, and installs the course packages.
# Safe to run multiple times: every step is skipped when already done.
#
# Note: the smoothest Windows experience for the C examples is WSL
# (wsl --install), because the course Makefiles assume a Unix-style toolchain.
# This script supports native Windows too if gcc (MinGW-w64/w64devkit) is on PATH.

$ErrorActionPreference = "Stop"
$repositoryRootDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repositoryRootDirectory

function Print-StatusOk($message)   { Write-Host "[ok] $message" }
function Print-StatusTodo($message) { Write-Host "[..] $message" }
function Print-StatusFail($message) { Write-Host "[!!] $message" }

# ---------------------------------------------------------------- C compiler
$gccCommand = Get-Command gcc -ErrorAction SilentlyContinue
if ($gccCommand) {
    $gccVersionFirstLine = (& gcc --version | Select-Object -First 1)
    Print-StatusOk "C compiler: $gccVersionFirstLine"
} else {
    Print-StatusFail "No gcc found on PATH."
    Print-StatusFail "Recommended: install WSL (run 'wsl --install' as administrator) and follow the Linux setup inside it."
    Print-StatusFail "Native alternative: install w64devkit (https://github.com/skeeto/w64devkit) and add it to PATH, then run this script again."
    Print-StatusFail "Continuing anyway - the Python examples work without a C compiler."
}

# ------------------------------------------------------------------------ uv
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    Print-StatusTodo "Installing uv (official installer from astral.sh)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # The installer adds uv to the user PATH, but this session's PATH predates
    # that change, so we extend it explicitly for the steps below.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Print-StatusOk "uv: $(uv --version)"

# ------------------------------------------------------- Python 3.12 venv
if (Test-Path ".venv\Scripts\python.exe") {
    Print-StatusOk "Python venv: .venv ($(& .venv\Scripts\python.exe --version))"
} else {
    Print-StatusTodo "Creating .venv with Python 3.12 (uv downloads Python if needed)..."
    uv venv --python 3.12 .venv
    Print-StatusOk "Python venv: .venv ($(& .venv\Scripts\python.exe --version))"
}

Print-StatusTodo "Installing course packages from requirements.txt (first run downloads ~2 GB for PyTorch)..."
$env:VIRTUAL_ENV = "$repositoryRootDirectory\.venv"
uv pip install --quiet -r requirements.txt
Print-StatusOk "Packages installed"

# ----------------------------------------------------------------- GPU check
& .venv\Scripts\python.exe -c @"
import torch

if torch.cuda.is_available():
    print(f'[ok] GPU: NVIDIA CUDA available ({torch.cuda.get_device_name(0)})')
else:
    print('[ok] GPU: no CUDA device found - everything still works on CPU (long trainings have --quick mode)')
    print('     If you have an NVIDIA GPU, install a recent driver and see SETUP.md.')
"@

Write-Host ""
Write-Host "Setup complete. Try it: .venv\Scripts\python chapters\00-setup\python\hello_ai.py"
