#!/usr/bin/env bash
set -Eeuo pipefail

# Set up this project on an NVIDIA CUDA server.
#
# What it does:
#   1. Ensures uv is available.
#   2. Creates/uses .venv and installs project dependencies.
#   3. Removes any CPU-only PyTorch build.
#   4. Installs CUDA-enabled torch/torchvision/torchaudio wheels.
#   5. Verifies that PyTorch was built with CUDA support.
#
# Usage:
#   bash scripts/setup_remote_cuda.sh
#
# Optional environment variables:
#   CUDA=cu126                         # PyTorch CUDA wheel tag: cu121, cu124, cu126, cu128, ...
#   PYTORCH_INDEX_URL=https://...      # Full PyTorch wheel index override
#   PYTHON=python3.12                  # Python executable used by uv
#   SKIP_TESTS=1                       # Skip unittest smoke test
#   NO_SHELL=1                         # Do not start an activated shell after setup

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CUDA_TAG="${CUDA:-cu126}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_TAG}}"
PYTHON_BIN="${PYTHON:-python3.12}"

log() {
  printf '\n\033[1;34m==> %s\033[0m\n' "$*"
}

warn() {
  printf '\n\033[1;33mWARNING: %s\033[0m\n' "$*" >&2
}

log "Checking NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || warn "nvidia-smi exists but failed; continuing so PyTorch can report the real error."
else
  warn "nvidia-smi was not found. Install NVIDIA drivers before training with --device cuda."
fi

log "Ensuring uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

log "Creating virtual environment"
if [[ ! -d .venv ]]; then
  uv venv --python "$PYTHON_BIN" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing project dependencies"
uv sync

log "Checking currently installed PyTorch build"
TORCH_STATUS="$($PROJECT_ROOT/.venv/bin/python - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"missing:{exc}")
else:
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_available = torch.cuda.is_available()
    print(f"version={torch.__version__} cuda_build={cuda_version} cuda_available={cuda_available}")
PY
)"
echo "$TORCH_STATUS"

if [[ "$TORCH_STATUS" == missing:* ]] || [[ "$TORCH_STATUS" == *"cuda_build=None"* ]]; then
  log "Installing CUDA-enabled PyTorch from ${PYTORCH_INDEX_URL}"
  uv pip uninstall -y torch torchvision torchaudio || true
  uv pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
else
  log "Existing PyTorch already has CUDA support; leaving it installed"
fi

log "Verifying CUDA-enabled PyTorch"
python - <<'PY'
import sys
import torch

print(f"torch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"device count: {torch.cuda.device_count()}")
if torch.cuda.device_count():
    print(f"device 0: {torch.cuda.get_device_name(0)}")

if torch.version.cuda is None:
    raise SystemExit("PyTorch is still CPU-only. Try CUDA=cu128 or set PYTORCH_INDEX_URL to the wheel index matching your server.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch has CUDA support, but CUDA is not available. Check NVIDIA drivers/container GPU passthrough.")
PY

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  log "Running smoke tests"
  uv run python -m unittest discover -s tests
fi

log "Setup complete"
echo "Example: uv run python src/rdlm/train_arc.py --arch structured_encoder --device cuda --data-dir /path/to/arc/tasks"

if [[ "${NO_SHELL:-0}" != "1" ]]; then
  log "Starting activated virtualenv shell"
  echo "Run 'exit' to leave the virtualenv shell."
  exec "${SHELL:-/bin/bash}" -i
else
  echo "Activate later with: source .venv/bin/activate"
fi
