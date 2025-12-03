#!/usr/bin/env bash
set -euo pipefail

# auto-detect CUDA and install matching torch wheel + backend requirements
# Usage: ./install_torch_auto.sh
# Run this from the repo root (so ./backend/requirements.txt exists)

BACKEND_DIR="./backend"
VENV_DIR="$BACKEND_DIR/venv"

echo "== VTON: Auto Torch + Backend installer (bash) =="

# Check nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi found. Querying CUDA version..."
  # Try to query the CUDA version reported by nvidia-smi
  CUDA_VERSION_RAW=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null || true)
  if [ -z "$CUDA_VERSION_RAW" ]; then
    # fallback to parsing generic output
    CUDA_VERSION_RAW=$(nvidia-smi | grep -i "cuda version" | awk -F'CUDA Version:' '{print $2}' | awk '{print $1}' || true)
  fi
  CUDA_VERSION=$(echo "$CUDA_VERSION_RAW" | tr -d '[:space:]')
  if [ -n "$CUDA_VERSION" ]; then
    echo "[INFO] Detected CUDA version: $CUDA_VERSION"
  else
    echo "[WARN] Could not detect CUDA version automatically."
  fi
else
  echo "[WARN] nvidia-smi not found. Cannot auto-detect CUDA. You will be prompted."
  CUDA_VERSION=""
fi

# If detection failed, ask the user
if [ -z "${CUDA_VERSION:-}" ]; then
  read -p "Enter CUDA major.minor version (example: 12.1 or 11.8). Or type 'cpu' to install CPU-only torch: " CUDA_VERSION
fi

# Decide which wheel to use
CASE_PREFIX=""
if [ "$CUDA_VERSION" = "cpu" ] || [ "$CUDA_VERSION" = "none" ]; then
  echo "[INFO] Installing CPU-only torch (no CUDA)."
  CASE_PREFIX="cpu"
else
  # Normalize to numeric major.minor
  # Examples: 12.2 -> 12.2, 12.1 -> 12.1, 11.8 -> 11.8
  # choose wheel: use cu121 for >=12.1 (12.x), cu118 for 11.8, cu117 for 11.7
  if [[ "$CUDA_VERSION" == 12.* ]] || [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12.2" ]]; then
    CASE_PREFIX="cu121"
  elif [[ "$CUDA_VERSION" == 11.8* ]]; then
    CASE_PREFIX="cu118"
  elif [[ "$CUDA_VERSION" == 11.7* ]]; then
    CASE_PREFIX="cu117"
  else
    echo "[WARN] Unrecognized CUDA version: $CUDA_VERSION"
    echo "Pick from options: cu121 (CUDA 12.1+), cu118 (CUDA 11.8), cu117 (CUDA 11.7), or 'cpu'"
    read -p "Enter wheel suffix (cu121/cu118/cu117/cpu): " CASE_PREFIX
  fi
fi

echo "[INFO] Selected backend wheel option: $CASE_PREFIX"

# Create venv if missing
mkdir -p "$BACKEND_DIR"
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install torch wheel
if [ "$CASE_PREFIX" = "cpu" ]; then
  echo "[INFO] Installing CPU-only torch (pip default)"
  pip install "torch" "torchvision" "torchaudio"
else
  echo "[INFO] Installing torch for $CASE_PREFIX via official index"
  pip install "torch" "torchvision" "torchaudio" --index-url "https://download.pytorch.org/whl/$CASE_PREFIX"
fi

echo "[INFO] Installing other backend dependencies from $BACKEND_DIR/requirements.txt"
if [ -f "$BACKEND_DIR/requirements.txt" ]; then
  pip install -r "$BACKEND_DIR/requirements.txt"
else
  echo "[WARN] $BACKEND_DIR/requirements.txt not found. Skipping."
fi

# Done. Verify
echo "-------------------------------------------"
python - <<PYCODE
import sys
import torch
print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("CUDA device name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("Could not query device name:", e)
PYCODE
echo "-------------------------------------------"
echo "[DONE] Backend environment prepared in $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
echo "Then run the backend: uvicorn main:app --host 0.0.0.0 --port 8502 --workers 1"
