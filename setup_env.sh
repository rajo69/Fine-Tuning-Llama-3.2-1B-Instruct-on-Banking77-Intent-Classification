#!/usr/bin/env bash
# =============================================================================
# setup_env.sh
# Environment setup for fine-tuning Llama-3.2-1B-Instruct on Intel Iris Xe.
# Uses `uv` for fast, reproducible dependency management.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, and pipe failures

# ---------------------------------------------------------------------------
# 0. Prerequisite check: ensure `uv` is installed
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[ERROR] 'uv' is not installed. Install it via:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v git &>/dev/null; then
    echo "[ERROR] 'git' is not installed. Please install git first."
    exit 1
fi

echo "========================================"
echo " Setting up Finetuning Environment"
echo " Target hardware: Intel Iris Xe iGPU"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. Create virtual environment using uv
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Creating virtual environment with uv..."
uv venv .venv --python 3.11
echo "      Virtual environment created at .venv/"

# Activate the venv for subsequent commands
# Note: On Windows/bash (Git Bash / WSL), source path differs
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32" ]]; then
    VENV_ACTIVATE=".venv/Scripts/activate"
else
    VENV_ACTIVATE=".venv/bin/activate"
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
echo "      Virtual environment activated."

# ---------------------------------------------------------------------------
# 2. Upgrade pip and install build tools
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Installing build tools..."
uv pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. Install PyTorch — CPU wheel as fallback for Intel Iris Xe
#    Intel Iris Xe (Gen 12 integrated) does not have full XPU driver support
#    in all environments; we install the CPU wheel as a safe fallback.
#    If you have Intel Extension for PyTorch (IPEX) and XPU drivers installed,
#    swap the index URL for: https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing PyTorch (CPU fallback for Intel Iris Xe)..."
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Optionally attempt Intel Extension for PyTorch (IPEX) for XPU acceleration.
# Comment out the block below if XPU drivers are not installed.
echo ""
echo "      [OPTIONAL] Attempting Intel Extension for PyTorch (IPEX) for XPU..."
uv pip install intel-extension-for-pytorch || {
    echo "      [WARNING] IPEX installation failed or not available."
    echo "      Continuing with CPU-only PyTorch. Training will be slower."
}

# ---------------------------------------------------------------------------
# 4. Clone and install Unsloth (Intel GPU branch)
#    The standard PyPI unsloth targets NVIDIA CUDA; the intel-gpu-torch290
#    extra is required for Intel hardware support.
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Installing Unsloth (Intel GPU branch)..."

if [ -d "unsloth" ]; then
    echo "      'unsloth/' directory already exists — pulling latest changes..."
    git -C unsloth pull
else
    git clone https://github.com/unslothai/unsloth.git
fi

cd unsloth
uv pip install ".[intel-gpu-torch290]" || {
    echo "      [WARNING] intel-gpu-torch290 extra failed."
    echo "      Falling back to standard unsloth install..."
    uv pip install .
}
cd ..

# ---------------------------------------------------------------------------
# 5. Install remaining ML/training dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Installing ML/training dependencies..."
uv pip install \
    transformers>=4.40.0 \
    datasets>=2.19.0 \
    trl>=0.8.6 \
    peft>=0.10.0 \
    accelerate>=0.30.0 \
    tensorboard>=2.16.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece \
    protobuf \
    scipy \
    tqdm \
    jsonlines

# ---------------------------------------------------------------------------
# 6. Create required project directories
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Creating project directory structure..."
mkdir -p data src models/final_lora checkpoints/hourly_backup logs

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Activate the venv:"
echo "      source $VENV_ACTIVATE"
echo "   2. Prepare the dataset:"
echo "      python src/1_prepare_data.py"
echo "   3. Run training:"
echo "      python src/2_train_intel.py"
echo "   4. Resume/inference:"
echo "      python src/3_resume_and_infer.py"
echo "========================================"
