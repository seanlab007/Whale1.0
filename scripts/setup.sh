#!/bin/bash
# Whale1.0 One-Click Setup Script
set -e

echo "🐋 Whale1.0 Setup - One-Click Environment Configuration"
echo "========================================================"

# Check Python
PYTHON=$(which python3 || which python)
if [ -z "$PYTHON" ]; then
    echo "❌ Python3 not found. Please install Python 3.10+"
    exit 1
fi
echo "✅ Python: $($PYTHON --version)"

# Check PyTorch
$PYTHON -c "import torch; print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || {
    echo "⚠️  PyTorch not found. Installing..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

# Install Whale1.0
echo "📦 Installing Whale1.0..."
pip install -e .
echo "✅ Whale1.0 installed!"

# Create directories
mkdir -p models outputs submodules

# Init submodules (optional)
echo ""
echo "📋 To initialize submodules from forked repos:"
echo "   git submodule add https://github.com/seanlab007/Wan2.2.git submodules/Wan2.2"
echo "   git submodule add https://github.com/seanlab007/CogVideo.git submodules/CogVideo"
echo "   git submodule add https://github.com/seanlab007/HunyuanVideo.git submodules/HunyuanVideo"
echo ""

echo "🎉 Setup complete! Run: python scripts/run_pipeline.py --prompt \"your prompt\""
