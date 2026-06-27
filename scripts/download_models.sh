#!/bin/bash
# Whale1.0 Model Weight Download Script
# Downloads model weights from Hugging Face Hub
set -e

echo "🐋 Downloading Whale1.0 model weights..."
echo "========================================="

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

download_model() {
    local repo_id=$1
    local target=$2
    echo ""
    echo "📥 Downloading $repo_id..."
    python3 -c "
from huggingface_hub import snapshot_download
print(f'Downloading {repo_id}...')
snapshot_download(repo_id='$repo_id', local_dir='$target', resume_download=True)
print(f'✅ Downloaded to $target')
" 2>&1 || echo "⚠️  Failed to download $repo_id (might need huggingface_hub installed)"
}

echo ""
echo "Available models (select what you need):"
echo "  1) Wan2.2-T2V-14B  (~30GB) - Best overall"
echo "  2) Wan2.2-I2V-14B  (~30GB) - Image-to-video"
echo "  3) CogVideoX-2b    (~20GB) - Chinese optimized"
echo "  4) HunyuanVideo-1.5(~16GB) - Lightweight"
echo "  5) All models (will take ~100GB+)"
echo ""
read -p "Select models to download [1-5]: " choice

case $choice in
    1) download_model "Wan-AI/Wan2.2-T2V-14B" "$MODELS_DIR/Wan2.2-T2V-14B" ;;
    2) download_model "Wan-AI/Wan2.2-I2V-14B" "$MODELS_DIR/Wan2.2-I2V-14B" ;;
    3) download_model "THUDM/CogVideoX-2b" "$MODELS_DIR/CogVideoX-2b" ;;
    4) download_model "Tencent/HunyuanVideo-1.5" "$MODELS_DIR/HunyuanVideo-1.5" ;;
    5)
        download_model "Wan-AI/Wan2.2-T2V-14B" "$MODELS_DIR/Wan2.2-T2V-14B"
        download_model "Wan-AI/Wan2.2-I2V-14B" "$MODELS_DIR/Wan2.2-I2V-14B"
        download_model "THUDM/CogVideoX-2b" "$MODELS_DIR/CogVideoX-2b"
        download_model "Tencent/HunyuanVideo-1.5" "$MODELS_DIR/HunyuanVideo-1.5"
        download_model "genmo/mochi-1-preview" "$MODELS_DIR/mochi-1-preview"
        ;;
    *) echo "Invalid choice" ;;
esac

echo ""
echo "🎉 Download complete!"
echo "Models stored in: $MODELS_DIR"
