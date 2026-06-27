#!/bin/bash
# init_submodules.sh - Clone all Whale1.0 submodule repos with shallow depth
# Usage: bash scripts/init_submodules.sh [--depth 1]

set -e

DEPTH="${2:-1}"  # Default: shallow clone
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE_DIR="$ROOT_DIR/submodules"

# All forked repos
REPOS=(
    "Wan2.2"
    "CogVideo"
    "HunyuanVideo"
    "Open-Sora"
    "mochi"
    "LTX-Video"
    "SkyReels-V1"
    "LivePortrait"
    "Wav2Lip"
    "AnimateDiff"
    "ECCV2022-RIFE"
)

mkdir -p "$SUBMODULE_DIR"
cd "$SUBMODULE_DIR"

echo "🐋 Whale1.0 - Initializing submodules..."
echo "   Depth: $DEPTH (use --depth 0 for full clone)"
echo ""

for repo in "${REPOS[@]}"; do
    if [ -d "$repo" ]; then
        echo "✅ $repo already exists"
        continue
    fi

    echo "📦 Cloning $repo..."
    if [ "$DEPTH" -gt 0 ]; then
        git clone --depth "$DEPTH" "https://github.com/seanlab007/${repo}.git" "$repo" 2>&1
    else
        git clone "https://github.com/seanlab007/${repo}.git" "$repo" 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "   ✅ $repo cloned successfully"
    else
        echo "   ❌ $repo failed to clone"
    fi
    echo ""
done

echo ""
echo "🎉 All submodules initialized!"
du -sh "$SUBMODULE_DIR"/*/ | sort -rh
