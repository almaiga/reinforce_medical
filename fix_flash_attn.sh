#!/bin/bash
set -e

echo "=========================================="
echo "Flash-Attention Fix for Blackwell GPU"
echo "=========================================="

echo ""
echo "The issue: flash-attention was pre-compiled without sm_120 support"
echo "Solution: Rebuild from source with CUDA compute capability 9.0"
echo ""

# Uninstall existing flash-attention
echo "Step 1: Removing pre-built flash-attention..."
pip uninstall -y flash-attn

# Install build dependencies
echo ""
echo "Step 2: Installing build dependencies..."
pip install ninja packaging wheel

# Set environment variables for CUDA architecture
echo ""
echo "Step 3: Setting CUDA architecture flags..."
export TORCH_CUDA_ARCH_LIST="9.0"  # Use compute_90 which is compatible
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4

# Build and install flash-attention from source
echo ""
echo "Step 4: Building flash-attention from source..."
echo "This will take 5-10 minutes..."
pip install flash-attn --no-build-isolation -v

echo ""
echo "=========================================="
echo "✅ Flash-Attention rebuilt successfully!"
echo "=========================================="
echo ""
echo "Next step: bash launch_training.sh"
