#!/bin/bash
set -e

echo "=========================================="
echo "PyTorch + CUDA Fix for RTX PRO 6000"
echo "=========================================="

# Clean up corrupted PyTorch installation
echo ""
echo "Step 1: Cleaning up corrupted PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip cache purge

# Remove any leftover files
echo "Removing leftover PyTorch files..."
rm -rf /workspace/miniconda3/envs/medical_reward/lib/python3.10/site-packages/torch* 2>/dev/null || true
rm -rf /workspace/miniconda3/envs/medical_reward/lib/python3.10/site-packages/torchgen* 2>/dev/null || true

# Install PyTorch with CUDA 12.4 (best available for Blackwell)
echo ""
echo "Step 2: Installing PyTorch 2.5.1 with CUDA 12.4..."
echo "Note: PyTorch 2.5.1 is the latest stable with better CUDA 12.x support"
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM (required for OpenRLHF)
echo ""
echo "Step 3: Installing vLLM..."
pip install vllm==0.6.3.post1

# Install flash-attn
echo ""
echo "Step 4: Installing flash-attention..."
pip install flash-attn --no-build-isolation

# Verify installation
echo ""
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "Note: You may still see CUDA capability warnings for sm_120,"
echo "but PyTorch should work in compatibility mode."
echo ""
echo "Next steps:"
echo "  1. Run: bash fix_red_team.sh"
echo "  2. Run: bash launch_training.sh"
