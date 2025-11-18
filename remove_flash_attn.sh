#!/bin/bash
set -e

echo "=========================================="
echo "Removing Flash-Attention"
echo "=========================================="

echo ""
echo "The issue: flash-attention was compiled for PyTorch 2.5.1"
echo "But vLLM requires PyTorch 2.4.0, causing ABI incompatibility"
echo ""
echo "Solution: Remove flash-attention completely"
echo "OpenRLHF will fall back to standard attention"
echo ""

# Uninstall flash-attention
echo "Removing flash-attention..."
pip uninstall -y flash-attn

# Verify it's gone
if python -c "import flash_attn" 2>/dev/null; then
    echo "❌ flash-attn still installed"
    exit 1
else
    echo "✅ flash-attn removed successfully"
fi

echo ""
echo "=========================================="
echo "✅ Flash-Attention Removed!"
echo "=========================================="
echo ""
echo "OpenRLHF will now use standard PyTorch attention"
echo "(slightly slower but fully functional)"
echo ""
echo "Next step: bash launch_training_no_flash.sh"
