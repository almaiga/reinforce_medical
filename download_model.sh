#!/bin/bash

# Download Fine-Tuned Model from HuggingFace
# Uses modern HuggingFace CLI with optional fast transfer

set -e

echo "=========================================="
echo "Downloading Fine-Tuned Model"
echo "=========================================="
echo ""

MODEL_NAME="Abdine/qwen3-4b-medical-selfplay-sft"
OUTPUT_DIR="trainer_output/qwen3-4b-medical-selfplay-sft"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if hf_transfer is available for faster downloads
if python -c "import hf_transfer" 2>/dev/null; then
    echo "✅ hf_transfer available - using fast download"
    export HF_HUB_ENABLE_HF_TRANSFER=1
else
    echo "ℹ️  hf_transfer not installed - using standard download"
    echo "   (Install with: pip install hf_transfer for faster downloads)"
fi

echo ""
echo "Downloading model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Download using modern HuggingFace CLI
huggingface-cli download "$MODEL_NAME" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "=========================================="
echo "✅ Model Downloaded Successfully!"
echo "=========================================="
echo ""

# Verify download
if [ -f "$OUTPUT_DIR/config.json" ]; then
    echo "✅ Model files verified"
    echo ""
    echo "Model contents:"
    ls -lh "$OUTPUT_DIR"
else
    echo "❌ Error: Model files not found"
    echo "Please check the download and try again"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Generate data: python scripts/create_rl_training_data.py && python scripts/convert_to_openrlhf_format.py"
echo "2. Train: ./launch_training.sh"
echo ""
