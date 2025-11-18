#!/bin/bash

# Medical Self-Play REINFORCE++ Training Launcher
# This script handles all setup and launches training

set -e  # Exit on error

echo "=========================================="
echo "Medical Self-Play REINFORCE++ Training"
echo "=========================================="
echo ""

# Configuration
MODEL_PATH="trainer_output/qwen3-4b-medical-selfplay-sft"
JUDGE_MODEL="google/medgemma-4b-it"
JUDGE_PORT=8000
DATA_DIR="data/medical_openrlhf"
TRAINING_DATA="$DATA_DIR/train.jsonl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Check if model exists
echo "Step 1: Checking model..."
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model not found at $MODEL_PATH"
    echo ""
    echo "Please download the model first:"
    echo "  mkdir -p $MODEL_PATH"
    echo "  huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \\"
    echo "      --local-dir $MODEL_PATH \\"
    echo "      --local-dir-use-symlinks False"
    exit 1
fi
print_status "Model found at $MODEL_PATH"

# Step 2: Check if OpenRLHF is installed
echo ""
echo "Step 2: Checking OpenRLHF installation..."
if ! python -c "import openrlhf" 2>/dev/null; then
    print_error "OpenRLHF not installed"
    echo ""
    echo "Please install OpenRLHF:"
    echo "  cd selfplay-redteaming-reference"
    echo "  pip install -e ."
    echo "  cd .."
    exit 1
fi
print_status "OpenRLHF installed"

# Step 3: Check if red_team module exists
echo ""
echo "Step 3: Checking red_team module..."
if [ ! -f "selfplay-redteaming-reference/red_team/__init__.py" ]; then
    print_warning "red_team module not found, copying medical_team..."
    rm -rf selfplay-redteaming-reference/red_team
    cp -r medical_team selfplay-redteaming-reference/red_team
    print_status "red_team module created"
else
    print_status "red_team module exists"
fi

# Step 4: Check/Generate training data
echo ""
echo "Step 4: Checking training data..."
if [ ! -f "$TRAINING_DATA" ]; then
    print_warning "Training data not found, generating..."
    
    # Generate intermediate format
    echo "  Generating intermediate data..."
    python scripts/create_rl_training_data.py \
        --input data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
        --output-dir data/medical_rl_training
    
    # Convert to OpenRLHF format
    echo "  Converting to OpenRLHF format..."
    python scripts/convert_to_openrlhf_format.py \
        --input data/medical_rl_training/train.jsonl \
        --output $TRAINING_DATA
    
    print_status "Training data generated"
else
    # Check sample count
    SAMPLE_COUNT=$(wc -l < "$TRAINING_DATA")
    print_status "Training data found ($SAMPLE_COUNT samples)"
fi

# Step 5: Check local reward function
echo ""
echo "Step 5: Checking local reward function..."
if [ ! -f "medical_team/local_reward_function.py" ]; then
    print_error "Local reward function not found"
    exit 1
fi
print_status "Local reward function found (no separate judge server needed)"

# Step 6: Final verification
echo ""
echo "Step 6: Final verification..."

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_error "CUDA not available"
    exit 1
fi
print_status "CUDA available"

# Check GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 10000 ]; then
    print_warning "Low GPU memory: ${GPU_MEM}MB free"
    echo "  Consider reducing batch sizes if training fails"
else
    print_status "GPU memory: ${GPU_MEM}MB free"
fi

# Step 7: Launch training
echo ""
echo "=========================================="
echo "🚀 Launching REINFORCE++ Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $TRAINING_DATA"
echo "  Judge: Local (medical_team/local_reward_function.py)"
echo "  GPU: RTX PRO 6000 (single GPU, all models colocated)"
echo ""
echo "Training will start in 3 seconds..."
sleep 3

# Make training script executable
chmod +x scripts/train_medical_reinforce.sh

# Launch training
./scripts/train_medical_reinforce.sh

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
