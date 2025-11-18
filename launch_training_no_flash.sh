#!/bin/bash

# Medical Self-Play REINFORCE++ Training Launcher (No Flash-Attention)
# Assumes model and data are already ready
# Flash-attention disabled for Blackwell GPU compatibility

set -e

echo "=========================================="
echo "Medical Self-Play REINFORCE++ Training"
echo "(Flash-Attention Disabled)"
echo "=========================================="
echo ""

# Configuration
MODEL_PATH="trainer_output/qwen3-4b-medical-selfplay-sft"
TRAINING_DATA="data/medical_openrlhf/train.jsonl"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Check model
echo "Step 1: Checking model..."
if [ ! -d "$MODEL_PATH" ]; then
    print_error "Model not found at $MODEL_PATH"
    exit 1
fi
print_status "Model found"

# Step 2: Check OpenRLHF
echo ""
echo "Step 2: Checking OpenRLHF..."
if ! python -c "import openrlhf" 2>/dev/null; then
    print_error "OpenRLHF not installed"
    echo "Run: pip install openrlhf"
    exit 1
fi
print_status "OpenRLHF installed"

# Step 3: Check training data
echo ""
echo "Step 3: Checking training data..."
if [ ! -f "$TRAINING_DATA" ]; then
    print_error "Training data not found at $TRAINING_DATA"
    exit 1
fi
SAMPLES=$(wc -l < "$TRAINING_DATA")
print_status "Training data found ($SAMPLES samples)"

# Step 4: Check reward function
echo ""
echo "Step 4: Checking local reward function..."
if [ ! -f "medical_team/local_reward_function.py" ]; then
    print_error "Local reward function not found"
    exit 1
fi
print_status "Local reward function ready"

# Step 5: Check CUDA
echo ""
echo "Step 5: Checking CUDA..."
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_error "CUDA not available"
    exit 1
fi
GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
print_status "CUDA available (${GPU_MEM}MB free)"

# Warning about flash-attention
echo ""
print_warning "Flash-attention disabled for Blackwell GPU compatibility"
print_warning "Training will be slightly slower but fully functional"

# Launch training
echo ""
echo "=========================================="
echo "🚀 Launching Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $TRAINING_DATA ($SAMPLES samples)"
echo "  Judge: Local (no server needed)"
echo "  GPU: RTX PRO 6000 (96GB VRAM)"
echo "  Flash-Attention: Disabled"
echo ""
echo "Starting in 3 seconds..."
sleep 3

chmod +x scripts/train_medical_reinforce_no_flash.sh
./scripts/train_medical_reinforce_no_flash.sh

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
