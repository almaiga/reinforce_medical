#!/bin/bash

# Quick Setup - Install Dependencies Only
# (Model and data already ready)

set -e

echo "=========================================="
echo "Medical Self-Play - Install Dependencies"
echo "=========================================="
echo ""

# Install OpenRLHF
echo "Installing OpenRLHF..."
pip install openrlhf

# Install other requirements
echo ""
echo "Installing other requirements..."
pip install -r requirements.txt

# Verify everything
echo ""
echo "Verifying setup..."

# Check model
if [ -d "trainer_output/qwen3-4b-medical-selfplay-sft" ]; then
    echo "✅ Model found"
else
    echo "❌ Model not found"
    exit 1
fi

# Check data
if [ -f "data/medical_openrlhf/train.jsonl" ]; then
    SAMPLES=$(wc -l < "data/medical_openrlhf/train.jsonl")
    echo "✅ Training data found ($SAMPLES samples)"
else
    echo "❌ Training data not found"
    exit 1
fi

# Check reward function
if [ -f "medical_team/local_reward_function.py" ]; then
    echo "✅ Local reward function ready"
else
    echo "❌ Local reward function not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Ready to Train!"
echo "=========================================="
echo ""
echo "Run: ./launch_training.sh"
echo ""
