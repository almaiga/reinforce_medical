#!/bin/bash

# Quick Start Script for Medical Self-Play Training
# Run this on your SSH server after cloning the repo

set -e

echo "=========================================="
echo "Medical Self-Play Quick Start"
echo "=========================================="
echo ""

# 1. Download model
echo "1. Downloading fine-tuned model..."
mkdir -p trainer_output/qwen3-4b-medical-selfplay-sft
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False

# 2. Install OpenRLHF
echo ""
echo "2. Installing OpenRLHF..."
cd selfplay-redteaming-reference
pip install -e .
cd ..

# 3. Setup medical_team as red_team
echo ""
echo "3. Setting up medical_team module..."
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team

# 4. Generate training data
echo ""
echo "4. Generating training data..."
python scripts/create_rl_training_data.py \
    --input data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
    --output-dir data/medical_rl_training

python scripts/convert_to_openrlhf_format.py \
    --input data/medical_rl_training/train.jsonl \
    --output data/medical_openrlhf/train.jsonl

# 5. Verify local reward function
echo ""
echo "5. Verifying local reward function..."
if [ -f "medical_team/local_reward_function.py" ]; then
    echo "✅ Local reward function ready"
    echo "   (No separate judge server needed - all on same GPU)"
else
    echo "❌ Local reward function not found"
    exit 1
fi

# 6. Ready to train
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  ./launch_training.sh"
echo ""
echo "Or directly:"
echo "  ./scripts/train_medical_reinforce.sh"
echo ""
