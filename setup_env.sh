#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo "Setting up Medical Reward Environment"
echo "=========================================="

# Install system dependencies
echo "Installing system dependencies (screen, git, wget)..."
sudo apt-get update
sudo apt-get install -y screen git wget curl

# Navigate to workspace directory
cd /workspace

# Install Miniconda3
echo "Installing Miniconda3..."
if [ ! -d "/workspace/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /workspace/miniconda3 -u
    rm miniconda.sh
    echo "Miniconda3 installed successfully"
else
    echo "Miniconda3 already installed, skipping..."
fi

# Initialize conda for bash
export PATH="/workspace/miniconda3/bin:$PATH"
eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"

# Add conda to PATH permanently for this session
echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc

# Clone the repository
echo "Cloning repository..."
if [ ! -d "/workspace/reinforce_medical" ]; then
    git clone https://github.com/almaiga/reinforce_medical.git
    echo "Repository cloned successfully"
else
    echo "Repository already exists, pulling latest changes..."
    cd /workspace/reinforce_medical
    git pull
    cd /workspace
fi

# Create conda environment
echo "Creating conda environment 'medical_reward'..."
if conda env list | grep -q "medical_reward"; then
    echo "Environment 'medical_reward' already exists, removing it..."
    conda env remove -n medical_reward -y
fi

conda create -n medical_reward python=3.10 -y

# Activate the environment
echo "Activating environment..."
source /workspace/miniconda3/bin/activate medical_reward

# Install requirements
echo "Installing requirements..."
cd /workspace/reinforce_medical
pip install -r requirements.txt

# Install verl for RL training
echo "Installing verl..."
pip install verl

# Install additional dependencies
echo "Installing additional dependencies..."
pip install ray[default]  # Ray with default dependencies
pip install omegaconf  # For configuration management
pip install wandb  # For logging

# Install huggingface_hub for model download
echo "Installing huggingface_hub..."
pip install huggingface_hub

# Login to HuggingFace and download model
echo ""
echo "=========================================="
echo "HuggingFace Login & Model Download"
echo "=========================================="
echo "Please login to HuggingFace..."
echo "You will need your HuggingFace token (get it from https://huggingface.co/settings/tokens)"
huggingface-cli login

# Create output directory and download model
echo ""
echo "Downloading model Abdine/qwen3-4b-medical-selfplay-sft..."
mkdir -p trainer_output/qwen3-4b-medical-selfplay-sft
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Model location: /workspace/reinforce_medical/trainer_output/qwen3-4b-medical-selfplay-sft"
echo ""
echo "To use the environment in future sessions, run:"
echo "  export PATH=\"/workspace/miniconda3/bin:\$PATH\""
echo "  conda activate medical_reward"
echo ""
echo "To run training in a detached screen session:"
echo "  screen -S training"
echo "  # Run your training command"
echo "  # Press Ctrl+A then D to detach"
echo "  # Use 'screen -r training' to reattach"
echo ""
echo "Useful screen commands:"
echo "  screen -ls              # List all screen sessions"
echo "  screen -r <name>        # Reattach to a session"
echo "  screen -X -S <name> quit # Kill a session"
