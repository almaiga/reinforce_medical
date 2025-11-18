#!/bin/bash

# Quick activation script for reconnecting to the server

echo "Activating medical_reward environment..."

# Add conda to PATH
export PATH="/workspace/miniconda3/bin:$PATH"

# Initialize conda
eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"

# Activate the environment
conda activate medical_reward

# Navigate to project directory
cd /workspace/reinforce_medical

# Pull latest changes
echo "Pulling latest changes from git..."
git pull

echo "Ready to work! You ar e now in /workspace/reinforce_medical with medical_reward environment active."
