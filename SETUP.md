# Setup Guide

## Prerequisites

- Linux server with NVIDIA GPU (CUDA 12.1+)
- At least 96GB VRAM (for 4B model)
- ~20GB disk space
- Internet connection

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/almaiga/reinforce_medical.git
cd reinforce_medical
```

### 2. Run Setup Script

```bash
bash setup_env.sh
```

This script will:
1. Install Miniconda3
2. Create `medical_reward` conda environment
3. Install all dependencies:
   - PyTorch with CUDA support
   - verl (RL training framework)
   - Ray (distributed computing)
   - vLLM (inference backend)
   - Transformers, datasets, etc.
4. Download the base model (Abdine/qwen3-4b-medical-selfplay-sft)

**Note**: You'll need to login to HuggingFace during setup. Get your token from https://huggingface.co/settings/tokens

### 3. Activate Environment

```bash
conda activate medical_reward
```

## Verify Installation

```bash
# Check verl installation
python scripts/verify_verl_installation.py

# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Next Steps

### 1. Prepare Data

```bash
# Convert dataset to verl format
python scripts/convert_dataset_to_verl.py

# Verify conversion
python scripts/verify_dataset.py
```

### 2. Configure Training

Edit `configs/verl_config.yaml` to adjust:
- Model path
- Batch sizes
- Learning rate
- Number of epochs
- GPU allocation

### 3. Run Training

```bash
# Start training
python scripts/train_medical_verl.py --config configs/verl_config.yaml

# Or run in screen session (recommended)
screen -S training
python scripts/train_medical_verl.py --config configs/verl_config.yaml
# Press Ctrl+A then D to detach
```

### 4. Monitor Training

```bash
# View logs
tail -f checkpoints/medical_verl_*/logs/training.log

# Check WandB dashboard (if configured)
# https://wandb.ai/your-project

# Reattach to screen session
screen -r training
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch sizes in `configs/verl_config.yaml`:
```yaml
rollout:
  batch_size: 32  # Reduce from 64
train:
  batch_size: 8   # Reduce from 16
```

### Import Errors

```bash
# Reinstall dependencies
conda activate medical_reward
pip install -r requirements.txt --force-reinstall
```

### Model Download Issues

```bash
# Use HF transfer for faster downloads
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Re-download model
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

### Ray Connection Issues

```bash
# Check Ray status
ray status

# Restart Ray
ray stop
ray start --head
```

## File Structure

```
reinforce_medical/
├── README.md                  # Main documentation
├── SETUP.md                   # This file
├── setup_env.sh              # Setup script
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── medical_team/            # Medical self-play components
│   ├── __init__.py
│   ├── medical_game_manager.py
│   ├── utils.py
│   ├── prompts.py
│   ├── medical_judge.py
│   └── remote_judge.py
│
├── scripts/                 # Training and utility scripts
│   ├── train_medical_verl.py
│   ├── convert_dataset_to_verl.py
│   └── verify_verl_installation.py
│
├── configs/                 # Configuration files
│   └── verl_config.yaml
│
├── data/                    # Datasets
│   └── medical_openrlhf/
│
├── tests/                   # Test files
│
└── .kiro/specs/            # Migration specifications
    └── migrate-to-verl/
```

## Environment Variables

Optional environment variables:

```bash
# HuggingFace
export HF_HOME=/path/to/cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# WandB
export WANDB_API_KEY=your_key
export WANDB_PROJECT=medical-selfplay

# Ray
export RAY_TMPDIR=/path/to/tmp
```

## Useful Commands

```bash
# List conda environments
conda env list

# Activate environment
conda activate medical_reward

# Deactivate environment
conda deactivate

# Update dependencies
pip install -r requirements.txt --upgrade

# Check GPU usage
watch -n 1 nvidia-smi

# List screen sessions
screen -ls

# Reattach to screen
screen -r training

# Kill screen session
screen -X -S training quit
```

## Support

For issues or questions:
1. Check the main README.md
2. Review the migration spec: `.kiro/specs/migrate-to-verl/`
3. Check verl documentation: https://verl.readthedocs.io/
4. Open an issue on GitHub
