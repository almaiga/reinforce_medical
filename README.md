# Medical Self-Play RL Training with verl

Reinforcement learning training for medical error detection using adversarial self-play, powered by [verl](https://github.com/volcengine/verl).

## Overview

This project implements a two-player adversarial self-play system for medical error detection:
- **Attacker**: Introduces realistic medical errors into clinical notes
- **Assessor**: Detects and classifies medical errors

The system uses verl's HybridFlow architecture for efficient RL training with PPO/GRPO algorithms.

## Quick Start

### 1. Setup Environment

On your server with GPU:

```bash
# Run the setup script
bash setup_env.sh

# This will:
# - Install Miniconda
# - Create medical_reward environment
# - Install all dependencies (PyTorch, verl, Ray, vLLM)
# - Download the base model
```

### 2. Activate Environment

```bash
conda activate medical_reward
```

### 3. Prepare Data

```bash
# Convert dataset to verl format
python scripts/convert_dataset_to_verl.py

# Verify conversion
python scripts/verify_dataset.py
```

### 4. Train

```bash
# Run training
python scripts/train_medical_verl.py --config configs/verl_config.yaml
```

## Project Structure

```
.
├── medical_team/              # Medical self-play components
│   ├── __init__.py
│   ├── medical_game_manager.py
│   ├── utils.py
│   ├── prompts.py
│   ├── medical_judge.py
│   └── remote_judge.py
├── scripts/                   # Training and utility scripts
│   ├── train_medical_verl.py
│   ├── convert_dataset_to_verl.py
│   └── verify_verl_installation.py
├── configs/                   # Configuration files
│   └── verl_config.yaml
├── data/                      # Datasets
│   └── medical_openrlhf/
├── .kiro/specs/              # Migration specifications
│   └── migrate-to-verl/
├── setup_env.sh              # Environment setup script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Architecture

### verl HybridFlow

```
┌─────────────────────────────────────────────────────────────┐
│                     verl Training Loop                       │
│                                                              │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │  Dataset   │───▶│   Rollout    │───▶│   Training     │ │
│  │  Loader    │    │  Generation  │    │   (PPO/GRPO)   │ │
│  └────────────┘    └──────────────┘    └────────────────┘ │
│                           │                      │          │
│                           ▼                      ▼          │
│                    ┌──────────────┐      ┌────────────┐   │
│                    │  Medical     │      │  Medical   │   │
│                    │  Game        │      │  Reward    │   │
│                    │  Manager     │      │  Function  │   │
│                    └──────────────┘      └────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Medical Self-Play

The system implements a 4-way game structure:
- **vanilla_harmful**: Copy error note as-is (EASY)
- **adversarial_harmful**: Modify/worsen error (HARD)
- **vanilla_benign**: Copy clean note as-is (EASY)
- **adversarial_benign**: Inject error into clean note (HARD)

## Configuration

Training is configured via YAML files in `configs/`:

```yaml
# Example: configs/verl_config.yaml
model:
  path: "trainer_output/qwen3-4b-medical-selfplay-sft"
  
algorithm:
  name: "ppo"
  gamma: 0.99
  lam: 0.95
  
rollout:
  batch_size: 64
  temperature: 0.7
  
train:
  batch_size: 16
  learning_rate: 1e-6
  num_epochs: 10
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX PRO 6000, 96GB VRAM)
- **Memory**: 96GB VRAM recommended for 4B model
- **Storage**: ~20GB for model and datasets

## Development

### Local Development (CPU)

You can develop and test locally without GPU:

```bash
# Install verl locally
conda activate medical_reward
pip install verl

# Test dataset conversion
python scripts/convert_dataset_to_verl.py --test

# Validate structure
python scripts/verify_verl_installation.py
```

### Migration from OpenRLHF

This project is migrating from OpenRLHF to verl. See `.kiro/specs/migrate-to-verl/` for:
- Requirements document
- Design document
- Implementation tasks

## Documentation

- **Setup Guide**: `START_HERE.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Training Guide**: `README_TRAINING.md`
- **Migration Spec**: `.kiro/specs/migrate-to-verl/`

## Key Features

- ✅ **Production-ready**: Built on verl's robust infrastructure
- ✅ **Flexible**: Easy to customize and extend
- ✅ **Efficient**: State-of-the-art throughput with vLLM backend
- ✅ **Scalable**: Supports single-GPU to multi-node training
- ✅ **Well-tested**: Comprehensive test suite

## Resources

- **verl Documentation**: https://verl.readthedocs.io/
- **verl GitHub**: https://github.com/volcengine/verl
- **HybridFlow Paper**: https://arxiv.org/pdf/2409.19256
- **Self-RedTeam Paper**: https://arxiv.org/abs/2506.07468

## License

Apache License 2.0

## Citation

If you use this code, please cite:

```bibtex
@article{verl2024,
  title={verl: Flexible and Efficient RL Training for LLMs},
  author={verl Team},
  year={2024}
}
```
