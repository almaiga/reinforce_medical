# Training Scripts Guide

## 🚀 Three Ways to Launch Training

### Option 1: Automated Launch (Recommended)
**Use this if you've already done setup**

```bash
./launch_training.sh
```

This script:
- ✅ Checks all prerequisites
- ✅ Generates data if missing
- ✅ Starts judge server if not running
- ✅ Verifies everything
- ✅ Launches training

**Best for:** Running training after initial setup

---

### Option 2: Quick Start (First Time)
**Use this for initial setup on SSH server**

```bash
./quick_start.sh
```

This script:
- Downloads your fine-tuned model
- Installs OpenRLHF
- Sets up medical_team module
- Generates training data
- Starts judge server

Then run:
```bash
./launch_training.sh
```

**Best for:** First-time setup on a new server

---

### Option 3: Direct Training
**Use this if everything is already set up**

```bash
./scripts/train_medical_reinforce.sh
```

This directly runs training without checks.

**Best for:** Quick restarts when you know everything is ready

---

## 📋 Prerequisites

Before running any script, ensure:
- CUDA-capable GPU available
- Python 3.8+ installed
- Git repository cloned
- Sufficient disk space (~20GB)

---

## 🔧 Script Details

### `launch_training.sh` (Main Launcher)

**What it does:**
1. Checks if model exists
2. Verifies OpenRLHF installation
3. Ensures red_team module is set up
4. Generates training data if missing
5. Starts judge server if not running
6. Verifies CUDA and GPU memory
7. Launches training

**Usage:**
```bash
./launch_training.sh
```

**Output:**
- Colored status messages (✅ ❌ ⚠️)
- Step-by-step progress
- Automatic error handling

---

### `quick_start.sh` (Initial Setup)

**What it does:**
1. Downloads model from HuggingFace
2. Installs OpenRLHF from Self-RedTeam fork
3. Copies medical_team as red_team
4. Generates 638 training samples
5. Starts judge server

**Usage:**
```bash
./quick_start.sh
```

**Time:** ~20-30 minutes (mostly model download)

---

### `scripts/train_medical_reinforce.sh` (Training Only)

**What it does:**
- Runs OpenRLHF REINFORCE++ training
- Uses your fine-tuned model
- Saves checkpoints every 50 steps

**Configuration:**
- Model: `trainer_output/qwen3-4b-medical-selfplay-sft`
- Data: `data/medical_openrlhf/train.jsonl` (638 samples)
- Judge: `http://localhost:8000/judge`
- Batch size: 64 rollout, 16 train
- Learning rate: 5e-7

**Usage:**
```bash
./scripts/train_medical_reinforce.sh
```

---

## 📊 Monitoring Training

### View Training Logs
```bash
# Find latest checkpoint directory
ls -lt checkpoints/ | head -5

# View logs
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log
```

### Check Judge Server
```bash
# View judge logs
tail -f judge_server.log

# Test judge endpoint
curl http://localhost:8000/health
```

### Monitor GPU
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

---

## 🚨 Troubleshooting

### Training Won't Start

**Check prerequisites:**
```bash
# Model exists?
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json

# Data exists?
wc -l data/medical_openrlhf/train.jsonl

# Judge running?
curl http://localhost:8000/health

# OpenRLHF installed?
python -c "import openrlhf; print('OK')"
```

### Judge Server Issues

**Restart judge:**
```bash
# Kill existing
pkill -f serve_medical_judge

# Start new
nohup python scripts/serve_medical_judge.py \
    --model google/medgemma-4b-it \
    --port 8000 \
    --device cuda \
    > judge_server.log 2>&1 &

# Wait and test
sleep 10
curl http://localhost:8000/health
```

### CUDA Out of Memory

**Edit `scripts/train_medical_reinforce.sh`:**
```bash
# Reduce batch sizes
--micro_rollout_batch_size 2  # was 4
--rollout_batch_size 32        # was 64

# Reduce GPU memory
--vllm_gpu_memory_utilization 0.5  # was 0.7
```

### Import Errors

**Reinstall red_team module:**
```bash
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team
```

---

## ⏱️ Expected Timeline

### First Time Setup (`quick_start.sh`)
- Model download: 10-15 min
- OpenRLHF install: 3-5 min
- Data generation: 2-3 min
- Judge server start: 1-2 min
- **Total: ~20-30 min**

### Training (`launch_training.sh`)
- Verification: 1 min
- Training: 2-4 hours (1 epoch, 638 samples)
- **Total: ~2-4 hours**

### Subsequent Runs
- Just run `./launch_training.sh`
- Skips setup steps
- Starts training immediately

---

## 📁 Output Files

### Checkpoints
```
checkpoints/medical_selfplay_RL_<timestamp>/
├── ckpt/                    # Model checkpoints
│   ├── step_50/
│   ├── step_100/
│   └── ...
├── logs/                    # Training logs
│   └── training.log
└── config.json              # Training config
```

### Logs
```
judge_server.log             # Judge server output
checkpoints/.../logs/        # Training logs
```

### Data
```
data/medical_rl_training/
└── train.jsonl              # Intermediate format

data/medical_openrlhf/
└── train.jsonl              # OpenRLHF format (638 samples)
```

---

## ✅ Quick Reference

### First Time on Server
```bash
./quick_start.sh
./launch_training.sh
```

### Subsequent Training Runs
```bash
./launch_training.sh
```

### Direct Training (No Checks)
```bash
./scripts/train_medical_reinforce.sh
```

### Stop Training
```bash
# Ctrl+C in terminal
# Or kill process
pkill -f train_ppo_ray
```

### Stop Judge Server
```bash
pkill -f serve_medical_judge
```

---

## 🎯 Success Indicators

Training is working correctly if you see:

1. ✅ Judge server responding to health checks
2. ✅ Training starting without errors
3. ✅ Both positive and negative rewards
4. ✅ Rewards sum to approximately zero (zero-sum)
5. ✅ Checkpoints being saved every 50 steps
6. ✅ GPU utilization 70-90%

---

## 📚 Additional Documentation

- `SSH_SERVER_SETUP.md` - Detailed server setup guide
- `TRAINING_READINESS_CHECKLIST.md` - Complete checklist
- `FINAL_SETUP_SUMMARY.md` - Quick summary
- `medical_team/README.md` - Component documentation

---

**Ready to train!** 🚀

Start with `./quick_start.sh` (first time) or `./launch_training.sh` (subsequent runs)
