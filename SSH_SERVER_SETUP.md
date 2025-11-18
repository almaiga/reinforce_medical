# SSH Server Setup Guide for REINFORCE++ Training

## 📋 Prerequisites on SSH Server

Before starting, ensure you have:
- CUDA-capable GPU
- Python 3.8+
- Sufficient disk space (~20GB for model + checkpoints)

## 🚀 Setup Steps on SSH Server

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd medical_reward_0
```

### 2. Download Fine-Tuned Model
```bash
# Create directory
mkdir -p trainer_output/qwen3-4b-medical-selfplay-sft

# Download from HuggingFace
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

### 3. Install Dependencies
```bash
# Install OpenRLHF (Self-RedTeam fork)
cd selfplay-redteaming-reference
pip install -e .
cd ..

# Install other requirements
pip install -r requirements.txt
```

### 4. Copy Medical Team Module
```bash
# OpenRLHF expects module named 'red_team'
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team
```

### 5. Generate Training Data
```bash
# Generate full dataset (638 samples from 319 error cases)
python scripts/create_rl_training_data.py \
    --input data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
    --output-dir data/medical_rl_training

# Convert to OpenRLHF format
python scripts/convert_to_openrlhf_format.py \
    --input data/medical_rl_training/train.jsonl \
    --output data/medical_openrlhf/train.jsonl
```

Expected output:
```
✅ Converted 638 records
📊 Distribution:
   - adversarial_benign: 159 (25.0%)
   - adversarial_harmful: 159 (25.0%)
   - vanilla_benign: 160 (25.0%)
   - vanilla_harmful: 160 (25.0%)
```

### 6. Start Judge Server
```bash
# Start in background
nohup python scripts/serve_medical_judge.py \
    --model google/medgemma-4b-it \
    --port 8000 \
    --device cuda > judge_server.log 2>&1 &

# Wait for it to start
sleep 10

# Test connection
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "model": "google/medgemma-4b-it"}
```

### 7. Run Training
```bash
# Make script executable
chmod +x scripts/train_medical_reinforce.sh

# Run training
./scripts/train_medical_reinforce.sh
```

## 📁 Expected Directory Structure on Server

```
medical_reward_0/
├── trainer_output/
│   └── qwen3-4b-medical-selfplay-sft/    # Your fine-tuned model
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files...
├── data/
│   ├── medical_rl_training/
│   │   └── train.jsonl                    # Intermediate format
│   └── medical_openrlhf/
│       └── train.jsonl                    # OpenRLHF format (638 samples)
├── selfplay-redteaming-reference/
│   ├── red_team/                          # Copied from medical_team/
│   └── openrlhf/                          # REINFORCE++ implementation
├── checkpoints/                           # Training checkpoints (created)
└── scripts/
    ├── train_medical_reinforce.sh         # Training script
    ├── serve_medical_judge.py             # Judge server
    └── ...
```

## 🔍 Verify Setup

Before training, check:

```bash
# 1. Model exists
ls -lh trainer_output/qwen3-4b-medical-selfplay-sft/

# 2. Training data exists
wc -l data/medical_openrlhf/train.jsonl
# Should show: 638

# 3. Judge server running
curl http://localhost:8000/health

# 4. OpenRLHF installed
python -c "import openrlhf; print('✅ OpenRLHF installed')"

# 5. red_team module exists
ls selfplay-redteaming-reference/red_team/
# Should show: __init__.py, utils.py, prompts.py, etc.
```

## 🎯 Training Configuration

The training script (`scripts/train_medical_reinforce.sh`) uses:

- **Model**: `trainer_output/qwen3-4b-medical-selfplay-sft` (your fine-tuned model)
- **Data**: `data/medical_openrlhf/train.jsonl` (638 samples)
- **Judge**: `http://localhost:8000/judge` (local server)
- **Batch size**: 64 rollout, 16 train
- **Learning rate**: 5e-7
- **Algorithm**: REINFORCE++ (advantage_estimator=reinforce)
- **Reward**: Zero-sum medical_general_sum

## 📊 Monitoring Training

### Check Progress
```bash
# View training logs
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log

# Check judge server logs
tail -f judge_server.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Expected Metrics
- **Rewards**: Both positive and negative (zero-sum)
- **Win rates**: ~50/50 attacker vs assessor
- **CoT violations**: Should decrease over time
- **Training time**: ~2-4 hours for 1 epoch (depends on GPU)

## 🚨 Troubleshooting

### Judge Server Issues
```bash
# Check if running
ps aux | grep serve_medical_judge

# Restart if needed
pkill -f serve_medical_judge
nohup python scripts/serve_medical_judge.py \
    --model google/medgemma-4b-it \
    --port 8000 \
    --device cuda > judge_server.log 2>&1 &
```

### CUDA Out of Memory
Edit `scripts/train_medical_reinforce.sh`:
```bash
# Reduce batch sizes
--micro_rollout_batch_size 2  # was 4
--rollout_batch_size 32        # was 64

# Reduce GPU memory
--vllm_gpu_memory_utilization 0.5  # was 0.7
```

### Import Errors
```bash
# Ensure red_team module is copied
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team

# Reinstall OpenRLHF
cd selfplay-redteaming-reference
pip install -e . --force-reinstall
cd ..
```

## ✅ Quick Start Checklist

On your SSH server, run these commands in order:

```bash
# 1. Download model
mkdir -p trainer_output/qwen3-4b-medical-selfplay-sft
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False

# 2. Install OpenRLHF
cd selfplay-redteaming-reference && pip install -e . && cd ..

# 3. Copy medical_team
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team

# 4. Generate data
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py

# 5. Start judge
nohup python scripts/serve_medical_judge.py \
    --model google/medgemma-4b-it --port 8000 --device cuda \
    > judge_server.log 2>&1 &

# 6. Wait and test
sleep 10
curl http://localhost:8000/health

# 7. Train!
chmod +x scripts/train_medical_reinforce.sh
./scripts/train_medical_reinforce.sh
```

## 🎉 Success!

If everything is working, you should see:
- Judge server responding to health checks
- Training starting without errors
- Rewards being computed (both positive and negative)
- Checkpoints being saved every 50 steps

Training will take several hours. Checkpoints are saved to:
`checkpoints/medical_selfplay_RL_<timestamp>/`

Good luck with your training! 🚀
