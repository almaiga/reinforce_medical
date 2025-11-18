# Final Setup Summary - Ready for SSH Server Training

## ✅ What's Complete (Local Development)

All development and preparation work is done:

1. ✅ **Data Pipeline**
   - `scripts/create_rl_training_data.py` - Creates 2x data (contrastive pairs)
   - `scripts/convert_to_openrlhf_format.py` - Converts to OpenRLHF format
   - Tested and working

2. ✅ **Medical Components**
   - `medical_team/` - All game logic, rewards, prompts
   - `scripts/serve_medical_judge.py` - Judge HTTP server
   - Verified against Self-RedTeam

3. ✅ **Training Script**
   - `scripts/train_medical_reinforce.sh` - Complete REINFORCE++ training
   - Configured for your fine-tuned model: `trainer_output/qwen3-4b-medical-selfplay-sft`

4. ✅ **Documentation**
   - `SSH_SERVER_SETUP.md` - Complete server setup guide
   - `TRAINING_READINESS_CHECKLIST.md` - Detailed checklist
   - `NEXT_STEPS.md` - Quick reference

## 🚀 On Your SSH Server

### Quick Start (Copy-Paste Ready)

```bash
# 1. Download your fine-tuned model
mkdir -p trainer_output/qwen3-4b-medical-selfplay-sft
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False

# 2. Install OpenRLHF
cd selfplay-redteaming-reference && pip install -e . && cd ..

# 3. Setup medical_team as red_team
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team

# 4. Generate training data (638 samples)
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py

# 5. Start judge server
nohup python scripts/serve_medical_judge.py \
    --model google/medgemma-4b-it --port 8000 --device cuda \
    > judge_server.log 2>&1 &
sleep 10
curl http://localhost:8000/health

# 6. Run training
chmod +x scripts/train_medical_reinforce.sh
./scripts/train_medical_reinforce.sh
```

## 📊 What You'll Get

### Training Output
- **Checkpoints**: `checkpoints/medical_selfplay_RL_<timestamp>/`
- **Frequency**: Every 50 steps
- **Format**: HuggingFace compatible

### Models Trained
- **Attacker**: Introduces realistic medical errors
- **Assessor**: Detects medical errors
- **Both**: Co-evolve through self-play

### Training Data
- **638 samples** (2x from 319 MEDEC error cases)
- **4-way structure**: 25% each category
  - vanilla_harmful (160 samples)
  - adversarial_harmful (159 samples)
  - vanilla_benign (160 samples)
  - adversarial_benign (159 samples)

## 🎯 Key Configuration

### Model
- **Base**: `trainer_output/qwen3-4b-medical-selfplay-sft` (your fine-tuned model)
- **Size**: Qwen3-4B
- **Already trained on**: Medical self-play SFT data

### Training
- **Algorithm**: REINFORCE++ (Monte Carlo Policy Gradient)
- **Reward**: Zero-sum medical_general_sum
- **Batch size**: 64 rollout, 16 train
- **Learning rate**: 5e-7
- **Epochs**: 1 (can adjust in script)

### Judge
- **Model**: google/medgemma-4b-it
- **Endpoint**: http://localhost:8000/judge
- **Function**: Evaluates medical notes for errors

## 📁 Files to Transfer to Server

All files are ready in your local repo. Just push to git and pull on server:

```bash
# Local (already done)
git add .
git commit -m "Ready for REINFORCE++ training"
git push

# On SSH server
git pull
```

## ⏱️ Estimated Timeline

- **Setup**: 20-30 minutes
  - Model download: 10 min
  - Dependencies: 5 min
  - Data generation: 5 min
  - Judge server: 2 min

- **Training**: 2-4 hours (depends on GPU)
  - 638 samples
  - 1 epoch
  - Checkpoints every 50 steps

## 🔍 Verification Commands

Before training, verify everything:

```bash
# Model exists
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json

# Data ready (should show 638)
wc -l data/medical_openrlhf/train.jsonl

# Judge running
curl http://localhost:8000/health

# OpenRLHF installed
python -c "import openrlhf; print('OK')"

# red_team module
ls selfplay-redteaming-reference/red_team/__init__.py
```

All should succeed before running training.

## 📚 Documentation Reference

- **`SSH_SERVER_SETUP.md`** - Detailed server setup
- **`scripts/train_medical_reinforce.sh`** - Training script
- **`TRAINING_READINESS_CHECKLIST.md`** - Complete checklist
- **`medical_team/README.md`** - Component documentation

## 🎉 You're Ready!

Everything is prepared and tested. Just:
1. Transfer code to SSH server (git push/pull)
2. Run the 6 setup commands
3. Start training!

The hard work is done - implementation, verification, and testing are all complete. Now it's just execution on your server! 🚀

---

**Questions?** Check `SSH_SERVER_SETUP.md` for troubleshooting and detailed explanations.
