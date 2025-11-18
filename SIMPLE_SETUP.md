# Simple Setup Guide

## ✅ What You Need

- RTX PRO 6000 GPU (96GB VRAM) ✅
- Python 3.8+
- CUDA installed

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Dependencies (5 min)

```bash
# Install OpenRLHF (official package)
pip install openrlhf

# Install other requirements
pip install -r requirements.txt
```

### Step 2: Download Model & Generate Data (25 min)

```bash
# Run the quick start script
./quick_start.sh
```

This will:
- Download your fine-tuned model (Abdine/qwen3-4b-medical-selfplay-sft)
- Generate 638 training samples
- Verify everything is ready

### Step 3: Train! (1-2 hours)

```bash
# Launch training
./launch_training.sh
```

Done! ✅

---

## 📊 What Happens During Training

1. **Loads models** on your GPU
   - Training model: Qwen3-4B (your fine-tuned)
   - Judge model: MedGemma-4B (local)

2. **Runs self-play games**
   - Attacker introduces errors
   - Assessor detects errors
   - Both improve through REINFORCE++

3. **Saves checkpoints** every 50 steps
   - Location: `checkpoints/medical_selfplay_RL_<timestamp>/`

---

## 🔍 Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log
```

Expected:
- GPU memory: 50-60GB / 96GB
- GPU utilization: 85-95%
- Training time: 1-2 hours

---

## ❓ Why Not Install from selfplay-redteaming-reference?

Good question! Here's why:

### We Use Official OpenRLHF
```bash
pip install openrlhf  # ✅ Official package from PyPI
```

### selfplay-redteaming-reference is Just Reference
- It's there for **understanding** the Self-RedTeam paper
- We **adapted** their approach for medical domain
- Our code is in `medical_team/` - ready to use!
- No need to install their fork

### Our Implementation
- `medical_team/` - All our medical components
- `medical_team/local_reward_function.py` - Local judge
- Works with official OpenRLHF package
- Simpler and cleaner!

---

## 📁 What You Have

```
medical_reward_0/
├── medical_team/              ← Our medical components (ready!)
│   ├── local_reward_function.py
│   ├── utils.py
│   ├── prompts.py
│   └── ...
├── scripts/
│   ├── train_medical_reinforce.sh  ← Training script
│   ├── create_rl_training_data.py
│   └── convert_to_openrlhf_format.py
├── selfplay-redteaming-reference/  ← Just for reference
│   └── (not used for training)
├── quick_start.sh             ← Setup script
└── launch_training.sh         ← Training launcher
```

---

## ✅ Checklist

Before training:

- [ ] OpenRLHF installed (`pip install openrlhf`)
- [ ] Model downloaded (run `./quick_start.sh`)
- [ ] Data generated (638 samples)
- [ ] GPU available (`nvidia-smi` shows RTX PRO 6000)

Then:
```bash
./launch_training.sh
```

---

## 🎉 That's It!

No complex setup, no installing from local folders, just:

1. `pip install openrlhf`
2. `./quick_start.sh`
3. `./launch_training.sh`

Training will complete in 1-2 hours on your RTX PRO 6000! 🚀
