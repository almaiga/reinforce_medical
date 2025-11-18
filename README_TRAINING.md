# Medical Self-Play Training - Ready to Go!

## ✅ What You Already Have

- ✅ Fine-tuned model: `trainer_output/qwen3-4b-medical-selfplay-sft/`
- ✅ Training data: `data/medical_openrlhf/train.jsonl` (638 samples)
- ✅ RTX PRO 6000 GPU (96GB VRAM - perfect!)

## 🚀 To Start Training (2 Steps)

### Step 1: Install Dependencies (5 min)

```bash
./quick_start.sh
```

This installs:
- OpenRLHF (official package)
- Other requirements

### Step 2: Train! (1-2 hours)

```bash
./launch_training.sh
```

That's it! ✅

---

## 📊 What Happens

1. **Checks** - Verifies model, data, and dependencies
2. **Trains** - Runs REINFORCE++ self-play training
3. **Saves** - Checkpoints every 50 steps

### Expected:
- GPU usage: 50-60GB / 96GB
- Training time: 1-2 hours
- Output: `checkpoints/medical_selfplay_RL_<timestamp>/`

---

## 🔍 Monitoring

```bash
# Watch GPU
watch -n 1 nvidia-smi

# View logs
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log
```

---

## 📁 What's Where

```
medical_reward_0/
├── trainer_output/
│   └── qwen3-4b-medical-selfplay-sft/  ← Your model (ready!)
├── data/
│   └── medical_openrlhf/
│       └── train.jsonl                  ← Training data (ready!)
├── medical_team/
│   └── local_reward_function.py         ← Local judge (ready!)
├── scripts/
│   └── train_medical_reinforce.sh       ← Training script
├── quick_start.sh                       ← Install dependencies
└── launch_training.sh                   ← Start training
```

---

## ❓ FAQ

### Do I need to download the model?
**No** - You already have it in `trainer_output/`

### Do I need to generate data?
**No** - You already have it in `data/medical_openrlhf/`

### Do I need a judge server?
**No** - Uses local judge on same GPU

### What does quick_start.sh do?
Just installs OpenRLHF and requirements. Skips model/data.

### What does launch_training.sh do?
Checks everything and starts training. No setup, just training.

---

## 🎉 Summary

You're ready! Just run:

```bash
./quick_start.sh      # Install deps (5 min)
./launch_training.sh  # Train (1-2 hours)
```

Everything else is already done! 🚀
