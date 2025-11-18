# 🚀 START HERE - Medical Self-Play Training

## ✅ Complete Setup in 5 Commands

```bash
# 1. Download Self-RedTeam (2 min)
./download_selfplay_redteaming.sh

# 2. Install dependencies (5-10 min)
./install_dependencies.sh

# 3. Download model (10-15 min)
pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
./download_model.sh

# 4. Generate data (2-3 min)
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py

# 5. Train! (1-2 hours)
./launch_training.sh
```

**Total time: ~2-2.5 hours from scratch to trained model!**

---

## 📚 Documentation

- **`SETUP_FROM_SCRATCH.md`** - Complete step-by-step guide
- **`QUICK_REFERENCE.md`** - One-page reference
- **`HUGGINGFACE_DOWNLOAD_GUIDE.md`** - Model download guide
- **`RTX_PRO_6000_OPTIMIZED.md`** - GPU-specific optimizations

---

## 🎯 What You're Building

**Medical Self-Play Training** using REINFORCE++ from the Self-RedTeam paper:
- **Attacker**: Introduces realistic medical errors
- **Assessor**: Detects medical errors
- **Both improve** through self-play co-evolution

---

## 💻 Your Hardware

- **GPU**: RTX PRO 6000 (96GB VRAM) - Perfect! 🚀
- **Memory usage**: ~53GB / 96GB (55%)
- **Training time**: 1-2 hours per epoch
- **Batch size**: 64 rollout, 16 train (optimized for 96GB)

---

## 📦 What Gets Installed

1. **Self-RedTeam Repository** (without .git)
   - OpenRLHF with REINFORCE++
   - Self-play game logic
   - Reference code

2. **Dependencies**
   - flash-attn (from conda-forge)
   - OpenRLHF (from Self-RedTeam fork)
   - PyTorch, transformers, etc.

3. **Your Fine-Tuned Model**
   - Abdine/qwen3-4b-medical-selfplay-sft
   - ~8GB download

4. **Training Data**
   - 638 samples (4-way balanced)
   - Generated from MEDEC dataset

---

## ✅ Verification

Before training, check:

```bash
# All should succeed:
python -c "import openrlhf; print('✅ OpenRLHF')"
ls selfplay-redteaming-reference/red_team/__init__.py && echo "✅ red_team"
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json && echo "✅ Model"
wc -l data/medical_openrlhf/train.jsonl && echo "✅ Data (638)"
nvidia-smi && echo "✅ GPU"
```

---

## 🚨 Common Issues

### OpenRLHF build error
```bash
pip install wheel setuptools build
pip install torch --index-url https://download.pytorch.org/whl/cu118
./install_dependencies.sh
```

### Model download slow
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
./download_model.sh
```

### Git conflicts
```bash
rm -rf selfplay-redteaming-reference
./download_selfplay_redteaming.sh
```

---

## 📊 Training Output

Checkpoints saved to:
```
checkpoints/medical_selfplay_RL_<timestamp>/
├── ckpt/
│   ├── step_50/
│   ├── step_100/
│   └── ...
└── logs/
    └── training.log
```

---

## 🎉 Ready to Start!

```bash
./download_selfplay_redteaming.sh
```

Then follow the prompts! 🚀

---

## 💡 Key Points

- ✅ **Self-RedTeam fork** (not official OpenRLHF) - has REINFORCE++
- ✅ **No .git directory** - avoids conflicts with your repo
- ✅ **medical_team → red_team** - OpenRLHF expects this name
- ✅ **Local judge** - no separate server needed
- ✅ **Single GPU** - all models colocated on RTX PRO 6000
- ✅ **Fast training** - 1-2 hours with 96GB VRAM

---

**Questions?** Check `SETUP_FROM_SCRATCH.md` for detailed guide!
