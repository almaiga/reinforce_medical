# Setup From Scratch - Complete Guide

## 🎯 Starting Fresh

This guide assumes you're starting from scratch on your SSH server.

---

## 📋 Step-by-Step Setup

### Step 1: Download Self-RedTeam Repository (2 min)

```bash
# Download Self-RedTeam (without .git to avoid conflicts)
./download_selfplay_redteaming.sh
```

This will:
- Clone https://github.com/mickelliu/selfplay-redteaming
- Remove .git directory (to avoid conflicts with your repo)
- Verify download

**Verify:**
```bash
ls selfplay-redteaming-reference/
# Should show: openrlhf/, red_team/, scripts/, etc.
```

---

### Step 2: Install Dependencies (5-10 min)

```bash
# Install OpenRLHF and dependencies
./install_dependencies.sh
```

This will:
- Install flash-attn from conda-forge (pre-built, fast)
- Install OpenRLHF from Self-RedTeam fork
- Copy medical_team as red_team module
- Install other requirements

**Verify:**
```bash
python -c "import openrlhf; print('✅ OpenRLHF OK')"
ls selfplay-redteaming-reference/red_team/__init__.py
```

---

### Step 3: Download Model (10-15 min)

```bash
# Optional: Install hf_transfer for faster downloads
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download model
./download_model.sh
```

**Verify:**
```bash
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json
# Should exist
```

---

### Step 4: Generate Training Data (2-3 min)

```bash
# Generate 638 training samples
python scripts/create_rl_training_data.py

# Convert to OpenRLHF format
python scripts/convert_to_openrlhf_format.py
```

**Verify:**
```bash
wc -l data/medical_openrlhf/train.jsonl
# Should show: 638
```

---

### Step 5: Launch Training! (1-2 hours)

```bash
./launch_training.sh
```

---

## 🚀 Quick Setup (All Commands)

```bash
# 1. Download Self-RedTeam
./download_selfplay_redteaming.sh

# 2. Install dependencies
./install_dependencies.sh

# 3. Download model (with fast transfer)
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
./download_model.sh

# 4. Generate data
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py

# 5. Train!
./launch_training.sh
```

---

## 📁 What Gets Created

```
medical_reward_0/
├── selfplay-redteaming-reference/  ← Downloaded (no .git)
│   ├── openrlhf/                   ← OpenRLHF with REINFORCE++
│   ├── red_team/                   ← Copied from medical_team/
│   └── ...
├── trainer_output/
│   └── qwen3-4b-medical-selfplay-sft/  ← Downloaded model
├── data/
│   ├── medical_rl_training/
│   │   └── train.jsonl             ← Intermediate format
│   └── medical_openrlhf/
│       └── train.jsonl             ← OpenRLHF format (638 samples)
└── checkpoints/                    ← Created during training
    └── medical_selfplay_RL_<timestamp>/
```

---

## ✅ Verification Checklist

Before training, verify everything:

```bash
# 1. Self-RedTeam downloaded
ls selfplay-redteaming-reference/openrlhf/

# 2. OpenRLHF installed
python -c "import openrlhf; print('OK')"

# 3. red_team module
ls selfplay-redteaming-reference/red_team/__init__.py

# 4. Model downloaded
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json

# 5. Data generated
wc -l data/medical_openrlhf/train.jsonl  # Should be 638

# 6. GPU available
nvidia-smi  # Should show RTX PRO 6000
```

All should succeed!

---

## 🚨 Troubleshooting

### Issue: Git conflicts

**Problem:** "fatal: destination path 'selfplay-redteaming-reference' already exists"

**Solution:**
```bash
rm -rf selfplay-redteaming-reference
./download_selfplay_redteaming.sh
```

---

### Issue: OpenRLHF build error

**Error:** "Failed to build 'rlhf'"

**Solution:**
```bash
# Install build dependencies
pip install wheel setuptools build

# Install PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Retry
./install_dependencies.sh
```

---

### Issue: flash-attn compilation

**Error:** "Failed building wheel for flash-attn"

**Solution:** Use conda-forge (pre-built):
```bash
conda install -c conda-forge flash-attn -y
```

This is already in `install_dependencies.sh`!

---

### Issue: Model download slow

**Solution:** Use hf_transfer:
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
./download_model.sh
```

---

## 📊 Expected Output

### After download_selfplay_redteaming.sh:
```
✅ Self-RedTeam Downloaded Successfully!
✅ OpenRLHF directory found
⚠️  red_team directory not found (will be replaced with medical_team)
```

### After install_dependencies.sh:
```
✅ flash-attn installed
✅ OpenRLHF installed
✅ medical_team copied as red_team
✅ Requirements installed
✅ OpenRLHF imported successfully
✅ red_team module ready
```

### After download_model.sh:
```
✅ Model Downloaded Successfully!
✅ Model files verified
```

### After data generation:
```
✅ Converted 638 records
📊 Distribution:
   - adversarial_benign: 159 (25.0%)
   - adversarial_harmful: 159 (25.0%)
   - vanilla_benign: 160 (25.0%)
   - vanilla_harmful: 160 (25.0%)
```

---

## ⏱️ Time Estimates

- Download Self-RedTeam: 2 min
- Install dependencies: 5-10 min
- Download model: 10-15 min (5 min with hf_transfer)
- Generate data: 2-3 min
- **Setup total: ~20-30 min**
- Training: 1-2 hours

**Total: ~2-2.5 hours from scratch to trained model!**

---

## 🎯 Why This Approach?

### Self-RedTeam Fork (Not Official OpenRLHF)
- ✅ Has REINFORCE++ implementation
- ✅ Has self-play game logic
- ✅ Tested with the paper
- ❌ Official OpenRLHF doesn't have these features

### No .git Directory
- ✅ Avoids conflicts with your main repo
- ✅ Cleaner git status
- ✅ Still get all the code
- ✅ Already in .gitignore

### medical_team as red_team
- ✅ OpenRLHF expects module named "red_team"
- ✅ Our medical_team is adapted from their red_team
- ✅ Drop-in replacement
- ✅ Works perfectly

---

## 📚 Key Files

- `download_selfplay_redteaming.sh` - Download Self-RedTeam (no .git)
- `install_dependencies.sh` - Install OpenRLHF & deps
- `download_model.sh` - Download fine-tuned model
- `launch_training.sh` - Main training launcher

---

## 🎉 You're Ready!

Once all steps complete:

```bash
./launch_training.sh
```

Training will:
- Load models on GPU (~53GB / 96GB)
- Run self-play games
- Save checkpoints every 50 steps
- Complete in 1-2 hours

Monitor with:
```bash
watch -n 1 nvidia-smi
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log
```

Good luck! 🚀
