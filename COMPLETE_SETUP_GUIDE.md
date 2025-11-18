# Complete Setup Guide - Step by Step

## 🎯 Your Current Status

You're on your SSH server with:
- ✅ Repository cloned
- ✅ Model directory exists (but empty)
- ❌ OpenRLHF not installed (build error)
- ❓ Training data status unknown

Let's fix everything!

---

## 📋 Step-by-Step Setup

### Step 1: Install Dependencies (5 min)

```bash
# Install OpenRLHF from Self-RedTeam fork
./install_dependencies.sh
```

This will:
- Install OpenRLHF with REINFORCE++ support
- Setup medical_team as red_team module
- Install other requirements

**If you get errors**, try:
```bash
# Install build dependencies first
pip install wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then retry
./install_dependencies.sh
```

---

### Step 2: Download Model (10-15 min)

```bash
# Download your fine-tuned model (using modern HF CLI)
mkdir -p trainer_output/qwen3-4b-medical-selfplay-sft

# Option 1: Using huggingface-cli (recommended)
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False

# Option 2: Using hf download (alternative)
# hf download Abdine/qwen3-4b-medical-selfplay-sft \
#     --local-dir trainer_output/qwen3-4b-medical-selfplay-sft
```

**Verify:**
```bash
ls trainer_output/qwen3-4b-medical-selfplay-sft/
# Should show: config.json, model files, tokenizer files
```

---

### Step 3: Generate Training Data (2-3 min)

```bash
# Generate 638 training samples
python scripts/create_rl_training_data.py \
    --input data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
    --output-dir data/medical_rl_training

# Convert to OpenRLHF format
python scripts/convert_to_openrlhf_format.py \
    --input data/medical_rl_training/train.jsonl \
    --output data/medical_openrlhf/train.jsonl
```

**Verify:**
```bash
wc -l data/medical_openrlhf/train.jsonl
# Should show: 638
```

---

### Step 4: Verify Everything (1 min)

```bash
# Check all components
python -c "import openrlhf; print('✅ OpenRLHF OK')"
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json && echo "✅ Model OK"
wc -l data/medical_openrlhf/train.jsonl && echo "✅ Data OK"
ls medical_team/local_reward_function.py && echo "✅ Reward function OK"
nvidia-smi && echo "✅ GPU OK"
```

All should pass!

---

### Step 5: Launch Training! (1-2 hours)

```bash
./launch_training.sh
```

---

## 🚨 Troubleshooting

### Issue: OpenRLHF Build Error

**Error:** `Failed to build 'rlhf'`

**Solution:**
```bash
# Install build dependencies
pip install wheel setuptools build

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install OpenRLHF
cd selfplay-redteaming-reference
pip install -e .
cd ..
```

---

### Issue: Model Directory Empty

**Problem:** `ls trainer_output/qwen3-4b-medical-selfplay-sft` shows nothing

**Solution:**
```bash
# Download the model
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

**If download fails:**
```bash
# Login to HuggingFace first
huggingface-cli login
# or
hf login

# Then retry download
```

**For faster downloads (optional):**
```bash
# Install hf_transfer for faster downloads
pip install hf_transfer

# Set environment variable
export HF_HUB_ENABLE_HF_TRANSFER=1

# Then download
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

---

### Issue: MEDEC Data Not Found

**Error:** `data_copy/MEDEC/MEDEC-MS/...csv not found`

**Solution:**
```bash
# Check if data exists
ls data_copy/MEDEC/MEDEC-MS/

# If not, you need to add the MEDEC dataset
# Contact me if you need help with this
```

---

### Issue: CUDA Not Available

**Error:** `CUDA not available`

**Solution:**
```bash
# Check CUDA
nvidia-smi

# If not working, check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Quick Verification Checklist

Before training, verify:

```bash
# 1. OpenRLHF installed
python -c "import openrlhf; print('OK')"

# 2. Model downloaded
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json

# 3. Data generated
wc -l data/medical_openrlhf/train.jsonl  # Should be 638

# 4. GPU available
nvidia-smi  # Should show RTX PRO 6000

# 5. red_team module
ls selfplay-redteaming-reference/red_team/__init__.py
```

All should succeed!

---

## 📊 Expected Output

### After install_dependencies.sh:
```
✅ OpenRLHF installed
✅ medical_team copied as red_team
✅ Requirements installed
✅ OpenRLHF imported successfully
✅ red_team module ready
```

### After model download:
```
trainer_output/qwen3-4b-medical-selfplay-sft/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
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

## 🚀 Ready to Train!

Once all checks pass:

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
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log
```

---

## 📞 Need Help?

If you encounter issues:

1. Check the error message carefully
2. Try the troubleshooting steps above
3. Verify all prerequisites are met
4. Check GPU memory with `nvidia-smi`

Common issues are usually:
- Missing dependencies (install with pip)
- Model not downloaded (run huggingface-cli download)
- Data not generated (run the scripts)
- CUDA not available (check nvidia-smi)

Good luck! 🎉
