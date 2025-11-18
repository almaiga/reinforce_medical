# Quick Reference - Medical Self-Play Training

## 🚀 Complete Setup (4 Commands)

```bash
# 1. Install dependencies
./install_dependencies.sh

# 2. Download model (with fast transfer)
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
./download_model.sh

# 3. Generate training data
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py

# 4. Train!
./launch_training.sh
```

---

## 📋 Individual Steps

### Install Dependencies
```bash
./install_dependencies.sh
```
- Installs OpenRLHF from Self-RedTeam fork
- Sets up medical_team as red_team module
- Installs requirements

### Download Model
```bash
./download_model.sh
```
- Downloads Abdine/qwen3-4b-medical-selfplay-sft
- Uses hf_transfer if available (faster)
- Verifies download

**Manual download:**
```bash
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

### Generate Data
```bash
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py
```
- Creates 638 training samples
- Converts to OpenRLHF format

### Train
```bash
./launch_training.sh
```
- Verifies everything is ready
- Launches REINFORCE++ training
- Saves checkpoints every 50 steps

---

## 🔍 Verification Commands

```bash
# Check OpenRLHF
python -c "import openrlhf; print('✅ OK')"

# Check model
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json

# Check data
wc -l data/medical_openrlhf/train.jsonl  # Should be 638

# Check GPU
nvidia-smi  # Should show RTX PRO 6000 with 96GB

# Check red_team module
ls selfplay-redteaming-reference/red_team/__init__.py
```

---

## 📊 Monitoring Training

```bash
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log

# Find latest checkpoint
ls -lt checkpoints/ | head -5
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

### CUDA not available
```bash
nvidia-smi  # Check GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📁 Key Files

- `install_dependencies.sh` - Install OpenRLHF & dependencies
- `download_model.sh` - Download fine-tuned model
- `launch_training.sh` - Main training launcher
- `scripts/train_medical_reinforce.sh` - Training script
- `medical_team/local_reward_function.py` - Local judge

---

## 📚 Documentation

- `COMPLETE_SETUP_GUIDE.md` - Detailed step-by-step guide
- `HUGGINGFACE_DOWNLOAD_GUIDE.md` - HF CLI usage
- `RTX_PRO_6000_OPTIMIZED.md` - GPU-specific optimizations
- `SINGLE_GPU_SETUP.md` - Single GPU configuration

---

## ⏱️ Time Estimates

- Install dependencies: 5 min
- Download model: 10-15 min (5 min with hf_transfer)
- Generate data: 2-3 min
- Training: 1-2 hours

**Total: ~2 hours from start to trained model!**

---

## ✅ Success Indicators

Training is working if:
- ✅ GPU memory: 50-60GB / 96GB
- ✅ GPU utilization: 85-95%
- ✅ Rewards: Both positive and negative
- ✅ Checkpoints saving every 50 steps
- ✅ No errors in logs

---

## 🎯 Your Hardware

- **GPU**: RTX PRO 6000 (96GB VRAM)
- **Batch size**: 64 rollout, 16 train
- **Memory usage**: ~53GB / 96GB (55%)
- **Training time**: 1-2 hours per epoch
- **Throughput**: ~100 steps/hour

Perfect for this task! 🚀
