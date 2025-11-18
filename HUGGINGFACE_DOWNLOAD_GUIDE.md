# HuggingFace Model Download Guide

## 🚀 Modern HuggingFace CLI Commands

### Recommended: `huggingface-cli download`

```bash
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

### Alternative: `hf download`

```bash
hf download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft
```

Both commands work with modern HuggingFace CLI (v0.20+).

---

## ⚡ Fast Downloads with hf_transfer

For **much faster** downloads (especially for large models):

### 1. Install hf_transfer

```bash
pip install hf_transfer
```

### 2. Enable it

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### 3. Download (same command)

```bash
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

**Speed improvement:** 2-5x faster! 🚀

---

## 🔐 Authentication

If the model requires authentication:

```bash
# Login once
huggingface-cli login
# or
hf login

# Then download
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False
```

---

## 📦 Using Our Script

We've created a convenient script:

```bash
./download_model.sh
```

This script:
- ✅ Creates output directory
- ✅ Checks for hf_transfer (uses if available)
- ✅ Downloads the model
- ✅ Verifies download completed
- ✅ Shows next steps

---

## 🔍 Verify Download

After downloading:

```bash
# Check files exist
ls -lh trainer_output/qwen3-4b-medical-selfplay-sft/

# Should show:
# - config.json
# - model files (.safetensors)
# - tokenizer files
# - generation_config.json
```

---

## 🚨 Troubleshooting

### Issue: Command not found

**Error:** `huggingface-cli: command not found`

**Solution:**
```bash
pip install --upgrade huggingface_hub
```

---

### Issue: Slow download

**Solution:** Use hf_transfer
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# Then retry download
```

---

### Issue: Authentication required

**Error:** `Repository not found` or `Access denied`

**Solution:**
```bash
huggingface-cli login
# Enter your HuggingFace token
# Then retry download
```

---

### Issue: Disk space

**Error:** `No space left on device`

**Check space:**
```bash
df -h .
```

**Model size:** ~8GB for Qwen3-4B

---

## 📊 Download Progress

The download will show:

```
Fetching 10 files: 100%|████████████| 10/10 [00:30<00:00,  3.00s/it]
```

For large models, this can take 10-30 minutes depending on your connection.

With `hf_transfer`, it's much faster!

---

## ✅ Complete Example

```bash
# 1. Install fast transfer (optional but recommended)
pip install hf_transfer

# 2. Enable it
export HF_HUB_ENABLE_HF_TRANSFER=1

# 3. Download model
huggingface-cli download Abdine/qwen3-4b-medical-selfplay-sft \
    --local-dir trainer_output/qwen3-4b-medical-selfplay-sft \
    --local-dir-use-symlinks False

# 4. Verify
ls trainer_output/qwen3-4b-medical-selfplay-sft/config.json
```

---

## 🎯 Quick Start

Just run our script:

```bash
./download_model.sh
```

It handles everything automatically! 🚀
