# Fix Import Error - Quick Guide

## 🚨 The Error

```
ImportError: cannot import name 'cot_format_check_and_extract' from 'red_team.utils'
```

## ✅ The Fix

The issue is that OpenRLHF expects specific function names from `red_team.utils`. I've added aliases to `medical_team/utils.py` to make it compatible.

### Step 1: Update red_team module

```bash
./update_red_team.sh
```

This will:
- Copy the updated `medical_team/` to `selfplay-redteaming-reference/red_team/`
- Verify all required functions are present

### Step 2: Retry training

```bash
./launch_training.sh
```

---

## 🔍 What Was Fixed

Added these aliases to `medical_team/utils.py`:

```python
# OpenRLHF expects these function names
cot_format_check_and_extract = medical_cot_format_check_and_extract
get_cot_formatting_reward = get_medical_cot_formatting_reward
get_redteaming_game_reward_general_sum = get_medical_game_reward_general_sum
get_redteaming_game_reward_zero_sum = get_medical_game_reward_general_sum
```

Now OpenRLHF can import the functions it expects!

---

## ⚠️ Additional Issue: PyTorch CUDA Compatibility

You also have this warning:

```
NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA capability sm_120 
is not compatible with the current PyTorch installation.
```

### Fix PyTorch for Blackwell (sm_120)

Your GPU is **very new** (Blackwell architecture). You need PyTorch with CUDA 12.4+ support:

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Or use PyTorch nightly (has latest GPU support):

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

---

## 📋 Complete Fix Steps

```bash
# 1. Update red_team module
./update_red_team.sh

# 2. Fix PyTorch for Blackwell GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Retry training
./launch_training.sh
```

---

## ✅ Verification

After running `update_red_team.sh`, you should see:

```
✅ __init__.py found
✅ utils.py found
✅ cot_format_check_and_extract found
✅ get_cot_formatting_reward found
✅ get_redteaming_game_reward_general_sum found
```

All checks should pass!

---

## 🎯 Why This Happened

OpenRLHF's `experience_maker.py` imports from `red_team.utils`:

```python
from red_team.utils import (
    cot_format_check_and_extract,
    get_cot_formatting_reward,
    get_redteaming_game_reward_general_sum,
    get_redteaming_game_reward_zero_sum
)
```

Our medical_team had different function names (with `medical_` prefix), so I added aliases to make them compatible.

---

## 🚀 Ready to Train!

After these fixes:

```bash
./update_red_team.sh
pip install torch --index-url https://download.pytorch.org/whl/cu124
./launch_training.sh
```

Training should start successfully! 🎉
