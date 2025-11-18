# PyTorch + CUDA Fix Guide

## Current Issues

1. **PyTorch Installation Corrupted** - Partial uninstall left broken files
2. **CUDA Compatibility** - RTX PRO 6000 Blackwell (sm_120) not supported by PyTorch 2.6.0
3. **Missing vLLM** - Required for OpenRLHF training

## Solution

Run the automated fix script:

```bash
bash fix_pytorch_cuda.sh
```

This script will:
1. Clean up corrupted PyTorch installation
2. Install PyTorch 2.5.1 with CUDA 12.4 support
3. Install vLLM 0.6.3.post1
4. Install flash-attention
5. Verify all installations

## What Gets Installed

- **PyTorch 2.5.1** - Latest stable with better CUDA 12.x support
- **CUDA 12.4** - Best available for Blackwell architecture
- **vLLM 0.6.3.post1** - Required for OpenRLHF inference
- **flash-attn** - For efficient attention computation

## Expected Warnings

You may still see this warning (it's safe to ignore):
```
NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA capability sm_120 
is not compatible with the current PyTorch installation.
```

**Why it's okay:** PyTorch will run in compatibility mode. The GPU will still work, just not with the latest Blackwell-specific optimizations.

## After Installation

Once the fix completes successfully:

1. **Apply import fixes:**
   ```bash
   bash fix_red_team.sh
   ```

2. **Launch training:**
   ```bash
   bash launch_training.sh
   ```

## Verification

The script will verify:
- ✅ PyTorch version and CUDA availability
- ✅ GPU detection (RTX PRO 6000)
- ✅ vLLM installation

## Troubleshooting

If you still encounter issues:

1. **Check CUDA version:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Test PyTorch manually:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Test vLLM:**
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

## Alternative: Manual Installation

If the script fails, install manually:

```bash
# Clean up
pip uninstall -y torch torchvision torchaudio
pip cache purge

# Install PyTorch with CUDA 12.4
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
pip install vllm==0.6.3.post1

# Install flash-attention
pip install flash-attn --no-build-isolation
```

## Why PyTorch 2.5.1 Instead of 2.6.0?

- Better CUDA 12.x compatibility
- More stable for production workloads
- Still supports all features needed for training
- Fewer edge case bugs with newer GPUs
