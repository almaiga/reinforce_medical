# Single GPU Setup - RTX PRO 6000

## ✅ Optimized for Your Hardware

All models run on the same GPU - no separate judge server needed!

### Hardware
- **GPU**: RTX PRO 6000 (96GB VRAM!) 🚀
- **Models**: All 4B models
  - Training model: Qwen3-4B (your fine-tuned)
  - Judge model: MedGemma-4B (loaded in-process)

### Memory Configuration
- **Training models**: ~55% GPU memory (53GB / 96GB)
- **Judge model**: Loaded on-demand, shares GPU
- **Batch sizes**: Optimized for 96GB (4/16 rollout/train)

---

## 🚀 Quick Start

```bash
# 1. Setup (20-30 min)
./quick_start.sh

# 2. Train (2-4 hours)
./launch_training.sh
```

---

## 📊 Training Configuration

### Optimized for 96GB VRAM

```bash
# Batch sizes (optimized for speed)
--micro_train_batch_size 4
--train_batch_size 16
--micro_rollout_batch_size 4
--rollout_batch_size 64

# GPU memory (plenty of headroom)
--vllm_gpu_memory_utilization 0.75

# All models colocated
--colocate_all_models
--actor_num_gpus_per_node 1
--ref_num_gpus_per_node 1
```

### Local Judge Evaluation

- **No HTTP server** - Judge runs in-process
- **Shared GPU** - All models on same device
- **Efficient** - No network overhead
- **Simple** - Fewer moving parts

---

## 💾 Memory Usage

### Expected GPU Memory

| Component | Memory | Notes |
|-----------|--------|-------|
| Training model (actor) | ~10GB | Qwen3-4B + larger batches |
| Reference model | ~10GB | Qwen3-4B |
| Judge model | ~10GB | MedGemma-4B (loaded on-demand) |
| VLLM engine | ~8GB | Generation |
| Activations/gradients | ~15GB | Training with larger batches |
| **Total** | **~53GB** | **Only 55% of 96GB!** ✅ |

### If You Want Even Faster Training

With 96GB, you have plenty of headroom! Edit `scripts/train_medical_reinforce.sh`:

```bash
# Increase batch sizes for maximum speed
--micro_rollout_batch_size 8
--rollout_batch_size 128

# Use more GPU memory
--vllm_gpu_memory_utilization 0.85
```

This will use ~70GB and train even faster!

---

## 🔍 How It Works

### Training Flow (Single GPU)

1. **Load models** on GPU
   - Actor (training model)
   - Reference model
   - VLLM engine

2. **Generate rollouts**
   - Actor generates responses
   - All on same GPU

3. **Evaluate with judge**
   - Load judge model temporarily
   - Evaluate batch
   - Unload judge model
   - Continue training

4. **Update policy**
   - Compute rewards
   - Calculate gradients
   - Update actor

### Local Reward Function

File: `medical_team/local_reward_function.py`

```python
def reward_func(queries, prompts, labels):
    # Load judge model (cached after first call)
    _load_judge_model()
    
    # Evaluate each query
    for query in queries:
        evaluation = _evaluate_medical_note(query)
        reward = compute_reward(evaluation, label)
        rewards.append(reward)
    
    return rewards
```

**Benefits:**
- ✅ No separate server process
- ✅ No network latency
- ✅ Simpler debugging
- ✅ Better GPU utilization

---

## 📈 Performance

### Expected Training Time

- **638 samples** (1 epoch)
- **~1-2 hours** on RTX PRO 6000 (96GB!)
- **~100 steps** per hour
- **Checkpoints** every 50 steps

### Throughput

- **Rollout**: 64 samples/batch
- **Training**: 16 samples/batch
- **Judge eval**: 64 samples/batch

**Much faster with 96GB VRAM!** 🚀

---

## 🚨 Troubleshooting

### OOM Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch sizes (see above)
2. Reduce GPU memory utilization
3. Enable gradient checkpointing
4. Use smaller models (3B instead of 4B)

### Slow Training

**Check:**
```bash
# GPU utilization should be 80-95%
nvidia-smi

# If low, increase batch sizes slightly
```

### Judge Model Loading

**First batch will be slower** - judge model loads on first use, then cached.

---

## ✅ Verification

Before training:

```bash
# Check GPU
nvidia-smi
# Should show ~48GB total memory

# Check models
ls trainer_output/qwen3-4b-medical-selfplay-sft/
# Should show config.json, model files

# Check data
wc -l data/medical_openrlhf/train.jsonl
# Should show 638

# Check reward function
python medical_team/local_reward_function.py
# Should run test without errors
```

---

## 📊 Monitoring

### GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Expected:
# - Memory: 35-45GB / 48GB
# - Utilization: 80-95%
# - Temperature: <85°C
```

### Training Progress

```bash
# View logs
tail -f checkpoints/medical_selfplay_RL_*/logs/training.log

# Look for:
# - Rewards (both positive and negative)
# - Loss decreasing
# - Checkpoints saving
```

---

## 🎯 Success Indicators

Training is working if:

1. ✅ GPU memory stable at 35-45GB
2. ✅ GPU utilization 80-95%
3. ✅ Rewards computed successfully
4. ✅ Both positive and negative rewards
5. ✅ Checkpoints saving every 50 steps
6. ✅ No OOM errors

---

## 📚 Key Files

- `scripts/train_medical_reinforce.sh` - Training script (single GPU config)
- `medical_team/local_reward_function.py` - Local judge evaluation
- `launch_training.sh` - Automated launcher
- `quick_start.sh` - Initial setup

---

## 🎉 Advantages of Single GPU Setup

- ✅ **Simpler** - No server management
- ✅ **Faster** - No network overhead
- ✅ **Efficient** - Better GPU utilization
- ✅ **Reliable** - Fewer failure points
- ✅ **Debuggable** - All in one process

Perfect for your RTX PRO 6000! 🚀
