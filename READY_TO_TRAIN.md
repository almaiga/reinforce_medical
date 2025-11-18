# ✅ Ready to Train - Single GPU Setup

## 🎉 Great News!

**No judge server needed!** Everything runs on one GPU using OpenRLHF's local reward function feature.

## 📦 What You Have

### ✅ Complete Implementation
1. **Data pipeline** - Creates 2x contrastive pairs
2. **Format conversion** - Medical → OpenRLHF format
3. **Medical components** - Game logic, rewards, prompts
4. **Local reward function** - Judge runs in-process
5. **Training script** - Ready to execute

### ✅ Key Files
- `scripts/create_rl_training_data.py` - Data generation
- `scripts/convert_to_openrlhf_format.py` - Format conversion
- `medical_team/local_reward_function.py` - **Local judge (no server!)**
- `scripts/train_medical_local.sh` - Training script
- `LOCAL_TRAINING_GUIDE.md` - Detailed guide

## 🚀 Quick Start (3 Commands)

### 1. Generate Data
```bash
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py
```

### 2. Setup OpenRLHF
```bash
cd selfplay-redteaming-reference && pip install -e . && cd ..
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team
```

### 3. Train!
```bash
chmod +x scripts/train_medical_local.sh
./scripts/train_medical_local.sh
```

## 🎯 How It Works

```
┌─────────────────────────────────────┐
│         Single GPU (48GB)            │
│                                      │
│  Actor (Qwen 3B)      ~6GB          │
│  Critic (Qwen 3B)     ~6GB          │
│  Judge (MedGemma 4B)  ~8GB          │
│  Activations          ~4GB          │
│  ─────────────────────────          │
│  Total                ~24GB          │
│                                      │
└─────────────────────────────────────┘
```

### Key Innovation:
Instead of `--remote_rm_url http://localhost:8000`, we use:
```bash
--remote_rm_url medical_team/local_reward_function.py
```

OpenRLHF loads the Python file and calls `reward_func()` directly!

## 📊 Training Data

- **Source**: 319 MEDEC error cases
- **Output**: 638 training samples (2x contrastive pairs)
- **Distribution**: 25% each category (perfect balance)
- **Format**: OpenRLHF-compatible JSONL

### Data Flow:
```
MEDEC CSV (319 errors)
    ↓
create_rl_training_data.py
    ↓
train.jsonl (638 samples)
    ↓
convert_to_openrlhf_format.py
    ↓
OpenRLHF format (vanilla/adversarial fields)
    ↓
Training!
```

## 🔧 Configuration

### Models:
- **Actor/Critic**: Qwen/Qwen2.5-3B-Instruct
- **Judge**: google/medgemma-4b-it (loaded in reward_func)

### Batch Sizes:
- `micro_rollout_batch_size`: 2
- `rollout_batch_size`: 32
- `train_batch_size`: 8

### Memory:
- `vllm_gpu_memory_utilization`: 0.5
- `zero_stage`: 2
- `gradient_checkpointing`: enabled

## 📈 What to Expect

### During Training:
```
🏥 Loading medical judge model...
✅ Medical judge loaded on cuda
🎮 Turn 0: 🚀 Generating attacks... 🔥
🎯 Computing rewards for 32 queries...
✅ Computed 32 rewards (avg: 0.15)
🎮 Turn 1: 🛡️ Generating defenses... 🛡️
🎯 Computing rewards for 32 queries...
✅ Computed 32 rewards (avg: -0.12)
Episode 1/1, Step 10/20
  Attacker reward: 0.23
  Assessor reward: -0.18
  Win rate: 52% attacker
```

### Checkpoints:
- Saved every 100 steps
- Location: `checkpoints/medical_selfplay_local_*/`
- Format: HuggingFace compatible

## 🎓 What You're Training

### Attacker Model:
- Learns to introduce realistic medical errors
- Gets reward when errors go undetected
- Improves through adversarial self-play

### Assessor Model:
- Learns to detect medical errors
- Gets reward when errors are caught
- Improves by facing harder attacks

### Result:
Both models improve together through **co-evolution**!

## 🔍 Monitoring

Watch for:
- ✅ Rewards are zero-sum (sum ≈ 0)
- ✅ Win rates balanced (~50/50)
- ✅ Both models receiving rewards
- ✅ No CUDA OOM errors
- ✅ Judge latency < 1s per batch

## 🚨 If You Run Out of Memory

Try these in order:

1. **Smaller model**:
   ```bash
   MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
   ```

2. **Smaller batches**:
   ```bash
   --micro_rollout_batch_size 1
   --rollout_batch_size 16
   ```

3. **Less VLLM memory**:
   ```bash
   --vllm_gpu_memory_utilization 0.3
   ```

4. **8-bit judge** (edit `local_reward_function.py`):
   ```python
   _judge_model = AutoModelForCausalLM.from_pretrained(
       model_name,
       load_in_8bit=True,
       device_map="auto"
   )
   ```

## 📚 Documentation

- `LOCAL_TRAINING_GUIDE.md` - Detailed setup guide
- `TRAINING_READINESS_CHECKLIST.md` - Complete checklist
- `medical_team/README.md` - Component documentation

## ✅ Pre-Flight Checklist

Before running training:

- [ ] Data generated (638 samples)
- [ ] Data converted to OpenRLHF format
- [ ] OpenRLHF installed
- [ ] medical_team copied to red_team
- [ ] Training script created
- [ ] GPU available (24GB+ free)
- [ ] CUDA working

## 🎉 You're Ready!

Everything is set up for **single GPU training** with no external dependencies!

Just run:
```bash
./scripts/train_medical_local.sh
```

**No judge server. No complexity. Just train!** 🚀

---

## 💡 Key Insight

The breakthrough is using OpenRLHF's **local reward function** feature:
- Pass a `.py` file instead of a URL
- OpenRLHF loads and calls `reward_func()` directly
- Judge model runs in the same process
- Everything on one GPU!

This is **much simpler** than the remote server approach! ✨
