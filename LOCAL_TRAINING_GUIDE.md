# Local Training Guide - Single GPU Setup

## ✅ No Judge Server Needed!

You can run everything on a **single GPU** using OpenRLHF's local reward function feature.

## 🎯 How It Works

Instead of running a separate judge server, OpenRLHF can load a **Python file** that contains a `reward_func()` function. This function is called directly during training to compute rewards.

### Architecture:

```
┌─────────────────────────────────────┐
│         Single GPU                   │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Actor Model (Qwen 3B)         │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Critic Model (Qwen 3B)        │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Judge Model (MedGemma 4B)     │ │
│  │  (loaded in reward_func)       │ │
│  └────────────────────────────────┘ │
│                                      │
└─────────────────────────────────────┘
```

## 📝 Setup Steps

### 1. Generate Training Data
```bash
# Generate full dataset (638 samples from 319 error cases)
python scripts/create_rl_training_data.py

# Convert to OpenRLHF format
python scripts/convert_to_openrlhf_format.py
```

### 2. Install OpenRLHF
```bash
cd selfplay-redteaming-reference
pip install -e .
cd ..
```

### 3. Copy Medical Team
```bash
# OpenRLHF expects module named 'red_team'
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team
```

### 4. Create Training Script

Create `scripts/train_medical_local.sh`:

```bash
#!/bin/bash
set -x

MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
PREFIX="medical_selfplay_local"
RUN_NAME="${PREFIX}_$(date +%m%dT%H:%M)"

# Use local reward function instead of remote server!
REWARD_FUNC="medical_team/local_reward_function.py"

# Custom configs for medical domain
CUSTOM_CONFIGS='{
    "max_turns": 2,
    "reward_type": "medical_general_sum",
    "remove_ties": true,
    "error_types": ["dosage", "diagnosis", "contraindication", "management"]
}'

python -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --remote_rm_url $REWARD_FUNC \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.5 \
    --pretrain $MODEL_PATH \
    --save_path checkpoints/${RUN_NAME} \
    --ckpt_path checkpoints/${RUN_NAME}/ckpt \
    --save_steps 100 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --micro_train_batch_size 2 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 32 \
    --prompt_data "data/medical_openrlhf/train.jsonl" \
    --max_samples 638 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --flash_attn \
    --zero_stage 2 \
    --num_episodes 1 \
    --bf16 \
    --seed 42 \
    --top_p 1.0 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --normalize_reward \
    --gradient_checkpointing \
    --advantage_estimator reinforce \
    --custom_configs "$CUSTOM_CONFIGS" \
    --actor_loss_coef 1.0 \
    --eval_steps 10
```

### 5. Run Training
```bash
chmod +x scripts/train_medical_local.sh
./scripts/train_medical_local.sh
```

## 🔧 Key Differences from Remote Setup

| Aspect | Remote Server | Local Function |
|--------|--------------|----------------|
| Judge Server | ✅ Required | ❌ Not needed |
| Setup | Start server separately | Just pass .py file |
| GPU Usage | Separate GPU | Same GPU |
| Latency | Network overhead | Direct function call |
| Debugging | Harder | Easier |
| Complexity | Higher | Lower |

## 📊 Memory Considerations

On a single GPU, you'll have:
- **Actor model**: ~6GB (Qwen 3B)
- **Critic model**: ~6GB (Qwen 3B)
- **Judge model**: ~8GB (MedGemma 4B)
- **Activations**: ~4GB
- **Total**: ~24GB

**Recommended GPU**: RTX 6000 (48GB) or A100 (40GB)

### Memory Optimization Tips:

1. **Use smaller models**:
   ```bash
   MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"  # Smaller actor/critic
   ```

2. **Reduce batch sizes**:
   ```bash
   --micro_rollout_batch_size 1
   --rollout_batch_size 16
   ```

3. **Use gradient checkpointing**:
   ```bash
   --gradient_checkpointing  # Already included
   ```

4. **Reduce memory utilization**:
   ```bash
   --vllm_gpu_memory_utilization 0.3  # Lower from 0.5
   ```

5. **Use 8-bit quantization** (if needed):
   ```python
   # In local_reward_function.py, modify model loading:
   _judge_model = AutoModelForCausalLM.from_pretrained(
       model_name,
       load_in_8bit=True,  # Add this
       device_map="auto"
   )
   ```

## 🎯 Advantages of Local Setup

1. **Simpler**: No need to manage separate server
2. **Faster**: No network latency
3. **Easier debugging**: All in one process
4. **Better for development**: Quick iterations
5. **Single GPU**: Everything on one device

## 🔍 How the Local Reward Function Works

The `medical_team/local_reward_function.py` file contains:

```python
def reward_func(queries, prompts, labels):
    """
    Args:
        queries: Generated responses from models
        prompts: Input prompts
        labels: Game types (vanilla_harmful, etc.)
    
    Returns:
        List of reward values
    """
    # Load judge model (once)
    _load_judge_model()
    
    # Evaluate each query
    for query, prompt, label in zip(queries, prompts, labels):
        # Extract medical note from CoT format
        # Evaluate with judge model
        # Compute reward based on game type
        ...
    
    return rewards
```

OpenRLHF automatically:
1. Loads this file at startup
2. Calls `reward_func()` during training
3. Uses returned rewards for policy updates

## 🚀 Quick Start (5 Steps)

```bash
# 1. Generate data
python scripts/create_rl_training_data.py
python scripts/convert_to_openrlhf_format.py

# 2. Install OpenRLHF
cd selfplay-redteaming-reference && pip install -e . && cd ..

# 3. Copy medical_team
rm -rf selfplay-redteaming-reference/red_team
cp -r medical_team selfplay-redteaming-reference/red_team

# 4. Create training script (see above)
nano scripts/train_medical_local.sh

# 5. Run!
chmod +x scripts/train_medical_local.sh
./scripts/train_medical_local.sh
```

## 📈 Expected Output

```
🏥 Loading medical judge model...
✅ Medical judge loaded on cuda
🎮 Turn 0: 🚀 Generating attacks... 🔥
🎯 Computing rewards for 32 queries...
✅ Computed 32 rewards (avg: 0.15)
🎮 Turn 1: 🛡️ Generating defenses... 🛡️
🎯 Computing rewards for 32 queries...
✅ Computed 32 rewards (avg: -0.12)
...
```

## ✅ You're Ready!

No judge server needed - everything runs on one GPU! 🎉
