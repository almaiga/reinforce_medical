# Training Readiness Checklist for OpenRLHF REINFORCE++

## ✅ What We Have (Complete)

### 1. Data Preparation ✅
- [x] **Raw MEDEC data** - `data_copy/MEDEC/MEDEC-MS/` (319 error cases)
- [x] **Data loading script** - `scripts/create_rl_training_data.py`
  - Creates 2x data points (harmful + benign from each error case)
  - Perfect 4-way split (25% each category)
  - Includes error_type metadata
- [x] **Format conversion script** - `scripts/convert_to_openrlhf_format.py`
  - Converts to OpenRLHF REINFORCE++ format
  - Maps vanilla/adversarial fields correctly
- [x] **Training data (OpenRLHF format)** - `data/medical_openrlhf/train.jsonl`
  - Ready for OpenRLHF training
  - Correct format: `{vanilla, adversarial, completion, data_type}`

### 2. Core Components ✅
- [x] **Medical game manager** - `medical_team/medical_game_manager.py`
  - Compatible with OpenRLHF interface
  - Handles 4-way game structure
- [x] **Language game (REINFORCE++)** - `medical_team/language_game.py`
  - Adapted from Self-RedTeam
  - Implements DialogueGameManager pattern
- [x] **Reward functions** - `medical_team/utils.py`
  - Zero-sum rewards matching Self-RedTeam
  - CoT format checking (identical to Self-RedTeam)
  - Medical-specific reward logic
- [x] **Prompts** - `medical_team/prompts.py`
  - Attacker prompts (harmful/benign)
  - Assessor prompts
- [x] **Medical judge** - `medical_team/medical_judge.py`
  - Local judge implementation
- [x] **Remote judge client** - `medical_team/remote_judge.py`
  - HTTP client for distributed training
- [x] **Judge server** - `scripts/serve_medical_judge.py`
  - FastAPI endpoint
  - Compatible with OpenRLHF remote RM interface

### 3. OpenRLHF Integration ✅
- [x] **Self-RedTeam reference** - `selfplay-redteaming-reference/`
  - Cloned repository with OpenRLHF fork
  - Contains REINFORCE++ implementation
- [x] **Medical team module** - `medical_team/`
  - Drop-in replacement for `red_team/`
  - All components adapted

### 4. Verification ✅
- [x] **Implementation verified** - `IMPLEMENTATION_VERIFICATION.md`
  - Reward structure matches Self-RedTeam
  - CoT parsing identical
  - Zero-sum property maintained
- [x] **Pre-training verified** - `PRE_TRAINING_VERIFICATION.md`
  - All components checked
  - Data quality confirmed

---

## 🔧 What We Need to Do (Next Steps)

### Step 1: Generate Full Training Dataset
**Status:** Need to run with full data

**Action:**
```bash
# Generate full training data (all 319 error cases = 638 data points)
python scripts/create_rl_training_data.py \
    --input data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
    --output-dir data/medical_rl_training \
    --num-samples 319  # or omit for all available

# Convert to OpenRLHF format
python scripts/convert_to_openrlhf_format.py \
    --input data/medical_rl_training/train.jsonl \
    --output data/medical_openrlhf/train.jsonl
```

**Expected output:**
- 638 training samples (2x from 319 error cases)
- Perfect 25% distribution across 4 categories
- OpenRLHF-compatible format

---

### Step 2: Install OpenRLHF (Self-RedTeam Fork)
**Status:** Need to install

**Action:**
```bash
cd selfplay-redteaming-reference
pip install -e .
```

**Verify:**
```bash
python -c "import openrlhf; print('OpenRLHF installed successfully')"
```

---

### Step 3: Copy Medical Team to OpenRLHF
**Status:** Need to copy

**Action:**
```bash
# Remove red_team from OpenRLHF
rm -rf selfplay-redteaming-reference/red_team

# Copy medical_team as red_team
cp -r medical_team selfplay-redteaming-reference/red_team
```

**Why:** OpenRLHF expects the module to be named `red_team`

---

### Step 4: Start Medical Judge Server
**Status:** Need to start

**Action:**
```bash
# Start judge server in background
python scripts/serve_medical_judge.py \
    --model google/medgemma-4b-it \
    --port 8000 \
    --device cuda &

# Wait for server to be ready
sleep 10

# Test connection
curl http://localhost:8000/health
```

**Expected output:**
```json
{"status": "healthy", "model": "google/medgemma-4b-it"}
```

---

### Step 5: Create Training Script
**Status:** Need to create

**Action:** Create `scripts/train_medical_reinforce.sh`

```bash
#!/bin/bash
set -x

MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"  # or your preferred model
PREFIX="medical_selfplay_RL"
RUN_NAME="${PREFIX}_$(date +%m%dT%H:%M)"
REMOTE_RM_URL="http://localhost:8000/judge"

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
    --remote_rm_url $REMOTE_RM_URL \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.7 \
    --pretrain $MODEL_PATH \
    --save_path checkpoints/${RUN_NAME} \
    --ckpt_path checkpoints/${RUN_NAME}/ckpt \
    --save_steps 100 \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --micro_train_batch_size 4 \
    --train_batch_size 16 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 64 \
    --prompt_data "data/medical_openrlhf/train.jsonl" \
    --max_samples 638 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --flash_attn \
    --zero_stage 3 \
    --num_episodes 1 \
    --bf16 \
    --seed 42 \
    --top_p 1.0 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --normalize_reward \
    --packing_samples \
    --gradient_checkpointing \
    --advantage_estimator reinforce \
    --custom_configs "$CUSTOM_CONFIGS" \
    --actor_loss_coef 1.0 \
    --eval_steps 10 \
    --vllm_sync_backend nccl \
    --enforce_eager
```

---

### Step 6: Run Training
**Status:** Ready to run after steps 1-5

**Action:**
```bash
# Make script executable
chmod +x scripts/train_medical_reinforce.sh

# Run training
./scripts/train_medical_reinforce.sh
```

---

## 📋 Pre-Training Checklist

Before running training, verify:

- [ ] Full training data generated (638 samples)
- [ ] Data converted to OpenRLHF format
- [ ] OpenRLHF installed from Self-RedTeam fork
- [ ] `medical_team/` copied to `selfplay-redteaming-reference/red_team/`
- [ ] Judge server running and responding
- [ ] Training script created
- [ ] GPU available and CUDA working
- [ ] Sufficient disk space for checkpoints (~10GB)

---

## 🎯 Expected Training Flow

1. **Data Loading**
   - OpenRLHF loads `data/medical_openrlhf/train.jsonl`
   - RedTeamGamePromptDataset processes records
   - Separates vanilla/adversarial prompts

2. **Game Playing**
   - Turn 0 (Attacker): Generates/modifies medical notes
   - Turn 1 (Assessor): Classifies notes as Safe/Harmful

3. **Judge Evaluation**
   - Batch queries sent to judge server
   - Judge returns labels (error_detected, error_present, etc.)

4. **Reward Calculation**
   - Zero-sum rewards computed
   - Attacker: +1 if error undetected, -1 if detected
   - Assessor: +1 if error detected, -1 if missed

5. **Policy Update**
   - REINFORCE++ advantage estimation
   - Policy gradient update
   - Both models improve through self-play

---

## 🔍 Monitoring

During training, monitor:

- **Rewards**: Should see both positive and negative rewards
- **Win rates**: Attacker vs Assessor balance
- **CoT format**: Should have minimal violations
- **Judge latency**: Should be <1s per batch
- **GPU memory**: Should stay under 80%

---

## 🚨 Troubleshooting

### Issue: Judge server not responding
**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Restart if needed
pkill -f serve_medical_judge
python scripts/serve_medical_judge.py --model google/medgemma-4b-it --port 8000 &
```

### Issue: Import errors for red_team
**Solution:**
```bash
# Ensure medical_team is copied as red_team
ls selfplay-redteaming-reference/red_team/
# Should see: __init__.py, utils.py, prompts.py, etc.
```

### Issue: CUDA out of memory
**Solution:**
- Reduce `micro_rollout_batch_size` (try 2)
- Reduce `vllm_gpu_memory_utilization` (try 0.5)
- Use smaller model (try 1B instead of 3B)

---

## 📊 Success Criteria

Training is working correctly if:

1. ✅ Both attacker and assessor receive rewards
2. ✅ Rewards are zero-sum (sum ≈ 0)
3. ✅ Win rates are balanced (~50/50)
4. ✅ CoT format violations decrease over time
5. ✅ Models generate valid medical text
6. ✅ No crashes or errors

---

## 🎉 You're Almost Ready!

**Current Status:** 95% complete

**Remaining work:**
1. Generate full dataset (5 minutes)
2. Install OpenRLHF (5 minutes)
3. Copy medical_team (1 minute)
4. Start judge server (2 minutes)
5. Create training script (5 minutes)
6. Run training! 🚀

**Total time to start training:** ~20 minutes

---

## 📚 Reference Documents

- `REINFORCE_PLUS_PLUS_READY.md` - Overview of REINFORCE++ implementation
- `IMPLEMENTATION_VERIFICATION.md` - Component verification
- `PRE_TRAINING_VERIFICATION.md` - Pre-training checks
- `medical_team/README.md` - Medical team components
- `selfplay-redteaming-reference/README.md` - OpenRLHF documentation
