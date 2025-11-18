# Medical Self-Play Training

This directory contains scripts for medical self-play training, adapted from the [Self-RedTeam](https://github.com/mickelliu/selfplay-redteaming) approach.

## Overview

The Self-RedTeam paper uses **online self-play reinforcement learning** where:
- **Attacker**: Introduces medical errors into clinical notes
- **Assessor**: Detects and classifies medical errors
- **Judge**: Evaluates whether errors were correctly identified

Both agents co-evolve through continuous interaction, following a **zero-sum game** structure.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Self-Play Loop                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Load Data (MEDEC)                                   │
│     ↓                                                    │
│  2. Attacker Turn: Generate/modify medical notes        │
│     ↓                                                    │
│  3. Assessor Turn: Classify notes (Safe/Harmful)        │
│     ↓                                                    │
│  4. Judge Evaluation: Determine ground truth            │
│     ↓                                                    │
│  5. Compute Rewards (zero-sum)                          │
│     ↓                                                    │
│  6. Update Models (REINFORCE++)                         │
│     ↓                                                    │
│  7. Repeat                                              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Scripts

### 1. `prepare_medical_data.py`
Prepares MEDEC data for training.

```bash
python scripts/prepare_medical_data.py \
    --num-samples 400 \
    --output-dir data/medical_test_4way \
    --train-ratio 0.9
```

**Output:**
- `data/medical_test_4way/train.jsonl` - Training data
- `data/medical_test_4way/val.jsonl` - Validation data

### 2. `serve_medical_judge.py`
Starts the medical judge HTTP server.

```bash
python scripts/serve_medical_judge.py \
    --model google/medgemma-4b-it \
    --port 8000 \
    --device cuda
```

**Endpoints:**
- `GET /health` - Health check
- `POST /judge` - Batch evaluation (OpenRLHF compatible)
- `POST /evaluate` - Single evaluation (testing)

### 3. `train_medical_selfplay_simple.py`
Simple self-play demonstration (single GPU).

```bash
# Start judge server first
python scripts/serve_medical_judge.py --model google/medgemma-4b-it &

# Run self-play
python scripts/train_medical_selfplay_simple.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --judge-url http://localhost:8000 \
    --num-samples 40 \
    --num-episodes 3
```

**Note:** This is a demonstration. For actual training, use:
- Your existing `script/selfplay/main.py` (TRL-based)
- Or adapt OpenRLHF's `train_ppo_ray.py` (distributed)

## Training Approaches

### Option A: TRL-based (Current - Recommended for Single GPU)

Your existing `script/selfplay/main.py` already implements self-play with TRL's GRPOTrainer.

**Pros:**
- ✅ Simpler code
- ✅ Works on single GPU
- ✅ Already tested and working
- ✅ Good for 4B models on RTX 6000

**Cons:**
- ❌ Limited to single GPU
- ❌ No distributed training

**Usage:**
```bash
python script/train_selfplay_advanced.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --judge_model_id google/medgemma-4b-it \
    --num_samples 400 \
    --max_rounds 10
```

### Option B: OpenRLHF Ray-based (For Distributed Training)

Adapt OpenRLHF's `train_ppo_ray.py` for medical domain.

**Pros:**
- ✅ Distributed training across multiple GPUs/nodes
- ✅ Better scaling for large models
- ✅ REINFORCE++ algorithm
- ✅ Production-ready infrastructure

**Cons:**
- ❌ More complex setup
- ❌ Requires Ray cluster
- ❌ Overkill for single GPU

**Setup:**
```bash
# 1. Start Ray cluster
ray start --head

# 2. Start judge server
python scripts/serve_medical_judge.py --port 8000 &

# 3. Run training (would need adaptation)
# python train_medical_ppo_ray.py --config configs/medical_selfplay_config.json
```

## Self-RedTeam Key Concepts

### 1. Zero-Sum Game Structure

**Attacker Rewards:**
- `RA_res_harm`: Opposite of assessor (zero-sum)
- `RA_res_refusal`: Opposite of assessor (zero-sum)
- `RA_revision`: Faithfulness to game category
- `RA_format`: CoT formatting

**Assessor Rewards:**
- `RD_res_harm`: Correct classification
- `RD_res_refusal`: Appropriate refusal behavior
- `RD_format`: CoT formatting

**Zero-sum property:**
```
RA_res_harm + RD_res_harm = 0
RA_res_refusal + RD_res_refusal = 0
```

### 2. 4-Way Game Structure

| Game Type | Source | Task | Difficulty |
|-----------|--------|------|------------|
| vanilla_harmful | Error note | Copy as-is | EASY |
| adversarial_harmful | Error note | Modify/worsen | HARD |
| vanilla_benign | Clean note | Copy as-is | EASY |
| adversarial_benign | Clean note | Inject error | HARD |

This balances:
- 50% easy vs 50% hard tasks
- 50% harmful vs 50% benign cases

### 3. REINFORCE++ Algorithm

The Self-RedTeam paper uses REINFORCE++ (policy gradient with advantages):

```python
# Simplified pseudocode
for episode in episodes:
    # 1. Collect experiences
    experiences = play_games(attacker, assessor)
    
    # 2. Evaluate with judge
    rewards = judge.evaluate(experiences)
    
    # 3. Compute advantages
    advantages = compute_advantages(rewards)
    
    # 4. Update policies
    attacker.update(advantages)
    assessor.update(advantages)
```

## Integration with Your Code

Your existing code already implements most of this! The key components:

### From `script/selfplay/`:
- ✅ `data.py` - 4-way game structure
- ✅ `prompts.py` - Attacker/assessor prompts
- ✅ `rewards.py` - Zero-sum reward structure
- ✅ `judge.py` - Judge evaluation
- ✅ `main.py` - Complete training loop

### From `medical_team/`:
- ✅ `MedicalDialogueGameManager` - Game orchestration
- ✅ `utils.py` - Reward functions
- ✅ `remote_judge.py` - Judge client

## Recommendations

### For Your Setup (Single RTX 6000, 4B Models):

1. **Use your existing TRL approach** (`script/selfplay/main.py`)
   - It already implements Self-RedTeam's core ideas
   - Simpler and works well for single GPU
   - No need for Ray complexity

2. **Add remote judge** (optional)
   - Run judge on same GPU but separate process
   - Prevents memory conflicts
   - Use `scripts/serve_medical_judge.py`

3. **Skip full OpenRLHF Ray** (for now)
   - Unnecessary complexity for single GPU
   - Consider only if scaling to multi-GPU cluster

### If You Want Full OpenRLHF Integration:

You would need to:
1. Clone Self-RedTeam repo
2. Adapt their `train_ppo_ray.py`
3. Replace `DialogueGameManager` with `MedicalDialogueGameManager`
4. Replace WildGuard with medical judge
5. Update reward functions

But honestly, your TRL approach is cleaner for single GPU!

## References

- **Self-RedTeam Paper**: https://arxiv.org/abs/2506.07468
- **Self-RedTeam Code**: https://github.com/mickelliu/selfplay-redteaming
- **OpenRLHF**: https://github.com/OpenRLHF/OpenRLHF
- **TRL**: https://github.com/huggingface/trl

## Next Steps

1. **Test the simple script:**
   ```bash
   python scripts/train_medical_selfplay_simple.py --num-samples 20
   ```

2. **Use your existing training:**
   ```bash
   python script/train_selfplay_advanced.py --num_samples 400
   ```

3. **Monitor training:**
   - Check reward trends
   - Verify zero-sum property
   - Track error detection accuracy

4. **Scale up:**
   - Increase samples
   - More training rounds
   - Larger models (if needed)
