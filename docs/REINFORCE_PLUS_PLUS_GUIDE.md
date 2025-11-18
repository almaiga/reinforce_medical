# Using REINFORCE++ for Medical Self-Play

## 🎯 What is REINFORCE++?

REINFORCE++ is the **key algorithm** from the Self-RedTeam paper. It's a policy gradient method that:
- Uses Monte Carlo returns (no value function needed)
- Computes advantages from actual game rewards
- Simpler than PPO (no clipping, no critic network)
- Perfect for self-play adversarial games

## ✅ What You Now Have

I've created `medical_team/language_game.py` which is a **direct adaptation** of Self-RedTeam's `DialogueGameManager` for medical error detection.

This is the **correct implementation** that works with OpenRLHF's REINFORCE++ trainer!

## 🚀 How to Use REINFORCE++

### Option 1: Use Self-RedTeam's OpenRLHF Fork (Recommended)

Since Self-RedTeam has modifications to OpenRLHF, you should use their fork:

```bash
# 1. Clone Self-RedTeam repo (already done - in selfplay-redteaming-reference/)
cd selfplay-redteaming-reference

# 2. Install their OpenRLHF
pip install -e .

# 3. Replace red_team/ with your medical_team/
rm -rf red_team/
cp -r ../medical_team ./

# 4. Start medical judge server
cd ..
python scripts/serve_medical_judge.py --model google/medgemma-4b-it --port 8000 &

# 5. Prepare data in OpenRLHF format
python scripts/prepare_medical_data.py --num-samples 400 --output-dir data/medical_openrlhf

# 6. Run REINFORCE++ training
cd selfplay-redteaming-reference
python -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --remote_rm_url "http://localhost:8000/judge" \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --pretrain "Qwen/Qwen2.5-3B-Instruct" \
    --save_path checkpoints/medical_selfplay \
    --micro_train_batch_size 8 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 128 \
    --prompt_data "../data/medical_openrlhf/train.jsonl" \
    --max_samples 40000 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --advantage_estimator reinforce \
    --custom_configs '{"max_turns":2,"reward_type":"medical_general_sum","error_types":["dosage","diagnosis","contraindication"]}' \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --normalize_reward \
    --bf16
```

### Option 2: Install OpenRLHF and Adapt

If you want to use the official OpenRLHF:

```bash
# 1. Install OpenRLHF
pip install openrlhf

# 2. You'll need to modify their code to use medical_team/
# This is more complex - Option 1 is recommended
```

## 📁 File Structure for REINFORCE++

```
your-project/
├── medical_team/
│   ├── __init__.py              # MedicalGameOutcome enum
│   ├── language_game.py         # ✅ NEW: MedicalDialogueGameManager for REINFORCE++
│   ├── utils.py                 # Reward functions
│   ├── prompts.py               # Medical prompts
│   └── remote_judge.py          # Judge client
│
├── selfplay-redteaming-reference/  # Self-RedTeam's OpenRLHF fork
│   ├── openrlhf/
│   │   ├── cli/train_ppo_ray.py    # Training script with REINFORCE++
│   │   └── trainer/
│   │       └── ppo_utils/
│   │           └── experience_maker.py  # Uses DialogueGameManager
│   └── red_team/  # ← Replace this with medical_team/
│
├── data/medical_openrlhf/
│   ├── train.jsonl              # Training data
│   └── val.jsonl                # Validation data
│
└── scripts/
    ├── serve_medical_judge.py   # Medical judge server
    └── prepare_medical_data.py  # Data preparation
```

## 🔑 Key Differences: REINFORCE++ vs GRPO

| Aspect | REINFORCE++ (Self-RedTeam) | GRPO (Your old script/) |
|--------|----------------------------|-------------------------|
| Algorithm | Policy gradient with MC returns | Group relative optimization |
| Value Function | None (Monte Carlo) | Implicit baseline |
| Infrastructure | OpenRLHF + Ray + vLLM | TRL + HuggingFace |
| Complexity | High (distributed) | Low (single GPU) |
| Paper | Self-RedTeam | TRL library |
| Best For | Multi-GPU clusters | Single GPU |

## 📊 Data Format for REINFORCE++

Your data needs to be in JSONL format with these fields:

```json
{
  "prompt": "medical_note_content",
  "prompt_type": "vanilla_harmful" | "adversarial_harmful" | "vanilla_benign" | "adversarial_benign",
  "completion": "expected_assessment"
}
```

The `prepare_medical_data.py` script handles this conversion.

## 🎮 How REINFORCE++ Training Works

```
1. Initialize Ray cluster
   ↓
2. Load medical notes dataset
   ↓
3. For each episode:
   ├─ Turn 0: Attacker introduces/modifies errors
   ├─ Turn 1: Assessor detects and classifies
   ├─ Judge evaluates correctness
   ├─ Compute rewards (zero-sum)
   ├─ Calculate advantages (REINFORCE++)
   └─ Update policies with policy gradient
   ↓
4. Repeat until convergence
```

## 🔧 Configuration

### Custom Configs for Medical Domain

```python
custom_configs = {
    "max_turns": 2,                    # Two-turn games
    "reward_type": "medical_general_sum",  # Medical reward function
    "error_types": [                   # Medical error types
        "dosage",
        "diagnosis",
        "contraindication",
        "drug_interaction"
    ],
    "direct_chat_no_cot": False,       # Enable CoT reasoning
    "no_attacker_turn": False,         # Set True for assessor-only training
    "no_assessor_turn": False          # Set True for attacker-only training
}
```

### Training Parameters

```bash
--advantage_estimator reinforce      # ← This enables REINFORCE++
--actor_learning_rate 5e-7           # Learning rate
--init_kl_coef 0.01                  # KL divergence coefficient
--normalize_reward                   # Normalize rewards
--rollout_batch_size 128             # Batch size for rollouts
--train_batch_size 32                # Batch size for training
```

## 🧪 Testing REINFORCE++ Setup

Before full training, test with small dataset:

```bash
# 1. Prepare small test dataset
python scripts/prepare_medical_data.py --num-samples 40 --output-dir data/medical_test

# 2. Start judge
python scripts/serve_medical_judge.py --model google/medgemma-4b-it --port 8000 &

# 3. Test training (1 episode)
cd selfplay-redteaming-reference
python -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --remote_rm_url "http://localhost:8000/judge" \
    --vllm_num_engines 1 \
    --pretrain "Qwen/Qwen2.5-3B-Instruct" \
    --prompt_data "../data/medical_test/train.jsonl" \
    --rollout_batch_size 32 \
    --max_samples 100 \
    --advantage_estimator reinforce \
    --custom_configs '{"max_turns":2,"reward_type":"medical_general_sum"}' \
    --bf16
```

## 📝 What Changed from Your Old Implementation

### Before (script/selfplay/ - GRPO):
```python
# Used TRL's GRPOTrainer
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_fn=reward_fn,
    # ... GRPO-specific args
)
```

### Now (medical_team/ - REINFORCE++):
```python
# Uses OpenRLHF's train_ppo_ray with REINFORCE++
# MedicalDialogueGameManager orchestrates games
# REINFORCE++ computes advantages from MC returns
# Ray handles distributed training
```

## 🎯 Why REINFORCE++ for Self-Play?

1. **No value function needed** - Simpler for adversarial games
2. **Monte Carlo returns** - Uses actual game outcomes
3. **Zero-sum games** - Perfect for attacker vs assessor
4. **Paper's algorithm** - Exactly what Self-RedTeam uses
5. **Proven results** - +65.5% on WildJailBreak benchmark

## 🚨 Important Notes

1. **You need Self-RedTeam's OpenRLHF fork** - They have custom modifications
2. **Replace `red_team/` with `medical_team/`** - Direct substitution
3. **Use `language_game.py`** - This is the correct DialogueGameManager
4. **Start judge server first** - Required for reward computation
5. **Data format matters** - Use `prepare_medical_data.py`

## 📚 References

- Self-RedTeam Paper: https://arxiv.org/abs/2506.07468
- Self-RedTeam Code: https://github.com/mickelliu/selfplay-redteaming
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- REINFORCE Algorithm: Williams, 1992

## ✅ Summary

You now have:
1. ✅ `medical_team/language_game.py` - Correct DialogueGameManager for REINFORCE++
2. ✅ `medical_team/utils.py` - Medical reward functions
3. ✅ `medical_team/prompts.py` - Medical prompts
4. ✅ `scripts/serve_medical_judge.py` - Judge server
5. ✅ `scripts/prepare_medical_data.py` - Data preparation

**To use REINFORCE++:**
- Use Self-RedTeam's OpenRLHF fork
- Replace `red_team/` with `medical_team/`
- Run `train_ppo_ray.py` with `--advantage_estimator reinforce`

**This is the correct way to implement the Self-RedTeam paper!** 🚀
