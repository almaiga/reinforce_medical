#!/bin/bash
set -x

# Medical Self-Play REINFORCE++ Training Script
# Uses fine-tuned Qwen3-4B model as starting point
# Local judge evaluation (no separate server needed)

# Model path (your fine-tuned model on SSH server)
MODEL_PATH="trainer_output/qwen3-4b-medical-selfplay-sft"

echo "Using model: $MODEL_PATH"

# Training configuration
PREFIX="medical_selfplay_RL"
RUN_NAME="${PREFIX}_$(date +%m%d_%H%M)"

# Use local reward function (no remote server)
REWARD_FUNCTION="medical_team/local_reward_function.py"

# Custom configs for medical domain
CUSTOM_CONFIGS='{
    "max_turns": 2,
    "reward_type": "medical_general_sum",
    "remove_ties": true,
    "error_types": ["dosage", "diagnosis", "contraindication", "management", "causalOrganism"],
    "judge_model": "google/medgemma-4b-it"
}'

# Check if data exists
if [ ! -f "data/medical_openrlhf/train.jsonl" ]; then
    echo "❌ Training data not found!"
    echo "Generate it with:"
    echo "  python scripts/create_rl_training_data.py"
    echo "  python scripts/convert_to_openrlhf_format.py"
    exit 1
fi
echo "✅ Training data found"

# Run REINFORCE++ training with local judge
python -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --remote_rm_url $REWARD_FUNCTION \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.75 \
    --pretrain $MODEL_PATH \
    --save_path checkpoints/${RUN_NAME} \
    --ckpt_path checkpoints/${RUN_NAME}/ckpt \
    --save_steps 50 \
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
