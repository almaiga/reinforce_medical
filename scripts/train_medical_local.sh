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
