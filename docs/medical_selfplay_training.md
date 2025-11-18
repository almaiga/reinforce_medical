# Medical Self-Play Training with OpenRLHF

This document explains how to use the medical self-play training system that adapts OpenRLHF for medical error detection.

## Overview

The medical self-play system trains two competing models:
- **Attacker**: Introduces or modifies medical errors in clinical notes
- **Assessor**: Detects and classifies medical errors as Safe/Harmful

The system uses a 4-way game structure:
1. **vanilla_benign**: Copy clean notes as-is → Assessor should say "Safe"
2. **vanilla_harmful**: Copy error notes as-is → Assessor should say "Harmful"  
3. **adversarial_benign**: Make benign changes to clean notes → Assessor should say "Safe"
4. **adversarial_harmful**: Modify/worsen error notes → Assessor should say "Harmful"

## Quick Start

### 1. Prepare Dataset

First, create a medical dataset in the 4-way format:

```bash
python3 scripts/convert_medical_dataset.py --create_sample --sample_size 50 --output_dir data/medical_selfplay
```

### 2. Run Training

Use the launcher script for easy training:

```bash
./scripts/run_medical_selfplay.sh
```

Or with custom parameters:

```bash
./scripts/run_medical_selfplay.sh \
    --pretrain "Qwen/Qwen2.5-3B-Instruct" \
    --dataset "data/medical_selfplay/train.jsonl" \
    --batch_size 16 \
    --max_len 2048
```

### 3. Monitor Training

The training will output logs showing:
- Game outcomes and rewards
- Medical judge evaluations
- Model performance metrics

## Components

### Medical DialogueGameManager

Orchestrates the 4-way medical self-play games:

```python
from medical_team.medical_game_manager import MedicalDialogueGameManager

game_manager = MedicalDialogueGameManager(
    tokenizer=tokenizer,
    medical_judge_fn=judge_function,
    strategy=strategy,
    custom_configs={
        "max_turns": 2,
        "reward_type": "medical_general_sum",
        "error_types": ["dosage", "diagnosis", "contraindication"]
    }
)
```

### Medical Judge Model

Evaluates medical notes for errors using MedGemma:

```python
from medical_team.medical_judge import load_medical_judge_model, create_medical_judge_remote_function

judge_model, judge_tokenizer, device = load_medical_judge_model("google/medgemma-4b-it")
judge_fn = create_medical_judge_remote_function(judge_model, judge_tokenizer, device)
```

### Medical Prompts

Simple, clean prompts optimized for Qwen3:

- **Attacker (harmful)**: "Create an error variation of this seed note"
- **Attacker (benign)**: "Verify this medical note is safe and output it unchanged"  
- **Assessor**: "Classify this medical note for errors: Safe/Harmful"

## Configuration

### Custom Configs

```json
{
  "max_turns": 2,
  "reward_type": "medical_general_sum",
  "error_types": ["dosage", "diagnosis", "contraindication", "drug_interaction"],
  "direct_chat_no_cot": false,
  "no_attacker_turn": false,
  "no_assessor_turn": false
}
```

### Training Parameters

```json
{
  "pretrain": "Qwen/Qwen2.5-3B-Instruct",
  "medical_judge_model": "google/medgemma-4b-it",
  "rollout_batch_size": 16,
  "max_len": 2048,
  "init_kl_coef": 0.01
}
```

## Dataset Format

The medical dataset should be in JSONL format with 4-way game structure:

```json
{"prompt": "Patient with diabetes. Prescribed metformin 10mg daily.", "completion": "Appropriate diabetes treatment.", "data_type": "vanilla_benign", "error_present": false}
{"prompt": "Patient with diabetes. Prescribed metformin 2000mg twice daily.", "completion": "Dosage error detected.", "data_type": "vanilla_harmful", "error_present": true, "error_type": "dosage"}
{"prompt": "Patient with diabetes. Prescribed metformin 10mg daily.", "completion": "Appropriate diabetes treatment.", "data_type": "adversarial_benign", "error_present": false}
{"prompt": "Patient with diabetes. Prescribed metformin 2000mg twice daily.", "completion": "Dosage error detected.", "data_type": "adversarial_harmful", "error_present": true, "error_type": "dosage"}
```

## Testing

### Test Components

```bash
python3 scripts/test_medical_training_simple.py
```

### Test Integration

```bash
python3 scripts/test_medical_integration.py
```

### Test Judge Model

```bash
python3 scripts/test_medical_judge.py
```

## Requirements

### Core Dependencies

- `transformers>=4.30.0`
- `torch>=2.0.0`
- `datasets>=2.0.0`
- `ray>=2.0.0`

### OpenRLHF Dependencies (for full training)

- `flash_attn>=2.0.0`
- `deepspeed>=0.9.0`
- `vllm>=0.2.0`

### Medical Judge Model

- `google/medgemma-4b-it` (default)
- Or any compatible medical language model

## Troubleshooting

### Common Issues

1. **Import Error: flash_attn**
   - Install: `pip install flash-attn`
   - Or use CPU-only mode for testing

2. **Dataset Not Found**
   - Run dataset conversion script first
   - Check dataset path in configuration

3. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Reduce max length: `--max_len 1024`
   - Use gradient checkpointing

4. **Judge Model Loading Failed**
   - Check internet connection for model download
   - Verify model name is correct
   - Use smaller judge model for testing

### Performance Tips

1. **Use vLLM for Generation**
   - Set `--vllm_num_engines 1` for faster generation
   - Enable prefix caching: `--enable_prefix_caching`

2. **Optimize Batch Size**
   - Start with small batch size (8-16)
   - Increase gradually based on GPU memory

3. **Model Colocation**
   - Use `--colocate_all_models` to save GPU memory
   - Trade memory for slightly slower training

## Advanced Usage

### Custom Medical Judge

```python
def custom_medical_judge_fn(url, batch_queries, score_key="labels"):
    # Your custom medical evaluation logic
    results = {}
    for query in batch_queries:
        # Evaluate medical note
        result = evaluate_medical_note(query["medical_note"])
        results[query["game_idx"]] = result
    return results
```

### Custom Reward Function

```python
def custom_medical_reward(gamedata, labels, reward_coeff_config):
    # Your custom reward computation
    role = gamedata["game_role"]
    error_detected = labels["error_detected"]
    error_present = labels["error_present"]
    
    if role == "attacker":
        reward = 1.0 if not error_detected and error_present else -1.0
    else:  # assessor
        reward = 1.0 if error_detected == error_present else -1.0
    
    return reward, set()
```

## References

- [OpenRLHF Documentation](https://github.com/OpenRLHF/OpenRLHF)
- [Self-RedTeam Paper](https://arxiv.org/abs/2506.07468)
- [MedGemma Model](https://huggingface.co/google/medgemma-4b-it)
- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)