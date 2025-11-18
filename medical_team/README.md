# Medical Team - OpenRLHF Adaptation

This module adapts the Self-RedTeam OpenRLHF implementation for medical error detection self-play training.

## Overview

The medical team module provides OpenRLHF-compatible components for training models to:
- **Attacker**: Introduce realistic medical errors into clinical notes
- **Assessor**: Detect and classify medical errors

This is adapted from the Self-RedTeam paper (arXiv:2506.07468) which uses adversarial self-play for LM safety alignment.

## Components

### Core Classes

1. **`MedicalGameOutcome`** (`__init__.py`)
   - Enum defining all possible game outcomes
   - Used for reward calculation and tracking

2. **`MedicalDialogueGameManager`** (`medical_game_manager.py`)
   - Orchestrates two-turn medical self-play games
   - Compatible with OpenRLHF's distributed training
   - Implements 4-way game structure:
     - `vanilla_harmful`: Copy error note as-is (EASY)
     - `adversarial_harmful`: Modify/worsen error (HARD)
     - `vanilla_benign`: Copy clean note as-is (EASY)
     - `adversarial_benign`: Inject error into clean note (HARD)

3. **Medical Utilities** (`utils.py`)
   - Reward calculation functions
   - CoT format parsing
   - Prompt generation helpers
   - Adapted from red_team/utils.py

4. **Medical Prompts** (`prompts.py`)
   - System prompts for attacker and assessor roles
   - CoT formatting templates
   - Medical-specific instructions

5. **Medical Judge** (`medical_judge.py`)
   - Judge model for evaluating medical responses
   - Replaces WildGuard with medical-specific evaluation
   - Supports both local and remote judge models

6. **OpenRLHF Adapter** (`openrlhf_adapter.py`)
   - Bridge between script/selfplay and OpenRLHF interfaces
   - Provides expected function signatures for OpenRLHF

## 4-Way Game Structure

The medical self-play uses a 4-way game structure to balance difficulty:

| Game Type | Source | Task | Difficulty |
|-----------|--------|------|------------|
| vanilla_harmful | Error note | Copy as-is | EASY |
| adversarial_harmful | Error note | Modify/worsen error | HARD |
| vanilla_benign | Clean note | Copy as-is | EASY |
| adversarial_benign | Clean note | Inject error | HARD |

This ensures:
- 50% easy tasks (vanilla) vs 50% hard tasks (adversarial)
- 50% harmful (error) vs 50% benign (clean)
- Balanced training for both attacker and assessor

## Reward Structure

Following the Self-RedTeam paper's reward structure:

### Attacker Rewards
- `RA_res_harm`: Response harmfulness (zero-sum with assessor)
- `RA_res_refusal`: Response refusal (zero-sum with assessor)
- `RA_revision`: Revision faithfulness (respects game category)
- `RA_format`: CoT formatting compliance

### Assessor Rewards
- `RD_res_harm`: Response harmfulness (zero-sum with attacker)
- `RD_res_refusal`: Response refusal (zero-sum with attacker)
- `RD_format`: CoT formatting compliance

## Usage

### Basic Usage

```python
from medical_team import MedicalDialogueGameManager, MedicalGameOutcome
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Initialize game manager
game_manager = MedicalDialogueGameManager(
    tokenizer=tokenizer,
    medical_judge_fn=medical_judge_function,
    strategy=ray_strategy,
    custom_configs={
        "max_turns": 2,
        "reward_type": "medical_general_sum",
        "error_types": ["dosage", "diagnosis", "contraindication"],
        "direct_chat_no_cot": False
    }
)

# Initialize games
game_manager.initialize_games(
    medical_notes=["Patient note 1", "Patient note 2"],
    completions=["Expected assessment 1", "Expected assessment 2"],
    data_types=["vanilla_harmful", "adversarial_benign"]
)

# Play games
results = game_manager.play_games(
    attacker_llm_generator=attacker_generator,
    assessor_llm_generator=assessor_generator
)

# Evaluate outcomes
batch_labels = game_manager.evaluate_game_outcomes()

# Compute rewards
attacker_outputs, attacker_states, assessor_outputs, assessor_states = \
    game_manager.filter_and_compute_rewards(batch_labels)
```

### Integration with OpenRLHF

The `MedicalDialogueGameManager` is designed to work with OpenRLHF's `train_ppo_ray.py`:

```python
# In your training script
from medical_team import MedicalDialogueGameManager
from medical_team.utils import (
    get_medical_game_reward_general_sum,
    convert_medical_game_history_to_messages,
    medical_cot_format_check_and_extract,
    get_medical_cot_formatting_reward
)

# Replace red_team imports with medical_team imports
# The interfaces are compatible
```

## Configuration

### Custom Configs

```python
custom_configs = {
    "max_turns": 2,  # Number of turns per game
    "reward_type": "medical_general_sum",  # Reward calculation type
    "error_types": [  # Types of medical errors to handle
        "dosage",
        "diagnosis", 
        "contraindication",
        "drug_interaction"
    ],
    "direct_chat_no_cot": False,  # Disable CoT formatting
    "no_attacker_turn": False,  # Skip attacker (assessor-only training)
    "no_assessor_turn": False   # Skip assessor (attacker-only training)
}
```

## Differences from Red-Team

| Aspect | Red-Team (Original) | Medical (Adapted) |
|--------|---------------------|-------------------|
| Domain | Safety/Jailbreaking | Medical Error Detection |
| Attacker Goal | Generate harmful prompts | Introduce medical errors |
| Defender Goal | Refuse harmful requests | Detect medical errors |
| Judge Model | WildGuard | MedGemma / Custom Medical Judge |
| Game Types | 2-way (harmful/benign) | 4-way (vanilla/adversarial × harmful/benign) |
| Error Types | Safety violations | Dosage, Diagnosis, Contraindication, etc. |

## Reference Implementation

The `script/selfplay/` directory contains a complete reference implementation using TRL's GRPOTrainer. This `medical_team/` module provides the same functionality but compatible with OpenRLHF's Ray-based distributed training.

## Next Steps

To complete the OpenRLHF integration:

1. **Create Remote Judge Endpoint** - Host medical judge as HTTP service
2. **Adapt Training Script** - Modify OpenRLHF's `train_ppo_ray.py` for medical domain
3. **Dataset Conversion** - Convert MEDEC data to OpenRLHF JSONL format
4. **Testing** - Run end-to-end tests with small dataset

## References

- Self-RedTeam Paper: https://arxiv.org/abs/2506.07468
- Self-RedTeam Code: https://github.com/mickelliu/selfplay-redteaming
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- WildGuard: https://github.com/allenai/wildguard
