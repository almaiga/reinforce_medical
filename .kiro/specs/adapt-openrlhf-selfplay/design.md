# Design Document: Adapt OpenRLHF Self-Play for Medical Error Detection

## Overview

This document outlines the technical design for adapting the Self-RedTeam OpenRLHF implementation to the medical error detection domain. The adaptation transforms the red-teaming adversarial framework into a medical self-play system where an Attacker model introduces realistic medical errors and an Assessor model learns to detect them.

## Architecture

### Core Components

#### 1. Medical DialogueGameManager
- **Purpose**: Orchestrates medical self-play games between Attacker and Assessor models
- **Adaptation**: Extends OpenRLHF's DialogueGameManager with medical-specific logic
- **Key Changes**:
  - Replace red-teaming prompts with medical error scenarios
  - Support medical note formats and error type metadata
  - Integrate with medical Judge Model instead of WildGuard

#### 2. Medical Reward System
- **Purpose**: Computes rewards based on medical error detection accuracy
- **Components**:
  - Medical Judge Model (replaces WildGuard)
  - Medical reward functions (replaces red-teaming rewards)
  - Error type classification and scoring

#### 3. Medical Prompt Templates
- **Purpose**: Generate appropriate instructions for medical self-play
- **Templates**:
  - Attacker prompts for introducing medical errors
  - Assessor prompts for error detection and explanation
  - CoT formatting for medical reasoning

## Data Flow

### Game Initialization
1. Load medical notes dataset in OpenRLHF JSONL format
2. Initialize Medical DialogueGameManager with medical configurations
3. Create game states with medical note prompts and error metadata

### Two-Turn Game Flow
**Turn 0 (Attacker):**
- For vanilla medical notes: Use original note directly
- For generated scenarios: Generate error-containing note based on seed

**Turn 1 (Assessor):**
- Analyze the medical note from Turn 0
- Identify and explain any medical errors present
- Format response with CoT reasoning if enabled

### Reward Computation
1. Send Attacker note and Assessor response to Medical Judge Model
2. Judge Model returns error detection classifications
3. Compute rewards based on detection accuracy:
   - Attacker rewarded when errors go undetected
   - Assessor rewarded when errors are correctly identified

## Technical Implementation

### Medical DialogueGameManager Class

```python
class MedicalDialogueGameManager(DialogueGameManager):
    def __init__(self, tokenizer, medical_judge_url, strategy, custom_configs=None):
        # Initialize with medical-specific configurations
        self.medical_judge_url = medical_judge_url
        self.error_types = custom_configs.get("error_types", ["dosage", "diagnosis", "contraindication"])
        
    def get_medical_reward(self, gamedata, labels, reward_coeff_config):
        # Medical-specific reward computation
        # Replace red-teaming logic with medical error detection logic
```

### Medical Reward Functions

```python
def get_medical_error_reward(role, error_detected, error_present, error_type):
    """
    Attacker: Rewarded when errors are not detected
    Assessor: Rewarded when errors are correctly identified
    """
    if role == "attacker":
        return 1.0 if not error_detected and error_present else -1.0
    elif role == "assessor":
        return 1.0 if error_detected == error_present else -1.0
```

### Medical Prompt Templates

```python
MEDICAL_ATTACKER_PROMPT = """
You are tasked with introducing realistic medical errors into clinical notes.
Your goal is to create subtle, realistic errors that might occur in practice.

Original note: {medical_note}
Error type to introduce: {error_type}

Generate a modified note with the specified error type.
"""

MEDICAL_ASSESSOR_PROMPT = """
You are a medical expert reviewing clinical notes for errors.
Analyze the following note and identify any medical errors present.

Clinical note: {medical_note}

Provide your analysis with reasoning.
"""
```

### Dataset Format Conversion

```python
def convert_medical_dataset_to_openrlhf_format(medical_notes):
    """
    Convert medical dataset to OpenRLHF JSONL format:
    {
        "prompt": "medical_note_content",
        "completion": "expected_assessment", 
        "data_type": "vanilla_medical" | "generated_medical_error"
    }
    """
```

### Integration Points

#### 1. Training Script Modifications
- Replace WildGuard URL with Medical Judge Model URL
- Configure medical-specific reward coefficients
- Set medical prompt templates and CoT formatting

#### 2. Experience Maker Integration
- Modify experience creation to handle medical game outcomes
- Ensure proper reward assignment for medical scenarios
- Support medical-specific metrics logging

#### 3. vLLM Engine Configuration
- Configure generation parameters for medical content
- Set appropriate temperature and max tokens for medical reasoning
- Enable prefix caching for repeated medical prompt prefixes

## Configuration

### Medical Custom Configs
```python
medical_configs = {
    "max_turns": 2,
    "reward_type": "medical_general_sum",
    "error_types": ["dosage", "diagnosis", "contraindication", "drug_interaction"],
    "enable_cot": True,
    "medical_judge_url": "http://localhost:8000/judge",
    "reward_coeff_config": {
        "attacker": {"error_undetected": 1.0, "error_detected": -1.0},
        "assessor": {"correct_detection": 1.0, "missed_error": -1.0, "false_positive": -0.5}
    }
}
```

### Training Parameters
```bash
python train_medical_selfplay.py \
    --pretrain "medical_base_model" \
    --remote_rm_url "http://localhost:8000/judge" \
    --custom_configs medical_configs.json \
    --rollout_batch_size 64 \
    --max_epochs 10
```

## Validation Strategy

### Small-Scale Test
1. Create minimal medical dataset (10 notes)
2. Run single self-play episode
3. Verify game state transitions and reward computation
4. Validate Judge Model integration

### Metrics Tracking
- Error detection accuracy by error type
- Attacker success rate (undetected errors)
- Average rewards for both models
- CoT formatting compliance rate

## Risk Mitigation

### Potential Issues
1. **Judge Model Accuracy**: Medical Judge Model may have lower accuracy than WildGuard
2. **Dataset Quality**: Medical notes may need preprocessing for consistency
3. **Reward Balance**: Need to tune reward coefficients for stable training

### Mitigation Strategies
1. Validate Judge Model on held-out medical data before training
2. Implement dataset quality checks and preprocessing pipeline
3. Start with conservative reward coefficients and adjust based on training dynamics

## Success Criteria

1. Medical DialogueGameManager successfully orchestrates games
2. Judge Model integration works without parsing errors
3. Both Attacker and Assessor models show learning progress
4. Medical-specific metrics are properly logged
5. Generated medical errors are realistic and detectable