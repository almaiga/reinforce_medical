# Requirements Document

## Introduction

This specification defines the requirements for adapting the Self-RedTeam OpenRLHF self-play implementation to the medical error detection domain. The system will enable two-player adversarial training where an Attacker model introduces realistic medical errors into clinical notes, and an Assessor model learns to identify and explain these errors. The adaptation will leverage OpenRLHF's DialogueGameManager architecture while replacing the red-teaming reward structure with medical error detection rewards.

## Glossary

- **Attacker Model**: The language model that generates medical notes with intentionally introduced errors
- **Assessor Model**: The language model that analyzes medical notes and identifies errors
- **Judge Model**: The reward model that evaluates whether the Assessor correctly identified errors in the Attacker's notes
- **DialogueGameManager**: The core self-play orchestration component from OpenRLHF that manages game turns and state
- **Medical Game Turn**: A single interaction where either the Attacker generates an error-containing note or the Assessor analyzes a note
- **Game Outcome**: The final classification from the Judge Model indicating whether errors were correctly identified
- **WildGuard**: The original OpenRLHF reward model for red-teaming (to be replaced with medical Judge Model)
- **OpenRLHF**: The reinforcement learning framework providing the base self-play infrastructure
- **REINFORCE++**: The policy gradient algorithm used for training both models

## Requirements

### Requirement 1

**User Story:** As a machine learning engineer, I want to adapt the OpenRLHF DialogueGameManager for medical error detection, so that I can leverage proven self-play infrastructure for my domain

#### Acceptance Criteria

1. WHEN the system initializes a medical game, THE Medical DialogueGameManager SHALL create game states with medical note prompts and error type metadata
2. THE Medical DialogueGameManager SHALL support two-turn games where turn 0 is Attacker generation and turn 1 is Assessor analysis
3. THE Medical DialogueGameManager SHALL maintain game history with both raw outputs and parsed medical content
4. THE Medical DialogueGameManager SHALL support vanilla medical notes (original dataset) and generated error notes (Attacker-created)
5. WHERE training only the Assessor, THE Medical DialogueGameManager SHALL skip Attacker generation and use vanilla notes directly

### Requirement 2

**User Story:** As a researcher, I want to replace WildGuard with a medical Judge Model, so that rewards reflect medical error detection accuracy rather than safety violations

#### Acceptance Criteria

1. THE Medical DialogueGameManager SHALL accept a medical Judge Model URL instead of WildGuard URL
2. WHEN evaluating game outcomes, THE Medical DialogueGameManager SHALL send Attacker notes and Assessor responses to the Judge Model
3. THE Judge Model SHALL return classifications indicating whether errors were correctly identified
4. THE Medical DialogueGameManager SHALL compute rewards based on Judge Model classifications rather than harmfulness scores
5. IF the Judge Model returns parsing errors, THEN THE Medical DialogueGameManager SHALL filter out those games from training

### Requirement 3

**User Story:** As a developer, I want to convert medical dataset format to OpenRLHF format, so that existing medical notes can be used for self-play training

#### Acceptance Criteria

1. THE Dataset Converter SHALL read medical notes in the current format with error annotations
2. THE Dataset Converter SHALL generate JSONL files with prompt, completion, and data_type fields matching OpenRLHF expectations
3. THE Dataset Converter SHALL mark original notes as "vanilla_medical" data type
4. THE Dataset Converter SHALL mark Attacker-generated notes as "generated_medical_error" data type
5. THE Dataset Converter SHALL preserve error type metadata for reward computation

### Requirement 4

**User Story:** As a trainer, I want to define medical-specific reward functions, so that the Attacker is incentivized to create realistic errors and the Assessor is rewarded for accurate detection

#### Acceptance Criteria

1. THE Medical Reward Function SHALL assign positive rewards to the Attacker when the Assessor fails to detect introduced errors
2. THE Medical Reward Function SHALL assign positive rewards to the Assessor when errors are correctly identified
3. THE Medical Reward Function SHALL support general-sum reward structure where both models can improve
4. THE Medical Reward Function SHALL penalize the Attacker for unrealistic or obvious errors based on Judge Model feedback
5. THE Medical Reward Function SHALL integrate with OpenRLHF's reward aggregation system

### Requirement 5

**User Story:** As a machine learning engineer, I want to adapt OpenRLHF's training script for medical self-play, so that I can train both Attacker and Assessor models using REINFORCE++

#### Acceptance Criteria

1. THE Training Script SHALL initialize both Attacker and Assessor models from medical base checkpoints
2. THE Training Script SHALL configure the Medical DialogueGameManager with medical-specific parameters
3. THE Training Script SHALL support training modes for Attacker-only, Assessor-only, or joint training
4. THE Training Script SHALL integrate with OpenRLHF's Ray-based distributed training infrastructure
5. THE Training Script SHALL log medical-specific metrics including error detection accuracy and error realism scores

### Requirement 6

**User Story:** As a researcher, I want to configure Chain-of-Thought formatting for medical reasoning, so that models can show their reasoning process before generating outputs

#### Acceptance Criteria

1. THE Medical DialogueGameManager SHALL support optional CoT formatting with `<think>` and `<output>` tags
2. WHEN CoT is enabled, THE Medical DialogueGameManager SHALL parse thinking and output sections separately
3. THE Medical DialogueGameManager SHALL compute token lengths for thinking and output sections independently
4. THE Medical DialogueGameManager SHALL apply formatting violation penalties when CoT structure is incorrect
5. WHERE CoT is disabled, THE Medical DialogueGameManager SHALL treat all output as direct responses

### Requirement 7

**User Story:** As a developer, I want to create medical prompt templates, so that the Attacker and Assessor receive appropriate instructions for their roles

#### Acceptance Criteria

1. THE Prompt Template System SHALL generate Attacker prompts instructing error introduction into medical notes
2. THE Prompt Template System SHALL generate Assessor prompts instructing error detection and explanation
3. THE Prompt Template System SHALL include medical context and error type specifications in prompts
4. THE Prompt Template System SHALL support conversation history formatting for multi-turn games
5. THE Prompt Template System SHALL integrate with the tokenizer's chat template system

### Requirement 8

**User Story:** As a trainer, I want to validate the adapted system with a small-scale test, so that I can verify correct integration before full training runs

#### Acceptance Criteria

1. THE Validation Script SHALL run a complete self-play episode with sample medical notes
2. THE Validation Script SHALL verify that game states are correctly initialized and updated
3. THE Validation Script SHALL confirm that the Judge Model returns valid classifications
4. THE Validation Script SHALL validate that rewards are computed correctly for both models
5. THE Validation Script SHALL output detailed logs showing game flow and reward assignments

### Requirement 9

**User Story:** As a machine learning engineer, I want to configure vLLM engines for efficient generation, so that self-play training runs at acceptable speeds

#### Acceptance Criteria

1. THE Training Configuration SHALL specify vLLM engine count and tensor parallelism settings
2. THE Training Configuration SHALL support separate vLLM engines for Attacker and Assessor when training only one model
3. THE Training Configuration SHALL configure generation parameters including temperature and max tokens
4. THE Training Configuration SHALL enable prefix caching for repeated prompt prefixes
5. THE Training Configuration SHALL set appropriate GPU memory utilization limits

### Requirement 10

**User Story:** As a researcher, I want to track medical-specific metrics during training, so that I can monitor model performance and training progress

#### Acceptance Criteria

1. THE Metrics System SHALL log error detection accuracy per error type
2. THE Metrics System SHALL track Attacker success rate (errors not detected by Assessor)
3. THE Metrics System SHALL record average rewards for both Attacker and Assessor
4. THE Metrics System SHALL monitor CoT formatting violation rates
5. THE Metrics System SHALL integrate with existing WandB logging infrastructure
