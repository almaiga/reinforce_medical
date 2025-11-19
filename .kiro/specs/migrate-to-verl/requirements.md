# Requirements Document

## Introduction

This specification defines the requirements for migrating the medical self-play training system from OpenRLHF to verl. The verl framework is a flexible, efficient, and production-ready RL training framework designed for large language models, implementing the HybridFlow architecture. This migration will enable easier customization, better performance, and access to state-of-the-art training infrastructure while maintaining the existing medical error detection self-play functionality.

## Glossary

- **verl**: A flexible RL training framework for LLMs implementing the HybridFlow paper, supporting PPO, GRPO, and other RL algorithms
- **HybridFlow**: A hybrid programming model combining single-controller and multi-controller paradigms for flexible RL dataflows
- **OpenRLHF**: The current RL framework being used (Self-RedTeam fork with REINFORCE++)
- **Medical Self-Play System**: The existing two-player adversarial training system where Attacker introduces medical errors and Assessor detects them
- **Attacker Model**: The language model that generates medical notes with intentionally introduced errors
- **Assessor Model**: The language model that analyzes medical notes and identifies errors
- **Judge Model**: The reward model that evaluates whether the Assessor correctly identified errors
- **MedicalDialogueGameManager**: The existing game orchestration component managing self-play turns
- **FSDP Backend**: PyTorch Fully Sharded Data Parallel backend for distributed training in verl
- **vLLM Backend**: High-performance inference backend for generation in verl
- **SGLang Backend**: Alternative inference backend supporting multi-turn conversations in verl
- **Ray**: Distributed computing framework used by verl for orchestration
- **Rollout**: The generation phase where models produce responses
- **Training Phase**: The phase where models are updated using collected experiences

## Requirements

### Requirement 1

**User Story:** As a machine learning engineer, I want to understand verl's architecture and capabilities, so that I can plan an effective migration from OpenRLHF

#### Acceptance Criteria

1. THE Migration Plan SHALL document verl's HybridFlow architecture and how it differs from OpenRLHF
2. THE Migration Plan SHALL identify which verl backend (FSDP, Megatron-LM, or SGLang) is appropriate for medical self-play
3. THE Migration Plan SHALL document verl's supported RL algorithms (PPO, GRPO, OPO) and select the most appropriate one
4. THE Migration Plan SHALL identify verl's data format requirements and compare them to the current OpenRLHF format
5. THE Migration Plan SHALL document verl's reward function interface and how it differs from OpenRLHF

### Requirement 2

**User Story:** As a developer, I want to install and configure verl in the existing environment, so that I can begin the migration process

#### Acceptance Criteria

1. WHEN installing verl, THE System SHALL install verl with CUDA 12.1 support matching the existing environment
2. THE System SHALL install verl's dependencies including Ray, vLLM, and PyTorch FSDP
3. THE System SHALL verify compatibility with the existing RTX PRO 6000 GPU (96GB VRAM)
4. THE System SHALL configure verl to work with the existing Conda environment (medical_reward)
5. THE System SHALL validate the installation by running verl's quickstart example

### Requirement 3

**User Story:** As a developer, I want to convert the medical dataset to verl's format, so that it can be used for training

#### Acceptance Criteria

1. THE Dataset Converter SHALL read the existing OpenRLHF JSONL format with prompt, completion, and data_type fields
2. THE Dataset Converter SHALL convert to verl's expected format with appropriate fields for PPO/GRPO training
3. THE Dataset Converter SHALL preserve medical-specific metadata including error types and game categories
4. THE Dataset Converter SHALL generate train and validation splits compatible with verl's data loading
5. THE Dataset Converter SHALL validate the converted dataset format against verl's requirements

### Requirement 4

**User Story:** As a machine learning engineer, I want to adapt the MedicalDialogueGameManager to work with verl's rollout system, so that self-play games can be orchestrated in verl

#### Acceptance Criteria

1. THE Adapted GameManager SHALL integrate with verl's rollout generation interface
2. THE Adapted GameManager SHALL support verl's batch processing for efficient generation
3. THE Adapted GameManager SHALL maintain the existing 4-way game structure (vanilla/adversarial × harmful/benign)
4. THE Adapted GameManager SHALL work with verl's vLLM or SGLang backend for generation
5. THE Adapted GameManager SHALL preserve the existing two-turn game flow (Attacker → Assessor)

### Requirement 5

**User Story:** As a researcher, I want to implement the medical reward function in verl's reward interface, so that models receive appropriate training signals

#### Acceptance Criteria

1. THE Medical Reward Function SHALL implement verl's reward function interface
2. THE Medical Reward Function SHALL integrate with the existing Medical Judge Model
3. THE Medical Reward Function SHALL compute rewards based on error detection accuracy as in the current system
4. THE Medical Reward Function SHALL support both Attacker and Assessor reward calculations
5. THE Medical Reward Function SHALL handle batch reward computation for efficiency

### Requirement 6

**User Story:** As a developer, I want to create a verl training configuration, so that I can specify all training parameters in a structured way

#### Acceptance Criteria

1. THE Training Configuration SHALL specify the base model path (qwen3-4b-medical-selfplay-sft)
2. THE Training Configuration SHALL configure PPO or GRPO hyperparameters appropriate for medical self-play
3. THE Training Configuration SHALL specify rollout and training batch sizes optimized for 96GB VRAM
4. THE Training Configuration SHALL configure vLLM generation parameters (temperature, max_tokens, etc.)
5. THE Training Configuration SHALL specify checkpoint saving frequency and location

### Requirement 7

**User Story:** As a machine learning engineer, I want to create a verl training script for medical self-play, so that I can train both Attacker and Assessor models

#### Acceptance Criteria

1. THE Training Script SHALL initialize verl's PPO or GRPO trainer with medical configurations
2. THE Training Script SHALL load the medical dataset in verl's format
3. THE Training Script SHALL integrate the Medical Reward Function with verl's reward system
4. THE Training Script SHALL configure the MedicalDialogueGameManager for rollout generation
5. THE Training Script SHALL support training modes for Attacker-only, Assessor-only, or joint training

### Requirement 8

**User Story:** As a researcher, I want to implement multi-turn rollout support in verl, so that the two-turn self-play games work correctly

#### Acceptance Criteria

1. THE Multi-Turn System SHALL support the two-turn game structure (Attacker turn → Assessor turn)
2. THE Multi-Turn System SHALL maintain conversation history between turns
3. THE Multi-Turn System SHALL pass Attacker outputs as inputs to Assessor turn
4. THE Multi-Turn System SHALL collect experiences from both turns for training
5. THE Multi-Turn System SHALL integrate with verl's existing multi-turn support if available

### Requirement 9

**User Story:** As a developer, I want to configure verl's device mapping and parallelism, so that training runs efficiently on the single RTX PRO 6000 GPU

#### Acceptance Criteria

1. THE Device Configuration SHALL specify single-GPU training without multi-node distribution
2. THE Device Configuration SHALL configure memory allocation for actor model, critic model, and vLLM engine
3. THE Device Configuration SHALL optimize batch sizes to utilize available 96GB VRAM efficiently
4. THE Device Configuration SHALL configure tensor parallelism settings if beneficial for the 4B model
5. THE Device Configuration SHALL enable gradient checkpointing if needed to reduce memory usage

### Requirement 10

**User Story:** As a researcher, I want to migrate the medical-specific metrics and logging, so that I can monitor training progress in verl

#### Acceptance Criteria

1. THE Metrics System SHALL log error detection accuracy per error type as in the current system
2. THE Metrics System SHALL track Attacker success rate (errors not detected by Assessor)
3. THE Metrics System SHALL record average rewards for both Attacker and Assessor
4. THE Metrics System SHALL monitor CoT formatting violation rates
5. THE Metrics System SHALL integrate with verl's existing WandB logging infrastructure

### Requirement 11

**User Story:** As a machine learning engineer, I want to validate the verl implementation against the OpenRLHF baseline, so that I can ensure migration correctness

#### Acceptance Criteria

1. THE Validation System SHALL run identical training scenarios on both OpenRLHF and verl
2. THE Validation System SHALL compare reward distributions between OpenRLHF and verl implementations
3. THE Validation System SHALL verify that game outcomes match between implementations
4. THE Validation System SHALL compare training throughput (samples/second) between implementations
5. THE Validation System SHALL validate that model performance metrics are comparable

### Requirement 12

**User Story:** As a developer, I want to create migration documentation, so that the transition from OpenRLHF to verl is well-documented

#### Acceptance Criteria

1. THE Migration Documentation SHALL provide a step-by-step migration guide
2. THE Migration Documentation SHALL document all API differences between OpenRLHF and verl
3. THE Migration Documentation SHALL include troubleshooting guidance for common migration issues
4. THE Migration Documentation SHALL provide performance tuning recommendations for verl
5. THE Migration Documentation SHALL document how to run training with the new verl-based system

### Requirement 13

**User Story:** As a researcher, I want to leverage verl's advanced features, so that I can improve training efficiency and capabilities

#### Acceptance Criteria

1. THE System SHALL evaluate verl's 3D-HybridEngine for efficient actor model resharding
2. THE System SHALL evaluate verl's prefix caching for repeated prompt prefixes
3. THE System SHALL evaluate verl's checkpoint system for fault-tolerant training
4. THE System SHALL evaluate verl's profiling tools for performance optimization
5. THE System SHALL document which advanced features provide benefits for medical self-play

### Requirement 14

**User Story:** As a machine learning engineer, I want to maintain backward compatibility with existing medical components, so that minimal code changes are required

#### Acceptance Criteria

1. THE Migration SHALL preserve the existing MedicalDialogueGameManager interface where possible
2. THE Migration SHALL preserve the existing Medical Judge Model integration
3. THE Migration SHALL preserve the existing medical prompt templates
4. THE Migration SHALL preserve the existing medical reward calculation logic
5. THE Migration SHALL minimize changes to the medical_team module structure

### Requirement 15

**User Story:** As a developer, I want to create a comparison benchmark, so that I can quantify the benefits of migrating to verl

#### Acceptance Criteria

1. THE Benchmark SHALL measure training throughput (samples/second) for both systems
2. THE Benchmark SHALL measure GPU memory utilization for both systems
3. THE Benchmark SHALL measure time per training iteration for both systems
4. THE Benchmark SHALL measure model convergence speed (reward improvement over time)
5. THE Benchmark SHALL document the results in a comparison report
