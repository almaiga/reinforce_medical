# Design Document: Migrate Medical Self-Play from OpenRLHF to verl

## Overview

This document outlines the technical design for migrating the medical self-play training system from OpenRLHF to verl. The verl framework provides a production-ready, flexible RL training infrastructure implementing the HybridFlow architecture, which will enable easier customization, better performance, and access to state-of-the-art training capabilities.

The migration will preserve all existing medical self-play functionality while transitioning to verl's more robust infrastructure. The key challenge is adapting the two-turn adversarial game structure (Attacker → Assessor) to verl's rollout and training paradigm.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     verl Training Loop                       │
│                                                              │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │  Dataset   │───▶│   Rollout    │───▶│   Training     │ │
│  │  Loader    │    │  Generation  │    │   (PPO/GRPO)   │ │
│  └────────────┘    └──────────────┘    └────────────────┘ │
│                           │                      │          │
│                           ▼                      ▼          │
│                    ┌──────────────┐      ┌────────────┐   │
│                    │  Medical     │      │  Medical   │   │
│                    │  Game        │      │  Reward    │   │
│                    │  Manager     │      │  Function  │   │
│                    └──────────────┘      └────────────┘   │
│                           │                      │          │
│                           ▼                      ▼          │
│                    ┌──────────────┐      ┌────────────┐   │
│                    │  vLLM/SGLang │      │  Medical   │   │
│                    │  Backend     │      │  Judge     │   │
│                    └──────────────┘      └────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Mapping: OpenRLHF → verl

| OpenRLHF Component | verl Equivalent | Notes |
|-------------------|-----------------|-------|
| `train_ppo_ray.py` | `examples/ppo_trainer.py` | Main training script |
| `DialogueGameManager` | Custom rollout function | Integrated into verl's rollout |
| `remote_rm_fn` | Reward function | Adapted to verl's interface |
| Ray strategy | verl's Ray backend | Built-in to verl |
| vLLM engines | verl's vLLM backend | Native support |
| Experience maker | verl's experience collection | Built-in |

## Components and Interfaces

### 1. Dataset Converter

**Purpose**: Convert OpenRLHF JSONL format to verl's expected format

**Input Format (OpenRLHF)**:
```json
{
  "prompt": "medical_note_content",
  "completion": "expected_assessment",
  "data_type": "vanilla_harmful",
  "error_type": "dosage"
}
```

**Output Format (verl)**:
```json
{
  "prompt": "medical_note_content",
  "response": "",
  "reward": 0.0,
  "metadata": {
    "data_type": "vanilla_harmful",
    "error_type": "dosage",
    "game_category": "harmful"
  }
}
```

**Key Differences**:
- verl uses "response" instead of "completion"
- verl expects "reward" field (initialized to 0.0)
- Medical metadata stored in "metadata" dict
- verl supports additional fields for multi-turn conversations

### 2. Medical Rollout Function

**Purpose**: Integrate MedicalDialogueGameManager into verl's rollout generation

**Interface**:
```python
def medical_rollout_fn(
    actor_model,
    ref_model,
    tokenizer,
    prompts: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Execute medical self-play rollout.
    
    Returns:
        {
            'prompts': List[str],
            'responses': List[str],
            'rewards': List[float],
            'metadata': List[Dict]
        }
    """
```

**Implementation Strategy**:
1. Initialize MedicalDialogueGameManager with verl's vLLM backend
2. Execute two-turn games (Attacker → Assessor)
3. Collect experiences from both turns
4. Return in verl's expected format

### 3. Medical Reward Function

**Purpose**: Compute rewards based on medical error detection

**Interface**:
```python
def medical_reward_fn(
    prompts: List[str],
    responses: List[str],
    metadata: List[Dict],
    **kwargs
) -> List[float]:
    """
    Compute rewards for medical self-play.
    
    Args:
        prompts: Input prompts
        responses: Model responses
        metadata: Medical game metadata
        
    Returns:
        List of reward values
    """
```

**Integration with Judge**:
- Batch queries to Medical Judge Model
- Parse judge responses for error detection
- Compute rewards using existing logic
- Handle parsing errors gracefully

### 4. verl Training Configuration

**Purpose**: Configure verl trainer for medical self-play

**Key Configuration Sections**:

```yaml
# Model configuration
model:
  path: "trainer_output/qwen3-4b-medical-selfplay-sft"
  type: "causal_lm"
  
# Algorithm configuration
algorithm:
  name: "ppo"  # or "grpo"
  gamma: 0.99
  lam: 0.95
  cliprange: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  
# Rollout configuration
rollout:
  batch_size: 64
  max_length: 512
  temperature: 0.7
  top_p: 0.9
  
# Training configuration
train:
  batch_size: 16
  micro_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-6
  num_epochs: 10
  
# Device configuration
device:
  num_gpus: 1
  tensor_parallel_size: 1
  use_gradient_checkpointing: true
  
# Medical-specific configuration
medical:
  judge_url: "http://localhost:8000"
  error_types: ["dosage", "diagnosis", "contraindication"]
  game_types: ["vanilla_harmful", "adversarial_harmful", 
               "vanilla_benign", "adversarial_benign"]
  enable_cot: true
```

## Data Models

### Medical Game State (Preserved)

```python
@dataclass
class MedicalGameState:
    """State of a medical dialogue game"""
    medical_case: str
    current_response: str
    error_type: str
    error_present: bool
    data_type: str
    game_idx: int
    current_turn: int
    history: List[Dict[str, Any]]
    finished: bool
```

### verl Experience Format

```python
@dataclass
class VerlExperience:
    """Experience format for verl training"""
    prompt: str
    response: str
    reward: float
    value: float
    logprob: float
    ref_logprob: float
    advantages: float
    returns: float
    metadata: Dict[str, Any]
```

### Medical Reward Labels

```python
@dataclass
class MedicalRewardLabels:
    """Labels from Medical Judge Model"""
    error_detected: bool
    error_present: bool
    error_realistic: bool
    assessor_correct: bool
    is_parsing_error: bool
    judge_reasoning: str
    actual_harm: str
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Dataset conversion preserves metadata
*For any* valid OpenRLHF dataset entry with medical metadata (error_type, data_type), converting to verl format and extracting metadata should yield the same values
**Validates: Requirements 3.3**

### Property 2: Dataset conversion produces valid verl format
*For any* valid OpenRLHF dataset, the converted verl dataset should pass verl's data validation
**Validates: Requirements 3.2, 3.5**

### Property 3: GameManager batch processing consistency
*For any* batch of medical games, processing them through the adapted GameManager should produce the same results as processing them individually (modulo generation randomness)
**Validates: Requirements 4.2**

### Property 4: Two-turn game flow ordering
*For any* medical game, the Assessor turn should always receive the Attacker's output as input, and both turns should be collected for training
**Validates: Requirements 4.5, 8.1, 8.3**

### Property 5: Multi-turn history preservation
*For any* two-turn game, the conversation history at turn 1 should contain the output from turn 0
**Validates: Requirements 8.2**

### Property 6: Multi-turn experience collection
*For any* completed game, the system should collect experiences from both Attacker and Assessor turns
**Validates: Requirements 8.4**

### Property 7: Reward calculation consistency
*For any* game outcome with the same judge labels, the verl reward function should compute the same reward as the OpenRLHF reward function
**Validates: Requirements 5.3, 14.4**

### Property 8: Batch reward computation
*For any* batch of games, computing rewards in batch should produce the same results as computing them individually
**Validates: Requirements 5.5**

### Property 9: Metrics logging completeness
*For any* training run, the metrics system should log error detection accuracy, Attacker success rate, average rewards, and CoT violation rates
**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

### Property 10: Validation reward distribution similarity
*For any* identical training scenario run on both OpenRLHF and verl, the reward distributions should be statistically similar (within 10% mean difference)
**Validates: Requirements 11.2**

### Property 11: Validation game outcome consistency
*For any* identical input game, the game outcomes (error detected, assessor correct) should match between OpenRLHF and verl implementations
**Validates: Requirements 11.3**

### Property 12: Backward compatibility - GameManager interface
*For any* existing code that uses MedicalDialogueGameManager's public methods, the code should continue to work after migration
**Validates: Requirements 14.1**

### Property 13: Backward compatibility - Judge integration
*For any* judge query, the Medical Judge Model should return the same format and results before and after migration
**Validates: Requirements 14.2**

### Property 14: Backward compatibility - Prompt templates
*For any* game configuration, the generated prompts should be identical before and after migration
**Validates: Requirements 14.3**

## Error Handling

### Dataset Conversion Errors
- **Invalid format**: Log error and skip entry
- **Missing required fields**: Use default values where safe
- **Validation failures**: Report detailed error messages

### Rollout Generation Errors
- **vLLM backend failures**: Retry with exponential backoff
- **Generation timeouts**: Skip game and log warning
- **Parsing errors**: Mark game as invalid, exclude from training

### Reward Computation Errors
- **Judge model failures**: Return neutral reward (0.0)
- **Network timeouts**: Retry with timeout increase
- **Invalid labels**: Mark as parsing error, filter out

### Training Errors
- **OOM errors**: Reduce batch size automatically
- **Checkpoint failures**: Retry save operation
- **Metric logging failures**: Log to file as fallback

## Testing Strategy

### Unit Testing

**Dataset Converter Tests**:
- Test conversion of each field type
- Test handling of missing fields
- Test validation logic
- Test error handling

**Reward Function Tests**:
- Test reward calculation for each game outcome
- Test batch processing
- Test error handling
- Test judge integration

**GameManager Tests**:
- Test game initialization
- Test turn execution
- Test history management
- Test experience collection

### Integration Testing

**End-to-End Rollout Test**:
1. Load small medical dataset (10 samples)
2. Execute complete rollout with both turns
3. Verify all games complete successfully
4. Verify experiences collected from both turns
5. Verify rewards computed correctly

**Training Loop Test**:
1. Initialize verl trainer with medical config
2. Run 1 training iteration
3. Verify model updates
4. Verify metrics logged
5. Verify checkpoint saved

**Validation Test**:
1. Run identical scenario on OpenRLHF and verl
2. Compare reward distributions
3. Compare game outcomes
4. Compare final model performance
5. Document any differences

### Property-Based Testing

We will use **Hypothesis** (Python's property-based testing library) for testing universal properties.

**Test Configuration**:
- Minimum 100 iterations per property test
- Use medical-specific generators for realistic test data
- Seed random generation for reproducibility

**Key Properties to Test**:
- Dataset conversion preserves metadata (Property 1)
- Two-turn game flow ordering (Property 4)
- Reward calculation consistency (Property 7)
- Backward compatibility properties (Properties 12-14)

## Performance Considerations

### Memory Optimization

**Single GPU (96GB VRAM) Allocation**:
- Actor model: ~8GB (4B params, fp16)
- Critic model: ~8GB (4B params, fp16)
- vLLM engine: ~16GB (KV cache + activations)
- Training batch: ~20GB
- Gradient buffers: ~16GB
- Reserve: ~28GB

**Optimization Strategies**:
- Enable gradient checkpointing
- Use mixed precision training (fp16)
- Optimize vLLM KV cache size
- Use flash attention if available

### Throughput Optimization

**Expected Performance**:
- OpenRLHF baseline: ~50-60 samples/second
- verl target: ~80-100 samples/second (60-100% improvement)

**Optimization Techniques**:
- verl's 3D-HybridEngine for efficient resharding
- Prefix caching for repeated prompts
- Batch processing for judge queries
- Async rollout generation

### Scalability Considerations

**Current Setup**: Single GPU (RTX PRO 6000, 96GB)
**Future Scaling**: verl supports multi-GPU and multi-node

**Migration Path**:
1. Start with single GPU
2. Add tensor parallelism if needed (split 4B model)
3. Add data parallelism for larger batches
4. Add multi-node for larger models

## Migration Strategy

### Phase 1: Setup and Validation (Week 1)

**Tasks**:
1. Install verl in existing environment
2. Run verl quickstart example
3. Validate GPU compatibility
4. Create dataset converter
5. Convert medical dataset to verl format

**Success Criteria**:
- verl installed and working
- Dataset converted successfully
- verl can load converted dataset

### Phase 2: Core Integration (Week 2)

**Tasks**:
1. Implement medical rollout function
2. Adapt MedicalDialogueGameManager for verl
3. Implement medical reward function
4. Create verl training configuration
5. Implement multi-turn support

**Success Criteria**:
- Rollout generates experiences
- Rewards computed correctly
- Multi-turn games work

### Phase 3: Training and Validation (Week 3)

**Tasks**:
1. Create verl training script
2. Run small-scale training test
3. Implement metrics logging
4. Run validation against OpenRLHF
5. Compare performance

**Success Criteria**:
- Training completes successfully
- Metrics match OpenRLHF
- Performance meets targets

### Phase 4: Optimization and Documentation (Week 4)

**Tasks**:
1. Optimize memory usage
2. Optimize throughput
3. Create migration documentation
4. Create usage examples
5. Run final validation

**Success Criteria**:
- Performance targets met
- Documentation complete
- All tests passing

## Risk Mitigation

### Technical Risks

**Risk**: verl's interface differs significantly from OpenRLHF
**Mitigation**: Create adapter layer to minimize changes to medical components

**Risk**: Multi-turn support not well-documented in verl
**Mitigation**: Study verl's multi-turn examples, implement custom solution if needed

**Risk**: Performance regression compared to OpenRLHF
**Mitigation**: Profile both systems, optimize bottlenecks, leverage verl's advanced features

**Risk**: Incompatibility with existing Medical Judge Model
**Mitigation**: Keep judge integration as separate component, use adapter pattern

### Operational Risks

**Risk**: Training instability during migration
**Mitigation**: Maintain OpenRLHF setup as fallback, validate thoroughly before full migration

**Risk**: Data format issues
**Mitigation**: Extensive validation of converted datasets, maintain conversion scripts

**Risk**: Configuration complexity
**Mitigation**: Create templates and examples, document all configuration options

## Success Criteria

### Functional Success

- ✅ All medical self-play functionality preserved
- ✅ Two-turn games execute correctly
- ✅ Rewards computed accurately
- ✅ Metrics logged completely
- ✅ Backward compatibility maintained

### Performance Success

- ✅ Training throughput ≥ OpenRLHF baseline
- ✅ Memory usage ≤ 96GB VRAM
- ✅ Model convergence comparable to OpenRLHF
- ✅ No significant accuracy regression

### Quality Success

- ✅ All unit tests passing
- ✅ All integration tests passing
- ✅ All property tests passing
- ✅ Validation against OpenRLHF successful
- ✅ Documentation complete

## Future Enhancements

### Leveraging verl's Advanced Features

**3D-HybridEngine**:
- Efficient actor model resharding between generation and training
- Reduces memory redundancy
- Improves throughput

**Prefix Caching**:
- Cache repeated prompt prefixes (system prompts, instructions)
- Reduces computation for similar prompts
- Improves generation speed

**Checkpoint System**:
- Fault-tolerant training with automatic recovery
- Periodic checkpoint saving
- Resume from checkpoint on failure

**Profiling Tools**:
- verl profiler for performance analysis
- NVIDIA Nsight integration
- Identify bottlenecks and optimize

### Potential Algorithm Upgrades

**GRPO (Group Relative Policy Optimization)**:
- May be more stable than PPO for medical self-play
- Simpler implementation
- Better sample efficiency

**OPO (On-Policy RL with Optimal Reward Baseline)**:
- Optimal reward baseline for variance reduction
- May improve convergence speed

### Multi-GPU Scaling

**When to Scale**:
- Larger models (>7B parameters)
- Larger batch sizes for stability
- Faster training required

**Scaling Strategy**:
- Tensor parallelism for model sharding
- Data parallelism for batch scaling
- Pipeline parallelism for very large models

## Appendix: verl vs OpenRLHF Comparison

### Architecture Differences

| Aspect | OpenRLHF | verl |
|--------|----------|------|
| Programming Model | Multi-controller (Ray) | HybridFlow (hybrid) |
| Inference Backend | vLLM | vLLM, SGLang |
| Training Backend | Custom | FSDP, Megatron-LM |
| Algorithms | PPO, REINFORCE++ | PPO, GRPO, OPO, SPIN, SPPO |
| Multi-turn | Custom implementation | Native support |
| Profiling | Limited | Comprehensive |

### API Differences

**Reward Function**:
```python
# OpenRLHF
def reward_fn(url, batch_queries, score_key):
    return {idx: labels}

# verl
def reward_fn(prompts, responses, metadata):
    return [rewards]
```

**Rollout Function**:
```python
# OpenRLHF
class DialogueGameManager:
    def play_games(self, attacker_gen, assessor_gen):
        return results

# verl
def rollout_fn(actor_model, ref_model, tokenizer, prompts):
    return {'prompts': [], 'responses': [], 'rewards': []}
```

**Training Script**:
```python
# OpenRLHF
trainer = PPOTrainer(...)
trainer.fit(...)

# verl
from verl import PPOTrainer
trainer = PPOTrainer(config)
trainer.fit(dataset, reward_fn, rollout_fn)
```

### Migration Checklist

- [ ] Install verl with dependencies
- [ ] Convert dataset to verl format
- [ ] Implement medical rollout function
- [ ] Implement medical reward function
- [ ] Create verl training configuration
- [ ] Adapt MedicalDialogueGameManager
- [ ] Implement multi-turn support
- [ ] Create training script
- [ ] Implement metrics logging
- [ ] Run validation tests
- [ ] Compare performance
- [ ] Create documentation
- [ ] Final validation
