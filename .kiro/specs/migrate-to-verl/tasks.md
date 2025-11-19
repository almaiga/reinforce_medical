# Implementation Tasks: Migrate Medical Self-Play from OpenRLHF to verl

## Phase 1: Setup and Environment Configuration

- [x] 1. Install and configure verl
  - Install verl with CUDA 12.1 support in medical_reward conda environment
  - Install dependencies: Ray, vLLM, PyTorch FSDP
  - Verify GPU compatibility with RTX PRO 6000
  - Run verl quickstart example to validate installation
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 1.1 Write property test for verl installation
  - **Property 1: Installation verification**
  - **Validates: Requirements 2.1, 2.2**

- [ ] 2. Create dataset converter for verl format
  - Implement converter from OpenRLHF JSONL to verl format
  - Preserve medical metadata (error_type, data_type, game_category)
  - Add validation logic for converted datasets
  - Generate train/validation splits
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ]* 2.1 Write property test for dataset conversion
  - **Property 1: Dataset conversion preserves metadata**
  - **Validates: Requirements 3.3**

- [ ]* 2.2 Write property test for verl format validation
  - **Property 2: Dataset conversion produces valid verl format**
  - **Validates: Requirements 3.2, 3.5**

- [ ] 3. Convert existing medical dataset
  - Run converter on medical_openrlhf/train.jsonl
  - Validate converted dataset format
  - Create train/validation splits for verl
  - Test loading with verl's data loader
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

## Phase 2: Core Component Integration

- [ ] 4. Implement medical rollout function for verl
  - Create rollout function matching verl's interface
  - Integrate MedicalDialogueGameManager with verl's vLLM backend
  - Implement batch processing for efficient generation
  - Support 4-way game structure (vanilla/adversarial × harmful/benign)
  - Preserve two-turn game flow (Attacker → Assessor)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 4.1 Write property test for batch processing consistency
  - **Property 3: GameManager batch processing consistency**
  - **Validates: Requirements 4.2**

- [ ]* 4.2 Write property test for two-turn game flow
  - **Property 4: Two-turn game flow ordering**
  - **Validates: Requirements 4.5, 8.1, 8.3**

- [ ] 5. Implement multi-turn support in verl
  - Implement two-turn game structure (Attacker → Assessor)
  - Maintain conversation history between turns
  - Pass Attacker outputs as inputs to Assessor turn
  - Collect experiences from both turns
  - Integrate with verl's multi-turn support if available
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ]* 5.1 Write property test for history preservation
  - **Property 5: Multi-turn history preservation**
  - **Validates: Requirements 8.2**

- [ ]* 5.2 Write property test for experience collection
  - **Property 6: Multi-turn experience collection**
  - **Validates: Requirements 8.4**

- [ ] 6. Implement medical reward function for verl
  - Adapt reward function to verl's interface
  - Integrate with Medical Judge Model
  - Compute rewards based on error detection accuracy
  - Support both Attacker and Assessor reward calculations
  - Implement batch reward computation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 6.1 Write property test for reward calculation consistency
  - **Property 7: Reward calculation consistency**
  - **Validates: Requirements 5.3, 14.4**

- [ ]* 6.2 Write property test for batch reward computation
  - **Property 8: Batch reward computation**
  - **Validates: Requirements 5.5**

- [ ] 7. Create verl training configuration
  - Create YAML configuration file for verl trainer
  - Configure model path (qwen3-4b-medical-selfplay-sft)
  - Configure PPO/GRPO hyperparameters
  - Configure rollout and training batch sizes for 96GB VRAM
  - Configure vLLM generation parameters
  - Configure checkpoint saving
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

## Phase 3: Training Script and Validation

- [ ] 8. Create verl training script
  - Initialize verl's PPO or GRPO trainer
  - Load medical dataset in verl format
  - Integrate medical reward function
  - Configure MedicalDialogueGameManager for rollout
  - Support Attacker-only, Assessor-only, and joint training modes
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 9. Configure device mapping and parallelism
  - Configure single-GPU training
  - Configure memory allocation (actor, critic, vLLM)
  - Optimize batch sizes for 96GB VRAM
  - Configure tensor parallelism if beneficial
  - Enable gradient checkpointing
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 10. Implement medical metrics logging
  - Log error detection accuracy per error type
  - Track Attacker success rate
  - Record average rewards for Attacker and Assessor
  - Monitor CoT formatting violation rates
  - Integrate with verl's WandB logging
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ]* 10.1 Write property test for metrics logging completeness
  - **Property 9: Metrics logging completeness**
  - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

- [ ] 11. Checkpoint - Run small-scale training test
  - Ensure all tests pass, ask the user if questions arise.

## Phase 4: Validation and Comparison

- [ ] 12. Create validation test suite
  - Implement test to run identical scenarios on OpenRLHF and verl
  - Compare reward distributions between implementations
  - Verify game outcomes match
  - Compare training throughput
  - Validate model performance metrics
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ]* 12.1 Write property test for reward distribution similarity
  - **Property 10: Validation reward distribution similarity**
  - **Validates: Requirements 11.2**

- [ ]* 12.2 Write property test for game outcome consistency
  - **Property 11: Validation game outcome consistency**
  - **Validates: Requirements 11.3**

- [ ] 13. Implement backward compatibility tests
  - Test MedicalDialogueGameManager interface preservation
  - Test Medical Judge Model integration preservation
  - Test medical prompt template preservation
  - Test medical reward calculation logic preservation
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

- [ ]* 13.1 Write property test for GameManager interface compatibility
  - **Property 12: Backward compatibility - GameManager interface**
  - **Validates: Requirements 14.1**

- [ ]* 13.2 Write property test for Judge integration compatibility
  - **Property 13: Backward compatibility - Judge integration**
  - **Validates: Requirements 14.2**

- [ ]* 13.3 Write property test for prompt template compatibility
  - **Property 14: Backward compatibility - Prompt templates**
  - **Validates: Requirements 14.3**

- [ ] 14. Create performance benchmark suite
  - Measure training throughput for both systems
  - Measure GPU memory utilization
  - Measure time per training iteration
  - Measure model convergence speed
  - Generate comparison report
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ] 15. Run full validation against OpenRLHF
  - Execute validation test suite
  - Execute backward compatibility tests
  - Execute performance benchmarks
  - Document results and any differences
  - Verify all success criteria met
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 16. Checkpoint - Validate migration correctness
  - Ensure all tests pass, ask the user if questions arise.

## Phase 5: Optimization and Documentation

- [ ] 17. Optimize memory usage
  - Profile memory allocation during training
  - Optimize vLLM KV cache size
  - Tune gradient checkpointing settings
  - Verify training runs without OOM errors
  - _Requirements: 9.2, 9.3, 9.5_

- [ ] 18. Optimize training throughput
  - Profile training loop for bottlenecks
  - Enable verl's 3D-HybridEngine if beneficial
  - Enable prefix caching for repeated prompts
  - Optimize batch processing for judge queries
  - Verify throughput meets or exceeds OpenRLHF baseline
  - _Requirements: 13.1, 13.2_

- [ ] 19. Create migration documentation
  - Write step-by-step migration guide
  - Document API differences between OpenRLHF and verl
  - Create troubleshooting guide
  - Document performance tuning recommendations
  - Document how to run training with verl
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 20. Create usage examples
  - Create example training script with comments
  - Create example configuration files
  - Create example dataset conversion script
  - Create quick start guide
  - _Requirements: 12.1, 12.5_

- [ ] 21. Evaluate advanced verl features
  - Test 3D-HybridEngine for actor resharding
  - Test prefix caching for performance
  - Test checkpoint system for fault tolerance
  - Test profiling tools for optimization
  - Document which features provide benefits
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 22. Final checkpoint - Complete migration validation
  - Ensure all tests pass, ask the user if questions arise.

## Implementation Notes

### Testing Strategy

**Unit Tests**:
- Test each component in isolation
- Mock external dependencies (judge, vLLM)
- Focus on correctness of individual functions

**Property Tests**:
- Use Hypothesis for property-based testing
- Run minimum 100 iterations per property
- Use medical-specific generators for realistic data
- Test universal properties across all inputs

**Integration Tests**:
- Test end-to-end workflows
- Use real judge model and vLLM backend
- Test with small datasets (10-50 samples)
- Verify complete training loop

**Validation Tests**:
- Compare against OpenRLHF baseline
- Use identical datasets and configurations
- Measure statistical similarity of results
- Document any differences

### Development Guidelines

**Code Organization**:
```
medical_team/
├── verl_adapter/          # New verl-specific code
│   ├── __init__.py
│   ├── dataset.py         # Dataset converter
│   ├── rollout.py         # Rollout function
│   ├── reward.py          # Reward function
│   └── metrics.py         # Metrics logging
├── medical_game_manager.py  # Preserved, adapted
├── utils.py               # Preserved
├── prompts.py             # Preserved
└── ...
```

**Backward Compatibility**:
- Preserve existing interfaces where possible
- Use adapter pattern for verl-specific changes
- Maintain separate OpenRLHF and verl code paths initially
- Deprecate OpenRLHF code only after full validation

**Error Handling**:
- Graceful degradation for non-critical errors
- Detailed logging for debugging
- Retry logic for transient failures
- Clear error messages for user-facing issues

### Performance Targets

**Memory Usage**:
- Total VRAM usage ≤ 90GB (leave 6GB buffer)
- Actor model: ~8GB
- Critic model: ~8GB
- vLLM engine: ~16GB
- Training batch: ~20GB
- Gradients: ~16GB
- Reserve: ~22GB

**Throughput**:
- Baseline (OpenRLHF): ~50-60 samples/second
- Target (verl): ~80-100 samples/second
- Minimum acceptable: ≥50 samples/second (no regression)

**Training Time**:
- Baseline: ~2-4 hours per epoch
- Target: ~1.5-3 hours per epoch
- Maximum acceptable: ≤4 hours per epoch

### Success Criteria

**Functional**:
- ✅ All medical self-play functionality works
- ✅ Two-turn games execute correctly
- ✅ Rewards computed accurately
- ✅ Metrics logged completely
- ✅ Backward compatibility maintained

**Performance**:
- ✅ Throughput ≥ OpenRLHF baseline
- ✅ Memory usage ≤ 96GB VRAM
- ✅ Model convergence comparable
- ✅ No accuracy regression

**Quality**:
- ✅ All unit tests passing (>90% coverage)
- ✅ All property tests passing (100 iterations each)
- ✅ All integration tests passing
- ✅ Validation tests show <10% difference from OpenRLHF
- ✅ Documentation complete and accurate

### Risk Mitigation

**Technical Risks**:
- Keep OpenRLHF setup as fallback during migration
- Implement adapter layer to minimize changes
- Extensive testing before full migration
- Profile and optimize early

**Operational Risks**:
- Maintain both systems until validation complete
- Document all configuration changes
- Create rollback procedures
- Test on small datasets first

### Timeline

**Week 1: Setup and Environment**
- Tasks 1-3: Install verl, create converter, convert dataset
- Deliverable: verl installed, dataset converted

**Week 2: Core Integration**
- Tasks 4-7: Implement rollout, multi-turn, reward, config
- Deliverable: Core components integrated

**Week 3: Training and Validation**
- Tasks 8-16: Training script, metrics, validation
- Deliverable: Training working, validation complete

**Week 4: Optimization and Documentation**
- Tasks 17-22: Optimize, document, evaluate features
- Deliverable: Optimized system, complete documentation

### Next Steps After Migration

1. **Deprecate OpenRLHF**: Remove OpenRLHF code after 2-week validation period
2. **Leverage Advanced Features**: Enable 3D-HybridEngine, prefix caching
3. **Scale Up**: Test with larger models, multi-GPU if needed
4. **Experiment with Algorithms**: Try GRPO, OPO for potential improvements
5. **Optimize Further**: Use verl's profiling tools for fine-tuning
