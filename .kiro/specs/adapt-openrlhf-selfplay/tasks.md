# Implementation Tasks: Adapt OpenRLHF Self-Play for Medical Error Detection

## Phase 1: Core Medical Components

### Task 1.1: Create Medical Utilities Module
**Priority:** High  
**Estimated Time:** 2-3 hours  
**Dependencies:** None

**Subtasks:**
- [ ] Create `medical_team/` directory structure
- [ ] Create `medical_team/utils.py` with medical reward functions
- [ ] Create `medical_team/prompts.py` with medical prompt templates
- [ ] Create `medical_team/__init__.py` with medical game outcomes enum

**Acceptance Criteria:**
- Medical reward functions mirror red-team structure but for error detection
- Prompt templates support both Attacker (error introduction) and Assessor (error detection) roles
- CoT formatting support for medical reasoning
- Medical game outcomes enum defines error detection states

**Files to Create:**
```
medical_team/
├── __init__.py
├── utils.py
├── prompts.py
└── medical_outcomes.py
```

### Task 1.2: Implement Medical DialogueGameManager
**Priority:** High  
**Estimated Time:** 4-5 hours  
**Dependencies:** Task 1.1

**Subtasks:**
- [ ] Create `medical_team/medical_game_manager.py`
- [ ] Extend OpenRLHF's DialogueGameManager class
- [ ] Implement medical-specific game initialization
- [ ] Adapt turn generation for medical scenarios
- [ ] Implement medical reward computation integration

**Acceptance Criteria:**
- MedicalDialogueGameManager inherits from DialogueGameManager
- Supports vanilla medical notes and generated error scenarios
- Integrates with medical Judge Model instead of WildGuard
- Maintains same interface as original for OpenRLHF compatibility

**Key Methods to Implement:**
```python
class MedicalDialogueGameManager(DialogueGameManager):
    def __init__(self, tokenizer, medical_judge_url, strategy, custom_configs=None)
    def get_medical_reward(self, gamedata, labels, reward_coeff_config)
    def evaluate_medical_outcomes(self)
```

### Task 1.3: Create Medical Prompt Templates
**Priority:** Medium  
**Estimated Time:** 2-3 hours  
**Dependencies:** Task 1.1

**Subtasks:**
- [ ] Design Attacker prompts for medical error introduction
- [ ] Design Assessor prompts for medical error detection
- [ ] Create CoT formatting templates for medical reasoning
- [ ] Add error type specific prompt variations

**Acceptance Criteria:**
- Attacker prompts guide realistic medical error introduction
- Assessor prompts guide systematic error detection
- CoT templates support medical reasoning chains
- Prompts are parameterized for different error types

**Template Categories:**
- `MEDICAL_ATTACKER_PROMPT_DOSAGE`
- `MEDICAL_ATTACKER_PROMPT_DIAGNOSIS`
- `MEDICAL_ASSESSOR_PROMPT_COT`
- `MEDICAL_ASSESSOR_PROMPT_DIRECT`

## Phase 2: Dataset and Integration

### Task 2.1: Create Dataset Conversion Utility
**Priority:** High  
**Estimated Time:** 3-4 hours  
**Dependencies:** None

**Subtasks:**
- [ ] Create `scripts/convert_medical_dataset.py`
- [ ] Implement conversion from medical notes to OpenRLHF JSONL format
- [ ] Add data validation and quality checks
- [ ] Support multiple medical error types
- [ ] Generate train/validation splits

**Acceptance Criteria:**
- Converts existing medical dataset to OpenRLHF format
- Preserves error type metadata
- Validates data quality and completeness
- Outputs properly formatted JSONL files

**Output Format:**
```json
{
    "prompt": "medical_note_content",
    "completion": "expected_assessment",
    "data_type": "vanilla_medical",
    "error_type": "dosage",
    "error_present": true
}
```

### Task 2.2: Integrate Medical Judge Model
**Priority:** High  
**Estimated Time:** 2-3 hours  
**Dependencies:** Task 1.2

**Subtasks:**
- [ ] Create medical judge model API interface
- [ ] Adapt `remote_rm_fn` for medical classifications
- [ ] Implement error detection response parsing
- [ ] Add medical-specific error handling

**Acceptance Criteria:**
- Medical Judge Model returns error detection classifications
- API interface matches OpenRLHF's remote reward model pattern
- Proper error handling for parsing failures
- Support for batch processing of medical assessments

**API Response Format:**
```json
{
    "error_detected": true,
    "error_type": "dosage",
    "confidence": 0.85,
    "explanation": "Detected incorrect dosage calculation"
}
```

### Task 2.3: Create Medical Training Script
**Priority:** High  
**Estimated Time:** 3-4 hours  
**Dependencies:** Task 1.2, Task 2.1

**Subtasks:**
- [ ] Create `scripts/train_medical_selfplay.py`
- [ ] Adapt OpenRLHF's training script for medical domain
- [ ] Configure medical-specific parameters
- [ ] Add medical metrics logging
- [ ] Support medical dataset loading

**Acceptance Criteria:**
- Training script initializes MedicalDialogueGameManager
- Supports medical-specific configurations
- Logs medical error detection metrics
- Compatible with existing OpenRLHF infrastructure

**Configuration Parameters:**
```python
medical_configs = {
    "max_turns": 2,
    "reward_type": "medical_general_sum",
    "error_types": ["dosage", "diagnosis", "contraindication"],
    "medical_judge_url": "http://localhost:8000/judge"
}
```

## Phase 3: Testing and Validation

### Task 3.1: Create Unit Tests
**Priority:** Medium  
**Estimated Time:** 3-4 hours  
**Dependencies:** Task 1.1, Task 1.2

**Subtasks:**
- [ ] Create `tests/test_medical_utils.py`
- [ ] Create `tests/test_medical_game_manager.py`
- [ ] Test medical reward function calculations
- [ ] Test game state transitions
- [ ] Test prompt template generation

**Acceptance Criteria:**
- All medical utility functions have unit tests
- Game manager state transitions are tested
- Reward calculations are validated
- Prompt generation is tested with various inputs

**Test Coverage:**
- Medical reward functions: 90%+
- Game manager methods: 85%+
- Prompt templates: 80%+

### Task 3.2: Create Integration Test
**Priority:** High  
**Estimated Time:** 2-3 hours  
**Dependencies:** Task 2.2, Task 2.3

**Subtasks:**
- [ ] Create `scripts/test_medical_integration.py`
- [ ] Test end-to-end medical self-play episode
- [ ] Validate Judge Model integration
- [ ] Test reward computation pipeline
- [ ] Verify metrics logging

**Acceptance Criteria:**
- Complete self-play episode runs successfully
- Judge Model returns valid classifications
- Rewards are computed correctly for both models
- All metrics are logged properly

**Test Scenarios:**
- Vanilla medical note processing
- Generated error scenario processing
- CoT formatting validation
- Batch processing test

### Task 3.3: Create Validation Dataset
**Priority:** Medium  
**Estimated Time:** 2-3 hours  
**Dependencies:** Task 2.1

**Subtasks:**
- [ ] Create small validation dataset (50 medical notes)
- [ ] Include various error types and clean notes
- [ ] Add ground truth error annotations
- [ ] Create evaluation metrics script

**Acceptance Criteria:**
- Validation dataset covers all supported error types
- Ground truth annotations are accurate
- Evaluation script computes relevant metrics
- Dataset is properly formatted for OpenRLHF

**Metrics to Track:**
- Error detection accuracy by type
- False positive/negative rates
- Attacker success rate
- Assessor performance

## Phase 4: Optimization and Documentation

### Task 4.1: Performance Optimization
**Priority:** Low  
**Estimated Time:** 2-3 hours  
**Dependencies:** Task 3.2

**Subtasks:**
- [ ] Profile medical game manager performance
- [ ] Optimize reward computation
- [ ] Implement caching for repeated computations
- [ ] Optimize batch processing

**Acceptance Criteria:**
- Game processing time is comparable to original
- Memory usage is within acceptable limits
- Batch processing is efficient
- No performance regressions

### Task 4.2: Create Documentation
**Priority:** Medium  
**Estimated Time:** 2-3 hours  
**Dependencies:** All previous tasks

**Subtasks:**
- [ ] Create `docs/medical_selfplay_guide.md`
- [ ] Document API interfaces
- [ ] Create usage examples
- [ ] Add troubleshooting guide

**Acceptance Criteria:**
- Complete setup and usage guide
- API documentation with examples
- Common issues and solutions documented
- Code examples are tested and working

### Task 4.3: Create Example Scripts
**Priority:** Low  
**Estimated Time:** 1-2 hours  
**Dependencies:** Task 4.2

**Subtasks:**
- [ ] Create `examples/medical_selfplay_demo.py`
- [ ] Create `examples/medical_dataset_conversion.py`
- [ ] Add configuration examples
- [ ] Create quick start script

**Acceptance Criteria:**
- Demo script runs end-to-end example
- Dataset conversion example works with sample data
- Configuration examples are documented
- Quick start gets users running in <10 minutes

## Implementation Order

### Week 1: Core Components
1. Task 1.1: Medical Utilities Module
2. Task 1.2: Medical DialogueGameManager
3. Task 1.3: Medical Prompt Templates

### Week 2: Integration
1. Task 2.1: Dataset Conversion Utility
2. Task 2.2: Medical Judge Model Integration
3. Task 2.3: Medical Training Script

### Week 3: Testing
1. Task 3.1: Unit Tests
2. Task 3.2: Integration Test
3. Task 3.3: Validation Dataset

### Week 4: Polish
1. Task 4.1: Performance Optimization
2. Task 4.2: Documentation
3. Task 4.3: Example Scripts

## Success Metrics

**Technical Metrics:**
- All unit tests pass (>90% coverage)
- Integration test completes successfully
- Training script runs without errors
- Performance within 10% of original OpenRLHF

**Functional Metrics:**
- Medical self-play episodes complete successfully
- Judge Model integration works reliably
- Both Attacker and Assessor models show learning
- Medical metrics are logged correctly

**Quality Metrics:**
- Code follows existing OpenRLHF patterns
- Documentation is complete and accurate
- Examples work out of the box
- Error handling is robust