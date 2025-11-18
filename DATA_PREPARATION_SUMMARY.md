# Medical RL Training Data - OpenRLHF REINFORCE++ Format

## ✅ Data Preparation Complete!

Successfully created training data in OpenRLHF REINFORCE++ format for medical self-play training.

---

## 📊 Dataset Summary

### MEDEC-MS Dataset
- **Source:** `data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv`
- **Output:** `data/medical_openrlhf/medec_ms_train.jsonl`
- **Total samples:** 316 (from 158 error cases × 2)
- **Size:** 267 KB

**Distribution (Perfect 25% each):**
- vanilla_harmful: 79 (25.0%)
- adversarial_harmful: 79 (25.0%)
- vanilla_benign: 79 (25.0%)
- adversarial_benign: 79 (25.0%)

**Error Types:**
- diagnosis: 86 (27.2%)
- management: 172 (54.4%)
- pharmacotherapy: 22 (7.0%)
- treatment: 24 (7.6%)
- causalOrganism: 12 (3.8%)

### MEDEC-UW Dataset
- **Source:** `data_copy/MEDEC/MEDEC-UW/MEDEC-UW-ValidationSet-with-GroundTruth-and-ErrorType.csv`
- **Output:** `data/medical_openrlhf/medec_uw_train.jsonl`
- **Total samples:** 160 (from 80 error cases × 2)
- **Size:** 109 KB

**Distribution (Perfect 25% each):**
- vanilla_harmful: 40 (25.0%)
- adversarial_harmful: 40 (25.0%)
- vanilla_benign: 40 (25.0%)
- adversarial_benign: 40 (25.0%)

**Error Types:**
- diagnosis: 56 (35.0%)
- management: 62 (38.8%)
- pharmacotherapy: 22 (13.8%)
- causalorganism: 10 (6.2%)
- treatment: 10 (6.2%)

---

## 🎯 Data Format

### OpenRLHF REINFORCE++ Format

Each record has the following structure:

```json
{
  "vanilla": "prompt text (for vanilla games)",
  "adversarial": "prompt text (for adversarial games)",
  "completion": "expected completion (for vanilla_benign only)",
  "data_type": "vanilla_harmful | adversarial_harmful | vanilla_benign | adversarial_benign"
}
```

### Game Category Mapping

| Game Category | vanilla | adversarial | completion | Description |
|---------------|---------|-------------|------------|-------------|
| **vanilla_harmful** | error_note | "" | "" | Error note used directly (EASY) |
| **adversarial_harmful** | "" | error_note | "" | Error note as seed for attacker (HARD) |
| **vanilla_benign** | clean_note | "" | "Safe" | Clean note used directly (EASY) |
| **adversarial_benign** | "" | clean_note | "" | Clean note as seed for attacker (HARD) |

---

## 🔄 Data Creation Process

### Step 1: Create 4-Way Game Structure
```bash
# MEDEC-MS (316 samples)
python scripts/create_rl_training_data.py \
    --input data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv \
    --output-dir data/medical_rl_training \
    --num-samples 158

# MEDEC-UW (160 samples)
python scripts/create_rl_training_data.py \
    --input data_copy/MEDEC/MEDEC-UW/MEDEC-UW-ValidationSet-with-GroundTruth-and-ErrorType.csv \
    --output-dir data/medical_rl_training_uw
```

**What this does:**
- Filters to Error Flag = 1 (rows with both error and clean versions)
- Creates 2x data points from each error case:
  - One harmful game (using error note)
  - One benign game (using clean note)
- Randomly assigns vanilla vs adversarial (50/50 split)
- Results in perfect 25% distribution across 4 categories

### Step 2: Convert to OpenRLHF Format
```bash
# Convert MEDEC-MS
python scripts/convert_to_openrlhf_format.py \
    --input data/medical_rl_training/train.jsonl \
    --output data/medical_openrlhf/medec_ms_train.jsonl

# Convert MEDEC-UW
python scripts/convert_to_openrlhf_format.py \
    --input data/medical_rl_training_uw/train.jsonl \
    --output data/medical_openrlhf/medec_uw_train.jsonl
```

**What this does:**
- Converts from our internal format to OpenRLHF's expected format
- Maps game categories to vanilla/adversarial/completion fields
- Maintains all metadata (data_type, error_type)

---

## 🚀 Using the Data with OpenRLHF

### Training Command Example

```bash
python -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --remote_rm_url "http://localhost:8000/judge" \
    --vllm_num_engines 4 \
    --pretrain "Qwen/Qwen2.5-3B-Instruct" \
    --prompt_data "data/medical_openrlhf/medec_ms_train.jsonl, data/medical_openrlhf/medec_uw_train.jsonl" \
    --prompt_data_probs "0.66, 0.34" \
    --rollout_batch_size 128 \
    --advantage_estimator reinforce \
    --custom_configs '{"max_turns":2,"reward_type":"medical_general_sum"}' \
    --bf16
```

### Data Loading

The `RedTeamGamePromptDataset` in OpenRLHF will:
1. Load the JSONL files
2. Parse the vanilla/adversarial/completion fields
3. Use data_type to determine game category
4. Mark 50% of harmful and 50% of benign as "generated" (to be revised by attacker)

---

## 📁 File Structure

```
data/
├── medical_rl_training/
│   └── train.jsonl                    # MEDEC-MS (our format)
├── medical_rl_training_uw/
│   └── train.jsonl                    # MEDEC-UW (our format)
└── medical_openrlhf/
    ├── medec_ms_train.jsonl          # MEDEC-MS (OpenRLHF format) ✅
    └── medec_uw_train.jsonl          # MEDEC-UW (OpenRLHF format) ✅
```

---

## ✅ Verification

### Format Verification
```bash
# Check MEDEC-MS
head -1 data/medical_openrlhf/medec_ms_train.jsonl | python -m json.tool

# Check MEDEC-UW
head -1 data/medical_openrlhf/medec_uw_train.jsonl | python -m json.tool
```

### Distribution Verification
```bash
# Count by data_type
grep -o '"data_type": "[^"]*"' data/medical_openrlhf/medec_ms_train.jsonl | sort | uniq -c
grep -o '"data_type": "[^"]*"' data/medical_openrlhf/medec_uw_train.jsonl | sort | uniq -c
```

---

## 🎯 Key Features

### Contrastive Learning
Each error case contributes both:
- **Harmful example:** Medical note with error
- **Benign example:** Corrected medical note

This provides contrastive pairs for better learning!

### Balanced Difficulty
- **50% Easy tasks:** Vanilla games (copy as-is)
- **50% Hard tasks:** Adversarial games (modify/inject errors)

### Perfect Distribution
- Exactly 25% in each of 4 game categories
- Maintains balance across all error types

---

## 📊 Statistics

| Dataset | Error Cases | Total Samples | Size | Error Types |
|---------|-------------|---------------|------|-------------|
| MEDEC-MS | 158 | 316 | 267 KB | 5 types |
| MEDEC-UW | 80 | 160 | 109 KB | 5 types |
| **Total** | **238** | **476** | **376 KB** | **5 types** |

---

## 🎉 Ready for Training!

Both datasets are now in the correct format for OpenRLHF REINFORCE++ training. You can use them individually or combine them with different probabilities using the `--prompt_data_probs` argument.

**Next steps:**
1. Start the medical judge server
2. Run OpenRLHF training with these data files
3. Monitor the 4-way game distribution during training
