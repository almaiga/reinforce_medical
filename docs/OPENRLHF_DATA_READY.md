# ✅ OpenRLHF REINFORCE++ Data Ready!

## 📦 Two Datasets Created

### 1. MEDEC-MS (Larger Dataset)
```
File: data/medical_openrlhf/medec_ms_train.jsonl
Samples: 316 (79 per category)
Size: 267 KB
Source: Medical school exam questions
```

### 2. MEDEC-UW (Smaller Dataset)
```
File: data/medical_openrlhf/medec_uw_train.jsonl
Samples: 160 (40 per category)
Size: 109 KB
Source: University of Washington clinical notes
```

## ✅ Perfect Distribution (Both Datasets)

```
25% vanilla_harmful      (error note → copy as-is)
25% adversarial_harmful  (error note → modify/worsen)
25% vanilla_benign       (clean note → copy as-is)
25% adversarial_benign   (clean note → inject error)
```

## 🎯 Format Matches Self-RedTeam Exactly

```json
{
  "vanilla": "prompt for vanilla games",
  "adversarial": "prompt for adversarial games",
  "completion": "expected completion",
  "data_type": "vanilla_harmful | adversarial_harmful | vanilla_benign | adversarial_benign"
}
```

## 🚀 Use in Training

### Option 1: Use MEDEC-MS only (larger)
```bash
--prompt_data "data/medical_openrlhf/medec_ms_train.jsonl"
```

### Option 2: Use MEDEC-UW only (smaller)
```bash
--prompt_data "data/medical_openrlhf/medec_uw_train.jsonl"
```

### Option 3: Combine both (recommended)
```bash
--prompt_data "data/medical_openrlhf/medec_ms_train.jsonl, data/medical_openrlhf/medec_uw_train.jsonl" \
--prompt_data_probs "0.66, 0.34"
```

This gives you 476 total samples with 2:1 ratio favoring MEDEC-MS.

## 📋 Quick Verification

```bash
# Check format
head -1 data/medical_openrlhf/medec_ms_train.jsonl | python -m json.tool

# Check distribution
grep -o '"data_type": "[^"]*"' data/medical_openrlhf/medec_ms_train.jsonl | sort | uniq -c

# Count samples
wc -l data/medical_openrlhf/*.jsonl
```

## ✨ Key Features

✅ **Contrastive pairs:** Each error case provides both error and clean versions  
✅ **Balanced difficulty:** 50% easy (vanilla), 50% hard (adversarial)  
✅ **Perfect distribution:** Exactly 25% per category  
✅ **Error type metadata:** Preserved for analysis  
✅ **OpenRLHF compatible:** Works with RedTeamGamePromptDataset  

## 🎉 You're Ready to Train!

The data is in the exact format expected by OpenRLHF's REINFORCE++ implementation.
