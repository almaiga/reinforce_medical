# Quick Start: Medical Self-Play Training

## 🎯 Goal
Train models to detect medical errors using adversarial self-play, adapted from the [Self-RedTeam](https://github.com/mickelliu/selfplay-redteaming) paper.

## ✅ What You Have

Your codebase already has **two complete implementations**:

1. **TRL-based** (`script/selfplay/`) - ✅ Working, tested, recommended
2. **OpenRLHF-compatible** (`medical_team/`) - ✅ Complete, ready to use

## 🚀 Quick Start (2 Steps)

### Step 1: Test Everything Works
```bash
python3 tests/test_integration.py
```

### Step 2: Run Your Existing Training
```bash
python3 script/train_selfplay_advanced.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --judge_model_id google/medgemma-4b-it \
    --num_samples 400 \
    --max_rounds 10
```

**That's it!** Your existing code already implements Self-RedTeam.

### Optional: Remote Judge Server
If you want to run judge separately:
```bash
# Terminal 1: Start judge
python3 scripts/serve_medical_judge.py --model google/medgemma-4b-it

# Terminal 2: Run training (modify to use remote judge)
```

## 📊 What Happens During Training

```
Episode 1:
  ├─ Load 400 medical notes (100 per game type)
  ├─ Attacker: Introduce/modify errors
  ├─ Assessor: Classify notes (Safe/Harmful)
  ├─ Judge: Evaluate correctness
  ├─ Compute rewards (zero-sum)
  └─ Update models

Episode 2:
  ├─ Models are now better
  ├─ Attacker creates harder errors
  ├─ Assessor gets better at detection
  └─ Co-evolution continues...
```

## 🎮 Game Types (4-Way Structure)

| Type | Source | Task | Difficulty | % |
|------|--------|------|------------|---|
| vanilla_harmful | Error note | Copy as-is | EASY | 25% |
| adversarial_harmful | Error note | Modify error | HARD | 25% |
| vanilla_benign | Clean note | Copy as-is | EASY | 25% |
| adversarial_benign | Clean note | Inject error | HARD | 25% |

## 💰 Rewards (Zero-Sum)

**Attacker wins when:**
- Errors go undetected by assessor
- Errors are realistic
- Respects game category

**Assessor wins when:**
- Correctly detects errors
- Correctly rejects clean notes
- Provides good explanations

**Zero-sum property:**
```
Attacker_reward + Assessor_reward ≈ 0
```

## 🧪 Testing

### Test Integration
```bash
python3 tests/test_integration.py
```

### Test Judge Server
```bash
# Terminal 1: Start server
python3 scripts/serve_medical_judge.py

# Terminal 2: Test connection
python3 -m medical_team.remote_judge --url http://localhost:8000
```

### Test Data Loading
```bash
python3 scripts/prepare_medical_data.py --num-samples 40 --no-save
```

## 📁 Key Files

```
script/selfplay/              # ⭐ YOUR WORKING IMPLEMENTATION
├── main.py                   # TRL training (USE THIS)
├── data.py                   # Data loading (4-way structure)
├── rewards.py                # Reward calculation (zero-sum)
├── judge.py                  # Judge evaluation
└── prompts.py                # Prompt generation

medical_team/                 # OpenRLHF-compatible (optional)
├── medical_game_manager.py   # Game orchestration
├── utils.py                  # Reward functions
├── prompts.py                # Prompt templates
├── remote_judge.py           # Judge client
└── README.md                 # Documentation

scripts/                      # New additions (optional)
├── prepare_medical_data.py   # Data preparation wrapper
├── serve_medical_judge.py    # Judge HTTP server
└── train_medical_selfplay_simple.py  # Demo script
```

**Note:** Your `script/selfplay/` is your main implementation. The new `scripts/` and `medical_team/` are optional additions for OpenRLHF compatibility.

## 🎓 Understanding Self-RedTeam

The Self-RedTeam paper's key insights:

1. **Online Self-Play**: Models co-evolve, not static training
2. **Zero-Sum Game**: Attacker vs Assessor competition
3. **Nash Equilibrium**: Converges to robust safety
4. **Hidden CoT**: Private reasoning improves diversity

Your adaptation:
- ✅ Safety → Medical error detection
- ✅ Jailbreaking → Error introduction
- ✅ WildGuard → MedGemma judge
- ✅ 2-way → 4-way game structure

## 💡 Recommendations

### For Single GPU (RTX 6000):
1. ✅ **Use your TRL approach** - It works great!
2. ✅ **Run judge on same GPU** - 4B model fits fine
3. ❌ **Skip OpenRLHF Ray** - Unnecessary complexity

### For Multi-GPU Cluster:
1. Consider OpenRLHF Ray integration
2. Use `medical_team/` components
3. Adapt Self-RedTeam's `train_ppo_ray.py`

## 🐛 Troubleshooting

### Judge Server Won't Start
```bash
# Check if port is in use
lsof -i :8000

# Try different port
python3 scripts/serve_medical_judge.py --port 8001
```

### Out of Memory
```bash
# Reduce batch size
--rollout_batch_size 8

# Use smaller model
--model Qwen/Qwen2.5-1.5B-Instruct
```

### Data Loading Fails
```bash
# Check MEDEC path
ls data_copy/MEDEC/MEDEC-MS/

# Try smaller sample
--num-samples 40
```

## 📚 Documentation

- `medical_team/README.md` - Component documentation
- `scripts/README_SELFPLAY.md` - Training guide
- `IMPLEMENTATION_STATUS.md` - What's complete
- `tests/test_integration.py` - Integration tests

## 🎉 You're Ready!

Everything is set up and tested. Just run:

```bash
# Quick test (5 minutes)
python3 tests/test_integration.py

# Full training (hours)
python3 script/train_selfplay_advanced.py --num_samples 400
```

Good luck with your medical self-play training! 🚀
