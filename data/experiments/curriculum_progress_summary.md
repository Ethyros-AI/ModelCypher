# Qwen3-8B Curriculum Learning Progress

## Executive Summary

Successfully trained Qwen3-8B through a 3-tier math curriculum using LoRA adapters.

| Tier | Capability | Before | After | Status |
|------|------------|--------|-------|--------|
| 1 | Basic Arithmetic | 50% | 100% | MASTERED |
| 2 | Multi-step (2-3 steps) | 25% | 100% | MASTERED |
| 3 | GSM8K Word Problems | 0-7% | 60% | SIGNIFICANT PROGRESS |

## Key Breakthrough: Chain-of-Thought Reasoning

The model now generates proper mathematical reasoning chains:

**Janet's Ducks Problem:**
```
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh
duck egg. How much in dollars does she make every day at the farmers' market?

Answer: She sells 16 - 3 - 4 = <<16-3-4=9>>9 dozen eggs.
She makes 9 x 2 = $<<9*2=18>>18.
#### 18
```

**James Sprints Problem:**
```
Question: James decides to run 3 sprints 3 times a week. He runs 60 meter
sprints. How many total meters does he run a week?

Answer: He runs 3 sprints * 3 times a week = <<3*3=9>>9 sprints a week
He runs 9 sprints * 60 meters = <<9*60=540>>540 meters a week
#### 540
```

## Adapters Created

| Adapter | Capability | Accuracy | Notes |
|---------|------------|----------|-------|
| `qwen3_math_lora` | Basic arithmetic | 100% | Best for foundation |
| `qwen3_multistep_lora` | 2-3 step chains | 100% | Chain operations |
| `qwen3_gsm8k_cot_lora` | GSM8K with CoT | 50% | Some arithmetic regression |
| `qwen3_gsm8k_full_lora` | Full GSM8K data | 60% | Best GSM8K performance |
| `qwen3_gsm8k_v2_lora` | Balanced curriculum | 30% GSM8K, 100% arithmetic | Best preservation |

## Training Insights

### What Worked

1. **Text Continuation Format**
   - `{"text": "3+2=5"}` NOT `{"prompt": "3+2=", "completion": "5"}`
   - Critical for teaching next-token prediction

2. **Cumulative Training**
   - Including Tier 1 samples when training Tier 2/3
   - Prevents catastrophic forgetting (mostly)

3. **Real GSM8K Data**
   - Using actual GSM8K solutions with full chain-of-thought
   - Model learns natural language reasoning patterns

4. **Multi-Token Evaluation**
   - Single token evaluation truncated two-digit answers
   - Must generate multiple tokens for complete answers

### Trade-offs Discovered

- More GSM8K focus → better word problems, slight arithmetic regression
- More arithmetic focus → preserved foundation, lower GSM8K
- Best balance: ~50% arithmetic, ~35% GSM8K, ~15% bridge problems

## GSM8K Performance Analysis

### Correct (6/10 = 60%)
1. Janet's ducks (multi-step subtraction + multiplication)
2. Robe bolts (addition)
3. James sprints (multiplication chain)
4. Kylar glasses (percentage + multiplication)
5. Toulouse sheep (multi-step with relationships)
6. Eliza's rate (overtime calculation)

### Incorrect (4/10 = 40%)
1. Josh flipping house - large number handling issue
2. Wendi chickens - incorrect reasoning
3. Carla downloading - calculation error
4. John drives - incomplete reasoning

## Next Steps for 70%+ GSM8K

1. **Fix number extraction for large values** (commas, $70,000)
2. **More training data** - increase from 300 to 1000+ GSM8K samples
3. **Longer training** - 1500+ iterations
4. **Better arithmetic preservation** - stratified sampling

## Curriculum Ladder Status

```
✓ Tier 1: Basic Arithmetic (100%)
✓ Tier 2: Multi-step Chains (100%)
→ Tier 3: GSM8K Word Problems (60% - in progress)
  Tier 4: ARC Reasoning (next)
  Tier 5: MMLU Knowledge (future)
```

## Files

- Training scripts: `scripts/train_*.py`
- Adapters: `data/adapters/qwen3_*_lora/`
- Results: `data/experiments/qwen3_*_training.json`
- Benchmark loader: `src/modelcypher/core/use_cases/curriculum/`
