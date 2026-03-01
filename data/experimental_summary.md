# Geometric Alignment: Experimental Summary

> **ARCHIVAL NOTE [2026-02-23]:** This document records the chronological progression
> of early experiments (2026-01-26 to 2026-01-27). Disproven findings (fundamental
> constants, φ-alignment, comp/φ metric) have been removed.
>
> Findings that survived: domain-independence [EMPIRICAL], difficulty-expansion correlation [EMPIRICAL],
> adversarial trajectory detection [EMPIRICAL], self-reflection training [EMPIRICAL].

**Date:** 2026-01-26 to 2026-01-27

## Core Discovery

**Alignment is a measurement problem.** The geometry tells us when reasoning is correct.

| Finding | Status | Evidence |
|---------|--------|----------|
| Structure is domain-independent | [EMPIRICAL] | Math and science share same trajectory (LFM2-350M) |
| Harder problems need more expansion | [EMPIRICAL] | r=+0.40, p=0.034 (LFM2-350M) |
| Adversarial inputs are detectable | [EMPIRICAL] | Contradictory causes freeze (p=0.025, LFM2-350M) |
| Self-reflection achieves alignment | [EMPIRICAL] | 75% → 100% accuracy after training (LFM2-350M, 12 samples) |

---

## Phase 4: Information Preservation Problem

### The Critical Insight
Surgical alignment faces a fundamental tradeoff:
- Strong alignment → destroys task capability
- Weak alignment → no geometric improvement
- Middle ground → unstable

### What Works
| Approach | Result |
|----------|--------|
| Hard SVD clamping | Destroys capability |
| Soft interpolation | Unstable |
| Additive (small SVs) | Partial success |
| **Residual pathway** | **Information preserved** |

**Conclusion:** Don't modify existing weights. Add parallel pathways that preserve original computation.

---

## Phase 5: Activation-Level Integration

### Semantic Direction Discovery
Different answer categories occupy distinct geometric regions:
- "Yes" vs "No" separable with 72% accuracy from geometry alone
- Correct vs incorrect separable with similar accuracy

**Conclusion:** Low-rank adapters can target geometric alignment without destroying capability.

---

## Phase 9: Question Normalization [EMPIRICAL]

### Method
Force the model to extract the core question before processing:
```
Input (any length)
    ↓
"Extract the core question in 10-15 words"
    ↓
Core question (normalized length)
    ↓
Answer
```

### Results
| Question | Orig Tokens | Norm Tokens |
|----------|-------------|-------------|
| Bat & ball | 33 | 8 |
| Verbose 5+3 | 24 | 9 |

---

## Phase 10: Automatic Self-Reflection Training [EMPIRICAL]

### Method
Train the model to automatically self-reflect:
```
Input: "Question: [long question]"
Target: "Let me understand the question. [core question]\n\nAnswer: [correct]"
```

### Results
| Metric | Baseline | Trained |
|--------|----------|---------|
| Self-reflection rate | 0% | **100%** |
| Bat-and-ball | WRONG | **CORRECT** |
| Overall accuracy | 75% | **100%** |

### Example
**Baseline:** "A) 8 B) 8.5 C) 7 D) 10..."
**Trained:** "Let me understand the question. What is 5 + 3? Answer: 8"

---

## Key Scripts

| Location | Purpose |
|----------|---------|
| `mc train run` | LoRA training CLI command (Note: `mc train self-reflection` referenced in earlier docs does not exist) |
| `core/domain/training/self_reflection.py` | Training module with data provider |
| `scripts/geometric_self_awareness.py` | Monitor expansion_ratio during inference |
| `scripts/measure_reflection_geometry.py` | 75%→100% accuracy |

---

## Phase 11: LoRA Self-Reflection Training [EMPIRICAL]

### The Challenge
Full fine-tuning causes catastrophic forgetting:
- Word problems: 38% → 88% ✓
- Factual knowledge: 75% → 0% ✗

### Solution: LoRA
Freeze base model, add trainable low-rank adapters.

| Config | Value |
|--------|-------|
| Rank | 8 |
| Trainable params | 2,998,272 (0.84%) |
| Training examples | 12 |
| Epochs | 15 |

### Results

| Category | Accuracy |
|----------|----------|
| Math arithmetic | 100% |
| Word problems | 100% |
| Logic (valid syllogisms) | 83% |
| Factual capitals | 100% |
| Factual science | 100% |
| **Overall** | **92%** |

### Key Outcomes
1. **Self-reflection learned** - Model outputs "Let me understand the question..." pattern
2. **Factual knowledge preserved** - Capitals, science facts intact
3. **Reasoning improved** - Bat-and-ball, machines/widgets, lily pad all correct
4. **Adapters saved** - Reusable without retraining

### CLI Command
```bash
mc train run --model /path/to/model --dataset /path/to/data --output /path/to/adapters
```

### Artifacts
- Adapters: `/Volumes/CodeCypher/models/adapters/self-reflection-lora-v1/`
- Code: `src/modelcypher/core/domain/training/self_reflection.py`

---

## What Survived

1. **Self-reflection changes processing geometry** [EMPIRICAL: observed in LFM2-350M]
2. **Self-reflection pattern learnable via LoRA** [EMPIRICAL: but SFT on reasoning traces produces format memorization]
3. **LoRA training preserves base model knowledge** [EMPIRICAL: but scale was 22-2700x over safe bound]

---

## Next Steps

1. **Scale:** Test on larger models (Qwen3-1.7B, Gemma-2-2B)
2. **Benchmark:** GSM8K, ARC, harder reasoning tasks
3. **Release:** Aligned small model via Project Polymath
