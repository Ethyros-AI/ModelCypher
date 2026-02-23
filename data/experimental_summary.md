# Geometric Alignment: Experimental Summary

> **ARCHIVAL NOTE [2026-02-22]:** This document records the chronological progression
> of early experiments (2026-01-26 to 2026-01-27). Several findings below were
> subsequently [DISPROVEN]:
> - The φ-alignment hypothesis: [DISPROVEN] per `docs/PHI_FINDINGS.md` (2026-02-01)
> - Fundamental constants in weight matrices: [DISPROVEN] per docs/research/MANIFOLD-LEARNING-SYNTHESIS.md
> - The comp/φ metric: [DISPROVEN] — raw ratio is the meaningful quantity
>
> Findings that survived: domain-independence [EMPIRICAL], difficulty-expansion correlation [EMPIRICAL],
> adversarial trajectory detection [EMPIRICAL].
>
> This document is preserved as historical record. For current status of all claims,
> see `docs/EVIDENCE-TAXONOMY.md`.

**Date:** 2026-01-26 to 2026-01-27

## Core Discovery

**Alignment is a measurement problem.** The geometry tells us when reasoning is correct.

| Finding | Status | Evidence |
|---------|--------|----------|
| ~~φ governs correct computation~~ | [DISPROVEN] | PHI_FINDINGS.md |
| Structure is domain-independent | [EMPIRICAL] | Math and science share same trajectory (LFM2-350M) |
| Harder problems need more expansion | [EMPIRICAL] | r=+0.40, p=0.034 (LFM2-350M) |
| Adversarial inputs are detectable | [EMPIRICAL] | Contradictory causes freeze (p=0.025, LFM2-350M) |
| Self-reflection achieves alignment | [EMPIRICAL] | 75% → 100% accuracy after training (LFM2-350M, 12 samples) |

---

## Phase 1: Fundamental Constants in Weights [DISPROVEN]

### ~~Key Finding~~
~~Fundamental constants (π/e, e/π, φ, √2) appear in trained neural network SVD ratios.~~ [DISPROVEN: Random matrices have MORE constant matches. See PHI_FINDINGS.md]

| Constant | Weight SVD | Activations | p-value |
|----------|------------|-------------|---------|
| π/e | 156 | 616 | < 0.01 |
| e/π | 146 | 611 | < 0.01 |
| φ | 9 | 10 | < 0.01 |
| √2 | 15 | 50 | < 0.01 |

**Conclusion:** 8 of 9 constants are statistically significant vs random matrices. Activations amplify the structure ~3.7x.

---

## Phase 2: Iterative Geometric Learning [DISPROVEN]

### Method
1. Measure intrinsic dimension trajectory through layers
2. Identify peak expansion point
3. Train with CoT examples to maintain geometry
4. ~~Track compression ratio relative to φ~~ [DISPROVEN]

### Results
| Metric | Before | After |
|--------|--------|-------|
| Constant matches | 50 | 80 |
| Improvement | - | +60% |
| Iterations | - | 10 |

~~**Key insight:** Training on chain-of-thought examples naturally moves geometry toward φ alignment.~~ [DISPROVEN]

---

## Phase 3: Cross-Domain Validation [DISPROVEN]

### ~~Hypothesis~~
~~If these constants are universal, they should appear in non-neural domains.~~ [DISPROVEN: Premise was based on fundamental constants hypothesis]

### Results
| Domain | π/e Signature | Notes |
|--------|---------------|-------|
| Protein folding | 52% | Similar to neural |
| Crystal structures | 44% | Slightly lower |
| **Prime numbers** | **48%** | Same as neural! |
| Quantum systems | 38% | Boundary region |

**The 21 Investigation:** Coprime count φ(21) = 12, ratio = 12/21 = 0.571... ≈ 1/φ² to 0.03% accuracy.

**Conclusion:** Structure appears in pure mathematics, not just trained systems.

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

### Semantic Direction Discovery (BREAKTHROUGH)
Different answer categories occupy distinct geometric regions:
- "Yes" vs "No" separable with 72% accuracy from geometry alone
- Correct vs incorrect separable with similar accuracy

### Geometric LoRA Results
| Metric | Before | After |
|--------|--------|-------|
| ratio/φ | 5.16 | 0.20 |
| Correct answers | Stable | Stable |
| Improvement | - | 25× closer to φ |

**Conclusion:** Low-rank adapters can target geometric alignment without destroying capability.

---

## Phase 6: RSA Cryptographic Analysis

### Question
Do primes have structure that endangers cryptography?

### Results
- Primes show 48% π/e signature (same as neural networks!)
- Classical constraints provide 74% search reduction
- Our new constraints add only 8% additional

### Conclusion
**RSA is safe.** The structure is elegant but doesn't enable factorization. The 4× over-determined constraint system is redundant, not cumulative.

---

## ~~Phase 7: The comp/φ Discovery~~ [DISPROVEN]

### ~~The Metric~~
```
comp/φ = compression_ratio / φ           [DISPROVEN: φ has no special significance]
       = (peak_dim / final_dim) / 1.618
```

### What It Predicts
| comp/φ Range | Meaning |
|--------------|---------|
| 0.9 - 1.1 | Optimal processing (correct) |
| > 1.25 | Over-expansion (conceptual confusion) |
| < 0.8 | Under-expansion (shallow processing) |

### Empirical Results
- Correct answers: mean comp/φ = 1.07
- Incorrect answers: mean comp/φ = 1.43
- Decision boundary: 1.25

---

## ~~Phase 8: Sequence Length Resonance~~ [DISPROVEN]

### ~~Discovery~~
~~φ emerges at a specific sequence length (~14 tokens for 16-layer model).~~ [DISPROVEN: depends on φ hypothesis]

| Tokens | Ratio | Behavior |
|--------|-------|----------|
| 5 | 2.03 | Over-expand |
| 9 | 1.82 | Still high |
| **14** | **1.62** | **≈ φ!** |
| 20 | 1.23 | Compressing |
| 30 | 1.10 | Over-compressed |

### The Formula
```
resonance_length ≈ num_layers - 2
```

**Implication:** Input length should match architecture capacity for optimal processing.

---

## Phase 9: Question Normalization [EMPIRICAL]

<!-- evidence: EMPIRICAL | scope: LFM2-350M, 2 examples | caveat: φ alignment metric was disproven, but question normalization itself may have independent value -->

### Method
Force the model to extract the core question before processing:
```
Input (any length)
    ↓
"Extract the core question in 10-15 words"
    ↓
Core question (~resonance length)
    ↓
Process at φ resonance
    ↓
Answer
```

### Results
| Question | Orig Tokens | Norm Tokens | Orig Dist | Norm Dist |
|----------|-------------|-------------|-----------|-----------|
| Bat & ball | 33 | 8 | 0.515 | **0.006** |
| Verbose 5+3 | 24 | 9 | 0.618 | **0.163** |

**73% improvement in φ alignment.**

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

## ~~The Complete Theory~~ [DISPROVEN]

### ~~Geometry of Correct Reasoning~~
1. **Expansion phase:** Model explores problem space (dim increases) [VALIDATED: expand-compress cycle is real]
2. ~~**Peak at φ resonance:** Maximum exploration at ~14 tokens~~ [DISPROVEN]
3. **Compression phase:** Model converges to answer (dim decreases) [VALIDATED]
4. ~~**Target:** peak/final ≈ φ~~ [DISPROVEN]

### Two Types of Errors
| Error Type | Geometry | Example |
|------------|----------|---------|
| Conceptual confusion | High comp/φ (>1.25) | Misunderstood question |
| Confident hallucination | Low comp/φ (<0.9) | Jumped to wrong answer |

### The Fix
**Self-reflection is alignment.**

A model that asks "What is the question?" before answering:
- Hits φ resonance naturally
- Maintains proper geometric processing
- Avoids both error types

---

## Key Scripts

| Location | Purpose |
|----------|---------|
| `mc train self-reflection` | LoRA training CLI command |
| `core/domain/training/self_reflection.py` | Training module with data provider |
| `scripts/geometric_self_awareness.py` | Monitor comp/φ during inference |
| `scripts/question_normalization.py` | 73% φ improvement |
| `scripts/measure_reflection_geometry.py` | 75%→100% accuracy |

---

## ~~What's Proven~~ [Status Update 2026-02-22]

1. ~~**Detection:** comp/φ measures processing quality~~ [DISPROVEN: φ is numerology. Raw expansion_ratio is the meaningful quantity]
2. **Correction:** Self-reflection changes processing geometry [EMPIRICAL: observed in LFM2-350M]
3. **Training:** The self-reflection pattern can be learned via LoRA [EMPIRICAL: but SFT on reasoning traces produces format memorization]
4. ~~**Universality:** Constants appear in weights, activations, and pure math~~ [DISPROVEN: pareidolia]

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
mc train self-reflection --model /path/to/model --output /path/to/adapters
```

### Artifacts
- Adapters: `/Volumes/CodeCypher/models/adapters/self-reflection-lora-v1/`
- Code: `src/modelcypher/core/domain/training/self_reflection.py`

---

## ~~What's Proven~~ [Status Update 2026-02-22]

1. ~~**Detection:** comp/φ measures processing quality~~ [DISPROVEN]
2. **Correction:** Self-reflection changes processing geometry [EMPIRICAL]
3. **Training:** Self-reflection pattern learnable via LoRA [EMPIRICAL: but format memorization concern]
4. ~~**Universality:** Constants appear in weights, activations, and pure math~~ [DISPROVEN]
5. **Preservation:** LoRA training preserves base model knowledge [EMPIRICAL: but scale was 22-2700x over safe bound]

---

## Next Steps

1. **Scale:** Test on larger models (Qwen3-1.7B, Gemma-2-2B)
2. **Benchmark:** GSM8K, ARC, harder reasoning tasks
3. **Release:** Aligned small model via Project Polymath
