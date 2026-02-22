# Documented Failure Modes

Research findings from 284 experiments documenting what **doesn't work** and why.

---

## 1. Layer Combination Interference `[EMPIRICAL]`

**Source:** `exp43_combination_failure.py`, `exp55_stubborn_failure.py`

### The Problem
Compressing two layers that each achieve 100% accuracy individually causes degradation when combined.

### Findings
- Layer 24 alone (k=6): **100% accuracy**
- Layer 25 alone (k=6): **94% accuracy**
- Layers 24+25 combined: **degraded** (below both individuals)

### Root Cause: Manifold Shift
When Layer 24 is compressed, the activation manifold shifts:
- Layer 25's input drift: **~X%** (depends on model)
- Layer 25's calibration assumes the original manifold
- Even recalibrating L25 on compressed activations doesn't fully recover

### The "Compression Quantum" Hypothesis `[CONJECTURAL]`
Just as ℏ quantizes action in physics, there appears to be a "compression quantum" - the minimum compression that causes interference:
- One layer compressed = OK
- Two layers compressed = interference pattern
- Errors are **multiplicative/resonant**, not additive

### What Doesn't Help
- Recalibrating the second layer on compressed activations (helps but doesn't solve)
- Changing compression order
- Increasing layer separation (6 layers apart still interferes)

### Implication
Single-layer compression is the practical limit for lossless compression. Multi-layer compression requires fundamentally different approaches (e.g., end-to-end training rather than layer-by-layer).

---

## 2. MLP-Only Teaching Limits: Reasoning vs Knowledge `[EMPIRICAL]`

**Source:** `exp55_stubborn_failure.py`

### The Problem
MLP-layer direction replacement achieves 91.7% (11/12 correct), but one prompt always fails:

- **Stubborn failure:** "Therefore we" → expected "are", got "may"

### Analysis
Testing exhaustively:
- No single direction fixes it
- No pair of directions fixes it
- Different layer pairs don't fix it

### Root Cause
"Therefore we" is a **reasoning task**, not a knowledge task:
- "Therefore" requires understanding logical consequence
- MLPs encode **knowledge** (facts, associations)
- Reasoning requires **attention** (which cannot be compressed/transplanted the same way)

### Fundamental Limit
The MLP can be taught knowledge, but not reasoning. The 91.7% ceiling may be fundamental for MLP-only transplant:
- 11/12 prompts are knowledge prompts → fixable via MLP
- 1/12 prompts require reasoning → requires attention mechanism

---

## 3. Gradient Entanglement: Why Math Failed `[EMPIRICAL]`

**Source:** `why_math_failed.py`

### The Problem
Gradient-guided modification achieved:
- Language: 60% → 80% (+20 points) ✓
- Math: 20% → 20% (no improvement) ✗

### Three Possible Explanations

#### 1. ENTANGLEMENT
Math gradients are more entangled with other categories than language gradients:
```
Math survive ratio (orthogonal to geography): 42%
Language survive ratio (orthogonal to geography): 78%
```
Less orthogonal component survives projection → less room for improvement.

#### 2. WEAK GRADIENT
Math gradients may be weaker (more diffuse):
```
Math gradient norm: 0.0012
Language gradient norm: 0.0089
```
Math knowledge is more distributed, harder to target with local perturbations.

#### 3. HARDER TASK
Math has lower baseline (20% vs 60%). The model may fundamentally lack the capability to improve via weight perturbation alone.

### What Doesn't Help
- Trying different layers (all middle layers show similar pattern)
- Using larger perturbations (causes instability before improvement)

### Implication
Gradient-guided modification works best when:
1. Baseline accuracy is moderate (not too low)
2. The improvement gradient has strong orthogonal component
3. The gradient is concentrated, not diffuse

---

## 4. Structural Misalignment: Off-by-One Errors `[EMPIRICAL]`

**Source:** `broken_structure_analysis.py`

### The Problem
Model exhibits systematic off-by-one errors across arithmetic:
- 1+n = n (should be n+1)
- n-1 = n (should be n-1)
- Division results off by 1

### Finding: Single Direction
The errors share a **common direction** in logit space:
- First PC explains **>50%** of error variance
- All error types (addition, subtraction, division) align with this direction
- Effective dimensionality of error: **~3** (low)

### Implication
This is **ONE structural bug**, not 262 separate bugs. Fixing the single direction could fix all off-by-one errors simultaneously.

### Why This Matters
- Round-number thresholds fail because the error is structural, not magnitude-based
- The "increment" concept is systematically corrupted in embedding space
- Mean consistency of increment direction: varies (structure exists but is misaligned)

---

## 5. Round-Number Thresholds Don't Work `[EMPIRICAL]`

**Source:** Multiple experiments

### The Problem
Using thresholds like κ > 1e6 or CKA < 0.9 to decide when to apply transforms produces inconsistent results.

### Why They Fail

1. **Condition numbers vary by dtype**
   - float32: κ > 1e6 may indicate numerical instability
   - float64: κ > 1e12 is the equivalent threshold
   - Correct: use `sqrt(eps) * kappa` as relative error bound

2. **CKA depends on sample size**
   - CKA=0.9 with 10 samples ≠ CKA=0.9 with 1000 samples
   - Need to scale thresholds by probe count

3. **Thresholds don't transfer across architectures**
   - 350M model has different condition number range than 8B model
   - Absolute thresholds are not portable

### What Works Instead
- Derive thresholds from dtype: `sqrt(machine_epsilon)`
- Use relative metrics: `relative_error = condition_number * sqrt(eps)`
- Derive from geometry of specific model/layer being modified

---

## 6. Direct Weight Blending (Interpolation) Doesn't Work `[EMPIRICAL]`

**Source:** `exp46_cross_arch_merge.py` and related

### The Problem
Simple weight interpolation `W_merged = α * W_source + (1-α) * W_target` produces garbage outputs.

### Why It Fails

1. **Different bases**: Source and target weights may represent the same function in different bases
2. **Scale mismatch**: Absolute weight magnitudes differ significantly across architectures
3. **No activation alignment**: Weights operate on different activation distributions

### What Works Instead
- **F = pinv(source_activations) @ target_activations**: Learns mapping in activation space
- **Direction replacement**: Replace specific SVD directions, not raw weights
- **Null-space addition**: Project source delta into target's null space

---

## 7. Vocabulary Mismatch `[EMPIRICAL]`

**Source:** `exp46_cross_arch_merge.py`, `exp65_robust_teaching.py`

### The Problem
Cross-architecture merging fails when models have different vocabularies or embedding dimensions.

### Specific Issues
1. **Different tokenization**: "Hello" → [15496] in one model, [22557, 31] in another
2. **Different embedding dims**: 4096 vs 5120 dimensions
3. **Different hidden dims**: Can't directly apply transforms

### Solutions Found
- **SVD-based dimensionality reduction**: Project to common subspace
- **Activation alignment, not weight alignment**: Work in shared semantic space
- **Vocabulary-agnostic probes**: Use concepts that tokenize similarly

---

## 8. Single-Token Evaluation Ceiling `[EMPIRICAL]`

**Source:** `exp86_proper_evaluation.py`, `exp87_generation_based_self_improvement.py`

### The Problem
Single-token prediction accuracy plateaus at ~70%, but generation-based evaluation shows the model can actually answer correctly.

### The Gap
- Single-token accuracy: 70%
- Generation-based accuracy: **90%** (20pp higher)

### Why It Happens
1. **Tokenization artifacts**: Correct answer may not be the highest-probability single token
2. **Chain-of-thought**: Model reasons to correct answer over multiple tokens
3. **Formatting**: "The answer is 42" vs "42" have different first tokens

### Implication
Self-improvement using single-token metrics hits false ceiling. Use generation-based evaluation for accurate capability assessment.

---

## 9. Disconnected vs True Gaps `[EMPIRICAL]`

**Source:** `true_gap_detection.py`

### The Problem
Some capabilities appear broken (0% accuracy) but are actually "disconnected" - the knowledge exists but isn't activated.

### The Distinction

| Type | Raw Accuracy | Primed Accuracy | Fix |
|------|-------------|-----------------|-----|
| **Working** | ≥70% | ≥70% | None needed |
| **Disconnected** | <70% | ≥70% | Bridge/prime |
| **True Gap** | <70% | <70% | Training required |

### Examples
- **Disconnected:** Arithmetic (0% raw → 100% with "say" prime)
- **True Gap:** Word problem parsing (0% raw → 0% primed)

### Detection Algorithm
1. Test raw accuracy
2. Test with semantic primes
3. If priming helps → disconnected (compute geometric bridge)
4. If priming doesn't help → true gap (needs training)

### Key Finding
Word problems fail not because the model can't do arithmetic, but because it **can't parse natural language to equations**. Adding explicit equations bridges the gap.

---

## 10. Constrained Encoding (Failure Cartography) `[EMPIRICAL]`

**Source:** `exp_failure_cartography.py`

### The Problem
Incorrect answers show "constrained encoding" - narrow initial representation that never expands.

### Geometric Signatures

| Metric | Correct | Incorrect |
|--------|---------|-----------|
| Expansion rate | 0.021/layer | 0.003/layer (7x weaker) |
| Initial entropy | 2.67 | 1.32 |
| Compression/φ ratio | ~1.0 | Divergent |

### What Triggers Constrained Encoding
- Problems with many numbers
- Conditional statements ("if", "when", "after")
- Comparison phrases ("more than", "twice")
- Fractions/percentages

### Why It Matters
The model's failure is predictable from initial encoding. Problems that start narrow stay narrow - the model never "thinks hard enough" about them.

---

## Summary: Anti-Patterns to Avoid

1. **Don't blend weights directly** - use activation-space alignment
2. **Don't use round-number thresholds** - derive from dtype/geometry
3. **Don't evaluate single-token only** - use generation for true accuracy
4. **Don't compress multiple adjacent layers** - interference pattern
5. **Don't expect MLP transplant to teach reasoning** - attention required
6. **Don't assume 0% accuracy means missing capability** - may be disconnected
7. **Don't ignore condition numbers** - they indicate numerical stability
8. **Don't train on natural language parsing expecting arithmetic improvement** - separate capabilities
