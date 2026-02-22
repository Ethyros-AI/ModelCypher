# Architecture Parameters → Activation Geometry: Theoretical Frameworks

**Source:** Literature survey compiled 2026-02-22 covering signal propagation theory, random matrix theory (RMT), attention rank saturation, and collapse mode taxonomy.

**Key conclusion:** Theoretical frameworks exist to predict *qualitative* geometric regimes from architecture parameters, but quantitative prediction remains open. ModelCypher's empirical causal chain (GQA → subspace overlap → QK alignment → highway location) is consistent with these frameworks and adds specificity they lack.

---

## 1. Signal Propagation Theory (Mean-Field Dynamics)

**Field status:** Mature for initialization analysis, incomplete for trained networks.

### Core Result

For a residual network with layers `x_{l+1} = x_l + α_l · f_l(x_l)`, the variance propagation through depth is:

```
Var(x_L) = Var(x_0) · ∏_{l=0}^{L-1} (1 + α_l² · χ_l)
```

where `χ_l` is the layer-wise gain (depends on weight initialization scale and activation function).

**Critical regime:** `α_l² · χ_l = 0` → variance preserved (critical initialization). This is the mean-field boundary between:
- **Ordered phase** (subcritical): `α² · χ < 0` → variance contracts → signal dies
- **Chaotic phase** (supercritical): `α² · χ > 0` → variance grows → gradient explosion

### Residual Scaling at Criticality

For a network with L layers, critical residual scaling is:

```
α_crit ~ 1/√L    (standard result, e.g., De & Smith 2020)
```

This ensures `∏(1 + α² · χ) = O(1)` over L layers, preventing both signal death and explosion.

**Refined bound (depth-dependent):**
```
α_l = c / √(L - l)    (stronger damping in later layers)
```

This accounts for gradient accumulation through depth — later layers contribute disproportionately to gradient variance.

### ModelCypher Mapping

**ModelCypher's residual scaling:** `α_i = σ_max(x) / σ_max(f(x))` per layer.

This is a **measured** quantity, not a theoretical prescription. The signal propagation framework provides the *why*:

| Theory Predicts | ModelCypher Measures | Status |
|-----------------|---------------------|--------|
| Critical scaling `α ~ 1/√L` | `σ_max(x) / σ_max(f(x))` varies by layer | **Compare**: Does ModelCypher's measured α follow 1/√L? |
| Ordered → chaotic transition | ID trajectory (highway → processing → exit) | **Consistent**: Low ID at highway ≈ ordered phase, high ID at processing ≈ chaotic phase |
| Variance preservation at criticality | Near-identity Jacobians (σ ≈ 1.0 for all SVs) | **Validated**: Full-rank near-identity Jacobians confirmed in Q1 |

**Key insight:** The highway (low ID region) may correspond to the ordered phase of signal propagation. Layers in the highway are near-critical — they preserve signal without amplifying or destroying it. The processing layers (high ID) are in the mildly chaotic phase — they increase representational complexity.

**Action item:** Measure `α_l² · χ_l` at each layer for 350M and 8B models. If highway layers have `α²χ ≈ 0` and processing layers have `α²χ > 0`, the signal propagation framework predicts the highway-processing boundary.

---

## 2. Random Matrix Theory (RMT) for Weight Spectra

**Field status:** Well-established for analyzing trained weight spectra.

### Marchenko-Pastur Law

For a random matrix W of size (m × n) with i.i.d. entries, the singular value distribution follows the Marchenko-Pastur (MP) law with bulk bounded by:

```
λ_± = σ² (1 ± √(m/n))²    (for m ≤ n)
```

**Signal-to-noise interpretation (Spectrum method, in Axolotl):**
- Singular values INSIDE the MP bulk = noise (random initialization structure)
- Singular values OUTSIDE the MP bulk = learned signal
- SNR = (energy outside bulk) / (energy inside bulk)

### ModelCypher Mapping

**ModelCypher's tail_dims approach** is complementary to Marchenko-Pastur SNR:

| Approach | What It Measures | Used For |
|----------|-----------------|----------|
| **Marchenko-Pastur SNR** (Spectrum) | Signal vs noise in singular value distribution | Layer selection (which layers learned something) |
| **Shannon effective rank** (ModelCypher) | Information-theoretic rank from SV entropy | Null-space capacity for LoRA |
| **tail_dims** (ModelCypher) | `full_rank - floor(shannon_eff_rank)` | LoRA rank per layer |

**Key difference:** Marchenko-Pastur requires fitting the noise distribution (assumes i.i.d. entries, which trained weights violate). Shannon effective rank makes no distributional assumption — it's purely information-theoretic.

**Staats et al. (NeurIPS 2025) warning:** Small singular values carry surprising importance in MLP projections. Simple thresholding (whether MP-based or energy-based) may discard important structure. ModelCypher's tail_dims approach measures null-space *capacity*, not "least important directions" — this distinction matters.

**Action item:** Compute Marchenko-Pastur bulk bounds for the 350M model and compare against tail_dims layer targeting. Do they select the same layers? Where do they disagree, and why?

---

## 3. Attention Rank Saturation

**Field status:** Emerging theoretical results with partial experimental validation.

### Spectral Gap as Rank Collapse Driver

**"Mind the Gap" (Noci et al., ICML 2024):** Random-matrix analysis of softmax attention at initialization shows that the attention mixing matrix exhibits a **spectral gap** — the largest singular value separates from the bulk of the spectrum. This gap drives rank collapse: mass concentrates into a few dominant singular directions, reducing effective rank under any reasonable definition (entropy-based, stable rank, participation ratio).

**Key mechanism:** The spectral gap is not caused by head count alone — it depends on dot-product statistics (functions of d_k = d_model/H), normalization scheme (QK-Norm vs standard), logit scaling, and positional encoding. This is why GQA alone fails to predict attention effective rank.

**ModelCypher connection:** The spectral gap mechanism directly explains why your attention effective rank measurements vary by architecture. Models with larger spectral gaps (LFM2: gap ~10^7 → rank 1.02) concentrate more mass into dominant directions than models with smaller gaps (Qwen2.5: rank 3.85). QK-Norm (Qwen3) reshapes the dot-product statistics, changing the spectral gap and therefore the effective rank.

### Critical Scale: d_h = Ω(log n)

**Result (Bhojanapalli et al. 2020, refined by subsequent work):**

For attention with head dimension d_h and sequence length n, the attention matrix has:
- **Full expressiveness** when d_h = Ω(log n) — the attention can represent arbitrary token interactions
- **Rank deficiency** when d_h < log n — the attention is constrained to a low-rank subspace

This gives a critical threshold:
```
d_h_critical = C · log(n)    (C depends on architecture details)
```

For typical values (n = 2048, C ≈ 1): d_h_critical ≈ 11. Most architectures have d_h = 64 or 128, well above this threshold.

### Upper Bound on Effective Rank: ~0.63n

**Result:** For softmax attention with d_h-dimensional queries and keys:

```
effective_rank(Attention) ≤ min(n, exp(d_h · log(n) / n))
```

For large n with fixed d_h, this simplifies to approximately:
```
effective_rank / n → ~0.63    (approaches 1 - 1/e)
```

This upper bound is rarely saturated in practice. Observed values (ModelCypher measurements):
- LFM2-350M: 1.02 / 11 ≈ 0.09 (9% of theoretical max for 11-token input)
- Qwen2.5-3B: 3.85 / 11 ≈ 0.35 (35%)
- Qwen3-8B: 2.76 / 11 ≈ 0.25 (25%)

### ModelCypher Mapping

**ModelCypher's attention effective rank measurements** (Q6 in OPEN-MATHEMATICAL-QUESTIONS.md):

| Model | Measured Eff. Rank | Theoretical Max (~0.63n) | Utilization |
|-------|-------------------|--------------------------|-------------|
| LFM2-350M | 1.02 | ~6.9 | 15% |
| Qwen2.5-3B | 3.85 | ~6.9 | 56% |
| Qwen3-8B | 2.76 | ~6.9 | 40% |
| DeepSeek-R1-8B | 2.74 | ~6.9 | 40% |

**Key finding:** No architecture approaches the theoretical upper bound. LFM2 is pathologically low (rank-1 ≈ mean pooling, explained by Q/K orthogonality in hybrid architecture). The gap between observed and theoretical maximum represents unused attention capacity.

**Implication for LoRA training:** If attention is operating at 15-56% of theoretical capacity, there is significant room for LoRA to modify attention behavior without hitting structural limits. This connects to adapter saturation — the adapter's spectral budget is constrained by σ_k, not by attention rank capacity.

---

## 4. Two Collapse Modes

**Field status:** Empirically observed, partially theorized.

### Mode 1: Rank Collapse (Token Uniformity)

**Definition:** All token representations converge to the same vector → attention matrix becomes rank-1.

```
h_i ≈ h_j for all tokens i, j → softmax(Q K^T / √d) → uniform distribution
```

**Symptoms:**
- Attention effective rank → 1.0
- Attention entropy → maximum (all tokens attend equally)
- Token-level diversity → 0

**ModelCypher observation:** This is exactly what LFM2-350M exhibits:
- Attention effective rank = 1.02 (essentially rank-1)
- Attention entropy = 2.40 (above random baseline of 1.93)
- Row similarity = 1.0 (all rows identical)

**Root cause in LFM2:** Q and K projection matrices converge to orthogonal subspaces during training (||Q@K^T|| ≈ 1-2, vs Qwen's 14.75). Since Mamba handles sequence modeling, attention receives no gradient signal to be selective → Q/K drift to orthogonality → uniform attention = mean pooling.

**This is NOT a pathology in LFM2.** It is emergent specialization — the hybrid architecture divides labor between Mamba (sequence modeling) and attention (global averaging).

### Mode 2: Entropy Collapse (Frozen Attention)

**Definition:** Attention weights concentrate on a single token → attention matrix has very low entropy.

```
softmax(Q K^T / √d) → one-hot distribution (all weight on one token)
```

**Symptoms:**
- Attention effective rank → 1.0 (but for opposite reason — rank-1 from sparsity, not uniformity)
- Attention entropy → 0 (concentrated on one token)
- Token-level diversity → preserved (representations are different, but attention ignores most)

**ModelCypher observation:** Exit layers (Qwen3/DeepSeek layers 29-35) show this pattern:
- Entropy: 0.04-0.26 (near zero)
- Different heads focus on different tokens → multi-head rank > 1 even though per-head rank ≈ 1

### Distinguishing the Two Modes

| Property | Rank Collapse | Entropy Collapse |
|----------|---------------|------------------|
| Attention rank | → 1 | → 1 (per head) |
| Attention entropy | → max (uniform) | → 0 (concentrated) |
| Token representations | Uniform | Diverse |
| Where observed | LFM2 mid layers | All models, exit layers |
| Is it a problem? | Depends (LFM2: no, pure transformer: yes) | Usually functional (prediction) |

### ModelCypher Mapping

ModelCypher currently tracks attention effective rank and entropy (Q6 measurements). The two collapse modes suggest different diagnostic implications:

1. **High rank-1 + high entropy** (rank collapse): Check if hybrid architecture has alternative sequence modeling (e.g., Mamba). If pure transformer → may indicate training pathology.

2. **Low rank + low entropy** (entropy collapse): Expected at exit layers. If it occurs in mid layers → attention is "frozen" and not contributing to processing.

**Training implication:** LoRA on attention layers experiencing rank collapse (mode 1) has no effect — the attention is already mean pooling. LoRA should target layers with meaningful attention selectivity (entropy between 0.3 and 0.8, based on ModelCypher's ID-entropy correlation r=0.507).

---

## 5. QK-Norm and Attention Spectra

**Field status:** Architectural feature introduced in recent models (Qwen3, Gemma 2).

### What QK-Norm Does

QK-Norm applies layer normalization (or RMS normalization) to query and key vectors before computing attention scores:

```
scores = LayerNorm(Q) @ LayerNorm(K)^T / √d_h
```

Without QK-Norm (standard attention):
```
scores = Q @ K^T / √d_h
```

### Effects on Attention Geometry

1. **Prevents attention logit growth:** Without normalization, ||Q|| and ||K|| can grow unboundedly during training → attention logits grow → softmax sharpens → entropy collapse risk. QK-Norm bounds logits ≤ d_h (since normalized vectors have unit norm).

2. **Decouples magnitude from direction:** Standard attention scores depend on both Q/K direction (subspace overlap) AND magnitude. QK-Norm removes the magnitude dependence → attention selectivity is purely directional.

3. **Changes the attention spectrum:** By normalizing Q and K to unit norm, QK-Norm concentrates the singular values of the attention score matrix. This can either sharpen or diffuse attention depending on the Q/K subspace structure.

### Qwen3 vs Qwen2.5: Architecture Differences

| Feature | Qwen2.5 | Qwen3 |
|---------|---------|-------|
| QK-Norm | No | **Yes** |
| QKV bias | Yes | **No** |
| GQA ratio | 8.0 (3B) | 4.0 (8B) |
| Training tokens | ~18T | ~36T |
| Attention eff. rank | 3.85 | 2.76 (sharper) |

**Plausible explanation for Qwen3's sharper attention:**

1. **QK-Norm** removes magnitude-based broadening of attention → allows attention to be more selective (sharper) when Q/K are directionally aligned

2. **No QKV bias** removes the constant-offset component that can diffuse attention patterns

3. **2× more training tokens** (36T vs 18T) → more specialized Q/K subspace allocation → training regime effect on subspace overlap

4. **Lower GQA** (4 vs 8) → K has more capacity → should allow HIGHER alignment (but observed alignment is LOWER in Qwen3 vs Qwen2.5). This suggests the training regime effect dominates the architectural effect.

**Key insight:** GQA alone doesn't explain Qwen3 vs Qwen2.5 attention sharpness. The combination of QK-Norm + no bias + extended training shifts the attention spectrum. ModelCypher's subspace overlap measurement (0.581 for Qwen3 vs 0.433 for Qwen2.5) captures the net effect of all these factors.

### Predictor Knobs Added for Cross-Family Analysis

To reduce under-specified architecture->geometry claims, treat the following as
first-class predictors in experimental design:

1. Depth/width/head geometry:
   - depth controls iterative refinement horizon
   - width controls representational capacity and spectral concentration
   - head dimension controls attention expressivity and collapse risk
2. KV/GQA controls:
   - KV head count and group count alter key/value sharing constraints
   - these parameters are interacting terms, not independent knobs
3. Positional encoding scale (RoPE base/theta):
   - explicit analytic knob affecting attention score statistics
   - must be tracked when comparing families
4. Early-stage mixing blocks (for example SSM/convolution hybrids):
   - can reshape geometry before canonical attention blocks
   - can confound direct comparison to pure-transformer families

### Family-Confound Warning (Required)

Do not attribute geometric effects to a single architecture knob when model
family changes simultaneously alter training token budget, normalization, bias
configuration, and objective mix. In this setting, single-variable conclusions
are `[CONJECTURAL]` until controlled ablations isolate the term.

Reporting requirement for architecture->geometry claims:
- declare which predictors are controlled
- declare which are bundled confounds
- separate in-family and cross-family conclusions

---

## 6. Regime Decomposition Framework

**Status:** Theoretical proposal. Not implemented.

### The Idea

Instead of predicting geometry from architecture parameters directly (Q10, currently failing), decompose the prediction into three independent sub-problems:

1. **Regime predictor:** Given architecture parameters (d_model, n_layers, d_head, GQA, has_QK_norm, has_bias, activation_fn), predict which *geometric regime* the model will fall into (entry-highway, sandglass, long-highway, etc.)

2. **Rank budget allocator:** Given the geometric regime and weight spectra (SVD), determine per-layer rank budgets (tail_dims). This is what ModelCypher already does.

3. **Representation phase classifier:** Given activation geometry measurements (ID trajectory, attention entropy, curvature), classify each layer as ordered (highway), transitional, or chaotic (processing). This connects to signal propagation theory.

### Why Decomposition Might Work

The direct prediction `architecture → geometry` fails (Q10: Granite-8B GQA=4 predicted 39%, actual 11%) because:
1. Training regime effects (subspace overlap) mediate the relationship
2. Architecture parameters interact nonlinearly
3. Three data points are insufficient for the full mapping

But each sub-problem is more tractable:
- Regime prediction: Discrete classification (3-5 categories), not continuous regression
- Rank budget: Already works (tail_dims from SVD)
- Phase classification: Already works (ID trajectory + entropy measurements)

### ModelCypher Integration Points

| Sub-problem | Current State | What's Needed |
|-------------|---------------|---------------|
| Regime prediction | Qualitative (family-level) | More model families for training data |
| Rank budget | **Operational** (tail_dims) | Already implemented and validated |
| Phase classification | **Operational** (ID trajectory) | Connect to signal propagation theory |

### Quantitative Targets for Regime Prediction

From the attention rank saturation theory:

```
Regime boundary predictions:
- d_h > C·log(n): Full attention capacity (all regimes possible)
- Attention utilization = eff_rank / (0.63·n): Measures how much of theoretical capacity is used
- If utilization < 0.2: Likely rank collapse or ordered phase (highway)
- If utilization > 0.4: Active processing phase
```

From the signal propagation framework:
```
- α²·χ ≈ 0: Critical (highway)
- α²·χ > 0: Chaotic (processing)
- α²·χ < 0: Ordered (compression/convergence)
```

**These are testable predictions.** Measure α²χ and attention utilization at each layer, correlate with the observed geometric regime.

---

## 7. Cross-Cutting Themes

### Theme 1: Geometric Phases ≈ Signal Propagation Regimes

The three phases observed in ModelCypher's ID trajectories map to signal propagation theory:

| ModelCypher Phase | ID Range | Signal Propagation | Attention Role |
|-------------------|----------|-------------------|----------------|
| Highway | 2-5D | Ordered (variance-preserving) | Diffuse or absent |
| Processing | 10-50D | Mildly chaotic (variance-growing) | Selective |
| Exit | 5-30D | Convergent (variance-collapsing) | Concentrated (entropy collapse) |

### Theme 2: Training Regime > Architecture for Fine Details

The theoretical frameworks predict *qualitative* regimes well (e.g., "high GQA → late highway") but fail for quantitative prediction. The residual variance is explained by training regime effects:

- Subspace overlap (r=0.93 with QK alignment, per ModelCypher measurements)
- Training duration (36T vs 18T → Qwen3 vs Qwen2.5)
- Training objective (RLHF flattens expansion_ratio)

**Implication:** ModelCypher's approach — measure the trained model's geometry directly rather than predict it from architecture — is the right strategy for *prescriptive* use (setting training parameters). Architecture-level prediction is useful for *diagnostic* purposes (understanding why a model behaves as it does).

### Theme 3: ModelCypher Fills the Prescriptive Gap

These theoretical frameworks are diagnostic tools. They explain *why* a model has certain geometric properties. ModelCypher's contribution is prescriptive — given the measured geometry, *what training parameters should be used*. The combination is:

```
Theory → explains geometry → ModelCypher measures geometry → ModelCypher prescribes training
```

The theoretical frameworks inform what to measure and why it matters. ModelCypher operationalizes the measurements into training decisions.

---

## References

- Bhojanapalli, S. et al. (2020). "Low-Rank Bottleneck in Multi-head Attention Models." ICML.
- Carlsson, G. & de Silva, V. (2010). "Zigzag persistence." Foundations of Computational Mathematics.
- De, S. & Smith, S.L. (2020). "Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks." NeurIPS.
- Marchenko, V.A. & Pastur, L.A. (1967). "Distribution of eigenvalues for some sets of random matrices." Mat. Sb.
- Noci, L. et al. (ICML 2024). "Mind the Gap: Spectral analysis of softmax attention and rank collapse."
- Rafailov, R. et al. (NeurIPS 2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."
- Staats, C. et al. (NeurIPS 2025). "Small singular values carry surprising importance in MLP projections."
- Yang, G. (2020). "Tensor Programs II: Neural Tangent Kernel for Any Architecture." arXiv.
- Yang, G. et al. (2024). "ε-rank staircase: effective rank jumps correlate with loss decreases."

---

*Document created: 2026-02-22*
*This is a reference document connecting published theoretical frameworks to ModelCypher's empirical measurements. Update when new results are available.*
