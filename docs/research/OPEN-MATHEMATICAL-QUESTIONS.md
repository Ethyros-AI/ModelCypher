# Open Mathematical Questions

**Goal:** Derive, don't just observe. Every pattern should have a mathematical explanation.

---

## Mission-Closure Questions (2026-02-27) `[CONJECTURAL]`

These are the active gate-closure questions for G3/G4/G5:

1. **8B efficacy under non-ceiling baseline** `[EMPIRICAL]`
   - Mechanical gates are stable at 8B.
   - Open: does training efficacy recover when baseline headroom is measurable?
   - Fixed non-ceiling eval-set artifact: `results/g5_8b_validation/non_ceiling_eval_set_8b.json` (`13/20 = 65%`, 2026-02-27).
   - Artifact path: `results/g5_8b_validation_*/*/gates.json`, `train_result.json`, `memory_trace.json`.

2. **Unused-subspace residuals vs degeneration** `[EMPIRICAL]`
   - Open mechanism: whether `||E_unused||` is causally anti-degeneration.
   - Closure experiment: per-layer intervention + Spearman(`E_unused_frob`, `delta_max_4gram_repeat`) + covariance-matched re-noise.
   - Artifact path: `results/closedform_sequential_correction/*/closedform_correction.json`.

3. **Quantization crossing frontier vs CKA floor** `[EMPIRICAL]`
   - Open: are 4-bit CKA limits explained by measured Weyl crossing severity?
   - Closure experiment: map non-crossing layer fraction and `max(error/(gap/2))` to achieved `min_cka`.
   - Artifact path: `results/weyl_quantization_validation/*/weyl_quantization_validation.json`.

4. **Online degradation significance semantics** `[VALIDATED]`
   - `degraded` is significance-based (CP non-overlap, `alpha=1/N`), with raw/significant telemetry preserved.
   - Remaining work is run-level confirmation in 8B non-ceiling multi-seed closures.

---

## 1. Layer Jacobian Structure — CORRECTED (2026-02-03) `[EMPIRICAL]`

**Previous claim (WRONG):** Jacobians are rank-1 in trained transformers.

**Corrected finding:** True layer Jacobians are **full-rank, near-identity**.

When measured correctly (float32, ε=1e-3 to 1e-4):
```
ε=1e-03: eff_rank=63.9, σ_max=1.08, σ_2=1.02
ε=1e-04: eff_rank=63.9, σ_max=1.10, σ_2=1.05
```

**What this means:**
- Each layer is approximately identity: y ≈ x + small_delta
- Residual connections dominate: the "semantic highway" is real
- All singular values ≈ 1.0 means equal contribution from all directions
- This is NOT rank-1, it's full-rank near-identity

**Cause of original error:**
- bf16 model precision (3-4 significant digits)
- Tiny epsilon (1e-5 to 1e-6) for finite differences
- Combined to make perturbations invisible → artificial rank collapse

**New question:** What determines the magnitude of the "small_delta"?
- Is it constant across layers? → **NO.** Angular curvature ranges 0.22-1.55 rad across layers (curvature accumulation analysis, 2026-02-26).
- Does it vary by layer type (attention vs MLP)? → **YES.** Attention contributes ~37% of directional change, MLP ~63%. This ratio is remarkably constant across architectures (std < 0.01).
- Does it correlate with highway position? → **FAMILY-DEPENDENT.** Cumulative curvature ↔ ID is positive for Qwen (r=0.55-0.77) but negative for Llama (r=-0.38). See curvature accumulation section.

**Experiments:**
- [x] Verify with float32 precision ✓
- [x] Test multiple epsilon values ✓
- [x] Measure delta magnitude across layers ✓ (curvature accumulation analysis 2026-02-26)
- [x] Compare attention delta vs MLP delta ✓ (attention ~37%, MLP ~63%, universal)

---

## 2. What Determines Highway Location? — PARTIAL UNDERSTANDING (2026-02-03) `[EMPIRICAL]`

**Observation:**
- LFM2: Entry compression (layers 0-1) at 0-6% of depth
- Granite: Early compression (layers 4-24) at 11-16% of depth
- Qwen/DeepSeek: Mid compression (layers 17-28) at 44-47% of depth

**FINDINGS:**

### Factor 1: Hybrid Architecture (LFM2) ✓ CONFIRMED

LFM2's entry highway is caused by **Mamba/SSM layers**, not transformer attention:
- Layers 0, 1, 3, 4, 6, 7, 9, 11, 13, 15 = Mamba (10 total)
- Layers 2, 5, 8, 10, 12, 14 = Attention (6 total)
- The highway (layers 0-1) consists of PURE MAMBA layers

**Why Mamba creates low ID:**
- SSM is a linear recurrence: h_t = A·h_{t-1} + B·x_t
- State h_t lives in a fixed-dimensional space
- This naturally creates low-dimensional compressed representations
- Then attention layers (starting layer 2) expand for processing

### Factor 2: Model Family (Pure Transformers) — NOT GQA!

**~~GQA formula~~ `[DISPROVEN]`:** The original "GQA formula" was spurious.

Validation on Granite-8B (GQA=4):
- **Predicted:** 39% (same as Qwen3-8B with GQA=4)
- **Actual:** 11%

The pattern is actually **model family**, not GQA:

| Model | GQA | Highway | attention_bias | RoPE θ |
|-------|-----|---------|----------------|--------|
| Granite-3B | 1.0 | 16% | True | 10M |
| Granite-8B | 4.0 | 11% | True | 10M |
| Qwen2.5-3B | 8.0 | 47% | False | 1M |
| Qwen3-8B | 4.0 | 44% | False | 1M |

**Within-family, GQA has opposite effects:**
- Granite: GQA=1→16%, GQA=4→11% (higher GQA = earlier)
- Qwen: GQA=8→47%, GQA=4→44% (higher GQA = later)

### Factor 3: Layer-0 Q/K Alignment — THE GEOMETRIC CAUSE (2026-02-03)

**UPDATED after testing Llama-3.2-3B:** ~~The attention_bias hypothesis~~ `[DISPROVEN]`.

| Model | attention_bias | L0 Q/K Align | Attn Entropy L0 | Highway |
|-------|----------------|--------------|-----------------|---------|
| Qwen | False | **0.041** | 2.70 (diffuse) | 44% |
| Llama | False | **0.157** | 1.62 (selective) | 0% |
| Granite | True | **0.177** | 2.78→1.24 | 11% |

**Key finding:** Llama has NO attention_bias but EARLY highway (like Granite, unlike Qwen).
The real cause is **Q/K alignment at layer 0**, not attention_bias!

**Q/K alignment across layers:**
- Layer 0: Llama=0.157, Granite=0.177, Qwen=0.041 (4× difference!)
- Layers 2+: All models ≈ 0.02-0.04 (similar)

The difference is ONLY at layer 0. This is a **learned property**, not architectural.

**Geometric explanation:**

Q/K alignment = ||W_q @ W_k^T|| / (||W_q|| × ||W_k||)

High alignment (Llama, Granite):
- Q and K project to OVERLAPPING subspaces
- Some input directions → high attention scores, others → low
- Softmax becomes selective → information filtered → low ID

Low alignment (Qwen):
- Q and K are nearly orthogonal
- All inputs → similar (low) scores
- Softmax is diffuse → all info preserved → high ID persists

This is the same mechanism as LFM2's uniform attention (extreme low alignment).

**The causal chain:**
```
High L0 Q/K alignment → selective attention from layer 0 → early compression → EARLY highway
Low L0 Q/K alignment → diffuse attention → compression delayed → LATE highway
```

### Summary

| Architecture | GQA | L0 Align | Highway | Geometric Cause |
|-------------|-----|----------|---------|-----------------|
| Hybrid (LFM2) | 2.0 | N/A | 0-6% | SSM layers create low-dim state |
| Low GQA (Granite-3B) | 1.0 | 0.28 | 16% | K can match Q → selective |
| Medium GQA (Llama, Granite-8B) | 3-4 | 0.16-0.18 | 0-11% | Moderate compression |
| High GQA (Qwen) | 4-8 | 0.03-0.04 | 44-47% | K must compress → diverges from Q |

### Factor 4: GQA Constrains Q/K Alignment — THE ROOT CAUSE (2026-02-03)

**GQA architecturally constrains Q/K alignment:**

| Model | GQA | L0 Q/K Align | Highway |
|-------|-----|--------------|---------|
| Granite-3B | **1.0** | **0.276** | 16% |
| Llama-3.2-3B | 3.0 | 0.157 | 0% |
| Granite-8B | 4.0 | 0.177 | 11% |
| Qwen3-8B | 4.0 | 0.041 | 44% |
| Qwen2.5-3B | **8.0** | **0.030** | 47% |

**Correlation: r(log(GQA), L0_align) = -0.88**

Formula: `L0_align ≈ 0.28 - 0.12 × log(GQA)`

**Why GQA affects Q/K alignment:**
- GQA=1: K has same dimensions as Q → K can match Q's structure → HIGH alignment
- GQA=8: K has 1/8th the dimensions → K must compress → K diverges from Q → LOW alignment

**The complete causal chain:**
```
GQA (architecture)
       ↓
K capacity constrained (K_dim = Q_dim / GQA)
       ↓
K must specialize differently from Q if GQA > 1
       ↓
Low Q/K alignment at layer 0
       ↓
Diffuse attention (Q·K produces uniform scores)
       ↓
All information preserved (high ID)
       ↓
LATE highway (compression delayed)
```

**Residual variance — EXPLAINED (2026-02-03):** Same GQA can give different alignments:

| Model | GQA | Subspace Overlap | QK Alignment | Highway |
|-------|-----|------------------|--------------|---------|
| Granite-8B | 4.0 | **0.777** | 0.177 | 11% |
| Qwen3-8B | 4.0 | 0.581 | 0.041 | 44% |
| Qwen2.5-3B | 8.0 | 0.433 | 0.030 | 47% |
| Llama-3.2-3B | 3.0 | 0.705 | 0.157 | **0%** |

**Subspace overlap** = ||V_q^T @ V_k||_F / sqrt(k), where V_q, V_k are top-k right singular vectors.

**Correlation: r(Subspace Overlap, QK Alignment) = 0.933**

**The proximate cause:**
- Granite: Q and K read from **similar input directions** (0.78 overlap)
- Qwen: Q and K read from **orthogonal input directions** (0.43-0.58 overlap)
- This is a **training regime effect**, not an architectural parameter

**Causal chain — protocol audit per `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`:**

Each link is labeled with its claim state. `[VALIDATED]` requires all eight protocol fields
satisfied. Claims missing architecture/scale terms or commensurability proofs are `[EXPLORATORY]`.

```
[PROVEN]      GQA (architecture) → K capacity
              K_dim = Q_dim / GQA. Architectural identity. No empirical test needed.
                    ↓
[EXPLORATORY] Training regime → Subspace allocation
              Observed: subspace overlap correlates with QK alignment (r=0.93).
              Mechanism for WHY training produces these allocations: NOT DERIVED.
              Architecture term: MISSING. Scale term: MISSING.
                    ↓
[EXPLORATORY] Subspace overlap → QK alignment (r=0.93, 4 models, 3 families)
              Geometry argument: overlapping Q/K subspaces → higher inner product →
              higher alignment scores. Argument is geometrically motivated.
              Formal derivation from attention mechanics: MISSING.
              Scale term: 3B–8B only.
                    ↓
[EXPLORATORY] QK alignment → Attention selectivity → Highway location
              Geometry argument: near-orthogonal Q,K → near-zero QK scores → uniform
              softmax → diffuse attention → late ID compression → late highway.
              Argument is derivable from the attention operator but has NOT been formalized.
              Architecture term: not conditioned on hybrid vs pure-attention families.
              Scale term: MISSING.
                    ↓
[PROVEN]       Attention selectivity ↔ Entropy
               Entropy = -Σ p_i log p_i. Selective weights → concentrated distribution
               → low entropy. By definition of Shannon entropy once "selectivity" is
               defined as weight concentration. No free assumption.
                    ↓
[EXPLORATORY, r=0.507] Logit Entropy → Δcurvature
               **The "entropy" in this link is logit entropy (Entropy-Lens), NOT
               attention weight entropy.** Logit entropy measures posterior certainty
               about the next token at depth l (project h_l through unembedding).
               Attention weight entropy (Shannon entropy of softmax weights) shows NO
               significant correlation with curvature on standard transformers
               (Qwen2.5-3B: r=-0.036, p=0.835). Correlation appears only on LFM2
               hybrid architecture (r=+0.829 for θ_attn, p=0.042).
               Architecture term: ARCHITECTURE-DEPENDENT. Sublayer sign opposition
               (P1: r(H,θ_attn) opposite sign from r(H,θ_mlp)) holds for LFM2 only.
               MLP gain varies 2-43× within models (CV 0.177-0.739, validated 3/3).
               Scale term: MISSING.
               Theoretical grounding (Agarwal et al. 2026, arXiv:2512.22471v3):
               Value manifold parameterized by posterior entropy. Logit entropy (which
               captures posterior state) connects to curvature through manifold
               dimensionality. Attention weight entropy is an upstream variable that
               does not directly predict curvature on standard transformers.
               Full sublayer analysis: `docs/research/entropy_curvature_derivation.md`.
               Data: `results/entropy_curvature/entropy_curvature_results.json`.
                    ↓
[EXPLORATORY, r=0.821] Cumulative curvature → ID
               Measured correlation. Mechanism: accumulated directional change →
               higher local manifold dimensionality (TwoNN). Geometrically motivated.
               Relationship between known curvature transformations and TwoNN estimator
               behavior: NOT DERIVED.
               Architecture term: MISSING. Scale term: MISSING.
               Theoretical grounding: same Bayesian manifold interpretation applies —
               cumulative curvature accumulation = progressive Bayesian suppression steps
               (each layer ablation causes >10× error increase in wind-tunnel tasks), and
               the final ID corresponds to the dimension of the posterior-entropy-
               parameterized value manifold.
                    ↓
[PROVEN]       ID → Phases
               Phases are defined by ID trajectory shape (minima = highway,
               accumulation = processing, stabilization = exit). True by construction.
                    ↓
[EXPLORATORY, LFM2-only] Highway location → C_ex peak
               Measured: LFM2-350M and LFM2-700M confirm.
               Qwen3.5-0.8B: C_ex peaks at layer 19 (late processing), not highway.
               Architecture term: LFM2 SSM-dominated entry phase only.
               Cross-family divergence NOT predicted by a pre-registered architecture/scale
               term → MECHANISM_UNDERSPECIFIED for cross-family claim.
```

**What C_ex = S_spec - log(ID) measures (geometric description, not derivation):**
High where spectral entropy is large relative to intrinsic dimension — many active spectral
directions but a compact local manifold. Whether this NECESSARILY peaks at the highway or
merely does so for LFM2's SSM-dominated entry phase requires a formal derivation from SSM
operator properties. Without that, C_ex at highway is an LFM2 observation, not a universal
chain extension.

**What needs formal derivation before any link is promoted to `[VALIDATED]`:**

1. **Logit entropy → curvature** (weakest link): The correlation (r=0.507) uses logit
   entropy, not attention weight entropy. Sublayer decomposition (2026-03-03) shows:
   attention weight entropy has NO significant correlation with curvature on standard
   transformers (Qwen2.5-3B: r=-0.036). MLP gain varies 2-43× within models. The
   mechanism is Bayesian manifold dimensionality (Agarwal 2026): logit entropy
   parameterizes value manifold dimension → curvature. Next step: replicate logit
   entropy → curvature with sublayer decomposition on ≥3 families.
   See `docs/research/entropy_curvature_derivation.md`.

2. **Cumulative curvature → ID**: Derive the TwoNN estimator's behavior under known curvature
   transformations. The TwoNN estimator is built on the manifold hypothesis — its response to
   accumulated directional change has a theoretical derivation that has not been worked out.

3. **QK alignment → highway timing**: Formalize the geometry argument. Given alignment = 0.04
   (Qwen) vs 0.18 (Llama), derive the expected depth at which cumulative curvature crosses the
   ID inflection threshold. Requires: alignment → per-layer entropy → curvature accumulation
   formula → crossing depth. Each step derivable in principle.

4. **C_ex at highway for SSM (Mamba)**: SSM is a linear recurrence in fixed-dimensional state,
   qualitatively different from attention. Derive why the SSM-dominated highway phase (LFM2
   layers 0–1) produces high S_spec relative to ID. The mechanism differs from the attention
   argument and requires separate derivation.

**Open questions (protocol-framed):**
- [ ] What training hyperparameters determine subspace allocation?
      (Required before promotable: causal operator identifying which gradient signals drive
       Q/K subspace separation toward or away from alignment)
- [ ] Can we predict subspace overlap from training recipe?
      (Required: architecture × training_duration × data_domain terms in the prediction form)
- [ ] Why does Qwen3.5-0.8B C_ex peak at layer 19 rather than highway?
      (Required: pre-registered architecture/scale prediction distinguishing LFM2 from Qwen
       before the claim becomes cross-family)

---

## 3. Why Does RLHF Flatten Geometry? `[CONJECTURAL]`

**Observation:** Specialist models (instruct, code, reasoning) have expansion_ratio variance ≈ 0.

**Hypothesis:** RL training creates stable attractors regardless of input type.

**What we need to derive:**
- What is the loss landscape geometry under RLHF?
- Does PPO/DPO explicitly or implicitly penalize geometric variance?
- Is this a side effect of reward hacking or intentional?

**Experiments to constrain:**
- [ ] Compare base vs instruct checkpoints of same model
- [ ] Measure geometry during RLHF training (if we have checkpoints)
- [ ] Test if flat geometry is necessary or sufficient for instruction following

---

## 4. Effective Rank & Recovery — RELATIONAL STRUCTURE (2026-02-03) `[EMPIRICAL]`

**GOAL: No arbitrary constants. Everything is a ratio of measurable quantities.**

### Core Measurables

All geometry reduces to these directly measurable ratios:

| Quantity | Definition | What It Measures |
|----------|------------|------------------|
| **Gap** | S₁/S₂ | Spike prominence |
| **Decay** | (S₁₀/S₂)^(1/8) | Plateau falloff rate |
| **Convergence** | ‖μ‖/‖x-μ‖ | Mean dominance |
| **V_rank/d** | eff_rank(W_v)/d_model | Value projection capacity |
| **Spike_frac** | S₁/Σ(S) | Variance in first mode |

### Effective Rank Formula (NO arbitrary constants)

```python
def rank_from_gap_decay(gap, decay, n=20):
    """Compute effective rank from gap and decay alone."""
    h = 1 / gap  # Height of plateau relative to spike
    S = np.zeros(n)
    S[0] = 1  # Spike normalized to 1
    for i in range(1, n):
        S[i] = h * (decay ** i)  # Plateau decays geometrically
    total = np.sum(S)
    p = S / total
    H = -np.sum(p * np.log(p + 1e-10))
    return np.exp(H)
```

**Key insight:** The SV distribution is NOT Gaussian or Zipf. It's **spike + decaying plateau**:
- S₁ = spike (variance from mean direction)
- S₂...Sₙ = plateau decaying by factor `decay` per mode

### What Determines Gap

**EXIT layers:** Gap ≈ convergence² (r = 0.99 when conv > 1)
- Large convergence → mean dominates → spike → high gap
- S₁ direction aligns with mean (correlation 0.986)

**HIGHWAY layers:** Different mechanism (conv ≈ 0.2, gap still present)
- Spike comes from attention selectivity, not mean dominance
- Need separate analysis

### What Determines Decay — SOLVED (2026-02-03)

**Layer decay is a norm-weighted average of component decays. NO arbitrary constants.**

```python
def layer_decay(attn_out, mlp_out, input_act):
    """Layer decay from component norms and decays."""
    norm_attn = np.linalg.norm(attn_out, axis=1).mean()
    norm_mlp = np.linalg.norm(mlp_out, axis=1).mean()
    norm_input = np.linalg.norm(input_act, axis=1).mean()
    total = norm_attn + norm_mlp + norm_input

    alpha = norm_attn / total  # ~0.15-0.25
    beta = norm_mlp / total    # ~0.19-0.21
    gamma = norm_input / total # ~0.54-0.66 (residual dominates)

    return alpha * attn_decay + beta * mlp_decay + gamma * input_decay
```

**Validation (error < 0.005):**

| Model | Layer | Predicted | Actual | Error |
|-------|-------|-----------|--------|-------|
| Qwen2.5-3B | 34 (exit) | 0.904 | 0.909 | 0.004 |
| Qwen3-8B | 34 (exit) | 0.862 | 0.865 | 0.002 |
| Qwen2.5-3B | 18 (mid) | 0.820 | 0.806 | 0.015 |
| Qwen3-8B | 18 (mid) | 0.863 | 0.857 | 0.005 |

**Component decay characteristics:**
- **Input decay**: Inherited from previous layer (recursive)
- **MLP decay**: High (~0.87-0.92), MLP is near full-rank
- **Attn decay**: Consistent (~0.89-0.91), V_rank doesn't directly determine it

**Attention decay analysis (2026-02-03):**
| Model | V_rank/d | V_decay | Attn_pattern_decay | Attn_out_decay |
|-------|----------|---------|-------------------|----------------|
| Qwen2.5-3B | 0.123 | 0.995 | 0.859 | 0.893 |
| Qwen3-8B | 0.238 | 0.989 | 0.457 | 0.906 |
| Granite-8B | 0.236 | 0.963 | 0.886 | 0.899 |
| Llama-3.2-3B | 0.302 | 0.991 | 0.825 | 0.886 |

V projection is near full-rank (decay 0.96-0.99), so it doesn't compress.
Attention output decay is consistent despite varying V_rank and pattern_decay.

**The earlier formula (decay ≈ 0.6 × V_rank + 0.8) was NOT fundamental:**
- The 0.6 and 0.8 were emergent from the norm-weighted mixing
- True formula has zero arbitrary constants

### Recovery Ratio

**~~Old formula~~ `[DISPROVEN]` (arbitrary constants):**
```
R = 4.26/N + 1.76 + T  ← This has no geometric meaning
```

**New understanding:**
Recovery ratio = f(exit_geometry) / f(highway_geometry)

Both geometries are determined by:
1. Gap at that layer → from convergence (exit) or attention selectivity (highway)
2. Decay at that layer → from V_rank and spike_frac

**What training changes:**
| Training Type | Exit Convergence | Result |
|---------------|------------------|--------|
| Base | High (3000+) | High gap → low rank → moderate recovery |
| Instruct | Moderate (~1400) | Lower gap → higher rank → higher recovery |
| Reasoning | Low (~800) | Lowest gap → highest rank → highest recovery |

### Complete Causal Chain

```
Architecture
    ↓
GQA → K capacity constraint
    ↓
Training → Q/K subspace allocation (r=0.93 with alignment)
    ↓
Subspace overlap → QK alignment → Attention selectivity
    ↓
Selectivity → Highway location (early vs late)
    ↓
V_rank → Plateau decay rate (r=0.73)
    ↓
Training type → Exit convergence
    ↓
Convergence → Exit spectral gap
    ↓
Gap + Decay → Effective rank (no free parameters)
    ↓
Exit_rank / Highway_rank = Recovery ratio
```

### What's Still Unknown (NO arbitrary constants allowed)

- [x] ~~Why 0.6 coefficient on V_rank term in decay formula?~~ → NOT fundamental, emergent from mixing
- [x] ~~Why 0.8 base in decay formula?~~ → NOT fundamental, residual dominance (~66%)
- [x] ~~What determines attention output decay?~~ → Consistent ~0.89-0.91 across models, V projects near full-rank
- [x] ~~What determines highway gap when convergence < 1?~~ → See note below on ID vs effective rank
- [x] ~~What training hyperparameters determine exit convergence?~~ → See training analysis below
- [x] ~~Why does reasoning training reduce exit convergence?~~ → Reduces exit mean norm, see below
- [x] Why was V_rank correlated (r=0.73) with layer decay if attn_out_decay is constant? → **Statistical artifact, see below**

### V_rank Correlation: Statistical Artifact (2026-02-03)

Original r=0.73 (V_rank vs layer decay) was noise: attn_decay range is only 0.019 across
4 models (0.888-0.907). V_rank has no meaningful effect on decay within this resolution.

### Exit Convergence: Training Reduces Mean Norm (2026-02-03)

Convergence = mean_norm / dev_norm. Reasoning training reduces exit mean norm by 2.1×
(Qwen3-8B base: 2895, DeepSeek-R1: 1364) while dev_norm stays constant (~1360).
Mechanism: diverse CoT → no single "default" direction dominates → lower mean.
No arbitrary constants — both norms are directly measurable.

### Expansion Ratio — RESOLVED (2026-02-03)

`expansion_ratio = peak_norm / final_norm`. Pure transformers always have ratio=1.0
(final MLP always increases norm → peak at last layer). LFM2 hybrids can have ratio>1.0
when final Mamba layer compresses. Variance does NOT predict quality (r=-0.47, spurious
confound with model size). It is a structural signature of architecture, not reasoning.

### Important Distinction: Effective Rank vs Intrinsic Dimension (2026-02-03)

**These measure DIFFERENT properties:**

| Metric | Definition | What It Measures |
|--------|------------|------------------|
| **Effective Rank** | exp(entropy of normalized SVs) | Global variance distribution |
| **Intrinsic Dimension** | Local manifold dimensionality | Local geometric complexity |

**Key finding from Qwen3-8B analysis:**
```
Layer  | Gap | Eff.Rank | Known ID
-------|-----|----------|----------
16-33  | 1.2-1.5 | ~18 | 2-3D (highway)
35     | 1.4 | 18.1 | 6.2D (exit)
```

**The "highway" (low ID) is NOT about having a spectral spike (high gap).**

A curved manifold can have:
- Low ID: Simple local structure (few coordinates needed locally)
- High effective rank: Variance spread across many global dimensions

**Implication:** Recovery ratio relates to ID trajectory, not effective rank trajectory.

### Intrinsic Dimension: Complete Causal Chain (2026-02-03)

**ID is determined by cumulative curvature, which is determined by logit entropy (Entropy-Lens).**

**Correlations found:**
| Relationship | Correlation |
|--------------|-------------|
| Logit entropy → Δcurvature | r = 0.507 (CONTAMINATED — norm confound, see below) |
| Cumulative curvature → ID | r = 0.821 |

**The mechanism:**
- **Entropy > 0.8 (diffuse)** → ADDS curvature (+0.043 avg)
- **Entropy < 0.3 (selective)** → REMOVES curvature (-0.044 avg)

**ID trajectory explained (Qwen3-8B):**
| Layers | Entropy | Δcurv | ID | Explanation |
|--------|---------|-------|------|-------------|
| 0-11 | ~1.0 | +0.10 | 2→5 | Diffuse attention adds curvature |
| 12-21 | 0.97→0.81 | -0.05 | 5→7→6 | Transition, curvature peaks |
| 22-35 | 0.62→0.09 | -0.05 | 6→4.5 | Selective attention removes curvature |

**Complete relational chain (NO arbitrary constants):**
```
Layer position
      ↓
QK alignment (learned, correlates with GQA)
      ↓
Logit entropy (Entropy-Lens: project h_l → unembedding → softmax → Shannon H)  [measurable]
      ↓
Δcurvature = curvature(attn_out) - curvature(attn_in)  [measurable]
      ↓  (r = 0.507)
Cumulative curvature = 1 - (top-2 variance fraction in local neighborhoods)
      ↓  (r = 0.821)
Intrinsic Dimension (MLE estimator from nearest neighbor ratios)
```

**Why logit entropy predicts Δcurvature (derived mechanism, D3.1–D3.5):**
- High H_logit_norm: attention is diffuse → D3.1 (centroid magnitude reduction) → r↓.
  D3.2 (centroid tangentiality) → sin(α)↑. D3.4: r-dominance → net θ↓.
- Low H_logit_norm: attention is concentrated → larger ||δ|| (D3.1 converse) → r↑ → θ↑.
- Architecture conditioning (D3.5): f_attn = fraction of QK-attention layers determines
  whether r or sin(α) absorbs the entropy coupling. Pure QK → r. Hybrid → sin(α).
- Note: This is NOT direct attention mixing. Attention weight entropy is upstream and
  architecture-dependent in its effect on curvature (see `entropy-curvature-derivation.md`).

**Critical clarification (2026-03-03):** The "entropy" in this chain is **logit entropy**
(Entropy-Lens), NOT attention weight entropy. Sublayer decomposition experiments show
attention weight entropy has NO significant correlation with angular curvature on standard
transformers (Qwen2.5-3B: r=-0.036, p=0.835). The mechanism is architecture-dependent:
sign opposition between attention and MLP sublayer correlations holds only for LFM2
hybrid architecture. MLP angular gain varies 2-43× within models (validated 3/3).
Full analysis: `docs/research/entropy_curvature_derivation.md`.

**Theoretical grounding (Agarwal et al. 2026, arXiv:2512.22471v3):**
In transformers that minimize cross-entropy (Theorem 1: CE minimizer = Bayesian posterior
predictive), the value manifold's dimensionality is parameterized by posterior entropy — at
the final checkpoint representations lie on a 1D manifold with entropy as coordinate. The
logit entropy→Δcurvature direction follows: logit entropy measures posterior certainty =
how many dimensions the value manifold needs = how much curvature accumulates per layer.
Attention weight entropy is upstream and architecture-dependent in its effect.
Full technical mapping: `docs/research/bayesian_geometry_connection.md`.

**Operator split resolved (2026-03-03, CR-EC-001):** The operator-split experiment
(`scripts/entropy_curvature_operator_split.py`) decisively confirms H_logit as the
primary entropy operator for curvature coupling on standard transformers:

| Model | r(H_logit, θ_attn) | r(H_attn, θ_attn) | r(H_logit, H_attn) |
|-------|--------------------|--------------------|---------------------|
| LFM2-700M | **+0.943** (p=0.005) | +0.829 (p=0.042) | +0.657 |
| Qwen3.5-0.8B | +0.600 | -0.429 | +0.086 |
| Qwen3.5-4B | +0.503 (p=0.003) | — | — |

The two operators are barely correlated on standard transformers (r=-0.086 to -0.299).

**F5 depth confound identified (2026-03-04):** The raw sign inconsistency (LFM2 negative,
Qwen positive) is a depth confound. Both H_logit and θ_total trend with depth, creating
spurious raw correlations. After depth control with derived detection floor (Fisher-SE MDE +
Bretherton 1999 autocorrelation correction): 7/10 models resolvable, all show **negative**
sign across 4 architecture families. F5 status: **CONSISTENT_SIGN** (threshold DERIVED).
Below floor: LFM2-700M, Qwen2.5-3B, Qwen3-8B (high autocorrelation → low n_eff → large MDE).

**10-model evidence (updated 2026-03-04, with proper Qwen3.5 GatedDeltaNet decomposition):**
F1 PASS 4/4, F3 PASS, F5 CONSISTENT_SIGN (7/10 resolvable, all negative, 4 families).
Gate check 10/10. Mechanism prediction 9/10 (Qwen3-8B sole mismatch). LFM2
competing_sublayers, Qwen3.5 core_pass_through, Llama/Mistral core_pass_through.
Qwen3.5 scale-validated (0.8B+2B+4B, all resolvable, all negative).
GQA conditioning (operator coupling): Spearman(GQA, r(H_logit, H_attn)) = -0.736, p=0.015, n=10.

**New falsifier closure (6-model rerun, 2026-03-04):**
F2_geometry_conditioned_E_mix = PASS (resolvable geometry effect in Qwen2.5-3B and
Llama-3.2-3B; others below measurement floor, not failures).

**B5 refinement (norm-coupling path):**
Spearman(GQA, R²(H_logit -> ||h||²)) = -0.632, permutation p=0.250, n=4 attention families:
NON_MONOTONE. Same-GQA counterexample: Qwen3-8B and Qwen3.5-0.8B both have GQA=4 but
R²=0.274 vs 0.000. Therefore B5 is two-variable (GQA + core operator type), not GQA-only.

CR-EC-001 remains [EMPIRICAL] (architecture-term gap still open).
Full results: `results/entropy_curvature_operator_split/`,
`results/f5_sign_law_analysis_6models/`, `results/gqa_norm_entropy_coupling/`.
Derivation: `docs/research/entropy-curvature-derivation.md` (Propositions B1-B3 proven,
B4-B5 exploratory; two-path framework with GQA+operator-conditioned cancellation).

**D3 tangential/radial decomposition (2026-03-04, formal derivation):**
The mechanism for negative r(H_logit_norm, θ) is now formally derived:
- D3.1 `[PROVEN]`: Centroid magnitude reduction (convexity) → r↓ as H↑
- D3.2 `[PROVEN]`: Centroid tangentiality (concentration of measure) → sin(α)↑ as H↑
- D3.3 `[PROVEN under A7]`: CE chain-rule selection bias → concentrated α receives score
  drift for above-average radial tokens (`r_t > R`)
- D3.4 `[PROVEN]`: r-dominance — O(√T) vs O(1), r wins the product θ ≈ r·sin(α)
- D3.5 `[PROVEN]`: Architecture conditioning — f_attn determines coupling strength
Remaining open item: validate A7 (radial-dominant downstream gradient) on real models.
All measured signs match across 3 architectures. See `entropy-curvature-derivation.md`.
Next falsifier protocol: `docs/research/ENTROPY-CURVATURE-GQA-FALSIFIER-PROTOCOL.md`.

**NORM CONFOUND DISCOVERED (2026-03-04):** The Entropy-Lens does NOT apply the model's
final RMSNorm before unembedding projection. Since `h @ W.T = ||h|| × (ĥ @ W.T)`,
softmax sharpness scales with ||h||, creating r(H_logit, ||h||²) ≈ -0.99 — a measurement
operator artifact, not a geometric coupling.

After adding H_logit_norm (RMSNorm applied before unembedding):
- Prediction 1 (r(H_logit_norm, ||h||²) ≈ 0): PASS for LFM2 (r=-0.065), FAIL for Qwen
  (r=-0.686, -0.552). RMSNorm reduces but doesn't eliminate norm coupling in Qwen.
- Prediction 2 (sign of r(H_logit_norm, θ_total | depth)): CONSISTENT NEGATIVE across
  3 families (LFM2: -0.390, Qwen3.5: -0.145, Qwen2.5: -0.468).
- Prediction 3 (r(H_logit_norm, H_logit) < 0.9): PASS for all (0.221, 0.689, 0.554).

**The r=0.507 was a confound artifact. The true sign is NEGATIVE.** Higher normalized
posterior entropy → less curvature. This reverses the chain mechanism prediction and
suggests the intuition was wrong: uncertain posterior → centroid-like output → LESS
angular displacement, not more.

Artifact: `results/entropy_curvature_operator_split/*/operator_split.json` (includes
both H_logit and H_logit_norm measurements).

**Causal perturbation test (2026-03-04, CORRECTED):** Direct causal intervention — boost
prefix attention weights by multiplier M, measure per-layer angular curvature change Δθ,
test Spearman(ΔH, Δθ) with exact permutation test (n=6 layers, all 720 permutations
enumerated) and Holm-Bonferroni correction across all M values with measurable |Δθ|.

| Model | Result | Best |ρ| | Sign | Raw p | Holm threshold | Testable M |
|-------|--------|---------|------|-------|----------------|------------|
| LFM2-350M | **FALSIFIED** | 0.771 | − | 0.103 | 0.000019 | 2696 |
| Qwen3.5-0.8B | **FALSIFIED** | 0.886 | + | 0.033 | 0.000446 | 112 |

**Both models FALSIFIED.** Initial run (before code review) reported Qwen as "NOT
FALSIFIED" (p=0.026) — false positive from: (1) no multiple-testing correction across
112 M values, (2) random-shuffle permutation instead of exact enumeration. After
Holm-Bonferroni, neither model's best p survives correction.

**Interpretation:** H_attn → curvature causal link is NOT supported by direct intervention.
Changing attention weight entropy does not produce statistically significant curvature changes
in either architecture. This is consistent with: (a) H_attn having no correlation with
curvature on standard transformers (Qwen2.5-3B: r=-0.036, p=0.835), (b) the norm confound
discovery showing r=0.507 was contaminated by ||h||² trends. The entropy→curvature link,
if real, operates through H_logit (posterior certainty), not H_attn (attention weight
redistribution). Direct attention perturbation is the wrong intervention for H_logit.

Artifact: `results/attention_validation/perturbation_experiment_corrected.txt`.

**GQA norm-entropy coupling B5 test (2026-03-04):** R²(H_logit → log||h||²) vs GQA ratio.

| Model | GQA | R²(H→||h||²) |
|-------|-----|---------------|
| Llama-3.2-3B | 3 | 0.826 |
| Qwen3-8B | 4 | 0.274 |
| Qwen3.5-0.8B | 4 | 0.000 |
| Qwen2.5-3B | 8 | 0.035 |

Spearman(GQA, R²) = -0.632; analytic p=0.368, exact-permutation p=0.250 (N=4, not significant). Direction consistent with B5
(higher GQA → weaker coupling) but insufficient power. LFM2-350M (GQA=N/A, hybrid)
excluded. B5 remains `[EXPLORATORY]` — needs more GQA-varied models.
Artifact: `results/gqa_norm_entropy_coupling/coupling_results.json`.

---

## 5. MLP Nonlinearity Geometry — SOLVED (2026-02-03) `[EMPIRICAL]`

**Question:** How does geometry change through the MLP nonlinearity?

MLP structure (gated, Llama/Qwen style):
```
gate = SiLU(W_gate @ h)
up = W_up @ h
h_intermediate = gate * up
h_out = W_down @ h_intermediate
```

### Key Finding: Gate × Up Multiplication is the Key Transformation

**SiLU activation has minimal geometric effect.** The elementwise multiplication is what matters.

Results from Qwen3-8B Layer 18 (mid-network):

| Stage | Eff.Rank | Gap | Decay | Sparsity | Curvature | ID |
|-------|----------|-----|-------|----------|-----------|-----|
| MLP input | 18.2 | 1.1 | 0.915 | 0.010 | 0.505 | 9.2 |
| Gate (pre-SiLU) | 18.1 | 1.2 | 0.913 | 0.006 | 0.492 | 9.1 |
| Gate (post-SiLU) | 18.1 | 1.2 | 0.913 | 0.005 | 0.492 | 9.5 |
| **Gate × Up** | **18.5** | 1.1 | **0.921** | **0.032** | **0.521** | 9.4 |
| MLP output | 18.4 | 1.2 | 0.928 | 0.009 | 0.516 | 9.3 |

### The Geometric Effects

**1. SiLU has negligible effect:**
- Pre-SiLU → Post-SiLU: Δsparsity = -0.001, Δcurvature = 0.000
- The activation function preserves geometry

**2. Gate × Up multiplication creates sparsity:**
- Sparsity jumps: 0.005 → 0.032 (6.4× increase)
- When gate values are small, the product is very small regardless of up values
- This is soft gating: SiLU(x) × y ≈ 0 when x << 0

**3. Gate × Up adds curvature:**
- Curvature increases: 0.492 → 0.521 (+0.029)
- Elementwise multiplication creates nonlinear combinations of linear projections
- This is the source of MLP's representational power

**4. Down projection reduces sparsity:**
- Sparsity drops: 0.032 → 0.009
- Linear mixing of sparse representations spreads activations
- But curvature is preserved: 0.521 → 0.516

### Summary

Gate × Up is a learned bilinear form: `SiLU(W_gate @ h) × (W_up @ h)`. This is the
only nonlinear operation that adds curvature (+0.03/layer, consistent across Qwen3-8B
and Llama-3.2-3B). SiLU alone: Δcurvature ≈ 0. Down projection: rotates back, mixes
sparsity. **The MLP's geometric role:** Add curvature through bilinear gating.

---

## 6. Attention Eigenvalue Distribution — INITIAL RESULTS `[EMPIRICAL]`

**Theoretical context (2026-02-22):** "Mind the Gap" (Noci et al., ICML 2024) shows via RMT that softmax attention matrices exhibit a **spectral gap** (largest SV separates from bulk) that drives rank collapse. The gap depends on dot-product statistics (d_k, normalization, positional encoding), not head count alone — explaining why GQA fails as a standalone predictor of effective rank.

**Finding (2026-02-03):** Attention matrices have dramatically lower effective rank than random.

| Model | Attn Eff. Rank | Entropy | Random | Rank Reduction |
|-------|----------------|---------|--------|----------------|
| LFM2-350M | **1.02** | 2.40 | 6.95 | **85%** |
| Qwen2.5-3B-Instruct | 3.85 | 1.48 | 6.95 | 45% |
| Qwen3-8B | 2.76 | 1.55 | 6.95 | **60%** |
| DeepSeek-R1-8B | 2.74 | 1.58 | 6.95 | **61%** |

**Key observations:**

1. **LFM2 attention is essentially rank-1** (eff_rank ≈ 1.02, spectral gap ~10^7)
   - This directly explains rank-1 Jacobians in LFM2
   - But entropy is HIGH (2.40 > random 1.93) - all rows attend similarly but broadly

2. **Qwen3 base is sharper than Qwen2.5-Instruct** (2.76 vs 3.85)
   - Counter-intuitive: specialist training doesn't always sharpen attention
   - The Qwen3 architecture itself may be responsible

3. **DeepSeek-R1 ≈ Qwen3 base** (2.74 vs 2.76)
   - Reasoning training didn't change attention rank
   - The geometry changes must be elsewhere (MLP? layer interactions?)

4. **Exit layers have near-zero entropy** (Qwen3/DeepSeek layers 29-35: entropy 0.04-0.26)
   - Final layers concentrate attention on 1-2 tokens
   - This is where the prediction is formed
   - Different heads focus on different tokens → rank > 1

**SOLVED PUZZLES:**

1. **LFM2: rank-1 + entropy > 0.8 (near-uniform) — EXPLAINED**

   LFM2 attention is **perfectly uniform**:
   ```
   [0.091 0.091 0.091 0.091 0.091 ...]
   [0.091 0.091 0.091 0.091 0.091 ...]
   [0.091 0.091 0.091 0.091 0.091 ...]
   ```
   - Row similarity = 1.0 (all rows identical) → rank-1
   - Uniform distribution over tokens → maximum entropy
   - **LFM2 attention has degenerated to mean-pooling**

   This explains rank-1 Jacobians: the attention contributes a rank-1 term because
   every position receives the same weighted sum of all positions.

   **Why does this happen?** LFM2 is a hybrid Mamba/attention architecture.
   The Mamba (SSM) layers may handle sequence modeling, leaving attention
   to simply aggregate information uniformly. Needs verification.

2. **Rank-1 Jacobian was a numerical artifact** — see §1 above. True Jacobians are
   full-rank near-identity (eff_rank=63.9, all σ ≈ 1.0).

3. **LFM2 uniform attention: Q/K orthogonality** — W_q^T @ W_k ≈ 0 (||Q@K^T|| = 1-2
   vs Qwen's 14.75). Mamba handles sequence modeling → attention receives no selectivity
   gradient → Q/K drift to orthogonality → uniform softmax = mean-pooling.

**REMAINING PUZZLES — PARTIALLY EXPLAINED (2026-02-22):**

1. **Why Qwen3 sharper than Qwen2.5?** Three factors: QK-Norm (removes magnitude
   broadening), no QKV bias, 2× training tokens. Architecture sets capacity; training
   determines usage. GQA × QK-Norm × training_duration → attention spectrum.
   Full analysis: `docs/research/architecture_geometry_theory.md` §5.

**Remaining experiments:**
- [x] Test more architectures (Qwen3, DeepSeek)
- [ ] Test at different training checkpoints
- [ ] Analytically derive Jacobian rank from attention rank
- [ ] Test pre-trained vs random initialization
- [ ] Layer-wise entropy trajectory analysis

---

## 6. Manifold Topology — INITIAL RESULTS (2026-02-03) `[EMPIRICAL]`

**Question:** What is the topology of the activation manifold? Does it change through the network?

### Key Finding: Topology is PRESERVED Through All Layers

Using persistent homology via ripser on LFM2-350M:

| Metric | Observation | Interpretation |
|--------|-------------|----------------|
| **β₀ (components)** | Stable at 2-4 throughout | Tokens remain topologically distinct |
| **β₁ (loops)** | ≈0 always | No persistent circular structures |
| **β₂ (voids)** | =0 always | No 3D holes |
| **Simplification ratio** | =1.00 | No topology change |

### Persistence Entropy Shows Highway Pattern

Persistence entropy H dips at highway (1.3-1.5) vs entry/exit (1.6-2.1), matching the
ID trajectory. Math prompts show brief β₁ loops at pre-highway layers.

### β₁ (Loops) Appear During Math Processing — [EMPIRICAL] (2026-02-03)

> **Note:** β₁ > 0 during math is an observed pattern (3 models). The separate claim
> that Δβ₁ *predicts reasoning correctness* was [DISPROVEN] (2026-02-22, 3/6 tests FAIL).
> Loops form during math; they do not reliably separate correct from incorrect.

**Cross-architecture validation:**

| Model | Math Prompts β₁ | Narrative Prompts β₁ |
|-------|-----------------|----------------------|
| LFM2-350M | 1-5 (peaks at layer 11) | 0 throughout |
| Llama-3.2-3B-Instruct | 1-2 (layers 3-27) | 0 throughout |
| Qwen3-8B | 1-4 (peaks at layer 22-23) | 0 throughout |

**Observed across 3 models (LFM2-350M, Llama-3.2-3B, Qwen3-8B):**
- **Math/reasoning prompts** → β₁ > 0 (topological loops form)
- **Narrative/descriptive prompts** → β₁ = 0 (no loops)

**Specific observations by prompt complexity:**

| Prompt | Max β₁ | Peak Layers | Interpretation |
|--------|--------|-------------|----------------|
| "What is 2+2?" | 1 | 6-7 | Simple arithmetic, brief loop |
| "15 times 7" | 1 | 7-11 (LFM2), 18-34 (Qwen) | Multiplication requires more steps |
| "3x + 5 = 20" | 4 | 22-23 (Qwen) | Algebra creates multiple relational loops |
| "train travels..." | 5 | 11 (LFM2), 22 (Qwen) | Word problem = complex relational structure |

**Geometric interpretation:**

A loop (β₁ > 0) in the token manifold means there exists a cycle in the nearest-neighbor graph of token representations. For reasoning:

1. Token A relates to Token B (e.g., "3x" relates to "=")
2. Token B relates to Token C (e.g., "=" relates to "20")
3. Token C relates back to Token A (e.g., "20" informs the value of "x" in "3x")

This circular dependency is the topological signature of **relational reasoning** — tokens must reference each other to compute the answer.

**Why narrative prompts have β₁ = 0:**
- "The quick brown fox" — sequential, no back-references
- "Once upon a time" — each token depends only on preceding context
- No circular dependencies → no loops in the manifold

**This is NOT an artifact of prompt length:**
- "train travels..." (16 tokens) has β₁=5
- "Once upon a time" (4 tokens) has β₁=0
- Length doesn't predict topology; semantic structure does

### ~~Δβ₁ Predicts Reasoning Success~~ [DISPROVEN: beta1_falsification, 2026-02-22]

Falsified (3/6 tests FAIL, n=50, LFM2-350M). F3 decisive: no metric shows significant
correct/incorrect separation. Original observation (3 prompts, 2 models) was underpowered.
Full report: `results/beta1_falsification/full/LFM2-350M/FALSIFICATION_REPORT.md`.

### Geometric Interpretation

Transformers preserve topology (β₀ constant, no tears/merges) while transforming geometry
(stretching, compressing, rotating). Consistent with near-identity Jacobians.
β₀ ≈ number of tokens (residual connections preserve individual token identity).

### Methodological Note: Zigzag Persistence

Current per-layer Ripser loses birth-death tracking across layers. Zigzag persistence
(Carlsson & de Silva 2010) would track features across the layer sequence. Not implemented.

### Experiments Completed

- [x] Compute persistent homology of activations at each layer
- [x] Track Betti numbers across layers
- [x] Compare topology across architectures (LFM2, Llama, Qwen — pattern holds)
- [x] Test if semantic categories occupy topologically distinct regions → **YES: reasoning creates loops, narrative doesn't**
- [x] Full falsification protocol (F1-F7, n=50) → **Δβ₁ as reasoning predictor DISPROVEN**
- [ ] Upgrade to zigzag persistence for cross-layer birth-death tracking

### Tools Created

`scripts/manifold_topology.py` — Computes persistent homology trajectory for any backend model:
```bash
poetry run python scripts/manifold_topology.py /path/to/model \
  --prompts "prompt 1" "prompt 2" \
  --output results.json
```

Uses PCA to reduce to 50 dimensions before ripser (standard TDA practice for high-dim data).

### Remaining Questions

- **Does β₁ > 0 on specific prompts predict anything?** (reasoning structure?)
- **Why does persistence entropy dip at the highway?**
- **Is topology preserved across ALL architectures?** (test Qwen, Llama)
- **Do semantically related tokens form distinct components?**

---

## 7. Layer-wise Invariants — PARTIALLY RESOLVED (2026-03-03)

**Question:** What properties are preserved vs transformed across layers?

Six hypotheses pre-registered, tested on 3 models (LFM2-350M, LFM2-700M, Qwen3.5-0.8B).
Data from `results/layer_invariants/`. Script: `scripts/layer_invariant_analysis.py`.

### Pre-Registered Hypotheses and Results

| # | Hypothesis | Test | 350M | 700M | 0.8B | Verdict |
|---|-----------|------|------|------|------|---------|
| I1 | Norm monotonicity | Spearman(layer, ‖h_l‖) > 0, p < 0.01 | r=0.69, p=3e-3 | r=0.93, p=2e-7 | r=0.96, p=7e-14 | **`[VALIDATED]`** |
| I2 | ID phase-invariance | ANOVA of ID by phase, p < 0.01 | untestable | F=8.01, p=0.014 | untestable | `[INCONCLUSIVE]` |
| I3 | Cosine preservation phase-conditional | highway cos > processing cos | untestable | REFUTED (diff=-0.007) | REFUTED (diff=-0.024) | **`[REFUTED]`** |
| I4 | Residual ratio phase-conditional | highway ratio < processing ratio | untestable | REFUTED (diff=+0.068) | REFUTED (diff=+0.166) | **`[REFUTED]`** |
| I5 | S_spec separates phases | ANOVA of S_spec by phase, p < 0.01 | untestable | REFUTED (F=0.88, p=0.37) | untestable | **`[REFUTED]`** |
| I6 | Effective rank tracks ID | Spearman(exp(S_spec), ID) > 0, p < 0.01 | r=0.72, p=2e-3 | r=-0.01, p=0.97 | r=0.48, p=0.02 | `[EMPIRICAL]` (1/3) |

"Untestable" = ANOVA/permutation test requires ≥2 phase groups with ≥2 members; model has only 1.

### Key Finding: Phase Labels Do Not Predict Bypass Metrics `[PROVEN]`

The "highway" phase label (assigned by ID trajectory in the information bridge) does NOT
predict residual-stream bypass behavior:
- Highway layers have residual ratio **≥** processing layers, not < (I4 refuted on 2/2 testable models)
- Highway layers have cosine preservation **≤** processing layers, not > (I3 refuted on 2/2 testable models)

This means the phase detection algorithm identifies ID-based regimes, but these regimes
correspond to **different** properties than what "highway" connotes in the residual stream
literature (where highway = skip connection dominates). The ID-based phase partition and
the residual-bypass partition are independent structures.

### Confirmed Invariant: Norm Growth `[VALIDATED]`

‖h_l‖ increases monotonically with depth (Spearman r > 0.69, p < 0.003 on all 3 models).
Strongest on dense architectures (Qwen3.5-0.8B: r=0.96). This is a geometric consequence
of the residual stream: h_{l+1} = h_l + F_l(h_l), where F_l has positive dot product with
h_l on average. The norm cannot decrease unless F_l opposes h_l strongly enough to
overcome the Pythagorean component.

### Phase-Conditional Summary (LFM2-700M, the only model with testable phase structure)

| Quantity | Highway (2 layers) | Processing (13 layers) | Exit (1 layer) |
|----------|--------------------|----------------------|----------------|
| S_spec (nats) | 3.03 ± 0.42 | 2.68 ± 0.46 | 3.28 |
| ID | 10.46 ± 0.06 | 11.23 ± 0.35 | 11.68 |
| Residual ratio | 0.83 ± 0.14 | 0.76 ± 0.68 | — |
| Cosine change | 0.85 ± 0.09 | 0.85 ± 0.11 | — |
| ‖h_l‖ | 0.75 ± 0.22 | 2.07 ± 1.39 | — |

### Remaining Questions

1. **Why does I6 hold on LFM2-350M but not LFM2-700M?** The effective rank–ID correlation is
   model-dependent. Possible explanation: LFM2-350M's smaller hidden dimension (d=1024) makes
   spectral entropy and TwoNN ID track similar low-dimensional structure, while at d=2048
   (LFM2-700M) they measure different aspects of the geometry.

2. **What is the right phase partition for bypass metrics?** The ID-based partition doesn't
   predict residual behavior. A residual-ratio-based partition (k-means on ‖F_l‖/‖h_l‖)
   might be more informative but would be post-hoc, not pre-registered.

3. **Norm growth rate:** Is d‖h_l‖/dl constant, linear in l, or architecture-dependent?
   The data suggests approximately linear growth (r=0.93–0.96 on larger models), but the
   rate varies by architecture (LFM2-350M has a norm anomaly at L6→L7 where ‖F‖/‖h‖=10.46).

---

## 8. Training Dynamics → Geometry `[CONJECTURAL]`

**Unknown:** How do training hyperparameters affect final geometry?

**Key hyperparameters:**
- Learning rate
- Batch size
- Weight decay
- Warmup schedule
- Dropout

**What we'd need:**
- Train same architecture with varied hyperparameters
- Measure final geometry
- Find functional relationships

**This is expensive but would give us predictive power.**

---

## 9. Information-Theoretic Characterization — PARTIALLY RESOLVED (2026-03-03)

**Question:** What information-theoretic invariant is preserved across layers and across
architectures?

### Resolution: Sigma Calibration + DPI Violation Mechanism

The measurement commensurability problem (previously `[MEASUREMENT_INVALID]`) is resolved.

**Sigma calibration (Regime 5):** Constraint-satisfaction calibration finds the σ interval
where ALL layers have non-degenerate Gram matrices (S₂ bounded away from 0 and log₂(N)),
then picks the geometric midpoint. All 3 models have wide feasible intervals — none is
intrinsically multi-scale. Design doc: `docs/research/sigma_calibration_design.md`.

| Model | σ* | Feasible Interval | Predictions Passed |
|-------|-----|-------------------|--------------------|
| LFM2-350M | 0.928 | [0.070, 12.228] | 3/9 |
| LFM2-700M | 1.744 | [0.097, 31.265] | 4/9 |
| Qwen3.5-0.8B | 1.602 | [0.050, 51.578] | 2/9 |

### Cross-Model Prediction Summary

| Prediction | LFM2-350M | LFM2-700M | Qwen3.5-0.8B | Verdict |
|------------|-----------|-----------|---------------|---------|
| P1: CKA decays with \|i-j\| | CONFIRMED | CONFIRMED | CONFIRMED | `[VALIDATED]` 3/3 |
| P2: Rényi MI decays with \|i-j\| | REFUTED | REFUTED | REFUTED | `[DISPROVEN]` 0/3 |
| P6: DPI holds at fixed σ | REFUTED | REFUTED | REFUTED | `[DISPROVEN]` 0/3 (explained below) |
| P7: C_ex peaks at highway | CONFIRMED | CONFIRMED | REFUTED | `[EMPIRICAL]` 2/3 LFM2 only |
| P8: CKA shows phase blocks | CONFIRMED | CONFIRMED | CONFIRMED | `[VALIDATED]` 3/3 |

### DPI Violation Mechanism (Normalized Matrix-Rényi Pipeline) `[PROVEN]` + `[VALIDATED]`

**Why DPI fails in the L2-normalized matrix-Rényi MI pipeline:**

Note: standard DPI (Cover & Thomas) applies to Shannon MI of random variables.
Matrix-based Rényi MI (Giraldo et al. 2014) is a kernel functional, not Shannon MI.
The claims below are scoped to this specific measurement pipeline.

The unnormalized chain h₀ → h₁ → ... → h_L is Markov (h_{l+1} = h_l + F_l(h_l) is
deterministic). For any MI functional satisfying DPI on deterministic channels,
I(h₀; h_{l+1}) ≤ I(h₀; h_l). `[PROVEN]`

Regime 5 L2-normalizes: X̃_l = h_l / ‖h_l‖. Given only X̃_l, h_l cannot be
reconstructed (scale is lost). So X̃_{l+1} is NOT a function of X̃_l alone —
the normalized chain is **not Markov**. DPI need not hold for any MI functional
evaluated on the normalized representations. `[PROVEN]`
(X̃_{l+1} = (h_l + F_l(h_l)) / ‖h_l + F_l(h_l)‖ requires knowing ‖h_l‖, not
just h_l / ‖h_l‖.)

**Empirical confirmation:** DPI violation magnitude correlates with residual bypass
strength (‖F_l(h_l)‖ / ‖h_l‖) across all 3 models. `[VALIDATED]`

| Model | Layers | ρ(\|Δ_l\|, residual_ratio) | p-value |
|-------|--------|---------------------------|---------|
| LFM2-350M | 16 | 0.849 | 6.25e-05 |
| LFM2-700M | 16 | 0.693 | 4.19e-03 |
| Qwen3.5-0.8B | 24 | 0.735 | 6.43e-05 |

Layers that change the representation more (higher ‖F_l‖/‖h_l‖) produce larger
MI changes in the normalized chain. The signed correlation is near zero (direction
of MI change is not predicted by bypass strength, only magnitude).

### Status Tags

- `[VALIDATED]` CKA depth-distance decay (P1, 3/3 models)
- `[VALIDATED]` CKA phase block structure (P8, 3/3 models)
- `[VALIDATED]` DPI violation ↔ bypass strength correlation (3/3 models, p < 0.01)
- `[PROVEN]` Unnormalized chain is Markov → DPI holds for true MI
- `[PROVEN]` L2 normalization breaks Markov property → DPI violations are genuine
- `[EMPIRICAL]` C_ex highway peak (P7, LFM2 family only, 2/3)
- `[DISPROVEN]` Rényi MI decays with layer distance (P2, 0/3)
- `[DISPROVEN]` DPI holds for normalized kernel MI (P6, 0/3, mechanism explained)

### Remaining Open Questions

1. **Signed violation direction:** What determines whether MI increases or decreases at
   a given layer? The bypass magnitude predicts |Δ| but not sign(Δ).
2. **P2 refutation:** MI does NOT decay with |i-j|. What DOES the all-pairs MI matrix
   structure reflect? (May connect to Section 7: layer-wise invariants.)
3. **C_ex universality:** Is the C_ex highway peak an LFM2 architectural property
   (hybrid attention-convolution) or does it appear in other families?

### Data and Artifacts

- Calibration: `results/information_bridge/{LFM2-350M,LFM2-700M,Qwen3.5-0.8B}/`
- DPI analysis: `results/dpi_analysis/{LFM2-350M,LFM2-700M,Qwen3.5-0.8B}/`
- Scripts: `scripts/information_bridge_experiment.py`, `scripts/dpi_violation_analysis.py`
- Calibration design: `docs/research/sigma_calibration_design.md`
- Derivation: `docs/research/information_bridge_derivation.md`
- Replacement observable derivation: `docs/research/linear_accessible_information_derivation.md`

---

## 10. The Fundamental Question — PARTIALLY UNDERSTOOD (2026-02-22) `[CONJECTURAL]`

**Can we write down an equation that predicts geometry from architecture?**

**Answer: Not a single equation, but decomposable into tractable sub-problems.**

### What We Learned

The "GQA formula" was a spurious correlation. Validation on Granite-8B (GQA=4) showed:
- Predicted: 39%
- Actual: 11%

The pattern is **model family**, not GQA ratio. Direct `architecture → geometry` mapping fails because training regime effects mediate the relationship.

### What We Can Predict (Qualitatively)

| Architecture Type | Highway Position |
|------------------|------------------|
| Hybrid (SSM + attention) | Entry (0-10%) |
| Granite family | Early (11-16%) |
| Qwen family | Mid (44-47%) |

But we can't predict which family a new architecture will behave like.

### Theoretical Frameworks (2026-02-22)

Three established frameworks provide partial predictions. See `docs/research/architecture_geometry_theory.md` for full analysis.

**1. Signal Propagation (Mean-Field Dynamics):**
- Residual network variance propagation: `Var(x_L) = Var(x_0) · ∏(1 + α_l² · χ_l)`
- Critical scaling: `α ~ 1/√L` (variance preservation)
- **Prediction:** Highway = ordered phase (α²χ ≈ 0), processing = mildly chaotic (α²χ > 0)
- **Testable:** Measure α²χ per layer, correlate with ID trajectory

**2. Random Matrix Theory (Marchenko-Pastur):**
- Weight SVs inside MP bulk = noise, outside = signal
- Spectrum method (in Axolotl) uses this for layer selection
- **ModelCypher alternative:** Shannon effective rank (no distributional assumption)
- **Testable:** Compare MP-SNR layer selection against tail_dims > 0 targeting

**3. Attention Rank Saturation:**
- Critical head dimension: d_h = Ω(log n) for full expressiveness
- Upper bound on effective rank: ~0.63n (approaches 1 - 1/e)
- **ModelCypher data:** All models operate at 15-56% of theoretical maximum
- **Testable:** Track attention utilization = eff_rank / (0.63n) per layer

### Regime Decomposition (Proposed)

Instead of direct `architecture → geometry`, decompose into three sub-problems:

| Sub-problem | Input | Output | Status |
|-------------|-------|--------|--------|
| **Regime prediction** | Architecture params (d_model, GQA, QK-Norm, etc.) | Geometric regime class (entry-highway, sandglass, long-highway) | Qualitative only (need more model families) |
| **Rank budget** | Weight SVD spectra | Per-layer tail_dims | **Operational** |
| **Phase classification** | Activation measurements (ID, entropy, curvature) | Per-layer phase (ordered/transitional/chaotic) | **Operational** (from ID trajectory + entropy) |

**Why this decomposition helps:** Each sub-problem is tractable individually. Regime prediction is discrete classification (3-5 categories), not continuous regression. The other two are already implemented in ModelCypher.

### Quantitative Targets for Regime Boundaries

From attention rank saturation + signal propagation theory:

```
Attention utilization = eff_rank / (0.63 × n)
  < 0.2 → likely ordered phase (highway)
  > 0.4 → active processing

Signal propagation: α²·χ per layer [DISPROVEN: see §R1]
  Mean-field theory does not apply to trained networks.
  α²·χ has no predictive power for phase classification (Spearman 0/5 pass).
```

### Candidate Causal Factors

The Granite vs Qwen difference correlates with:
1. ~~**attention_bias**~~ `[DISPROVEN]`: Llama has no bias but early highway like Granite
2. ~~**RoPE theta**~~ `[DISPROVEN]`: Similar locality despite 10× difference
3. **QK-Norm**: Qwen3 has it, Qwen2.5/Granite/Llama don't (affects attention spectrum)
4. **Training regime**: Subspace overlap (r=0.93 with alignment) is a learned property
5. **Training duration**: 36T (Qwen3) vs 18T (Qwen2.5) → more specialized subspace allocation

### What We Still Can't Predict

- **Highway position**: Qualitative family-level predictions only; quantitative requires subspace overlap which is a training outcome
- **Attention rank**: Know QK-Norm + training duration affect it (Qwen3 vs Qwen2.5), but no formula
- **Expansion ratio variance**: Architectural (hybrid vs transformer), not quality-related

### ~~Depth/Width Ratio Hypothesis (2026-02-22)~~ `[DISPROVEN: see §R3]`

**TESTED 2026-02-26.** 5 models, 60 probes, 10 pairwise comparisons. Partial Spearman(L/d | L) = 0.018, p = 0.96. L alone (r = 0.515) predicts ID trajectory shape; the ratio L/d adds zero information after controlling for depth. Family effects dominate (same-family Procrustes = 0.18 vs cross-family = 1.38).

**Why L/d fails:** L = processing stages (each adds curvature). d = representational capacity. Since ID << d (peak ~10 vs min d = 1024), width is never the bottleneck. The CompleteP result applies to training stability, not trained geometry.

### The Path Forward

1. ~~**Measure signal propagation regimes**~~ `[TESTED, REFUTED — §R1]`: α²χ does not predict phases
2. ~~**Compare MP-SNR vs tail_dims**~~ `[TESTED, REFUTED — §R2]`: MP model wrong for learned attention
3. **More model families**: Test Llama, Mistral, Phi to build regime classification training data
4. **Controlled experiments**: Train same architecture with/without QK-Norm to isolate its effect
5. ~~**Test L/d scaling**~~ `[TESTED, REFUTED — §R3]`: L/d has zero signal after controlling for L

### Lessons Learned

Three data points aren't enough. The GQA formula had R²=0.941 but was completely wrong.
Always validate on held-out data before claiming a relationship.

**Theories for random/initialized networks do not apply to trained networks** (2026-02-26). Mean-field α²χ, Marchenko-Pastur spectral predictions, and L/d scaling were all derived for random initialization or infinite-width limits. Trained models have learned structure that violates these assumptions. The correct framework is the empirical causal chain: GQA → QK alignment → entropy → curvature → ID → phases.

---

## 11. Step Size from Geometry — PARTIALLY SOLVED (2026-02-22) `[EMPIRICAL]`

**Question:** Given a Cayley-Stiefel preconditioned optimizer on the Stiefel manifold with per-layer Weyl perturbation constraints, what geometric quantity correctly determines step size?

### The Failure: Lipschitz LR Derivation `[DISPROVEN]`

The original approach — η ≤ 2/(L × λ_max(P)) where L = λ_max(Hessian) via central-difference HVP + power iteration — is fundamentally broken.

**Ablation evidence (2026-02-22):**

| Exp | Config | LR | Result (from 18/25 baseline) |
|-----|--------|-----|-----|
| 0 | Default (CE+REINFORCE) | 0.996 | 5/25 (-13) |
| 1 | CE-only | 1.64 | 13/25 (-5) |
| 2 | LR/10 | 0.072 | 16/25 (-2) |
| 3 | LR/100 | 0.0037 | 17/25 (-1) |
| 8 | 10-batch Lipschitz | 1.13 | 11/25 (-7) |

**Root cause:** The loss surface has (L₀,L₁)-relaxed smoothness (Zhang et al. ICLR 2020) — local Lipschitz constant correlates positively with gradient norm and varies by orders of magnitude across minibatches. Central-difference HVP measurements span 3 OOM (0.1 to 193). Median of 3-OOM-spread noise is still noise.

### MASS: The Implemented Solution

**MASS (Measured-Adaptive Step Size)** replaces curvature estimation with per-step measurement + geometric bounds:

```
eta_step = min(eta_ceiling, eta_sps, eta_weyl)
```

| Layer | Formula | Source | What It Bounds |
|-------|---------|--------|----------------|
| Static ceiling | `σ_k_min / σ_max` | Weyl perturbation theory | Total adapter contribution relative to base model |
| SPS | `f(x_t) / \|\|d_t\|\|²` | Loizou et al. 2020 | Per-step rate from actual loss and preconditioned gradient |
| Weyl displacement | `σ_k_min / \|\|d_t\|\|` | Weyl 1912 | Per-step displacement relative to crossing threshold |
| Val backoff | `eta_ceiling *= val_loss_ratio` | Measured | Floor at √ε_f32 |

**Why MASS works:** SPS measures what the loss surface actually allows at each step — no curvature estimation needed. The Weyl bounds provide geometric safety rails independent of loss landscape smoothness.

### Open Questions

**Q11.1: Per-layer vs global η** `[CONJECTURAL]`

MASS uses global `σ_k_min` and `σ_max` (minimums/maximums across all LoRA layers). Per-layer ceilings `η_ceiling_i = σ_k_i / σ_max_i` would respect per-layer geometry but adds complexity (separate step sizes per layer).

- When does this matter? Likely when layers have very different condition numbers.
- The Cayley-Stiefel preconditioner already adapts per-layer (P = M M^T per layer). Does this make per-layer η redundant?
- Experiment: compare global vs per-layer ceiling on 350M.

**Q11.2: √N budget distribution** `[EMPIRICALLY CONFIRMED]`

**Validated (2026-02-22).** Per-step Weyl ceiling alone is insufficient. Over N steps per epoch, accumulated displacement scales as √N × per_step_displacement (Brownian scaling).

Without √N correction (ceiling = σ_k_min/σ_max = 0.1064): catastrophic overfitting. Repetition 60%, entropy collapsing, adapter saturation 67% in 1 epoch.

With √N correction (ceiling /= √N = 0.0157): healthy training. Monotonically decreasing val_loss, modest repetition, CKA min=0.965.

```
eta_ceiling = σ_k_min / (σ_max × √N)
```

**SPS does NOT sidestep this** — SPS is non-binding because its f* = 0 assumption is wrong for fine-tuning (loss is never near zero). The ceiling is the active constraint. √N correction is implemented in `mlx_training_adapter.py` (MASS √N budget block).

**REINFORCE gradient compounding — RESOLVED (2026-02-22):**

The 3× gap between √N-corrected ceiling (0.012) and empirical sweet spot (0.004) was caused by REINFORCE drawing from the same Weyl budget as CE without being accounted for. CE consumed `update_norm` of the `sigma_k_min` budget; REINFORCE then added N_re more steps, exceeding the total displacement bound.

Fix: Weyl remainder budget. After CE phase, REINFORCE gets `(sigma_k_min - update_norm) / sqrt(N_re)` per step. If CE exhausts the budget, REINFORCE is skipped entirely. Every quantity is measured or from SVD — no new hyperparameters.

**Remaining questions:**
- Is √N the right scaling, or should it be N^α for some α ∈ (0.5, 1)?
- Per-epoch vs per-training budget: should √N use steps per epoch or total steps?

**Q11.3: SPS and (L₀,L₁)-relaxed smoothness** `[CONJECTURAL]`

Zhang et al. (ICLR 2020): local smoothness L(θ) = L₀ + L₁ × ||∇f(θ)||.

SPS: η = f(x) / ||d||². When ||d|| is large, η naturally decreases.

- SPS's ||d||² dependence provides automatic gradient-norm-dependent scaling
- But SPS depends on loss (numerator), not gradient norm directly
- Is f(x)/||d||² ≤ 1/L(θ) guaranteed? Under what conditions?
- The L₁ term suggests η should scale as 1/(L₀ + L₁||g||). SPS scales as f/||d||². These are the same only if f ∝ ||g|| (approximately true near a quadratic).

**Q11.4: Convergence of min(ceiling, SPS, Weyl)** `[CONJECTURAL]`

Each MASS component has individual convergence properties (SPS converges for convex objectives; Weyl bounds are safety constraints). The min of three convergent sequences converges. But:

- Does the min unnecessarily slow convergence?
- Which component binds in practice? (Ceiling binds on all tested 350M runs — measure on larger models)
- Is there a regime where all three give contradictory signals?

### Fallback Candidates (Not Implemented)

If MASS proves insufficient at larger scale, two alternatives have been analyzed:

1. **D-Adaptation** (Defazio & Mishchenko, ICML 2023): Derives LR from distance-to-solution ||θ* - θ₀||. No curvature estimation. Would need adaptation for Stiefel manifold (geodesic distance vs Euclidean).

2. **Spectral-norm step control** (Muon-inspired): Bound ||δW||₂ ≤ c × σ_k directly. Per-layer natural. c derivable from Weyl: c = spectral_gap / σ_k.

See `docs/research/lr_derivation_analysis.md` for full analysis.

---


## Experiment Refutations — Moved

This section moved to keep this file under the one-shot review budget.

See [OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md](OPEN-MATHEMATICAL-QUESTIONS-REFUTATIONS.md).
