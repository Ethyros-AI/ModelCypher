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
[EXPLORATORY, r=0.507] Entropy → Δcurvature
               Measured correlation only. Causal operator: NOT DERIVED.
               Proposed mechanism: diffuse attention mixes many token directions →
               output spans more local directions → higher curvature. Geometrically
               motivated but not formalized from attention + MLP mechanics.
               Architecture term: MISSING. Scale term: MISSING.
                    ↓
[EXPLORATORY, r=0.821] Cumulative curvature → ID
               Measured correlation. Mechanism: accumulated directional change →
               higher local manifold dimensionality (TwoNN). Geometrically motivated.
               Relationship between known curvature transformations and TwoNN estimator
               behavior: NOT DERIVED.
               Architecture term: MISSING. Scale term: MISSING.
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

1. **Entropy → curvature** (weakest link): Derive from the attention operator. Given uniform
   attention weights α_ij = 1/T (diffuse), output_i = mean(V). Given concentrated weights
   (α_ij ≈ δ_{jk}), output_i ≈ V_k. How does the geometry of {output_i}_i over the input
   distribution change between these cases? This is computable from attention mechanics.

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

**Original finding:** V_rank correlated with layer decay (r=0.73)

**Investigation results:**

| Model | V_rank/d | Attn Decay | Layer Decay |
|-------|----------|------------|-------------|
| Qwen2.5-3B | 0.123 | 0.907 | 0.914 |
| Qwen3-8B | 0.238 | 0.892 | 0.912 |
| Granite-8B | 0.236 | 0.888 | 0.918 |
| Llama-3.2-3B | 0.302 | 0.891 | 0.922 |

**The paradox:** V_rank correlates NEGATIVELY with attn_decay (r=-0.88) but POSITIVELY with layer_decay (r=0.63). This contradicts the layer decay = weighted average formula.

**Resolution:** The variation is noise.
- Attention decay range: 0.888-0.907 (Δ=0.019)
- Layer decay range: 0.912-0.922 (Δ=0.011)
- With only 4 data points and ~1% variation, correlations are unstable

**Conclusion:** The original r=0.73 was likely a statistical artifact. The true relationship is:
- Attention output decay ≈ 0.89-0.91 (approximately constant)
- Layer decay ≈ 0.91-0.92 (approximately constant)
- V_rank has no meaningful effect on decay within this resolution

### Exit Convergence: Training Reduces Mean Norm (2026-02-03)

**Convergence = mean_norm / dev_norm**

Comparing same architecture (Qwen3-8B) with different training:

| Layer | Base Mean | Base Dev | Reason Mean | Reason Dev |
|-------|-----------|----------|-------------|------------|
| 0 | 9.1 | 7.9 | 9.6 | 8.7 |
| 18 | 138.3 | 260.3 | 215.3 | 409.4 |
| 35 | **2894.9** | 1355.7 | **1364.1** | 1369.2 |

**Key finding:** Reasoning training reduces EXIT MEAN NORM by 2.1×, not deviation norm.

| Model | Exit Mean | Exit Dev | Convergence |
|-------|-----------|----------|-------------|
| Qwen3-8B (base) | 2895 | 1356 | 2.14 |
| DeepSeek-R1-Qwen3 (reasoning) | 1364 | 1369 | 1.00 |

**Why reasoning reduces mean norm:**
1. Mean direction = "average next token prediction"
2. Base models predict common continuations → activations cluster toward common token embeddings
3. Reasoning models generate diverse CoT → no single "default" direction dominates
4. Lower mean = activations spread more uniformly in direction space

**The causal chain:**
```
Training diversity → Output token diversity → Exit mean norm
                                                    ↓
                                          Convergence = mean/dev
```

**No arbitrary constants.** Mean norm is measurable, dev norm is measurable,
convergence is their ratio.

### Expansion Ratio: Determined by Final Layer Behavior (2026-02-03)

**Definition:** `expansion_ratio = peak_norm / final_norm`

**Key finding:** Expansion ratio depends on whether the last layer increases or decreases norm.

| Model | Last Layer Type | Final Δnorm | Expansion Ratio |
|-------|-----------------|-------------|-----------------|
| LFM2 (retrieval) | Mamba (SSM) | -1.0 | 1.056 |
| LFM2 (reasoning) | Mamba (SSM) | +2.2 | 1.000 |
| Qwen3 (any task) | Transformer | +2448 | 1.000 |

**Why pure transformers have expansion_ratio = 1.0:**
- Final transformer layer always increases norm (MLP expansion)
- Peak is always at the last layer
- Therefore peak = final → ratio = 1.0

**Why LFM2 can have expansion_ratio > 1.0:**
- Final layer is Mamba (SSM), which can compress
- Some tasks cause the final Mamba layer to decrease norm
- Peak is at second-to-last layer
- Therefore peak > final → ratio > 1.0

**Why RLHF "flattens" expansion_ratio:**
- Not actually flattening - it was already 1.0 in pure transformers
- Qwen2.5-Instruct showed slight variance because some prompts peaked at layer 35 instead of 36

**The complete picture:**
```
Architecture (Mamba vs Transformer final layer)
              ↓
Final layer behavior (compress vs expand)
              ↓
Peak location (last layer vs earlier)
              ↓
expansion_ratio = peak / final
```

**Variance across tasks comes from:**
1. Task content affecting final layer compression (hybrid architectures only)
2. Pure transformers: no variance (always ratio = 1.0)

### Expansion Ratio Variance vs Benchmark Performance — RESOLVED (2026-02-03)

**Question:** Does expansion_ratio variance correlate with benchmark performance?

**Results from 6 models, 8 task types:**

| Model | Type | Variance | MMLU |
|-------|------|----------|------|
| LFM2-350M | hybrid | 0.027 | 35% |
| LFM2-700M | hybrid | 0.000 | 42% |
| LFM2-1.2B | hybrid | 0.001 | 55% |
| Qwen2.5-3B | transformer | 0.017 | 65% |
| Qwen3-8B | transformer | 0.000 | 70% |
| Llama-3.2-3B | transformer | 0.000 | 63% |

**Correlations:**
- r(variance, MMLU) = -0.47 (overall)
- r(variance, MMLU) = -0.74 (hybrids only)

**Key findings:**

1. **Variance does NOT predict quality.** The negative correlation is spurious.

2. **Variance is confounded with model size:**
   - Within LFM2 family: larger models are more stable (lower variance) AND smarter (higher MMLU)
   - This is because larger models have more parameters to regularize behavior

3. **Most transformers have variance = 0:**
   - Qwen3-8B, Llama-3.2-3B: peak always at last layer
   - Qwen2.5-3B: anomaly — peak at layer 35/36 for most prompts

4. **The Qwen2.5-3B anomaly:**
   - Creative writing: ratio = 1.42 (significant compression in last layer)
   - Factual/logic: ratio = 1.0 (normal)
   - This may reflect training recipe differences, not quality

**Conclusion:** Expansion_ratio variance is a **structural signature**, not a quality predictor.
- High variance = hybrid architecture with small model size
- Zero variance = pure transformer OR large hybrid model
- Does NOT indicate reasoning quality

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

**ID is determined by cumulative curvature, which is determined by attention entropy.**

**Correlations found:**
| Relationship | Correlation |
|--------------|-------------|
| Attention entropy → Δcurvature | r = 0.507 |
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
Attention entropy = -Σ p log p / log(T)  [measurable]
      ↓
Δcurvature = curvature(attn_out) - curvature(attn_in)  [measurable]
      ↓  (r = 0.507)
Cumulative curvature = 1 - (top-2 variance fraction in local neighborhoods)
      ↓  (r = 0.821)
Intrinsic Dimension (MLE estimator from nearest neighbor ratios)
```

**Why entropy predicts Δcurvature:**
- Diffuse attention (entropy ≈ 1): Mixes many token representations → output spans many local directions → curvature increases
- Selective attention (entropy < 0.3): Focuses on few tokens → output constrained to subspace → curvature decreases

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

### Why Gate × Up is Geometrically Special

The gate × up operation is a **learned bilinear form**:

```
h_intermediate[i] = SiLU(W_gate[i,:] @ h) × (W_up[i,:] @ h)
                  ≈ SiLU(a_i) × b_i
```

where `a_i` and `b_i` are linear projections of h.

**Geometric interpretation:**
- Two linear projections define two subspaces
- SiLU gates one based on its magnitude
- Product combines them nonlinearly
- Result: manifold gains curvature (can't be approximated by hyperplane)

**This is why MLPs add representational capacity:**
- Attention is approximately linear (softmax → linear combination of V)
- MLP's gate × up is fundamentally nonlinear
- The curvature increase (+0.03 per layer) accumulates

### Consistent Across Models and Layers

Llama-3.2-3B Layer 14 (mid-network):

| Stage | Sparsity | Curvature | ID |
|-------|----------|-----------|-----|
| MLP input | 0.005 | 0.474 | 10.7 |
| Gate (post-SiLU) | 0.004 | 0.477 | 10.5 |
| **Gate × Up** | **0.024** | **0.518** | 10.6 |
| MLP output | 0.004 | 0.499 | 10.9 |

Same pattern:
- SiLU: Δcurvature ≈ 0
- Gate × Up: Δcurvature ≈ +0.04, Δsparsity ≈ +0.02
- Down proj: Δcurvature ≈ -0.02, Δsparsity ≈ -0.02

### Summary

**No arbitrary constants.** The geometry changes are determined by the algebra:

```
Operation              | Geometric Effect
-----------------------|------------------
Linear projection      | Rotates/scales (preserves linearity)
SiLU activation        | Negligible (continuous + monotonic)
Elementwise multiply   | Creates curvature (bilinear form)
Down projection        | Rotates back, mixes sparsity
```

**The MLP's geometric role:** Add curvature to the representation through bilinear gating.

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

2. **CORRECTION: Rank-1 Jacobian was a numerical artifact!** (2026-02-03)

   Previous finding (WRONG): Jacobians are rank-1 in trained transformers.

   **Actual finding:** When measured correctly (float32, ε=1e-3 to 1e-4):
   ```
   ε=1e-03: eff_rank=63.9, σ_max=1.08, σ_2=1.02
   ε=1e-04: eff_rank=63.9, σ_max=1.10, σ_2=1.05
   ```

   The true layer Jacobian is:
   - **Full rank** (~64 effective rank for 64-probe measurement)
   - **Near-identity** (all singular values ≈ 1.0)
   - Each layer makes small incremental changes

   The rank-1 result was caused by:
   - bf16 model precision (3-4 significant digits)
   - Tiny finite difference epsilon (1e-5 to 1e-6)
   - These combined to make small perturbations invisible

   **Correct interpretation:** Transformer layers are approximately identity
   transformations with small incremental modifications. The "semantic highway"
   is real (residual connections dominate), but it's not rank-1 - it's full-rank
   near-identity.

3. **Why does LFM2 learn uniform attention? — EXPLAINED** (2026-02-03)

   **Answer:** Q and K projection matrices converge to **nearly orthogonal subspaces**.

   **Evidence:**
   | Layer | ||Q@K^T|| | ||Q@K^T|| (Qwen) |
   |-------|----------|------------------|
   | LFM2 | 1.0-2.0 | - |
   | Qwen | - | **14.75** (7.4×) |

   When W_q^T @ W_k ≈ 0, attention scores are near-zero regardless of input.
   Softmax of near-zero uniform scores → uniform attention.

   **Why does training converge to this?**
   1. LFM2 is a hybrid: 10 Mamba layers + 6 Attention layers
   2. Mamba handles sequence modeling (token dependencies)
   3. Attention layers receive no gradient signal to be selective
   4. Q/K projections drift toward orthogonality (stable attractor)
   5. Result: attention = mean-pooling

   **This is emergent specialization, not a bug.** The model learned that
   global averaging is sufficient when Mamba handles sequence modeling.

**REMAINING PUZZLES — PARTIALLY EXPLAINED (2026-02-22):**

1. **Why does Qwen3 have sharper attention than Qwen2.5?** — THREE CONTRIBUTING FACTORS

   | Feature | Qwen2.5 | Qwen3 | Effect on Sharpness |
   |---------|---------|-------|---------------------|
   | QK-Norm | No | **Yes** | Removes magnitude-based broadening → allows sharper selectivity |
   | QKV bias | Yes | **No** | Removes constant-offset component → less attention diffusion |
   | Training tokens | ~18T | ~36T | More training → more specialized Q/K subspace allocation |
   | GQA | 8.0 (3B) | 4.0 (8B) | Lower GQA → more K capacity (should diffuse, but doesn't) |

   **Why QK-Norm sharpens attention:** Without normalization, attention scores depend on both Q/K direction AND magnitude. QK-Norm removes magnitude dependence → selectivity is purely directional → trained attention can be more discriminating.

   **Why lower GQA doesn't diffuse:** GQA=4 gives K more capacity than GQA=8, which SHOULD allow higher alignment and broader attention. But Qwen3's measured subspace overlap (0.581) is HIGHER than Qwen2.5 (0.433). The combination of QK-Norm + 2× training duration shifts how Q/K use their capacity — they become more aligned but more selective.

   **Key insight:** Architecture parameters (QK-Norm, bias, GQA) set the *capacity* for attention sharpness. Training regime determines how that capacity is *used*. Qwen3's architectural choices enable sharper attention; extended training realizes it.

   See `docs/research/architecture_geometry_theory.md` §5 for full analysis.

2. **GQA and attention sharpness relationship** — NOT A SIMPLE FUNCTION
   - Qwen2.5 (GQA=8) has higher attention rank than Qwen3 (GQA=4)
   - This is NOT counter-intuitive once QK-Norm is accounted for
   - GQA constrains K capacity (fewer parameters), but QK-Norm changes how that capacity is used
   - The relationship is: GQA × QK-Norm × training_duration → attention spectrum
   - No single architecture parameter determines attention sharpness

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

### Detailed Layer-by-Layer Results

**Prompt: "The quick brown fox" (4 tokens):**
```
Layer  0: β₀=2, β₁=0, β₂=0, H=1.61
Layer  7: β₀=2, β₁=0, β₂=0, H=1.27  ← Highway dip
Layer 15: β₀=2, β₁=0, β₂=0, H=1.58
```

**Prompt: "What is 2+2?" (7 tokens):**
```
Layer  0: β₀=4, β₁=0, β₂=0, H=2.05
Layer  6: β₀=4, β₁=1, β₂=0, H=2.10  ← Brief loop appears!
Layer  7: β₀=4, β₁=0, β₂=0, H=1.39  ← Highway dip
Layer 15: β₀=4, β₁=0, β₂=0, H=2.00
```

### Persistence Entropy Shows Highway Pattern

The persistence entropy (H) follows the same "dip" pattern as expansion_ratio:

| Phase | Layers | H (persistence entropy) | Interpretation |
|-------|--------|-------------------------|----------------|
| Entry | 0-6 | ~1.6-2.1 | High feature diversity |
| Highway | 7-10 | ~1.3-1.5 | Concentrated features |
| Exit | 11-15 | ~1.4-2.0 | Diversification for output |

**This matches the intrinsic dimension trajectory** — the highway compresses representation complexity.

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

> **ARCHIVAL NOTE [2026-02-22]:** The claim below was subjected to a pre-registered
> falsification protocol (6 tests, 50 samples, LFM2-350M) and failed 3/6 tests.
> See `results/beta1_falsification/full/LFM2-350M/FALSIFICATION_REPORT.md` for the
> full report. Preserved as historical record only. Do not cite as current findings.

**Original hypothesis (2026-02-03):** Models that fail at math would show β₁ = 0 even on math prompts.

**Original observation:** Weak models still show β₁ > 0, but Δβ₁ sign appeared to correlate with correctness on a small sample (3 prompts, 2 models).

**Falsification results (2026-02-22, LFM2-350M, n=50, 58% accuracy):**

| Test | Result | Detail |
|------|--------|--------|
| F1: Metric robustness | **FAIL** | 70% agreement across metrics (threshold: 80%) |
| F2: Generality | INCONCLUSIVE | 0 degenerate samples |
| F3: Held-out replication | **FAIL** | No metric shows significant correct/incorrect separation (all CIs include 0) |
| F5: Subsample stability | **FAIL** | 57.8% sign stability (threshold: 80%) |
| F6: Null-shuffle control | PASS | Shuffling destroys signal (d=0.22) |
| F7: Layer-window calibration | PASS | 2/3 windows show d>0.3 |

**Verdict:** The original observation was based on too few samples (3 prompts) and did not survive robustness controls. F3 (the core claim — Δβ₁ separates correct from incorrect) fails across all 4 independent metrics. F5 shows the sign of Δβ₁ is unstable under token subsampling, indicating sensitivity to point cloud composition rather than genuine topological signal.

**What survived:** F6 confirms temporal structure matters (shuffling destroys whatever signal exists), and F7 confirms the signal is not specific to one window choice. But F3's failure is decisive — there is no statistically significant separation between correct and incorrect outputs.

~~**The pattern:**~~
~~- **Δβ₁ > 0** (loops grow toward exit) → **Correct reasoning**~~
~~- **Δβ₁ < 0** (loops collapse before exit) → **Incorrect reasoning**~~

**Original data (retained for context, not current findings):**

| Model | Answer | Early β₁ (L0-10) | Late β₁ (L-5 to end) | Δβ₁ |
|-------|--------|------------------|----------------------|-----|
| Qwen3-8B | ✓ x=5 | 1.2 | 3.8 | **+2.62** |
| Qwen-0.5B | ✗ x=10 | 2.6 | 2.0 | **-0.64** |

### Geometric Interpretation

**Transformers preserve topology while transforming geometry.**

The activation manifold:
- **Stretches and compresses** (changing intrinsic dimension)
- **Rotates** (changing which directions have variance)
- **DOES NOT tear or merge** (β₀ constant, no new holes persist)

This is consistent with the near-identity Jacobian finding — each layer makes small incremental changes without topological surgery.

### Why β₀ = number of tokens

The connected components (β₀) roughly equals the number of tokens because:
1. Each token starts as a distinct point in embedding space
2. Attention mixes representations but doesn't fully merge them
3. Residual connections preserve individual token identity

**Prediction:** Longer sequences should show higher β₀ (more components).

### Methodological Upgrade: Zigzag Persistence (2026-02-22)

Current implementation (`scripts/manifold_topology.py`) computes persistent homology per layer independently, then compares β₁ across layers to get Δβ₁. This loses birth-death tracking — a loop at layer 10 and a loop at layer 11 may or may not be "the same" feature.

**Zigzag persistence** (Carlsson & de Silva 2010) tracks topological features as they're born, persist, and die *across* a sequence of spaces (layers). This gives:
- Birth-death pairs for each β₁ feature across layer depth
- Persistence diagrams indexed by layer (not just per-layer snapshots)
- Direct measurement of "later-born features are more long-lived" (reported in LLM zigzag persistence studies)

**Implication for Δβ₁:** Instead of computing `mean(β₁[-5:]) - mean(β₁[:10])`, zigzag persistence would give the actual persistence of each loop — how many layers it survives. Correct reasoning should show loops with high persistence (born early, die late or never). Incorrect reasoning should show loops with low persistence (born, die quickly).

**Status:** Not implemented. Would require replacing per-layer Ripser calls with a zigzag persistence library (e.g., Dionysus 2, or zigzag module in GUDHI).

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

## 7. Layer-wise Invariants `[CONJECTURAL]`

**Unknown:** What properties are preserved vs transformed across layers?

**Candidates:**
- Norm (preserved? scaled?)
- Angles between vectors (preserved?)
- Rank of activation matrix (preserved?)
- Intrinsic dimension (varies - but by how much?)

**Experiments:**
- [ ] Track multiple geometric properties layer-by-layer
- [ ] Identify which are approximately preserved
- [ ] Derive why certain properties must be preserved given architecture

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

## 9. Information-Theoretic Characterization — NOT RESOLVED (2026-03-03) `[MEASUREMENT_INVALID]`

**Question:** What information-theoretic invariant is preserved across layers and across
architectures?

### Bedrock Diagnosis

This thread is not resolved. The prior mixed outcomes were promoted too early.
Cross-model MI conclusions remain blocked until measurement commensurability is proven.

Current split:

1. **Stable signal:** CKA depth-distance decay is validated cross-model.
2. **Family-level signal:** `C_ex` highway peak appears in LFM2, not universal.
3. **Blocked signal:** Rényi MI cross-model promotion is measurement-invalid under current
   bandwidth calibration assumptions.

### Why the MI Claims Are Blocked

Kernel MI uses bandwidth `sigma`. When `sigma` calibration is not commensurable across compared
layers/models, `I_2(X_0, X_l)` becomes dominated by bandwidth regime artifacts rather than causal
geometry.

Observed mechanism from current runs:

1. Depth strongly predicts sigma growth (dominant term in multiple models).
2. Architecture transitions can create local residual jumps, but these are secondary once depth
   scale is accounted for.
3. Claims that ignore this decomposition mix causal terms and measurement terms.

This means the right model is not "architecture only" and not "noise only"; it is a coupled
depth + architecture measurement problem that must be specified before interpretation.

### Required Claim Contract (Before Any New MI Promotion)

No MI claim is promotable without:

1. `observable = f(geometry_state, architecture_state, scale_state, measurement_operator)`
2. Explicit depth/scale term
3. Explicit architecture/operator-family term
4. Commensurability proof for kernel calibration
5. Directional prediction registered before run
6. Falsifier

Source of truth: `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`.

### Immediate Next Tests

1. L2-normalize activations before kernel construction and re-run full prediction set.
2. Fit depth-only vs depth+architecture sigma models; reject architecture-only narratives unless
   residual variance requires architecture terms.
3. Treat boundary-local effects as residual diagnostics, not primary causal claims, unless they
   survive depth-controlled falsification.
4. Keep CKA + `C_ex` as active bedrock diagnostics while MI commensurability remains unresolved.

### Status Tags

- `[VALIDATED]` CKA depth-distance decay (cross-model)
- `[EMPIRICAL]` `C_ex` highway peak (LFM2 family)
- `[MEASUREMENT_INVALID]` cross-model Rényi MI promotion under current calibration assumptions
- `[CONJECTURAL]` architecture-conditioned sigma regime derivation
- `[CONJECTURAL]` commensurable MI operator for heterogeneous layer families

### Data and Artifacts

- Results: `results/information_bridge/{LFM2-350M,LFM2-700M,Qwen3.5-0.8B}/`
- Script: `scripts/information_bridge_experiment.py`
- Derivation: `docs/research/information_bridge_derivation.md`

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
