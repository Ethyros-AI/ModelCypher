# Open Mathematical Questions — Experiment Refutations

Extracted from `OPEN-MATHEMATICAL-QUESTIONS.md` to keep each file one-shot reviewable.

## Information Bridge Experiment — MEASUREMENT FAILURE (2026-03-03) `[MEASUREMENT_INVALID]`

**8 pre-registered predictions (P1–P8) tested across 3 models (LFM2-350M, LFM2-700M, Qwen3.5-0.8B).**

**Confirmed (survived):**
- **P1** — CKA decays with |i-j|: all 3 models, r ≈ -0.42 to -0.64, p < 1e-12. Solid.
- **P3** — CKA and Rényi MI correlate: confirmed for 350M and Qwen (r ≈ 0.29–0.30, p < 0.01). INCONCLUSIVE for 700M (p=0.038).
- **P7** — C_ex peaks at highway: confirmed for both LFM2 models. Refuted for Qwen (peak C_ex at late processing layer 19, not highway).

**Refuted (5 predictions): P2, P4, P5, P6, P8.**

**Root cause: per-layer RBF sigma creates incommensurable kernels.**

The sigma derivation uses the maximum relative gap in the sorted pairwise distance distribution — a valid data-derived scale for a single distribution, but one that produces wildly different values across heterogeneous layer types. Sigma logging confirmed:

| Model | sigma_0 | max sigma | max/min ratio | Worst jump |
|-------|---------|-----------|---------------|------------|
| LFM2-350M | 0.0485 | 3.886 | 124× | L13→L14: 23.97× |

Layer 14 in LFM2-350M is a full attention layer; its activation distribution produces sigma_l = 3.25 vs sigma_0 = 0.049. The Hadamard product K₀ ⊙ K₁₄ mixes kernels calibrated at completely different length scales. The result is not mutual information — it's a bandwidth mismatch artifact.

**What the I₂ drops mean:** Layers 14-15 in LFM2-350M (I₂ = 1.03, 1.09 bits from a baseline of ~7) and layers 8-13 in Qwen3.5-0.8B (I₂ = 0.59–0.74 bits) correlate with specific architectural layer types. Whether these drops are real information compression events or pure bandwidth artifact cannot be determined with the current measurement. The fixed-sigma I₂_fixed fails in the opposite direction (monotone non-decreasing) because sigma_0 is too small for later layers.

**P4 also had a test design error:** the original test checked `fraction_below_median` rather than actual global minimum. Fixed to check whether highway layers contain the global I₂ minimum. Refuted 3/3 under the correct test: the global minimum is at late processing layers (L14 for 350M), not at the highway.

**Chain extension confirmed:** C_ex peaks at highway for LFM2 (2/3 models). The validated chain gains: `→ C_ex peak → Highway`. Qwen failure may reflect the phase classifier not being tuned to Qwen's architecture.

**Root cause confirmed: sigma grows with depth (residual stream scale accumulation).**

- LFM2-350M: Spearman(depth, sigma) = 0.756, p = 7e-4
- LFM2-700M: Spearman(depth, sigma) = 0.915, p = 7e-7
- Qwen3.5-0.8B: Spearman(depth, sigma) = 0.476, p = 0.019 (marginal; L19 outlier sigma=13.2)

S_spec and attention type are NOT the drivers. Spearman(sigma, S_spec): 0.04, 0.34, -0.22 across the three models — no consistent signal. P9 (attention type predicts sigma): REFUTED (Qwen geometric ratio=1.563, p=0.578). Boundary-local transition test: one-sided p=0.066, not significant at p<0.01, and not independent of the depth trend.

**Fix for MI measurement:** L2-normalize activations before computing kernels. This removes depth-driven scale while preserving directional structure (geometry). Per-layer sigma comparison becomes valid when all activation vectors have unit norm. Run this before adding further predictions.

**Verdict for Rényi MI as cross-layer measure:** NOT suitable with per-layer RBF kernels on raw activations. The depth-driven scale makes sigmas incommensurable by construction. L2-normalize activations first, then retest P2/P4/P5/P6. CKA is the reliable tool in the interim.

**Artifacts:** `results/information_bridge/LFM2-350M/`, `LFM2-700M/`, `Qwen3.5-0.8B/` — each contains `predictions.json`, `report.md` (with kernel bandwidth diagnostics table and layer type column), `trajectories.json`, `cka_matrix.json`, `renyi_mi_matrix.json`.

---

## Experiment Refutations — 5/5 H1 REFUTED (2026-02-26) `[EMPIRICAL]`

Five experiments tested whether external theories (mean-field, RMT, scaling ratios, TDA, SPS) could predict the geometry of trained networks. All five refuted. **The unifying root cause: all five theories were derived for random/initialized networks and do not apply to trained networks.**

The validated causal chain (§2, §4) already explains what these experiments tried to predict:

```
GQA → K capacity → QK alignment → Attention selectivity → Entropy → Curvature → ID → Phases
Training → Exit convergence → Gap → Decay → Effective rank → Recovery ratio
```

### R1: Mean-Field α²χ Phase Classification — REFUTED `[DISPROVEN]`

**Hypothesis:** Mean-field signal propagation (De & Smith 2020) predicts α²χ ≈ 0 at highway layers and α²χ > 0 at processing layers, with Spearman(α²χ, ID_gradient) > 0.5.

**Results (5 models, 60 probes per model):**

| Model | Spearman(α²χ, ID_grad) | Highway CI includes 0? | Processing mean > 0? |
|-------|------------------------|----------------------|---------------------|
| LFM2-350M | -0.353 | No (0.05, 0.77) | Yes (0.23) |
| LFM2-1.2B | -0.365 | No (0.28, 0.83) | Yes (0.68) |
| Qwen2.5-3B | +0.221 | No (0.04, 10178) | Yes (0.04) |
| Qwen3-8B | -0.022 | No (0.06, 123) | Yes (0.05) |
| Llama-3.2-3B | -0.144 | No (4.76, 4.76) | Yes (1.28) |

**Verdict:** Spearman 0/5 pass. Highway CI 0/5 includes 0. Processing 5/5 positive.

**Root cause:** Mean-field theory assumes i.i.d. weights (initialization). In trained networks, α = ||delta||/||h_in|| is task-dependent signal routing, χ = Var(delta)/Var(h_in) reflects learned layer behavior. Their product has no physical meaning post-training. The wild CIs (0.04 to 10178 for Qwen2.5) are not precision artifacts — χ is computed from float32 activations, not finite differences. The metric simply measures something that does not predict phase structure.

**Correct framework:** Phase boundaries are determined by cumulative curvature driven by attention entropy (§2: selectivity → compression). This is the existing causal chain, not a random-initialization theory.

### R2: RMT Marchenko-Pastur Spectral Gap — REFUTED `[DISPROVEN]`

**Hypothesis:** Marchenko-Pastur (Noci et al. 2024) predicts attention spectral gap σ₁/σ₂ from architecture parameters (d_head, seq_len, QK-Norm). Models with similar gap distributions have similar attention geometry.

**Results (4 models, 30 probes per model):**

| Model | Predicted Gap | Measured Gap Range | CV |
|-------|---------------|-------------------|-----|
| Qwen3-8B | 1.00 | 1.5 – 3448 | 9.37 |
| Qwen2.5-3B | 1.65 | 2.7 – 2058 | 5.52 |
| Llama-3.2-3B | 1.65 | 12.4 – 2735 | 5.81 |
| LFM2-350M | 1.65 | 2.8 – 4.3 | 0.22 |

**Verdict:** Spearman(predicted, measured) = 0.037. RMT predicts a CONSTANT; reality varies 2200× within a single model.

**Root cause:** Marchenko-Pastur describes the spectrum of random matrices with i.i.d. entries. Post-softmax attention is row-stochastic, causal-masked, and learned. The gap at each layer is determined by QK alignment and attention entropy at that layer — learned properties, not architectural parameters. Late layers have extreme selectivity (gap 1000+); early layers are diffuse (gap 1-5). LFM2 is stable (CV=0.22) because SSM layers have no learned attention.

**Correct framework:** The spectral gap is a consequence of attention selectivity (§2: QK alignment → entropy). It carries no independent information beyond what entropy already measures.

### R3: L/d Ratio Scaling — REFUTED `[DISPROVEN]`

**Hypothesis:** ID trajectory similarity is governed by L/d ratio, not L or d independently. Models with similar L/d should have similar expansion ratios and Procrustes-aligned ID trajectories.

**Results (5 models, 60 probes, 10 pairwise comparisons):**

| Metric | Value |
|--------|-------|
| Spearman(L/d_distance, Procrustes) | -0.321 (wrong sign) |
| Spearman(L_distance, Procrustes) | +0.515 (L alone is better) |
| Partial Spearman(L/d \| L) | +0.018, p=0.96 (zero signal) |
| Same-family Procrustes (LFM2-350M↔LFM2-1.2B) | 0.181 |
| Cross-family Procrustes (LFM2-1.2B↔Qwen3-8B) | 1.380 |

**Verdict:** After controlling for L, L/d has literally zero correlation (r=0.018, p=0.96). The original L/d hypothesis is dead.

**Root cause:** L and d have geometrically different roles. L = number of processing stages (each adds/removes curvature). d = representational capacity per stage. Since ID << d for all models (peak ID ≈ 10, min d = 1024), width is never the bottleneck. The Dey et al. CompleteP result concerns training stability (covariance propagation during optimization), not the geometry of the trained model. Family effects dominate: same-architecture models are 7.6× closer in trajectory shape than cross-family models with similar L/d.

**Correct framework:** L (depth) determines number of curvature accumulation steps. Architecture family determines attention selectivity pattern (§2). Their interaction determines ID trajectory shape. The ratio L/d is a meaningless composite.

### R4: Zigzag Persistence Phase Detection — MIXED OUTCOME `[EMPIRICAL]`

**Hypothesis:** VR persistence detects topological phase boundaries aligned with ID inflection points, and math prompts produce higher H1 (loop) persistence than narrative prompts.

**Results (3 models, 30 probes — 10 per category):**

| Model | KW p-value | Boundary align | Math H1 | Narrative H1 | MW p (math>narr) |
|-------|-----------|---------------|---------|-------------|-------------------|
| LFM2-350M | 1.000 (FAIL) | 0/0 (FAIL) | 0.003 | 0.008 | 0.065 |
| Qwen3-8B | 0.0002 (PASS) | 1/1 (PASS) | 0.401 | 0.340 | 0.237 |
| Llama-3.2-3B | 0.006 (PASS) | 2/2 (PASS) | 0.044 | 0.070 | 0.743 |

**Verdict:** Phase detection 2/3 pass, boundary alignment 2/3 pass. Loop ordering 0/3 pass.

**What was confirmed:** VR persistence on per-layer point clouds detects the same geometric transitions as ID trajectory analysis. Phase boundaries align with ID inflection points. This is redundant confirmation of the ID trajectory, not an independent discovery.

**What was refuted:** Math does NOT have higher H1 persistence than narrative. In 2/3 models, the ordering is inverted (narrative > math). β₁ as a task-type predictor was already disproven (§6: 3/6 FAIL in previous testing). Loops form when manifold complexity reaches a threshold — that threshold is layer-dependent (driven by learned geometry), not task-dependent.

**Root cause:** Topological features (H1 loops) are a consequence of the ID trajectory, not a cause or independent predictor. High-ID layers have more room for stable loops; low-ID layers do not.

### R5: SPS f* from Measured Geometry — UNTESTABLE `[MEASUREMENT_INVALID]`

**Hypothesis:** Three geometric methods (RMT noise floor, exponential tail fit, signal propagation highway fraction) agree on f* within 10×, and f*>0 causes SPS to bind >10% of final-quarter iterations with better CKA.

**Results (1 run, 50 iterations, Qwen3-1.7B):**

| Method | f* Estimate | Status |
|--------|-------------|--------|
| A (RMT) | NaN | No RMT results file for Qwen3-1.7B |
| B (Exponential) | 0.000 | Only 30 post-warmup points, poor fit |
| C (Signal Prop) | NaN | Qwen3-1.7B not in Exp 1 model set |

**Root cause:** Operational — Exp 1 tested {LFM2-350M, LFM2-1.2B, Qwen2.5-3B, Qwen3-8B, Llama-3.2-3B} but not Qwen3-1.7B (the training target). No RMT noise floor analysis was run for Qwen3-1.7B.

**Critical observation:** With f*=0, SPS binds on 50/50 iterations (100%). This means eta_sps < eta_ceiling ALWAYS. Setting f*>0 reduces the SPS step size further (tighter bound), which may hurt convergence rather than help. The hypothesis that f*>0 enables SPS binding is backwards — SPS already binds without it.

**Re-run results (2026-02-26, max_iters=500, RMT wired):**

| Method | f* | final_loss | SPS binds (final 25%) | CKA |
|--------|-----|------------|----------------------|-----|
| Baseline | 0.000 | 0.0205 | 98.4% | 0.9938 |
| RMT (A) | 0.002 | 0.0189 | 100.0% | 0.9938 |
| Exponential (B) | 0.000 | — | — | — |
| Signal prop (C) | NaN | — | — | — |

SPS binds ~100% at both f*=0 and f*=0.002. CKA identical. Loss difference within noise (losses oscillate 0.01-0.09). Method B estimated f*=0 (asymptote of exponential fit). Method C failed due to model name mismatch (now fixed).

**Verdict: REFUTED.** SPS is always the binding constraint. f*>0 does not meaningfully change convergence or alignment quality. The SPS step size mechanism already operates at the floor; further tightening has no effect.

---

### Curvature Accumulation Decomposition (2026-02-26) `[EMPIRICAL]`

**Goal:** Strengthen the entropy→curvature weak link (r=0.507, only 25% variance explained) by decomposing per-layer curvature into attention and MLP contributions.

**Method:** Angular change (arccos of cosine similarity) between hidden states at sub-layer boundaries:
- **Attention curvature:** angular change from h_in to h_post_attn (after residual)
- **MLP curvature:** angular change from h_post_attn to h_out (after residual)
- **Total curvature:** angular change from h_in to h_out

Script: `scripts/curvature_accumulation_analysis.py` — 6 models, 60 probes, 4 transformer architectures.

**Results:**

| Model | cum_curv↔ID r | p | attn_frac mean | attn_frac↔ID r | p |
|-------|--------------|---|---------------|----------------|---|
| Qwen3-1.7B | **0.767** | <0.001 | 0.379 | **-0.573** | 0.001 |
| Qwen2.5-3B | **0.698** | <0.001 | 0.375 | **-0.523** | 0.001 |
| Qwen3-8B | **0.554** | <0.001 | 0.363 | -0.310 | 0.066 |
| Llama-3.2-3B | **-0.384** | 0.044 | 0.364 | **+0.532** | 0.004 |
| LFM2 (both) | N/A | — | N/A | N/A | — |

LFM2 models have 0/16 layers with decomposition (hybrid attention-convolution, no standard self_attn/mlp sub-layers).

**Three findings:**

1. **Attention fraction is universally ~37%** across all 4 transformer architectures (range: 0.363-0.379). MLP contributes ~63% of directional change per layer. The std across models is < 0.01. This is a new architectural constant.

2. **Cumulative curvature ↔ ID is family-dependent.** Positive for Qwen family (0.554-0.767), negative for Llama (-0.384). The curvature→ID mapping is not simply "more cumulative curvature = higher ID." The distribution of curvature across layers matters, and different families distribute it differently.

3. **Attention fraction ↔ ID sign flips across families.** Negative for Qwen (higher-ID layers have lower attention fraction), positive for Llama (higher-ID layers have higher attention fraction). This suggests architecturally different learning strategies.

**Per-layer component correlations with ID gradient are mostly insignificant** (p > 0.1 for most). The decomposition does not explain ID gradient variance better than total curvature alone.

**Verdict:** The decomposition does NOT close the r=0.507 weak link. The attention/MLP split is remarkably constant (~37/63), so it cannot explain variance in curvature. The entropy→curvature r=0.507 may be the true ceiling for per-layer measurements — the remaining variance likely comes from inter-layer interactions (how curvature at layer l depends on the state shaped by all previous layers).

**Positive discovery:** The ~37% attention fraction universal is new and worth investigating. It may emerge from the residual stream structure: Pre-LN transformers add attention delta and MLP delta to the residual. If both sub-layers contribute similarly-scaled perturbations, the MLP consistently dominates because it has more parameters (d×4d vs d×d per head).

---

## Priority Ranking

| Question | Tractability | Impact | Priority | Status |
|----------|--------------|--------|----------|--------|
| Highway location | High | High | **1** | **EXPLAINED** - subspace overlap→alignment→selectivity |
| Attention eigenvalues | High | High | **2** | PARTIAL - LFM2 explained |
| Jacobian structure | High | High | **3** | CORRECTED - not rank-1, is near-identity |
| Recovery ratio function | High | Medium | **4** | **[DISPROVEN]** - R=4.26/N+1.76+T has no geometric meaning (arbitrary constants). New understanding: R = f(exit_geometry) / f(highway_geometry) |
| Manifold topology | Medium | Medium | 5 | NOT STARTED |
| RLHF flattening | Low | Medium | 6 | NOT STARTED |
| Layer invariants | High | Medium | 7 | NOT STARTED |
| Training dynamics | Low | High | 8 | BLOCKED (need training runs) |
| Information theory | Medium | Medium | 9 | NOT STARTED |
| **Step size from geometry** | **High** | **Critical** | **10** | **PARTIAL — MASS implemented, open Qs remain** |

---

## Next Steps

1. **Verify attention_bias explanation** - Find model with bias=True but Qwen-like config
2. **Test more model families** - Llama, Mistral, Phi - predict highway from attention_bias
3. **Derive recovery ratio formula** - Fit functional form to size vs recovery data
4. **Persistent homology** - Compute Betti numbers across layers to understand topology

**Completed:**
- ✓ Attention eigenvalue analysis (LFM2 explained - Q/K orthogonality)
- ✓ Jacobian structure (corrected from rank-1 to near-identity)
- ✓ Hybrid architecture highway (Mamba/SSM causes entry compression)
- ✓ Pure transformer highway explained via GQA → Q/K alignment chain
- ✓ Validated on Llama-3.2-3B (downloaded and tested)

**Falsified:**
- ✗ ~~Original GQA formula (highway% = f(GQA))~~ `[DISPROVEN]` - too simplistic
- ✗ ~~RoPE theta hypothesis (similar locality despite 10x difference)~~ `[DISPROVEN]`
- ✗ ~~attention_bias hypothesis (Llama has no bias but early highway)~~ `[DISPROVEN]`
- ✗ ~~Mean-field α²χ phase prediction~~ `[DISPROVEN: 2026-02-26]` - Spearman 0/5, theory for initialization not trained nets (§R1)
- ✗ ~~Marchenko-Pastur spectral gap prediction~~ `[DISPROVEN: 2026-02-26]` - gap varies 2200×, MP constant (§R2)
- ✗ ~~L/d ratio scaling hypothesis~~ `[DISPROVEN: 2026-02-26]` - partial r=0.018 p=0.96, L alone works (§R3)
- ✗ ~~Task-type loop ordering (math > narrative H1)~~ `[DISPROVEN: 2026-02-26]` - inverted in 2/3 models (§R4)
- ✗ ~~β₁ as reasoning predictor~~ `[DISPROVEN]` - 3/6 FAIL, now 0/3 on task ordering

**The complete geometric chain:**
```
GQA (architecture)
       ↓
K capacity = Q_dim / GQA (constrained)
       ↓
High GQA → K must compress → K diverges from Q → LOW L0 alignment
Low GQA → K can match Q → HIGH L0 alignment
       ↓
L0 alignment → attention selectivity → information filtering
       ↓
Early selectivity → early compression → EARLY highway
Late selectivity → late compression → LATE highway
```

**Correlation: r(log(GQA), L0_align) = -0.88**

*The goal is to move from "we measured X" to "X must be true because Y".*
