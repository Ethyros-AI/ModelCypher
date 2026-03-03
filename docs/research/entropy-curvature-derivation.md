# Entropy -> Curvature Derivation (Population Form)

**Status:** Derivation draft (2026-03-03)  
**Scope:** Attention entropy -> angular curvature -> TwoNN ID (layer-local, population-level)  
**Taxonomy:** Mixed ([PROVEN] identities + [EXPLORATORY] operator bridge)

---

## Objective

Derive a first-principles map from attention entropy to measured layer curvature using the
actual measurement operators in this repository:

1. Attention output map: `y(x) = W_O V α(x)`
2. Curvature operator: `θ(x) = arccos(cos(h_in(x), h_out(x)))` (radians)
3. Intrinsic dimension operator: TwoNN on geodesic distances (`μ = r2/r1`)

Target claim form:

```
observable = f(geometry_state, architecture_state, scale_state, measurement_operator)
```

For this document:

- `observable`: expected angular curvature and TwoNN ID
- `geometry_state`: covariance spectrum of `y(x)` over `x ~ P`
- `architecture_state`: `W_O`, `V`, residual map, MLP Jacobian
- `scale_state`: layer width, depth, family-dependent value geometry
- `measurement_operator`: angular change + geodesic TwoNN

---

## Notation

- `x ~ P`: input distribution at a fixed layer
- `α(x) in Δ^(T-1)`: attention weights over `T` tokens (simplex)
- `H(α) = -Σ_i α_i log α_i`: Shannon entropy (nats)
- `k_eff(α) = exp(H(α))`: Shannon effective support size
- `V in R^(d_v x T)`: value columns
- `W_O in R^(d_h x d_v)`: output projection
- `y(x) = W_O V α(x) in R^(d_h)`: attention contribution before residual
- `h_in(x), h_out(x) in R^(d_h)`: pre/post layer representations
- `δ(x) = h_out(x) - h_in(x)`
- `P_perp(h) = I - hh^T / ||h||^2`: projection orthogonal to `h`
- `Σ_α = Cov_x[α(x)]`, `Σ_y = Cov_x[y(x)]`, `Σ_δ = Cov_x[δ(x)]`

---

## Assumptions (Explicit)

- **A1 [PROVEN: architecture]:** Attention output is affine in `α`: `y = W_O V α`.
- **A2 [PROVEN: definition]:** `α` lies on simplex: `α_i >= 0`, `Σ_i α_i = 1`.
- **A3 [EXPLORATORY]:** Local small-angle regime for curvature approximation:
  `||δ|| / ||h_in|| << 1` on the probe population used for layer measurements.
- **A4 [EXPLORATORY]:** Local value non-degeneracy on active supports:
  selected columns of `W_O V` are not rank-collapsed.
- **A5 [EXPLORATORY]:** Local manifold regularity for TwoNN (approximately bi-Lipschitz
  chart around sampled points).
- **A6 [EXPLORATORY]:** MLP contribution can be linearized locally via Jacobian
  `J_mlp(x)` for covariance propagation.

Any failed assumption is a direct derivation failure mode, not an interpretation issue.

---

## Step 1: Output Covariance from Entropy-Indexed Support

### Proposition 1: Covariance Pushforward [PROVEN]

For `y(x) = W_O V α(x)`:

```
Σ_y = W_O V Σ_α V^T W_O^T
```

This is linear covariance propagation; no approximation.

### Proposition 2: Entropy-Support Constraint [PROVEN]

For any simplex vector `α` with support size `s(α) = |{i: α_i > 0}|`:

```
H(α) <= log s(α)  =>  s(α) >= exp(H(α)) = k_eff(α)
```

So higher entropy enforces a larger minimum active support.

### Proposition 3: Rank Envelope [PROVEN]

Because `Σ_i α_i = 1`, simplex fluctuations live in a `T-1` dimensional affine subspace:

```
rank(Σ_α) <= T - 1
rank(Σ_y) <= min(rank(W_O V), rank(Σ_α))
```

In a local region with active support `S` of size `s`:

```
rank(Σ_α|S) <= s - 1
rank(Σ_y|S) <= min(rank((W_O V)_S), s - 1)
```

This is the first bridge: entropy lower-bounds feasible support size; support size upper-bounds
covariance rank.

---

## Step 2: Local Dimensionality Bounds from `V` and `α`

### Lemma 1: Local Affine-Hull Bound [PROVEN]

For fixed support `S`, outputs lie in the affine hull of selected columns of `W_O V`:

```
y in Aff({(W_O V)_i : i in S})
dim_local(y) <= rank((W_O V)_S) - 1
```

Hence local dimensionality cannot exceed selected value-subspace rank.

### Lemma 2: Entropy-Conditioned Local Rank Envelope [EXPLORATORY]

If A4 holds (non-degenerate selected columns), then locally:

```
dim_local(y) ~ O(s - 1),  with  s >= k_eff
```

Operationally: larger `k_eff` increases the attainable local output dimension by activating
more independent value directions.

### Practical Proxy

Define a geometry operator per layer:

```
G = V^T W_O^T W_O V
```

Then entropy-weighted mixing energy is:

```
E_mix = E_x[ α(x)^T G α(x) ] - ||E_x[W_O V α(x)]||^2
```

`E_mix` is computable and isolates value-subspace mixing under `α`.

---

## Step 3: Curvature Operator Dependence on Covariance Spectrum

### Proposition 4: Small-Angle Expansion of Angular Curvature [PROVEN: under A3]

With `h = h_in`, `δ = h_out - h_in`, `θ = arccos(cos(h, h+δ))`, for `||δ||/||h||` small:

```
θ^2 = ||P_perp(h) δ||^2 / ||h||^2 + O((||δ||/||h||)^3)
```

So angular curvature is governed (to second order) by orthogonal energy of the update.

### Proposition 5: Expected Curvature as Projected Covariance Trace [EXPLORATORY]

Under local stationarity and A3:

```
E[θ^2] ~ E[ tr(P_perp(h) Σ_δ(x) P_perp(h)) / ||h||^2 ]
```

If attention term dominates local variation:

```
Σ_δ ~ Σ_y + Σ_mlp + cross terms
Σ_y = W_O V Σ_α V^T W_O^T
```

Then entropy affects curvature through `Σ_α -> Σ_y -> projected orthogonal energy`.

This is the explicit operator bridge missing from correlation-only claims.

---

## TwoNN Link: Covariance Spectrum -> Estimated ID

### Lemma 3: TwoNN Local Scaling Dependence [EXPLORATORY]

TwoNN estimates dimension from local distance-ratio scaling (`μ = r2/r1` and regression on
`log μ`). For locally regular manifolds (A5), estimated ID increases with the number of
non-negligible local covariance eigenvalues.

Equivalent operational statement: when `Σ_y` (or `Σ_δ`) spreads mass across more significant
eigen-directions, TwoNN ID should increase.

### Bridge Claim

```
higher H(α) -> larger attainable support -> higher-rank/more isotropic Σ_α
-> broader Σ_y spectrum -> larger projected orthogonal update energy
-> higher E[θ] and higher TwoNN ID (through local scaling)
```

This bridge remains `[EXPLORATORY]` until Lemma 3 is formalized with estimator-specific bounds.

---

## Derived Predictions (Pre-Registered Form)

These are not post-hoc interpretations; each is tied to the derivation structure above.

1. **P-EC1 (Direction):** Within layer families where A3/A4 hold, entropy-conditioned
   curvature slope is non-negative:
   `∂ E[θ^2 | layer] / ∂ H >= 0`.
2. **P-EC2 (Geometry dependence):** At fixed entropy, layers with larger effective value
   subspace rank (`rank((W_O V)_S)` proxy) show higher curvature.
3. **P-EC3 (Operator split, architecture-qualified):** Relative dominance between
   `corr(H, θ_attn)` and `corr(H, θ_mlp)` is family-dependent (e.g., hybrid LFM2 may show
   attention-dominant coupling while standard transformer families may show MLP-dominant or
   mixed coupling). No universal dominance direction is assumed.
4. **P-EC4 (ID response):** Layers with larger entropy-driven projected covariance trace
   have higher TwoNN ID after controlling for norm scale.

If these fail, the derivation fails.

---

## Falsifiers (Designed to Detect Coincidence)

`r=0.507` alone is insufficient. The following falsifiers distinguish mechanism from coincidence.

### F1: Sign Falsifier (Mechanism Direction)

**Test:** Fit within-model, per-layer regression of `θ^2` on `H` with architecture controls.  
**Derivation prediction:** non-negative coefficient for entropy term.  
**Failure criterion:** significant negative coefficient in >=2 architecture families.

Interpretation: if violated, entropy is not the causal driver under this operator.

### F2: Geometry-Conditioned Falsifier (Not Just Entropy)

**Test:** Hold entropy bins fixed; compare curvature across layers with different
`rank((W_O V)_S)` or `E_mix`.  
**Derivation prediction:** curvature differs with value geometry at fixed entropy.  
**Failure criterion:** no geometry effect after entropy matching.

Interpretation: if violated, `V/W_O` geometry term is unnecessary in the model.

### F3: Attention-vs-MLP Falsifier (Operator Decomposition)

**Test:** Compare `corr(H, θ_attn)` vs `corr(H, θ_mlp)` using decomposition script.  
**Derivation prediction:** LFM2-qualified only: attention-side correlation dominates in LFM2;
no universal dominance sign is assumed for non-LFM2 families.  
**Failure criterion:** any LFM2 run with `|corr(H, θ_attn)| <= |corr(H, θ_mlp)|`.

Interpretation: if violated, even the qualified attention-path claim fails; if non-LFM2
families diverge, treat as architecture-term requirement, not refutation.

### F4: Permutation Falsifier (Coincidence Check)

**Test:** Permute entropy assignments within layer-depth strata; recompute explained variance
for the full derivation model (`H`, geometry term, projected covariance proxy).  
**Derivation prediction:** real-data fit exceeds permutation null by large margin.  
**Failure criterion:** real fit within null envelope (e.g., not beyond 95th percentile).

Interpretation: if violated, observed `r=0.507` is compatible with accidental alignment.

### F5: Scale/Family Falsifier (Missing Terms)

**Test:** Evaluate coefficients jointly across LFM2/Qwen/Llama with architecture interactions.  
**Derivation prediction:** same sign, family-specific magnitude is allowed.  
**Failure criterion:** sign flips explained only by omitted architecture terms.

Interpretation: if violated, claim is `[MECHANISM_UNDERSPECIFIED]` not validated.

---

## Minimal Experimental Bundle (Directly Runnable)

1. Extend `scripts/curvature_accumulation_analysis.py` to log per-layer:
   - mean `H(α)`
   - `E_mix` proxy
   - `θ_attn`, `θ_mlp`, `θ_total`
   - TwoNN ID
2. Run stratified regressions:
   - `θ_attn^2 ~ H + E_mix + depth + family + interactions`
   - `ID ~ H + E_mix + depth + family + interactions`
3. Run permutation null for F4.
4. Report whether F1-F5 pass/fail.

Until F1-F5 are executed and passed, this bridge remains `[EXPLORATORY]`.

---

## Current Evidence Status

- Covariance pushforward and entropy-support/rank identities: `[PROVEN]`
- Curvature small-angle expansion: `[PROVEN]` under A3
- Entropy -> curvature operator bridge with model-family robustness: `[EXPLORATORY]`
- TwoNN response mapping from covariance spectrum: `[EXPLORATORY]`

This document is a derivation contract, not a promotion claim.

**Empirical status from companion document** (`entropy_curvature_derivation.md`):
- P1 (sign opposition): REFUTED as universal — LFM2-only (1/3 pass)
- P4 (MLP gain varies): VALIDATED 3/3 (CV 0.177–0.739)
- Attention weight entropy shows no significant correlation with curvature on standard
  transformers (Qwen2.5-3B: r=-0.036, p=0.835). The causal chain's r=0.507 uses logit
  entropy (Entropy-Lens), a different quantity.
- F3 qualified to LFM2-only based on these results.

---

## Entropy Operator Reconciliation (ACT-016, 2026-03-03)

**Status:** Measured. Conclusion: H_attn and H_logit are different quantities.

### Context

The derivation in this document is written for H_attn = Shannon entropy of QK softmax weights
α. The empirical r=0.507 reported in the causal chain uses H_logit = Shannon entropy of the
token distribution obtained by projecting h_l through the final norm and unembedding matrix
(Entropy-Lens). These are distinct operators. ACT-016 required measuring corr(H_attn, H_logit)
per family to determine whether the derivation path is valid in spirit.

### Operator Correlation Results

| Family | Model | corr(H_attn, H_logit) | Interpretation |
|--------|-------|----------------------|----------------|
| Qwen | Qwen2.5-3B | r = -0.094 | Different quantities |

**Threshold:** r > 0.7 → proxies (derivation valid in spirit). r < 0.3 → different quantities.

Result: r = -0.094 on Qwen2.5-3B is firmly in the "different quantities" regime.
H_attn and H_logit are not proxies for the same underlying posterior uncertainty on standard
transformers. The existing derivation (H(α) → Cov[y] → curvature) does not theoretically
explain the empirical r=0.507 chain, which operates on H_logit.

**Note:** Full multi-family results (LFM2, Llama) pending. Qwen2.5-3B is the only confirmed
datapoint. Additional families required to complete the per-family table.

### F1 Operator Split Results

| Family | F1_attn (H_attn → θ²) | F1_logit (H_logit → θ²) |
|--------|----------------------|------------------------|
| Qwen2.5-3B | r = -0.036, p = 0.835 (FAIL) | H_logit significant (PASS) |

H_attn fails F1 on standard transformers. H_logit passes F1 (consistent with r=0.507 baseline).

### Sign-Law Decomposition Results

From `f5_sign_law_decomposition` falsifier (2026-03-03):

**Equation:** `log(θ_total²) ≈ log||P_perp(h)δ||² - log||h||²`

| Family | beta_theta | p-value | Significant | sign_match |
|--------|-----------|---------|-------------|------------|
| Qwen2.5 | -0.2773 | 0.0030 | Yes | True |
| Llama | -0.0649 | 0.0188 | Yes | True |
| LFM2 | -0.0549 | 0.3971 | No | True |
| Qwen3 | +0.0113 | 0.8909 | No | False (non-significant) |

**F5 sign-law result:** PASSES — no sign mismatches in significant families.

**Observation:** The beta direction is negative for significant families — higher H_logit
predicts lower θ_total² (lower angular curvature). This is counter to the naive expectation
from the derivation direction (higher entropy → higher curvature). The sign-law test confirms
the decomposition structure is consistent (numerator-denominator competition governs sign),
but the mechanism for the negative direction remains open.

### Implications for the Derivation

1. **This derivation's scope:** The formal derivation (Steps 1–3 above) explains H_attn's
   effect on curvature through the population covariance path. This path is theoretically
   sound but empirically inoperative on standard transformers — H_attn does not correlate
   with curvature on Qwen/Llama.

2. **The empirical chain uses H_logit:** The r=0.507 link requires a separate derivation
   framed around H_logit. H_logit measures posterior uncertainty about the next token; its
   relationship to angular curvature must be derived from the unembedding geometry, not
   from the attention operator V/W_O path.

3. **Two separate derivation targets:**
   - **Target A (this document's scope):** H_attn → Cov[y] → curvature (theoretically clean;
     empirically not the operative path on standard transformers)
   - **Target B (open):** H_logit → curvature (empirically operative, r=0.507; derivation
     requires understanding why higher posterior uncertainty at position l implies lower
     angular change — the negative sign is the key puzzle)

4. **Reframing required for empirical chain closure:** The causal chain's H_logit → Δcurvature
   link cannot be derived from the attention mechanics path developed here. A separate derivation
   starting from the unembedding projection and its geometric relationship to layer output
   changes is needed.

---

## References

- Facco et al. (2017), TwoNN intrinsic dimension estimator.
- Agarwal, Dalal, Misra (2026), arXiv:2512.22471 / 2512.22473 / 2512.23752.
- `docs/research/entropy_curvature_derivation.md`: Empirical validation companion.
  Contains P1-P6 prediction outcomes on 3 models (LFM2-700M, Qwen3.5-0.8B, Qwen2.5-3B),
  the logit-vs-attention entropy distinction, architecture-dependent sublayer sign tables,
  MLP gain variation data, and contrast with Codex falsifier tests. This document provides
  the empirical findings; the present document provides the formal derivation framework.
- `docs/research/causal-chain-evidence-map.md`: Updated chain with H_attn/H_logit split;
  ACT-016 results recorded at the `H_attn ↔ H_logit [EXPLORATORY]` link.
- `docs/research/bayesian_geometry_connection.md`: Agarwal 2026 alignment with causal chain.
- Curvature operator implementation:
  `scripts/curvature_accumulation_analysis.py` (`angular_change = arccos(cosine_similarity)`).
  Contains: `try_compute_logit_entropy`, `mean_h_logit`, `operator_correlation`,
  `f1_sign_logit`, `f3_logit_vs_attn_operator`, `f4_permutation_logit`, `f5_family_logit`,
  `f5_sign_law_decomposition`.
- TwoNN implementation:
  `src/modelcypher/core/domain/geometry/intrinsic_dimension.py`.
- Empirical data: `results/entropy_curvature/entropy_curvature_results.json`.
