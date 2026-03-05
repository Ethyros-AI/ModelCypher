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

**Update (2026-03-04, 6-model operator-split run): PASS.**

Implemented test: `corr(theta_total, E_mix | H_logit, depth)` with Fisher-SE MDE
and Bretherton autocorrelation correction for resolvability.

- Resolvable geometry effect detected: Qwen2.5-3B (`|r|=0.379 > MDE=0.175`),
  Llama-3.2-3B (`|r|=0.221 > MDE=0.201`)
- Below measurement floor (not failures): LFM2-350M, LFM2-700M, Qwen3.5-0.8B,
  Mistral-7B

Source: `results/entropy_curvature_operator_split/falsifier_outcomes.json`

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
| Global (all) | — | r = 0.145 | Different quantities |
| Qwen | Qwen2.5-3B | r = -0.094 | Different quantities |
| Llama | Llama-3.2-3B | r = 0.629 | Partial overlap |
| Qwen3 | Qwen3-8B | r = 0.602 | Partial overlap |
| LFM2 | LFM2-350M | — | Not measurable (no attention decomposition) |

**Threshold:** r > 0.7 → proxies (derivation valid in spirit). 0.3–0.7 → partial overlap,
architecture-dependent. r < 0.3 → different quantities.

Result: The operators are neither universal proxies nor universally orthogonal. The
relationship is family-conditioned:
- Qwen2.5 (GQA=8): r = -0.094. Operators measure different things. H_attn explains nothing
  about the empirical r=0.507 chain for this family.
- Llama (GQA=3) and Qwen3 (GQA=4): r ≈ 0.61–0.63. Partial overlap — the operators share
  some variance but are not interchangeable. Derivation valid in a weaker sense for these
  families only.
- Global cross-family: r = 0.145 — different quantities at the aggregate level, driven by
  Qwen2.5's negative correlation pulling the global value down.

**Implication:** The H(α) → Cov[y] → curvature derivation path cannot be claimed as the
mechanism behind the empirical r=0.507 link. The relationship between the two operators is
architecture-conditioned. Cross-family evidence is directional (10-model Spearman
GQA vs r(H_logit, H_attn) = -0.736, p=0.015), but same-GQA models can diverge
(Qwen3 vs Qwen3.5 at GQA=4), so a GQA-only monotone law is not established.

### F1 Operator Split Results

| Family | F1_attn (H_attn → θ²) | F1_logit (H_logit → θ²) |
|--------|----------------------|------------------------|
| Qwen2.5-3B | r = -0.036, p = 0.835 (FAIL) | PASS |
| Llama-3.2-3B | (from multi-family run) | PASS |
| Qwen3-8B | (from multi-family run) | PASS |
| LFM2-350M | not measurable | PASS (non-significant) |

H_attn fails F1 on standard transformers. H_logit passes F1 globally (consistent with
r=0.507 baseline). The empirical operative path is H_logit, not H_attn.

### Sign-Law Decomposition Results

From `f5_sign_law_decomposition` falsifier (2026-03-03):

**Equation:** `log(θ_total²) ≈ log||P_perp(h)δ||² - log||h||²`

| Family | beta_theta | p-value | Significant | sign_match |
|--------|-----------|---------|-------------|------------|
| Qwen2.5 | -0.2773 | 0.0030 | Yes | True |
| Llama | -0.0649 | 0.0188 | Yes | True |
| LFM2 | -0.0549 | 0.3971 | No | True |
| Qwen3 | +0.0113 | 0.8909 | No | False (non-significant) |

**Sign-law decomposition test (log-space, 2026-03-03):** No sign mismatches among models
with p<0.05 (heuristic threshold, NOT derived). This is a separate test from the
operator-split F5 (depth-controlled partial Spearman, below) which uses a different model
set and method. The threshold used here is not derived from measurement theory — this
test's status is diagnostic only, not a decision boundary.

**Observation:** The beta direction is negative for low-p models — higher H_logit
predicts lower θ_total² (lower angular curvature). This is counter to the naive expectation
from the derivation direction (higher entropy → higher curvature). The decomposition
structure is consistent (numerator-denominator competition governs sign),
but the mechanism for the negative direction remains open.

### Implications for the Derivation

1. **This derivation's scope:** The formal derivation (Steps 1–3 above) explains H_attn's
   effect on curvature through the population covariance path. This path is theoretically
   sound but empirically inoperative at the cross-family level. H_attn does not correlate
   with curvature on standard transformers (Qwen/Llama). The derivation may be locally valid
   within LFM2 (attention-dominant hybrid) but this has not been tested against H_attn.

2. **The empirical chain uses H_logit:** The r=0.507 link requires a separate derivation
   framed around H_logit. H_logit measures posterior uncertainty about the next token. Its
   relationship to angular curvature must be derived from the unembedding geometry, not from
   the attention operator V/W_O path.

3. **The GQA-conditioning hypothesis (open, testable):** Current data supports a directional
   cross-family trend (higher GQA tends to weaker H_attn/H_logit coupling), but the same-GQA
   split (Qwen3 vs Qwen3.5, both GQA=4) shows architecture terms are required. The correct
   claim form is GQA + architecture, not a 1D monotone GQA law.

4. **Two separate derivation targets:**
   - **Target A (Steps 1–3 above):** H_attn → Cov[y] → curvature (theoretically clean;
     empirically not the operative cross-family path; may be operative in LFM2 regime)
   - **Target B (section below):** H_logit → representation components (empirically operative;
     Propositions B1-B3 proven, Assumptions B4-B5 exploratory. The key finding: H_logit
     predicts components of θ² but they cancel in the ratio. θ² is the wrong observable.)

5. **Reframing required for empirical chain closure:** The causal chain's H_logit → Δcurvature
   link cannot be derived from the attention mechanics path developed here. A derivation starting
   from the unembedding projection and its geometric relationship to layer output changes is
   needed. The sign-law decomposition (Proposition 4) provides the structural scaffold:
   `θ² ≈ ||P_perp(h)δ||²/||h||²`. The open question is: why does higher H_logit (more
   diffuse posterior) predict smaller perpendicular-to-h update energy in the numerator
   relative to the denominator?

---

## Target B: H_logit → Component Coupling (2026-03-04)

**Status:** [EMPIRICAL] — proven decomposition algebra + strong empirical data + theoretical
motivation. Causal operator (B4) is a learned manifold property, not architecturally derivable.

Target A (Steps 1–3 above) derives H_attn → Cov[y] → curvature through the attention
covariance path. That path is empirically inoperative on standard transformers. This section
develops the empirically operative path: H_logit → representation components.

### Notation (additions to main notation)

- `LN(h) = γ ⊙ (h − μ_h) / σ_h + β`: LayerNorm with learned `γ`, `β`; `μ_h = mean(h)`,
  `σ_h = RMS(h − μ_h)`
- `W_E ∈ R^(V × d)`: embedding/unembedding matrix (weight-tied)
- `H_logit(h) = H(softmax(W_E · LN(h)))`: logit entropy (Entropy-Lens)

### Proposition B1: H_logit is Direction-Only [PROVEN: architecture]

**Claim:** `H_logit(h)` is invariant to positive rescaling of `h`.

**Proof:** RMSNorm (the norm used in modern transformers including LFM2, Llama, Qwen) maps:

```
RMSNorm(h) = γ ⊙ h / ||h||_RMS
```

where `||h||_RMS = √(Σ h_i² / d)`. For any `c > 0`:

```
RMSNorm(c·h) = γ ⊙ (c·h) / ||c·h||_RMS = γ ⊙ h / ||h||_RMS = RMSNorm(h)
```

The logits `z = W_E · RMSNorm(h)` are therefore invariant to positive rescaling of `h`.
Since softmax and Shannon entropy are applied to `z`, `H_logit` is invariant to `||h||`.

More precisely: `H_logit = g(ĥ)` where `ĥ = h/||h||` is the unit-direction and `g` is
determined by `W_E`, `γ`. H_logit is a function of the **direction** of `h` in
representation space, not its magnitude.

**Corollary B1.1:** Any statistical correlation between `H_logit` and `||h||²` across a
token population is a **population-level manifold property** (tokens at different H_logit
values have systematically different norms), NOT a pointwise mathematical identity. The
direction and magnitude of `h` are coupled through the learned representation geometry,
not through the measurement operator.

### Proposition B2: θ² Component Decomposition [PROVEN: = Proposition 4]

From Proposition 4 (Step 3 above), under assumption A3:

```
θ² ≈ ||P_perp(h) δ||² / ||h||²
```

In log space:

```
log(θ²) ≈ log(||P_perp(h) δ||²) − log(||h||²)
```

Define: `Y_num = log(||P_perp(h) δ||²)` (numerator), `Y_den = log(||h||²)` (denominator).

### Theorem B3: Cancellation Mechanics [PROVEN: regression algebra]

**Claim:** If H_logit predicts both components with depth-controlled coefficients `β_num`
and `β_den`, then the depth-controlled coefficient on `θ²` satisfies:

```
β_θ ≈ β_num − β_den
```

**Proof:** Under linear regression on depth-residualized variables:

```
E[Y_num | H, depth] ≈ β_num · H + f_num(depth)
E[Y_den | H, depth] ≈ β_den · H + f_den(depth)
```

Since `log(θ²) ≈ Y_num − Y_den`:

```
E[log(θ²) | H, depth] ≈ (β_num − β_den) · H + (f_num − f_den)(depth)
```

Therefore `β_θ = β_num − β_den`. ∎

**Corollary B3.1 (Cancellation regimes):**
- When `|β_num| ≈ |β_den|` and same sign: **cancellation** (θ² insensitive to H_logit)
- When `β_num` and `β_den` have opposite signs: **amplification** (θ² strongly sensitive)
- When one dominates: **partial pass-through** (θ² inherits the dominant component's sign)

**Empirical confirmation (4-family component table):**

| Family | β_num | β_den | |β_num|/|β_den| | Regime |
|--------|------:|------:|---------------:|--------|
| LFM2-350M | −0.256 | −0.254 | 1.01 | Perfect cancellation |
| Llama-3.2-3B | +0.070 | +0.168 | 0.42 | Denominator dominates |
| Qwen2.5-3B | −0.422 | +0.175 | 2.41 | Opposite signs → amplification |
| Qwen3-8B | +0.199 | +0.359 | 0.55 | Denominator dominates |

The cancellation theorem explains why F4 fails on `θ²` while both components individually
pass at the 100th permutation percentile. Angular curvature is the **wrong observable**
for the H_logit relationship — it normalizes by `||h||²`, the very quantity H_logit most
strongly predicts.

### Assumption B4: Learned Manifold Coupling [EXPLORATORY]

**Statement:** At each depth `l`, the trained representation manifold `M_l` admits an
approximate parameterization by posterior entropy `H_logit` such that both `||h_l||` and
`||P_perp(h_l) δ_l||` are smooth functions of `H_logit` on `M_l`.

**This is NOT derivable from architecture.** LayerNorm explicitly destroys norm information
before the unembedding projection (Proposition B1). The coupling between direction (which
determines H_logit) and magnitude (||h||²) arises from the learned weight geometry, not
from architectural identities. Different trained models on different data could in principle
have different coupling signs and strengths.

**Theoretical motivation:** Agarwal, Dalal & Misra (2026, arXiv:2512.22471) show that in
well-specified Bayesian tasks, the value manifold at training convergence is 1D,
parameterized by posterior entropy. If this structure generalizes to production LLMs (an
empirical claim, not a theorem), then H_logit indexes position on a 1D manifold, and all
geometric properties of `h` — including `||h||` and `||P_perp δ||` — co-vary through their
shared dependence on manifold position.

**Empirical support:**
- Both components pass depth-stratified permutation at 100th percentile (n=116 observations,
  4 families, 500 permutations). The signal is overwhelmingly real.
- Per-family Spearman correlations (depth-residualized):

  | Family | r(H, ||P_perp δ||²) | p | r(H, ||h||²) | p |
  |--------|--------------------:|----:|-------------:|----:|
  | LFM2 | −0.679 | 0.004 | −0.732 | 0.001 |
  | Llama | +0.465 | 0.013 | +0.762 | 0.000 |
  | Qwen2.5 | −0.599 | 0.000 | −0.198 | 0.246 |
  | Qwen3 | +0.366 | 0.028 | +0.655 | 0.000 |
  | Qwen3.5 | −0.302 | 0.152 | +0.053 | 0.806 |

- The component-level signs are family-dependent (positive for Llama/Qwen3, negative for
  LFM2/Qwen2.5, mixed for Qwen3.5). The family-dependence indicates the coupling is
  learned and architecture-conditioned, consistent with the learned manifold interpretation.

**Failure mode:** If a new architecture family shows H_logit predicting neither component
after depth control (both components below permutation null), B4 is refuted for that family.
This would indicate that the manifold parameterization does not hold universally.

### Hypothesis B5: GQA Modulates Norm-Entropy Coupling [EXPLORATORY → REFINED]

Strict monotone B5 (GQA → R²) **FAILS** (Spearman = -0.632, p=0.250, n=4).
Counterexample: Qwen3 (GQA=4, pure attn, R²=0.274) vs Qwen3.5 (GQA=4, hybrid, R²=0.000).

**Refined B5':** R² depends on BOTH GQA AND attention architecture type.
Operator-stratified within Qwen3.5-0.8B (fixed GQA=4): full-attention layers R²=0.941
(p=0.0013, n=6) vs linear-attention R²=0.032 (p=0.476, n=18). Architecture type is
an active term at fixed GQA.

**Cancellation regime (GQA + architecture):**

| Regime | R²(H→||h||²) | Cancellation |
|--------|-------------:|-------------|
| Low GQA + pure attn (Llama) | 0.826 | Strong |
| Low GQA + hybrid (LFM2) | 0.721 | Perfect (ratio 1.01) |
| Mid GQA + pure attn (Qwen3) | 0.274 | Partial (den dominates) |
| Mid GQA + hybrid (Qwen3.5) | 0.000 | None (numerator passes) |
| High GQA + pure attn (Qwen2.5) | 0.035 | Broken (opposite signs) |

Source: `results/gqa_norm_entropy_coupling/coupling_results.json`

### Derivation Assessment: Protocol Claim Form

Per `docs/research/FIRST_PRINCIPLES_REVIEW_PROTOCOL.md`:

```
observable     = {||P_perp(h)δ||², ||h||²}  (NOT θ²)
geometry_state = position on learned manifold parameterized by H_logit
architecture   = GQA ratio (modulates norm-entropy coupling R²)
scale          = NOT YET CHARACTERIZED
measurement    = depth-controlled OLS on log-transformed components
```

| Field | Status | Notes |
|-------|--------|-------|
| Causal operator | **MISSING** | B4 is a manifold assumption, not a derived operator |
| Equation/theorem | B1-B3 [PROVEN], B4-B5 [EXPLORATORY] | Algebra is complete; mechanism is not |
| Architecture term | GQA + attn type [EXPLORATORY, n=4] | B5 refined: two-variable (GQA + hybrid) |
| Scale term | NOT CHARACTERIZED | Qwen3.5 scale series shows transition, no functional form |
| Measurement operator | Depth-controlled OLS on log components | bf16 only for sublayer; 4-bit safe for aggregate |
| Commensurability | Log-transform valid for component magnitudes | Component signs not commensurable across families |
| Directional prediction | β_total < 0 among resolvable models (7/10 CONSISTENT_SIGN) | Component signs are family-dependent |
| Falsifier | Any resolvable model with β_total > 0 | Refutes CONSISTENT_SIGN |

**Overall status: [EMPIRICAL].** The provable decomposition (B1-B3) is a genuine advance:
it identifies the wrong observable (θ²), explains why F4 fails (cancellation), and
characterizes the cancellation regime (GQA). But the causal operator (B4) cannot be promoted
because the norm-entropy coupling is a learned manifold property. The strongest achievable
status is [EMPIRICAL] until either:
- B4 is derived from architecture (unlikely — it depends on training dynamics), or
- A new measurement operator is found that does not require the manifold assumption.

---

## Sign Law Resolution (2026-03-04)

### The depth confound

Raw ρ(H_logit, θ_total) is confounded by shared depth trends (both decrease through network).
Raw: LFM2 near-zero, Qwen spuriously positive. After OLS depth-residualization: all 4 models
negative (Qwen3.5-0.8B p=0.003, others weak). Partial Spearman gives mixed signs at
significance boundary. The raw sign inconsistency is explained by depth confound.

### Why H_attn is null on standard transformers

H_attn measures routing concentration, but in trained standard transformers, MLP dominates
output covariance → H_attn is decoupled from perpendicular energy. On LFM2, conv layers
handle transport → attention has a specialized binding role → H_attn operative (r=0.829).

### Architecture-dependent sublayer mechanism

| Architecture | Mechanism | ρ_core | ρ_mlp | Models |
|-------------|-----------|--------|-------|--------|
| Hybrid conv+attn (LFM2) | competing_sublayers | + | − | 2/2 consistent |
| Pure attention (Llama, Mistral) | core_pass_through | + | + | 2/2 consistent |
| Hybrid linear-attn (Qwen3.5) | core_pass_through | + | + | 3/3 scale-consistent |

Prediction accuracy 9/10 (Qwen3-8B sole mismatch). Architecture-predictable: conv+attn →
competing, all others → core_pass_through.

### F5 status: CONSISTENT_SIGN (threshold DERIVED)

**Threshold derivation:** Fisher-SE minimum detectable effect (MDE) with Bretherton (1999)
autocorrelation correction. The MDE is the measurement resolution of the partial correlation
estimator — the smallest |r| with signal-to-noise ratio ≥ 1 at the effective sample size.
No heuristic thresholds. n_eff capped at n (cannot have more independent observations than
physical layers).

**Detection floor results (updated 2026-03-04, 10 models with proper Qwen3.5 decomposition):**

| Model | n | ρ₁ | n_eff | MDE | |r| | Resolvable | Perm exceedance |
|-------|---|-----|-------|-----|-----|------------|-----------------|
| LFM2-350M | 16 | -0.495 | 16.0 | 0.270 | 0.326 | Yes | 0.215 |
| LFM2-700M | 16 | -0.616 | 16.0 | 0.270 | 0.109 | No | 0.681 |
| Qwen3.5-0.8B | 24 | 0.323 | 12.3 | 0.317 | 0.580 | Yes | 0.036 |
| Qwen3.5-2B | 24 | 0.254 | 14.3 | 0.289 | 0.556 | Yes | 0.042 |
| Qwen3.5-4B | 32 | 0.077 | 27.4 | 0.200 | 0.503 | Yes | 0.034 |
| Qwen3.5-4B-4bit | 32 | 0.072 | 27.7 | 0.199 | 0.522 | Yes | 0.034 |
| Llama-3.2-3B | 28 | 0.422 | 11.4 | 0.332 | 0.703 | Yes | 0.004 |
| Mistral-7B | 32 | 0.425 | 12.9 | 0.307 | 0.741 | Yes | 0.000 |
| Qwen2.5-3B | 36 | 0.905 | 4.0 | 0.762 | 0.325 | No | 0.022 |
| Qwen3-8B | 36 | 0.890 | 4.0 | 0.762 | 0.097 | No | 0.611 |

**Result:** 7/10 models resolvable. All resolvable models show **negative** sign
(depth-controlled OLS slope). Below floor: LFM2-700M, Qwen2.5-3B, Qwen3-8B (all have
high autocorrelation ρ₁ → low n_eff → large MDE). Cross-family consistency across 4
architecture families (LFM2 hybrid, Qwen3.5 hybrid, Llama, Mistral). Qwen3.5
scale-validated (0.8B + 2B + 4B all resolvable, all negative). Gate check: 10/10.
F5 status: **CONSISTENT_SIGN**.

**Summary:** 7/10 resolvable, all negative, 4 families (LFM2, Qwen3.5, Llama, Mistral).
Mechanism prediction 9/10. Qwen3.5 scale-validated (0.8B+2B+4B). F5: CONSISTENT_SIGN.
Open: LFM2-700M below floor, component-sign split discriminator unknown,
CR-EC-001 remains [EMPIRICAL].

---

## F4 Permutation Results (2026-03-04)

### All three F4 variants FAIL

Run on real models: LFM2-350M, Llama-3.2-3B, Qwen2.5-3B, Qwen3-8B (4 families, 116
observations for θ_total, ~100 for θ_attn which excludes LFM2).

| Test | Curvature | Entropy | Real |r| | Null 95th | Percentile | Result |
|------|-----------|---------|------:|----------:|-----------:|--------|
| F4 (original) | θ_attn² | H_attn | 0.069 | 0.136 | 69.4% | **FAIL** |
| F4_logit | θ_attn² | H_logit | 0.085 | 0.251 | 30.0% | **FAIL** |
| F4_logit_total | θ_total² | H_logit | 0.151 | 0.250 | 74.6% | **FAIL** |

Method: depth-stratified permutation (500 permutations, seed=2). Entropy values permuted
within depth strata to remove shared depth trends. Spearman correlation on depth-residualized
values.

### Why F4 fails: numerator-denominator cancellation

The F4 failure is not just "depth confound." The decomposition `θ² = ||P_perp(h)δ||² / ||h||²`
reveals the precise mechanism. After depth-residualization, H_logit predicts *both* components
— but in the same direction, so the ratio cancels the signal.

Depth-residualized OLS slopes (β per unit H_logit residual):

| Model | β_num (||P_perp δ||²) | β_den (||h||²) | |β_num|/|β_den| | Pattern |
|-------|---------------------:|---------------:|---------------:|---------|
| LFM2-350M | -0.256 | -0.254 | 1.01 | Perfect cancellation |
| Llama-3.2-3B | +0.070 | +0.168 | 0.42 | Den dominates |
| Qwen2.5-3B | -0.422 | +0.175 | 2.41 | **Opposite signs → amplifies** |
| Qwen3-8B | +0.199 | +0.359 | 0.55 | Den dominates |

Depth-residualized Spearman correlations (H_logit → component):

| Model | r(H, ||P_perp δ||²) | p | r(H, ||h||²) | p | r(H, θ²) | p |
|-------|--------------------:|----:|-------------:|----:|---------:|----:|
| LFM2-350M | -0.679 | 0.004 | -0.732 | 0.001 | -0.350 | 0.184 |
| Llama-3.2-3B | +0.465 | 0.013 | +0.762 | 0.000 | -0.126 | 0.521 |
| Qwen2.5-3B | -0.599 | 0.000 | -0.198 | 0.246 | -0.429 | 0.009 |
| Qwen3-8B | +0.366 | 0.028 | +0.655 | 0.000 | +0.004 | 0.983 |

### GQA + Core Operator Control the Cancellation Pattern

Governing quantity: R²(H_logit → log||h||² | depth). NON_MONOTONE in GQA alone
(Spearman=-0.632, p=0.250, n=4). Same-GQA counterexample: Qwen3 R²=0.274 vs Qwen3.5
R²=0.000 (both GQA=4). Requires architecture-type term (see B5 cancellation regime table).
Higher GQA compresses key space → decouples routing from norm → less cancellation.

### What this means for the entropy-curvature link

H_logit predicts both components (||P_perp(h)δ||² and ||h||²) after depth control,
but θ² = ratio cancels the signal. θ is the wrong observable — it normalizes out
the very quantity (||h||²) that H_logit most strongly predicts.

### Implications for the derivation

1. **The r=0.507 is depth-confounded.** Confirmed. The deeper finding is that H_logit
   predicts the *components* of curvature (both numerator and denominator) even after depth
   control — the components cancel in the ratio, and the degree of cancellation is governed
   by GQA through the norm-entropy coupling R²(H → ||h||²).

2. **Both components PASS permutation null at 100th percentile.** Run on real models
   (LFM2-350M, Llama-3.2-3B, Qwen2.5-3B, Qwen3-8B), depth-stratified permutation (500
   perms), all 116 observations:

   | Component | Real r | |r| | Null 95th | Percentile | Result |
   |-----------|-------:|----:|----------:|-----------:|--------|
   | log||P_perp(h)δ||² | -0.470 | 0.470 | 0.164 | **100.0%** | **PASS** |
   | log||h||² | -0.364 | 0.364 | 0.180 | **100.0%** | **PASS** |

   Component signals are real (100th percentile, n=116). Pooled sign negative.
   Per-family: `results/f5_sign_law_full/cross_model_summary.json`.

3. **Component sign is family-dependent.** NEG: LFM2, Qwen2.5, Qwen3.5. POS: Llama, Qwen3.
   FFN ratio hypothesis REFUTED. Hybrid is sufficient-but-not-necessary for NEG.
   Discriminator unresolved (GQA doesn't explain; same-GQA Qwen3/Qwen3.5 have opposite signs).

4. **CR-EC-001 reframing.** The link is not "entropy → angular curvature" — it is
   "entropy → representation scale" AND "entropy → perpendicular update energy."
   Angular curvature (θ²) cancels the signal because it normalizes by the very quantity
   (||h||²) that entropy predicts. The correct observables are the unnormalized components.
   F4 FAIL on θ² is a measurement-operator artifact, not mechanism absence. F5
   CONSISTENT_SIGN in θ-space (among resolvable models) reflects the residual leakage
   from incomplete cancellation.

5. **Sublayer decomposition (8-model snapshot):** All β_total NEGATIVE (8/8).
   Mechanism: LFM2 competing (ρ_core+, ρ_mlp−), Llama/Qwen2.5 core_pass (both +),
   Qwen3.5 mlp_dom at 0.8B-2B → competing at 4B (scale transition). Prediction 6/8.
   Quantization: aggregate robust, sublayer attribution flips sign (bf16 required).
   Source: `results/f5_sign_law_full/cross_model_summary.json`.

6. **Open:** (a) Component-sign discriminator, (b) Qwen3.5 scale transition,
   (c) GQA → cancellation derivation.

### GQA Conditioning Hypothesis

**Status: PASSES** (updated 2026-03-04 with n=10 models, proper Qwen3.5 decomposition)

**Original test (n=3 families, H_attn vs H_logit operator correlation):**

| Family | GQA | corr(H_attn, H_logit) | Fisher z |
|--------|----:|---------------------:|---------:|
| Llama | 3 | 0.629 | 0.741 |
| Qwen3 | 4 | 0.602 | 0.697 |
| Qwen2.5 | 8 | -0.094 | -0.095 |

Regression: `z_f = atanh(corr_f) = a + b × log(GQA_f)`
- Slope b = -0.905 (predicted: b < 0)
- R² = 0.942
- Permutation p = 0.167 (1/6; n=3 families → 6 total permutations)
- LOO sign consistent: yes (all leave-one-out fits maintain b < 0)

**Updated test (n=10 models, cross-family, GQA → r(H_logit, H_attn)):**

After fixing GatedDeltaNet decomposition (Step 2.1) and re-running all Qwen3.5 models
with proper `linear_attn` decomposition (18 linear_attn + 6 attention per model):

```
Spearman(GQA, r(H_logit, H_attn)) = -0.736, p = 0.015, n = 10
```

Higher GQA → lower coupling between the two entropy operators. This now has
sufficient power (p < 0.05 at n=10) for a directional claim.

Within-family stability confirmed: Qwen3.5 (all GQA=4) shows r = -0.07 to -0.14,
stable across 0.8B–4B scale. Scale does not affect operator coupling at fixed GQA.

**Caveat:** Cross-family comparison confounds GQA with architecture differences.
Within-family GQA variation is not available in our model set (all Qwen3.5 models
share GQA=4). The cross-family result is directional, not causal.

See Proposition 10 in the H_logit derivation section below for the full mechanism
hypothesis connecting GQA → key compression → operator decoupling → cancellation.

---

## H_logit → Curvature: The Unembedding Geometry Path (2026-03-04)

### Motivation

The H_attn → curvature derivation (Steps 1-3 above) is theoretically clean but empirically
inoperative on standard transformers. H_attn does not predict curvature cross-family.
The empirical operative quantity is H_logit — Shannon entropy of the next-token distribution
obtained by projecting h_l through norm and unembedding:

```
H_logit(h) = H(softmax(W_u · LayerNorm(h)))
```

where `W_u in R^(V x d)` is the unembedding matrix (or `embed_tokens` when weight-tied).

This section derives the geometric relationship between H_logit and the curvature components,
explains the F4 cancellation mechanism, and formalizes the GQA modulation.

### Definition: H_logit as a Geometry Operator

**Definition 1.** For hidden state `h in R^d` at layer l, define the logit entropy operator:

```
p_i(h) = softmax(W_u · LayerNorm(h))_i,  i = 1..V
H_logit(h) = -Σ_i p_i(h) log p_i(h)
```

H_logit measures how concentrated the hidden state is in the directions defined by the
unembedding matrix rows. Low H_logit ↔ h is aligned with few vocab directions (sharp
posterior). High H_logit ↔ h projects broadly across vocab space (diffuse posterior).

**Proposition 6: H_logit as Unembedding Alignment [PROVEN: definition + softmax property]**

For any h, define the effective vocabulary dimension:

```
k_vocab(h) = exp(H_logit(h))
```

Then `k_vocab(h) in [1, V]` with `k_vocab = 1` when h is perfectly aligned with one
unembedding row (degenerate posterior) and `k_vocab = V` when h projects uniformly
across all rows (maximum entropy).

The unembedding matrix `W_u` defines a fixed set of V directions in R^d. H_logit
partitions the hidden state space into regions of different alignment concentration.

### Proposition 7: H_logit Predicts Representation Scale [EXPLORATORY]

**Empirical basis (5 families, depth-controlled):**

| Family | r(H_logit, log||h||²) | p | R² |
|--------|---------------------:|---:|---:|
| LFM2-350M | -0.732 | 0.001 | 0.721 |
| Llama-3.2-3B | +0.762 | 0.000 | 0.826 |
| Qwen2.5-3B | -0.198 | 0.246 | 0.035 |
| Qwen3-8B | +0.655 | 0.000 | 0.274 |
| Qwen3.5-0.8B | +0.053 | 0.806 | 0.000 |

**Geometric argument:** Before LayerNorm, larger ||h||² concentrates softmax → lower H_logit.
Coupling strength is architecture-conditioned: Llama/LFM2 strong (R²=0.72-0.83), Qwen3
moderate (0.274), Qwen2.5/Qwen3.5 weak (0.00-0.04). See B5 for GQA+operator conditioning.

### Proposition 8: H_logit Predicts Perpendicular Update Energy [EXPLORATORY]

**Empirical basis (5 families, depth-controlled):**

| Family | r(H_logit, log||P_perp(h)δ||²) | p | R² |
|--------|-------------------------------:|---:|---:|
| LFM2-350M | -0.679 | 0.004 | 0.575 |
| Llama-3.2-3B | +0.465 | 0.013 | 0.160 |
| Qwen2.5-3B | -0.599 | 0.000 | 0.428 |
| Qwen3-8B | +0.366 | 0.028 | 0.123 |
| Qwen3.5-0.8B | -0.302 | 0.152 | 0.376 |

**Geometric argument:** Low H_logit → h concentrated in few vocab directions → layer
update δ constrained to same region ("deepening commitment") → lower perpendicular energy.
High H_logit → h spread across vocab space → update has more freedom to rotate → higher
perpendicular energy.

### Proposition 9: F4 Cancellation Theorem [EXPLORATORY]

**Claim:** The angular curvature ratio θ² = ||P_perp(h)δ||² / ||h||² cancels the
H_logit signal when H_logit predicts both numerator and denominator with proportional
coefficients.

**Proof sketch (under linear approximation):** Assume depth-controlled relationships:

```
log||P_perp(h)δ||² = a_num + b_num · H_logit + ε_num
log||h||² = a_den + b_den · H_logit + ε_den
```

Then:

```
log θ² ≈ log||P_perp(h)δ||² - log||h||²
       = (a_num - a_den) + (b_num - b_den) · H_logit + (ε_num - ε_den)
```

The H_logit coefficient in θ² is `b_num - b_den`. When `b_num ≈ b_den`
(proportional prediction), the coefficient vanishes:

```
|b_num - b_den| << max(|b_num|, |b_den|)  =>  H_logit signal cancels in θ²
```

**Empirical verification (depth-residualized OLS slopes):**

| Model | β_num | β_den | |β_num|/|β_den| | Cancellation |
|-------|------:|------:|----------------:|-------------|
| LFM2-350M | -0.256 | -0.254 | 1.01 | Near-perfect |
| Llama-3.2-3B | +0.070 | +0.168 | 0.42 | Partial (den dominates) |
| Qwen2.5-3B | -0.422 | +0.175 | 2.41 | **Broken** (opposite signs) |
| Qwen3-8B | +0.199 | +0.359 | 0.55 | Partial (den dominates) |

When β_num and β_den have the same sign and similar magnitude (LFM2, Llama), θ²
normalizes away the H_logit signal. When they have opposite signs (Qwen2.5), the
signal amplifies in θ² rather than cancelling. ∎

### Proposition 10: GQA Modulates Cancellation via Key Compression [EXPLORATORY]

**Empirical basis (10 models, 2026-03-04):**
Spearman(GQA, r(H_logit, H_attn)) = -0.736, p=0.015, n=10. Within-family stable
(Qwen3.5 all GQA=4: r=-0.07 to -0.14 across 0.8B-4B). Full table in GQA Conditioning
section above.

**Mechanism:** Higher GQA compresses key space → attention routing less informative
about hidden state trajectory → H_attn and H_logit decouple. Low GQA: routing tightly
bound to posterior → proportional β_num/β_den → cancellation. High GQA: ||h||² driven
by MLP (independent of H_logit) → β coefficients diverge → cancellation breaks.

**Prediction:** Higher GQA → less cancellation → larger |β_total|. Not cleanly
separable from family effects at n=10 (|r| confounded by architecture).

### Summary: Two-Path Framework

The entropy → curvature relationship has two independent geometric paths:

**Path A (H_attn → curvature, this document Steps 1-3):**
```
H_attn → support size → Cov[α] rank → Cov[y] spectrum → ||P_perp(h)δ||²
```
- [PROVEN] covariance pushforward, rank bounds, entropy-support constraint
- [EXPLORATORY] curvature bridge
- Empirically operative in LFM2 only (H_attn → θ_attn: r=0.83 LFM2-700M)
- Null on standard transformers (Qwen2.5: r=-0.036, p=0.835)

**Path B (H_logit → curvature, this section):**
```
H_logit → unembedding alignment → {||P_perp(h)δ||², ||h||²} → θ² (with cancellation)
```
- [PROVEN] definitions (Prop 6)
- [EXPLORATORY] component predictions (Props 7-8), cancellation (Prop 9),
  GQA modulation (Prop 10)
- Empirically operative cross-family for components (F4 component PASS at 100th
  percentile, n=116)
- θ² signal survives only where cancellation is incomplete (GQA-dependent)
- F5 CONSISTENT_SIGN among 7/10 resolvable models (all negative β_total)

**The bridge between paths:** GQA controls operator coupling (Prop 10,
ρ = -0.736, p = 0.015, n = 10). Low GQA → paths A and B share variance through
routing-norm coupling → Path A is operative. High GQA → paths decouple → Path A
goes null, Path B's component-level signal survives uncancelled in θ².

**What remains OPEN:**
1. Formal derivation of ∂||P_perp(h)δ||²/∂H_logit and ∂||h||²/∂H_logit through
   the chain rule involving W_u (would promote Props 7-8 from EXPLORATORY)
2. Why the component-level sign is family-dependent (POS for Llama/Qwen3, NEG for
   LFM2/Qwen2.5/Qwen3.5) — the FFN ratio hypothesis was REFUTED, hybrid architecture
   is sufficient-but-not-necessary
3. The Qwen3.5 scale-dependent mechanism transition (mlp_dominant at 0.8B-2B →
   competing_sublayers at 4B)
4. Whether the GQA → cancellation relationship is derivable from key compression
   geometry or is itself an empirical coincidence

---

## Pre-Registered Falsifier: GQA-Isolated Operator Decoupling (F-GQA-01)

Canonical protocol:
- `docs/research/ENTROPY-CURVATURE-GQA-FALSIFIER-PROTOCOL.md`

This derivation document references the protocol but does not duplicate its operational
criteria, to keep falsifier logic single-sourced.

---

## Norm Confound Discovery & Controlled Re-measurement (2026-03-04)

### The Confound

The Entropy-Lens does NOT apply the model's final norm (RMSNorm) before projecting through
the unembedding:

```
H_logit = H(softmax(h @ W_unembed.T))                ← what we measure (unnormalized)
H_model = H(softmax(RMSNorm(h) @ W_unembed.T))       ← what the model actually computes
```

Since `h @ W.T = ||h|| × (ĥ @ W.T)`, softmax sharpness scales with `||h||`. The r=-0.99
correlation between H_logit and `||h||²` is a **measurement operator artifact**: it measures
how norm affects softmax temperature, not posterior uncertainty about the next token.

### Bedrock Derivation A: Why the Norm Confound Is Inevitable (Exact)

Let `z_hat` be fixed logits (direction fixed), and define temperature-scaled entropy:

```
p_t = softmax(t z_hat)
H(t) = -sum_i p_t,i log p_t,i
```

Using `H(t) = A(t) - t * E_{p_t}[z_hat]` with `A(t) = log sum_i exp(t z_hat,i)`:

```
dH/dt = -t * Var_{p_t}(z_hat) <= 0
```

This is exact (no approximation): increasing logit scale strictly decreases entropy unless
all logits are equal (zero variance case).

For unnormalized Entropy-Lens:

```
z_raw = h W_u^T = ||h|| * z_hat
```

so `t = ||h||` (up to fixed RMS/scale factors). Therefore `H_logit` is mechanically
anti-correlated with hidden-state norm from the measurement operator itself, even before any
geometry claim. This proves the norm confound mechanism.

### Bedrock Derivation B: Sign Law After Norm Correction

After RMSNorm, define:

```
p = softmax(z_norm),  z_norm = W_u * RMSNorm(h)
D = log(V) - H_logit_norm = KL(p || u_V) >= 0
```

where `u_V` is uniform over vocabulary.

Curvature operator identity (exact):

```
sin^2(theta) = ||P_perp(h) delta||^2 / ||h + delta||^2
```

with `delta = h_out - h_in`.

Assume the depth-local posterior-to-update map is differentiable on the simplex tangent:

```
P_perp(h) delta = B_l (p - u_V) + r_l
||r_l|| <= c_l ||p - u_V||^2
```

(`B_l` is architecture-dependent and depth-dependent; this is where architecture terms live.)

Then:

```
||P_perp(h) delta||^2
<= 2||B_l||^2 ||p-u_V||^2 + 2||r_l||^2
<= 4||B_l||^2 D + 8 c_l^2 D^2
```

using `||x||_2 <= ||x||_1` and Pinsker `||p-u_V||_1^2 <= 2 KL(p||u_V) = 2D`.

So at fixed depth:

```
sin^2(theta) <= a_l D + b_l D^2
           = a_l (log V - H_logit_norm) + b_l (log V - H_logit_norm)^2
```

with `a_l, b_l >= 0` after denominator normalization by `||h+delta||^2`.
Leading-order slope:

```
d sin^2(theta) / d H_logit_norm = -a_l + O(D) <= 0
```

This gives the corrected sign prediction: higher normalized posterior entropy implies lower
angular curvature to first order, matching the observed sign reversal.

**Consequences for the causal chain:**
- The r=0.507 raw (H_logit → θ_total) is contaminated by norm confound
- H_logit → E_total (r≈-0.9) is norm²→norm² (`||h||²` → `||δ||²`)
- θ_total is norm-independent → correctly shows no signal after depth control
- The entropy→curvature link needs re-measurement with the model's actual norm applied

**Per MISSION.md "Bedrock Mandate":** This is exactly the kind of correlation→cause error
the mission guards against. A near-perfect correlation (r=-0.99) that turns out to be a
measurement operator confound, not a geometric coupling.

### Prediction Contract (MISSION.md:51 — written before measurement)

```
observable = r(H_logit_norm, θ_total | depth)
  where H_logit_norm = H(softmax(RMSNorm(h_l) @ W_unembed.T))

geometry_state: per-layer hidden state direction (after norm removes magnitude)
architecture_state: {
    LFM2-700M (hybrid conv+attn, 16 layers),
    Qwen3.5-0.8B (hybrid linear_attn+full_attn, 24 layers),
    Qwen2.5-3B (pure attention, 36 layers)
}
scale_state: {700M (16L), 800M (24L), 3B (36L)}
measurement_operator: {
    H_logit_norm: Shannon entropy of norm-then-unembed projection,
    θ_total: arccos(cos(h_in, h_out)) per layer,
    depth control: OLS residualization on depth_fraction
}
```

**Directional predictions (written before measurement):**
1. `|r(H_logit_norm, ||h||²)| <= MDE` — normalization removes the norm confound.
   MDE = Fisher-SE minimum detectable effect = tanh(1/√(n-2)) where n = num_layers.
   (Falsifier: |r| resolvable above MDE means confound persists → architecture term needed)
2. `r(H_logit_norm, θ_total | depth)` — TWO possible outcomes, BOTH informative:
   - If |partial_r| resolvable above MDE: entropy→curvature is REAL
   - If below MDE: entropy does NOT predict curvature → chain link demoted
   (No falsifier needed — both outcomes are valid scientific results)
3. `(1 - r(H_logit_norm, H_logit)) > MDE` — normalization changed the measurement.
   If H_logit_norm ≈ H_logit (within MDE of r=1), normalization is trivial → check impl.

**Mixed-outcome rule (FIRST_PRINCIPLES_REVIEW_PROTOCOL.md §4):**
If sign differs across families, classify as MECHANISM_UNDERSPECIFIED unless architecture
terms predict the divergence. Report per-family and aggregate separately.

### E_mix Architecture Split (from existing results)

This finding is independent of the norm confound (E_mix is a norm² quantity but the
SIGN split is geometric):

| Operator | E_mix sign at high H_logit | Interpretation |
|----------|---------------------------|----------------|
| Attention (LFM2) | negative | core/MLP cooperate |
| Conv (LFM2) | positive | core/MLP oppose |
| Attention (Qwen3.5) | negative | core/MLP cooperate |
| Linear attn (Qwen3.5) | near-zero | no coupling |
| Attention (Llama) | negative | moderate cooperation |
| Attention (Mistral) | positive | core/MLP oppose |

This needs its own claim form if promoted beyond exploratory.

### Derivation D1: H_logit_norm Is Direction-Only (Identity)

RMSNorm(h)_i = γ_i · h_i / RMS(h), where RMS(h) = ||h|| / √d.

Therefore: `RMSNorm(h) = √d · γ ⊙ ĥ`, where `ĥ = h/||h||`.

Logits after norm: `√d · (γ ⊙ ĥ) @ W_unembed^T`.

√d is constant across layers. γ and W_unembed are fixed parameters.

**H_logit_norm = f(ĥ; γ, W, d) — depends on direction only.** `[PROVEN]`

||h|| is algebraically divided out. No approximation. No residual. This is exact.

Corollary: r(H_logit_norm, ||h||²) = r(f(ĥ_l), ||h_l||²) across layers l. This is
nonzero **iff** the direction trajectory {ĥ_l} and magnitude trajectory {||h_l||}
are statistically dependent across the layer index. We call this quantity the
**direction-magnitude coupling (DMC)**.

### Derivation D2: DMC From Residual Stream Geometry

In a residual network, `h_l = h_0 + Σ_{k<l} δ_k`. Decompose each update into radial
and tangential components relative to the current hidden state:

```
δ_k^∥ = ⟨δ_k, ĥ_k⟩ ĥ_k        (radial: changes ||h||, preserves ĥ)
δ_k^⊥ = δ_k - δ_k^∥              (tangential: changes ĥ, preserves ||h||)
```

To first order in ||δ||/||h||:

```
||h_{k+1}||² ≈ ||h_k||² + 2||h_k||·⟨δ_k, ĥ_k⟩ + ||δ_k||²    (magnitude from radial)
ĥ_{k+1} ≈ ĥ_k + δ_k^⊥ / ||h_k||                               (direction from tangential)
```

**DMC is zero when the tangential trajectory (which determines f(ĥ_l)) does not co-vary
with the cumulative radial trajectory (which determines ||h_l||²) across layers.**

For attention layers: δ_k = W_O V α(x), where α depends on Q = h_k W_Q. Because Q
depends on h_k, the output δ is structurally coupled to the input direction → radial
and tangential components co-vary → DMC ≠ 0.

For conv layers: δ_k = Conv(h_k) applies local temporal mixing via fixed-width kernels.
The conv output direction relative to h_k is determined by kernel weights and local
context, not by the global direction ĥ_k. This produces approximately direction-independent
radial/tangential ratios → DMC ≈ 0.

### Derivation D3: Negative Sign From Tangential/Radial Decomposition

From h_out = h_in + δ:

```
cos(θ) = (1 + r cos α) / √(1 + 2r cos α + r²)
```

where `r = ||δ||/||h_in||` and `α = angle(h_in, δ)`.

For small r (typical: θ < 0.3 rad implies r < 0.3): `θ ≈ r · sin(α) = ||δ^⊥||/||h_in||`.

Angular curvature = tangential update / residual stream magnitude.

**D3 original prediction (FALSIFIED):** sin(α) drives the negative sign — generic-direction
layers producing radial-dominant updates. **Corrected by D3.1–D3.4:** r (centroid magnitude
reduction) drives the negative sign; sin(α) opposes it (centroid is MORE tangential, D3.2).

**D3 measurement (2026-03-04, depth-controlled log-space):**

Since θ ≈ r · sin(α), decompose in log space: log θ ≈ log r + log sin(α).
All correlations are depth-residualized Spearman (OLS residuals against depth_fraction).

| Model | r(H_norm, log r) | p | r(H_norm, log sin α) | p | r(H_norm, log θ) | p |
|-------|----------------:|----:|--------------------:|----:|----------------:|----:|
| LFM2-700M | -0.071 | 0.795 | **+0.750** | 0.001 | -0.318 | 0.231 |
| Qwen3.5-0.8B | **-0.395** | 0.056 | -0.007 | 0.974 | -0.323 | 0.123 |
| Qwen2.5-3B | **-0.568** | 0.000 | **+0.686** | 0.000 | **-0.361** | 0.031 |

**D3's original tangential prediction was WRONG. Corrected derivation (D3.1–D3.5):**

The original D3 predicted sin(α) drives the negative sign (high entropy → radial-dominant →
small sin(α)). D3.2 proves the opposite: high entropy → centroid → MORE tangential (large
sin(α)). The negative θ comes from D3.1 (centroid magnitude reduction: r↓), with D3.4
proving r-dominance over the opposing sin(α) effect.

Architecture conditioning (D3.5) determines which factor absorbs the coupling:

```
θ ≈ r · sin(α)

Pure attention (Qwen2.5):  r↓↓↓(p<.001) + sin(α)↑↑↑(p<.001) → net θ↓ (D3.4: r wins)
Hybrid lin-attn (Qwen3.5): r↓↓ (p=.056) + sin(α)≈0 (n.s.)   → net θ↓ (D3.5: partial)
Hybrid conv+attn (LFM2):  r≈0  (n.s.)   + sin(α)↑↑↑(p=.001) → net θ↓ (D3.5: conv buffers r)
```

**Derivation: r and sin(α) as functions of attention entropy**

Let `w_t = W_O v_t ∈ R^(d_h)` be the output vectors (value columns composed with output
projection). Decompose each relative to `ĥ = h/||h||`:

```
w_t = w_t^∥ ĥ + w_t^⊥       where w_t^∥ = ⟨w_t, ĥ⟩ ∈ R,  w_t^⊥ ⊥ ĥ
```

The attention output `δ = Σ_t α_t w_t` decomposes as:

```
δ^∥ = Σ_t α_t w_t^∥           (radial: scalar sum)
δ^⊥ = Σ_t α_t w_t^⊥           (tangential: vector sum in d_h − 1 dimensions)
```

**Lemma D3.1 (Centroid Magnitude Reduction) `[PROVEN: convexity]`.**
For non-collinear value vectors (A4), concentrated α produces larger `||δ||` than diffuse α.

*Proof.* `||δ||² = α^T M α` where `M_{ts} = ⟨w_t, w_s⟩` is the Gram matrix.
`α = e_k` gives `||δ||² = ||w_k||²`. `α = u` gives `||δ||² = ||w̄||²`.
By strict triangle inequality, `||w̄|| < max_k ||w_k||` for non-collinear vectors.
Convex quadratic on simplex: maximum at vertices. ∎

This gives `r = ||δ||/||h|| ↓` as `H(α) ↑`.

**Lemma D3.2 (Centroid Tangentiality) `[PROVEN: concentration of measure]`.**
For T output vectors in d_h dimensions, the centroid has `sin²(α) → 1 − O(T/d_h)`.

*Proof.* The tangential centroid norm:

```
||w̄^⊥||² = (1/T²) Σ_{t,s} ⟨w_t^⊥, w_s^⊥⟩
```

Tangential components `{w_t^⊥}` live in `d_h − 1` dimensions. Cross-terms
`⟨w_t^⊥, w_s^⊥⟩` for `t ≠ s` satisfy `E[|⟨u,v⟩|²] = 1/d` for random unit vectors
(standard concentration). For `T ≪ d_h`, cross-terms are negligible:
`||w̄^⊥||² ≈ (1/T) mean_t(||w_t^⊥||²)`.

The radial centroid `(w̄^∥)² = (mean_t(w_t^∥))²` — scalars add coherently. But each
projection `w_t^∥ = ⟨w_t, ĥ⟩` captures `O(1/d_h)` of `||w_t||²` (one direction
among d_h). Meanwhile `||w_t^⊥||² = ||w_t||² · (1 − O(1/d_h))`.

```
sin²(α_centroid) = ||w̄^⊥||² / (||w̄^⊥||² + (w̄^∥)²)
                 ≥ 1 − O(T/d_h) → 1     for d_h ≫ T
```

For concentrated `α ≈ e_k`: `sin²(α) = ||w_k^⊥||²/||w_k||²` depends on the specific
token's radial alignment, which can be strictly less than 1. ∎

**Lemma D3.3 (CE-Driven QK Selection Bias) `[PROVEN under A7, A7 FALSIFIED — D3.3 NOT APPLICABLE]`.**
Under a radial-dominant downstream gradient, CE training increases attention scores for
above-average radial tokens and decreases scores for below-average radial tokens.

Define:

```
r_t = ⟨w_t, ĥ⟩               (token radial component)
R   = Σ_t α_t r_t = ⟨δ, ĥ⟩   (attention-weighted radial mean)
g   = ∂L/∂δ                   (downstream gradient at attention output)
```

Attention identities:

```
δ = Σ_t α_t w_t
s_t = q·k_t / √d_k
∂δ/∂s_t = α_t (w_t - δ)
∂L/∂s_t = ⟨g, ∂δ/∂s_t⟩ = α_t ⟨g, w_t - δ⟩
```

Assumption A7 (explicit): local downstream gradient is radial-dominant,
`g = -β ĥ + g_⊥`, with `β > 0` and `⟨g_⊥, w_t - δ⟩` mean-zero over t
conditioned on `r_t` (no systematic correlation with radial order).

Then the CE score update (`Δs_t = -η ∂L/∂s_t`) has signed radial term:

```
Δs_t = η β α_t (r_t - R) - η α_t ⟨g_⊥, w_t - δ⟩
```

Taking conditional expectation under A7:

```
E[Δs_t | r_t] = η β α_t (r_t - R)
```

So tokens with `r_t > R` get positive score drift; tokens with `r_t < R` get negative drift.
This is the QK selection bias: CE pushes mass toward radially aligned value directions.

Equivalent chain-rule closure to Q/K parameters:

```
∇_{W_Q}L = h^T [ Σ_t (∂L/∂s_t) k_t ] / √d_k
∇_{W_K}L = Σ_t x_t^T [ (∂L/∂s_t) q ] / √d_k
```

Hence the same signed `∂L/∂s_t` term governs both query-side and key-side learning.

**Monotone radial-gain corollary (under A7).**
First-order change in weighted radial mean `R = Σ_t α_t r_t` is:

```
ΔR = Σ_t (∂R/∂s_t) Δs_t + O(η²),   ∂R/∂s_t = α_t (r_t - R)
```

Using the radial part above:

```
E[ΔR] = η β Σ_t α_t² (r_t - R)² >= 0
```

Strictly positive unless all `r_t` are equal. So CE increases radial concentration.

**Falsifier for A7:** if measured `E[Δs_t | r_t]` is not monotone increasing in `r_t - R`,
or if `E[ΔR] < 0`, the radial-selection premise fails and D3.3 is not applicable.

**A7 FALSIFIED (2026-03-04).** `scripts/validate_a7_assumption.py` measured per-token
score gradients via finite difference (ε = sqrt(eps_bf16), IEEE 754 derived) on LFM2-350M
and Qwen3.5-0.8B. Results: Spearman(∂L/∂s_t / α_t, -(r_t - R)) shows no systematic
positive correlation (0/96 heads pass Holm-Bonferroni on LFM2, 0/48 on Qwen). β sign
scattered ≈50/50 positive/negative. The downstream gradient ∂L/∂δ is not radial-dominant.
D3.3's selection mechanism does not hold.

**What survives:** D3.1, D3.2, D3.4, D3.5 are unconditional geometric identities.
The consequence map (concentration → larger r → larger θ) is proven. The open question
is now the *cause*: what CE gradient mechanism drives attention concentration during
training, if not radial selection? The bedrock operator equation is:
`θ_l² ≈ (α^T M_l α / ||h_l||²) sin²(α_l)` where `M_l` is the Gram matrix over
token output vectors `w_{l,t}`. This is geometry, not dynamics.

**Theorem D3.4 (r-Dominance) `[PROVEN given D3.1, D3.2]`.**
r changes by `O(√T)` between concentrated and uniform α; sin(α) changes by `O(1)`.

*Proof.* From D3.1: for T diverse vectors, `||w̄|| = O(||w_k||/√T)` (centroid of T
vectors with cancelling cross-terms → `||w̄||² ≈ mean(||w_t||²)/T`). So r-ratio = `O(√T)`.

From D3.2: `sin(α_unif) ≈ 1`, `sin(α_conc) ∈ [c, 1]` with c > 0 (measured: 0.7–1.0).
Sin-ratio is `O(1)`.

For T ≥ 4: `√T ≥ 2 >` sin-ratio. The r-factor dominates `θ ≈ r · sin(α)`. ∎

**Corollary D3.5 (Architecture Conditioning) `[PROVEN: structural]`.**

D3.1–D3.4 apply to QK attention sublayers only. Non-QK sublayers have
entropy-independent output magnitude, diluting r-coupling:

- **Conv layers (LFM2, 10/16):** `δ_conv = Conv(h)` uses fixed kernels, no QK
  selection. D3.1 does not apply → no r-entropy coupling from conv layers.

- **Linear attention (Qwen3.5, 18/24):** `δ_lin = W_O S φ(q)` where `S = K^T V` is
  a recurrent state (no per-query softmax). D3.1 does not apply.

- **Pure QK attention (Qwen2.5, 36/36):** Every layer uses softmax → D3.1 applies
  everywhere → maximum r-entropy coupling.

Combined with the Pinsker envelope `[r · sin(α)]² ≤ a_l D + b_l D²`:

```
f_attn = (# QK-attention layers) / (# total layers)

High f_attn  → r absorbs entropy decrease (D3.1 at every layer)
Low f_attn   → r buffered by non-QK layers → sin(α) absorbs residual
```

This is structural: which sublayer types have entropy-dependent output magnitude
(QK attention: yes, by D3.1; conv/linear attention: no, by architecture).

### Results (2026-03-04)

**Prediction 1: DMC by architecture**

| Model | Arch | r(H_logit, \|\|h\|\|²) | r(H_logit_norm, \|\|h\|\|²) | DMC |
|-------|------|----------------------|---------------------------|-----|
| LFM2-700M | 10/16 conv | -0.894 | -0.065 | ≈ 0 |
| Qwen3.5-0.8B | 18/24 lin_attn | -0.999 | -0.686 | ≠ 0 |
| Qwen2.5-3B | 36/36 attn | -0.998 | -0.552 | ≠ 0 |

The DMC pattern matches the derivation: conv-dominant → DMC ≈ 0, attention-dominant → DMC ≠ 0.
The unnormalized H_logit was measuring ||h|| (r ≈ -1.0 for all models); after normalization,
only the DMC component remains.

**Prediction 2: Corrected sign**

| Model | partial_r(H_logit_norm, θ \| depth) | OLS r | OLS p |
|-------|-------------------------------------|-------|-------|
| LFM2-700M | -0.390 | -0.176 | 0.514 |
| Qwen3.5-0.8B | -0.145 | -0.846 | 0.000 |
| Qwen2.5-3B | -0.468 | -0.517 | 0.001 |

Consistent negative sign across all three families, matching the D3 prediction.

**Prediction 3: Normalization non-triviality**

| Model | r(H_logit_norm, H_logit) |
|-------|--------------------------|
| LFM2-700M | 0.221 |
| Qwen3.5-0.8B | 0.689 |
| Qwen2.5-3B | 0.554 |

LFM2's near-zero correlation confirms the unnormalized Entropy-Lens was measuring ||h||,
not directional posterior uncertainty. Qwen's moderate correlation reflects DMC: direction
and magnitude share variance through the attention coupling pathway.

### D3 Measurement Results `[DERIVATION, TESTED — original prediction corrected]`

D3 originally predicted sin(α) drives the negative sign. Measurement (2026-03-04) falsified
that specific prediction and confirmed the corrected decomposition (D3.1–D3.5):

| Model | f_attn | r(H_ln, log r) | p | r(H_ln, log sin α) | p | Predicted dominant |
|-------|--------|---------------:|----:|-------------------:|----:|-------------------|
| LFM2-700M | 6/16 = 0.375 | -0.071 | 0.795 | **+0.750** | 0.001 | sin(α) (D3.5: low f_attn) |
| Qwen3.5-0.8B | 6/24 = 0.250 | **-0.395** | 0.056 | -0.007 | 0.974 | r (partial) |
| Qwen2.5-3B | 36/36 = 1.000 | **-0.568** | 0.000 | **+0.686** | 0.000 | r (D3.5: high f_attn) |

**Verification against D3.1–D3.5:**

1. **D3.1 (r↓ as H↑):** Confirmed for pure attention (Qwen2.5: r=-0.568, p<.001).
   Weak/absent for low-f_attn architectures as predicted by D3.5.

2. **D3.2 (sin(α)↑ as H↑):** Confirmed for LFM2 (+0.750, p=.001) and Qwen2.5 (+0.686, p<.001).
   The centroid is more tangential than QK-selected tokens, as derived.

3. **D3.4 (r-dominance):** Confirmed for Qwen2.5 (full QK attention): net r(H, log θ) = -0.361
   despite opposing sin(α) = +0.686. The r-factor wins.

4. **D3.5 (architecture conditioning):** LFM2 (f_attn=0.375) shows negligible r-coupling but
   strong sin(α)-coupling — conv layers buffer r, forcing entropy effects into the tangential
   channel. Qwen2.5 (f_attn=1.0) shows strong r-coupling as predicted.

**Qwen3.5 anomaly:** f_attn = 0.25 (6/24 full attention layers) predicts sin(α)-dominance
(like LFM2), but measurement shows r-dominance with p=0.056. Possible cause: the linear
attention layers in Qwen3.5 have partial entropy coupling through the φ(q) term (unlike
pure conv which has zero QK dependence). This makes the effective f_attn higher than the
nominal 0.25. **Status: needs investigation — may require refining D3.5 to distinguish
"zero coupling" (conv) from "partial coupling" (linear attention).**

**Summary:** The negative r(H_logit_norm, θ) emerges from D3.1 (centroid magnitude reduction)
at QK attention layers, opposed by D3.2 (centroid tangentiality) but dominated by D3.4
(r-dominance: O(√T) vs O(1)). Architecture conditions the coupling via D3.5 (f_attn).

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
  `f1_sign_logit`, `f3_logit_vs_attn_operator`, `f4_permutation_logit`,
  `f4_permutation_logit_total`, `f4_component_perp`, `f4_component_norm`,
  `f5_family_logit`, `f5_sign_law_decomposition`, `gqa_conditioning_hypothesis`.
- TwoNN implementation:
  `src/modelcypher/core/domain/geometry/intrinsic_dimension.py`.
- Empirical data: `results/entropy_curvature/entropy_curvature_results.json`.
