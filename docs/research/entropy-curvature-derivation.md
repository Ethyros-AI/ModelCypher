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
GQA-conditioned: higher GQA (more key compression) → operators decouple. This is itself a
testable claim (r vs GQA monotone hypothesis: Qwen2.5 GQA=8 r=-0.094, Qwen3 GQA=4 r=0.602,
Llama GQA=3 r=0.629 — consistent with monotone decrease as GQA increases).

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

3. **The GQA-conditioning hypothesis (open, testable):** The operator correlation pattern
   (Qwen2.5 GQA=8: r=-0.094; Qwen3 GQA=4: r=0.602; Llama GQA=3: r=0.629) is consistent
   with a monotone relationship: higher GQA → more K capacity compression → greater decoupling
   of routing (H_attn) from posterior uncertainty (H_logit). This is a new testable prediction
   from this measurement that links back to the GQA → K capacity → QK alignment chain.

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

**Original statement:** `R²(H_logit → ||h||² | depth)` is monotonically decreasing with
GQA ratio across attention-based architecture families.

**Extended test (n=4, adding Qwen3.5-0.8B at GQA=4):**

| Family | GQA | Hybrid? | R²(H → ||h||²) | R²(H → ||Pδ||²) | |β_num/β_den| |
|--------|----:|---------|---------------:|----------------:|-------------:|
| Llama-3.2-3B | 3 | No | 0.826 | 0.160 | 0.42 |
| Qwen3-8B | 4 | No | 0.274 | 0.123 | 0.55 |
| Qwen3.5-0.8B | 4 | Yes | 0.000 | 0.376 | 90.12 |
| Qwen2.5-3B | 8 | No | 0.035 | 0.428 | 2.41 |
| LFM2-350M | — | Yes | 0.721 | 0.575 | 1.01 |

Spearman(GQA, R²) = −0.632 (n=4, permutation p = 0.250). **NON_MONOTONE.**

**B5 is REFINED, not confirmed.** The strict monotone hypothesis fails because Qwen3
(GQA=4, pure attention, R²=0.274) and Qwen3.5 (GQA=4, hybrid linear-attention, R²=0.000)
have the same GQA but different norm-entropy coupling strengths. GQA alone does NOT
determine the coupling.

**Refined hypothesis (B5'):** Norm-entropy coupling R² depends on BOTH GQA ratio AND
attention architecture type. At fixed GQA, hybrid/linear-attention architectures have
weaker norm-entropy coupling than pure full-attention architectures.

Evidence:
- At GQA=4: Qwen3 (pure attn) R²=0.274 > Qwen3.5 (hybrid) R²=0.000
- LFM2 (hybrid conv+attn, GQA≈2): R²=0.721, high but with near-perfect cancellation
  (|β_num/β_den|=1.01). LFM2's conv layers couple norm and entropy similarly in both
  components, producing high R² for the denominator but zero net effect on θ².

**Operator-stratified evidence inside Qwen3.5-0.8B (same model, fixed GQA=4):**
- Full-attention layers: `r(H_logit -> E_mix | depth) = -0.970`, `R²=0.941`,
  `p=0.0013`, `n=6`
- Linear-attention layers: `r(H_logit -> E_mix | depth) = +0.179`, `R²=0.032`,
  `p=0.476`, `n=18`

This isolates architecture type (core operator) as an active term at fixed GQA.
The same-GQA divergence is therefore not explainable by GQA alone.

**The two-variable structure is consistent with the broader sign split investigation:**
Hybrid architectures (Qwen3.5, LFM2) handle the norm-entropy relationship differently
than pure-attention architectures, even at the same GQA. This is the same observation
as the component-level POS/NEG sign split: hybrid architectures are in a different
geometric regime.

**Consequence for Theorem B3 (cancellation):** The cancellation regime is determined by
both GQA and architecture type:
- Low GQA + pure attention (Llama): strong norm coupling → cancellation
- Low GQA + hybrid (LFM2): strong norm coupling → perfect cancellation (ratio ≈ 1.01)
- Mid GQA + pure attention (Qwen3): moderate norm coupling → denominator dominates
- Mid GQA + hybrid (Qwen3.5): zero norm coupling → numerator passes through uncancelled
- High GQA + pure attention (Qwen2.5): no norm coupling → opposite-sign amplification

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

Raw Spearman ρ(H_logit, θ_total) is confounded by depth. Both quantities trend with depth:
H_logit generally decreases (posterior sharpens through the network) and θ_total generally
decreases (later layers make smaller angular changes). This shared depth trend creates a
**spurious positive** raw correlation on Qwen architectures while LFM2 (with its hybrid
conv+attention structure) shows a near-zero raw correlation.

The raw sign disagreement across families is what caused the original F5 FAIL:

| Model | Architecture | Raw ρ(H,θ_total) | p |
|-------|-------------|------------------:|---:|
| LFM2-350M | lfm2 (hybrid) | -0.012 | 0.966 (n.s.) |
| LFM2-700M | lfm2 (hybrid) | -0.044 | 0.871 (n.s.) |
| Qwen3.5-0.8B | qwen3.5 | +0.595 | 0.002 |
| Qwen2.5-3B | qwen2.5 | +0.487 | 0.003 |

### Depth-controlled results: method-dependent sign direction

Two deconfounding methods were applied. They agree on significance patterns but disagree
on sign direction for borderline cases:

| Model | Partial Spearman r | PS p | OLS β_total | OLS p |
|-------|-------------------:|-----:|------------:|------:|
| LFM2-350M | -0.008 | 0.978 | -28.21 | 0.217 |
| LFM2-700M | +0.016 | 0.956 | -2.88 | 0.687 |
| Qwen3.5-0.8B | -0.240 | 0.269 | -22.94 | 0.003 |
| Qwen2.5-3B | +0.346 | 0.042 | -0.039 | 0.053 |

**OLS residualization** (remove linear depth trend, fit residuals): all 4 negative.
One significant (Qwen3.5-0.8B, p=0.003); Qwen2.5-3B borderline (p=0.053).

**Partial Spearman** (remove rank-correlation with depth): mixed signs. One significant
(Qwen2.5-3B, p=0.042, positive); others non-significant.

The sign disagreement occurs at the significance boundary and in non-significant effects.
**The coupling direction after depth control is too weak and method-dependent to claim as
a universal sign law.** The robust finding is that the raw sign inconsistency (F5 FAIL)
is explained by depth confound — not that a specific direction emerges.

### Why H_attn is null on standard transformers

The formal derivation (Steps 1-3) predicts H_attn → curvature through the covariance path:
higher H_attn → larger attention support → broader Σ_α → more output covariance → more
curvature. This prediction requires that attention weights dominate the output covariance
spectrum.

In trained standard transformers, the MLP's nonlinear transform dominates the output
covariance. Attention distributes information but the MLP concentrates it. H_attn measures
routing concentration (how many keys receive attention mass), which is geometrically
decoupled from the quantity that governs curvature (how much perpendicular energy the
full sublayer update produces).

On LFM2 (hybrid conv+attention), H_attn shows significant correlation (r=0.829) because
the conv layers handle spatial transport, leaving attention with a more specialized binding
role where routing concentration directly affects output covariance.

### Architecture-dependent sublayer mechanism

The sublayer decomposition reveals architecture-dependent coupling patterns (raw Spearman,
which is reliable for sublayer correlations where depth confound is less severe):

- **Hybrid (LFM2):** ρ_core > 0, ρ_mlp < 0 → **competing_sublayers**. Conv core and
  attention handle transport and binding; MLP opposes the core signal. Confirmed on both
  LFM2-350M (ρ_core=+0.438, ρ_mlp=-0.447) and LFM2-700M (ρ_core=+0.556, ρ_mlp=-0.641).
  Cross-scale consistent.

- **Pure attention (Llama, Mistral):** ρ_core > 0, ρ_mlp positive →
  **core_pass_through**. Attention handles both transport and binding; MLP is cooperative.
  Confirmed on Llama-3.2-3B (ρ_core=+0.869, ρ_mlp=+0.723), Mistral-7B
  (ρ_core=+0.887, ρ_mlp=+0.790).

- **Hybrid linear-attention (Qwen3.5):** ρ_core > 0, ρ_mlp positive →
  **core_pass_through**. Linear attention + full attention layers (properly decomposed
  after Step 2.1 GatedDeltaNet fix: 0.8B/2B have 18 linear_attn + 6 attention = 24
  layers; 4B has 24 linear_attn + 8 attention = 32 layers). Core operator handles
  transport. Confirmed on Qwen3.5-0.8B (ρ_core=+0.713, ρ_mlp=+0.577), Qwen3.5-2B
  (ρ_core=+0.732, ρ_mlp=+0.528), and Qwen3.5-4B (ρ_core=+0.611, ρ_mlp=+0.234).
  Cross-scale consistent within Qwen3.5 family (3 models, 0.8B–4B).

The mechanism classification is architecture-predictable: hybrid (conv+attn) → competing,
pure attention or hybrid linear-attention → core_pass_through. Prediction accuracy 9/10
on the 10-model F5 set (6 families; Qwen3-8B is the sole mismatch — predicted
core_pass_through, observed competing_sublayers; source:
`results/f5_sign_law/cross_model_summary.json`).

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

**What the evidence supports:**
1. The raw sign disagreement across families is a depth confound, not a genuine
   architecture effect
2. The sublayer mechanism is architecture-dependent (competing vs pass-through)
3. H_logit is the primary operator (F1 PASS 4/4 on original set)
4. After depth control, the sign is consistently **negative** among 7/10 resolvable models
   across 4 architecture families (higher logit entropy → less angular change at fixed depth)
5. Mechanism prediction is 9/10 across all tested models (Qwen3-8B sole mismatch)
6. Qwen3.5 family scale-validated (0.8B + 2B + 4B all resolvable, all negative, all core_pass_through)

**What remains open:**
- LFM2-700M below detection floor (|r|=0.109, low effect size)
- Architecture term for component-sign split is still unknown at component level
- CR-EC-001 remains [EMPIRICAL] until architecture-term gap is closed

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

The cancellation pattern is not architecture-random, but it is also not a 1D monotone
function of GQA alone. The governing quantity is
R²(H_logit -> log||h||² | depth): how much of the representation norm's non-depth variance
is explained by posterior entropy, conditioned by both GQA and core operator type.

| Model | GQA | R²(H → ||h||²) | Cancels? |
|-------|----:|---------------:|----------|
| Llama-3.2-3B | 3 | 0.826 | Yes — denominator tracks numerator |
| Qwen3-8B | 4 | 0.274 | Yes — denominator partially tracks |
| Qwen3.5-0.8B | 4 | 0.000 | No — denominator decoupled (hybrid linear-attn) |
| Qwen2.5-3B | 8 | 0.035 | No — denominator independent |
| LFM2-350M | — | 0.721 | Yes — near-perfect (β ratio 1.01) |

Updated attention-family test (n=4: Llama, Qwen3, Qwen3.5, Qwen2.5):
Spearman(GQA, R²) = -0.632, permutation p = 0.250. **NON_MONOTONE.**

Critical counterexample: Qwen3 and Qwen3.5 both have `GQA=4` but
`R²=0.274` vs `R²=0.000`. This same-GQA split falsifies the strict monotone-GQA hypothesis
and requires an architecture-type term.

**Mechanism hypothesis (n=4, not derived):** Higher GQA compresses the key space, which
decouples the attention routing pattern from the representation norm trajectory. When routing
(which drives H_logit through the unembedding projection) is decoupled from norm, the
denominator ||h||² has independent variance that does not cancel the numerator signal.

When GQA is low (Llama, GQA=3), routing and norm are tightly coupled (R²=0.826) — H_logit
predicts ||h||² strongly, so the angular curvature ratio normalizes away much of the signal.
When GQA is high (Qwen2.5, GQA=8), routing and norm are decoupled (R²=0.035). At mid GQA,
architecture splits the regime: Qwen3 (pure attention) retains moderate coupling, while
Qwen3.5 (hybrid linear/full attention) is fully decoupled.

**This is not an "exception."** It is a GQA-conditioned and architecture-conditioned regime boundary. The question is
whether this regime boundary can be derived from the key compression geometry, or whether it
is itself an empirical coincidence at n=4.

### What this means for the entropy-curvature link

**H_logit does predict the geometry — but the geometry it predicts is not angular curvature.**
H_logit significantly predicts both ||P_perp(h)δ||² (perpendicular update energy) and ||h||²
(representation norm) after depth control. The correlations are strong (|r| = 0.37–0.76,
p < 0.03 in most families). The F4 failure for θ² occurs because these two effects *cancel
in the ratio* — and the degree of cancellation is itself GQA-conditioned.

The angular curvature operator `θ = arccos(cos(h_in, h_out))` normalizes out the very
quantity (||h||²) that H_logit most strongly predicts. It is the wrong observable for this
relationship.

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

   The signal that cancels in θ² is **overwhelmingly real** in each component separately.
   H_logit genuinely predicts both perpendicular update energy and representation norm
   after removing depth confound. The pooled sign is negative (higher entropy → smaller
   components) because the negative-sign families (LFM2, Qwen2.5) contribute more strongly.

   Per-family component correlations (depth-residualized Spearman):

   | Family | r(H, ||P_perp δ||²) | p | R² | r(H, ||h||²) | p | R² |
   |--------|--------------------:|----:|----:|-------------:|----:|----:|
   | LFM2 | -0.679 | 0.004 | 0.575 | -0.732 | 0.001 | 0.721 |
   | Llama | +0.465 | 0.013 | 0.160 | +0.762 | 0.000 | 0.826 |
   | Qwen2.5 | -0.599 | 0.000 | 0.428 | -0.198 | 0.246 | 0.035 |
   | Qwen3 | +0.366 | 0.028 | 0.123 | +0.655 | 0.000 | 0.274 |
   | Qwen3.5 | -0.302 | 0.152 | 0.376 | +0.053 | 0.806 | 0.000 |

   Qwen3.5 component-level: perp ρ=-0.302 (negative direction, p=0.152 — below significance
   with only 6 decomposable full-attention layers out of 24 total). Norm ρ=+0.053 (no signal).
   The f5 sign-law decomposition (all 24 layers) gives β_θ=-0.379 (p=7.2e-7), β_num=-0.340
   (p=0.0015), β_den=+0.004 (p=0.958) — sign_match=True. Qwen3.5 is in the NEGATIVE group.

3. **The sign direction is family-dependent. Hybrid architecture is a candidate discriminator.**

   | Family | Sign | FFN ratio | Hybrid? | GQA |
   |--------|------|----------:|---------|----:|
   | LFM2-350M | NEG | 6.50 | Yes | ≈2 |
   | Qwen2.5-3B | NEG | 5.38 | No | 8 |
   | Qwen3.5-0.8B | NEG | 3.50 | Yes | 4 |
   | Llama-3.2-3B | POS | 2.67 | No | 3 |
   | Qwen3-8B | POS | 3.00 | No | 4 |

   **FFN expansion ratio hypothesis: REFUTED.** Predicted Qwen3.5 (FFN=3.50) → POSITIVE
   (like Llama 2.67, Qwen3 3.00). Observed: NEGATIVE. The sign split is NOT a simple FFN
   ratio threshold.

   **Hybrid architecture hypothesis: EXPLORATORY / MECHANISM_UNDERSPECIFIED.** Both hybrid
   models (LFM2, Qwen3.5) are NEGATIVE — consistent (2/2). But Qwen2.5 is also NEGATIVE
   and is pure transformer, so hybrid is sufficient-but-not-necessary for NEG sign. The
   pure-transformer discriminator (Qwen2.5 NEG vs Llama/Qwen3 POS) is unresolved. Not
   promotable to a law until the full sign can be predicted from architecture alone.

   GQA does not explain it (LFM2 GQA≈2 groups with Qwen2.5 GQA=8, not with Llama GQA=3).
   Qwen3.5 GQA=4 matches Qwen3 GQA=4 but they have opposite signs.

4. **CR-EC-001 reframing.** The link is not "entropy → angular curvature" — it is
   "entropy → representation scale" AND "entropy → perpendicular update energy."
   Angular curvature (θ²) cancels the signal because it normalizes by the very quantity
   (||h||²) that entropy predicts. The correct observables are the unnormalized components.
   F4 FAIL on θ² is a measurement-operator artifact, not mechanism absence. F5
   CONSISTENT_SIGN in θ-space (among resolvable models) reflects the residual leakage
   from incomplete cancellation.

5. **Sublayer decomposition (historical 8-model snapshot, 2026-03-04):**

   Full θ²_total = θ²_core + θ²_mlp + cross_energy decomposition. All depth-controlled
   β_total are NEGATIVE in this 8-model snapshot (8/8). The component-level "POS/NEG" from
   curvature_accumulation is a different metric (raw Spearman on unnormalized components);
   the operator_split depth-controlled OLS is the definitive sign.

   | Model | ρ_core | ρ_mlp | ρ_cross | β_total | Mechanism | Resolvable |
   |-------|-------:|------:|--------:|--------:|-----------|:----------:|
   | LFM2-350M | +0.438 | -0.447 | -0.003 | -28.2 | competing | Yes |
   | LFM2-700M | +0.556 | -0.641 | +0.041 | -2.88 | competing | No |
   | Llama-3.2-3B | +0.869 | +0.722 | -0.825 | -0.180 | core_pass | Yes |
   | Qwen2.5-3B | +0.867 | +0.084 | -0.465 | -0.039 | core_pass | No |
   | Qwen3.5-0.8B | -0.241 | +0.284 | +0.230 | -17.4 | mlp_dom | Yes |
   | Qwen3.5-2B | -0.281 | +0.100 | +0.335 | -2.27 | mlp_dom | Yes |
   | Qwen3.5-4B bf16 | +0.199 | -0.338 | -0.069 | -0.587 | competing | Yes |
   | Qwen3.5-4B 4bit | -0.313 | -0.333 | +0.454 | -0.574 | mixed_flat | Yes |

   Observations:
   - **ρ_core is positive for 5/8 models.** The attention/core component correlates
     positively with logit entropy in most architectures. The three exceptions are all
     Qwen3.5 (0.8B, 2B, 4B-4bit) — interpretable as linear-attention layers (75% of
     layers) having inverted core geometry. Qwen3.5-4B bf16 is the exception within
     the family (+0.199), suggesting this property is marginal at scale.
   - **Mechanism class does NOT discriminate the component-level sign split.** Multiple
     mechanism classes appear in both POS and NEG groups.
   - **Qwen3.5 within-family scale progression:** 0.8B and 2B are mlp_dominant (ρ_core
     negative, ρ_mlp positive). At 4B, the mechanism transitions to competing_sublayers
     (ρ_core flips positive, ρ_mlp flips negative). The 75% linear-attention architecture
     produces MLP-dominated geometry at small scale; as model width increases (d: 1024 →
     2048 → 2560), the full-attention layers gain enough capacity to compete. This is a
     scale-dependent mechanism transition within a single architecture family.
   - **Mechanism prediction accuracy: 6/8** on the 8-model set. Qwen3.5-4B bf16 and
     Qwen3.5-4B 4bit both mispredicted (predicted mlp_dominant from architecture, observed
     competing_sublayers and mixed_or_flat respectively). The candidate law's mechanism
     prediction does not account for the scale-dependent transition in Qwen3.5.

   **Quantization impact on sublayer decomposition (Qwen3.5-4B bf16 vs 4-bit):**

   | Metric | bf16 | 4-bit | Δ | Interpretation |
   |--------|-----:|------:|--:|----------------|
   | β_total | -0.587 | -0.574 | +0.013 | Consistent (both negative, <3% difference) |
   | ρ_core | +0.199 | -0.313 | -0.512 | **Sign flip** — core sublayer coupling inverted |
   | ρ_mlp | -0.338 | -0.333 | +0.005 | Consistent (both negative, <2% difference) |
   | ρ_cross | -0.069 | +0.454 | +0.523 | **Sign flip** — cross-energy coupling inverted |
   | Mechanism | competing | mixed_flat | — | Classification changes |
   | r_value | -0.468 | -0.461 | +0.007 | Consistent |
   | Resolvable | Yes | Yes | — | Both above MDE threshold |

   Key finding: **Quantization preserves aggregate geometry but distorts sublayer
   attribution.** β_total and r_value (the depth-controlled coupling strength) are nearly
   identical between bf16 and 4-bit — the overall H_logit → θ² relationship is robust to
   4-bit quantization. But the sublayer decomposition diverges: ρ_core flips sign (+0.199 →
   -0.313), ρ_cross flips sign (-0.069 → +0.454), and the mechanism classification changes.

   This means: (a) 4-bit quantization is safe for aggregate measurements (β_total, overall
   sign, resolvability), (b) sublayer-level attribution (ρ_core, ρ_cross, mechanism class)
   requires bf16 precision, (c) the ρ_core sign flip is consistent with quantization noise
   in the unembedding projection — H_logit is computed via weight-tied logits, and 4-bit
   quantization of embed_tokens introduces reconstruction error that propagates into the
   logit entropy estimate. The MLP sublayer (ρ_mlp) is unaffected because it depends on
   activation geometry, not the unembedding matrix.

   Source: `results/f5_sign_law_full/cross_model_summary.json` (8-model run)

6. **Open questions (ordered by priority):**
   a. **What distinguishes the component-level POS families (Llama) from NEG
      (LFM2, Qwen2.5, Qwen3.5)?** The operator_split shows ALL models have negative
      depth-controlled β_total, so the "sign split" is a property of the unnormalized
      component metric (curvature_accumulation), not the θ²-level OLS. The most prominent
      sublayer differentiator is ρ_cross: Llama has strongly negative cross-energy
      correlation (−0.83), meaning sublayer updates become more orthogonal at higher
      entropy. Whether this explains the component-level sign reversal in
      curvature_accumulation remains EXPLORATORY.
   b. **Qwen3.5 scale-dependent mechanism transition.** At 0.8B–2B, Qwen3.5 is
      mlp_dominant. At 4B, it transitions to competing_sublayers. Is there a critical
      width (or attention-layer capacity) at which full-attention layers overcome the
      linear-attention majority? The 3-point scale series (0.8B/2B/4B) is insufficient
      to locate the transition precisely.
   c. **Can the GQA → cancellation pattern be derived from key compression geometry?** The
      monotone relationship (n=3) needs both more families and a formal derivation connecting
      GQA ratio to the coupling between routing entropy and representation norm.

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

**Geometric argument:** In networks where the unembedding matrix W_u is approximately
orthogonal (or has well-spread singular directions), the softmax entropy of W_u h is
monotonically related to the effective projection dimension of h in vocab space. When
||h||² is large, the logit vector z = W_u h has larger magnitude, which (before
LayerNorm) concentrates the softmax and reduces H_logit. LayerNorm removes the norm
dependence, but the *residual stream* norm (before LayerNorm) carries information about
how the network has accumulated evidence.

The sign and strength of this coupling depends on how tightly the norm trajectory is
bound to the entropy trajectory — which is architecture-conditioned (GQA + core operator):

- **Llama (GQA=3), LFM2:** Strong coupling (R²=0.72–0.83). The attention routing
  pattern (which drives the hidden state norm through value accumulation) is tightly
  correlated with the unembedding projection (which H_logit measures). When routing
  concentrates on informative keys, both ||h||² grows (value accumulation) and H_logit
  drops (sharper posterior). The coupling is near-deterministic.

- **Qwen3 (GQA=4, pure attention):** Moderate coupling (R²=0.274). Norm still tracks
  entropy enough to partially cancel numerator signal.

- **Qwen2.5 (GQA=8), Qwen3.5 (GQA=4, hybrid):** Weak coupling (R²=0.00-0.04). In Qwen2.5,
  high GQA key compression is consistent with decoupling. In Qwen3.5, the same-GQA
  comparison to Qwen3 indicates hybrid linear-attention architecture adds an independent
  decoupling mechanism.

### Proposition 8: H_logit Predicts Perpendicular Update Energy [EXPLORATORY]

**Empirical basis (5 families, depth-controlled):**

| Family | r(H_logit, log||P_perp(h)δ||²) | p | R² |
|--------|-------------------------------:|---:|---:|
| LFM2-350M | -0.679 | 0.004 | 0.575 |
| Llama-3.2-3B | +0.465 | 0.013 | 0.160 |
| Qwen2.5-3B | -0.599 | 0.000 | 0.428 |
| Qwen3-8B | +0.366 | 0.028 | 0.123 |
| Qwen3.5-0.8B | -0.302 | 0.152 | 0.376 |

**Geometric argument:** The perpendicular update `P_perp(h)δ` measures how much the
layer's output rotates the hidden state away from its current direction. When H_logit
is low (sharp posterior), h is concentrated in a low-dimensional subspace of vocab
space. The layer's update δ (from attention + MLP) is constrained by the same
representational bottleneck that produces the sharp posterior — the network has
committed to a specific interpretation, and the update respects this commitment.

Formally, if we decompose δ into vocab-aligned and vocab-orthogonal components:

```
δ = P_W δ + P_W_perp δ
```

where `P_W` projects onto the column span of `W_u^T` (the unembedding subspace),
then ||P_perp(h)δ||² depends on both the magnitude of δ and the angle between δ
and h. When h is in a low-entropy (concentrated) region of vocab space, the layer's
update tends to stay within the same concentrated region (the network "deepens its
commitment" rather than exploring new directions), reducing perpendicular energy.

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

**Updated empirical basis (10 models, 2026-03-04):**

Cross-family Spearman(GQA, r(H_logit, H_attn)):

```
ρ = -0.736, p = 0.015, n = 10
```

| Model | GQA | r(H_logit, H_attn) | Family |
|-------|----:|-------------------:|--------|
| LFM2-350M | 2 | +0.600 | LFM2 |
| LFM2-700M | 3 | +0.657 | LFM2 |
| Llama-3.2-3B | 3 | +0.294 | Llama |
| Mistral-7B | 4 | -0.136 | Mistral |
| Qwen3-8B | 4 | +0.645 | Qwen3 |
| Qwen3.5-0.8B | 4 | -0.086 | Qwen3.5 |
| Qwen3.5-2B | 4 | -0.086 | Qwen3.5 |
| Qwen3.5-4B | 4 | -0.071 | Qwen3.5 |
| Qwen3.5-4B-4bit | 4 | -0.143 | Qwen3.5 |
| Qwen2.5-3B | 8 | -0.299 | Qwen2.5 |

**Within-family stability (Qwen3.5, all GQA=4):** r = -0.07 to -0.14, stable across
0.8B–4B scale range. Scale does not affect operator coupling at fixed GQA.

**Mechanism:** GQA ratio = n_heads / n_kv_heads. Higher GQA means more query heads
share each key-value head. This compresses the key space:

1. **Low GQA (e.g., 2-3):** Each KV head serves 2-3 query heads. Key vectors
   maintain high-dimensional selectivity. Attention routing (α) strongly determines
   which values accumulate → routing entropy (H_attn) and posterior entropy (H_logit)
   are coupled through the value accumulation path.

2. **High GQA (e.g., 8):** Each KV head serves 8 query heads. Key vectors must
   represent a compressed subspace to serve diverse queries. Attention routing
   becomes less informative about the specific hidden state trajectory → H_attn
   and H_logit decouple.

**Connection to cancellation:** When GQA is low and operators are coupled, routing
(which drives ||h||² through value accumulation) is tightly bound to posterior
(which H_logit measures). Both numerator and denominator of θ² respond to the same
underlying routing signal → proportional β coefficients → cancellation.

When GQA is high and operators are decoupled, ||h||² is driven by MLP processing
(which is independent of H_logit) while ||P_perp(h)δ||² retains some H_logit
dependence through the attention-specific update. The β coefficients diverge →
cancellation breaks → θ² retains H_logit signal.

**Prediction (testable):** At fixed depth, models with higher GQA should show larger
|β_total| (depth-controlled H_logit → θ² coefficient), i.e., less cancellation.
Among the 7 resolvable models in the F5 analysis:

| Model | GQA | |β_total| | |r| |
|-------|----:|--------:|----:|
| Qwen3.5-0.8B | 4 | 22.94 | 0.580 |
| Qwen3.5-2B | 4 | 2.21 | 0.556 |
| Qwen3.5-4B | 4 | 0.46 | 0.503 |
| Llama-3.2-3B | 3 | 0.18 | 0.703 |
| Mistral-7B | 4 | 8.34 | 0.741 |
| LFM2-350M | 2 | 28.21 | 0.326 |

β_total magnitude is confounded by model scale (different hidden dimensions produce
different absolute β). The |r| values do not show a clean GQA monotonicity within
the resolvable set because the cancellation degree also depends on architecture family.
The prediction is directional (higher GQA → less cancellation on average) but not
yet cleanly separable from family effects at n=10.

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

**Date:** 2026-03-04
**Status:** PRE-REGISTERED (not yet executed)

### Hypothesis

GQA ratio causally controls the coupling between H_logit and H_attn, independent of
architecture family. Specifically: within models of the same architecture family,
increasing GQA (by modifying the number of KV heads while holding architecture constant)
decreases r(H_logit, H_attn).

### Current evidence and its limitation

Cross-family Spearman(GQA, r(H_logit, H_attn)) = -0.736, p=0.015, n=10 (6 families).
This is **confounded by architecture**: GQA co-varies with family (all Qwen3.5 models
have GQA=4; Qwen2.5 has GQA=8; etc.). The correlation could reflect architecture
differences rather than GQA per se.

### Test design

**Approach 1 (Within-family GQA variation — gold standard):**

Find or construct models from the same family with different GQA ratios. Candidates:
- Llama-3.2-1B (GQA=4) vs Llama-3.2-3B (GQA=3 — actually n_kv_heads=8, n_heads=24)
  → verify actual GQA ratios differ
- Qwen2.5-0.5B through Qwen2.5-7B if different sizes use different GQA configurations
- Any family that varies KV head count across scale

For each within-family pair with different GQA:
```
Prediction: model with higher GQA has lower |r(H_logit, H_attn)|
Falsifier: model with higher GQA has HIGHER |r(H_logit, H_attn)| (p < 0.05)
```

Minimum: 3 families with ≥2 GQA values each → 3 within-family tests.

**Approach 2 (Partial correlation controlling for family — silver standard):**

Pool all 10+ models. Compute partial Spearman(GQA, r(H_logit, H_attn) | family),
treating family as a categorical covariate.

```
Prediction: partial ρ < 0 (GQA effect persists after family control)
Falsifier: partial ρ ≥ 0 (GQA effect explained entirely by family)
```

This is weaker because within-family GQA variation is limited in our current model set,
so the partial correlation is dominated by cross-family variation.

**Approach 3 (Norm-coupling regression — mechanistic):**

If GQA controls cancellation through R²(H_logit → ||h||² | depth), then:

```
R²_norm(model) = R²(H_logit → log||h||² | depth)
Prediction: Spearman(GQA, R²_norm) < 0, controlling for family
Falsifier: Spearman(GQA, R²_norm) ≥ 0
```

Current data (n=3 attention-based families): Spearman = -1.000 (but p=0.167 = 1/6).
Need n ≥ 5 families for p < 0.05.

### Promotion criteria

To promote Hypothesis B5 (GQA modulates norm-entropy coupling) from [EXPLORATORY]:
1. Approach 1 passes on ≥ 2/3 within-family pairs, OR
2. Approaches 2 + 3 both pass with p < 0.05 on n ≥ 5 families

To refute:
1. Approach 1 fails on ≥ 2/3 within-family pairs (opposite direction), OR
2. Both approaches 2 and 3 show ρ ≥ 0

### Required models (not yet available)

Priority order for acquisition:
1. **Llama-3.2-1B** — same family as Llama-3.2-3B, likely different GQA
2. **Qwen2.5-1.5B or Qwen2.5-7B** — same family as Qwen2.5-3B, may differ in GQA
3. **Gemma-2-2B and Gemma-2-9B** — new family entirely, known to vary GQA across scale

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
