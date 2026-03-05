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

3. **The GQA-conditioning hypothesis (F-GQA-01, INCONCLUSIVE):** Cross-family regression
   (n=9) gives b_g = -0.503 (p=0.063), direction consistent but CI crosses zero. Within-family
   LFM2 shows opposite sign (z_couple increases with GQA). The correct claim form is
   GQA + architecture, not a 1D monotone GQA law. See GQA Conditioning Hypothesis section
   for full F-GQA-01 results.

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

**Status: PASSES (operator coupling), INCONCLUSIVE (regression falsifier)** (updated 2026-03-04)

**F-GQA-01 Falsifier Protocol (2026-03-04):** Pre-registered test of GQA modulation claim.
Design matrix `[1, log(GQA), I(hybrid), log(d)]` with 9 models. VIF(log(GQA)) = 1.22
(well-identified). cond(X'X) = 37952 (driven by intercept scale, not predictor collinearity).

z_couple regression (n=9, DOF=5, R²=0.686) [EXPLORATORY — includes incommensurable models]:
- b_g = -0.503 (SE=0.211, t=-2.39, p=0.063, 95% CI [-1.044, 0.038])
- CI crosses zero → **F1: INCONCLUSIVE** (direction consistent but not significant at 95%)

Commensurable-only regression (n=4, DOF=0): **UNDERPOWERED** (zero degrees of freedom).

c_cancel regression (n=9, DOF=5, R²=0.854):
- d_g = 0.535 (p=0.003, CI excludes zero) → **F2: SUPPORTED**. Unaffected by
  commensurability (c_cancel uses all layers, not attention-only).

F3 (within-family LFM2): **INCOMMENSURABLE.** Both LFM2 models have saturated H_logit
(resid range 0.007, 0.022 < log(2)=0.693 nats). z_couple correlates noise, not posterior
concentration. See `ENTROPY-CURVATURE-GQA-FALSIFIER-PROTOCOL.md` for full commensurability
table and H_logit Saturation Gate derivation (5/9 models incommensurable).

Artifacts: `results/gqa_falsifier_protocol/*/`

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

### Derivation C: `B_l` Jacobian Factorization (Gap-2 Target)

Gap-2 asks for measurable `a_l, b_l` via a depth-local Jacobian object. The reduced
algebra is:

```
f_l(h)=P_perp(h)delta(h),  g_l(h)=p(h)-u_V,  f_l(h)=B_l g_l(h)+r_l(h)
B_l = J_f J_g^+,  ||B_l|| <= ||J_f||/sigma_min(J_g)
```

with

```
J_f = d(P_perp delta)/d xi,   J_g = J_softmax(z) W_u J_RMSNorm(h)
```

and

```
a_l = 4||B_l||^2 / ||h+delta||^2
    <= 4||J_f||^2 / (||h+delta||^2 sigma_min(J_g)^2)
```

Interpretation: architecture terms bound the ceiling (`W_O,W_V,W_Q,W_K,d_k,W_u`), while
input-state terms (`alpha,p,r,Cov_alpha(v,k),sigma_min(J_g)`) tighten the realized value.
This is the required architecture/input split for the Gap-2 measurable.

**Operator note (pipeline integration):** `pipeline_gate_v1` consumes only these measured
invariants (spectral safety, stop-basis degradation, CKA-vs-bound bundle, saturation, gain)
as promotability checks. It does not rely on open-cause mechanism claims (e.g., A7/D3.3).

#### Gap-2 Measurement Results (2026-03-04)

Measured via `scripts/estimate_bl_jacobian.py` on 3 probes per layer.
Full data: `results/bl_estimation/full_local2/`.
Perturbation scale ε = √(eps_bf16) = 8.84e-02 (IEEE 754 derived).

**LFM2-350M** (σ_d(W_u) = 2.733, 6 attention layers of 16):

| Layer | a_l ceiling | a_l measured | ratio | b_l measured |
|------:|------------:|-------------:|------:|-------------:|
| 2 | 7.33e+09 | 1.83e+07 | 0.0024 | 2.46e+07 |
| 5 | 7.62e+06 | 2.43e+04 | 0.0036 | 1.63e+05 |
| 8 | 2.16e+05 | 1.33e+03 | 0.0063 | 1.07e+05 |
| 10 | 1.33e+05 | 1.77e+03 | 0.013 | 2.56e+05 |
| 12 | 5.14e+04 | 1.11e+03 | 0.022 | 6.97e+04 |
| 14 | 1.11e+07 | 4.26e+05 | 0.042 | 7.77e+04 |

**Qwen3.5-0.8B** (σ_d(W_u) = 3.534, 6 full-attention layers of 24):

| Layer | a_l ceiling | a_l measured | ratio | b_l measured |
|------:|------------:|-------------:|------:|-------------:|
| 3 | 1.51e+04 | 1.68e+03 | 0.111 | 7.90e+04 |
| 7 | 1.09e+04 | 1.72e+03 | 0.157 | 2.34e+05 |
| 11 | 1.17e+04 | 1.01e+03 | 0.085 | 1.35e+05 |
| 15 | 3.96e+03 | 7.82e+02 | 0.204 | 7.56e+04 |
| 19 | 3.90e+03 | 7.95e+02 | 0.201 | 1.44e+04 |
| 23 | 1.26e+06 | 1.43e+05 | 0.125 | 4.76e+03 |

**Interpretation:**

1. **Ceiling is never tight.** Ratio ∈ [0.002, 0.042] for LFM2, [0.085, 0.204] for Qwen3.5.
   Input-state terms dominate the realized coupling — the architecture ceiling is 5–400×
   above the measured value.

2. **Depth trend differs by architecture.** LFM2 ratio increases with depth (0.002 → 0.042),
   meaning deeper layers use a larger fraction of their architectural capacity. Qwen3.5 ratio
   is roughly flat (~0.1–0.2). This is consistent with the architecture conditioning in D3.5.

3. **Quadratic remainder is non-negligible.** At several layers b_l ≫ a_l (e.g., LFM2
   layer 10: b_l = 2.56e+05 vs a_l = 1.77e+03). The Pinsker envelope
   `sin²(θ) ≤ a_l·D + b_l·D²` needs both coefficients — the linear term alone understates
   the coupling strength at large KL divergence.

Detailed derivation steps (previous C1-C5 expansion) are preserved in project artifacts and
summarized in [OPEN-MATHEMATICAL-QUESTIONS.md](./OPEN-MATHEMATICAL-QUESTIONS.md).

**Runbook: reading `cond_raw` vs `cond_reg` in `results/bl_estimation/*`:**
- `cond_raw` is the condition number of `P_fit^T P_fit` before ridge.
- `cond_reg` is the condition number after adding `ridge_lambda I`.
- Target rule is fixed by IEEE precision: `kappa_target = 1/sqrt(eps_f32)`.
- `cond_reg <= kappa_target` means the local solve is numerically resolved at float32 scale.
- Large gap (`cond_raw >> cond_reg`) indicates ridge is doing essential stabilization.
- High `solve_fail_count` or `nonfinite_fail_count` with low `holdout_used/attempted`
  marks layers where local Jacobian inversion remains fragile and should be interpreted
  as measurement-limited, not mechanism-falsifying.

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

Directional predictions (pre-registered):
1. `|r(H_logit_norm, ||h||²)| <= MDE` (norm confound removed).
2. `r(H_logit_norm, θ_total | depth)` adjudicates the coupling link.
3. `1 - r(H_logit_norm, H_logit) > MDE` (normalization is non-trivial).

**Mixed-outcome rule (FIRST_PRINCIPLES_REVIEW_PROTOCOL.md §4):**
If sign differs across families, classify as MECHANISM_UNDERSPECIFIED unless architecture
terms predict the divergence. Report per-family and aggregate separately.

### E_mix Architecture Split (from existing results)

Independent of the norm confound, `E_mix` still shows architecture-conditioned sign split:
attention channels mostly negative (core/MLP cooperation), conv and some families positive
(opposition), linear-attention near zero. Keep as exploratory unless promoted with its own
claim form and falsifier set.

### Derivation D1: H_logit_norm Is Direction-Only (Identity)

RMSNorm(h)_i = γ_i · h_i / RMS(h), where RMS(h) = ||h|| / √d.

Therefore: `RMSNorm(h) = √d · γ ⊙ ĥ`, where `ĥ = h/||h||`.

Thus `H_logit_norm = f(ĥ;γ,W,d)` depends on direction only (`||h||` exactly divided out).
Residual correlation with `||h||²` after normalization is therefore DMC
(direction-magnitude coupling), not direct norm leakage.

### Derivation D2: DMC From Residual Stream Geometry

In a residual stream `h_l = h_0 + Σ_{k<l} δ_k`, decompose updates into radial and
tangential components:

```
δ_k^∥ = ⟨δ_k, ĥ_k⟩ ĥ_k        (radial: changes ||h||, preserves ĥ)
δ_k^⊥ = δ_k - δ_k^∥              (tangential: changes ĥ, preserves ||h||)
```

First-order:

```
||h_{k+1}||² ≈ ||h_k||² + 2||h_k||·⟨δ_k, ĥ_k⟩ + ||δ_k||²    (magnitude from radial)
ĥ_{k+1} ≈ ĥ_k + δ_k^⊥ / ||h_k||                               (direction from tangential)
```

Hence DMC is near zero when tangential and radial trajectories decouple across depth.
Empirically this is architecture-conditioned: pure attention retains coupling; conv-heavy
paths suppress it.

### Derivation D3: Negative Sign From Tangential/Radial Decomposition

From h_out = h_in + δ:

```
cos(θ) = (1 + r cos α) / √(1 + 2r cos α + r²)
```

where `r = ||δ||/||h_in||` and `α = angle(h_in, δ)`.

For small `r`, `θ ≈ r·sin(α)`. The original D3 mechanism (`sin(α)` dominance) was falsified;
the corrected mechanism is `r` dominance with architecture-conditioned buffering:
QK-attention-rich stacks transmit `r` coupling strongly, while conv/linear-attention-heavy
stacks partially absorb it.

Compact empirical summary (2026-03-04):
- **DMC by architecture:** `r(H_logit_norm, ||h||²)` is near zero for conv-dominant LFM2 and
  nonzero for Qwen families.
- **Corrected sign:** `partial_r(H_logit_norm, θ | depth)` is negative in all tested families.
- **Normalization non-triviality:** `r(H_logit_norm, H_logit)` far from 1 confirms operator
  change (not cosmetic renaming).
- **A7 status:** radial-selection training mechanism (old D3.3) is falsified across 5 models
  (0/528 heads, 2 families, 3 scales). Diagnostic: R²(radial) ≈ 0.16 mean — radial projection
  explains <18% of gradient variance. Best correlate is token position (mean |ρ|≈0.37).
  This does not invalidate geometric consequence equations (D3.1/D3.2/D3.4/D3.5).

Numerical tables and per-head diagnostics are kept in:
- `results/entropy_curvature_operator_split/*`
- `results/gqa_falsifier_protocol/*`
- [OPEN-MATHEMATICAL-QUESTIONS.md](./OPEN-MATHEMATICAL-QUESTIONS.md)
- [causal-chain-evidence-map.md](./causal-chain-evidence-map.md)

---

### Proposition B6: Three-Component Decomposition [EMPIRICAL, ARCHITECTURE-DEPENDENT]

**Statement:** The curvature numerator decomposes as:

```
||P_perp(h)δ||² = ||δ||² sin²(α)    where α = angle(h, δ)
```

In log space: `log(||P_perp(h)δ||²) = log(||δ||²) + 2·log(sin(α))`.

Three sub-components carry independent geometric meaning:
1. `||δ||²` = E_total — update magnitude (D3.1: centroid averaging)
2. `sin²(α)` — tangential fraction (D3.2: tangentiality)
3. `||h||²` = h_in_norm_sq — hidden state norm (B5/B7: norm-entropy coupling)

**Question:** Which sub-component carries the H_logit_norm signal?

**Method:** Depth-residualized Spearman correlation r(H_logit_norm, Y_X | depth) for each
component X, with AR(1)-corrected effective df (Bretherton 1999), Fisher-SE MDE, and
500-permutation exceedance test.

**10-model results (2026-03-04):**

| Model | L | MDE | r(δ²) | r(sin²) | r(h²) | r(num) | Dominant |
|-------|--:|----:|------:|--------:|------:|-------:|----------|
| LFM2-350M | 16 | 0.762 | -0.688 | +0.594 | **-0.797*** | -0.715 | ||h||² |
| LFM2-700M | 16 | 0.762 | -0.656 | +0.750 | **-0.918*** | -0.641 | ||h||² |
| Llama-3.2-3B | 28 | 0.487 | +0.485 | +0.339 | **+0.768*** | +0.466 | ||h||² |
| Mistral-7B | 32 | 0.762 | +0.430 | +0.696 | +0.602 | +0.433 | NONE |
| Qwen2.5-3B | 36 | 0.635 | -0.626 | **+0.686*** | -0.201 | -0.595 | sin²(α) |
| Qwen3-8B | 36 | 0.762 | +0.400 | +0.225 | +0.706 | +0.414 | NONE |
| Qwen3.5-0.8B | 24 | 0.266 | -0.256 | -0.007 | +0.099 | -0.252 | NONE |
| Qwen3.5-2B | 24 | 0.345 | -0.238 | +0.034 | -0.043 | -0.265 | NONE |
| Qwen3.5-4B | 32 | 0.314 | -0.174 | +0.297 | -0.067 | -0.156 | NONE |
| Qwen3.5-4B-4bit | 32 | 0.328 | -0.201 | +0.143 | +0.113 | -0.185 | NONE |

\* = resolvable (|r| > MDE). Closure checks: all 10 models PASS (max_rel_gap < eps_bf16).

**D3 prediction pass rates:**
- D3.1 (r(H, log(||δ||²)) < 0): 7/10 (70%). Fails on Llama, Mistral, Qwen3-8B (positive sign).
- D3.2 (r(H, 2·log(sin(α))) > 0): 9/10 (90%). Fails on Qwen3.5-0.8B only.
- D3.4 (|r_delta| > |r_sin|, magnitude dominates): 6/10 (60%).

**Cross-model consistency: INCONSISTENT.** Dominant component is architecture-dependent:

| Architecture group | Dominant | D3.1 sign | Models |
|-------------------|----------|-----------|--------|
| Hybrid conv+attn (LFM2) | ||h||² | negative (correct) | 350M, 700M |
| Pure attention (Llama) | ||h||² | **positive** (reversed) | 3.2-3B |
| Pure attention (Mistral) | NONE (below floor) | **positive** (reversed) | 7B |
| GQA=8 (Qwen2.5) | sin²(α) | negative (correct) | 3B |
| GQA=4 (Qwen3) | NONE (below floor) | **positive** (reversed) | 8B |
| Hybrid linear+full attn (Qwen3.5) | NONE (below floor) | negative (correct) | 0.8B, 2B, 4B, 4B-4bit |

**Architecture-conditioned findings:**
1. **LFM2 + Llama: ||h||² dominant.** The hidden state norm carries the entropy signal.
   But LFM2 has negative D3.1 (as predicted) while Llama has positive D3.1 (reversed).
   The conv-vs-attention path difference in how ||δ|| responds to entropy is a real signal.
2. **Qwen2.5: sin²(α) dominant.** High GQA (8) pushes the coupling into the tangential fraction.
3. **Qwen3.5 (all 4 models): below detection floor.** Despite lowest MDE (0.27-0.33), all
   correlations are weak (|r| < 0.30). The GatedDeltaNet + linear attention architecture
   decouples all three sub-components from H_logit_norm.
4. **Llama/Mistral/Qwen3-8B: D3.1 sign flip.** r(H_logit_norm, log(||δ||²)) is POSITIVE,
   not negative. Higher normalized entropy → LARGER update magnitude in these architectures,
   contradicting the centroid averaging prediction. This is a genuine architecture-dependent
   reversal, not a measurement artifact (all pass closure checks).

**Status: [EMPIRICAL, ARCHITECTURE-DEPENDENT].** B6 confirms the decomposition identity
(closure: all pass) but the dominant sub-component is not universal. The which-component
question has an architecture-dependent answer. No single sub-component explains the
H_logit_norm → curvature coupling across all families.

Source: `results/entropy_curvature_three_component/cross_model_summary.json`,
`scripts/entropy_curvature_three_component.py`.

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
