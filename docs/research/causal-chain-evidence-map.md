# Causal Chain Evidence Map

**Status:** Derivation memo (2026-03-04, updated with perturbation experiment results)
**Purpose:** Map each link in the validated causal chain to its evidence level using the
three-tier taxonomy from EVIDENCE-TAXONOMY.md, and record precisely what the Bayesian
Geometry trilogy (arXiv 2512.22471/22473/23752) formally adds vs. what remains an open
derivation gap.

---

## The Chain

```
GQA (architecture)
    ↓ [PROVEN]
K capacity
    ↓ [EXPLORATORY]
Training regime → subspace allocation
    ↓ [EXPLORATORY]
QK alignment (L0)
    ↓ [EXPLORATORY]
Attention selectivity
    ↓ [PROVEN]
H_attn (attention weight entropy)
    ↓ [EXPLORATORY] ← OPERATOR RECONCILIATION NEEDED
H_logit (logit entropy / Entropy-Lens)
    ↓ [EXPLORATORY: SIGN-REVERSED, r_norm≈-0.3] ← NORM CONFOUND (2026-03-04)
Δcurvature
    ↓ [EXPLORATORY]
Cumulative curvature → ID (r=0.821)
    ↓ [PROVEN]
Phases (highway / processing / exit)
```

**Operator note:** The `[PROVEN]` link to H_attn is definitional (attention selectivity = Shannon
entropy of QK softmax weights). The empirically measured r=0.507 uses H_logit (Entropy-Lens:
project h_l through unembedding, compute token distribution entropy). Whether H_attn and H_logit
proxy the same underlying posterior uncertainty is open (CR-EC-001). See ACT-016.

## Pipeline Boundary (Vision Scope)

What this chain controls for `mc train run`:
- Geometric monitoring operators (curvature/budget diagnostics) are valid when derived on
  the norm-corrected operator path (`H_logit_norm`, depth-controlled statistics).
- Consequence map is usable now: concentration geometry determines curvature response under
  the bedrock operator equation.
- Operational decision boundary is now explicit in code: `pipeline_gate_v1` hard-gates
  strict promotability using measured spectral/CKA/budget and online-eval stop-basis checks.

What this chain does **not** currently gate:
- Objective selection (CE vs REINFORCE variants) — this is determined by CI-based baseline
  headroom/regime selection, not by D3.3/A7.
- CLI promotion of the training pipeline — A7 falsification narrows mechanism claims but does
  not invalidate the derived training controls already wired into `mc train run`.

---

## Link-by-Link Evidence

### GQA → K capacity  `[PROVEN]`

`K_dim = Q_dim / GQA`

Architectural identity. No empirical test needed. Every implementation instantiates this
by construction.

Trilogy contribution: **none needed**. This is a definition.

---

### Training regime → Subspace allocation  `[EXPLORATORY]`

**Measured:** r(subspace_overlap, QK_alignment) = 0.93 across 4 models, 3 families.

**Gap:** Why training produces these allocations is not derived. The gradient signals that
drive Q/K subspace separation toward or away from alignment are not identified.

**Architecture term:** MISSING. Scale term: 3B–8B only.

**Trilogy contribution (arXiv 2512.22473):** First-order training dynamics account. CE
training shapes routing geometry (frame) early; content geometry (value manifold) improves
later. This provides theoretical backing for WHY training determines subspace allocation
(CE gradient acts differentially on Q/K vs. V weights). Does NOT derive the specific
allocation pattern (why Qwen ends at 0.04, Llama at 0.16).

**What promotes this to `[VALIDATED]`:** Derive the causal operator — which gradient
signals drive Q/K subspace separation. Requires a training dynamics analysis.

---

### QK alignment → Attention selectivity → Highway location  `[EXPLORATORY]`

**Measured (L0 Q/K alignment):**
| Model | GQA | L0_align | Highway |
|-------|-----|----------|---------|
| Granite-3B | 1.0 | 0.276 | 16% |
| Llama-3.2-3B | 3.0 | 0.157 | 0% |
| Granite-8B | 4.0 | 0.177 | 11% |
| Qwen3-8B | 4.0 | 0.041 | 44% |
| Qwen2.5-3B | 8.0 | 0.030 | 47% |

r(log(GQA), L0_align) = −0.88

**Geometry argument** (derivable but not formalized): Near-orthogonal Q,K → near-zero QK
inner products → uniform softmax → diffuse attention → no information filtering → high ID
persists → highway delayed.

**Gap:** The geometry argument has not been derived from attention mechanics. Given
alignment = 0.04 vs. 0.18, the expected crossing depth has not been calculated.
Architecture term not conditioned on hybrid (SSM) vs. pure-attention families.

**Trilogy contribution (arXiv 2512.22471):** Layer 0 forms near-orthogonal key bases in
small controlled models (37% reduction in off-diagonal cosines vs. random). Directly
corroborates the alignment observation. Does NOT formalize the alignment → highway timing
derivation. Their setup (small fixed architecture) does not provide the GQA variation term
that ModelCypher has.

**What promotes this to `[VALIDATED]`:** Formalize from attention mechanics. Given
alignment = a, derive expected entropy per layer, derive cumulative curvature crossing
depth. Each step is computable from the attention operator.

---

### Attention selectivity ↔ H_attn  `[PROVEN]`

`H_attn = -Σ α_i log α_i` where α_i are softmax attention weights (QK^T / √d_k).

Selectivity (concentration of α) and H_attn are definitionally equivalent. No empirical
test needed. This is Shannon entropy evaluated on the attention distribution.

Trilogy contribution: **none needed**. Theorem 1 (arXiv 2512.22471) confirms CE minimizers
track Bayesian posteriors, and entropy appears throughout as the key coordinate — but the
definitional equivalence does not require the paper.

---

### H_attn ↔ H_logit  `[EXPLORATORY]` ← **OPERATOR RECONCILIATION NEEDED**

**H_logit** (Entropy-Lens): project the layer's output hidden state `h_l` through the model's
final norm and unembedding matrix, compute Shannon entropy of the resulting token distribution.

**H_attn** and **H_logit** are distinct operators measuring different things:
- H_attn: concentration of attention over input tokens (architectural routing)
- H_logit: uncertainty about the next output token (posterior uncertainty)

**Open question:** Are they proxies for the same underlying posterior uncertainty? If
correlated (r > 0.7 per family), the derivation in `entropy-curvature-derivation.md`
(which is written for H_attn) is valid in spirit. If uncorrelated (r < 0.3), the derivation
path must be reframed around H_logit.

**Empirical evidence:** Qwen2.5-3B: r(H_attn, curvature) = -0.036, p=0.835 — not significant.
The r=0.507 reported in the causal chain uses H_logit (Entropy-Lens), not H_attn.
ACT-016 requires: run corr(H_attn, H_logit) per family and per-layer, report per-family.

**Causal perturbation test (2026-03-04, CORRECTED):** Direct intervention on H_attn (boost
prefix attention weights), exact permutation test (all 720 permutations for n=6 layers),
Holm-Bonferroni across all testable M values.
- LFM2-350M: FALSIFIED (best |ρ|=0.771, sign=−, raw p=0.103, Holm threshold=0.000019)
- Qwen3.5-0.8B: FALSIFIED (best |ρ|=0.886, sign=+, raw p=0.033, Holm threshold=0.000446)
Both models FALSIFIED. Initial report of Qwen "NOT FALSIFIED" was a false positive from
uncorrected multiple testing (112 M values) + random-shuffle permutation.
H_attn perturbation does not produce statistically significant curvature changes.
Artifact: `results/attention_validation/perturbation_experiment_corrected.txt`.

**What promotes this to `[VALIDATED]`:** Run ACT-016. Report corr(H_attn, H_logit)
per family in `results/entropy_curvature_operator_split/<model>/`.

---

### H_logit → Δcurvature  `[EXPLORATORY, SIGN-REVERSED]` ← **NORM CONFOUND DISCOVERED**

**Original measurement:** r = 0.507 (H_logit → Δcurvature, cross-family). Contaminated
by ||h||² confound (r(H_logit, ||h||²) ≈ -0.99 for all models).

**Corrected measurement (2026-03-04):** Using H_logit_norm (RMSNorm before unembedding):
- LFM2-700M: partial_r = -0.390 (negative)
- Qwen3.5-0.8B: partial_r = -0.145 (negative)
- Qwen2.5-3B: partial_r = -0.468 (negative)
- F5_norm: CONSISTENT_SIGN (all negative)

The true relationship is **negative**: higher normalized posterior entropy → less curvature.
The original positive r=0.507 was an artifact of the norm confound and depth trends.

**Operator:** Entropy-Lens (logit entropy), NOT attention weight entropy H_attn.

**Prior mechanism (REFUTED by sign reversal):** The pre-correction hypothesis predicted
high entropy → high curvature (diffuse attention → span convex hull → higher-D → more
curvature). The corrected measurement shows the **opposite**: high normalized entropy →
less curvature. The prior mechanism text is retained here for the record:
*"Diffuse attention → output = mean(V) → output directions span the convex hull of V
columns → higher-D local patch → higher curvature."* — This prediction is falsified.

**Corrected mechanism (derived sign law):**
`docs/research/entropy-curvature-derivation.md` now provides a formal two-step derivation:
1. Exact entropy-temperature identity:
   `dH/dt = -t Var_{p_t}(z_hat) <= 0` proves the raw Entropy-Lens norm confound.
2. Norm-corrected curvature bound:
   with `D = log(V) - H_logit_norm = KL(p||u_V)` and local map
   `P_perp(h) delta = B_l(p-u_V) + r_l`,
   Pinsker gives `||p-u_V||^2 <= 2D`, yielding
   `sin^2(theta) <= a_l D + b_l D^2`.
   Leading-order slope is therefore negative:
   `d sin^2(theta) / dH_logit_norm = -a_l + O(D) <= 0`.

So the sign reversal is no longer intuition-only; it is the expected leading-order behavior
of the norm-corrected operator under an explicit architecture-conditioned local map.

**D3 tangential/radial decomposition (2026-03-04, formal derivation):**
`entropy-curvature-derivation.md` now contains D3.1–D3.5:
- D3.1 `[PROVEN]`: Centroid magnitude reduction — diffuse α → smaller ||δ|| (convexity).
- D3.2 `[PROVEN]`: Centroid tangentiality — diffuse α → sin(α) → 1 (concentration of measure).
- D3.3 `[PROVEN under A7, A7 FALSIFIED]`: CE chain-rule selection bias — the math is correct
  *given* a radial-dominant downstream gradient, but A7 is empirically false. The downstream
  gradient ∂L/∂δ is not radial-dominant (0/96 LFM2-350M, 0/48 Qwen3.5-0.8B, 0/144 LFM2-700M,
  0/192 LFM2-1.2B, 0/48 Qwen3.5-2B — 5 models, 2 families, 3 scales). D3.3 is not applicable
  to real models.
- D3.4 `[PROVEN]`: r-dominance — r changes O(√T) vs sin(α) changes O(1), r wins.
- D3.5 `[PROVEN]`: Architecture conditioning — f_attn determines which factor absorbs coupling.

**Bedrock operator equation (unconditional):**
`θ_l² ≈ (α^T M_l α / ||h_l||²) sin²(α_l)` where `M_l` is the Gram matrix over `w_{l,t}`.
D3.1+D3.2+D3.4 prove: concentration raises r, r dominates θ. This is geometry, no assumptions.

**B6 three-component decomposition (2026-03-04, 10 models):**
`||P_perp(h)δ||² = ||δ||² sin²(α)`. Which sub-component carries the H_logit_norm signal?
- LFM2: ||h||² dominant (r=-0.80, -0.92). D3.1 sign correct (negative).
- Llama-3.2-3B: ||h||² dominant (r=+0.77). D3.1 sign **reversed** (positive).
- Qwen2.5-3B: sin²(α) dominant (r=+0.69). GQA=8 pushes coupling to tangential fraction.
- Qwen3.5 (4 models): all below detection floor. GatedDeltaNet decouples all components.
- Mistral-7B, Qwen3-8B: below detection floor. D3.1 sign reversed (positive).
- Cross-model: INCONSISTENT. Dominant component is architecture-dependent.
- D3.1: 7/10 pass (70%), D3.2: 9/10 pass (90%), D3.4: 6/10 pass (60%).
Source: `results/entropy_curvature_three_component/cross_model_summary.json`.

**Remaining gaps:**
1. ~~Validate A7 in-model~~ → **A7 FALSIFIED** (2026-03-04, 5 models, `scripts/validate_a7_assumption.py`).
   Diagnostic (`scripts/diagnose_a7_gradient_structure.py`): R²(radial) mean=0.18 (LFM2-350M),
   0.15 (Qwen3.5-0.8B) — radial projection explains <18% of gradient variance. Best correlate
   is token position (mean |ρ|≈0.37), not radial geometry. Gradient is moderately concentrated
   (k_eff/T≈0.53). Open question: what CE gradient mechanism drives attention concentration
   during training?
2. ~~Estimate `a_l, b_l` from measured Jacobians (`B_l`) per architecture~~ →
   **MEASURED** (2026-03-04, `scripts/estimate_bl_jacobian.py`,
   `results/bl_estimation/full_local2/`).
   - LFM2-350M (σ_d(W_u) = 2.733): ceiling ratio ∈ [0.002, 0.042], increases with depth.
     b_l ∈ [2.5e+04, 2.6e+05].
   - Qwen3.5-0.8B (σ_d(W_u) = 3.534): ceiling ratio ∈ [0.085, 0.204], roughly flat.
     b_l ∈ [4.8e+03, 2.3e+05].
   - Estimator schema now versioned (`estimator_version=bl_jacobian_v2`) and records
     conditioning diagnostics (`ptp_cond_raw`, `ptp_cond_reg`, `ridge_lambda`) plus
     holdout quality counters (`holdout_attempted`, `holdout_used`, `solve_fail_count`,
     `nonfinite_fail_count`) for numerical auditability.
   - Ceiling never tight (input-state dominates). Quadratic remainder non-negligible
     (b_l ≫ a_l at several layers). Full tables in `entropy-curvature-derivation.md`.
3. Qwen3.5 anomaly: effective f_attn may differ from nominal (linear attention has partial coupling).

**arXiv 2512.23752 contribution:** Entropy-aligned axis appears robust across larger model
families — partial support for generalization. Does not close the derivation gap.

**Causal perturbation test (2026-03-04, CORRECTED):** H_attn intervention → Δθ, with
exact permutation test + Holm-Bonferroni. Both models FALSIFIED:
- LFM2-350M: best |ρ|=0.771 (sign=−), raw p=0.103, Holm threshold=0.000019
- Qwen3.5-0.8B: best |ρ|=0.886 (sign=+), raw p=0.033, Holm threshold=0.000446
Direct attention weight perturbation does not produce significant curvature changes.
This narrows the derivation target: if entropy→curvature exists, it operates through
H_logit (posterior certainty / Bayesian manifold), not H_attn (attention weight entropy).

**GQA conditioning on norm-entropy coupling (2026-03-04, B5 test):** Spearman(GQA, R²(H→||h||²))
= -0.632, p=0.250 (exact permutation, N=4). Direction consistent with B5 but not significant.

**F-GQA-01 falsifier protocol (2026-03-04, updated with H_logit commensurability gate):**
- z_couple full regression (R²=0.686): b_g = -0.503 (p=0.063). **EXPLORATORY ONLY** —
  5 of 9 models have incommensurable z_couple (H_logit saturated, z_couple correlates noise).
- z_couple commensurable-only regression (n=4, DOF=0): **UNDERPOWERED** — cannot adjudicate.
- F1: INCONCLUSIVE (full regression CI crosses zero; commensurable-only underpowered).
- c_cancel regression (R²=0.854): d_g = 0.535 (p=0.003). F2: **SUPPORTED** — higher GQA
  produces less complete numerator/denominator cancellation. Unaffected by commensurability
  (c_cancel uses all layers, H_logit has real variation even for saturated models).
- F3 (within-family LFM2): **INCOMMENSURABLE** — both LFM2 models have saturated H_logit
  (depth-residualized range 0.007 and 0.022, both far below log(2)=0.693 nats).
  z_couple comparison is mathematically invalid.
- Overall: INCONCLUSIVE (F3 no longer triggered — incommensurable, not contradicted).
  Artifacts: `results/gqa_falsifier_protocol/*/`.

**What promotes this to `[VALIDATED]`:**
First, reconcile the operator (ACT-016): determine whether H_attn ≈ H_logit or whether
H_logit is the correct quantity to derive from. Then derive from the population map
`x → α(x) → y(x) = W_O V α(x)`:
1. `Cov[y]` spectrum as a function of `H(α)` via `k_eff = exp(H(α))`
2. Local output dimensionality bounds from V and α
3. Angle-based curvature and TwoNN response as functions of `Cov[y]` spectrum
See `entropy-curvature-derivation.md` for the formal derivation target.

---

### Cumulative curvature → ID  `[EXPLORATORY]`

**Measured:** r = 0.821 (cumulative Δcurvature → TwoNN ID, cross-family).

**Proposed mechanism:** Accumulated directional change → higher local manifold dimensionality
as measured by TwoNN nearest-neighbor distance ratios.

**Gap:** The TwoNN estimator's behavior under known curvature transformations has not been
derived. The estimator is built on the manifold hypothesis — its response to accumulated
directional change has a theoretical derivation that has not been worked out.

**Trilogy contribution (arXiv 2512.22471):** Progressive QK sharpening — each layer
provides a non-interchangeable refinement step (ablating any single middle layer increases
error >10×). This is consistent with cumulative curvature accumulation representing
cumulative Bayesian suppression steps. Theoretical grounding for cumulative interpretation,
not derivation of TwoNN behavior.

**What promotes this to `[VALIDATED]`:** Derive TwoNN estimator's expected response to
known curvature transformations. This is a mathematical question about the estimator,
independent of the neural network context.

---

### ID → Phases  `[PROVEN]`

Phases are defined by ID trajectory shape (local minima = highway, accumulation = processing,
stabilization = exit). True by construction — the taxonomy is a labeling of ID trajectory
features, not a claim about behavior.

Trilogy contribution: **none needed**. Definitions are not empirical claims.

---

## What the Trilogy Formally Adds

| Paper | Link supported | What it adds | What remains open |
|-------|---------------|--------------|-------------------|
| 2512.22471 | QK alignment → orthogonality | Empirical confirmation at Layer 0 (37% reduction); wind-tunnel quantification | Derivation of timing/crossing depth |
| 2512.22471 | Entropy → geometry direction | Value manifold 1D at final checkpoint, entropy = coordinate | Derivation from attention mechanics (curvature calculation) |
| 2512.22471 | Theorem 1 (CE → Bayes) | Architecture-agnostic proof that CE minimizer tracks posterior | Does not close entropy→curvature derivation |
| 2512.22473 | Training regime → subspace allocation | CE training dynamics differentially shapes Q/K vs V geometry | Specific allocation mechanism still underived |
| 2512.23752 | Entropy → geometry (scale) | Entropy-aligned axis robust across larger families | Scale quantification still missing for our curvature chain |

**What the trilogy does NOT add:**

- GQA cross-architecture variation term (fixed small architecture in all three papers)
- Quantitative curvature derivation (papers measure manifold ID, not Riemannian distances)
- Cumulative curvature → TwoNN ID (not studied)
- Hybrid SSM/attention architectures (Mamba tested separately, not hybrid)
- The MI/injectivity argument (independent — residual stream injectivity → MI constant)

---

## The Open Derivation Gap: H_logit → Curvature (with Operator Reconciliation)

This is the key step needed to close the chain theoretically. It has two sub-problems.

### Sub-problem 1: Operator Reconciliation (ACT-016)

**What we have:** r=0.507 (H_logit → curvature), but r=-0.036 p=0.835 (H_attn → curvature,
Qwen2.5-3B). The derivation in `entropy-curvature-derivation.md` uses H_attn (H(α)).

**What is needed:** Measure corr(H_attn, H_logit) per family (LFM2, Qwen, Llama). If correlated
(r > 0.7), the derivation is valid in spirit. If not, reframe around H_logit.

**Required artifact:** `results/entropy_curvature_operator_split/<model>/` with:
- Per-family corr(H_attn, H_logit) across layers
- Per-operator F1 falsifier outcome
- Per-family sign table for both operators

### Sub-problem 2: The Derivation Gap

**What we have:** r=0.507 (H_logit → curvature), directional theoretical grounding
from Bayesian value manifold.

**What is missing:** A derivation of the form:

```
Given attention weights α with entropy H(α),
and value matrix V with columns {v_1,...,v_T},
the expected curvature C of the layer output distribution
is a function f(H(α), V).
```

The derivable path (population framing):

The correct object is the population map `x → α(x) → y(x)` where `y(x) = W_O V α(x)`.
Curvature and ID are properties of the set `{y(x) : x ~ P(x)}`, not of a single output.

1. **Step 1 — Output covariance from α distribution:**
   Derive `Cov[y] = W_O V · Cov[α] · V^T W_O^T` over `P(x)`.
   The rank and spectrum of `Cov[y]` is bounded by `rank(Cov[α])`, which is in turn
   bounded by `k_eff(x) = exp(H(α(x)))` — the effective number of tokens attended to.
   Low entropy → small `k_eff` → low-rank `Cov[α]` → low-rank `Cov[y]` → low ID.
   High entropy → large `k_eff` → higher-rank `Cov[α]` → higher-rank `Cov[y]` → higher ID.
   **Consistent with Bayesian 5-cluster geometry (arXiv 2512.22471): peaked α → cluster
   structure in y, one cluster per attended token.**

2. **Step 2 — Local dimensionality bounds:**
   Derive bounds on local output dimensionality from `V` and `α`. For a neighborhood of
   inputs with similar `α(x)`, the local covariance rank is bounded by `k_eff` and the
   geometry of the attended V-columns. This gives local ID as a function of `H(α)` and the
   spectral structure of the selected V submatrix.

3. **Step 3 — Curvature from covariance spectrum:**
   Derive how the angle-based curvature operator (Riemannian distance between adjacent
   hidden states) depends on `Cov[y]`. Then map the covariance spectrum to expected TwoNN
   response. Both steps are tractable from the definitions of the curvature measure and
   TwoNN estimator.

**Status (updated 2026-03-04):** D3.1–D3.5 formally derive the sign and architecture
conditioning. D3.1, D3.2, D3.4, D3.5 are `[PROVEN]`; D3.3 is `[PROVEN under A7]`
(radial-dominant downstream gradient condition). A7 is **FALSIFIED** (5 models, 2 families,
3 scales: 0/528 heads show predicted monotonicity). `a_l, b_l` are **MEASURED** (ceiling never
tight, quadratic remainder non-negligible). The Pinsker envelope + D3 decomposition close
the geometric consequence; the remaining promotion blocker is identifying the cause
mechanism (training dynamics that drive attention concentration → curvature change).
B6 three-component decomposition (10 models) shows the dominant sub-component is
architecture-dependent: ||h||² for LFM2/Llama, sin²(α) for Qwen2.5, below floor
for Qwen3.5. D3.1 sign reversed on Llama/Mistral/Qwen3-8B.
Formal derivation: `entropy-curvature-derivation.md`.

---

## Promotion Criteria Summary

| Link | Current | Requires for promotion |
|------|---------|----------------------|
| GQA → K capacity | `[PROVEN]` | — |
| Training regime → subspace allocation | `[EXPLORATORY]` | Gradient signal causal operator |
| QK alignment → selectivity → highway | `[EXPLORATORY]` | Formal derivation: alignment → expected crossing depth |
| Attention selectivity ↔ H_attn | `[PROVEN]` | — |
| H_attn ↔ H_logit | `[EXPLORATORY]` | ACT-016: corr(H_attn, H_logit) per family; determine if proxies |
| H_logit → Δcurvature | `[EXPLORATORY, SIGN-REVERSED]` | D3.1+D3.2+D3.4 derive consequence (concentration → curvature). D3.3 (cause: radial selection) FALSIFIED. `a_l, b_l` MEASURED (ceiling never tight; quadratic remainder non-negligible). Geometric consequence map is proven; remaining blocker is cause mechanism (training dynamics). |
| Cumulative curvature → ID | `[EXPLORATORY]` | Derive: TwoNN behavior under curvature transformations |
| ID → Phases | `[PROVEN]` | — |

---

## MI/Injectivity Argument (Independent)

The MI impossibility proof is structurally independent of the Bayesian geometry papers:

```
h_l = h_0 + Σ_{k<l} δ_k   (residual stream)
→ injective map (fixed weights, deterministic)
→ Shannon MI(h_0; h_l) = H(h_0) for all l (constant, cannot decay)
→ therefore: residual stream reorganizes geometry, does not compress information
→ observable must be geometric (not MI)
→ linear CKA (second-order relational geometry) is the correct replacement
```

The Bayesian Geometry papers establish *what* the geometry reorganizes toward (Bayesian
posterior tracking). The injectivity argument establishes *why* MI is the wrong observable.
These are complementary, not redundant.

**Trilogy contribution to MI/injectivity:** Theorem 1 provides the *target* of the
geometric reorganization. Together: CE training → Bayesian posterior tracking (Theorem 1)
+ residual stream preserves all information (injectivity) → representation geometry moves
toward posterior tracking without information loss → linear CKA measures progress along
the posterior-tracking manifold.

---

## Citation

```
@misc{agarwal2026bayesian,
    title={The Bayesian Geometry of Transformer Attention},
    author={Naman Agarwal and Siddhartha R. Dalal and Vishal Misra},
    year={2026},
    eprint={2512.22471},
    archivePrefix={arXiv}
}
```

Companion papers: arXiv:2512.22473 (training dynamics), arXiv:2512.23752 (scale extension).
Full connection document: `docs/research/bayesian_geometry_connection.md`.
