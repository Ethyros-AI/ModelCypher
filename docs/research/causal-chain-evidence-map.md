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
    ↓ [EXPLORATORY: r=0.507] ← KEY DERIVATION GAP
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

**Causal perturbation test (2026-03-04):** Direct intervention on H_attn (boost prefix
attention weights). LFM2-350M: FALSIFIED (best ρ=+0.371, p=0.46). Qwen3.5-0.8B: NOT
FALSIFIED (ρ=+0.886, p=0.026). Architecture-dependent — LFM2's conv layers (10/16)
absorb the entropy perturbation. Consistent with operator split finding: H_attn ↔ H_logit
correlation is architecture-dependent (r=+0.657 LFM2, r=+0.086 Qwen3.5).
Artifact: `results/attention_validation/perturbation_experiment.txt`.

**What promotes this to `[VALIDATED]`:** Run ACT-016. Report corr(H_attn, H_logit)
per family in `results/entropy_curvature_operator_split/<model>/`.

---

### H_logit → Δcurvature  `[EXPLORATORY]` ← **KEY DERIVATION GAP**

**Measured:** r = 0.507 (H_logit → Δcurvature, cross-family). Operator: Entropy-Lens
(logit entropy), NOT attention weight entropy H_attn.

**Proposed mechanism:** Diffuse attention → output = mean(V) → output directions span the
convex hull of V columns → higher-D local patch → higher curvature. Concentrated attention
→ output ≈ single V_k → low-D local patch → lower curvature.

**Gap:** This argument has not been derived from attention + MLP mechanics. Two things
are missing:

1. **Attention step:** Derive how the distribution over V columns (determined by entropy of
   α) affects the geometry of {output_i} over an input distribution. This is computable:
   uniform α → output = (1/T)ΣV_k, meaning all outputs are the same centroid point → ID=1,
   zero curvature. Concentrated α → outputs are distinct V_k selections → higher ID. The
   direction is correct but the quantitative relationship is not formalized.

2. **MLP step:** After the attention output, a nonlinear MLP acts. The curvature measure
   (Riemannian distance between adjacent hidden states) aggregates both. How the MLP
   modulates the attention-derived geometry is not derived.

**Trilogy contribution (arXiv 2512.22471):** Value manifold at the final training
checkpoint is 1D, parameterized by posterior entropy. This is the strongest available
theoretical support for the entropy→geometry direction:

- Low entropy → most hypotheses eliminated → value manifold 1D → minimum curvature.
- High entropy → many hypotheses remaining → value manifold higher-D → higher curvature.

This is theoretical grounding for the **direction** of the relationship, not a derivation
of the curvature calculation from attention mechanics. The paper's manifold result is a
measurement (intrinsic dimension of value representations), not a derivation from the
attention operator that would let us predict curvature from entropy without measurement.

**arXiv 2512.23752 contribution:** Entropy-aligned axis appears robust across larger model
families — partial support for generalization. Does not close the derivation gap.

**Causal perturbation test (2026-03-04):** H_attn intervention → Δθ. LFM2-350M: FALSIFIED
(ρ=+0.371, p=0.46). Qwen3.5-0.8B: NOT FALSIFIED (ρ=+0.886, p=0.026). The H_attn → curvature
causal path is architecture-dependent. Conv-dominated architectures absorb the perturbation.
This narrows the derivation target: the H_logit → curvature mechanism works through posterior
certainty (Bayesian manifold), not directly through attention weight redistribution.

**GQA conditioning on norm-entropy coupling (2026-03-04, B5 test):** Spearman(GQA, R²(H→||h||²))
= -0.632, p=0.368 (N=4). Direction consistent with B5 but not significant. Needs more models.

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

**Status:** The population framing is correct and the path is derivable. The derivation has
not been executed. Until step 3 is formalized, entropy→curvature stays `[EXPLORATORY]`.
Formal derivation target: `entropy-curvature-derivation.md`.

---

## Promotion Criteria Summary

| Link | Current | Requires for promotion |
|------|---------|----------------------|
| GQA → K capacity | `[PROVEN]` | — |
| Training regime → subspace allocation | `[EXPLORATORY]` | Gradient signal causal operator |
| QK alignment → selectivity → highway | `[EXPLORATORY]` | Formal derivation: alignment → expected crossing depth |
| Attention selectivity ↔ H_attn | `[PROVEN]` | — |
| H_attn ↔ H_logit | `[EXPLORATORY]` | ACT-016: corr(H_attn, H_logit) per family; determine if proxies |
| H_logit → Δcurvature | `[EXPLORATORY]` | ACT-016 + derive: H(α), V → expected ID/curvature of output distribution |
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
