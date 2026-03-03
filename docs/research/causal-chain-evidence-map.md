# Causal Chain Evidence Map

**Status:** Derivation memo (2026-03-03)
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
Entropy
    ↓ [EXPLORATORY] ← KEY DERIVATION GAP
Δcurvature (r=0.507)
    ↓ [EXPLORATORY]
Cumulative curvature → ID (r=0.821)
    ↓ [PROVEN]
Phases (highway / processing / exit)
```

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

### Attention selectivity ↔ Entropy  `[PROVEN]`

`H = -Σ α_i log α_i` where α_i are softmax attention weights.

Selectivity (concentration of α) and entropy are definitionally equivalent. No empirical
test needed. This is Shannon entropy evaluated on the attention distribution.

Trilogy contribution: **none needed**. Theorem 1 (arXiv 2512.22471) confirms CE minimizers
track Bayesian posteriors, and entropy appears throughout as the key coordinate — but the
definitional equivalence does not require the paper.

---

### Entropy → Δcurvature  `[EXPLORATORY]` ← **KEY DERIVATION GAP**

**Measured:** r = 0.507 (Entropy → Δcurvature, cross-family).

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

**What promotes this to `[VALIDATED]`:**
- Derive: given entropy H(α) of attention weights, expected dimensionality of {output_i}
  over input distribution.
- This requires: (a) distribution over V-column selections, (b) expected rank of their
  convex hull, (c) connection to curvature measure.
- Steps (a) and (b) are tractable from linear algebra. Step (c) requires working out the
  relationship between manifold dimensionality (as measured by TwoNN or Riemannian distance)
  and attention entropy.

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

## The Open Derivation Gap: Entropy → Curvature

This is the key step needed to close the chain theoretically.

**What we have:** r=0.507, directional theoretical grounding from Bayesian value manifold.

**What is missing:** A derivation of the form:

```
Given attention weights α with entropy H(α),
and value matrix V with columns {v_1,...,v_T},
the expected curvature C of the layer output distribution
is a function f(H(α), V).
```

The derivable path:

1. **Uniform case (H max):** α_i = 1/T → output_i = (1/T)Σ_j v_j for all i. All outputs
   collapse to the same centroid. ID = 0, curvature = 0 by degeneracy. (This is the wrong
   direction — need to think about the population of inputs, not a single input.)

2. **Correct framing:** Over a distribution of inputs P(x), different inputs produce
   different attention patterns α(x). Low-entropy models produce peaked α(x) concentrated
   on a few v_j. High-entropy models produce diffuse α(x). The question is: what is the
   intrinsic dimensionality of {output(x) : x ~ P(x)} under each regime?

3. **Low entropy regime:** outputs ≈ discrete selections from {v_j} → outputs form clusters
   (one per attended token) → within-cluster variation low → low ID. **Consistent with
   Bayesian 5-cluster Mamba geometry (arXiv 2512.22471).**

4. **High entropy regime:** outputs ≈ weighted averages across many v_j → outputs fill the
   convex hull of V → higher-D → higher ID. The dimension is bounded by rank(V), which is
   typically full-rank.

5. **Connection to curvature:** Riemannian curvature as measured (adjacent hidden state
   angle) correlates with ID because both measure local manifold dimensionality. This
   connection needs to be derived from the TwoNN estimator's behavior and the angle-based
   curvature definition.

**Status:** The path is visible. The derivation has not been executed. Until step 5 is
formalized, entropy→curvature stays `[EXPLORATORY]`.

---

## Promotion Criteria Summary

| Link | Current | Requires for promotion |
|------|---------|----------------------|
| GQA → K capacity | `[PROVEN]` | — |
| Training regime → subspace allocation | `[EXPLORATORY]` | Gradient signal causal operator |
| QK alignment → selectivity → highway | `[EXPLORATORY]` | Formal derivation: alignment → expected crossing depth |
| Attention selectivity ↔ Entropy | `[PROVEN]` | — |
| Entropy → Δcurvature | `[EXPLORATORY]` | Derive: H(α), V → expected ID/curvature of output distribution |
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
