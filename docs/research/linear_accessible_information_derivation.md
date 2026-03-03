# Linear-Accessible Information in Residual Networks

**Status:** Q3 frontier derivation artifact (2026-03-03)
**Purpose:** Replace invalid Shannon-MI depth-decay predictions with a protocol-compliant
observable for deterministic residual networks.

---

## 1. Claim Contract

```text
observable = f(geometry_state, architecture_state, scale_state, measurement_operator)
```

For the replacement of old P2/P4-style MI claims:

- `observable`: `CKA_linear(H_i, H_j)` (or `CKA_linear(H_0, H_l)` trajectory)
- `geometry_state`: residual decomposition `h_l = h_0 + sum_{k<l} delta_k`
- `architecture_state`: ordered operator schedule (attention/SSM/MLP, routing pattern)
- `scale_state`: depth index `l`, normalized depth `l/L`, hidden width `d`, model family
- `measurement_operator`: linear CKA from centered dot-product Gram matrices
- `falsifier`: non-negative depth-distance slope in at least one architecture family under
  pre-registered significance criteria

This document only promotes claims that satisfy this contract.

---

## 2. Why Total MI Is the Wrong Depth Observable

### 2.1 Deterministic residual chain

For fixed parameters:

```text
h_{l+1} = h_l + F_l(h_l),
h_l = G_l(h_0)
```

`G_l` is deterministic. In residual architectures, `h_0` remains a direct summand in each
state update.

### 2.2 Total Shannon MI depth decay is structurally invalid `[PROVEN]`

For deterministic continuous maps, Shannon MI is not a finite depth-varying quantity:

- Continuous activations: `I(h_0; h_l)` is infinite for deterministic continuous maps.
- Discrete finite case with injective map: `I(h_0; h_l) = H(h_0)` (constant in `l`).

So "does total MI decay with depth?" is not a valid frontier question in this setting.

References: Goldfeld et al. (2019), Cover & Thomas (deterministic channel identity).

---

## 3. Operational Quantity: Linear-Accessible Information

We need a nontrivial, finite, comparable quantity that captures what changes across layers.

Define the second-order relational overlap:

```text
L(i,j) := CKA_linear(H_i, H_j)
```

where `H_i` is the activation matrix at layer `i` over a fixed probe set.

### 3.1 Measurement operator `[PROVEN]`

Linear CKA uses centered dot-product Gram matrices:

```text
K_i = H_i H_i^T
L(i,j) = <K_i^c, K_j^c>_F / (||K_i^c||_F ||K_j^c||_F)
```

Properties:

- Basis-invariant under orthogonal coordinate changes
- Global-scale-invariant
- Cross-layer commensurable without kernel bandwidth selection

Implementation exists: `compute_linear_cka_from_activations()` in
`src/modelcypher/core/domain/geometry/cka.py`.

---

## 4. Information-Theoretic Interpretation

### 4.1 Gaussianized linear channel bridge `[PROVEN]`

For jointly Gaussian `(X,Y)`, mutual information depends on canonical correlations `rho_k`:

```text
I_G(X;Y) = -1/2 * sum_k log(1 - rho_k^2)
```

For small `rho_k`:

```text
I_G(X;Y) = 1/2 * sum_k rho_k^2 + O(sum_k rho_k^4)
```

So the sum of squared canonical correlations is the first nontrivial term of linear
Gaussian MI.

### 4.2 Why CKA is the practical proxy `[EXPLORATORY]`

Linear CKA is a normalized second-order overlap statistic. It is not equal to total MI,
but it is an operational proxy for linearly accessible relational overlap and aligns with
existing validated behavior (P1).

This is the quantity we can measure robustly across layers and families today.

---

## 5. Replacement Predictions (Protocol-Compliant)

### P2-R (replacement for invalid MI-depth claim)

**Claim:** `CKA_linear(H_0, H_l)` decreases with depth for fixed architecture family.

- State: `[EXPLORATORY]` until depth-slope is re-run with the linear CKA operator
- Scope: family-conditioned, not universal over all possible architectures
- Falsifier: non-negative Spearman slope for `l` vs `CKA_linear(H_0, H_l)` with
  pre-registered alpha in any tested family

### P4-R (replacement for invalid bottleneck-from-MI-min claim)

**Claim:** phase transitions are expressed as block structure in linear CKA, not MI minima.

- State: `[EXPLORATORY]` until block tests are re-run with the linear CKA operator
- Falsifier: within-phase and cross-phase CKA become indistinguishable under permutation test

### Total-MI statement

**Claim:** total Shannon MI depth decay in deterministic residual chains is not an admissible
experimental hypothesis.

- State: `[PROVEN]`
- Consequence: remove MI-decay-style predictions from frontier planning.

---

## 6. Architecture and Scale Terms

### 6.1 Architecture term `[EXPLORATORY]`

Slope and block structure of `CKA_linear` are expected to depend on operator schedule:

- hybrid attention/SSM alternation,
- full vs linear attention boundaries,
- residual branch strength patterns.

These terms must be declared before cross-family claims.

### 6.2 Scale term `[EXPLORATORY]`

Use at minimum:

- normalized depth `l/L`,
- width `d`,
- model family.

Do not claim universality from raw layer index alone.

---

## 7. Experiment Spec (Next Run)

### 7.1 Pre-registration Contract

```text
observable      = CKA_linear(H_i, H_j), all (i,j) pairs for L layers per model
geometry_state  = h_l = h_0 + Σ_{k<l} δ_k (residual stream; h_0 direct summand)
architecture    = {LFM2 hybrid attention/SSM, Qwen dense transformer, same two families
                   as information_bridge run}
scale_state     = normalized depth l/L, hidden_dim d, model family label
measurement_op  = compute_linear_cka_from_activations — dot-product Gram matrices,
                  centered, no bandwidth parameter, returns float in [0,1]
direction_pred  = Spearman(l, CKA_linear(H_0, H_l)) < 0 per architecture family
falsifier_P2R   = non-negative Spearman slope at p < 0.05 for any tested family
falsifier_P4R   = within_phase_CKA_mean <= cross_phase_CKA_mean for 2/3+ models
```

### 7.2 Script Change

`scripts/information_bridge_experiment.py`, lines 706–717.
Current import/call uses `compute_cka` (geodesic RBF, returns `CKAResult`, extract `.best`).
Change to:

```python
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

# in the L×L loop:
cka_matrix[i][j] = compute_linear_cka_from_activations(
    layer_activations[sorted_layers[i]],
    layer_activations[sorted_layers[j]],
    backend,
)
cka_matrix[j][i] = cka_matrix[i][j]
```

`compute_linear_cka_from_activations` returns `float` directly — no `.best` needed.
Output directory: `results/information_bridge_linear_cka/` (separate from Regime 5 results).

### 7.3 Run Commands

```bash
# LFM2-350M (~10 min)
poetry run python scripts/information_bridge_experiment.py \
    --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
    --output results/information_bridge_linear_cka/LFM2-350M/ \
    --probes 200

# LFM2-700M (~15 min)
poetry run python scripts/information_bridge_experiment.py \
    --model /Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16 \
    --output results/information_bridge_linear_cka/LFM2-700M/ \
    --probes 200

# Qwen3.5-0.8B (~30 min)
poetry run python scripts/information_bridge_experiment.py \
    --model /Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16 \
    --output results/information_bridge_linear_cka/Qwen3.5-0.8B/ \
    --probes 200
```

### 7.4 Pass/Fail Thresholds

| Prediction | Pass | Fail (→ classification) |
|------------|------|--------------------------|
| P1-R (self-check) | Spearman(`\|i-j\|`, CKA_linear) < 0, p < 0.01, all 3 models | sign flip vs geodesic P1 → geodesic CKA P1 needs qualification |
| P2-R (depth decay) | Spearman(l, CKA_linear(H_0, H_l)) < 0, p < 0.01, 2/3+ models, sign-consistent across LFM2 + Qwen | sign flip → `[MECHANISM_UNDERSPECIFIED]` (architecture term); consistent non-significant → `[EXPLORATORY]` |
| P4-R (phase blocks) | within_phase_mean > cross_phase_mean AND permutation p < 0.01, 2/3+ models | ratio ≤ 1.0 or p ≥ 0.01 → `[DISPROVEN]` |

All thresholds pre-registered here before data is examined. p = 0.01 is the same significance
floor used in the original information_bridge experiment.

### 7.5 Rules

1. Use linear CKA only as primary observable.
2. Re-run P1/P8 with linear CKA as self-consistency check before interpreting P2-R/P4-R.
3. Keep MI estimators as secondary diagnostics only unless a DPI-satisfying estimator is derived.
4. Report sign-flip failures as `[MECHANISM_UNDERSPECIFIED]` with missing architecture term named.
5. Do NOT classify consistent-non-significant as `[DISPROVEN]` — that is `[EXPLORATORY]` pending
   more probes or a scale term that predicts the weak signal.

---

## 8. Results — Linear-CKA Rerun (2026-03-03)

**Models:** LFM2-350M, LFM2-700M, Qwen3.5-0.8B  **Probes:** 200  **Operator:** `compute_linear_cka_from_activations`

### 8.1 P1-R: CKA Depth-Distance Decay

| Model | Spearman r | p-value | Pass (p<0.01) |
|-------|-----------|---------|----------------|
| LFM2-350M | -0.319 | 3.85e-4 | yes |
| LFM2-700M | -0.168 | 0.066 | **no** |
| Qwen3.5-0.8B | -0.413 | 8.21e-13 | yes |

Pre-registered threshold: all 3 models at p < 0.01. Not met (2/3). Sign is consistently
negative. Per Rule 5 (§7.5): consistent-non-significant ≠ `[DISPROVEN]`.

**Classification: `[EXPLORATORY]`.**

Secondary finding: geodesic CKA P1 was `[VALIDATED]` 3/3 under shared-sigma σ* calibration
(σ*=0.928 for 350M, σ*=1.744 for 700M). Linear CKA gives p=0.066 for 700M — not significant.
Interpretation: the geodesic shared-sigma selection amplified the distance-decay correlation
for 700M. The original geodesic P1 `[VALIDATED]` result now requires a scope qualifier:
"holds under geodesic RBF with calibrated σ* per model." The linear-CKA operator, which has
no bandwidth parameter, isolates the structural signal without sigma amplification.

Additional scope qualifier (sampling regime): this rerun does not yet apply sparse/high-dim
sampling-bias corrections for linear CKA estimators. Therefore, the 700M non-significant
result is currently confounded by two unresolved factors: (1) operator/sigma interaction and
(2) potential sparse-sampling bias. Keep P1-R as `[EXPLORATORY]` until ACT-007-style debiased
re-estimation is completed across all three models.

### 8.2 P4-R: Phase Block Structure

| Model | Within-phase | Cross-phase | Ratio | Pass (ratio>1) |
|-------|-------------|-------------|-------|-----------------|
| LFM2-350M | 0.769 | 0.759 | 1.013 | yes |
| LFM2-700M | 0.866 | 0.845 | 1.025 | yes |
| Qwen3.5-0.8B | 0.897 | 0.863 | 1.040 | yes |

Ratio > 1 for all 3 models. Pre-registered falsifier also requires permutation p < 0.01, which
is not reported by the current script. The ratio condition is met 3/3.

**Classification: `[EMPIRICAL]`** (measured, 3 models, no permutation test in reports).

Upgrade path: add permutation test to P8 evaluation in the script → `[VALIDATED]` on re-run
if p < 0.01 across 2/3+ models.

### 8.3 P2-R: Input Similarity Trajectory

Not extracted in this run. `cka_matrix.json` contains the full L×L matrix; row 0 =
CKA_linear(H_0, H_l). Requires post-processing. **Status: unmeasured.**

---

## 9. Claim-State Summary

- `[PROVEN]` Total Shannon MI depth decay is not a valid observable in deterministic residual chains.
- `[PROVEN]` Linear CKA is a commensurable, basis-invariant second-order overlap operator.
- `[VALIDATED]` CKA depth-distance decay and phase block structure (geodesic CKA, 3/3 families).
  Scope qualifier: holds under geodesic RBF with calibrated σ* per model — see §8.1.
- `[EXPLORATORY]` Linear-CKA depth-distance decay (P1-R): 2/3 families, direction consistent.
- `[EMPIRICAL]` Linear-CKA phase block structure (P4-R): 3/3 families, ratio > 1.
- `[EXPLORATORY]` P2-R input similarity trajectory: unmeasured, pending cka_matrix.json extraction.
- `[EXPLORATORY]` Quantitative architecture and scale law for CKA slope magnitude.

---

## 10. References

- Goldfeld, Z. et al. (2019). *Estimating Information Flow in Deep Neural Networks*.
- Kornblith, S. et al. (2019). *Similarity of Neural Network Representations Revisited*.
- Cover, T. & Thomas, J. *Elements of Information Theory* (deterministic channel identities).
- Hotelling, H. (1936). *Relations Between Two Sets of Variates* (canonical correlation).
