# LoRA Isometry Ratio - Mathematical Derivation `[CONJECTURAL]`

**Status**: Draft
**Author**: Gemini
**Date**: 2026-02-05

---

## 1. Problem Statement

A **LoRA adapter** modifies a base weight matrix $W \in \mathbb{R}^{m \times n}$ by adding a low-rank update:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, and $r \ll \min(m, n)$.

**Question**: How do we measure whether this update *preserves* the geometric structure of the original weight matrix?

---

## 2. What is Geometric Preservation?

### 2.1 Isometry Definition

A transformation $f: V \to V$ is **isometric** if it preserves distances:

$$\|f(x) - f(y)\| = \|x - y\| \quad \forall x, y \in V$$

For linear transformations, this means $W^T W = I$ (orthogonal columns).

### 2.2 Relaxed Isometry for LoRA

LoRA updates are not isometric (they change the matrix). Instead, we measure **how much** the update deviates from isometry.

**Key insight**: A "good" LoRA adapter should:
1. Operate in a *subspace* of the weight matrix
2. *Not* disturb directions orthogonal to its action
3. Preserve the *spectral structure* of the original

---

## 3. Metric Candidates

### 3.1 Subspace Overlap (Already Implemented)

Measures how much the LoRA action aligns with the base weight's principal directions.

$$\text{SubspaceOverlap} = \frac{\|U_W^T \Delta W\|_F}{\|\Delta W\|_F}$$

where $U_W$ contains left singular vectors of $W$.

**Properties**:
- Range: $[0, 1]$
- $1$ = LoRA operates entirely in base subspace
- $0$ = LoRA operates in null space

### 3.2 Spectral Preservation Ratio (SPR) - NEW

Measures preservation of singular value distribution.

**Definition**:
$$\text{SPR} = \frac{\sum_{i=1}^{k} \min(\sigma_i(W'), \sigma_i(W))}{\sum_{i=1}^{k} \sigma_i(W)}$$

where $\sigma_i$ are ordered singular values and $k = \text{rank}(W)$.

**Derivation**:
- Each singular value $\sigma_i$ represents variance in direction $i$
- Preservation means $\sigma_i(W') \approx \sigma_i(W)$
- The $\min$ captures *reduction* in variance (catastrophic forgetting)
- Normalization bounds to $[0, 1]$

**Properties**:
- Range: $[0, 1]$
- $1$ = Perfect spectral preservation
- $< 1$ = Some directions lost variance
- Does NOT penalize adding new directions

### 3.3 Grassmann Distance - EXISTING

Measures angle between subspaces spanned by $W$ and $W'$.

**Definition**:
$$\theta_G = \arccos(\sigma_{\min}(U_W^T U_{W'}))$$

where $U_W, U_{W'}$ are left singular vectors.

**Properties**:
- Range: $[0, \pi/2]$
- $0$ = Identical subspaces
- Already computed in `_align_subspaces`

### 3.4 Relative Frobenius Deviation (RFD)

Simple magnitude measure.

$$\text{RFD} = \frac{\|\Delta W\|_F}{\|W\|_F}$$

**Properties**:
- Range: $[0, \infty)$
- Simple to compute
- Does NOT capture subspace alignment

---

## 4. Recommended Metric: Isometry Ratio (IR)

After analysis, we propose **combining** SPR and SubspaceOverlap:

$$\text{IR} = \text{SPR} \times \text{SubspaceOverlap}$$

**Rationale**:
- SPR ensures spectral preservation (no forgetting)
- SubspaceOverlap ensures action in meaningful directions
- Product penalizes either failure mode

**Interpretation**:
- $\text{IR} \approx 1$: Adapter preserves geometry (safe to use)
- $\text{IR} < 0.8$: Some geometric drift (monitor carefully)
- $\text{IR} < 0.5$: Significant deviation (may cause issues)

---

## 5. Hyperparameters

### 5.1 Derived from First Principles

| Parameter | Value | Derivation |
|-----------|-------|------------|
| `rank_threshold` | $\sqrt{\epsilon_{\text{machine}}} \cdot \sigma_1$ | From `numerical_stability.py` |
| `layers_to_measure` | All linear layers | Complete picture |
| `k` (SPR top-k) | Effective rank of $W$ | Data-driven |

### 5.2 NOT Tunable

No hyperparameters requiring tuning. All derived from:
- Machine precision
- Data dimensionality
- Mathematical definitions

---

## 6. Invariants

### 6.1 Scale Invariance

$$\text{IR}(\alpha W, \alpha \Delta W) = \text{IR}(W, \Delta W)$$

**Proof**: SPR and SubspaceOverlap are both scale-invariant by construction (ratios).

### 6.2 Rotation Invariance

$$\text{IR}(Q W R, Q \Delta W R) = \text{IR}(W, \Delta W)$$

for orthogonal $Q, R$.

**Proof**: SVD is rotation-invariant. $U_{QW} = QU_W$.

---

## 7. Open Questions

1. **Should we include null-space activation?**
   - Currently tracked but not in IR
   - Null-space adaptation can be *good* (new capabilities)

2. **Layer-wise vs global?**
   - Currently per-layer
   - Should we aggregate? How?

3. **Validation threshold?**
   - Proposed 0.8 threshold is heuristic
   - Need empirical validation in Phase 3

---

## 8. Implementation Notes

### Where to compute:
- `lora_diagnostic_service.py` - Add SPR computation
- `LayerSVDReport` - Add `isometry_ratio` field

### Dependencies:
- `backend.svd()` - Already used
- `svd_rank_threshold()` - Already used

---

## References

- [GeLoRA] Intrinsic dimension as rank lower bound
- [Uni-LoRA] Isometric projection definition
- [SpectralFT] Top spectral space preservation
