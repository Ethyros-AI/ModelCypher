# Model Merging Algorithm Synthesis: 2024-2025 Research

**Date**: 2025-12-23
**Status**: Research Complete
**Sources**: ICML 2025, CVPR 2025, ICLR 2025, arXiv preprints

---

## Executive Summary

Recent research (2024-2025) has converged on a fundamental insight: **task-specific knowledge in neural networks occupies low-dimensional linear subspaces**. This validates ModelCypher's geometric approach and provides concrete algorithms for improving merge quality.

Key findings:
1. Task vectors are ~3% rank (TSV-Merge)
2. Interference occurs when subspaces overlap (WUDI-Merging)
3. LLM manifolds have negative Ricci curvature (stratified manifolds)
4. Fisher Information defines natural geometry on parameter space (CAMEx)
5. Null-space projection eliminates interference by construction (MINGLE)

---

## Algorithm 1: WUDI-Merging (ICML 2025)

**Paper**: "Understanding and Improving Model Merging via Task Vector Subspaces"
**arXiv**: 2503.08099
**Improvement**: 10.9% over prior SOTA

### Core Insight

Task vectors (weight deltas from fine-tuning) don't point in arbitrary directions—they form **approximate linear subspaces** that are task-specific. Interference occurs when these subspaces overlap.

### Mathematical Formulation

```
τ_A = W_finetuned_A - W_base  # Task vector for task A
τ_B = W_finetuned_B - W_base  # Task vector for task B

# WUDI projects τ_B to avoid τ_A's subspace
τ_B_safe = τ_B - P_A(τ_B)

where P_A is the projection onto span(τ_A)
```

### Integration with ModelCypher

**Current**: `InterferencePredictor` measures volume overlap between concept distributions.

**Enhancement**: Add subspace overlap detection:

```python
@dataclass
class SubspaceInterference:
    overlap_coefficient: float  # cos(angle between subspaces)
    shared_dimensions: int      # dim(intersection)
    projection_residual: float  # ||τ_B - P_A(τ_B)|| / ||τ_B||

def compute_subspace_overlap(
    task_vector_a: mx.array,  # Shape: [d]
    task_vector_b: mx.array,  # Shape: [d]
) -> SubspaceInterference:
    """Measure overlap between task vector subspaces."""
    # SVD to find principal directions
    # Compute angle between dominant singular vectors
    # Return overlap coefficient
```

**File**: `src/modelcypher/core/domain/geometry/subspace_interference.py`

---

## Algorithm 2: TSV-Merge (CVPR 2025)

**Paper**: "Task Singular Vectors: Reducing Task Interference in Model Merging"
**arXiv**: 2412.00081
**Key Finding**: 3% of singular components retain 98.5% accuracy

### Core Insight

Task matrices (reshaped task vectors) are **inherently low-rank**. By extracting only the dominant singular vectors, we:
1. Reduce storage/compute
2. Eliminate noise
3. Enable selective transfer

### Mathematical Formulation

```
T = reshape(τ, [out_features, in_features])  # Task matrix
U, S, V = SVD(T)

# Keep top-k singular components (typically k << rank(T))
T_reduced = U[:, :k] @ diag(S[:k]) @ V[:k, :]

# Selective merging: only transfer directions with high singular values
```

### Integration with ModelCypher

**Current**: `unified_manifold_merger.py` uses full weight blending.

**Enhancement**: SVD-based selective merging:

```python
@dataclass
class TSVMergeConfig:
    rank_fraction: float = 0.03      # Keep top 3% of singular values
    min_singular_ratio: float = 0.01 # Drop if σ_i/σ_1 < threshold
    per_layer_rank: bool = True      # Adaptive rank per layer

def tsv_selective_merge(
    source_weights: mx.array,
    target_weights: mx.array,
    base_weights: mx.array,
    alpha: float,
    config: TSVMergeConfig,
) -> mx.array:
    """Merge using only dominant Task Singular Vectors."""
    task_source = source_weights - base_weights
    task_target = target_weights - base_weights

    # SVD decomposition
    U_s, S_s, V_s = svd(reshape(task_source, [-1, dim]))
    U_t, S_t, V_t = svd(reshape(task_target, [-1, dim]))

    # Select top-k components
    k_s = select_rank(S_s, config)
    k_t = select_rank(S_t, config)

    # Reconstruct reduced task vectors
    task_s_reduced = reconstruct(U_s, S_s, V_s, k_s)
    task_t_reduced = reconstruct(U_t, S_t, V_t, k_t)

    # Blend reduced representations
    return base_weights + alpha * task_s_reduced + (1 - alpha) * task_t_reduced
```

**File**: Extend `src/modelcypher/core/domain/merging/unified_manifold_merger.py`

---

## Algorithm 3: Stratified Manifolds & Ricci Curvature

**Paper**: "What is the geometry of neural network token spaces?"
**arXiv**: 2410.08993
**Key Finding**: LLM token spaces have uniformly negative Ricci curvature

### Core Insight

Neural network activation spaces are not flat Euclidean spaces—they are **stratified manifolds** with hyperbolic geometry. The Ricci curvature (average of sectional curvatures) is consistently negative, meaning:
- Geodesics diverge exponentially
- Euclidean distance underestimates true distance
- Volume grows faster than Euclidean prediction

### Mathematical Formulation

```
# Sectional curvature K(u,v) at point p in direction of plane spanned by u,v
# Ricci curvature Ric(v,v) = sum of sectional curvatures over orthogonal planes

Ric(v) = Σ_i K(v, e_i)  # where e_i are orthonormal basis vectors

# Negative Ricci → hyperbolic geometry → exponential divergence
d_geodesic(p, q) ≈ d_euclidean(p, q) * exp(|K| * d_euclidean / 2)
```

### Integration with ModelCypher

**Current**: `manifold_curvature.py` computes sectional curvature.

**Enhancement**: Add Ricci curvature and health metrics:

```python
@dataclass
class RicciCurvatureResult:
    mean_ricci: float              # Average Ricci curvature
    ricci_by_direction: np.ndarray # Per-direction Ricci
    hyperbolic_factor: float       # Correction for geodesic distances
    manifold_health: str           # "healthy" / "degenerate" / "collapsed"

def compute_ricci_curvature(
    activations: mx.array,  # Shape: [n_samples, d]
    n_directions: int = 10,
) -> RicciCurvatureResult:
    """Compute Ricci curvature as trace of sectional curvatures."""
    sectional = compute_sectional_curvatures(activations, n_directions)
    ricci = trace_over_planes(sectional)

    # Negative Ricci is normal for LLMs; positive may indicate collapse
    if mean(ricci) > 0:
        health = "collapsed"
    elif mean(ricci) > -0.1:
        health = "degenerate"
    else:
        health = "healthy"

    return RicciCurvatureResult(
        mean_ricci=mean(ricci),
        ricci_by_direction=ricci,
        hyperbolic_factor=exp(abs(mean(ricci))),
        manifold_health=health,
    )
```

**File**: Extend `src/modelcypher/core/domain/geometry/manifold_curvature.py`

---

## Algorithm 4: CAMEx - Curvature-Aware Expert Merging (ICLR 2025)

**Paper**: "CAMEx: Curvature-Aware Merging of Experts"
**arXiv**: 2502.18821
**Key Finding**: Fisher Information defines natural geometry

### Core Insight

Parameters are not equally important—the Fisher Information Matrix (FIM) quantifies how much the loss changes when parameters change. High Fisher = high importance = preserve during merge.

### Mathematical Formulation

```
# Fisher Information Matrix
F = E[(∇_θ log p(x|θ))(∇_θ log p(x|θ))^T]

# Fisher-weighted distance
d_Fisher(θ_1, θ_2) = (θ_1 - θ_2)^T F (θ_1 - θ_2)

# Fisher-weighted merging
θ_merged = (F_1 θ_1 + F_2 θ_2) / (F_1 + F_2)
```

### Integration with ModelCypher

**Current**: `fisher_blending.py` exists but is incomplete.

**Enhancement**: Complete Fisher-weighted merging:

```python
@dataclass
class FisherBlendingConfig:
    sample_count: int = 100        # Samples for FIM estimation
    diagonal_only: bool = True     # Use only diagonal (memory efficient)
    regularization: float = 1e-6   # Prevent division by zero
    blend_strength: float = 0.5    # How much to trust Fisher weights

def compute_fisher_weights(
    model: nn.Module,
    sample_prompts: list[str],
    tokenizer: Tokenizer,
) -> dict[str, mx.array]:
    """Estimate diagonal Fisher Information for each parameter."""
    fisher = {name: zeros_like(param) for name, param in model.named_parameters()}

    for prompt in sample_prompts:
        # Forward + backward pass
        loss = compute_loss(model, prompt, tokenizer)
        grads = compute_gradients(loss, model.parameters())

        # Accumulate squared gradients (diagonal FIM estimate)
        for name, grad in grads.items():
            fisher[name] += grad ** 2

    # Normalize
    for name in fisher:
        fisher[name] /= len(sample_prompts)

    return fisher

def fisher_weighted_merge(
    source_weights: dict[str, mx.array],
    target_weights: dict[str, mx.array],
    fisher_source: dict[str, mx.array],
    fisher_target: dict[str, mx.array],
    config: FisherBlendingConfig,
) -> dict[str, mx.array]:
    """Merge weights using Fisher Information as importance weighting."""
    merged = {}
    for name in source_weights:
        f_s = fisher_source[name] + config.regularization
        f_t = fisher_target[name] + config.regularization

        # Fisher-weighted average
        merged[name] = (f_s * source_weights[name] + f_t * target_weights[name]) / (f_s + f_t)

    return merged
```

**File**: Complete `src/modelcypher/core/domain/geometry/fisher_blending.py`

---

## Algorithm 5: Null-Space Filtering (MINGLE)

**Paper**: "MINGLE: Mixture of Null-space Gated Experts"
**arXiv**: 2509.21413
**Improvement**: 7-9% over baseline merging

### Core Insight

If we project expert updates into the **null space** of prior task representations, interference is eliminated by construction. The null space contains directions that don't affect prior task performance.

### Mathematical Formulation

```
# Prior task representation matrix (from activations)
A = stack([act_1, act_2, ..., act_n])  # Shape: [n_samples, d]

# Null space of A
N = I - A^+ @ A  # Where A^+ is pseudoinverse

# Project update to null space
Δw_safe = N @ Δw

# The safe update is orthogonal to all prior activations
```

### Integration with ModelCypher

**Current**: No null-space filtering.

**Enhancement**: Create null-space filter module:

```python
@dataclass
class NullSpaceFilterConfig:
    rank_threshold: float = 0.01   # Eigenvalues below this are "null"
    max_null_dim: int = 100        # Cap null space dimension
    activation_samples: int = 50   # Samples for activation matrix

@dataclass
class NullSpaceFilterResult:
    filtered_delta: mx.array       # Δw projected to null space
    null_space_dim: int            # Dimension of null space
    projection_loss: float         # ||Δw - Δw_safe|| / ||Δw||
    preserved_fraction: float      # ||Δw_safe|| / ||Δw||

def compute_null_space_projection(
    activation_matrix: mx.array,  # Shape: [n_samples, d]
    config: NullSpaceFilterConfig,
) -> mx.array:
    """Compute projection matrix onto null space of activations."""
    # SVD to find null space
    U, S, Vh = svd(activation_matrix)

    # Null space = span of right singular vectors with small singular values
    null_mask = S < config.rank_threshold * S[0]
    null_vectors = Vh[null_mask]

    # Projection matrix
    P_null = null_vectors.T @ null_vectors
    return P_null

def filter_to_null_space(
    weight_delta: mx.array,
    prior_activations: mx.array,
    config: NullSpaceFilterConfig,
) -> NullSpaceFilterResult:
    """Project weight delta to null space of prior activations."""
    P_null = compute_null_space_projection(prior_activations, config)

    # Project delta
    delta_flat = weight_delta.flatten()
    delta_safe = P_null @ delta_flat

    return NullSpaceFilterResult(
        filtered_delta=delta_safe.reshape(weight_delta.shape),
        null_space_dim=int(trace(P_null)),
        projection_loss=norm(delta_flat - delta_safe) / norm(delta_flat),
        preserved_fraction=norm(delta_safe) / norm(delta_flat),
    )
```

**File**: `src/modelcypher/core/domain/geometry/null_space_filter.py`

---

## Unified Theory: The Geometry of Knowledge Transfer

These five algorithms reveal a coherent picture:

1. **Knowledge is low-rank**: Task-specific information occupies ~3% of parameter space (TSV-Merge)

2. **Interference is overlap**: When two tasks use the same subspace, merging causes interference (WUDI)

3. **Geometry is hyperbolic**: LLM manifolds have negative curvature, so Euclidean intuitions fail (Stratified Manifolds)

4. **Importance is curvature**: Fisher Information measures parameter importance via loss landscape curvature (CAMEx)

5. **Orthogonality is safety**: Null-space projection guarantees no interference (MINGLE)

### Synthesis: ModelCypher's Unified Approach

```
InterferencePrediction (existing)
      │
      ▼
┌─────────────────────────────────────────┐
│  Pre-Merge Analysis                     │
│  - Subspace overlap (WUDI)              │
│  - Ricci curvature health check         │
│  - ConceptVolume interference score     │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Weight Selection                       │
│  - TSV extraction (keep dominant SVs)   │
│  - Fisher importance weighting          │
│  - Per-dimension correlation            │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Safe Blending                          │
│  - Null-space filtering                 │
│  - Curvature-corrected alpha            │
│  - Domain signal adjustment             │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Post-Merge Validation                  │
│  - Knowledge retention probes           │
│  - CKA similarity metrics               │
│  - Domain waypoint verification         │
└─────────────────────────────────────────┘
```

---

## Implementation Priority

| Algorithm | Effort | Impact | Priority |
|-----------|--------|--------|----------|
| Null-Space Filtering | Medium | High | 1 |
| TSV Selective Merge | Medium | High | 2 |
| WUDI Subspace Detection | Low | Medium | 3 |
| Ricci Curvature | Low | Medium | 4 |
| Fisher Completion | Medium | Medium | 5 |

### Recommended Next Steps

1. **Implement Null-Space Filter** (`null_space_filter.py`)
   - Highest impact: eliminates interference by construction
   - Mathematically elegant, straightforward to implement

2. **Add TSV extraction to merger**
   - Extends existing SVD infrastructure
   - Validates low-rank hypothesis empirically

3. **Integrate subspace overlap into InterferencePredictor**
   - Enhances existing interference analysis
   - Provides pre-merge guidance

---

## References

1. WUDI-Merging: arXiv:2503.08099 (ICML 2025)
2. TSV-Merge: arXiv:2412.00081 (CVPR 2025)
3. Stratified Manifolds: arXiv:2410.08993
4. CAMEx: arXiv:2502.18821 (ICLR 2025)
5. MINGLE: arXiv:2509.21413

---

## Related ModelCypher Modules

| Module | Relevance |
|--------|-----------|
| `interference_predictor.py` | Pre-merge interference analysis |
| `riemannian_density.py` | ConceptVolume distribution modeling |
| `manifold_curvature.py` | Sectional curvature → extend to Ricci |
| `fisher_blending.py` | Stub → complete with CAMEx insights |
| `unified_manifold_merger.py` | Integration point for all algorithms |
| `domain_geometry_waypoints.py` | Validation anchors (439 probes) |
