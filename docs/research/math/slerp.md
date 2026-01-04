# Spherical Linear Interpolation (SLERP)

> Geodesic interpolation on the hypersphere for diagnostics and visualization.

---

## Why This Exists in ModelCypher

SLERP provides spherical geodesic interpolation for diagnostics, visualization,
and representation analysis. It is **not** used for model merging because
interpolation discards information; ModelCypher merges with null-space addition.

**In ModelCypher**: Implemented in `vector_math.py` with explicit warnings against
using SLERP for merges.

---

## The Core Insight

Neural network weights, especially in attention layers, often lie on or near hyperspheres (normalized or approximately normalized). SLERP interpolates along the **great circle arc** rather than the chord:

- **Linear interpolation**: $(1-t) \cdot v_0 + t \cdot v_1$ (chord through sphere interior)
- **SLERP**: Follows the surface of the sphere (geodesic path)

---

## Formal Definition

### Original Formulation (Shoemake, 1985)

Given two unit vectors $v_0, v_1$ on the unit sphere and interpolation parameter $t \in [0, 1]$:

$$\text{SLERP}(v_0, v_1, t) = \frac{\sin((1-t)\theta)}{\sin\theta} v_0 + \frac{\sin(t\theta)}{\sin\theta} v_1$$

where $\theta = \arccos(v_0 \cdot v_1)$ is the angle between the vectors.

### Properties

1. **Constant angular velocity**: $t$ varies linearly → rotation angle varies linearly
2. **Geodesic path**: Shortest path on the sphere
3. **Magnitude preservation**: Output has unit norm (for unit inputs)

### Edge Cases

When $\theta \approx 0$ (vectors nearly parallel):
$$\text{SLERP}(v_0, v_1, t) \approx (1-t) v_0 + t v_1$$

When $\theta \approx \pi$ (vectors nearly opposite), ModelCypher falls back to
linear interpolation to avoid instability.

---

## Algorithm for Neural Network Weights

```python
def slerp(v0, v1, t, epsilon=None, interpolate_magnitude=True):
    """
    Spherical linear interpolation between weight vectors.

    Args:
        v0: First weight vector (will be normalized)
        v1: Second weight vector (will be normalized)
        t: Interpolation factor in [0, 1]
        epsilon: Threshold for near-parallel/antipodal detection

    Returns:
        Interpolated vector on the great circle arc
    """
    if epsilon is None:
        epsilon = dtype_epsilon(v0)

    # Normalize inputs (geodesic norms on backend)
    v0_norm = v0 / geodesic_norm(v0)
    v1_norm = v1 / geodesic_norm(v1)

    # Compute angle
    dot = clip(dot_product(v0_norm, v1_norm), -1, 1)
    theta = arccos(dot)

    # Handle near-parallel or near-antipodal case
    if theta < epsilon or theta > (pi - epsilon):
        return (1 - t) * v0 + t * v1

    # SLERP formula
    sin_theta = sin(theta)
    s0 = sin((1 - t) * theta) / sin_theta
    s1 = sin(t * theta) / sin_theta

    # Interpolate and rescale to original magnitude
    result = s0 * v0_norm + s1 * v1_norm

    # Optionally rescale to interpolated magnitude
    if interpolate_magnitude:
        mag = (1 - t) * geodesic_norm(v0) + t * geodesic_norm(v1)
        return result * mag
    return result
```

---

## When to Use SLERP (ModelCypher)

- Visualizing smooth transitions between representations
- Interpolating embeddings for diagnostic sweeps
- Animation or inspection of latent trajectories

For merges, use null-space addition instead of interpolation.

---

## Geometric Interpretation

### Great Circle Path

On a sphere, SLERP traces the **great circle** connecting two points—the analog of a straight line in curved space. This is the geodesic on the sphere.

### Connection to Quaternions

Originally developed for 3D rotation interpolation:
- Unit quaternions form a 3-sphere ($S^3$)
- SLERP on quaternions gives uniform angular velocity rotation
- This same principle applies to normalized weight vectors

### Notes on Scope

SLERP is a spherical geodesic; it does not imply that the full weight space is
globally spherical. Use it for local, diagnostic interpolation only.

---

## Code Implementation

**Primary Location**: [`src/modelcypher/core/domain/geometry/vector_math.py`](../../../../src/modelcypher/core/domain/geometry/vector_math.py)

**Key entry points**:
- `VectorMath.slerp()` / `slerp_batch()` - CPU fallback
- `BackendVectorMath.slerp()` / `slerp_batch()` / `slerp_matrix()` - GPU path
- `get_vector_math()` - factory for backend-aware math

**Usage**:
```python
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.vector_math import get_vector_math

backend = get_default_backend()
vm = get_vector_math(backend)  # Returns BackendVectorMath for GPU acceleration
result = vm.slerp(v0, v1, 0.5)
```

**Design decisions**:
1. **Explicit warning**: SLERP is for diagnostics/visualization, not merging.
2. **Geodesic norms**: Uses backend geodesic norms and cosine similarity.
3. **Numerical stability**: Epsilon derives from backend dtype via `division_epsilon`.
4. **Edge cases**: Linear fallback for near-parallel (θ≈0) or near-antipodal (θ≈π).
5. **Magnitude handling**: Optional `interpolate_magnitude` for rescaled outputs.

---

## Relationship to Other Methods

| Method | Path Type | Handles Magnitude | Handles Sign Conflicts |
|--------|-----------|-------------------|----------------------|
| **SLERP** | Spherical geodesic | Yes (interpolates) | No |
| Linear | Chord | Yes | No |
| TIES | Task vector | Trims/elects | Yes |
| DARE | Sparse linear | Yes | No |

Do not use SLERP as a merge step; use null-space addition for knowledge merging.

---

## Citations

### Foundational

1. **Shoemake, K.** (1985). "Animating Rotation with Quaternion Curves." *SIGGRAPH 1985*, Computer Graphics, 19(3), 245-254. [DOI:10.1145/325334.325242](https://doi.org/10.1145/325334.325242)
   - *Original SLERP formulation for quaternions*

2. **Davis, G.** (attributed). Symmetric weighted sum formula for geometric SLERP.
   - *Dimension-independent formulation*

### Neural Network Applications

3. **Kao, W.-C., Gur, I., Polymenakos, E., Bansal, K., & Ravi, S.** (2023). "SLERP: Spherical Linear Interpolation between Neural Networks." [arXiv:2305.17493](https://arxiv.org/abs/2305.17493)
   - *SLERP for LLM merging*

4. **ACL 2025 Industry Track** (2025). "Model Merging for Knowledge Editing." [ACL Anthology](https://aclanthology.org/)
   - *Comparative analysis of SLERP vs other methods*

5. **Nature Communications: Materials** (2025). "Fine-tuning large language models for domain adaptation." [DOI:10.1038/s41524-025-01564-y](https://doi.org/10.1038/s41524-025-01564-y)
   - *SLERP effectiveness in domain adaptation*

### 2025 Practical Applications

6. **MergeKit Documentation** (2024-2025). SLERP implementation in mergekit. [GitHub](https://github.com/cg123/mergekit)
   - *Standard tooling for SLERP merging*

7. **Hugging Face Hub** (2024-2025). SLERP-merged models collection. [HuggingFace](https://huggingface.co/models?search=slerp)
   - *Production deployment patterns*

---

## Related Concepts

- [task_singular_vectors.md](task_singular_vectors.md) - Orthogonalization before SLERP
- [procrustes_analysis.md](procrustes_analysis.md) - Alignment before interpolation
- [geodesic_distance.md](geodesic_distance.md) - Why geodesics matter on manifolds

---

*SLERP respects the spherical geometry of normalized weight spaces, providing smooth interpolation for diagnostics and visualization.*
