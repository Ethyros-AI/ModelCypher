# Gromov-Wasserstein Distance

Gromov-Wasserstein (GW) distance measures structural similarity between metric spaces without requiring point-to-point correspondence. This makes it ideal for comparing representation manifolds across models with different architectures.

## Mathematical Foundation

### Problem Formulation

Given two metric spaces (X, dX) and (Y, dY) with probability measures μ and ν, the Gromov-Wasserstein distance finds an optimal "soft matching" (coupling) between points that preserves pairwise distance relationships.

The GW objective minimizes:

```
GW(μ, ν) = min_γ ∑_{i,j,k,l} L(dX(xi, xk), dY(yj, yl)) · γij · γkl
```

Where:
- γ is a coupling matrix with marginals μ and ν
- L is a loss function (typically squared: L(a,b) = (a-b)²)
- dX(xi, xk) is the distance between points i and k in source space
- dY(yj, yl) is the distance between points j and l in target space

### Interpretation

The GW distance quantifies how much the internal geometry of one space must be "distorted" to match another:
- **GW ≈ 0**: Spaces are nearly isomorphic (identical structure)
- **GW small**: Similar geometry with minor distortions
- **GW large**: Fundamentally different structural organization

## Entropic Regularization

Direct GW optimization is computationally expensive (O(n⁴) per iteration). We use entropic regularization to make the problem tractable:

```
GW_ε(μ, ν) = min_γ GW(γ) + ε · H(γ)
```

Where H(γ) is the entropy of the coupling matrix.

### Epsilon Parameter Effects

| Epsilon | Effect | Use Case |
|---------|--------|----------|
| ε < 0.01 | Sharp coupling, slow convergence | Final refinement |
| ε ≈ 0.05 | Balanced (default) | General use |
| ε > 0.1 | Diffuse coupling, fast convergence | Quick screening |

The implementation uses epsilon annealing: start with larger ε for fast initial progress, then reduce for precision.

## Sinkhorn Algorithm

The entropic GW problem is solved via alternating optimization:

1. **Outer loop**: Update the cost matrix based on current coupling
2. **Inner loop**: Sinkhorn iterations to project onto transport polytope

### Sinkhorn Step

Given cost matrix C and regularization ε:

```python
K[i,j] = exp(-C[i,j] / ε)    # Gibbs kernel
u, v = 1, 1                   # Dual variables

for iteration in range(max_iter):
    u = μ / (K @ v)           # Row scaling
    v = ν / (K.T @ u)         # Column scaling

coupling = diag(u) @ K @ diag(v)
```

Key properties:
- Convergence is exponential in number of iterations
- Numerical stability requires log-domain computation for small ε
- Each iteration is O(nm) for n×m coupling

## Coupling Matrix Interpretation

The output coupling γ shows how probability mass flows between spaces:

```
γ[i,j] = "strength of correspondence between source point i and target point j"
```

Analysis patterns:
- **Diagonal dominance**: Direct 1-to-1 correspondence
- **Block structure**: Clustered correspondence
- **Diffuse rows/columns**: Ambiguous matching (structural mismatch)

## Implementation Details

### Cost Matrix Computation

```python
cost[i,j] = ∑_{i',j'} L(dX[i,i'], dY[j,j']) · coupling[i',j']
```

This O(n²m²) computation is the bottleneck. For large point clouds, sampling or approximation is necessary.

### Convergence Criteria

The solver terminates when either:
1. **Coupling change** < threshold: max|γ_new - γ_old| < 1e-5
2. **Objective change** < threshold: |GW_new - GW_old| / GW_old < 1e-5
3. **Maximum iterations** reached

### Normalized Distance

Raw GW values are scale-dependent. The normalized distance provides a [0,1] interpretable measure:

```python
normalized = 1 - exp(-distance)
```

- 0.0: Identical structures
- 0.5: Moderate structural difference
- 1.0: Completely different

### Compatibility Score

The inverse exponential provides a similarity metric:

```python
compatibility = exp(-distance)
```

This is useful for merge safety assessment: higher scores indicate safer merging.

## Use in ModelCypher

GW distance is used for:
1. **Pre-merge compatibility**: Assess if models can be safely merged
2. **Layer correspondence**: Find structurally similar layers across architectures
3. **Training monitoring**: Detect geometric drift during fine-tuning

### Example Usage

```python
from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

service = GeometryMetricsService()
result = service.compute_gromov_wasserstein(
    source_points=[[0.1, 0.2], [0.3, 0.4], ...],
    target_points=[[0.15, 0.25], [0.35, 0.45], ...],
    epsilon=0.05,
    max_iterations=50,
)

print(f"Distance: {result.distance:.4f}")
print(f"Compatibility: {result.compatibility_score:.2%}")
print(f"Interpretation: {result.interpretation}")
```

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Pairwise distances | O(n²d) | d = embedding dimension |
| Cost matrix update | O(n²m²) | Bottleneck operation |
| Sinkhorn iteration | O(nm) | Per inner iteration |
| Total per outer iteration | O(n²m² + nm·I) | I = inner iterations |

For n=m=1000, expect ~seconds per outer iteration on CPU.

## References

1. Peyré, G., & Cuturi, M. (2019). *Computational Optimal Transport*. Foundations and Trends in Machine Learning.

2. Mémoli, F. (2011). *Gromov-Wasserstein distances and the metric approach to object matching*. Foundations of Computational Mathematics.

3. Solomon, J., et al. (2016). *Entropic metric alignment for correspondence problems*. ACM Transactions on Graphics.

4. Titouan, V., et al. (2019). *Optimal Transport for structured data with application on graphs*. ICML.
