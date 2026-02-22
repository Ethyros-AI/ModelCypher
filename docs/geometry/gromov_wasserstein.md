# Gromov-Wasserstein Distance

Gromov-Wasserstein (GW) distance measures structural similarity between metric spaces without requiring point-to-point correspondence. This makes it well-suited for comparing representation manifolds across models with different architectures.

## Mathematical Foundation [PROVEN]

Given two metric spaces (X, dX) and (Y, dY) with probability measures μ and ν, the GW objective minimizes:

```
GW(μ, ν) = min_γ ∑_{i,j,k,l} L(dX(xi, xk), dY(yj, yl)) · γij · γkl
```

where γ is a coupling matrix with marginals μ and ν and L is typically squared loss.

For squared loss L(a, b) = (a-b)², the objective decomposes into terms that allow efficient tensor products:

```
L(a,b) = a² + b² - 2ab
```

This decomposition enables O(n²m + nm²) computation per outer iteration instead of O(n²m²).

## Algorithm Used in ModelCypher [PROVEN]

ModelCypher implements GW using a **Frank–Wolfe (conditional gradient)** solver, following Peyré, Cuturi, and Solomon (2016).

High-level steps:
1. Compute the GW gradient using the loss decomposition.
2. Solve a linear OT subproblem to obtain a descent direction.
3. Perform an analytic line search to choose the Frank–Wolfe step size.
4. Update the coupling with the convex combination.

**Important:** The update `T ← (1-α)T + αG` is the Frank–Wolfe step, not heuristic blending.

### Linear OT Subproblem

The linear OT subproblem is solved with Sinkhorn iterations. The Sinkhorn epsilon is **derived from the cost matrix scale** (median cost × √machine_epsilon). There are **no user-tuned hyperparameters**; convergence thresholds are derived from dtype precision.

### Convergence Criteria

The solver checks change in objective value:
- Absolute change below √machine_epsilon, or
- Relative change below √machine_epsilon

after a minimum number of iterations. The maximum outer iterations are capped.

### Exact Permutation for Small n

When n = m and n ≤ 8, the solver exhaustively searches permutations to return an exact solution.

## Normalized Distance

The implementation exposes a normalized distance:

```
normalized_distance = 1 - exp(-distance)
```

This is used for reporting only; the core distance remains the raw GW value.

## CLI Usage

```bash
mc analyze geodesic-compare source_points.json target_points.json
```

Output fields:
- `distance`
- `normalizedDistance`
- `aligned`
- `converged`
- `iterations`
- `couplingShape`

Example output shape (values are illustrative):

```json
{
  "_schema": "mc.geometry.gromov_wasserstein.v1",
  "distance": 0.0,
  "normalizedDistance": 0.0,
  "aligned": true,
  "converged": true,
  "iterations": 12,
  "couplingShape": [128, 128]
}
```

## Use in ModelCypher

GW distance is used in:
- `GeometryMetricsService` (`mc analyze geodesic-compare`)
- Geometry validation fixtures (identity/permutation checks)
- Cross-dimensional projection utilities
- Low-rank GW experiments

## References

1. Peyré, G., Cuturi, M., & Solomon, J. (2016). *Gromov-Wasserstein Averaging of Kernel and Distance Matrices*. ICML. [PMLR](https://proceedings.mlr.press/v48/peyre16.html)
2. [Peyré, G., & Cuturi, M. (2019)](../references/arxiv/Peyre_2018_Computational_Optimal_Transport.pdf). *Computational Optimal Transport*. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
3. Mémoli, F. (2011). *Gromov-Wasserstein distances and the metric approach to object matching*. Foundations of Computational Mathematics, 11(4), 417-487. [DOI:10.1007/s10208-011-9093-5](https://doi.org/10.1007/s10208-011-9093-5)
