# Topological Fingerprints

Topological fingerprinting uses persistent homology to characterize the "shape" of representation manifolds. This provides architecture-invariant descriptors that capture structural properties invisible to standard metrics.

## Core Concepts

### Persistent Homology

Persistent homology studies topological features (connected components, loops, voids) across multiple scales. Instead of choosing a single threshold, we track when features appear ("birth") and disappear ("death") as we vary the scale parameter.

Key insight: Features that persist across many scales are "real" structure; short-lived features are noise.

### Betti Numbers

Betti numbers count topological features by dimension:

| Betti Number | Meaning | Interpretation for Representations |
|--------------|---------|-----------------------------------|
| β₀ | Connected components | Number of distinct clusters |
| β₁ | 1-dimensional holes (loops) | Cyclic/periodic structure |
| β₂ | 2-dimensional voids | Higher-order organization |

For representation analysis, β₀ and β₁ are most informative:
- **β₀ = 1**: Single coherent representation space
- **β₀ > 1**: Fragmented representations (potential problem)
- **β₁ > 0**: Cyclic structure (may indicate periodicity or redundancy)

## Vietoris-Rips Filtration

The Vietoris-Rips complex is built by connecting points within distance ε:

```
At scale ε:
- Vertices: All points
- Edges: Pairs with distance ≤ ε
- Triangles: Triples where all pairs have distance ≤ ε
- Higher simplices: Similarly defined
```

As ε increases from 0 to max(distances):
1. Points start isolated (β₀ = n)
2. Nearby points connect, merging components (β₀ decreases)
3. Loops may form when edges complete cycles (β₁ increases)
4. Triangles fill loops (β₁ decreases)

### Persistence Diagram

A persistence diagram records (birth, death) pairs for each topological feature:

```
    death
      ↑
      |     •
      |   •   •
      |  •  •
      | • • •
      +--------→ birth
```

Points far from the diagonal (high persistence) represent significant structure.

## Implementation Details

### 0-Dimensional Persistence (Components)

Uses Union-Find with the "elder rule":
1. Sort edges by length
2. Process edges in order
3. When edge connects two components:
   - Older component survives (lower birth time)
   - Younger component "dies" at current edge length
4. Record (birth=0, death=edge_length) for dying component

```python
def union_with_persistence(i, j, current_distance):
    root_i, root_j = find(i), find(j)
    if root_i == root_j:
        return None  # Already connected

    # Elder rule: older component survives
    if birth[root_i] < birth[root_j]:
        survivor, dying = root_i, root_j
    else:
        survivor, dying = root_j, root_i

    parent[dying] = survivor
    return PersistencePoint(birth=birth[dying], death=current_distance, dimension=0)
```

### 1-Dimensional Persistence (Loops)

Detecting cycles requires tracking when edges form loops without merging components:

```python
for edge (i, j, distance) in sorted_edges:
    if find(i) == find(j):
        # Edge creates a cycle (points already connected via other path)
        cycle_birth = distance

        # Cycle dies when a triangle fills it
        cycle_death = find_triangle_filling(i, j, distances)

        if cycle_death > cycle_birth:
            record(birth=cycle_birth, death=cycle_death, dimension=1)
    else:
        union(i, j)
```

Triangle filling: A cycle through edge (i,j) dies when vertex k exists such that all three edges (i,j), (i,k), (j,k) are present.

### Persistence Summary Statistics

```python
@dataclass
class TopologySummary:
    component_count: int     # β₀ at final scale
    cycle_count: int         # β₁ at final scale
    average_persistence: float  # Mean lifetime of features
    max_persistence: float      # Longest-lived feature
    persistence_entropy: float  # Distribution uniformity
```

**Persistence Entropy**: Measures how uniformly distributed feature lifetimes are.
```python
entropy = -∑ (p_i * log(p_i))  where p_i = persistence_i / total_persistence
```
- High entropy: Many features of similar importance
- Low entropy: Dominated by few features

## Distance Metrics

### Bottleneck Distance

The bottleneck distance is the maximum cost of optimally matching persistence diagrams:

```
d_B(D₁, D₂) = inf_γ max_{p ∈ D₁} ||p - γ(p)||_∞
```

Where γ is a bijection (matching) between diagrams, and unmatched points can be matched to the diagonal.

Properties:
- Captures worst-case structural difference
- Robust to small perturbations
- O(n³) for optimal computation (we use greedy approximation)

### Wasserstein Distance

The p-Wasserstein distance sums all matching costs:

```
W_p(D₁, D₂) = [inf_γ ∑_{p ∈ D₁} ||p - γ(p)||^p]^(1/p)
```

For p=1 (used in implementation):
```
W_1(D₁, D₂) = inf_γ ∑_{p ∈ D₁} ||p - γ(p)||_1
```

Properties:
- Captures total structural difference
- More sensitive to multiple small differences than bottleneck
- Better for continuous optimization

### Greedy Matching Approximation

Full optimal transport is O(n³). We use greedy matching:

```python
def greedy_bottleneck(diagram_a, diagram_b):
    max_cost = 0
    used_b = set()

    for a in diagram_a.points:
        best_match = None
        best_cost = infinity

        for j, b in enumerate(diagram_b.points):
            if j in used_b:
                continue
            cost = max(|a.birth - b.birth|, |a.death - b.death|)
            if cost < best_cost:
                best_cost = cost
                best_match = j

        diagonal_cost = (a.death - a.birth) / 2
        if best_match and best_cost < diagonal_cost:
            used_b.add(best_match)
            max_cost = max(max_cost, best_cost)
        else:
            max_cost = max(max_cost, diagonal_cost)

    return max_cost
```

## Comparison and Compatibility

### Similarity Score

Combined metric using bottleneck, Wasserstein, and Betti differences:

```python
scale = max(fp_a.max_persistence, fp_b.max_persistence)
score = exp(-bottleneck/scale) * exp(-wasserstein/scale) * (1 / (1 + betti_diff))
```

### Compatibility Assessment

```python
is_compatible = (
    betti_difference <= 2 and
    bottleneck < scale * 0.5
)
```

Interpretation thresholds:
| Score Range | Interpretation |
|-------------|----------------|
| > 0.8, β_diff=0 | Identical topological structure |
| > 0.6 | Similar topological structure |
| > 0.3 | Moderate topological similarity |
| ≤ 0.3 | Different topological structure |

## Use in ModelCypher

Topological fingerprints are used for:

1. **Model comparison**: Architecture-invariant similarity
2. **Merge safety**: Detect incompatible representation structures
3. **Training monitoring**: Track manifold collapse or fragmentation
4. **Anomaly detection**: Identify unusual representation topology

### Example Usage

```python
from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

service = GeometryMetricsService()
result = service.compute_topological_fingerprint(
    points=[[0.1, 0.2], [0.3, 0.4], ...],
    max_dimension=1,
    num_steps=50,
)

print(f"Components (β₀): {result.betti_0}")
print(f"Loops (β₁): {result.betti_1}")
print(f"Persistence entropy: {result.persistence_entropy:.3f}")
print(f"Interpretation: {result.interpretation}")
```

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Pairwise distances | O(n²d) | d = embedding dimension |
| Edge sorting | O(n² log n) | For Rips filtration |
| Union-Find | O(n² α(n)) | α = inverse Ackermann |
| Cycle detection | O(n³) | Triangle search |
| Bottleneck (greedy) | O(k²) | k = diagram points |

For n=1000 points, expect ~seconds for full computation.

## Limitations

1. **Computational cost**: Scales poorly beyond ~5000 points
2. **Approximations**: Greedy matching is not optimal
3. **1D persistence**: Simplified cycle detection may miss some features
4. **Higher dimensions**: β₂+ requires full Rips complex (exponential)

For large point clouds, consider:
- Subsampling (random or landmark-based)
- Witness complexes
- Approximate persistent homology

## References

1. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.

2. Carlsson, G. (2009). *Topology and Data*. Bulletin of the American Mathematical Society.

3. Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). *Stability of Persistence Diagrams*. Discrete & Computational Geometry.

4. Zomorodian, A., & Carlsson, G. (2005). *Computing Persistent Homology*. Discrete & Computational Geometry.

5. Chazal, F., et al. (2014). *Stochastic Convergence of Persistence Landscapes and Silhouettes*. SoCG.
