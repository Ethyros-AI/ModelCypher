# Topological Fingerprints

Topological fingerprinting uses persistent homology to characterize the "shape" of representation manifolds. This provides architecture-invariant descriptors that capture structural properties invisible to standard metrics.

Integration context: `docs/research/deep_research_integration_2026_02.md`.

## Core Concepts

### Persistent Homology [PROVEN]

Persistent homology studies topological features (connected components, loops, voids) across multiple scales. Instead of choosing a single threshold, we track when features appear ("birth") and disappear ("death") as we vary the scale parameter.

Key insight: Features that persist across many scales are stable structure; short-lived features are likely noise.

### Betti Numbers

Betti numbers count topological features by dimension:

| Betti Number | Meaning | Interpretation for Representations |
|--------------|---------|-----------------------------------|
| β₀ | Connected components | Number of distinct clusters |
| β₁ | 1-dimensional holes (loops) | Cyclic/periodic structure |
| β₂ | 2-dimensional voids | Higher-order organization |

For representation analysis, β₀ and β₁ are most informative:
- **β₀ = 1**: Single coherent representation space
- **β₀ > 1**: Fragmented representations (compare to baseline)
- **β₁ > 0**: Cyclic structure (may indicate periodicity or redundancy)

## Vietoris-Rips Filtration [PROVEN]

The Vietoris–Rips complex is built by connecting points within distance ε:

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

### Distance Matrix

Pairwise distances are computed via geodesic distances (`geodesic_distance_matrix`) with k-neighbors set to n-1 to preserve cycles in small point sets.

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

Cycles are detected using a triangle approximation:

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

Cycle candidates are capped (20) for stability on large point sets.

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

**Persistence Entropy** measures how uniformly distributed feature lifetimes are:

```
entropy = -∑ (p_i * log(p_i))  where p_i = persistence_i / total_persistence
```

## Distance Metrics

### Bottleneck Distance

The bottleneck distance is the maximum cost of optimally matching persistence diagrams:

```
d_B(D₁, D₂) = inf_γ max_{p ∈ D₁} ||p - γ(p)||_∞
```

### Wasserstein Distance

The p-Wasserstein distance sums all matching costs (p=1 in the implementation):

```
W_1(D₁, D₂) = inf_γ ∑_{p ∈ D₁} ||p - γ(p)||_1
```

Both distances use an optimal assignment (Hungarian algorithm).

## Comparison and Reporting

`TopologicalFingerprint.compare(...)` returns raw metrics plus a similarity score:

```python
comparison = TopologicalFingerprint.compare(fp_a, fp_b)
# comparison.bottleneck_distance
# comparison.wasserstein_distance
# comparison.betti_difference
# comparison.similarity_score
# comparison.betti_numbers_match
```

Report raw metrics and `betti_numbers_match`. If thresholds are required, derive them from baselines for your model family and domain.

## Deployment Modes for `beta_1` Monitoring

> **NOTE (2026-02-22):** `delta_beta_1` as a reasoning correctness predictor has been
> [DISPROVEN] on LFM2-350M. The modes below remain valid for topology *measurement*,
> but `delta_beta_1` should NOT be used as a reasoning signal until the claim is
> validated on a different model or with a different methodology (e.g., zigzag persistence).

### Mode A: Offline exact PH (research validation)

Use exact/near-exact persistent homology when the goal is mechanism testing:
- Full Vietoris-Rips or high-fidelity approximation on curated point sets
- Multi-layer sweeps for `delta_beta_1` birth/death analysis
- Method-selection studies (distance metric, filtration, threshold sensitivity)

Recommended for:
- Falsification work
- Architecture comparisons
- Baseline construction for proxy calibration

### Mode B: Online proxy `beta_1` (inference-time gating)

~~Use fast graph-based proxies when latency is constrained:~~
~~- Track trend statistics (`delta_beta_1`, moving-window slope) instead of single-shot values~~

**Status: NOT DEPLOYABLE.** `delta_beta_1` does not reliably separate correct from
incorrect outputs (F3 FAIL on all metrics). Online proxy gating based on this signal
would be no better than random for reasoning quality prediction.

Graph-based proxy measurement (cycle rank = |E| - |V| + beta_0) remains valid for
topology *monitoring*, but should not be used as an abstention/gating criterion until
the underlying signal is validated.

## `delta_beta_1` Robustness Protocol [DISPROVEN: LFM2-350M, 2026-02-22]

> **FALSIFICATION RESULT (2026-02-22):** The protocol below was executed on LFM2-350M
> (n=50, 58% accuracy on hard arithmetic). Result: 3/6 tests FAIL. The claim that
> `delta_beta_1` predicts reasoning correctness does not survive robustness controls.
> See `results/beta1_falsification/full/LFM2-350M/FALSIFICATION_REPORT.md`.

Before interpreting `delta_beta_1` as a reasoning signal, run all checks below:

1. Distance sensitivity: **FAIL** (70% agreement, threshold 80%)
   Compare Euclidean, cosine, spectral, and geodesic distances. The sign of
   `delta_beta_1` must remain stable across admissible metrics.
2. Subsampling stability: **FAIL** (57.8% stable, threshold 80%)
   Bootstrap token/point subsets and verify confidence intervals for
   `delta_beta_1` sign.
3. Null-shuffle controls: **PASS** (d=0.22, threshold 0.3)
   Shuffle token order / prompt-label pairing to establish a null distribution.
   Report effect size against this baseline.
4. Layer-window calibration: **PASS** (2/3 windows show d>0.3)
   Identify where signal is strongest (for example, mid/late windows) offline.
   Freeze this window before online deployment.
5. Held-out replication: **FAIL** (no metric shows significant separation, all CIs include 0)
   `delta_beta_1` must separate correct from incorrect outputs with d>0.3, p<0.05,
   and bootstrap CI excluding zero in at least one independent metric.
6. Numerical consistency:
   Re-run under precision and neighbor-count perturbations (`k`, threshold
   schedule) to confirm trend-level robustness. (Not tested — superseded by failures above.)

## Proxy-Validity Gate (Required Before Deployment)

> **BLOCKED (2026-02-22):** The underlying `delta_beta_1` reasoning signal is
> [DISPROVEN]. Proxy deployment gates are moot until the signal itself is validated.
> Graph proxy correlation (Spearman rho=0.56, n=800) shows the proxy *tracks* the
> offline PH metric, but tracking a non-predictive signal is not useful.

A proxy `beta_1` monitor is deployable only if all criteria pass:

1. Correlation gate:
   Proxy score has stable positive correlation with offline PH metric on a held
   out validation set.
2. Sign gate:
   Proxy and offline PH agree on `delta_beta_1` sign above chance on held-out
   prompts.
3. Reliability gate:
   Proxy score improves selective-risk behavior versus confidence-only baselines
   in the intended deployment domain.
4. Drift gate:
   Proxy behavior remains stable under minor prompt perturbations and dataset
   refresh.

If any gate fails, keep proxy use research-only and continue offline PH.

## Use in ModelCypher

Topological fingerprints are used in:
1. `GeometryMetricsService` (`mc analyze spectral-trajectory`)
2. Dimension-constraint invariance checks
3. Standalone comparison workflows

### Example Usage

```bash
mc analyze spectral-trajectory points.json
```

```python
from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

service = GeometryMetricsService()
result = service.compute_topological_fingerprint(
    points=[[0.1, 0.2], [0.3, 0.4], ...],
)

print(f"Components (β₀): {result.betti_0}")
print(f"Loops (β₁): {result.betti_1}")
print(f"Persistence entropy: {result.persistence_entropy:.3f}")
```

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Pairwise distances | O(n²d) | d = embedding dimension |
| Edge sorting | O(n² log n) | For Rips filtration |
| Union-Find | O(n² α(n)) | α = inverse Ackermann |
| Cycle detection | O(n³) | Triangle search |
| Bottleneck/Wasserstein | O(k³) | k = diagram points |

## Limitations

1. **Computational cost**: scales poorly for large point sets.
2. **1D persistence**: triangle approximation may miss higher-order cycles.
3. **Higher dimensions**: β₂+ requires full Rips complexes (exponential).

For large point clouds, consider subsampling (random or landmark-based).
