# Persistent Homology and Topological Data Analysis

> Extracting multi-scale topological features from neural representations.

---

## Why This Matters for Model Merging

Persistent homology captures **shape** in activation point clouds:
- **H0**: connected components
- **H1**: loops/cycles

These features provide architecture-invariant fingerprints for comparing
representations across models.

**In ModelCypher**: Implemented in
`src/modelcypher/core/domain/geometry/topological_fingerprint.py`.

---

## Core Concepts (Background)

- **Vietoris-Rips filtration**: connect points within increasing thresholds
- **Persistence diagram**: (birth, death) pairs for topological features
- **Betti numbers**: counts of features by dimension

---

## ModelCypher Implementation

`TopologicalFingerprint` computes persistent homology for **H0** and **H1**:
1. Compute pairwise **geodesic** distances.
2. Build a Vietoris-Rips filtration (edges sorted by distance).
3. Track H0 via Union-Find with the elder rule.
4. Track H1 via cycle/triangle detection.
5. Summarize with Betti numbers and persistence entropy.

The implementation is optimized for n < 5000 points and does not compute H2+.

---

## Comparison Metrics

`TopologicalFingerprint.compare` returns raw distances:
- **Bottleneck distance** (L-infinity matching)
- **Wasserstein distance** (sum of matched distances)
- **Betti difference** (absolute count delta)
- **Similarity score** (geometric decay; no qualitative labels)

---

## Code Implementation

**Primary Location**: `src/modelcypher/core/domain/geometry/topological_fingerprint.py`

**Key types**:
- `PersistencePoint`
- `PersistenceDiagram`
- `TopologicalFingerprint`
- `ComparisonResult`

---

## Citations

1. **Edelsbrunner, H., Letscher, D., & Zomorodian, A.** (2002). "Topological Persistence and Simplification." *Discrete & Computational Geometry*, 28(4), 511-533. [DOI:10.1007/s00454-002-2885-2](https://doi.org/10.1007/s00454-002-2885-2)
2. **Zomorodian, A., & Carlsson, G.** (2005). "Computing Persistent Homology." *Discrete & Computational Geometry*, 33(2), 249-274. [DOI:10.1007/s00454-004-1146-y](https://doi.org/10.1007/s00454-004-1146-y)
3. **Carlsson, G.** (2009). "Topology and Data." *Bull. AMS*, 46(2), 255-308. [DOI:10.1090/S0273-0979-09-01249-X](https://doi.org/10.1090/S0273-0979-09-01249-X)

---

## Related Concepts

- [geodesic_distance.md](geodesic_distance.md) - Geodesic distance inputs
- [topological_fingerprints.md](../../geometry/topological_fingerprints.md) - Implementation guide

---

*Topology captures structure that distances alone miss. Use it as an invariant fingerprint, not a heuristic score.*
