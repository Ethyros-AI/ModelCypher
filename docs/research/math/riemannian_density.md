# Riemannian Density Estimation (Concept Volumes)

> Modeling concept activations as probability mass on a curved manifold.

---

## Why This Matters for Model Merging

Many ModelCypher workflows need more than “distance between points” — they need an estimate of **how much space a concept occupies** and **where the manifold is sparse**.

**In ModelCypher**: Density/volume estimates are used for (at least) interference prediction and knowledge density profiling.

---

## Core Idea

Given a set of activation samples for a concept, ModelCypher models that concept as a distribution over representation space, but with two constraints:

1. **Distances are geodesic**, not Euclidean (k-NN graph shortest paths).
2. **Local geometry matters**, so covariance/extent should be curvature-aware.

This is an *operational* density model: it is designed to stay on-backend, be numerically stable, and support downstream comparisons (including cross-dimensional comparisons via Gram matrices where needed).

---

## ModelCypher Implementation Notes

ModelCypher represents a concept volume with:
- A centroid (mean location in activation space)
- A covariance (regularized and curvature-aware)
- A geodesic radius (extent along the manifold)
- An influence type (Gaussian / Laplacian / Student-t / uniform), derived from data characteristics

The design principle is “no hand-tuned knobs”: parameters are derived from data and numerical precision limits.

---

## Code Implementation

**Primary Location**: `src/modelcypher/core/domain/geometry/riemannian_density.py`

**Key types**:
- `ConceptVolume`
- `InfluenceType`

**CLI surface area** (density profiling):
- `mc geometry density profile`
- `mc geometry density diff`

---

## Related Concepts

- [geodesic_distance.md](geodesic_distance.md) — geodesic distances on k-NN graphs
- [frechet_mean.md](frechet_mean.md) — mean/centroid on curved spaces
- [manifold_curvature.md](manifold_curvature.md) — curvature estimation used for local corrections

