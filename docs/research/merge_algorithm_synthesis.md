# Model Merging Algorithm Synthesis (Research Notes)

**Status**: Research notes with ModelCypher implementation status
**Updated**: 2025-01-04

---

## Overview

ModelCypher merges models via geometric alignment and null-space addition.
This document summarizes external algorithms and tracks whether they are
implemented in the codebase.

---

## Implementation Map

| Topic | External Reference | Status | ModelCypher Location |
|------|---------------------|--------|----------------------|
| WUDI interference | ICML 2025 | Implemented (metrics only) | `src/modelcypher/core/domain/geometry/wudi_interference.py` |
| TSV-Merge | CVPR 2025 | Not implemented | `docs/research/math/task_singular_vectors.md` |
| Curvature signals | arXiv 2024 | Implemented (raw metrics) | `src/modelcypher/core/domain/geometry/manifold_curvature.py` |
| Fisher/CAMEx | ICLR 2025 | Not implemented | `docs/research/math/fisher_information.md` |
| Null-space filtering | MINGLE-like | Implemented | `src/modelcypher/core/domain/geometry/geodesic_null_space.py` |

---

## Implemented Topics

### WUDI Interference (ICML 2025)

ModelCypher implements a deterministic WUDI-style interference signal. It:
- Groups task vectors by weight shape
- Computes WUDI loss per group
- Reports overlap metrics (mean/max)

All outputs are raw measurements with no thresholds or labels.

### Curvature Signals

`manifold_curvature.py` computes sectional and Ollivier-Ricci curvature from
geodesic graphs. The module returns raw curvature values only. Interpretation
is left to callers using baselines.

### Geodesic Null-Space Filtering

`geodesic_null_space.py` provides the merge-safe projection used during
transplant. It computes geodesic-orthogonal directions from activation
manifolds and projects deltas into that space.

---

## Not Implemented (Research Only)

### TSV-Merge (CVPR 2025)

No TSV implementation exists in core code. Any future addition must:
- Use backend `geodesic_svd`
- Derive rank from data, not fixed heuristics
- Preserve null-space addition as the merge operator

### Fisher/CAMEx (ICLR 2025)

There is no Fisher matrix computation in core. If added, Fisher information
must be used as a diagnostic or constraint, not as a blending weight.

---

## Notes

- Parameter-space averaging and interpolation are not used for merging.
- All thresholds in core code are derived from data or machine epsilon.
