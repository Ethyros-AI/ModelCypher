# Geometric Safety Synthesis (Repo-Backed)

This document synthesizes existing ModelCypher docs into one coherent
through-line. It does not add new claims; it only connects what is already
implemented and documented.

Primary sources:
- `docs/GEOMETRY-GUIDE.md`
- `docs/WHY-GEOMETRY-MATTERS.md`
- `docs/research/math/procrustes_analysis.md`
- `docs/research/math/relative_representations.md`
- `docs/research/math/geodesic_distance.md`
- `docs/research/math/manifold_curvature.md`
- `docs/research/math/riemannian_density.md`
- `docs/research/merge_algorithm_synthesis.md`
- `docs/research/entropy_differential_safety.md`

## 1) The invariant skeleton

ModelCypher treats LLM representations as geometry. The key invariant is that
structure is preserved under alignment:
- Procrustes alignment gives a closed-form orthogonal transform between
  representation spaces (`docs/research/math/procrustes_analysis.md`).
- Relative representations are invariant to isometries and scale, enabling
  dimension-agnostic transfer (`docs/research/math/relative_representations.md`).
- Evidence suite measurements show aligned CKA = 1.0 on training probes by
  construction, with generalization depending on probe coverage
  (`docs/GEOMETRY-GUIDE.md`).

This is the skeleton key: after alignment, relationships are comparable even
when coordinates differ.

## 2) Manifold-first measurements

Distances and local structure are measured on the manifold, not in Euclidean
space:
- Geodesic distance on k-NN graphs is the foundational distance used throughout
  the stack (`docs/research/math/geodesic_distance.md`).
- Curvature (sectional, Ricci, scalar, Ollivier-Ricci) is computed from
  geodesic structure and reported as raw values
  (`docs/research/math/manifold_curvature.md`).
- Density and concept volumes are curvature-aware and geodesic-first
  (`docs/research/math/riemannian_density.md`).

These measurements are intended to stay on-backend and avoid heuristic
thresholds. When a threshold is required, use dtype-derived epsilons
(`docs/GEOMETRY-GUIDE.md`).

## 3) Merge and continual learning operator

Merging is geometric alignment plus null-space addition:
- Parameter averaging is not used; merging preserves target behavior and adds
  source knowledge via null-space projection
  (`docs/WHY-GEOMETRY-MATTERS.md`, `docs/research/merge_algorithm_synthesis.md`).
- Geodesic null-space filtering is the merge-safe projection used in
  transplant (`docs/research/merge_algorithm_synthesis.md`).
- Curvature signals and interference metrics are computed as raw diagnostics.

The operator is addition in unused directions, not blending.

## 4) Safety as pre-emission geometry signals

Safety is treated as a signal, not a classifier:
- Entropy differential (Delta-H) measures instability shifts during inference
  and is interpreted relative to baselines
  (`docs/research/entropy_differential_safety.md`).
- Circuit breaker logic aggregates raw signals (entropy, refusal distance,
  persona drift, oscillation patterns) without keyword matching.
- The geometry guide explicitly requires raw measurements with no qualitative
  labels (`docs/GEOMETRY-GUIDE.md`).

This positions safety as a geometric monitoring problem rather than a
post-hoc output filter.

## 5) Evidence and reproducible checks

The evidence suite produces raw, reproducible measurements for:
- Alignment generalization (CKA on train/holdout probes)
- Geodesic convergence on synthetic manifolds
- Curvature convergence on known curvature surfaces
- Causal intervention measurements

See the commands and example outputs in `docs/GEOMETRY-GUIDE.md` and the
result files under `docs/research/`.

## 6) Implementation map (code touchpoints)

Alignment and comparison:
- `src/modelcypher/core/domain/geometry/generalized_procrustes.py`
- `src/modelcypher/core/domain/geometry/relative_representation.py`

Manifold geometry:
- `src/modelcypher/core/domain/geometry/riemannian_utils.py`
- `src/modelcypher/core/domain/geometry/manifold_curvature.py`
- `src/modelcypher/core/domain/geometry/riemannian_density.py`

Merge operator:
- `src/modelcypher/core/domain/geometry/geodesic_null_space.py`
- `src/modelcypher/core/domain/geometry/transplant.py`
- `src/modelcypher/core/use_cases/merge/stages/transplant.py`

Safety signals:
- `src/modelcypher/core/use_cases/thermo_service.py`
- `src/modelcypher/core/domain/safety/circuit_breaker_integration.py`

## 7) Synthesis in one line

Alignment reveals a shared invariant structure; geodesic/curvature/density
measure that structure; null-space addition preserves it during merging;
entropy-derived signals monitor it during inference.

All of this is already in the repo. This document is the connective tissue.
