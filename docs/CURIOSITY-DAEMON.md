# Curiosity Daemon (Geometry-First Exploration)

This note documents the math and flow behind the Curiosity Daemon. The design
is geometry-first: all decisions are derived from manifold measurements and
machine precision, not heuristics.

## Premise [CONJECTURAL]

LLM behavior is manifold geometry. Curiosity is implemented as targeted
sampling of sparse, high-structure regions while respecting null-space
capacity for new knowledge.

## Core Measurements

Inputs:
- Corpus activations: existing activation vectors that define the manifold.
- Candidate activations: proposed probe locations in the same activation space.

Derived:
- coverage_radius: k-center radius of the corpus (global coverage).
- mean_local_id: mean local intrinsic dimension (structural complexity).
- sparse_fraction: fraction of corpus with local ID above modal + sqrt(eps).
- coverage_rate: relative change in coverage_radius between iterations.

## Core Formulas [PROVEN]

Expected Free Energy (Active Inference):
- risk = (1 - capacity_fraction)^2
- ambiguity = eigenscore
- EFE = risk + ambiguity (lower is better)

Epistemic value (probe ranking):
- epistemic_value = eigenscore * capacity_fraction

Core-set coverage (k-center, geodesic):
- coreset_contribution = min_{s in corpus} d_geo(candidate, s)
- coverage_radius = max_{x in corpus} min_{s in corpus, s != x} d_geo(x, s)

Directional coverage and complexity:
- alignment = |cos(theta(candidate_tangent, sparse_direction))|
- id_factor = local_id / modal_id
- coverage_contribution = alignment * id_factor * (max_gap_angle / pi)
- density_contribution = id_factor

Composite acquisition weighting (geometry-derived):
- w = 1 / (1 + coverage_radius / mean_local_id)
- coreset = min_distance / coverage_radius (normalized k-center contribution)
- score = (1 - w) * coreset + w * (coverage + density)

Exploration temperature:
- T = mean_eigenscore / sqrt(eps)

Convergence:
- coverage_rate > 0 and coverage_rate < sqrt(eps)
- sparse_fraction < sqrt(eps)

## Dataflow and State Machine

1) SELECTING: rank candidates by epistemic value (EFE policy).
2) ACQUIRING: score candidates with composite acquisition.
3) EXECUTING: run probes and collect activations.
4) MEASURING: update coverage metrics.
5) CONSOLIDATING: optional consolidation when geometry indicates.
6) CONVERGED: coverage rate reaches precision floor.

## Module Map

- `src/modelcypher/core/domain/continual/curiosity_policy.py`
  EFE policy, epistemic value, exploration temperature.
- `src/modelcypher/core/domain/geometry/acquisition_coreset.py`
  Geodesic k-center distances and coverage radius.
- `src/modelcypher/core/domain/geometry/acquisition_manifold.py`
  Directional gaps + local intrinsic dimension.
- `src/modelcypher/core/domain/geometry/acquisition_composite.py`
  Geometry-derived weighting and composite score.
- `src/modelcypher/core/use_cases/curiosity_daemon.py`
  Async orchestration and convergence detection.

## Guarantees [PROVEN]

All thresholds and decisions are derived from:
- sqrt(machine_epsilon)
- manifold geometry (distance, curvature, intrinsic dimension)

No fixed empirical heuristics are used.
