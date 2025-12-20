# ModelCypher Geometry Guide (for AI + humans)

This guide explains what the geometry tooling measures and how to explain the outputs in plain language.
It is written for AI agents that call the CLI/MCP tools and then summarize results for humans.

## Mental model (plain language)

- We treat weights, activations, and response trajectories as points in a very high-dimensional space.
- Geometry metrics summarize shape: curvature (flat vs sharp), distance (similar vs different),
  and direction (is training moving toward a known risk direction).
- Most outputs are already normalized or scored. Smaller usually means closer/more similar.
  Larger usually means farther/more different. When in doubt, use the provided interpretation field.

## Quick translation rules

- Prefer the `interpretation` string when provided. It already encodes thresholds.
- If a metric is missing or null, say "not enough signal" rather than guessing.
- Use 1 to 2 sentences in human summaries. Focus on stability, drift, and any warnings.

## Tool-by-tool explanations

### tc geometry training status

Key fields:
- `flatnessScore` (0 to 1): Higher is flatter and safer. >0.7 is good, 0.4 to 0.7 is moderate, <0.4 is sharp.
- `gradientSNR`: Signal-to-noise ratio in gradients. >10 strong, >1 adequate, <1 noisy.
- `circuitBreakerSeverity`: Aggregated risk score from entropy/refusal/persona/oscillation.
  >=0.75 is tripped, 0.5 to 0.75 is warning, <0.5 nominal.
- `activeLayers`: Layers with notable gradient activity (when available).

How to explain:
- "Training looks stable with [flatness assessment] curvature and [SNR assessment] gradients."
- "Circuit breaker [tripped/nominal], so [pause/continue] is recommended."

### tc geometry training history

Key fields:
- `flatnessHistory`, `snrHistory`, `parameterDivergenceHistory`: Trend lines over training steps.

How to explain:
- "Trends are [stable/improving/worsening]." Call out direction, not just magnitude.
- If empty, say metrics were not captured for this run.

### tc geometry safety circuit-breaker

Key fields:
- `severity`: 0 to 1 aggregate safety score.
- `tripped`: true when severity >= 0.75.
- `interpretation` + `recommendedAction`: Use these in the human summary.

How to explain:
- "Safety signals are [nominal/warning/tripped]; recommended action is [X]."

### tc geometry safety persona

Key fields:
- `overallDriftMagnitude`: 0 to 1 estimate of alignment drift.
- `driftAssessment`: minimal (<0.1), moderate (<0.3), significant (<0.5), critical (>=0.5).
- `driftingTraits`: Which persona traits are moving most.
- `refusalDistance`: Lower means closer to the refusal direction (riskier).

How to explain:
- "Persona drift is [assessment]; alignment is [stable/at risk]."

### tc geometry adapter sparsity (DARE)

Key fields:
- `effectiveSparsity`: Fraction of adapter deltas that are small enough to drop.
- `qualityAssessment`: excellentForMerging/good/moderate/dense/concerninglyDense.
- `recommendedDropRate`: Suggested pruning rate.

How to explain:
- "Adapter updates are [sparse/dense]; merge readiness is [qualityAssessment]."

### tc geometry adapter decomposition (DoRA)

Key fields:
- `magnitudeChangeRatio`: Average scale change in weights (0.1 means about 10% change).
- `directionalDrift`: Average angular change (0 means no rotation).
- `learningType`: magnitude_dominant, direction_dominant, balanced, minimal.

How to explain:
- "Adapter mainly [scales/rotates] weights; impact is [minimal/moderate/strong]."

### tc geometry path detect

Key fields:
- `detectedGates`: Sequence of computational gates detected in the response.
- `meanConfidence`: Confidence in the gate sequence.

How to explain:
- "The response follows a [gate sequence]; confidence is [high/medium/low]."

### tc geometry path compare

Key fields:
- `normalizedDistance` (0 to 1): Lower means more similar gate trajectories.
- `alignmentCount`: Number of aligned gate steps.

How to explain:
- "The two responses follow [similar/divergent] computational paths."

### tc geometry validate

Key fields:
- `passed`: True means the geometry invariants are behaving as expected.
- `gromovWasserstein`, `traversalCoherence`, `pathSignature`: Each includes thresholds.

How to explain:
- "Geometry validation [passed/failed]; core distance and path tests are [within/outside] thresholds."

## Glossary (short)

- Gromov-Wasserstein distance: Compares the shape of two point clouds without requiring exact alignment.
- Procrustes alignment: Best-fit rotation (and optional scaling) to align two spaces.
- DoRA decomposition: Splits weight changes into magnitude (scale) and direction (rotation).
- DARE sparsity: Estimates how sparse adapter updates are and how aggressively they can be dropped.
- Gate path: A sequence of detected computational motifs in a response.

## If you only remember one thing

Use the `interpretation` and `recommendedAction` fields when present. They are the safest summary for humans.
