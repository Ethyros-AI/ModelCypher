# ModelCypher Geometry Guide (for AI + humans)

This guide explains what the geometry tooling measures and how to explain the outputs in plain language.
It is written for AI agents that call the CLI/MCP tools and then summarize results for humans.

Related docs:
- `docs/MATH-PRIMER.md` - Intuition for the underlying geometry (distance/angle/alignment).
- `docs/AI-ASSISTANT-GUIDE.md` - Safe summarization patterns across CLI + MCP.

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

## What these metrics can and cannot tell you

They *can* help you:
- detect that something changed (drift, instability, unusual updates),
- localize where it changed (layers/components, when captured),
- decide when to pause and investigate (circuit breaker style signals).

They *cannot*:
- prove a model is “safe”,
- replace eval suites, policy review, or red teaming,
- guarantee causality (“metric went up, therefore X happened”).

## Tool-by-tool explanations

### mc geometry training status

Key fields:
- `flatnessScore` (0 to 1): Higher is flatter and safer. >0.7 is good, 0.4 to 0.7 is moderate, <0.4 is sharp.
- `gradientSNR`: Signal-to-noise ratio in gradients. >10 strong, >1 adequate, <1 noisy.
- `circuitBreakerSeverity`: Aggregated risk score from entropy/refusal/persona/oscillation.
  >=0.75 is tripped, 0.5 to 0.75 is warning, <0.5 nominal.
- `activeLayers`: Layers with notable gradient activity (when available).

How to explain:
- "Training looks stable with [flatness assessment] curvature and [SNR assessment] gradients."
- "Circuit breaker [tripped/nominal], so [pause/continue] is recommended."

### mc geometry training history

Key fields:
- `flatnessHistory`, `snrHistory`, `parameterDivergenceHistory`: Trend lines over training steps.

How to explain:
- "Trends are [stable/improving/worsening]." Call out direction, not just magnitude.
- If empty, say metrics were not captured for this run.

### mc geometry training levels

Purpose:
- Lists available instrumentation levels and which metrics each level collects.

How to explain:
- "Higher levels collect more metrics (more overhead) and enable deeper geometry analysis."
  If a metric you expect is missing, confirm the job captured it at the chosen level.

### mc geometry safety circuit-breaker

Key fields:
- `severity`: 0 to 1 aggregate safety score.
- `tripped`: true when severity >= 0.75.
- `interpretation` + `recommendedAction`: Use these in the human summary.

How to explain:
- "Safety signals are [nominal/warning/tripped]; recommended action is [X]."

### mc geometry safety persona

Key fields:
- `overallDriftMagnitude`: 0 to 1 estimate of alignment drift.
- `driftAssessment`: minimal (<0.1), moderate (<0.3), significant (<0.5), critical (>=0.5).
- `driftingTraits`: Which persona traits are moving most.
- `refusalDistance`: Lower means closer to the refusal direction (riskier).

How to explain:
- "Persona drift is [assessment]; alignment is [stable/at risk]."

### mc geometry adapter sparsity (DARE)

Key fields:
- `effectiveSparsity`: Fraction of adapter deltas that are small enough to drop.
- `qualityAssessment`: excellentForMerging/good/moderate/dense/concerninglyDense.
- `recommendedDropRate`: Suggested pruning rate.

How to explain:
- "Adapter updates are [sparse/dense]; merge readiness is [qualityAssessment]."

### mc geometry adapter decomposition (DoRA)

Key fields:
- `magnitudeChangeRatio`: Average scale change in weights (0.1 means about 10% change).
- `directionalDrift`: Average angular change (0 means no rotation).
- `learningType`: magnitude_dominant, direction_dominant, balanced, minimal.

How to explain:
- "Adapter mainly [scales/rotates] weights; impact is [minimal/moderate/strong]."

### mc geometry path detect

Key fields:
- `detectedGates`: Sequence of computational gates detected in the response.
- `meanConfidence`: Confidence in the gate sequence.

How to explain:
- "The response follows a [gate sequence]; confidence is [high/medium/low]."

### mc geometry path compare

Key fields:
- `normalizedDistance` (0 to 1): Lower means more similar gate trajectories.
- `alignmentCount`: Number of aligned gate steps.

How to explain:
- "The two responses follow [similar/divergent] computational paths."

### mc geometry validate

Key fields:
- `passed`: True means the geometry invariants are behaving as expected.
- `gromovWasserstein`, `traversalCoherence`, `pathSignature`: Each includes thresholds.

How to explain:
- "Geometry validation [passed/failed]; core distance and path tests are [within/outside] thresholds."

## Example translations (JSON -> human)

### Example: mc geometry training status

**JSON output:**
```json
{
  "jobId": "job-123",
  "step": 420,
  "flatnessScore": 0.78,
  "flatnessAssessment": "Flat (good)",
  "gradientSNR": 12.4,
  "snrAssessment": "Strong signal",
  "circuitBreakerSeverity": 0.22,
  "circuitBreakerTripped": false,
  "activeLayers": ["layer.4", "layer.5"]
}
```

**Human summary:**
Training looks stable with flat curvature and strong gradient signal; no safety trip detected.

### Example: mc geometry safety circuit-breaker

**JSON output:**
```json
{
  "tripped": true,
  "severity": 0.82,
  "state": "tripped",
  "interpretation": "ALIGNMENT DRIFT: Persona vectors shifting from baseline",
  "recommendedAction": "Stop and request human review"
}
```

**Human summary:**
Circuit breaker tripped due to alignment drift; stop generation and request human review.

### Example: mc geometry adapter sparsity (DARE)

**JSON output:**
```json
{
  "checkpointPath": "./adapters/adapter.npz",
  "baseModelPath": "./models/base",
  "effectiveSparsity": 0.91,
  "qualityAssessment": "good",
  "interpretation": "Effective sparsity 91.00% (good). Recommended drop rate 0.90.",
  "nextActions": [
    "mc geometry adapter decomposition --checkpoint './adapters/adapter.npz'",
    "mc checkpoint export --path './adapters/adapter.npz'"
  ]
}
```

**Human summary:**
Adapter deltas are sparse and merge-ready; a ~0.90 drop rate is reasonable.

### Example: mc geometry adapter decomposition (DoRA)

**JSON output:**
```json
{
  "checkpointPath": "./adapters/adapter.npz",
  "baseModelPath": "./models/base",
  "magnitudeChangeRatio": 0.18,
  "directionalDrift": 0.04,
  "learningType": "magnitude_dominant",
  "interpretation": "Adapter primarily amplifies existing features (magnitude +18%)",
  "nextActions": [
    "mc geometry adapter sparsity --checkpoint './adapters/adapter.npz'",
    "mc checkpoint export --path './adapters/adapter.npz'"
  ]
}
```

**Human summary:**
Adapter changes mostly scale existing features with minimal rotation.

## Glossary (short)

- Gromov-Wasserstein distance: Compares the shape of two point clouds without requiring exact alignment.
- Procrustes alignment: Best-fit rotation (and optional scaling) to align two spaces.
- DoRA decomposition: Splits weight changes into magnitude (scale) and direction (rotation).
- DARE sparsity: Estimates how sparse adapter updates are and how aggressively they can be dropped.
- Gate path: A sequence of detected computational motifs in a response.

## If you only remember one thing

Use the `interpretation` and `recommendedAction` fields when present. They are the safest summary for humans.
