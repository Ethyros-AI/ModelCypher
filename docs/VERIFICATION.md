# ModelCypher Verification: Reproducible Checks

ModelCypher is built on the principle of **Falsifiability**. This document outlines what to measure and which fields to compare when validating geometry-based merging and safety detection. Replace any example values with outputs from your own runs; this repo does not ship canonical baselines.

Notes:
- In this repo, run commands as `poetry run mc ...`.
- Global CLI options can appear anywhere on the command line (example: `mc model profile ./model --output text`).

## 1. Merging Stability: Geometry Metrics

Command:

```bash
poetry run mc geometry interference predict <source_model> <target_model>
```

Inspect these fields in the output:
- `globalMetrics.meanOverlap`
- `globalMetrics.meanCka`
- `globalMetrics.meanCurvatureDivergence`
- `globalMetrics.meanDistance`

Compare these raw measurements across merge strategies you test.

## 2. 3D Spatial Grounding: Spatial Metrics

Command:

```bash
poetry run mc geometry spatial probe-model <model_path>
```

Inspect these fields in the output:
- `world_model_score`
- `gravity_gradient.mass_correlation`
- `volumetric_density.inverse_square_compliance`
- `axis_orthogonality` (mean in text output)

## 3. Safety: Pre-Emission Detection (Delta H)

For the full architecture and theory, see [Entropy Differential Safety](research/entropy_differential_safety.md).

Generate calibration first:

```bash
poetry run mc geometry safety calibrate-safety --model <model_path> --prompts <safe_prompts.json> --output-file <calibration.json>
```

Command:

```bash
poetry run mc geometry safety jailbreak-test --model <model_path> --prompts <prompts.json> --calibration <calibration.json>
```

Inspect these fields in the output:
- `vulnerabilitiesFound`
- `meanThresholdExceedance`
- `vulnerabilityDetails[].baselineEntropy`
- `vulnerabilityDetails[].attackEntropy`
- `vulnerabilityDetails[].deltaH`

---

## Reproducing These Results

```bash
# Verify domain geometry waypoints
poetry run mc geometry waypoint validate <source_model> <merged_model>

# Merge analysis metrics
poetry run mc geometry interference predict ./model-A ./model-B

# Spatial grounding probe
poetry run mc geometry spatial probe-model ./model

# Safety calibration
poetry run mc geometry safety calibrate-safety --model ./model --prompts ./safe_prompts.json --output-file ./calibration.json

# Safety jailbreak testing
poetry run mc geometry safety jailbreak-test --model ./model --prompts ./prompts.json --calibration ./calibration.json
```

For formal derivations and extended writeups, see [**Research Papers**](../papers/README.md).

---

## Verification Log (Template)

Use this format to record your own runs:

```
### YYYY-MM-DD: <Model> (<Hardware>)

Command: `poetry run mc geometry spatial probe-model <model_path>`

Results:
- world_model_score: <value>
- gravity_gradient.mass_correlation: <value>
- volumetric_density.inverse_square_compliance: <value>
- axis_orthogonality_mean: <value>
```
