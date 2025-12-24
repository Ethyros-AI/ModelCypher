# Interference Prediction: Pre-Merge Quality Estimation

**Status**: Implemented
**Date**: 2025-12-23
**Module**: `core/domain/geometry/interference_predictor.py`
**CLI**: `mc geometry interference predict`

---

## Abstract

Interference prediction uses Riemannian density estimation to model concepts as probability distributions (not single points) and predict whether merging two models will result in constructive or destructive interference. This enables pre-merge quality assessment without expensive post-merge evaluation.

**Key Insight**: A concept in neural network latent space is not a single point, but a distribution of activations that vary with input phrasing, context, and model stochasticity. By measuring volume overlap and curvature mismatch BEFORE merging, we can predict merge quality.

---

## Theoretical Foundation

### Concepts as Distributions

Traditional merge analysis treats concepts as point centroids. This ignores:
- **Activation variance**: Different phrasings of the same concept produce different activations
- **Context sensitivity**: The same word in different contexts activates different regions
- **Model stochasticity**: Floating-point non-determinism creates activation noise

**ConceptVolume** models concepts as multivariate distributions with:
- **Centroid**: Mean position in activation space
- **Covariance**: Shape and extent of the distribution
- **Geodesic Radius**: Extent along the manifold (curvature-corrected)
- **Local Curvature**: Riemannian geometry at the centroid

### Interference Types

| Type | Description | Implication |
|------|-------------|-------------|
| **CONSTRUCTIVE** | Concepts reinforce each other | Good merge quality, enhanced capabilities |
| **NEUTRAL** | Minimal interaction | Safe to merge, no significant changes |
| **PARTIAL_DESTRUCTIVE** | Some conflict detected | Risky, apply mitigations before merge |
| **DESTRUCTIVE** | Major conflict | High risk, review before proceeding |

### Interference Mechanisms

| Mechanism | Description | Mitigation |
|-----------|-------------|------------|
| **VOLUME_OVERLAP** | Physical overlap in activation space | Reduce alpha for overlapping layers |
| **CURVATURE_MISMATCH** | Different local geometries | Use curvature-corrected alpha |
| **SUBSPACE_CONFLICT** | Misaligned principal directions | Apply Procrustes alignment |
| **BOUNDARY_COLLISION** | Edge effects at volume boundaries | Apply Gaussian smoothing |
| **SEMANTIC_COLLISION** | Same region, different meanings | Use knowledge probes post-merge |

---

## Mathematical Foundations

### Bhattacharyya Coefficient

Measures similarity between two Gaussian distributions:

```
BC = exp(-D_B)

D_B = (1/8)(μ_a - μ_b)^T Σ^{-1} (μ_a - μ_b) + (1/2)ln(det(Σ)/sqrt(det(Σ_a)det(Σ_b)))

where Σ = (Σ_a + Σ_b)/2
```

Higher BC (closer to 1) indicates more overlap.

### Geodesic Distance with Curvature Correction

For manifold with sectional curvature K:
- **K > 0** (spherical): `s = (1/sqrt(K)) * arcsin(sqrt(K) * d_euclidean)`
- **K < 0** (hyperbolic): `s = (1/sqrt(-K)) * arcsinh(sqrt(-K) * d_euclidean)`
- **K = 0** (flat): `s = d_euclidean`

### Curvature-Corrected Covariance

Standard covariance assumes flat space. For curved manifolds:

```
Cov_corrected = Cov * correction_factor

correction_factor = 1 + K*r²/6  (for K > 0)
                  = 1/(1 - K*r²/6)  (for K < 0)
```

### Safety Score Computation

Composite score from multiple factors:

```
safety = 0.4 * overlap_safety * distance_modifier
       + 0.2 * curvature_safety
       + 0.2 * alignment_safety
       + 0.2 * distance_score

where:
  overlap_safety = max(1 - overlap_score, distance_score)
  curvature_safety = 1 - curvature_mismatch
  alignment_safety = subspace_alignment
  distance_modifier = 0.5 + 0.5 * normalized_distance
```

---

## Integration with Domain Waypoints

Interference prediction integrates with the 4 validated domain geometries:

| Domain | Probes | Interference Relevance |
|--------|--------|------------------------|
| **Spatial** | 23 | Physics/3D world model conflicts |
| **Social** | 25 | Power hierarchy distortions |
| **Temporal** | 23 | Temporal ordering conflicts |
| **Moral** | 30 | Ethical valence collisions |

The CLI command `mc geometry interference predict` analyzes all domains and provides:
- Per-domain safety scores
- Domain-specific critical pairs
- Aggregate recommendation

---

## CLI Usage

```bash
# Predict interference between two models
mc geometry interference predict /path/to/source /path/to/target

# Analyze specific domains only
mc geometry interference predict /path/to/source /path/to/target --domains moral,social

# Compute volume for a single concept
mc geometry interference volume /path/to/model "justice" --samples 10

# Save detailed report
mc geometry interference predict source target -o interference_report.json
```

---

## Example Output

```
======================================================================
INTERFERENCE PREDICTION REPORT
======================================================================

Source: Qwen2.5-0.5B-Instruct-bf16
Target: Qwen2-0.5B-Instruct-4bit
Layer: last

--------------------------------------------------
VERDICT: SAFE
Overall Safety: 100.0%
--------------------------------------------------

Per-Domain Analysis:
  MORAL:
    Concepts: 30
    Safety: 100.0% (min: 100.0%)
    neutral: 30

Recommendation:
  LOW RISK: Models have compatible concept geometry.
```

---

## Interpretation Guide

### Safety Score Thresholds

| Score | Interpretation |
|-------|----------------|
| ≥ 0.8 | LOW RISK - Proceed with standard merge |
| 0.6-0.8 | ACCEPTABLE - Minor monitoring recommended |
| 0.4-0.6 | MODERATE RISK - Apply mitigations |
| < 0.4 | HIGH RISK - Review and reconsider |

### Critical Pair Analysis

When critical pairs are detected, the report identifies:
1. **Concept ID**: Which concept has interference
2. **Mechanism**: Root cause of interference
3. **Mitigation**: Recommended action

---

## Limitations

1. **Sample Requirements**: Volume estimation requires multiple activation samples per concept; single-sample analysis treats concepts as point masses
2. **Curvature Estimation**: Requires sufficient samples (n ≥ d+2) for reliable curvature estimation
3. **Dimensionality**: In very high dimensions (896+), covariance becomes singular without many samples
4. **Semantic Validation**: Interference prediction doesn't verify semantic meaning preservation

---

## Files

| File | Purpose |
|------|---------|
| `core/domain/geometry/riemannian_density.py` | ConceptVolume, RiemannianDensityEstimator |
| `core/domain/geometry/interference_predictor.py` | InterferencePredictor, GlobalInterferenceReport |
| `core/domain/geometry/manifold_curvature.py` | Curvature estimation |
| `cli/commands/geometry/interference.py` | CLI commands |
| `docs/research/interference_prediction.md` | This document |

---

## Future Directions

1. **Multi-Sample Probing**: Generate multiple prompt variations for robust volume estimation
2. **Cross-Domain Interference**: Detect conflicts between different domains (e.g., moral-social)
3. **Temporal Evolution**: Track interference changes during fine-tuning
4. **Merge Guidance Integration**: Feed interference predictions into alpha adjustment

---

## Related Work

- **Riemannian Geometry in ML**: Pennec, 2006 - Intrinsic Statistics on Riemannian Manifolds
- **Manifold Learning**: Belkin & Niyogi, 2003 - Laplacian Eigenmaps
- **ModelCypher Domain Waypoints**: Domain-aware merge guidance using validated geometries
- **Ghost Anchor Synthesis**: CABE-1 uses interference prediction for concept transplantation
