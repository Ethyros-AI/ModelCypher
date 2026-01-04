# Merge Analysis: Interference Geometry (Pre-Merge)

**Status**: Implemented
**Date**: 2025-12-23
**Module**: `src/modelcypher/core/domain/geometry/interference_predictor.py`
**CLI**: `mc geometry interference predict`

---

## Abstract

Merge analysis models concepts as Riemannian distributions and reports raw geometric measurements
that summarize how two models align before a merge. The CLI extracts domain probe activations,
computes per-domain metrics, and emits overlap, curvature divergence, distance, and CKA alignment
signals without prescribing thresholds or verdicts.

---

## Theoretical Foundation

### Concepts as Distributions

Concepts are treated as probability distributions over the representation manifold (ConceptVolume):
- **Centroid**: Frechet mean on the geodesic graph
- **Covariance**: Tangent-space covariance with curvature correction
- **Geodesic radius**: 95th percentile geodesic distance from centroid
- **Local curvature**: Sectional curvature at the centroid

### Relation Computation

When raw activations are available, relations are computed with CKA (dimension-agnostic).
Otherwise, the estimator falls back to geodesic comparisons using the k-NN geodesic graph.

---

## Mathematical Foundations

### Bhattacharyya Coefficient

Used for Gaussian overlap when geodesic fallback is required:

```
BC = exp(-D_B)

D_B = (1/8)(mu_a - mu_b)^T Sigma^{-1} (mu_a - mu_b)
    + (1/2) ln(det(Sigma) / sqrt(det(Sigma_a) det(Sigma_b)))

Sigma = (Sigma_a + Sigma_b) / 2
```

### CKA-Derived Relations

When raw activations are stored, CKA is used to define overlap and alignment:
- overlap_coefficient = CKA
- jaccard_index = CKA
- bhattacharyya_coefficient = CKA
- subspace_alignment = CKA
- distance = 1 - CKA

### Curvature-Corrected Covariance

Curvature correction scales covariance using the effective radius:

```
correction = 1 + K * r^2 / 6          (K > 0)
correction = 1 / (1 - K * r^2 / 6)    (K < 0)
```

The correction factor is clamped to a stable range before application.

---

## Domains and Probes

`mc geometry interference predict` analyzes all domains in `AtlasDomain` using
`UnifiedAtlasInventory` probes:

- mathematical, logical, linguistic, mental
- computational, structural
- affective, relational
- temporal, spatial
- moral, safety
- philosophical, factual

---

## CLI Usage

```bash
# Predict merge geometry between two models
mc geometry interference predict /path/to/source /path/to/target

# Save a detailed JSON report
mc geometry interference predict /path/to/source /path/to/target --output-file interference_report.json

# Compute ConceptVolume for a single concept
mc geometry interference volume /path/to/model "justice"

# Inspect null-space capacity (related diagnostic)
mc geometry interference null-space /path/to/model
```

---

## Example Output (Text)

```
======================================================================
MERGE ANALYSIS REPORT
======================================================================

Source: Qwen2.5-0.5B-Instruct-bf16
Target: Qwen2-0.5B-Instruct-4bit
Layer: last

--------------------------------------------------
Per-Domain Analysis:
  MORAL:
    Concepts: 30
    Mean Overlap: 0.82
    Domain CKA: 0.9134
    Domain Aligned: False
    Mean Curvature Divergence: 0.07
    Mean Distance: 0.41
```

---

## Understanding Output

### Per-Domain Fields

- **Concepts**: Number of shared probes used for that domain.
- **Mean Overlap**: Average overlap score (mean of Bhattacharyya, overlap coefficient, Jaccard).
- **Domain CKA**: CKA over stacked concept activations (dimension-agnostic).
- **Domain Aligned**: True only when `abs(domain_cka - 1.0) <= machine_epsilon`.
- **Mean Curvature Divergence**: Average curvature divergence across concepts.
- **Mean Distance**: Average normalized geodesic distance.

### Global Metrics (JSON)

The JSON report includes:
- `meanOverlap`
- `meanCka`
- `meanCurvatureDivergence`
- `meanDistance`

These are raw measurements; interpret relative to baselines from similar model pairs.

---

## Limitations

1. **Single-sample volumes in predict**: Each concept is extracted from a single probe prompt,
   so per-concept volumes are point masses; CKA is computed at the domain level instead.
2. **Layer scope**: `predict` currently analyzes the last layer only (`layer = -1`).
3. **Probe coverage**: Results depend on available probes for each domain.

---

## Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/geometry/riemannian_density.py` | ConceptVolume + relation metrics |
| `src/modelcypher/core/domain/geometry/interference_predictor.py` | MergeAnalyzer + report types |
| `src/modelcypher/core/domain/geometry/manifold_curvature.py` | Curvature estimation |
| `src/modelcypher/cli/commands/geometry/interference.py` | CLI commands |
| `docs/research/interference_prediction.md` | This document |

---

## Future Directions

1. **Multi-sample probing**: Multiple prompt variants per concept for true volumes.
2. **Cross-domain analysis**: Track interference across domain boundaries.
3. **Temporal tracking**: Monitor drift across fine-tuning checkpoints.
4. **Merge planning integration**: Use metrics to prioritize domains for transplant.

---

## Related Work

- Pennec (2006): Intrinsic Statistics on Riemannian Manifolds
- Belkin & Niyogi (2003): Laplacian Eigenmaps
- Ainsworth et al. (2022): Git Re-Basin
