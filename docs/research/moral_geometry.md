# Moral Geometry: Latent Ethicist Measurements

**Status**: Measurement-only (no thresholds or verdicts)
**Atlas**: 30 moral anchors across 6 moral foundations
**Probe Schemas**: `mc.geometry.moral.probe_model.v1`, `mc.geometry.moral.analyze.v1`

---

## Overview

ModelCypher probes moral structure using anchor activations from the Moral Atlas
(Haidt-style foundations). The analyzer returns raw geometric measurements:
axis orthogonality, gradient consistency, foundation clustering, virtue-vice
opposition, principal component variance, and a composite moral manifold score.

At least 15 anchors are required to run the analysis.

---

## Metrics Computed (Raw)

1. **Axis orthogonality**
   - Compute principal direction per axis via geodesic SVD.
   - Orthogonality = `1 - |cos|` between axis directions.

2. **Gradient consistency**
   - Spearman correlation between concept level and PC1 projection.
   - Monotonic flag is `abs(corr) > 0` (measurement, not a thresholded verdict).

3. **Foundation clustering**
   - Geodesic cosine similarity matrix is cached.
   - Within-foundation vs between-foundation means.
   - Separation ratio = within / (between + eps).
   - Reports most distinct foundation and most overlapping pair.

4. **Virtue-vice opposition**
   - `1 - cosine` between endpoints:
     - compassion ↔ cruelty
     - justice ↔ exploitation
     - devotion ↔ betrayal

5. **Principal components variance**
   - Top-5 variance ratios from geodesic SVD of centered anchors.

6. **Moral manifold score (MMS)**
   - Mean of: axis orthogonality mean, mean absolute gradient correlation,
     min(1, separation ratio), and mean opposition.

---

## CLI Usage

```bash
# List moral anchors
mc geometry moral anchors

# Probe a model (runs full analysis)
mc geometry moral probe-model /path/to/model

# Filter anchors
mc geometry moral anchors --foundation care_harm
mc geometry moral anchors --axis valence

# Analyze pre-computed activations
mc geometry moral analyze ./activations.json
```

---

## Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/moral_atlas.py` | Moral concept inventory |
| `src/modelcypher/core/domain/geometry/moral_geometry.py` | MoralGeometryAnalyzer |
| `src/modelcypher/cli/commands/geometry/moral.py` | CLI commands |
| `docs/research/moral_geometry.md` | This document |
