# Social Geometry: Latent Sociologist Measurements

> **Status**: Measurement-only (no thresholds or verdicts)
> **Implementation**: `src/modelcypher/core/domain/geometry/social_geometry.py`
> **CLI**: `mc geometry social anchors`, `mc geometry social probe-model`,
>          `mc geometry social analyze`

---

## Overview

Social Geometry probes implicit social structure in model representations using
anchor concepts from the Social Atlas. It reports raw geometric measurements:
axis orthogonality, gradient consistency, power gradient correlation, PCA
variance, and a composite social manifold score.

The Social Atlas contains 25 probes across 5 categories:
- Power hierarchy
- Kinship
- Formality
- Status markers
- Age

---

## Metrics Computed (Raw)

1. **Axis orthogonality**
   - Axes are defined by low/high anchor pairs:
     - power: `slave → emperor`
     - kinship: `enemy → family`
     - formality: `hey → salutations`
   - Orthogonality = `1 - |cos|` between axis vectors.

2. **Gradient consistency**
   - Spearman-style correlation between expected ordering and PC1 positions.
   - Monotonic flag from sign consistency of ordered anchor positions.

3. **Power gradient analysis**
   - Correlation between PC1 positions and expected power levels.
   - Power direction computed from Fréchet mean of low vs high status anchors.

4. **Principal components variance**
   - Top-5 variance ratios from PCA on anchor activations.

5. **Social manifold score (SMS)**
   - Weighted combination of:
     - mean axis orthogonality (30%)
     - mean gradient correlation (40%)
     - absolute power correlation (30%)

All outputs are raw values; interpretation is left to callers.

---

## CLI Usage

```bash
# List social anchors
mc geometry social anchors

# Probe a model (runs full analysis)
mc geometry social probe-model /path/to/model

# Analyze pre-computed activations
mc geometry social analyze ./activations.json
```

---

## Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/social_atlas.py` | Social concept inventory |
| `src/modelcypher/core/domain/geometry/social_geometry.py` | SocialGeometryAnalyzer |
| `src/modelcypher/cli/commands/geometry/social.py` | CLI commands |
| `docs/research/social_geometry.md` | This document |
