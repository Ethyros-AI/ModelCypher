# Temporal Topology: Latent Chronologist Measurements

> **Status**: Measurement-only (no thresholds or verdicts)
> **Implementation**: `src/modelcypher/core/domain/geometry/temporal_topology.py`
> **Atlas**: `src/modelcypher/core/domain/agents/temporal_atlas.py`
> **CLI**: `mc geometry temporal anchors`, `mc geometry temporal probe-model`,
>          `mc geometry temporal analyze`

---

## Overview

Temporal Topology probes time-like structure in model representations using
anchor concepts from the Temporal Atlas. It reports raw geometric measurements
for axis orthogonality, gradient consistency, arrow-of-time detection, PCA
variance, and temporal manifold component scores.

The Temporal Atlas contains 25 probes across 5 categories:
- tense
- duration
- causality
- lifecycle
- sequence

---

## Metrics Computed (Raw)

1. **Axis orthogonality**
   - Principal direction per axis (direction, duration, causality) via geodesic SVD.
   - Orthogonality = `1 - |cos|` between axis directions.

2. **Gradient consistency**
   - Spearman correlation between concept level and PC1 projection.
   - Monotonic flag when correlation is non-zero.

3. **Arrow of time**
   - Correlation between expected past→future ordering and PC1 projection.
   - Reports anchor sets used for past and future.

4. **Principal components variance**
   - Top-5 variance ratios from geodesic SVD of centered anchors.

5. **Temporal manifold components**
   - `orthogonality_score`, `gradient_score`, `arrow_score` (raw values).
   - `temporal_manifold_score` is the mean of these three components.

All outputs are raw measurements; interpretation is left to callers.

---

## CLI Usage

```bash
# List temporal anchors
mc geometry temporal anchors

# Probe a model (runs full analysis)
mc geometry temporal probe-model /path/to/model

# Analyze pre-computed activations
mc geometry temporal analyze activations.json
```

---

## Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/temporal_atlas.py` | Temporal concept inventory |
| `src/modelcypher/core/domain/geometry/temporal_topology.py` | TemporalTopologyAnalyzer |
| `src/modelcypher/cli/commands/geometry/temporal.py` | CLI commands |
| `docs/research/temporal_topology.md` | This document |
