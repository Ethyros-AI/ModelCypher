# Spatial Grounding: 3D World Model Measurements

> **Status**: Measurement-only (no thresholds or verdicts)
> **Implementation**: `src/modelcypher/core/domain/geometry/spatial_3d.py`,
>                    `src/modelcypher/core/domain/geometry/cross_grounding_transfer.py`
> **CLI**: `mc geometry spatial ...`

---

## Overview

Spatial Grounding probes whether a model’s representations encode consistent
3D structure using a spatial anchor inventory. It reports raw geometric metrics
for axis orthogonality, gravity correlation, volumetric density, stereoscopy,
occlusion, and a composite world model score.

---

## Spatial Atlas

The Spatial Atlas contains 23 anchors across 5 categories:
- **vertical** (e.g., ceiling, floor, sky, ground, cloud, basement)
- **lateral** (left_hand, right_hand, west, east)
- **depth** (foreground, background, horizon, here, there)
- **mass** (balloon, stone, feather, anvil)
- **furniture** (chair, table, lamp, rug)

Axis anchors used for orthogonality:
- X-axis: `right_hand - left_hand`
- Y-axis: `ceiling - floor`
- Z-axis: `foreground - background`

---

## Metrics Computed (Raw)

1. **Axis orthogonality**
   - `1 - |cos|` between inferred X/Y/Z axes.

2. **Gravity gradient**
   - Correlation between mass anchors and vertical axis alignment.

3. **Volumetric density**
   - Density per anchor (norm / sqrt(variance)).
   - Density–mass correlation, perspective attenuation, inverse-square compliance.

4. **Stereoscopy**
   - Parallax consistency across viewpoint prompts.

5. **Occlusion**
   - Z-shift magnitude under front/back swap probes.

6. **World model score**
   - Mean of axis, gravity, density, stereoscopy, and occlusion scores.

All outputs are raw measurements; interpretation is left to callers.

---

## CLI Commands

```bash
# List spatial anchors
mc geometry spatial anchors

# Probe a model (extract activations + full analysis)
mc geometry spatial probe-model /path/to/model

# Analyze saved activations
mc geometry spatial analyze ./activations.json

# Component-level probes
mc geometry spatial gravity --model /path/to/model
mc geometry spatial density --model /path/to/model

# Cross-grounding
mc geometry spatial cross-grounding-feasibility source.json target.json
mc geometry spatial cross-grounding-transfer source.json target.json -o ghost_anchors.json
```

---

## Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/spatial_atlas.py` | Spatial anchor inventory |
| `src/modelcypher/core/domain/geometry/spatial_3d.py` | Spatial3DAnalyzer |
| `src/modelcypher/core/domain/geometry/cross_grounding_transfer.py` | Cross-grounding engine |
| `src/modelcypher/cli/commands/geometry/spatial.py` | CLI commands |
| `docs/research/spatial_grounding.md` | This document |
