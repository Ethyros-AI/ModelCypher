# Atlas-Based Geometry: Domain-Specific Measurements

> **Status**: Measurement-only (no thresholds or verdicts)
> **Purpose**: Probe implicit structure in model representations using domain-specific anchor inventories

---

## Overview

ModelCypher probes representation geometry using curated anchor inventories (atlases). Each domain uses the same measurement pipeline:

1. Collect activations for domain-specific anchor concepts
2. Compute axis orthogonality, gradient consistency, and domain-specific metrics
3. Return raw measurements; interpretation is left to callers

All analyzers share common metric types:
- **Axis orthogonality**: `1 - |cos|` between principal directions
- **Gradient consistency**: Spearman correlation between concept level and PC1 projection
- **Principal components variance**: Top-5 variance ratios from geodesic SVD
- **Domain manifold score**: Composite of domain-specific metrics

---

## Moral Geometry

**Atlas**: 30 moral anchors across 6 moral foundations (Haidt-style)
**Implementation**: `src/modelcypher/core/domain/geometry/moral_geometry.py`
**Probe Schemas**: `mc.geometry.moral.probe_model.v1`, `mc.geometry.moral.analyze.v1`

### Metrics Computed

1. **Axis orthogonality** - Principal direction per axis via geodesic SVD
2. **Gradient consistency** - Spearman correlation between concept level and PC1
3. **Foundation clustering** - Within vs between-foundation similarity, separation ratio
4. **Virtue-vice opposition** - `1 - cosine` between endpoints (compassion↔cruelty, justice↔exploitation, devotion↔betrayal)
5. **Moral manifold score (MMS)** - Mean of axis orthogonality, gradient correlation, separation ratio, and opposition

### CLI Commands

```bash
# Probe a model (runs full analysis)
mc analyze concept-volume /path/to/model --domain moral

# Analyze pre-computed activations
mc analyze concept-volume ./activations.json --domain moral
```

### Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/moral_atlas.py` | Moral concept inventory |
| `src/modelcypher/core/domain/geometry/moral_geometry.py` | MoralGeometryAnalyzer |
| `src/modelcypher/cli/commands/geometry/moral.py` | CLI commands |

---

## Social Geometry

**Atlas**: 25 probes across 5 categories (power hierarchy, kinship, formality, status markers, age)
**Implementation**: `src/modelcypher/core/domain/geometry/social_geometry.py`

### Metrics Computed

1. **Axis orthogonality** - Axes defined by low/high anchor pairs:
   - power: `slave → emperor`
   - kinship: `enemy → family`
   - formality: `hey → salutations`
2. **Gradient consistency** - Spearman-style correlation between expected ordering and PC1 positions
3. **Power gradient analysis** - Correlation between PC1 positions and expected power levels
4. **Social manifold score (SMS)** - Weighted: axis orthogonality (30%), gradient correlation (40%), power correlation (30%)

### CLI Commands

```bash
# Probe a model (runs full analysis)
mc analyze concept-volume /path/to/model --domain social

# Analyze pre-computed activations
mc analyze concept-volume ./activations.json --domain social
```

### Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/social_atlas.py` | Social concept inventory |
| `src/modelcypher/core/domain/geometry/social_geometry.py` | SocialGeometryAnalyzer |
| `src/modelcypher/cli/commands/geometry/social.py` | CLI commands |

---

## Temporal Topology

**Atlas**: 25 probes across 5 categories (tense, duration, causality, lifecycle, sequence)
**Implementation**: `src/modelcypher/core/domain/geometry/temporal_topology.py`

### Metrics Computed

1. **Axis orthogonality** - Principal direction per axis (direction, duration, causality) via geodesic SVD
2. **Gradient consistency** - Spearman correlation between concept level and PC1
3. **Arrow of time** - Correlation between expected past→future ordering and PC1 projection
4. **Temporal manifold components** - `orthogonality_score`, `gradient_score`, `arrow_score` (raw values)
5. **Temporal manifold score** - Mean of the three components

### CLI Commands

```bash
# Probe a model (runs full analysis)
mc analyze concept-volume /path/to/model --domain temporal

# Analyze pre-computed activations
mc analyze concept-volume activations.json --domain temporal
```

### Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/temporal_atlas.py` | Temporal concept inventory |
| `src/modelcypher/core/domain/geometry/temporal_topology.py` | TemporalTopologyAnalyzer |
| `src/modelcypher/cli/commands/geometry/temporal.py` | CLI commands |

---

## Spatial Grounding

**Atlas**: 23 anchors across 5 categories (vertical, lateral, depth, mass, furniture)
**Implementation**: `src/modelcypher/core/domain/geometry/spatial_3d.py`, `cross_grounding_transfer.py`

### Axis Anchors

- X-axis: `right_hand - left_hand`
- Y-axis: `ceiling - floor`
- Z-axis: `foreground - background`

### Metrics Computed

1. **Axis orthogonality** - `1 - |cos|` between inferred X/Y/Z axes
2. **Gravity gradient** - Correlation between mass anchors and vertical axis alignment
3. **Volumetric density** - Density per anchor (norm / sqrt(variance)), density–mass correlation, perspective attenuation, inverse-square compliance
4. **Stereoscopy** - Parallax consistency across viewpoint prompts
5. **Occlusion** - Z-shift magnitude under front/back swap probes
6. **World model score** - Mean of axis, gravity, density, stereoscopy, and occlusion scores

### CLI Commands

```bash
# Probe a model (extract activations + full analysis)
mc model info /path/to/model
mc analyze concept-volume /path/to/model --domain spatial

# Analyze saved activations
mc analyze concept-volume ./activations.json --domain spatial

# Cross-grounding transfer
mc merge run -s source.json -t target.json -o ghost_anchors.json
```

### Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/spatial_atlas.py` | Spatial anchor inventory |
| `src/modelcypher/core/domain/geometry/spatial_3d.py` | Spatial3DAnalyzer |
| `src/modelcypher/core/domain/geometry/cross_grounding_transfer.py` | Cross-grounding engine |
| `src/modelcypher/cli/commands/geometry/spatial.py` | CLI commands |

---

## Semantic Primes

**Atlas**: NSM semantic primes (English 2014 inventory)
**Implementation**: `src/modelcypher/core/domain/agents/semantic_primes.py`
**Data**: `semantic_primes.json`, `semantic_prime_multilingual.json`, `semantic_prime_frames.json`

### Purpose

Semantic primes are treated as a small, standardized anchor set for cross-model comparison. ModelCypher uses the English 2014 inventory to probe embedding-space structure and compute CKA-based coherence metrics.

### Inventory Sources

- `semantic_primes.json` - English 2014 prime list and categories
- `semantic_prime_multilingual.json` - Multilingual variants (for future analysis)
- `semantic_prime_frames.json` - Frame-based variants (for future analysis)

### CLI Commands

```bash
# Probe a local model directory (writes optional JSON)
mc analyze concept-volume /path/to/model --output-file primes.json

# Compare two activation JSON files
mc analyze reasoning-geometry-validation model_a_primes.json model_b_primes.json
```

### Implementation Details

1. Encode each prime's first English exponent
2. Run a forward pass to the final layer and mean-pool activations
3. Compute CKA for all primes (overall coherence) and within each category

`compare` computes CKA between two activation JSONs and reports the most similar/divergent primes. If dimensions differ, it falls back to a centroid similarity heuristic for per-prime ranking.

### Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/agents/semantic_primes.py` | English 2014 inventory |
| `src/modelcypher/core/domain/agents/unified_atlas.py` | Unified atlas access |
| `src/modelcypher/data/semantic_primes.json` | Prime data |

---

## Common Patterns

### Minimum Anchor Requirements

Most analyzers require a minimum number of anchors:
- Moral geometry: 15 anchors minimum
- Other domains: varies by analysis type

### Output Format

All analyzers return structured results with:
- Raw metric values (no thresholds or verdicts)
- Probe metadata (model, layer, anchor count)
- Optional JSON output for downstream analysis

### Cross-Domain Analysis

Use `mc analyze dimension-profile` to compare intrinsic dimension across all domains for a model. This reveals domain-specific encoding strategies and bottleneck characteristics.

---

## Related

- [RESEARCH-MAP.md](../RESEARCH-MAP.md) - How atlas measurements connect to research + testable predictions
- [math/centered_kernel_alignment.md](math/centered_kernel_alignment.md) - CKA methodology
