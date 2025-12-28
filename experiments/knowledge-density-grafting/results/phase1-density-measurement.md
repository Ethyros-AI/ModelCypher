# Phase 1: Knowledge Density Measurement

**Date**: 2025-12-28
**Models**: SmolLM-360M-Instruct-4bit, Qwen2-0.5B-Instruct-4bit

## Method

Computed per-concept density using intrinsic dimension at representative layers:
- SmolLM: Layer 5 (mid-network)
- Qwen2: Layer 12 (mid-network)

Density formula: `density = 1 / normalized_intrinsic_dimension`

## Results

### SmolLM-360M (Layer 5, Spatial Domain)

| Metric | Value |
|--------|-------|
| Mean density | 0.450 |
| Sparse concepts | 19 |
| Dense concepts | 19 |

**Most Sparse** (incomplete representations):
- right_hand: 0.295
- here: 0.317
- left_hand: 0.338

**Most Dense** (well-learned):
- be_someone: 0.893
- where: 0.772

### Qwen2-0.5B (Layer 12, Spatial Domain)

| Metric | Value |
|--------|-------|
| Mean density | 0.947 |
| Sparse concepts | 8 |
| Dense concepts | 30 |

**Most Sparse**:
- sky: 0.643
- be_somewhere: 0.648
- floor: 0.662

**Most Dense**:
- 30 concepts at density 1.0 (fully compressed)

## Key Finding

**Intrinsic dimension correlates inversely with density.**
- Lower intrinsic dimension = higher density = more compressed/efficient representation
- Qwen2 is significantly denser overall (0.947 vs 0.450)

## Raw Data

```json
{
  "smolm_spatial_l5": {
    "mean_density": 0.450,
    "sparse_count": 19,
    "dense_count": 19,
    "domain": "spatial",
    "layer": 5
  },
  "qwen2_spatial_l12": {
    "mean_density": 0.947,
    "sparse_count": 8,
    "dense_count": 30,
    "domain": "spatial",
    "layer": 12
  }
}
```

## Commands Used

```bash
mc geometry research concept-density \
  /Volumes/CodeCypher/models/mlx-community/SmolLM-360M-Instruct-4bit \
  --domain spatial --layer 5 --output json

mc geometry research concept-density \
  /Volumes/CodeCypher/models/mlx-community/Qwen2-0.5B-Instruct-4bit \
  --domain spatial --layer 12 --output json
```
