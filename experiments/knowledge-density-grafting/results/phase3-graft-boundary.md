# Phase 3: Graft Boundary Detection

**Date**: 2025-12-28
**Models**: Qwen2-0.5B → SmolLM-360M

## Method

Binned concepts by density into 5 brackets and measured mean graft opportunity per bracket.

## Results

| Density Bracket | Concepts | Mean Opportunity | Recommendation | Layer Distribution |
|-----------------|----------|------------------|----------------|-------------------|
| 0.0-0.3 | 10 | +0.123 | consider_graft | L0: 7, L5: 3 |
| 0.3-0.5 | 53 | +0.159 | consider_graft | L0: 23, L5: 27, L10: 1, L15: 2 |
| 0.5-0.7 | 30 | +0.011 | neutral | L10: 15, L15: 15 |
| 0.7-0.9 | 14 | -0.161 | do_not_graft | L10: 6, L15: 8 |
| 0.9-1.0 | 13 | -0.270 | do_not_graft | L10: 8, L15: 5 |

## Key Findings

### Graft Boundary: density = 0.5

- Below 0.5: Positive opportunity → grafting helps
- Above 0.5: Negative opportunity → grafting harms or wastes computation

### Layer Distribution

- **Early layers (0, 5)**: Concentrate sparse concepts (opportunities)
- **Late layers (10, 15)**: Concentrate dense concepts (preserve)

### Null Space Availability

Mean null fraction: 0.999 (99.9% of capacity available for grafting)

Capacity is NOT the bottleneck. Density difference is.

## Implications

1. **Only graft into layers 0-5** for SmolLM
2. **Preserve layers 10-15** completely
3. **Use alpha scaling**: alpha=0.3 for sparse layers, alpha=0.0 for dense layers

## Raw Data

```json
{
  "graft_boundary": 0.5,
  "brackets": [
    {"range": "0.0-0.3", "count": 10, "opportunity": 0.123, "action": "graft"},
    {"range": "0.3-0.5", "count": 53, "opportunity": 0.159, "action": "graft"},
    {"range": "0.5-0.7", "count": 30, "opportunity": 0.011, "action": "neutral"},
    {"range": "0.7-0.9", "count": 14, "opportunity": -0.161, "action": "preserve"},
    {"range": "0.9-1.0", "count": 13, "opportunity": -0.270, "action": "preserve"}
  ],
  "mean_null_fraction": 0.999,
  "recommendation": {
    "graft_layers": [0, 5],
    "preserve_layers": [10, 15],
    "suggested_alpha": {
      "sparse": 0.3,
      "dense": 0.0
    }
  }
}
```

## Commands Used

```bash
mc geometry research graft-boundary \
  --source /Volumes/CodeCypher/models/mlx-community/Qwen2-0.5B-Instruct-4bit \
  --target /Volumes/CodeCypher/models/mlx-community/SmolLM-360M-Instruct-4bit \
  --density-brackets 0.3,0.5,0.7,0.9 \
  --output-path graft-boundary.json
```
