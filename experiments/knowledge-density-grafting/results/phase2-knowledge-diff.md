# Phase 2: Knowledge State Diff

**Date**: 2025-12-28
**Models**: SmolLM-360M vs Qwen2-0.5B (bidirectional comparison)

## Method

Computed knowledge diff: `graft_opportunity = source_density - target_density`

- Positive opportunity: source can help target (target is sparse where source is dense)
- Negative opportunity: target already denser (grafting would waste computation or harm)

## Results

### SmolLM → Qwen2 (Can SmolLM help Qwen2?)

| Metric | Value |
|--------|-------|
| Overall opportunity | **-0.278** (negative) |
| High graft opportunities | 2 |
| No-graft concepts | 45 |

**Interpretation**: Qwen2 is already denser on most concepts. SmolLM cannot help Qwen2.

High opportunities (only 2):
- foreground
- sky

### Qwen2 → SmolLM (Can Qwen2 help SmolLM?)

| Metric | Value |
|--------|-------|
| Overall opportunity | **+0.278** (positive) |
| High graft opportunities | 45 |
| No-graft concepts | 2 |

**Interpretation**: Qwen2 can help SmolLM on 45 concepts.

Top graft opportunities:
| Concept | Opportunity |
|---------|-------------|
| right_hand | 0.705 |
| here | 0.683 |
| left_hand | 0.662 |
| ... | ... |

Only 2 no-graft concepts:
- sky (SmolLM is denser)
- foreground (SmolLM is denser)

## Key Finding

**Merge direction matters.**
- Qwen2 → SmolLM: 45 graft opportunities
- SmolLM → Qwen2: only 2 graft opportunities

This is NOT symmetric. The right direction is from denser to sparser model.

## Raw Data

```json
{
  "smolm_to_qwen2": {
    "overall_opportunity": -0.278,
    "high_opportunity_count": 2,
    "no_graft_count": 45,
    "direction": "sparse_to_dense",
    "recommendation": "do_not_merge"
  },
  "qwen2_to_smolm": {
    "overall_opportunity": 0.278,
    "high_opportunity_count": 45,
    "no_graft_count": 2,
    "direction": "dense_to_sparse",
    "recommendation": "graft_sparse_regions"
  }
}
```

## Commands Used

```bash
mc geometry research knowledge-diff \
  /Volumes/CodeCypher/models/mlx-community/SmolLM-360M-Instruct-4bit \
  /Volumes/CodeCypher/models/mlx-community/Qwen2-0.5B-Instruct-4bit \
  --output-path smolm-to-qwen2-diff.json

mc geometry research knowledge-diff \
  /Volumes/CodeCypher/models/mlx-community/Qwen2-0.5B-Instruct-4bit \
  /Volumes/CodeCypher/models/mlx-community/SmolLM-360M-Instruct-4bit \
  --output-path qwen2-to-smolm-diff.json
```
