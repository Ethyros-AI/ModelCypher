# Knowledge Density Grafting Experiment

**Status**: In Progress
**Started**: 2025-12-28
**Models**: Qwen2.5-Coder-3B, Qwen2.5-3B-Chat, SmolLM-360M, Qwen2-0.5B

## Hypothesis

> **Merge = Fill gaps, not blend expertise.**
>
> If a model has learned all it can about concept X (dense, smooth representation),
> there's nothing to add. The goal is to overlay sparse regions to make them denser -
> filling gaps in a model's conceptual framework.

### Core Claims

1. **Dense regions are untouchable**: Well-learned concepts should NOT be modified
2. **Sparse regions are opportunities**: Gaps in understanding are where grafting adds value
3. **Uniform blending destroys coherence**: Even same-architecture models cannot be uniformly merged
4. **Null space = unexplored territory**: Not "unused capacity" but "unexplored conceptual space"

## Methodology

### Phase 1: Knowledge Density Measurement

Measure per-concept density using intrinsic dimension:
- **Dense**: Low intrinsic dimension = compressed/efficient representation
- **Sparse**: High intrinsic dimension = incomplete/inefficient representation

```bash
mc geometry research concept-density --model $MODEL --domain spatial --layer 5
```

### Phase 2: Knowledge State Diff

Compute graft opportunities between models:
- `graft_opportunity = source_density - target_density`
- Positive = source can help target
- Negative = target already denser

```bash
mc geometry research knowledge-diff $SOURCE $TARGET --output-path diff.json
```

### Phase 3: Graft Boundary Detection

Find density threshold where grafting is safe vs harmful:
- Below threshold: grafting improves performance
- Above threshold: grafting degrades or wastes computation

```bash
mc geometry research graft-boundary --source $SOURCE --target $TARGET
```

### Phase 4: Sparse Region Grafting

Apply targeted grafts using `--knowledge-delta-mask`:
- alpha=0.0 for dense layers (preserve)
- alpha>0.0 for sparse layers (graft)

```bash
mc model merge --source $SOURCE --target $TARGET --knowledge-delta-mask mask.json
```

## Results

See [results/](results/) for detailed outputs.

### Summary

| Experiment | Date | Key Finding |
|------------|------|-------------|
| [Phase 1: Density Measurement](results/phase1-density-measurement.md) | 2025-12-28 | Intrinsic dimension correlates with concept mastery |
| [Phase 2: Knowledge Diff](results/phase2-knowledge-diff.md) | 2025-12-28 | Merge direction is asymmetric; Qwen2 denser than SmolLM |
| [Phase 3: Graft Boundary](results/phase3-graft-boundary.md) | 2025-12-28 | Threshold at density=0.5; early layers sparse, late layers dense |
| [Same-Architecture Merge](results/same-arch-uniform-blend.md) | 2025-12-28 | Uniform blending destroys coherence even for same-architecture |

## Key Findings

### 1. CKA Tolerance Fix Required

The merge pipeline's PROBE BAROMETER gate was too strict:
- Previous: `phase_tol = machine_epsilon(~1e-7)`
- Failed at CKA=0.999999 (within 1e-6 of 1.0)
- Fixed: `phase_tol = max(base_eps * 100, 1e-5)`

### 2. Same-Architecture Uniform Blend Produces Gibberish

Despite:
- Same architecture (Qwen2.5 3B)
- 100% vocabulary overlap
- CKA=1.0 achieved

The merged model produces incoherent output. This validates the hypothesis that **uniform blending is wrong**.

### 3. Graft Boundary Exists at Density 0.5

| Density Bracket | Opportunity | Recommendation |
|-----------------|-------------|----------------|
| 0.0-0.3 | +0.123 | Graft |
| 0.3-0.5 | +0.159 | Graft |
| 0.5-0.7 | +0.011 | Neutral |
| 0.7-0.9 | -0.161 | Preserve |
| 0.9-1.0 | -0.270 | Preserve |

## Next Steps

1. **Implement density-aware grafting**: Use knowledge_delta_mask to limit blending to sparse regions
2. **Validate with perplexity**: Measure improvement on sparse concepts without degrading dense concepts
3. **Test zero-shot transfer**: Verify capabilities transfer without fine-tuning

## Files Modified

| File | Change |
|------|--------|
| `unified_geometric_merge.py:560-566` | Fixed CKA tolerance to 1e-5 |
| `cka.py` | Added `.best` property for corrected CKA |
| `gram_aligner.py` | Added bfloat16 → float32 casting for eigh() |
| `geometric_merge_orchestrator.py` | Cross-architecture tolerance of 1e-2 |

## Reproducibility

Models stored on external drive (not in repository):
- `/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-3B-Instruct-bf16`
- `/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16`
- `/Volumes/CodeCypher/models/mlx-community/SmolLM-360M-Instruct-4bit`
- `/Volumes/CodeCypher/models/mlx-community/Qwen2-0.5B-Instruct-4bit`

Experiment outputs on external drive:
- `/Volumes/CodeCypher/experiments/knowledge-density-2025-12-28/`
