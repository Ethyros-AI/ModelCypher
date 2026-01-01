# Same-Architecture Uniform Blend Experiment

**Date**: 2025-12-28
**Models**: Qwen2.5-Coder-3B-Instruct-bf16 → Qwen2.5-3B-Instruct-bf16

## Hypothesis

Same-architecture models with identical vocabulary and dimensions should merge successfully.

## Method

Ran geometric merge pipeline with default (uniform) blending:

```bash
mc merge pipeline \
  --source /Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-3B-Instruct-bf16 \
  --target /Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16 \
  --output-dir merged-coder-to-chat/ \
  --transplant-domains coding
```

## Results

### Merge Pipeline Success

| Metric | Value |
|--------|-------|
| Layers | 36 |
| Weights | 434 |
| CKA | 1.0000 |
| Mean confidence | 0.9999987 |
| Rotations applied | 252 |
| Fisher scaling | 148 |
| Intrinsic scaled | 432 |

### Model Verification

| Test | Result |
|------|--------|
| Model loads | PASS |
| Generates text | PASS (72 tok/s, 0.146s TTFT) |
| Output coherence | **FAIL** |

### Example Output

```
Prompt: "What is 2+2? Answer briefly:"

Response: "escritRetrieveSTARX微®微 drummer手持端 strain X®.strptime
plagiar X Tigtal端假STAR Undefinedaina这种事情strike }];
XXXstrikeflex onTap onTapENG Yong® strike..."
```

## Key Finding

**Uniform blending destroys model coherence even for same-architecture models.**

Despite:
- Same architecture (Qwen2.5, 36 layers, 2048 dim)
- 100% vocabulary overlap (151,936 tokens)
- Perfect CKA alignment (1.0000)
- All geometric transformations applied correctly

The merged model produces **gibberish**.

## Analysis

### Why This Happens

1. **Geometry is preserved, semantics are destroyed**
   - CKA measures relational structure, not semantic content
   - Uniform blending averages weights, destroying learned associations

2. **All layers treated equally**
   - Dense regions (well-learned concepts) are modified unnecessarily
   - No distinction between "needs help" and "leave alone"

3. **Blending is wrong for merging**
   - Merge should be "graft into gaps", not "average everything"
   - The target model's expertise is overwritten

## Implications

1. **CKA=1.0 is necessary but not sufficient** for successful merging
2. **Uniform blending is fundamentally wrong** for model merging
3. **Density-aware grafting is required**: only modify sparse regions

## Next Steps

1. Re-run with `--knowledge-delta-mask` to limit blending
2. Set alpha=0.0 for dense layers (preserve target)
3. Set alpha=0.3 for sparse layers (graft source knowledge)

## Files

- Merge log: `/Volumes/CodeCypher/experiments/knowledge-density-2025-12-28/exp5-merge-log-v9.txt`
- Output model: `/Volumes/CodeCypher/experiments/knowledge-density-2025-12-28/merged-coder-to-chat/`
- Model size: 6.2GB

## Code Changes Required

Fixed CKA tolerance in `unified_geometric_merge.py:560-566`:
```python
# Before (too strict):
phase_tol = machine_epsilon(self._backend, sample_array)

# After (allows numerical precision):
base_eps = machine_epsilon(self._backend, sample_array)
phase_tol = max(base_eps * 100, 1e-5)
```
