# Bug Report: Rank Decrease During Augmentation

## Observed Behavior

During rank augmentation, the source model's rank DECREASES instead of increasing.

**Initial state (after 4596 probes):**
- Layer 6: src=2870/4096
- Layer 7: src=3095/4096

**After Round 2 (added 50 probes):**
- Layer 6: src=224/4096 (DECREASED by 2646)
- Layer 7: src=273/4096 (DECREASED by 2822)

## Mathematical Impossibility

Adding independent samples to a matrix can only:
1. Increase rank (if the new sample is linearly independent)
2. Keep rank the same (if the new sample is linearly dependent)

Rank CANNOT decrease when adding samples. This indicates a bug in the accumulation code.

## Suspected Causes

1. **Overwriting instead of appending**: Data might be replaced rather than concatenated
2. **Shape mismatch**: Augmentation activations might have different shapes than initial probes
3. **List/Array type confusion**: The code handles both lists and arrays; a transition might lose data
4. **Lazy evaluation (MLX)**: Tensors might be evaluated at wrong time, corrupting data
5. **Layer index mismatch**: Source (36 layers) vs Target (16 layers) might cause confusion

## Code Location

The bug is likely in `probe.py` lines 560-600, where augmentation activations are accumulated:

```python
for lidx in source_layer_activations.keys():
    src_act = src_result.get(lidx)
    ...
    if isinstance(source_layer_activations[lidx], list):
        source_layer_activations[lidx].append(src_act)
    else:
        source_layer_activations[lidx] = b.concatenate(
            [source_layer_activations[lidx], b.expand_dims(src_act, 0)],
            axis=0,
        )
```

## Next Steps

1. Add debugging to log array shapes before/after each operation
2. Verify that `source_layer_activations[lidx]` is not being overwritten elsewhere
3. Check if there's a global variable issue or aliasing problem
4. Verify that `collect_hidden_activations` returns consistent shapes

## Impact

This bug prevents any merge that requires rank augmentation (source_dim > initial_probe_count).
