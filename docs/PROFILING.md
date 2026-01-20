# Performance Profiling Guide

This guide covers profiling techniques, caching strategies, and optimization tips for ModelCypher operations.

In this repo, run CLI commands as `poetry run mc ...` (instead of `mc ...`).

## Quick Profiling

### CLI Timing

Use the shell's `time` command for simple timing:

```bash
# Time a geometry operation
time poetry run mc geometry crm build --model ./model --output-path ./crm.json

# Time with verbose output
poetry run mc --log-level debug geometry validate
```

### Memory Monitoring

Monitor memory during operations:

```bash
# Watch system status (if `watch` is available)
watch -n 2 poetry run mc system status

# Portable loop
while true; do poetry run mc system status; sleep 2; done

# MLX-specific memory info
poetry run python -c "import mlx.core as mx; print(mx.metal.get_active_memory() / 1e9, 'GB')"

# macOS Activity Monitor equivalent
top -pid $(pgrep -f modelcypher)
```

## Caching Strategies

ModelCypher uses several cache layers to avoid redundant computation.

### Fingerprint Cache

Stores precomputed geometry fingerprints.

| Property | Value |
|----------|-------|
| Location | `$HOME/Library/Caches/ModelCypher/fingerprints/` |
| Key | Model path hash + config hash + model mtime |
| TTL | 30 days |
| Size | Varies by probe count |

**Invalidation:**
```bash
rm -rf "$HOME/Library/Caches/ModelCypher/fingerprints/"
```

### Geometry Metrics Cache

Caches expensive point-cloud metrics (GW, intrinsic dimension, topological fingerprint,
spectral signature).

| Property | Value |
|----------|-------|
| Location | `$HOME/Library/Caches/ModelCypher/geometry_metrics/` |
| TTL | 7 days |

### Computation Cache (Memory-Only)

Session-scoped cache for Gram matrices, geodesic distances, SVDs, and Fréchet means.
Cleared when the process exits.

### CRM Cache

Concept Response Matrices are stored at the specified output path.

```bash
# Reuse existing CRM
poetry run mc geometry crm compare --source ./crm1.json --target ./crm2.json
```

## Optimizing Geometry Operations

### Atlas Dimensionality

Atlas dimensionality uses geometry-derived settings (no user-tuned batching or pooling):

```bash
poetry run mc geometry atlas dimensionality /path/to/model
```

## Python Profiling

### cProfile

For detailed function-level timing:

```python
import cProfile
import pstats

from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

svc = GeometryMetricsService()

# Profile a specific operation
profiler = cProfile.Profile()
profiler.enable()
result = svc.compute_topological_fingerprint(points=[[0.0, 0.0], [1.0, 1.0]])
profiler.disable()

# Print top 20 functions by cumulative time
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Line Profiler

For line-by-line analysis (requires `line_profiler`):

Add `@profile` decorator to functions of interest, then run:

```bash
kernprof -l -v script.py
```

Note: `kernprof` comes from `line_profiler` and is not installed by default in this repo's Poetry environment.

## MLX-Specific Profiling

### Metal GPU Trace

Capture GPU execution for analysis in Instruments:

```python
import mlx.core as mx

mx.metal.start_capture()
# ... your geometry operations ...
mx.metal.stop_capture("trace.gputrace")
```

Open `trace.gputrace` in Xcode Instruments.

### Memory Tracking

```python
import mlx.core as mx

# Before operation
before = mx.metal.get_active_memory()

# ... geometry operation ...
mx.eval(result)

# After operation
after = mx.metal.get_active_memory()
print(f"Memory used: {(after - before) / 1e9:.2f} GB")
```

## Common Performance Issues

### Issue: Slow First Run

**Cause:** JIT compilation (JAX) or lazy graph building (MLX)

**Solution:**
- JAX: Use `jax.jit` for repeated operations
- MLX: First run is slower; subsequent runs are faster

### Issue: Out of Memory

**Cause:** Activation collection across too many layers at once

**Solutions:**
1. Reduce scope (smaller models or fewer operations).
2. Close other applications.

### Issue: Slow CRM Build

**Cause:** All probes × N layers × batch inference

**Solutions:**
1. Reuse existing CRM outputs instead of rebuilding.
2. Run on the fastest available backend (MLX/CUDA).
3. Use targeted probe commands when a full CRM is unnecessary (e.g., `mc geometry primes probe-model`).

### Issue: Fingerprint Mismatch After Model Update

**Cause:** Stale cache

**Solution:** Clear fingerprint cache:
```bash
rm "$HOME/Library/Caches/ModelCypher/fingerprints/"*model_name*
```

## Performance Benchmarks

Use local measurements for accuracy:

```bash
poetry run mc system benchmark cache
```

End-to-end timings vary widely by model size, backend, and probe set.
