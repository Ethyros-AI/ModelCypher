# Performance Profiling Guide

This guide covers profiling techniques, caching strategies, and optimization tips for ModelCypher operations.

## Quick Profiling

### CLI Timing

Use the shell's `time` command for simple timing:

```bash
# Time a geometry operation
time mc geometry crm-build --model ./model --output crm.json

# Time with verbose output
MC_LOG_LEVEL=debug mc geometry validate ./model
```

### Memory Monitoring

Monitor memory during operations:

```bash
# Watch system status
mc system status --watch

# MLX-specific memory info
python -c "import mlx.core as mx; print(mx.metal.get_active_memory() / 1e9, 'GB')"

# macOS Activity Monitor equivalent
top -pid $(pgrep -f modelcypher)
```

## Caching Strategies

ModelCypher uses several cache layers to avoid redundant computation.

### Fingerprint Cache

Stores precomputed geometry fingerprints.

| Property | Value |
|----------|-------|
| Location | `~/.modelcypher/caches/fingerprints/` |
| Key | Model hash + probe configuration |
| TTL | Indefinite (invalidate manually) |
| Size | ~1-5 MB per model |

**Invalidation:**
```bash
rm -rf ~/.modelcypher/caches/fingerprints/
```

### Activation Cache

Stores layer activations for probe texts.

| Property | Value |
|----------|-------|
| Location | `/Volumes/CodeCypher/caches/activations/` |
| Key | Model + layer + probe batch |
| TTL | 7 days (configurable) |
| Size | 10-100 MB per model |

**Usage:**
```bash
# Enable activation caching
mc geometry crm-build --cache-activations ./model
```

### CRM Cache

Concept Response Matrices are stored at the specified output path.

```bash
# Reuse existing CRM
mc geometry crm-compare --crm-a ./crm1.json --crm-b ./crm2.json
```

## Optimizing Geometry Operations

### Batch Size Tuning

Larger batches improve throughput but require more memory:

```bash
# Memory-constrained (e.g., 8GB RAM)
mc geometry crm-build --batch-size 4 ./model

# High-memory systems (32GB+)
mc geometry crm-build --batch-size 32 ./model
```

**Guidelines:**
| RAM | Recommended Batch Size |
|-----|------------------------|
| 8 GB | 2-4 |
| 16 GB | 8-16 |
| 32 GB+ | 16-32 |

### Layer Selection

Probe only specific layers to reduce computation:

```bash
# Every 6th layer (typical for 24-layer models)
mc geometry crm-build --layers 0,6,12,18,24 ./model

# Just the last few layers
mc geometry crm-build --layers -4,-3,-2,-1 ./model
```

### Probe Subset

Use fewer probes for quick estimates:

```bash
# Use only semantic primes (65 probes instead of 343)
mc geometry crm-build --atlas semantic_primes ./model
```

## Python Profiling

### cProfile

For detailed function-level timing:

```python
import cProfile
import pstats

from modelcypher.core.use_cases.geometry_service import GeometryService

svc = GeometryService()

# Profile a specific operation
profiler = cProfile.Profile()
profiler.enable()
result = svc.compute_fingerprint(model_path)
profiler.disable()

# Print top 20 functions by cumulative time
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Line Profiler

For line-by-line analysis (requires `line_profiler`):

```python
# Add @profile decorator to functions of interest
# Then run with kernprof
kernprof -l -v script.py
```

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
backend.eval(result)

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

**Cause:** Batch size too large or too many layers

**Solutions:**
1. Reduce batch size: `--batch-size 4`
2. Select fewer layers: `--layers 0,12,24`
3. Use streaming mode where available
4. Close other applications

### Issue: Slow CRM Build

**Cause:** 343 probes × N layers × batch inference

**Solutions:**
1. Use layer subset: `--layers 0,6,12,18,24`
2. Use probe subset: `--atlas semantic_primes`
3. Enable caching: `--cache-activations`
4. Increase batch size if memory allows

### Issue: Fingerprint Mismatch After Model Update

**Cause:** Stale cache

**Solution:** Clear fingerprint cache:
```bash
rm ~/.modelcypher/caches/fingerprints/*model_name*
```

## Performance Benchmarks

Typical timings on M1 Max (32GB):

| Operation | Time | Notes |
|-----------|------|-------|
| Model probe (0.5B) | ~5s | Full 343 probes |
| Model probe (7B) | ~30s | Full 343 probes |
| CRM build (0.5B) | ~15s | All layers |
| CRM build (7B) | ~2min | All layers |
| Merge validation | ~10s | Two adapters |
| Intrinsic dimension | ~2s | Per layer |

*Timings vary based on hardware, model architecture, and configuration.*
