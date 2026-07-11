# Backend Comparison Guide

ModelCypher supports multiple platform backends. This guide helps you select the right backend for your environment.

Notes:
- In this repo, run CLI commands as `poetry run mc ...`.
- JAX CPU is an engineering fallback for CI and domain tests. It does not imply
  end-to-end Linux model-loader or expert-CLI parity.

## Quick Selection

| Platform | Backend | Install Command | Product status |
|----------|---------|-----------------|----------------|
| macOS Apple Silicon | MLX | `poetry install` | Primary end-to-end path |
| Linux + NVIDIA GPU | CUDA | `poetry install --extras cuda` | Protocol available; loader/CLI parity partial |
| Linux + TPU/GPU | JAX | `poetry install --extras jax` | Protocol available; loader/CLI parity partial |
| Linux CPU | JAX | `poetry install --extras jax` | CI/domain-test fallback only |

## Performance Characteristics

### macOS Backend (Apple Silicon)

**Strengths:**
- Unified memory architecture (no CPU↔accelerator copies)
- Lazy evaluation with automatic fusion
- Quantization support via the Backend protocol (`quantize` / `dequantize`)

**Weaknesses:**
- macOS only
- Smaller ecosystem of third-party tooling
- Limited batch sizes on memory-constrained devices

**Typical use cases:** Local development on Mac, memory-efficient inference

**Key Pattern:**
```python
# Some backends require explicit evaluation
result = backend.matmul(a, b)
backend.eval(result)  # Forces computation
```

### JAX Backend

**Strengths:**
- JIT compilation for optimized kernels
- TPU support for large-scale training
- Functional execution model

**Weaknesses:**
- Compilation overhead on first run
- Debugging can be complex (traced execution)
- Less intuitive for imperative code

**Typical use cases:** Backend-protocol research and CPU CI; accelerated
end-to-end workflows remain subject to the parity table below.

### CUDA Backend

**Strengths:**
- Mature tooling and debugging
- Wide hardware support
- Production-ready inference

**Weaknesses:**
- Explicit memory management
- Linux-focused ecosystem

**Typical use cases:** Production inference, large-scale training on NVIDIA hardware

**Key Pattern:**
```python
# Some backends require explicit synchronization
result = backend.matmul(a, b)
backend.eval(result)
```

## Backend-Specific Notes

### Lazy Evaluation

Some backends use lazy evaluation: operations are not executed until explicitly evaluated.
Call `backend.eval()` before:
- Timing operations
- Extracting values (use `backend.tolist()` / `backend.to_scalar()`)

### Random Keys

Some backends require explicit random state management:

```python
backend.random_seed(42)
samples = backend.random_categorical(logits, num_samples=10)
```

### Device Placement

Accelerator tensors are created on device by default:

```python
tensor = backend.zeros((100, 100))
values = backend.tolist(tensor)
```

## Selecting a Backend at Runtime

The backend is selected based on platform detection, with an environment override (`MC_BACKEND`, alias: `MODELCYPHER_BACKEND`).
Use `mc system probe backends` to discover backend keys on the current machine.

```python
from modelcypher.backends import detect_default_backend_type, get_backend

backend = get_backend(detect_default_backend_type())
```

## Memory Considerations

| Backend | Typical Memory Usage | Notes |
|---------|---------------------|-------|
| macOS backend | Lower | Unified memory, lazy evaluation |
| TPU backend | Higher | JIT compilation caches |
| NVIDIA backend | Medium | Explicit allocation |

## Troubleshooting

### "Backend not available"

Install the platform-appropriate backend extra from `pyproject.toml`, then re-run:
```bash
poetry run mc system probe backends
```

---

## Backend Parity Checklist

Goal: All platform backends should get the same capabilities, with backend-appropriate
performance defaults and no platform-only blockers in shared paths.

### Current Snapshot

- Shared domain geometry is expressed through the Backend protocol.
- Backend selection and tensor-level operations are covered by cross-platform tests.
- Model loading, activation collection, evaluation, and adapter workflows are
  not yet end-to-end portable.

### Parity Status

| Capability | MLX/macOS | JAX | CUDA |
|------------|-------|-----|--------|
| Backend selection via MC_BACKEND | ✓ | ✓ | ✓ |
| System health reports | ✓ | ✓ | ✓ |
| Unified inference engine | ✓ | ✓ | ✓ |
| Training engine + checkpoints | ✓ | ✓ | ✓ |
| CLI geometry commands | ✓ | Partial | Partial |
| Activation probing (merge) | ✓ | Planned | Planned |
| Evaluation service | ✓ | Planned | Planned |
| Entropy calibration | ✓ | Planned | Planned |
| Adapter wrapping | ✓ | Planned | Planned |

### Backlog (Prioritized)

1. **Platform loaders for CLI geometry probes**: Provide model loaders and tokenizers for non-macOS backends.
2. **Merge pipeline activation collection**: Implement activation collection for non-macOS backends.
3. **Evaluation + calibration**: Add non-macOS implementations in evaluation and entropy calibration services.
4. **Adapter tooling parity**: Add backend-agnostic wrapper with explicit layout metadata.
5. **Parity tests**: Add tests for system service and inference platform selection across backends.
