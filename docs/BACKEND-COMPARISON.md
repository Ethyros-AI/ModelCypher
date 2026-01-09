# Backend Comparison Guide

ModelCypher supports multiple compute backends for different platforms. This guide helps you select the right backend for your environment.

Notes:
- In this repo, run CLI commands as `poetry run mc ...`.
- ModelCypher does not auto-fallback to CPU. If no GPU backend is available, it fails fast with an install hint.

## Quick Selection

| Platform | Default Backend | Install Command |
|----------|---------------------|-----------------|
| macOS Apple Silicon | MLXBackend | `poetry install` |
| Linux + NVIDIA GPU | CUDABackend | `poetry install -E cuda` |
| Linux + TPU | JAXBackend | `poetry install -E jax` |

## Performance Characteristics

### MLX (Apple Silicon)

**Strengths:**
- Unified memory architecture (no CPU↔GPU copies)
- Lazy evaluation with automatic fusion
- Quantization support via the Backend protocol (`quantize` / `dequantize`)

**Weaknesses:**
- macOS only
- Smaller ecosystem than PyTorch
- Limited batch sizes on memory-constrained devices

**Typical use cases:** Local development on Mac, memory-efficient inference

**Key Pattern:**
```python
# MLX requires explicit evaluation
result = backend.matmul(a, b)
backend.eval(result)  # Forces computation
```

### JAX (TPU/GPU)

**Strengths:**
- JIT compilation for optimized kernels
- TPU support for large-scale training
- Functional programming model
- Common in research workflows due to JIT + TPU support

**Weaknesses:**
- Compilation overhead on first run
- Debugging can be complex (traced execution)
- Less intuitive for imperative code

**Typical use cases:** TPU training, research, large-scale experiments

### CUDA (NVIDIA)

**Strengths:**
- Largest ecosystem (PyTorch ecosystem support)
- Mature tooling and debugging
- Wide hardware support
- Production-ready inference

**Weaknesses:**
- Explicit memory management
- Linux-focused ecosystem

**Typical use cases:** Production inference, large-scale training on NVIDIA hardware

**Key Pattern:**
```python
# CUDA requires explicit synchronization
result = backend.matmul(a, b)
backend.eval()  # torch.cuda.synchronize()
```

## Backend-Specific Notes

### MLX Lazy Evaluation

MLX uses lazy evaluation - operations are not executed until explicitly evaluated:

```python
a = backend.zeros((1000, 1000))
b = backend.ones((1000, 1000))
c = backend.matmul(a, b)  # Not yet computed!
backend.eval(c)            # Now it runs
```

Always call `backend.eval()` before:
- Timing operations
- Extracting values (use `backend.tolist()` / `backend.to_scalar()`)

### JAX Random Keys

JAX uses explicit random state management:

```python
backend.random_seed(42)  # Sets the initial key
samples = backend.random_categorical(logits, num_samples=10)
```

### CUDA Device Placement

All CUDA tensors are created on the GPU:

```python
# Automatically on CUDA device
tensor = backend.zeros((100, 100))  # device="cuda"
tensor_list = backend.tolist(tensor)  # Extracts to Python values
```

## Selecting a Backend at Runtime

The backend is selected based on platform detection, with an environment override (`MC_BACKEND`, alias: `MODELCYPHER_BACKEND`):

```python
from modelcypher.backends import detect_default_backend_type, get_backend

backend = get_backend(detect_default_backend_type())
```

## Memory Considerations

| Backend | Typical Memory Usage | Notes |
|---------|---------------------|-------|
| MLX | Lower | Unified memory, lazy evaluation |
| JAX | Higher | JIT compilation caches |
| CUDA | Medium | Explicit allocation |

## Troubleshooting

### "torch is required for the CUDA backend"
Install PyTorch with CUDA support:
```bash
poetry install -E cuda
```
If you need a specific CUDA wheel, follow the official PyTorch install instructions for your platform/CUDA version.

### "mlx is required for the MLX backend"
Ensure you're on macOS with Apple Silicon:
```bash
poetry install
poetry run python -c "import mlx; print(mlx.__version__)"
```

### "jax is required for the JAX backend"
Install JAX for your platform:
```bash
poetry install -E jax
# For TPU/GPU, see https://github.com/google/jax#installation
```
