# Backend Comparison Guide

ModelCypher supports multiple compute backends for different platforms. This guide helps you select the right backend for your environment.

## Quick Selection

| Platform | Recommended Backend | Install Command |
|----------|---------------------|-----------------|
| macOS Apple Silicon | MLXBackend | `poetry install` |
| Linux + NVIDIA GPU | CUDABackend | `poetry install -E cuda` |
| Linux + TPU | JAXBackend | `poetry install -E jax` |

## Capability Matrix

| Feature | MLX | JAX | CUDA |
|---------|-----|-----|------|
| Unified Memory | Yes | No | No |
| GPU Acceleration | Metal | TPU/GPU | CUDA |
| Quantization (4/8-bit) | Full | Partial | No |
| Training | Yes | Yes | Yes |
| Inference | Yes | Yes | Yes |
| SOTA Performance APIs | Yes | Partial | No |

## Performance Characteristics

### MLX (Apple Silicon)

**Strengths:**
- Unified memory architecture (no CPU↔GPU copies)
- Lazy evaluation with automatic fusion
- Native quantization support (4-bit, 8-bit)
- Zero-copy operations

**Weaknesses:**
- macOS only
- Smaller ecosystem than PyTorch
- Limited batch sizes on memory-constrained devices

**Best for:** Local development on Mac, memory-efficient inference

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
- Excellent for research workflows

**Weaknesses:**
- Compilation overhead on first run
- Debugging can be complex (traced execution)
- Less intuitive for imperative code

**Best for:** TPU training, research, large-scale experiments

**Key Pattern:**
```python
# JAX benefits from JIT compilation
@jax.jit
def compute(a, b):
    return backend.matmul(a, b)
```

### CUDA (NVIDIA)

**Strengths:**
- Largest ecosystem (PyTorch compatibility)
- Mature tooling and debugging
- Wide hardware support
- Production-ready inference

**Weaknesses:**
- Explicit memory management
- Linux-focused ecosystem
- No native quantization in this backend

**Best for:** Production inference, large-scale training on NVIDIA hardware

**Key Pattern:**
```python
# CUDA requires explicit synchronization
result = backend.matmul(a, b)
backend.eval()  # torch.cuda.synchronize()
```

### NumPy (Testing)

**Strengths:**
- Universal compatibility
- Deterministic behavior
- Easy debugging
- No GPU dependencies

**Weaknesses:**
- No GPU acceleration
- Too slow for real models
- Not suitable for production

**Best for:** Unit tests, CI/CD, algorithm development

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
- Converting to numpy
- Checking values

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
numpy_array = backend.to_numpy(tensor)  # Moves to CPU
```

## Selecting a Backend at Runtime

The backend is typically selected based on environment:

```python
import os

if os.environ.get("MODELCYPHER_BACKEND") == "cuda":
    from modelcypher.backends.cuda_backend import CUDABackend
    backend = CUDABackend()
elif os.environ.get("MODELCYPHER_BACKEND") == "jax":
    from modelcypher.backends.jax_backend import JAXBackend
    backend = JAXBackend()
else:
    # Default to MLX on macOS
    from modelcypher.backends.mlx_backend import MLXBackend
    backend = MLXBackend()
```

## Memory Considerations

| Backend | Typical Memory Usage | Notes |
|---------|---------------------|-------|
| MLX | Lower | Unified memory, lazy evaluation |
| JAX | Higher | JIT compilation caches |
| CUDA | Medium | Explicit allocation |
| NumPy | Highest | Full precision, no optimization |

For memory-constrained environments:
1. Use smaller batch sizes (`--batch-size 4`)
2. Select specific layers (`--layers 0,6,12,18,24`)
3. Enable streaming mode where available

## Troubleshooting

### "torch is required for the CUDA backend"
Install PyTorch with CUDA support:
```bash
poetry install -E cuda
# or
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "mlx is required for the MLX backend"
Ensure you're on macOS with Apple Silicon:
```bash
poetry install
python -c "import mlx; print(mlx.__version__)"
```

### "jax is required for the JAX backend"
Install JAX for your platform:
```bash
poetry install -E jax
# For TPU/GPU, see https://github.com/google/jax#installation
```
