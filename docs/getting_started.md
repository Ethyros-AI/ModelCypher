# Getting Started with ModelCypher

ModelCypher is a CLI-first toolkit for measuring the geometry of LLM representations (intrinsic dimension, curvature, entropy, and similarity).

It supports multiple compute backends:

| Platform | Backend | Notes |
| :--- | :--- | :--- |
| **macOS** (Apple Silicon) | MLX | Default. Unified memory, fast local inference. |
| **Linux** (NVIDIA GPU) | CUDA | PyTorch CUDA backend for NVIDIA GPUs. |
| **Linux/Cloud** (TPU/GPU) | JAX | Google TPU pods, JAX GPU backends. |

## Prerequisites

### macOS (MLX Backend - Default)
- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4). 16GB+ RAM.
- **OS**: macOS 14.0+ (Sonoma or later).
- **Python**: 3.11+

### Linux (CUDA Backend)
- **Hardware**: Linux with NVIDIA GPU.
- **Python**: 3.11+
- **Note**: Install with `poetry install -E cuda` and set `MC_BACKEND=cuda`.

### Linux/Cloud (JAX Backend)
- **Hardware**: Any Linux system with TPU or GPU.
- **Python**: 3.11+
- **Note**: Install with `poetry install -E jax` and set `MC_BACKEND=jax`.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ethyros-AI/ModelCypher.git
   cd ModelCypher
   ```

2. **Install dependencies**:
   ```bash
   # macOS (MLX backend - default)
   poetry install

   # Linux (CUDA backend)
   poetry install -E cuda

   # Linux/Cloud (JAX backend)
   poetry install -E jax
   ```

3. **Verify installation**:
   ```bash
   poetry run mc --help
   ```

## Key Commands

All examples below assume you are in the repo root and run `mc` via `poetry run mc …`.
See [CLI-REFERENCE.md](CLI-REFERENCE.md) for the full command map and global flags (`--ai`, `--pretty`, `--log-level`, etc).

### 1. Model management (`mc model …`)
Use this to fetch/register models and probe local model directories.

```bash
# Fetch a model (downloads to ModelCypher storage; prints the local path)
poetry run mc model fetch mlx-community/Qwen2.5-0.5B-Instruct-bf16 --auto-register --alias qwen

# Probe a local model directory (architecture + summary)
poetry run mc model probe /path/to/model
```

### 2. Geometry + safety diagnostics (`mc geometry …`, `mc thermo …`)
Use these to interpret training stability, safety signals, and probe-based geometry.

```bash
# Probe a model for 3D spatial geometry
poetry run mc geometry spatial probe-model /path/to/model

# Measure thermodynamic/entropy signals (see CLI reference for details)
poetry run mc thermo measure --model /path/to/model "Your prompt here"
```

### 3. Model merging (`mc merge …`)
Merge takes knowledge from SOURCE and adds it to TARGET via null-space projection.

```bash
poetry run mc merge run -s /path/to/source -t /path/to/target -o /path/to/output_dir
```

## Training

Training commands require a dataset path and explicit hyperparameters. For full workflows and guidance, see [TRAINING-GUIDE.md](TRAINING-GUIDE.md).

```bash
poetry run mc train preflight --help
poetry run mc train start --help
```

## Next Steps

- Start with [START-HERE.md](START-HERE.md) for reading paths.
- Read about [High Dimensional Geometry](geometry/manifold_stitching.md).
- Explore the [Architecture](ARCHITECTURE.md).
