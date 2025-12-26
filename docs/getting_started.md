# Getting Started with ModelCypher

ModelCypher is a high-dimensional geometry engine for Large Language Models. It supports multiple compute backends:

| Platform | Backend | Notes |
| :--- | :--- | :--- |
| **macOS** (Apple Silicon) | MLX | Default. Unified memory, fast local inference. |
| **Linux** (NVIDIA GPU) | CUDA | PyTorch CUDA backend for NVIDIA GPUs. |
| **Linux/Cloud** (TPU/GPU) | JAX | Google TPU pods, JAX GPU backends. |
| **Any** | NumPy | Testing and CI (no GPU required). |

## Prerequisites

### macOS (MLX Backend - Default)
- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4). 16GB+ RAM recommended.
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
   mc --help
   ```

## Key Commands

ModelCypher exposes its functionality through a single CLI: `mc` (or `modelcypher`).
See `CLI-REFERENCE.md` for the full command map and global flags (`--output json`, `--ai`, etc).

### 1. Model management (`mc model …`)
Use this to fetch/register models and probe local model directories.

```bash
# Fetch a model (downloads to ModelCypher storage; prints the local path)
mc model fetch mlx-community/Llama-2-7b-chat-mlx --auto-register

# Probe a local model directory (architecture + layer/tensor summary)
mc model probe ./models/Llama-2-7b-chat-mlx --output json
```

### 2. Training (`mc train …`)
Train LoRA adapters (including "sidecar"-style adapters) and manage training jobs.

```bash
# Preflight a training configuration (fit + rough ETA estimates)
mc train preflight \
    --model ./models/Mistral-7B-v0.1-mlx \
    --dataset data/safety.jsonl \
    --lora-rank 8 \
    --lora-alpha 16

# Start training
mc train start \
    --model ./models/Mistral-7B-v0.1-mlx \
    --dataset data/safety.jsonl \
    --lora-rank 8 \
    --lora-alpha 16 \
    --out adapters/safety_sidecar
```

### 3. Geometry + safety diagnostics (`mc geometry …`, `mc thermo …`)
Use these to interpret training stability, safety signals, and probe-based geometry.

```bash
# Training stability + “flatness”/SNR summaries
mc geometry training status --job <job_id>

# Probe semantic prime anchors (list/probe/compare)
mc geometry primes list

# Measure thermodynamic/entropy signals (see CLI reference for details)
mc thermo measure --help
```

## Next Steps

- Read about [High Dimensional Geometry](geometry/manifold_stitching.md).
- Explore the [Architecture](ARCHITECTURE.md).
