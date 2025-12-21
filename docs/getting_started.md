# Getting Started with ModelCypher

ModelCypher is a high-dimensional geometry engine for Large Language Models. It runs on **Apple Silicon (Mac)** using the `MLX` framework.

## Prerequisites

- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4). 16GB+ RAM recommended.
- **OS**: macOS 14.0+ (Sonoma or later).
- **Python**: 3.10+
- **Package Manager**: `uv` (recommended) or `pip`.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ModelCypher/ModelCypher.git
   cd ModelCypher
   ```

2. **Install dependencies**:
   Using `uv` (faster):
   ```bash
   uv sync
   ```
   Using `pip`:
   ```bash
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   mc-inspect --help
   ```

## Key Commands

ModelCypher exposes its functionality through several CLI tools.

### 1. `mc-inspect`: Geometric Analysis
Use this to "fingerprint" a model or check its intersection with another.

```bash
# Scan a model's rotational roughness
mc-inspect scan --model mlx-community/Llama-2-7b-chat-mlx

# Compare two models (Intersection Map)
mc-inspect intersection --source mlx-community/Llama-2-7b-mlx --target mlx-community/Llama-2-7b-chat-mlx
```

### 2. `mc-train`: Geometric Training
Train adapters (LoRA) with geometric constraints.

```bash
# Train a "Sidecar" adapter for safety
mc-train lora \
    --model mlx-community/Mistral-7B-v0.1-mlx \
    --data paths/to/safety_data.jsonl \
    --rank 8 \
    --alpha 16 \
    --output adapters/safety_sidecar
```

### 3. `mc-dynamics`: Training Dynamics
Analyze the "loss landscape" and entropy during training.

```bash
# Analyze a training run's gradient smoothness
mc-dynamics analyze-gradients --run-id <run_id>
```

## Next Steps

- Read about [High Dimensional Geometry](geometry/manifold_stitching.md).
- Explore the [Architecture](architecture.md).
