# ModelCypher Examples

This directory contains example scripts demonstrating common ModelCypher workflows.

## Prerequisites

- Python 3.11+
- ModelCypher installed (`poetry install`)
- macOS with Apple Silicon for MLX backend (most examples)
- Local model weights (download from Hugging Face mlx-community)

## Examples

### 01. Basic Geometry Probe

Probe a model to inspect its geometric properties and layer structure.

```bash
python examples/01_basic_geometry_probe.py /path/to/model
```

**What it does:**
- Loads model weights
- Analyzes layer structure
- Reports parameter counts and hidden dimensions
- Computes geometric metrics (intrinsic dimension, effective rank)

### 02. Safety Audit

Run safety probes against an adapter to detect potential risks.

```bash
python examples/02_safety_audit.py /path/to/adapter.safetensors
```

**What it does:**
- Static analysis for threat indicators
- Entropy baseline verification
- Pattern analysis for distress signals

### 03. Adapter Inspection

Inspect LoRA adapters without blending or interpolation.

```bash
python examples/03_adapter_inspection.py /path/to/adapter_dir
```

**What it does:**
- Reports adapter rank, alpha, sparsity, and layer count
- Outputs raw measurements only

### 04. Entropy Analysis

Analyze entropy patterns in model outputs using thermodynamic metrics.

```bash
python examples/04_entropy_analysis.py /path/to/model --prompt "Your prompt here"
```

**What it does:**
- Measures entropy, temperature, and free energy
- Detects phase transitions (ridges)
- Provides interpretable results

### 05. Model Merge

Merge two models using geometric alignment.

```bash
python examples/05_model_merge.py base_model model_a model_b -o merged_output
```

**What it does:**
- Aligns model representations using Procrustes analysis
- Applies spectral regularization
- Produces a merged model preserving capabilities from both sources

## Getting Models

Download MLX-compatible models from Hugging Face:

```bash
# Using huggingface-cli
huggingface-cli download mlx-community/Qwen2.5-0.5B-Instruct-bf16 --local-dir ./models/qwen2.5-0.5b

# Or use the ModelCypher CLI to fetch a model
mc model fetch mlx-community/Qwen2.5-0.5B-Instruct-bf16
```

## Tips

1. **Start small**: Use smaller models (0.5B-3B) for faster iteration
2. **Check memory**: Run `mc system memory` before large operations
3. **Use the CLI**: Many examples have CLI equivalents (`mc geometry spatial probe-model`)
4. **Read the output**: Raw measurements are returned; you decide meaning based on context
