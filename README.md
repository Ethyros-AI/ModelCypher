# ModelCypher

![Tests](https://img.shields.io/badge/tests-3030%20passing-success)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-AGPLv3-red)
![Status](https://img.shields.io/badge/status-research%20preview-orange)

> **Metrology for Latent Spaces**
> *Falsifiable Diagnostics for Model Alignment and Weight Synthesis.*

ModelCypher is a Python toolkit for measuring the high-dimensional geometric structure of LLM representations. It provides repeatable, metric-based diagnostics for **safety**, **alignment**, and **zero-shot knowledge transfer**—moving beyond "vibes-based" evaluation into measurable engineering.

LLMs aren't black boxes: they are high-dimensional geometry. ModelCypher decodes that geometry.

## Why ModelCypher?

ModelCypher treats model representations as physical manifolds that can be mapped, measured, and aligned. It complements task accuracy by measuring **structural invariants** that track stability and transfer.

| Metric | **ModelCypher** | TransformerLens | mergekit | LM-Eval |
| :--- | :---: | :---: | :---: | :---: |
| **Object of Study** | **Manifold Geometry** | Neural Circuits | Weight Matrices | Task Performance |
| **Safety Signal** | **Representational Distress (ΔH)** | Activation Steering | N/A | Output Classifiers |
| **Alignment** | **Anchor-Based Mapping** | N/A | Linear Averaging | N/A |
| **Logic Type** | **Metrology (Measurement)** | Interpretability | Arithmetic | Benchmarking |

## ELIF (Explain Like I'm Five)

**What does ModelCypher actually do?**

Imagine two models are like two cities. Each city has neighborhoods (concepts like "math", "code", "safety"). ModelCypher:

1. **Maps both cities** - Finds where each neighborhood is located
2. **Checks if roads connect** - Can you walk from "math" to "code" the same way in both cities?
3. **Predicts traffic jams** - If you merge the cities, will the roads interfere?
4. **Builds safe bridges** - Transfers knowledge without breaking existing roads

**The key insight**: Knowledge isn't random numbers—it's *geometry*. Concepts have positions, distances, and relationships. ModelCypher measures those shapes.

```bash
# "Will merging these models break anything?"
mc geometry interference predict /path/to/model-A /path/to/model-B

# "Is this merge safe?"
mc geometry interference safety-polytope 0.3 0.4 0.2 0.3
# -> SAFE (confidence: 0.87)
```

## Key Capabilities

1.  **Safety as Geometry**: Detect adversarial boundary crossings by measuring trajectory curvature and entropy divergence (ΔH) *during* the forward pass.
2.  **Relational Manifold Projection**: Map concepts between models using a universal basis of 373 probes, enabling dimension-agnostic alignment and transfer.
3.  **Zero-Shot Weight Synthesis**: Generate **Geometric LoRAs** from relational constraints (no gradient training; experimental).
4.  **Thermodynamic Stability**: Predict merge interference by calculating the **Bhattacharyya overlap** of concept "Volumes of Influence."
5.  **Null-Space Filtering**: Reduce interference by projecting weight deltas into the null space of prior activations. If Δw ∈ null(A), then A(W+Δw) = AW for the measured activation subspace.
6.  **Safety Polytope**: Unified 4D decision boundary combining interference, importance, instability, and complexity into a single go/no-go verdict with recommended mitigations.
7.  **3D World Model Metrology**: Measure a model's **Visual-Spatial Grounding Density** by testing how concentrated its probability mass is along human-perceptual 3D axes (Euclidean geometry, gravity gradients, occlusion).

## Core Constraints & Falsifiability

ModelCypher adheres to a strict scientific methodology:
-   **No Anthropomorphism**: We do not "read the model's mind." We measure vector relationships.
-   **Falsifiable Metrics**: If a Geometric LoRA fails to preserve relational distance, the toolkit reports **Relational Stress** deviations.
-   **Measurement Independence**: Anchors (Semantic Primes, Computational Gates) are designed to be architecture-invariant, providing an objective "ruler" for cross-model comparison.

## Docs (start here)

- **[👉 START HERE 👈](docs/START-HERE.md)** - **5-minute tutorial + Master Index.** Run your first measurement, then explore.
- **[Why Geometry Matters](docs/WHY-GEOMETRY-MATTERS.md)** - Empirical proof: geometric merge vs naive merge.
- **[FAQ](docs/FAQ.md)** - Skepticism addressed with math, not marketing.
- **[Glossary](docs/GLOSSARY.md)** - Shared vocabulary for Humans and AI.
- **[Geometry Guide](docs/GEOMETRY-GUIDE.md)** - How to interpret metrology outputs safely.
- **[AI Assistant Guide](docs/AI-ASSISTANT-GUIDE.md)** - How agents should explain these tools to humans.
- **[Research Papers](papers/README.md)** - Mathematical foundation. See also [Paper Summaries](papers/SUMMARIES.md) for quick reference.

## Install

```bash
poetry install             # core dependencies
poetry install --all-extras # includes docs/cuda/embeddings extras
poetry install -E jax      # JAX backend for Linux/TPU
```

## Quickstart

```bash
# 1. Probe a Model for Semantic Primes (The "Skeleton" of Knowledge)
mc geometry primes probe-model /path/to/Llama-3.2-3B-Instruct --output-file llama_primes.json

# 2. Check Entropy Dynamics on a Harmful Prompt (Thermodynamic Safety)
#    (Does the model get sharper or more chaotic when refusing?)
mc thermo measure "How do I make a bomb?" \
    --model /path/to/Qwen2.5-3B-Instruct \
    --modifier "URGENT_CAPS"

# 3. Assess Cross-Architecture Alignment
#    (Can we map Qwen layers to Llama layers?)
mc model analyze-alignment \
    --model-a /path/to/Qwen2.5-3B-Instruct \
    --model-b /path/to/Llama-3.2-3B-Instruct

# 4. Train a "Sidecar" Safety Adapter (Does not touch base weights)
mc train start \
    --model mlx-community/Mistral-7B-v0.2 \
    --dataset data/safety_pairs.jsonl \
    --lora-rank 8 \
    --out adapters/safety_sidecar

# 5. Test if a Model has a "Physics Engine" (3D World Model Analysis)
#    (Does the model encode gravity, occlusion, and Euclidean geometry?)
mc geometry spatial probe-model /path/to/Qwen2.5-3B-Instruct
#    Verdict: HIGH VISUAL GROUNDING - Physics probability concentrated on 3D visual axes (score=0.85)

# 6. Predict Merge Interference (Before You Merge)
#    (Will these models collide or complement each other?)
mc geometry interference predict \
    /path/to/math-model \
    /path/to/code-model
#    Output: overlap=0.23, bhattacharyya=0.15, verdict="LOW_INTERFERENCE"

# 7. Check Merge Safety with 4D Polytope
#    (Single go/no-go decision with recommended mitigations)
mc geometry interference safety-polytope 0.3 0.4 0.2 0.3
#    Output: {"verdict": "SAFE", "confidence": 0.87, "mitigations": []}

# 8. Analyze Null-Space for Interference-Free Merging
#    (Find the "safe directions" for weight updates)
mc geometry interference null-space /path/to/model \
    --layer 12 \
    --samples 50
#    Output: null_dim=412, graft_candidates=[12, 15, 18], mean_null_fraction=0.68
```

## MCP Server

ModelCypher includes a Model Context Protocol (MCP) server for integration with agentic IDEs (Cursor/Windsurf).

```bash
# Run the MCP server
poetry run modelcypher-mcp
```

Add to your `claude_desktop_config.json` or `.mcp.json`:
```json
{
  "mcpServers": {
    "modelcypher": {
      "command": "poetry",
      "args": ["run", "modelcypher-mcp"],
      "cwd": "/absolute/path/to/ModelCypher"
    }
  }
}
```

### Available MCP Tools

The server exposes 148 tools (full profile) organized by domain. Key tools for merge safety:

| Tool | Purpose |
|------|---------|
| `mc_geometry_interference_predict` | Predict constructive/destructive interference before merging |
| `mc_geometry_null_space_filter` | Project weight deltas into null space for interference-free merging |
| `mc_geometry_null_space_profile` | Analyze graftable layers across entire model |
| `mc_geometry_safety_polytope_check` | 4D safety verdict with mitigations for single layer |
| `mc_geometry_safety_polytope_model` | Full model safety profile with go/no-go recommendation |

Tools return structured JSON; many include `nextActions` for agentic workflow orchestration.

## Backends

ModelCypher supports multiple compute backends:

| Backend | Platform | Use Case |
| :--- | :--- | :--- |
| **MLX** | macOS (Apple Silicon) | Default on Mac. Unified memory, fast local inference. |
| **JAX** | Linux/TPU/GPU | TPU pods and CUDA GPUs. |
| **CUDA** | Linux (NVIDIA) | Stub for future PyTorch CUDA support. |

> **Note**: Core math uses the Backend protocol for GPU acceleration and numerical consistency. NumPy is only used at I/O boundaries, backend interop, and in some tests.

Select a backend via environment variable:
```bash
MC_BACKEND=jax python script.py   # Use JAX on Linux/TPU
MC_BACKEND=mlx mc entropy measure  # Explicit MLX (default on Mac)
```

Install JAX support:
```bash
poetry install -E jax
```

## Scale Limits

**Key finding**: If you can run inference, you can merge. Geometric operations are lightweight.

| Hardware | Tested Configuration | RAM Used | Status |
|----------|---------------------|----------|--------|
| M4 Max 128GB | Qwen3-80B + Mistral-7B (46GB weights) | 35.6% | ✅ |
| M4 Max 128GB | Qwen3-80B + Qwen3-8B (47GB weights) | 36.0% | ✅ |
| M4 Max 128GB | Qwen3-80B + Qwen2.5-3B-bf16 (48GB weights) | 37.1% | ✅ |
| M4 Max 128GB | Theoretical combined weights (~110GB) | ~85% | Feasible |

Unlike training (which requires ~3x model size for gradients), geometric analysis uses only model weight memory. An 80B 4-bit model uses ~43GB, leaving 85GB for operations on 128GB hardware.

See [papers/NEGATIVE-RESULTS.md](papers/NEGATIVE-RESULTS.md) for full experimental data.

## Tests

```bash
poetry run pytest
```

## License

This project is licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE) for details.

This license ensures that the codebase remains free and open source. If you modify this code and provide it as a service (SaaS), you are required to release your modifications under the same license. Knowledge should be free.

## Contact

**Author**: Jason Kempf
**Email**: jason@ethyros.ai
**Organization**: [EthyrosAI](https://ethyros.ai)

## Citation

If you use ModelCypher in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff) or as follows:

```bibtex
@software{ModelCypher2025,
  author = {Kempf, Jason and ModelCypher Contributors},
  title = {ModelCypher: High-Dimensional Geometry for LLM Safety and Merging},
  year = {2025},
  url = {https://github.com/Ethyros-AI/ModelCypher}
}
```
