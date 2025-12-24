# ModelCypher

![Tests](https://img.shields.io/badge/tests-2828%20passing-success)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20preview-orange)

> **Metrology for Latent Spaces**
> *Falsifiable Diagnostics for Model Alignment and Weight Synthesis.*

ModelCypher is a Python toolkit for measuring the high-dimensional geometric structure of LLM representations. It provides reproducible, metric-based diagnostics for **safety**, **alignment**, and **zero-shot knowledge transfer**—moving beyond "vibes-based" evaluation into deterministic engineering.

```mermaid
graph LR
    subgraph "The Problem (Vibes)"
        A[Prompt] -->|?| B[Black Box]
        B -->|?| C[Output]
        style B fill:#f9f,stroke:#333
    end

    subgraph "The ModelCypher Way (Metrology)"
        D[Prompt] --> E[Trajectory Analysis]
        E -->|Entropy $\Delta H$| F[Boundary Monitor]
        E -->|Sectional Curvature $K$| G[Stability Diagnostic]
        E -->|Relational Footprint| H[Manifold Map]
        F & G & H --> I[Falsifiable Signal]
    end
```

## Why ModelCypher?

ModelCypher treats model representations as physical manifolds that can be mapped, measured, and aligned. Unlike standard evaluation suites that measure *task accuracy*, ModelCypher measures the **structural invariants** that enable that accuracy.

| Metric | **ModelCypher** | TransformerLens | mergekit | LM-Eval |
| :--- | :---: | :---: | :---: | :---: |
| **Object of Study** | **Manifold Geometry** | Neural Circuits | Weight Matrices | Task Performance |
| **Safety Signal** | **Representational Distress ($\Delta H$)** | Activation Steering | N/A | Output Classifiers |
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
# → SAFE (confidence: 0.87)
```

## Key Capabilities

1.  **Safety as Geometry**: Detect adversarial boundary crossings by measuring trajectory curvature and entropy divergence ($\Delta H$) *during* the forward pass.
2.  **Relational Manifold Projection**: Map concepts from a Source Model to a Target Model using a universal basis of 237 anchors, enabling 1:1 knowledge transfer.
3.  **Zero-Shot Weight Synthesis**: Generate **Geometric LoRAs** that "print" new relational footprints into a model's latent space without a retraining run.
4.  **Thermodynamic Stability**: Predict merge interference by calculating the **Bhattacharyya overlap** of concept "Volumes of Influence."
5.  **Null-Space Filtering**: Guarantee interference-free merging by projecting weight deltas into the null space of prior activations. Mathematical proof: if Δw ∈ null(A), then A(W+Δw) = AW.
6.  **Safety Polytope**: Unified 4D decision boundary combining interference, importance, instability, and complexity into a single go/no-go verdict with recommended mitigations.
7.  **3D World Model Metrology**: Measure a model's **Visual-Spatial Grounding Density** by testing how concentrated its probability mass is along human-perceptual 3D axes (Euclidean geometry, gravity gradients, occlusion).

## Core Constraints & Falsifiability

ModelCypher adheres to a strict scientific methodology:
-   **No Anthropomorphism**: We do not "read the model's mind." We measure vector relationships.
-   **Falsifiable Metrics**: If a Geometric LoRA fails to preserve relational distance, the toolkit reports a **Relational Stress** error.
-   **Measurement Independence**: Our anchors (Semantic Primes, Computational Gates) are architecture-invariant, providing an objective "ruler" for cross-model comparison.

## Docs (start here)

- **[👉 START HERE 👈](docs/START-HERE.md)** - **The Master Index.** Pathways for Tinkerers, Researchers, and Auditors.
- **[Glossary](docs/GLOSSARY.md)** - Shared vocabulary for Humans and AI (Analogy-to-Math).
- **[Geometry Guide](docs/GEOMETRY-GUIDE.md)** - How to interpret metrology outputs safely.
- **[AI Assistant Guide](docs/AI-ASSISTANT-GUIDE.md)** - How agents should explain these tools to humans.
- **[Research Papers](papers/README.md)** - The mathematical foundation (1936-2025).

## Install

We recommend using `uv` for fast, reliable dependency management.

```bash
uv sync
```

Alternatively, standard pip works:

```bash
pip install -e .
```

## Quickstart

```bash
# 1. Probe a Model for Semantic Primes (The "Skeleton" of Knowledge)
mc geometry primes probe --model mlx-community/Llama-3.2-3B-Instruct --output llama_primes.json

# 2. Check Entropy Dynamics on a Harmful Prompt (Thermodynamic Safety)
#    (Does the model get sharper or more chaotic when refusing?)
mc entropy measure \
    --model mlx-community/Qwen2.5-3B-Instruct \
    --prompt "How do I make a bomb?" \
    --modifier "URGENT_CAPS"

# 3. Assess Cross-Architecture Alignment
#    (Can we map Qwen layers to Llama layers?)
mc model analyze-alignment \
    --source mlx-community/Qwen2.5-3B-Instruct \
    --target mlx-community/Llama-3.2-3B-Instruct

# 4. Train a "Sidecar" Safety Adapter (Does not touch base weights)
mc train start \
    --model mlx-community/Mistral-7B-v0.2 \
    --dataset data/safety_pairs.jsonl \
    --lora-rank 8 \
    --out adapters/safety_sidecar

# 5. Test if a Model has a "Physics Engine" (3D World Model Analysis)
#    (Does the model encode gravity, occlusion, and Euclidean geometry?)
mc geometry spatial probe-model /Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct
#    Verdict: HIGH VISUAL GROUNDING - Physics probability concentrated on 3D visual axes (score=0.85)

# 6. Predict Merge Interference (Before You Merge)
#    (Will these models collide or complement each other?)
mc geometry interference predict \
    --source /path/to/math-model \
    --target /path/to/code-model
#    Output: overlap=0.23, bhattacharyya=0.15, verdict="LOW_INTERFERENCE"

# 7. Check Merge Safety with 4D Polytope
#    (Single go/no-go decision with recommended mitigations)
mc geometry interference safety-polytope 0.3 0.4 0.2 0.3
#    Output: {"verdict": "SAFE", "confidence": 0.87, "mitigations": []}

# 8. Analyze Null-Space for Interference-Free Merging
#    (Find the "safe directions" for weight updates)
mc geometry interference null-space \
    --model /path/to/model \
    --layer 12 \
    --samples 50
#    Output: null_dim=412, graft_candidates=[12, 15, 18], mean_null_fraction=0.68
```

## MCP Server

ModelCypher includes a Model Context Protocol (MCP) server for integration with agentic IDEs (Cursor/Windsurf).

```bash
# Run the MCP server
uv run modelcypher-mcp
```

Add to your `claude_desktop_config.json` or `.mcp.json`:
```json
{
  "mcpServers": {
    "modelcypher": {
      "command": "uv",
      "args": ["run", "modelcypher-mcp"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/ModelCypher/src"
      }
    }
  }
}
```

### Available MCP Tools

The server exposes 150+ tools organized by domain. Key tools for merge safety:

| Tool | Purpose |
|------|---------|
| `mc_geometry_interference_predict` | Predict constructive/destructive interference before merging |
| `mc_geometry_null_space_filter` | Project weight deltas into null space for interference-free merging |
| `mc_geometry_null_space_profile` | Analyze graftable layers across entire model |
| `mc_geometry_safety_polytope_check` | 4D safety verdict with mitigations for single layer |
| `mc_geometry_safety_polytope_model` | Full model safety profile with go/no-go recommendation |

All tools return structured JSON with `nextActions` for agentic workflow orchestration.

## Backends

ModelCypher supports multiple compute backends:

| Backend | Platform | Use Case |
| :--- | :--- | :--- |
| **MLX** | macOS (Apple Silicon) | Default on Mac. Unified memory, fast local inference. |
| **JAX** | Linux/TPU/GPU | Google TPU pods, Anthropic infrastructure, CUDA GPUs. |
| **CUDA** | Linux (NVIDIA) | Stub for future PyTorch CUDA support. |
| **NumPy** | Any | Testing and CI (no GPU required). |

Select a backend via environment variable:
```bash
MC_BACKEND=jax python script.py   # Use JAX on Linux/TPU
MC_BACKEND=mlx mc entropy measure  # Explicit MLX (default on Mac)
```

Install JAX support:
```bash
uv sync --extra jax
```

## Tests

```bash
uv run pytest
```

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Citation

If you use ModelCypher in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff) or as follows:

```bibtex
@software{ModelCypher2025,
  author = {Kempf, Jason and ModelCypher Contributors},
  title = {ModelCypher: High-Dimensional Geometry for LLM Safety and Merging},
  year = {2025},
  url = {https://github.com/EthyrosAI/ModelCypher}
}
```
