# ModelCypher

ModelCypher is a Python toolkit for **geometric analysis of large language models**. It bridges the gap between theoretical frameworks (Linear Representation Hypothesis, Semantic Entropy) and practical engineering by providing reproducible diagnostics for representation structure, safety, and cross-model alignment.

This repository implements methodology from 37 foundational papers (see `docs/references/BIBLIOGRAPHY.md`) and provides a comprehensive suite of CLI tools and Python modules for measuring:

- **Representation Geometry**: Centered Kernel Alignment (CKA), topological fingerprints, and intrinsic dimension.
- **Entropy Dynamics**: Thermodynamic profiling of prompt sensitivity and base-adapter divergence ($\Delta H$).
- **Model Merging**: Cross-architecture transfer via anchor-locked Procrustes alignment.

> **Status**: Active Research Toolkit. Implements 222 domain modules with 1,116 passing tests.
3.  **Experiment** with model/adaptor merges (e.g., Llama + Qwen) and quantify retention vs drift with explicit diagnostics (benchmark harness is in-progress; see `docs/PARITY.md`).

It runs on **macOS (MLX)** for local research and supports **CUDA** for scale.

## Key Capabilities

1.  **Safety as Geometry**: Measure refusal dynamics as trajectories in representation space rather than relying on chat-based red teaming.
2.  **Thermodynamic Monitoring**: Track entropy ($\Delta H$) and intrinsic dimension to detect hallucinations and safety boundary violations.
3.  **Cross-Architecture Transfer**: Align and merge adapters between disjoint model families (e.g., Qwen $\to$ Llama) using geometric stitching.

All capabilities are grounded in falsifiable metrics. See [**Papers**](papers/README.md) for methodology.

## Docs (start here)

- **[👉 START HERE 👈](docs/START-HERE.md)** - **The Master Index.** Everything starts here.
- **[Glossary](docs/GLOSSARY.md)** - Shared vocabulary for Humans and AI.
- **[Getting Started](docs/getting_started.md)** - Installation, setup, and key commands (`mc train`, `mc model`, `mc geometry`).
- **[Architecture](docs/ARCHITECTURE.md)** - Understanding the Hexagonal Architecture and core domains.
- **[Geometry Guide](docs/GEOMETRY-GUIDE.md)** - How to interpret geometry outputs safely.
- **[AI Assistant Guide](docs/AI-ASSISTANT-GUIDE.md)** - How agents should use these tools.
- **[CLI Reference](docs/CLI-REFERENCE.md)** - Full command documentation.
- **[Security](docs/security.md)** - Policy on secrets and safe tensors.
- **[Contributing](CONTRIBUTING.md)** - How to help us build the future of geometric AI.
  - Implementation status: see **[Parity](docs/PARITY.md)**.

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
# Verify installation
mc --help

# (Optional) Fetch a model from Hugging Face (requires network + HF_TOKEN for gated repos)
mc model fetch mlx-community/Llama-2-7b-chat-mlx --auto-register

# Probe a local model directory
mc model probe ./models/Llama-2-7b-chat-mlx --output json

# Train a LoRA adapter ("sidecar"-style)
mc train start \
    --model ./models/Mistral-7B-v0.1-mlx \
    --dataset data/safety.jsonl \
    --lora-rank 8 \
    --lora-alpha 16 \
    --out adapters/safety_sidecar
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

## Backends

ModelCypher uses **MLX** on macOS by default for unified memory efficiency. A CUDA backend stub exists providing a path for future Linux support.

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
  url = {https://github.com/anon57396/ModelCypher}
}
```
