# ModelCypher

> **The Toolkit for Geometric AI Alignment.**
> *Turn "Vibes" into Vectors.*

ModelCypher is a Python framework that accurately measures, tracks, and aligns the **Geometry** of Large Language Models. It assumes that "Knowledge" is a high-dimensional shape and that "Safety" is a boundary condition on that shape.

## ⚡️ The 30-Second Summary

**The Problem**: AI Alignment today is based on "Vibes" (Chat with the model -> It seems nice -> Ship it). This is unscientific and fragile.
**The Solution**: ModelCypher gives you **Geometric Metrics** (Entropy, Intrinsic Dimension, Topological Fingerprints) so you can:
1.  **Measure** if a model is actually safe, or just refusing superficially.
2.  **Monitor** if a model is "hallucinating" (high entropy) or "reasoning" (low entropy).
3.  **Merge** disjoint models (e.g., Llama + Qwen) by aligning their manifolds.

It runs on **macOS (MLX)** for local research and supports **CUDA** for scale.

## Why geometry?

- Training moves a model through a high-dimensional space. Geometry metrics tell you whether that path is stable or risky.
- Distances, angles, and curvature can reveal drift before it shows up in loss curves.
- The CLI and MCP outputs include interpretation strings so agents can summarize results safely.

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
