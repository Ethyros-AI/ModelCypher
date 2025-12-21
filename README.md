# ModelCypher

ModelCypher is a Python port of TrainingCypher's CLI and MCP tooling with core training, merging, and geometry engines. It uses MLX on macOS and keeps domain logic backend-agnostic so CUDA backends can be swapped in later.

ModelCypher focuses on high-dimensional geometry for training health, alignment drift, and model comparison. The goal is to make those signals explainable to AI agents and the humans they support.

Key capabilities include:
- **Geometry**: Metaphor convergence, manifold clustering, and dimensionality estimation.
- **Safety**: Circuit breakers, regex content filters, and intervention execution.
- **Training Dynamics**: Gradient smoothness, idle scheduling, and regime state detection.
- **Semantics**: Compositional probes and topological fingerprinting.

## Why geometry?

- Training moves a model through a high-dimensional space. Geometry metrics tell you whether that path is stable or risky.
- Distances, angles, and curvature can reveal drift before it shows up in loss curves.
- The CLI and MCP outputs include interpretation strings so agents can summarize results safely.

## Docs (start here)

- **[👉 START HERE 👈](docs/START-HERE.md)** - **The Master Index.** Everything starts here.
- **[Glossary](docs/GLOSSARY.md)** - Shared vocabulary for Humans and AI.
- **[Getting Started](docs/getting_started.md)** - Installation, setup, and key commands (`mc-train`, `mc-inspect`).
- **[Architecture](docs/architecture.md)** - Understanding the Hexagonal Architecture and core domains.
- **[Geometry Guide](docs/geometry/manifold_stitching.md)** - Deep dive into Manifold Stitching and Intersection Maps.
- **[AI Assistant Guide](docs/AI-ASSISTANT-GUIDE.md)** - How agents should use these tools.
- **[CLI Reference](docs/CLI-REFERENCE.md)** - Full command documentation.
- **[Security](docs/security.md)** - Policy on secrets and safe tensors.
- **[Contributing](CONTRIBUTING.md)** - How to help us build the future of geometric AI.

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
mc-inspect --help

# Scan a model's geometric profile
mc-inspect scan --model mlx-community/Llama-2-7b-chat-mlx --output json

# Train a geometric safety adapter
mc-train lora \
    --model mlx-community/Mistral-7B-v0.1-mlx \
    --data data/safety.jsonl \
    --rank 8 \
    --alpha 16 \
    --output adapters/safety_sidecar
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
