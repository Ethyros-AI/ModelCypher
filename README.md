# ModelCypher

ModelCypher is a Python port of TrainingCypher's CLI and MCP tooling with core training, merging, and geometry engines. It uses MLX on macOS and keeps domain logic backend-agnostic so CUDA backends can be swapped in later.

ModelCypher focuses on high-dimensional geometry for training health, alignment drift, and model comparison. The goal is to make those signals explainable to AI agents and the humans they support.

## Why geometry?

- Training moves a model through a high-dimensional space. Geometry metrics tell you whether that path is stable or risky.
- Distances, angles, and curvature can reveal drift before it shows up in loss curves.
- The CLI and MCP outputs include interpretation strings so agents can summarize results safely.

## Docs (start here)

- `docs/START-HERE.md` - Executive summary + repo tour.
- `docs/AI-ASSISTANT-GUIDE.md` - How agents should call tools and explain outputs safely.
- `docs/GEOMETRY-GUIDE.md` - How to explain metrics and outputs in plain language.
- `docs/MATH-PRIMER.md` - The geometry intuition behind the metrics.
- `docs/ARCHITECTURE.md` - Codebase structure (core vs adapters, CLI vs MCP).
- `docs/CLI-REFERENCE.md` - Command shapes and output fields (authoritative).
- `docs/MCP.md` - MCP tools/resources and how to run the server.
- `docs/PARITY.md` - What is fully implemented vs stubbed.
- `KnowledgeasHighDimensionalGeometryInLLMs.md` - Foundational bibliography for “knowledge as geometry”.

## Install

```bash
poetry install
```

Optional extras for document conversion:

```bash
poetry install --extras docs
```

## Quickstart

```bash
# Inventory
poetry run tc inventory --output json

# Validate dataset
poetry run tc dataset validate ./data.jsonl --output json

# Start a small training job
poetry run tc train start --model demo --dataset ./data.jsonl --output json

# Geometry health snapshot
poetry run tc geometry training status --job job-<id> --output json
```

## CLI Usage

The CLI is compatible with TrainingCypher's `tc` interface.

```bash
# Training
poetry run tc train start --model demo --dataset data.jsonl --output json
poetry run tc train start --model demo --dataset data.jsonl --output json
poetry run tc train status job-<id> --output json

# Agents
poetry run tc agents primes signature --text "Hello world" --output json
poetry run tc agents diversion assess --expected "coding" --observed "poetry" --output json

# Geometry
poetry run tc geometry training status --job job-<id> --format summary --output json
poetry run tc geometry safety circuit-breaker --job job-<id> --output json
poetry run tc geometry adapter sparsity --checkpoint ./adapters/adapter.npz --output json
poetry run tc geometry transport merge --source ./modelA --target ./modelB --output json

# Docs to dataset
poetry run tc doc convert --input ./docs --output-path ./dataset.jsonl --output json
```

See full command reference in `docs/CLI-REFERENCE.md`.

## MCP Server

Start the MCP server (stdio transport):

```bash
poetry run modelcypher-mcp
```

The legacy `trainingcypher-mcp` entrypoint is also available for compatibility.

Example `.mcp.json`:

```json
{
  "mcpServers": {
    "modelcypher": {
      "command": "poetry",
      "args": ["run", "modelcypher-mcp"],
      "env": {
        "TC_MCP_PROFILE": "training"
      }
    }
  }
}
```

MCP tools and resources are documented in `docs/MCP.md`.

## Backends

ModelCypher uses MLX on macOS by default. A CUDA backend stub is provided in
`src/modelcypher/backends/cuda_backend.py` and can be enabled by installing
the optional `torch` extra and swapping the backend in adapters.

## Tests

```bash
poetry run pytest
```
