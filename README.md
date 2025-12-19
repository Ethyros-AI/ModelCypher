# ModelCypher

ModelCypher is a Python port of TrainingCypher's CLI and MCP tooling with core training, merging, and geometry engines. It uses MLX on macOS and keeps domain logic backend-agnostic so CUDA backends can be swapped in later.

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
```

## CLI Usage

The CLI is compatible with TrainingCypher's `tc` interface.

```bash
# Training
poetry run tc train start --model demo --dataset data.jsonl --output json
poetry run tc train status job-<id> --output json

# Models
poetry run tc model register demo --path ./models/demo --architecture custom --output json
poetry run tc model search "llama 3" --output json

# Geometry
poetry run tc geometry training status --job job-<id> --format summary --output json
poetry run tc geometry safety circuit-breaker --job job-<id> --output json
poetry run tc geometry adapter sparsity --checkpoint ./adapters/adapter.npz --output json

# Docs to dataset
poetry run tc doc convert --input ./docs --output ./dataset.jsonl --output json
```

See full command reference in `docs/CLI-REFERENCE.md`.

## MCP Server

Start the MCP server (stdio transport):

```bash
poetry run modelcypher-mcp
```

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
