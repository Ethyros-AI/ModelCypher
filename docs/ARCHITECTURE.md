# Architecture (ModelCypher)

This repo is intentionally structured so that:

- **Core geometry/training logic is reusable** across backends.
- **IO/backends are swappable** (MLX today; CUDA later).
- **CLI and MCP are thin shells** over the same use cases.

If you are an AI agent or a human trying to explain “why this output looks like this”, this is the map.

## High-level flow

```
CLI (tc) ─────┐
              ├──> Use cases (src/modelcypher/core/use_cases/*)
MCP tools ────┘
                       │
                       ▼
            Domain logic (src/modelcypher/core/domain/*)
                       │
                       ▼
      Adapters / backends (src/modelcypher/adapters/*, src/modelcypher/backends/*)
                       │
                       ▼
        Filesystem + MLX + (optional) network calls
```

## “Core” vs “adapters” (why it matters)

Most of the math lives in `src/modelcypher/core/domain`. This code should be:

- deterministic (same input → same output),
- easy to unit test,
- not coupled to filesystem, subprocesses, or network calls.

Adapters live under `src/modelcypher/adapters` and handle:

- local storage layout (jobs/checkpoints),
- packaging formats (e.g., ASIF),
- inference integration.

This split is how ModelCypher stays explainable: outputs are mostly the product of pure functions + explicit inputs.

## Interfaces exposed to agents

### CLI

- Entry point: `src/modelcypher/cli/app.py`
- Output formatting: `src/modelcypher/cli/output.py`
- AI mode detection: `src/modelcypher/cli/context.py`

CLI outputs are designed to be machine-readable (`--output json` or `--ai`) and then summarized for humans using the interpretation guidance in `docs/GEOMETRY-GUIDE.md`.

### MCP

- Entry point: `src/modelcypher/mcp/server.py`
- Tool profiles: selected by `TC_MCP_PROFILE`

The MCP server exposes the same use cases as tools, with JSON schemas and annotations (`readOnlyHint`, `destructiveHint`, etc.) to help AI clients decide what is safe to call.

## Backend notes (MLX)

On macOS, ModelCypher uses MLX. MLX has a few “gotchas” that affect correctness and explainability:

- **Evaluation is required** after certain operations (lazy execution).
- **Weight layout matters**; many operations assume a `[out, in]` layout.

When adding new geometry operations, keep backend assumptions explicit and add tests where reasonable.

