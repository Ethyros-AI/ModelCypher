# Start Here (ModelCypher)

If you are looking at this repository and thinking “what is this?”, this document is the shortest path to being productive.

ModelCypher is built for **AI agents** (CLI + MCP) that need to run training/analysis tasks and then **explain the results to humans** without hand-wavy math.

## What ModelCypher is

- A Python port of TrainingCypher’s CLI + MCP interface.
- A set of **“geometry” analyzers**: tools that treat training signals (weights/gradients/trajectories) as points in a high‑dimensional space and summarize what changed.
- A pragmatic on-device workflow: **macOS + MLX** is the default backend; other backends are designed to be pluggable.

If you only read one more document after this one, read `docs/GEOMETRY-GUIDE.md`.

## What “geometry” means here (plain language)

Training moves a model through a very large space (millions/billions of numbers). We can’t visualize that space directly, but we can compute:

- **Distance**: “How far did we move?” (drift, divergence, update magnitude)
- **Angle / direction**: “Did we rotate into a risky direction?” (alignment/persona/refusal directions)
- **Shape**: “Is the training surface sharp?” (flatness / curvature proxies)

These metrics are **heuristics**: they are useful early-warning signals, not proof of safety or quality.

## How an agent should use this repo (CLI workflow)

The intended agent loop is:

1. **Discover state**: `tc inventory --output json`
2. **Validate inputs**: `tc dataset validate <dataset.jsonl> --output json`
3. **Run the job**: `tc train start --model <id> --dataset <path> --output json`
4. **Monitor health**: `tc geometry training status --job <jobId> --output json`
5. **Explain results**: use `interpretation` / `recommendedAction` fields when present (see `docs/GEOMETRY-GUIDE.md`).

Tip: `--ai` mode defaults to JSON when stdout is not a TTY; combine with `--pretty` if a human is reading the raw output.

## How an agent should use this repo (MCP workflow)

MCP is the same capabilities as the CLI, but as tools:

1. Start the server: `poetry run modelcypher-mcp`
2. Call `tc_inventory` first.
3. Call specific tools (training, geometry, dataset editing) using the schemas in `docs/MCP.md`.

## “Where is the important stuff?” (repo tour)

- `src/modelcypher/cli/` — CLI commands and output formatting.
- `src/modelcypher/mcp/` — MCP server and tool definitions.
- `src/modelcypher/core/` — core logic (domain + use cases). This layer should not import adapters directly.
- `src/modelcypher/adapters/` — filesystem/inference/packaging integration points.
- `docs/CLI-REFERENCE.md` — authoritative command shapes and output fields.
- `docs/PARITY.md` — what is implemented vs stubbed.

## What is implemented vs aspirational?

ModelCypher tracks parity with TrainingCypher, but not everything is fully wired into end-to-end training yet.

If you’re unsure whether a command’s output is “real” or “stubbed”, check `docs/PARITY.md` and prefer commands that are marked DONE or explicitly described as returning computed metrics.

