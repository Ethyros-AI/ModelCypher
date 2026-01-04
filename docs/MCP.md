# ModelCypher MCP Server Documentation

The MCP server exposes system, model, training, inference, geometry, safety/entropy,
agent, adapter, task, thermo, and evaluation tools. Tools accept data paths and
return raw measurements; geometry metrics operate on point clouds or model paths.

## Run

```bash
poetry run modelcypher-mcp
```

## Tools

See `docs/MCP-TOOLS-CATALOG.md` for the full tool list and signatures.

## Geometry Input Format

Point clouds are JSON arrays of arrays. Intrinsic-dimension also accepts a JSON object
of activation vectors (values are treated as points in sorted key order).
