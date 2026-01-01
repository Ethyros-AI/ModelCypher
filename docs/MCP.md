# ModelCypher MCP Server Documentation

The MCP server exposes geometry metrics only. There is no configuration or tool profiling.
Tools accept data paths and return raw measurements.

## Run

```bash
poetry run modelcypher-mcp
```

## Tools

- `mc_geometry_gromov_wasserstein(source_file, target_file)`
- `mc_geometry_intrinsic_dimension(points_file)`
- `mc_geometry_topological_fingerprint(points_file)`
- `mc_geometry_spectral_signature(points_file)`

## Input Format

Point clouds are JSON arrays of arrays. Intrinsic-dimension also accepts a JSON object
of activation vectors (values are treated as points in sorted key order).
