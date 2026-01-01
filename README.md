# ModelCypher

ModelCypher measures high-dimensional geometry. It exposes geometry metrics for point clouds only.
There is no configuration surface; inputs are model/data paths.

## Install

```bash
poetry install
```

## CLI

```bash
mc geometry metrics gromov-wasserstein <source_file> <target_file>
mc geometry metrics intrinsic-dimension <points_file>
mc geometry metrics topological-fingerprint <points_file>
mc geometry metrics spectral-signature <points_file>
```

Point clouds are JSON arrays of arrays. Intrinsic-dimension also accepts a JSON object
of activation vectors (values are treated as points in sorted key order).

## MCP

```bash
poetry run modelcypher-mcp
```

Tools:
- `mc_geometry_gromov_wasserstein(source_file, target_file)`
- `mc_geometry_intrinsic_dimension(points_file)`
- `mc_geometry_topological_fingerprint(points_file)`
- `mc_geometry_spectral_signature(points_file)`
