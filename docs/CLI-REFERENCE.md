# CLI Reference

ModelCypher CLI exposes geometry metrics only. Output is JSON to stdout; diagnostics go to stderr.

## Commands

```bash
mc geometry metrics gromov-wasserstein <source_file> <target_file>
mc geometry metrics intrinsic-dimension <points_file>
mc geometry metrics topological-fingerprint <points_file>
mc geometry metrics spectral-signature <points_file>
mc geometry density profile <model_dir>
mc geometry density diff <source_model_dir> <target_model_dir>
```

## Input Format

Point clouds are JSON arrays of arrays. Intrinsic-dimension also accepts a JSON object
of activation vectors (values are treated as points in sorted key order).
