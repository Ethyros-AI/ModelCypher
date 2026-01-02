# CLI Reference

ModelCypher CLI. Output is JSON to stdout; diagnostics go to stderr.

## Model Merging

The primary operation. Takes knowledge from source and adds it to target via null-space projection.

```bash
mc merge -s SOURCE -t TARGET -o OUTPUT -d DOMAINS

# Full example
mc merge \
  --source /path/to/qwen \
  --target /path/to/smol \
  --output-dir /path/to/merged \
  --transplant-domains mathematical,logical,spatial
```

**Options:**
- `-s, --source`: Path to source model (knowledge donor)
- `-t, --target`: Path to target model (receives knowledge)
- `-o, --output-dir`: Output directory for merged model
- `-d, --transplant-domains`: Comma-separated domains (mathematical, logical, spatial, temporal, social, computational)
- `--skip-pre-analysis`: Skip interference analysis
- `-f, --output-file`: Save full result to JSON file

## Geometry Commands

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
