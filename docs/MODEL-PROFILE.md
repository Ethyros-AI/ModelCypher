# Unified ModelProfile

The ModelProfile is the unified schema for capturing everything needed to understand a model's high-dimensional geometry. It provides a one-stop shop for stripping away the black box.

Related docs:
- [GEOMETRY-GUIDE.md](GEOMETRY-GUIDE.md) - What geometry tooling measures
- [GLOSSARY.md](GLOSSARY.md) - Shared vocabulary
- [CLI-REFERENCE.md](CLI-REFERENCE.md) - Command reference

## Philosophy [EMPIRICAL]

A ModelProfile answers: "What does this model look like on the inside?"

For any two models, the unified profile enables:
- **Alignment assessment**: How similar is their geometry?
- **Alignment planning**: What transformations are needed to merge?
- **Capability mapping**: Where does each model store what knowledge?
- **Transfer prediction**: What will survive a merge?

## Profile Sections

The profile is organized into sections that can be computed independently:

| Section | What It Captures | Cost |
|---------|------------------|------|
| `identity` | Model path, architecture, dimensions | Fast (config.json) |
| `geometry` | Per-layer curvature, intrinsic dimension | Medium |
| `topology` | Persistent homology, Betti numbers | Expensive |
| `semantic` | Semantic prime signatures | Requires inference |
| `density` | Knowledge density, concept regions | Expensive |
| `entropy` | Per-layer Shannon/Renyi entropy (LayerProfile fields) | Medium |

## Schema Overview

### ModelProfile

The top-level profile contains:

```python
@dataclass
class ModelProfile:
    # Identity
    model_path: str
    model_id: str            # Stable identity hash (config + weight metadata)
    model_family: str          # "llama", "qwen", "mistral", "smollm"
    architecture: str          # "llama", "qwen2", etc.
    parameter_count: int
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int

    # Per-layer geometry
    layer_profiles: list[LayerProfile]

    # Global curvature aggregates
    global_sectional_mean: float
    global_sectional_std: float
    global_ollivier_ricci_mean: float
    global_ollivier_ricci_std: float
    global_intrinsic_dimension_mean: float

    # Optional summaries
    topology_summary: TopologySummary | None
    semantic_signature: SemanticSignature | None
    density_summary: DensitySummary | None

    # Domain-specific metrics (optional)
    domain_metrics: dict[str, dict[str, float]]

    # Metadata
    profile_version: str       # Schema version
    computed_sections: list[str]
    computed_at: str           # ISO timestamp
    probe_corpus_hash: str
    probe_cache: dict[str, dict[str, Any]]  # Per-model probe cache index
    backend_used: str
    extraction_config: dict[str, Any]
```

### LayerProfile

Per-layer geometry:

```python
@dataclass
class LayerProfile:
    layer_idx: int
    layer_name: str

    # Curvature
    sectional_curvature_mean: float
    sectional_curvature_std: float
    ollivier_ricci_mean: float
    ollivier_ricci_std: float
    dominant_curvature_sign: str  # "positive", "negative", "flat", "mixed"

    # Intrinsic dimension
    intrinsic_dimension: float
    intrinsic_dimension_uncertainty: float
    intrinsic_dimension_method: str

    # Optional
    shannon_entropy: float | None
    renyi_entropy_alpha2: float | None
    betti_0: int | None
    betti_1: int | None
    max_persistence: float | None
    gradient_norm: float | None
    condition_number: float | None
    manifold_regions: list[ManifoldRegion]
```

### ProfileComparison

When comparing two profiles, the ProfileComparison reports raw geometric measurements:

```python
@dataclass
class ProfileComparison:
    source_path: str
    target_path: str

    # Structural
    architecture_match: bool
    hidden_dim_ratio: float
    layer_count_ratio: float
    vocab_overlap: float

    # Geometric diffs
    mean_sectional_curvature_diff: float
    mean_ollivier_ricci_diff: float
    mean_intrinsic_dimension_diff: float

    # Topology diffs
    topology_betti_diff: int | None
    topology_persistence_diff: float | None

    # Semantic measurement
    semantic_cosine_similarity: float | None

    # Layer correspondence
    layer_mapping: dict[int, int]  # source -> target
    layer_comparisons: list[LayerComparison]

    # Alignment flag
    aligned: bool

    # Baseline-relative z-scores (optional)
    sectional_z_score: float | None
    ricci_z_score: float | None
    dimension_z_score: float | None
    baseline_family: str | None
    baseline_model_count: int | None
```

## CLI Commands

In this repo, run commands as `poetry run mc ...` (instead of `mc ...`).

### Inspect a profile

```bash
# View summary
poetry run mc profile inspect profile.json

# View specific section
poetry run mc profile inspect profile.json --section geometry

# View specific layer
poetry run mc profile inspect profile.json --layer 5
```

### Compare two profiles

```bash
# Compare and show geometric diffs
poetry run mc profile compare source.json target.json

# Save comparison result
poetry run mc profile compare source.json target.json --save comparison.json

# Compare with baseline for z-score computation
poetry run mc profile compare source.json target.json --baseline qwen-baseline.json
```

When a baseline is provided, the comparison includes z-scores that show how many
standard deviations the differences are from typical within-family variation:

- `ricci_z_score`: How unusual the Ollivier-Ricci curvature difference is
- `dimension_z_score`: How unusual the intrinsic dimension difference is
- `sectional_z_score`: How unusual the sectional curvature difference is

Z-scores are raw measurements relative to the provided baseline; interpret them
in the context of your own family distributions.

### Import from existing formats

```bash
# Import from CurvatureProfile
poetry run mc profile import curvature.json --type curvature -o unified.json

# Import and merge into existing profile
poetry run mc profile import curvature.json --type curvature --base unified.json -o updated.json
```

Currently supported types: `curvature`.

### Merge partial profiles

```bash
# Combine multiple partial profiles
poetry run mc profile merge geometry.json topology.json semantic.json -o complete.json
```

### Generate a profile

```bash
# Identity-only profile (fast, no model loading)
poetry run mc profile generate /path/to/model -o profile.json --identity-only
```

### Update an existing profile

```bash
# Add identity data from a model directory
poetry run mc profile update profile.json --model /path/to/model -o updated.json
```

## Alignment Flags

`ProfileComparison.aligned` is a raw boolean flag indicating whether the
comparison met the internal alignment checks. Use baseline-relative z-scores
for context instead of hard thresholds.

## Family Baselines

Family baselines capture typical curvature distributions for a model family
(e.g., Qwen, LLaMA, Mistral). They enable z-score comparisons that express
differences relative to family variation, not arbitrary thresholds.

### Building a Baseline

```bash
# Generate per-model geometry profiles
poetry run mc analyze geodesic-profile --model /path/to/Qwen2-0.5B --prompt "test" -o qwen-0.5b.json
poetry run mc analyze geodesic-profile --model /path/to/Qwen2.5-3B --prompt "test" -o qwen-3b.json
poetry run mc analyze geodesic-profile --model /path/to/Qwen3-0.6B --prompt "test" -o qwen-0.6b.json
```

### Using a Baseline

```bash
# Compare with baseline for z-score computation
poetry run mc profile compare model_a.json model_b.json --baseline qwen-baseline.json
```

The comparison result includes:
- `baseline_family`: Which family the baseline represents
- `baseline_model_count`: How many models contributed to the baseline
- `ricci_z_score`, `dimension_z_score`, `sectional_z_score`: Z-scores for differences

### Existing Baselines

Pre-computed baselines are stored in experiment directories:
- `/path/to/experiments/curvature-profiles-YYYY-MM-DD/baselines/qwen-baseline.json`

## Importing Existing Profiles

The unified ModelProfile can import from existing profile formats in the codebase:

### CurvatureProfile

Files from `mc analyze geodesic-profile` or stored in experiments directories:

```bash
poetry run mc profile import /path/to/experiments/curvature-profiles-YYYY-MM-DD/SmolLM-360M.json \
    --type curvature -o smolm-unified.json
```

## Incremental Profile Building

Profiles can be built incrementally as sections are computed:

```bash
# Start with curvature
poetry run mc profile import curvature.json --type curvature -o partial.json

# Add density later
poetry run mc profile import density.json --type density --base partial.json -o updated.json

# Merge multiple profiles
poetry run mc profile merge partial.json topology.json -o complete.json
```

## Probe Cache (Per-Model)

When probes run, dense activation caches are stored under:

```
$HOME/.modelcypher/profiles/models/<model_id>/probe_cache/
```

The `probe_cache` field indexes available caches by
`<probe_mode>:<probe_corpus_hash>` and records raw metadata like probe counts,
stored spaces (hidden/intermediate/attention/embedding), and update time.
`probe_corpus_hash` is set to the corpus used for the current profile.

## JSON Schema

Profiles use JSON with schema versioning:

```json
{
  "_schema": "mc.model_profile.v1",
  "profile_version": "mc.model_profile.v1",
  "model_path": "/path/to/model",
  "model_family": "qwen",
  "architecture": "qwen2",
  "computed_sections": ["geometry"],
  "layer_profiles": [
    {
      "layer_idx": 0,
      "sectional_curvature_mean": 0.0,
      "ollivier_ricci_mean": 0.039,
      "intrinsic_dimension": 7.64
    }
  ]
}
```

## Example: Full Comparison Workflow

```bash
# 1. Import existing curvature profiles
poetry run mc profile import /path/to/experiments/SmolLM.json \
    --type curvature -o smolm.json
poetry run mc profile import /path/to/experiments/Qwen2.json \
    --type curvature -o qwen2.json

# 2. Compare for geometric diffs
poetry run mc profile compare smolm.json qwen2.json --save comparison.json

# 3. Check the results
poetry run mc --output json profile compare smolm.json qwen2.json | jq '{
  aligned: .aligned,
  mean_sectional_diff: .mean_sectional_curvature_diff,
  mean_ricci_diff: .mean_ollivier_ricci_diff,
  mean_dim_diff: .mean_intrinsic_dimension_diff
}'
```

## Files

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/geometry/model_profile.py` | Core ModelProfile schema |
| `src/modelcypher/core/domain/geometry/profile_comparison.py` | Comparison logic |
| `src/modelcypher/cli/commands/profile.py` | CLI commands |
| `tests/test_model_profile.py` | Profile tests |
| `tests/test_profile_comparison.py` | Comparison tests |
