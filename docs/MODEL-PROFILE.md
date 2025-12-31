# Unified ModelProfile

The ModelProfile is the unified schema for capturing everything needed to understand a model's high-dimensional geometry. It provides a one-stop shop for stripping away the black box.

Related docs:
- [GEOMETRY-GUIDE.md](GEOMETRY-GUIDE.md) - What geometry tooling measures
- [GLOSSARY.md](GLOSSARY.md) - Shared vocabulary
- [CLI-REFERENCE.md](CLI-REFERENCE.md) - Command reference

## Philosophy

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
| `entropy` | Shannon/Renyi entropy per layer | Medium |

## Schema Overview

### ModelProfile

The top-level profile contains:

```python
@dataclass
class ModelProfile:
    # Identity
    model_path: str
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

    # Optional summaries
    topology_summary: TopologySummary | None
    semantic_signature: SemanticSignature | None
    density_summary: DensitySummary | None

    # Metadata
    profile_version: str       # Schema version
    computed_sections: list[str]
    computed_at: str           # ISO timestamp
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
    intrinsic_dimension_method: str

    # Optional
    shannon_entropy: float | None
    betti_0: int | None
    betti_1: int | None
    max_persistence: float | None
```

### ProfileComparison

When comparing two profiles, the ProfileComparison tells the alignment story:

```python
@dataclass
class ProfileComparison:
    source_path: str
    target_path: str

    # Structural
    architecture_match: bool
    hidden_dim_ratio: float
    layer_count_ratio: float

    # Geometric alignment (0-1 scale)
    curvature_alignment: float
    ricci_alignment: float
    dimension_alignment: float
    overall_alignment: float

    # Layer correspondence
    layer_mapping: dict[int, int]  # source -> target
    layer_comparisons: list[LayerComparison]
    critical_layers: list[int]     # High effort layers

    # Alignment metrics
    total_alignment_effort: float
    mean_alignment_effort: float
    max_alignment_effort: float
    recommended_strategy: str  # "procrustes", "projection_first", "curvature_flow"
```

## CLI Commands

### Inspect a profile

```bash
# View summary
mc profile inspect profile.json

# View specific section
mc profile inspect profile.json --section geometry

# View specific layer
mc profile inspect profile.json --layer 5
```

### Compare two profiles

```bash
# Compare and show alignment story
mc profile compare source.json target.json

# Save comparison result
mc profile compare source.json target.json --save comparison.json
```

### Import from existing formats

```bash
# Import from CurvatureProfile
mc profile import curvature.json --type curvature -o unified.json

# Import and merge into existing profile
mc profile import density.json --type density --base unified.json -o updated.json
```

### Merge partial profiles

```bash
# Combine multiple partial profiles
mc profile merge geometry.json topology.json semantic.json -o complete.json
```

## Alignment Scores

Alignment scores are in [0, 1] and computed via exponential decay of differences.
They are comparative metrics, not pass/fail thresholds. If you need thresholds,
derive them from baseline distributions for the model family.

The `recommended_strategy` field records which alignment heuristic was selected:

| Strategy | Heuristic trigger |
|----------|-------------------|
| `procrustes` | Similar dimensions, curvature signs match |
| `projection_first` | Dimension ratios > 1.5x difference |
| `curvature_flow` | Curvature sign mismatches or high effort |

## Importing Existing Profiles

The unified ModelProfile can import from existing profile formats in the codebase:

### CurvatureProfile

Files from `mc geometry curvature profile` or stored in experiments directories:

```bash
mc profile import /Volumes/CodeCypher/experiments/curvature-profiles-2025-12-31/SmolLM-360M.json \
    --type curvature -o smolm-unified.json
```

### Planned Import Types

- `density`: ModelDensityProfile from knowledge density analysis
- `topology`: TopologicalFingerprint from persistent homology
- `semantic`: SemanticPrimeSignature from semantic analysis

## Incremental Profile Building

Profiles can be built incrementally as sections are computed:

```bash
# Start with curvature
mc profile import curvature.json --type curvature -o partial.json

# Add density later
mc profile import density.json --type density --base partial.json -o updated.json

# Merge multiple profiles
mc profile merge partial.json topology.json -o complete.json
```

## JSON Schema

Profiles use JSON with schema versioning:

```json
{
  "$schema": "mc.model_profile.v1",
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
mc profile import /Volumes/CodeCypher/experiments/SmolLM.json \
    --type curvature -o smolm.json
mc profile import /Volumes/CodeCypher/experiments/Qwen2.json \
    --type curvature -o qwen2.json

# 2. Compare for alignment story
mc profile compare smolm.json qwen2.json --save comparison.json

# 3. Check the results
mc --output json profile compare smolm.json qwen2.json | jq '{
  alignment: .overall_alignment,
  strategy: .recommended_strategy,
  effort: .mean_alignment_effort
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
