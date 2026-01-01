# CLI Reference

ModelCypher CLI is standalone. Use `mc` (or `modelcypher`) for all commands.

## Output + AI Mode

- `stdout` is structured output (JSON/YAML/text).
- `stderr` is diagnostics (logs, progress).
- `--ai` forces JSON output and suppresses prompts/logs; `MC_AI_MODE=1` enables the same.
- `MC_NO_AI=1` disables AI mode.

## Global Options

- `--output {text,json,yaml}`
- `--ai`
- `--pretty`
- `--quiet`, `--very-quiet`
- `--yes`, `--no-prompt`
- `--trace-id <value>`
- `--log-level {trace,debug,info,warn,error}`

Environment variables:
- `MC_AI_MODE`, `MC_NO_AI`
- `MC_OUTPUT`

## Command Map

Primary workflows:
- `mc train` (start/preflight/status/pause/resume/cancel/export/logs)
- `mc job` (list/show/attach/delete)
- `mc checkpoint` (list/delete/export)
- `mc merge` (pipeline)
- `mc model` (list/register/delete/fetch/search/probe/validate-merge/validate-knowledge/analyze-alignment/vocab-compare)
- `mc profile` (generate/inspect/compare/update/import/merge)
- `mc doc` (convert/validate)
- `mc infer` (run/suite)
- `mc storage` (status/usage/cleanup)
- `mc inventory`, `mc system`

Research + diagnostics:
- `mc geometry` (path/training/safety/adapter/atlas/baseline/concept/cross-cultural/primes/crm/metrics/sparse/refusal/persona/manifold/refinement/invariant/emotion/merge-entropy/transfer/spatial/social/temporal/moral/waypoint/interference)
- `mc thermo` (analyze/path/path-integration/entropy/measure/detect/detect-batch/ridge-detect/phase/sweep/benchmark/parity)
- `mc entropy` (analyze/detect-distress/verify-baseline/window/conversation-track/dual-path/calibrate)
- `mc safety` (adapter-probe)
- `mc agent` (trace-import/trace-analyze/validate-action)
- `mc eval` (run/list/show)
- `mc compare` (run/list/show/checkpoints/baseline/score)
- `mc adapter` (inspect/project/wrap-mlx/smooth/merge)
- `mc calibration` (run/status/apply)
- `mc stability` (run/report)
- `mc agent-eval` (run/results)
- `mc dashboard` (metrics/export)
- `mc ensemble` (create/run/list/delete)
- `mc research` (sparse-region/afm)
- `mc help` (ask/completions/schema)
- `mc explain`
- `mc validate` (train), `mc estimate` (train)

## Geometry Atlas Commands
```bash
mc geometry atlas dimensionality <model_path> --layer <n>
mc geometry atlas dimensionality-study <model_path> --layer <n> [-l ...]
```

## Geometry Concept Commands
```bash
mc geometry concept detect "Text to analyze"
mc geometry concept detect "Prompt" --model <path>
mc geometry concept compare --text-a "First" --text-b "Second"
mc geometry concept compare --model-a <path> --model-b <path> --prompt "Test input"
```

## Geometry CRM Commands
```bash
mc geometry crm build --model <path> --output-path <path>
mc geometry crm compare --source <crm.json> --target <crm.json>
mc geometry crm delta-mask --source <crm.json> --target <crm.json> --output-path <mask.json>
```

## Geometry Metrics Commands
```bash
mc geometry metrics gromov-wasserstein <source_file> <target_file>
mc geometry metrics intrinsic-dimension <points_file>
mc geometry metrics topological-fingerprint <points_file>
mc geometry metrics spectral-signature <points_file>
mc geometry metrics dimension-constraint <points_file> --pad-dim <n>
```

## Geometry Sparse Commands
```bash
mc geometry sparse domains
mc geometry sparse locate <domain_stats.json> <baseline_stats.json> --domain <name> --use-dare-alignment
mc geometry sparse neurons --model <path> [--domain <name> | --prompts <prompts.json>]
```

### Sparse Locate Output Schema
```json
{
  "_schema": "mc.geometry.sparse_locate.v1",
  "domain": "math",
  "sparseLayers": [2, 3, 4],
  "skipLayers": [0],
  "sparsityThreshold": 0.3125,
  "layerSparsity": { "0": 0.0, "1": 0.12, "2": 0.41 },
  "dareAlignment": {
    "highDroppabilityLayers": [2, 4],
    "overlapWithSparse": 0.5
  }
}
```

### Spectral Signature Output Schema
```json
{
  "_schema": "mc.geometry.spectral_signature.v1",
  "eigenvalues": [0.0, 0.18, 0.62, 1.02],
  "eigenvalueCount": 4,
  "eigenvaluesTruncated": false,
  "heatTrace": [3.74, 2.11],
  "heatTimes": [0.1, 1.0],
  "spectralEntropy": 0.942,
  "algebraicConnectivity": 0.18,
  "componentCount": 1,
  "nodeCount": 4,
  "edgeCount": 3,
  "kNeighbors": 2,
  "kernelBandwidth": 0.75,
  "normalizedLaplacian": true,
  "connected": true
}
```

### Dimension Constraint Invariance Output Schema
```json
{
  "_schema": "mc.geometry.dimension_constraint_invariance.v1",
  "baseDimension": 2,
  "paddedDimension": 4,
  "sampleCount": 5,
  "kNeighbors": 3,
  "gramCka": 1.0,
  "geodesicDiff": {
    "meanAbs": 0.0,
    "maxAbs": 0.0
  },
  "spectral": {
    "eigenMeanAbsDiff": 0.0,
    "eigenMaxAbsDiff": 0.0,
    "spectralEntropyBase": 0.123,
    "spectralEntropyPadded": 0.123,
    "heatTraceBase": [4.89, 2.31],
    "heatTracePadded": [4.89, 2.31],
    "heatTimes": [0.1, 1.0]
  },
  "topology": {
    "bettiNumbersBase": { "0": 1, "1": 0 },
    "bettiNumbersPadded": { "0": 1, "1": 0 },
    "componentCountBase": 1,
    "componentCountPadded": 1,
    "cycleCountBase": 0,
    "cycleCountPadded": 0,
    "persistenceEntropyBase": 0.0,
    "persistenceEntropyPadded": 0.0,
    "maxPersistenceBase": 0.912,
    "maxPersistencePadded": 0.912
  }
}
```

### Dimension Constraint Example (Semantic Primes)
```bash
mc geometry primes probe-model ./model --output-file primes.json
mc geometry metrics dimension-constraint primes.json --pad-dim 4096 --output json
```
Use `primes.json` directly; the command consumes activation dict values as points.

### Geometry Validate Output Schema (Spectral + Dimension Constraint Excerpt)
```json
{
  "_schema": "mc.geometry.validation.v1",
  "spectralSignature": {
    "eigenvalueMin": 0.0,
    "eigenvalueMax": 2.0,
    "algebraicConnectivity": 0.0,
    "componentCount": 2,
    "heatTrace": [3.64, 2.27, 2.0],
    "heatTimes": [0.1, 1.0, 10.0],
    "connected": false,
    "passed": true
  },
  "spectralSignatureConnected": {
    "eigenvalueMin": 0.0,
    "eigenvalueMax": 2.0,
    "algebraicConnectivity": 0.4,
    "componentCount": 1,
    "heatTrace": [3.71, 2.19, 1.0],
    "heatTimes": [0.1, 1.0, 10.0],
    "connected": true,
    "passed": true
  },
  "dimensionConstraint": {
    "baseDimension": 2,
    "paddedDimension": 4,
    "sampleCount": 5,
    "kNeighbors": 3,
    "gramCka": 1.0,
    "geodesicDiff": { "meanAbs": 0.0, "maxAbs": 0.0 },
    "spectral": {
      "eigenMeanAbsDiff": 0.0,
      "eigenMaxAbsDiff": 0.0,
      "spectralEntropyBase": 0.123,
      "spectralEntropyPadded": 0.123,
      "heatTraceBase": [4.89, 2.31],
      "heatTracePadded": [4.89, 2.31],
      "heatTimes": [0.1, 1.0]
    },
    "topology": {
      "bettiNumbersBase": { "0": 1, "1": 0 },
      "bettiNumbersPadded": { "0": 1, "1": 0 },
      "componentCountBase": 1,
      "componentCountPadded": 1,
      "cycleCountBase": 0,
      "cycleCountPadded": 0,
      "persistenceEntropyBase": 0.0,
      "persistenceEntropyPadded": 0.0,
      "maxPersistenceBase": 0.912,
      "maxPersistencePadded": 0.912
    },
    "passed": true
  }
}
```

## Geometry Cross-Cultural Commands
```bash
mc geometry cross-cultural analyze <input_json>
```

## Geometry Transplant Commands
```bash
mc geometry transplant run --source <path> --target <path> --output-dir <path> --core-domain <domain>
mc geometry transplant run --source <path> --target <path> --output-dir <path> --core-domain <domain> --target-layer <n>
```

## Geometry Primes Commands
```bash
mc geometry primes probe-model <model_path> --layer <n>
mc geometry primes probe-model <model_path> --layer <n> --output-file <path>
mc geometry primes compare <activations_a.json> <activations_b.json>
```

### Primes Probe Output Schema
```json
{
  "_schema": "mc.geometry.primes.probe.v1",
  "model_path": "/path/to/model",
  "layer": 23,
  "primes_probed": 44,
  "total_primes": 63,
  "overall_coherence": 0.99,
  "overall_coherence_raw": 0.97,
  "category_coherence": {
    "structural": 0.97
  }
}
```
Note: `overall_coherence` uses bias-corrected CKA (AUTO estimator); `overall_coherence_raw` is uncorrected.

### Primes Compare Output Schema
```json
{
  "_schema": "mc.geometry.primes.compare.v1",
  "model_a": "/path/to/activations_a.json",
  "model_b": "/path/to/activations_b.json",
  "common_primes": 42,
  "cka_similarity": 0.98,
  "cka_raw": 0.96,
  "most_similar_primes": ["want", "know"],
  "most_divergent_primes": ["before", "feel"]
}
```
Note: `cka_similarity` uses bias-corrected CKA (AUTO estimator); `cka_raw` is uncorrected.

## Geometry Spatial Commands
```bash
mc geometry spatial anchors
mc geometry spatial probe-model <model_path>
mc geometry spatial analyze <activations_file>
mc geometry spatial euclidean <activations_file>
mc geometry spatial gravity <activations_file>
mc geometry spatial density <activations_file>
mc geometry spatial cross-grounding-feasibility <source_activations> <target_activations>
mc geometry spatial cross-grounding-transfer <source_activations> <target_activations> --concepts <file>
```

## Geometry Baseline Commands
```bash
# List available baselines
mc geometry baseline list
mc geometry baseline list --domain spatial

# Extract baseline from a reference model
mc geometry baseline extract <model_path> --domain spatial
mc geometry baseline extract <model_path> --domain social --layer -1 --k-neighbors 10

# Compare model against baselines (baseline-relative deltas)
mc geometry baseline validate <model_path>
mc geometry baseline validate <model_path> --domains spatial,social

# Compare two models
mc geometry baseline compare <model1_path> <model2_path> --domain spatial
```

### Baseline Output Schema
```json
{
  "_schema": "mc.geometry.baseline.extract.v1",
  "domain": "spatial",
  "modelFamily": "qwen",
  "modelSize": "0.5B",
  "ollivierRicciMean": -0.189,
  "ollivierRicciStd": 0.045,
  "manifoldHealthDistribution": {
    "healthy": 1.0,
    "degenerate": 0.0,
    "collapsed": 0.0
  },
  "intrinsicDimension": 12.4
}
```

### Baseline Validation Output Schema
```json
{
  "_schema": "mc.geometry.baseline.validate.v1",
  "model_path": "/path/to/model",
  "results": [
    {
      "domain": "spatial",
      "baseline_found": true,
      "baseline_model": "qwen-0.5B",
      "current_model": "/path/to/model",
      "missing_metrics": [],
      "notes": [],
      "metrics": {
        "ollivier_ricci_mean": {
          "current": -0.18,
          "baseline": -0.23,
          "baseline_std": 0.08,
          "delta": 0.05,
          "relative_delta": 0.217,
          "z_score": 0.62,
          "percentile": 0.5
        }
      }
    }
  ]
}
```

### Curvature Reference
- **Negative Ricci curvature (< -0.1)**: Hyperbolic geometry - characteristic of high-capacity representations
- **Near-zero curvature (-0.1 to 0.1)**: Flat (Euclidean) geometry
- **Positive curvature (> 0.1)**: Spherical geometry - often indicates low-rank representations

Compare measurements against model family baselines to determine significance.

## Selected Commands

### Safety Commands
```bash
mc safety adapter-probe --adapter <path>    # Run adapter safety probes
```

### Model Merge Commands

Merge uses **null-space constrained transplant** (validated by AlphaEdit, ICLR 2025 Outstanding Paper).
The mathematical guarantee: `A_boundary @ W' = A_boundary @ W_target` (boundary preservation).

```bash
# Basic transplant (requires --transplant-domains)
mc model merge --source <path> --target <path> --output-dir <path> --transplant-domains mathematical

# Multiple domains
mc model merge --source <path> --target <path> --output-dir <path> --transplant-domains mathematical,logical

# With boundary tuning
mc model merge --source <path> --target <path> --output-dir <path> --transplant-domains <domains> --transplant-boundary-k <k> --transplant-geodesic-k <k>

# With per-layer alpha mask
mc model merge --source <path> --target <path> --output-dir <path> --transplant-domains <domains> --knowledge-delta-mask <mask.json>
```

**Pipeline**: `VOCAB → PROBE → TRANSPLANT → VALIDATE`

**Note**: Alpha-blending (`rotate_blend`) was removed - it produces gibberish even for same-architecture models.

### Program Commands (Multi-Donor Transplant)

Multi-donor transplant programs automate sequential transplants from multiple donor models into base models. Programs are defined in YAML config files.

```bash
# Execute a program
mc program run ./configs/program_a.yaml
mc program run ./configs/program_a.yaml --parallel --max-workers 2
mc program run ./configs/program_a.yaml --dry-run
mc program run ./configs/program_a.yaml --base qwen3-8b  # Only process one base

# Resume from checkpoint
mc program run ./configs/program_a.yaml --resume

# Status and listing
mc program status <program_id>
mc program list
mc program show ./configs/program_a.yaml

# Compare results across programs
mc program compare A:./out-A B:./out-B C:./out-C
mc program compare ./out-A ./out-B --output-json comparison.json --output-md comparison.md
```

#### Program Config Schema (YAML)

```yaml
_schema: "mc.program.transplant.v1"
name: "Program A - Permissive Multi-Specialist"
description: |
  Multi-donor transplant into Qwen3/Ministral bases.

bases:
  - id: "qwen3-8b"
    source: "Qwen/Qwen3-8B"
    alias: "qwen3"  # optional short name for output dirs

donors:
  - id: "deepseek-v3"
    source: "deepseek-ai/DeepSeek-V3.2"
    domains: ["reasoning", "logical"]
    priority: 3  # higher priority donors applied first
  - id: "devstral-coding"
    source: "mistralai/Devstral-Small-2507"
    domains: ["coding"]
    priority: 2

evaluation:
  after_each_donor: true
  after_program_complete: true
  benchmarks: ["mmlu_pro", "gpqa_diamond"]
  smoke_test_prompts:
    - "What is 15 * 17?"
    - "Write a Python function that reverses a string."

output:
  base_dir: "~/.modelcypher/merged/program-A"
```

#### Predefined Programs

ModelCypher includes three predefined programs (all MIT/Apache-2.0 licensed for redistribution):

| Program | Base | Focus | Description |
|---------|------|-------|-------------|
| A | Qwen3-8B, Ministral | General | Multi-specialist with math, code, medical, legal |
| B | Ministral | Mistral-centric | Same-tokenizer donors for maximum tokenizer alignment |
| C | Qwen3-8B, Granite | Qwen-centric | Same-tokenizer Qwen ecosystem |

Programs located at: `src/modelcypher/data/programs/`

#### Program Result Schema

```json
{
  "_schema": "mc.result.multi_donor.v1",
  "program_id": "abc123",
  "program_name": "Program A",
  "base_results": [
    {
      "base_id": "qwen3-8b",
      "base_alias": "qwen3",
      "output_path": "~/.modelcypher/merged/program-A/qwen3/final",
      "total_cka_improvement": 0.15,
      "mean_boundary_preserved": 0.92,
      "total_donors_applied": 5,
      "status": "completed",
      "donor_stages": [...]
    }
  ],
  "total_duration_seconds": 3600.0,
  "status": "completed"
}
```

### Entropy Commands
```bash
mc entropy window '[[3.5, 0.2], [3.6, 0.1]]' --size 50          # Sliding window entropy tracking
mc entropy conversation-track --session <file>                  # Multi-turn conversation analysis
mc entropy dual-path '[{"base": [3.5, 0.2], "adapter": [3.8, 0.3]}]'  # Base vs adapter divergence
```

### Agent Commands
```bash
mc agent trace-import --file <path>         # Import OpenTelemetry/Monocle traces
mc agent trace-analyze --trace <file>       # Analyze agent traces
mc agent validate-action --action <json>    # Validate agent actions
```

## Profile Commands

Unified model profile operations for geometric analysis, comparison, and merge planning.

```bash
# Generate a profile from a model
mc profile generate /path/to/model -o profile.json --identity-only
mc profile generate /path/to/model -o profile.json

# Inspect a profile
mc profile inspect profile.json
mc profile inspect profile.json --section geometry
mc profile inspect profile.json --section topology
mc profile inspect profile.json --layer 5

# Compare two profiles
mc profile compare source.json target.json
mc profile compare source.json target.json --save comparison.json
mc profile compare source.json target.json --baseline qwen-baseline.json

# Update a profile with additional info
mc profile update profile.json --model /path/to/model
mc profile update profile.json --model /path/to/model -o updated.json

# Import from legacy profile formats
mc profile import curvature.json --type curvature --output unified.json
mc profile import density.json --type density --base unified.json --output updated.json

# Merge multiple partial profiles
mc profile merge geometry.json topology.json semantic.json --output complete.json
```

### Profile Output Schema
```json
{
  "_schema": "mc.profile.summary.v1",
  "model_path": "/path/to/model",
  "model_family": "qwen",
  "architecture": "Qwen2ForCausalLM",
  "identity": {
    "parameter_count": 500000000,
    "hidden_dim": 896,
    "num_layers": 24,
    "num_attention_heads": 14,
    "vocab_size": 151936
  },
  "computed_sections": ["identity", "geometry"],
  "layer_profile_count": 24,
  "curvature": {
    "global_sectional_mean": -0.15,
    "global_sectional_std": 0.08,
    "global_ollivier_ricci_mean": -0.23,
    "global_ollivier_ricci_std": 0.12,
    "global_intrinsic_dimension_mean": 12.4
  }
}
```

### Profile Comparison Output Schema
```json
{
  "source_path": "/path/to/source",
  "target_path": "/path/to/target",
  "architecture_match": true,
  "hidden_dim_ratio": 1.0,
  "layer_count_ratio": 1.0,
  "vocab_overlap": 0.82,
  "curvature_alignment": 0.92,
  "ricci_alignment": 0.94,
  "dimension_alignment": 0.90,
  "overall_alignment": 0.92,
  "topology_similarity": 0.88,
  "semantic_alignment": 0.91,
  "layer_mapping": {"0": 0, "12": 12, "23": 23},
  "layer_comparisons": [],
  "critical_layers": [0, 12, 23],
  "total_alignment_effort": 1.2,
  "mean_alignment_effort": 0.05,
  "max_alignment_effort": 0.2,
  "recommended_strategy": "procrustes"
}
```

### Profile Sections

| Section | Description |
|---------|-------------|
| `identity` | Model architecture (layers, hidden dim, attention heads, vocab size) |
| `geometry` | Curvature metrics (sectional, Ollivier-Ricci, intrinsic dimension) |
| `topology` | Topological fingerprint (Betti numbers, persistence entropy) |
| `semantic` | Semantic primes signature (dominant concepts, vector embedding) |
| `density` | Activation density distribution across layers |
| `layers` | Per-layer geometric profiles |

## Streaming

- `mc doc convert --stream` emits NDJSON events for conversion progress.
- `mc train logs --follow` tails training logs.

## Schemas + Completions

- `mc help schema <command>` emits JSON schema for a command.
- `mc help completions {bash,zsh,fish}` generates shell completions.
