# ModelCypher Research Papers

This directory contains publication-quality research manuscripts.

## Paper Series

| Paper | Title | Status | Focus |
|-------|-------|--------|-------|
| [Paper 0](paper-0-the-shape-of-knowledge.md) | The Shape of Knowledge | Framework | Geometric Knowledge Thesis |
| [Paper 1](paper-1-invariant-semantic-structure.md) | Invariant Semantic Structure Across Language Model Families | Empirical | CKA = 0.82 for primes |
| [Paper 2](paper-2-entropy-safety-signal.md) | Intensity Modifiers Reduce Entropy | Empirical | ΔH AUROC = 0.85 |
| [Paper 3](paper-3-cross-architecture-transfer.md) | Cross-Architecture Adapter Transfer | Empirical | 65-78% skill retention |
| [Paper 4](paper-4-modelcypher-toolkit.md) | ModelCypher Toolkit | Systems | 274 modules, 2972 tests |

## Quality Standards

All papers follow arXiv/NeurIPS conventions:

- **Abstract**: Single paragraph summarizing contribution
- **Methodology**: Mathematical definitions, algorithms, protocols
- **Falsification**: Explicit criteria for rejecting hypotheses
- **Related Work**: Inline citations to foundational papers
- **Reproducibility**: CLI commands, code pointers, seeds

## Experimental Status

### Validated Research (2025-12-23)

Foundational geometry hypotheses have been validated with empirical results:

| Hypothesis | Status | Results |
|------------|--------|---------|
| Spatial Grounding (Blind Physicist) | ✅ Validated | [spatial_grounding.md](../docs/research/spatial_grounding.md) |
| Social Geometry (Latent Sociologist) | ✅ Validated | [social_geometry.md](../docs/research/social_geometry.md) |
| Temporal Topology (Latent Chronologist) | ⚠️ Partial | [temporal_topology.md](../docs/research/temporal_topology.md) |
| Moral Geometry (Latent Ethicist) | ✅ Validated | [moral_geometry.md](../docs/research/moral_geometry.md) |

### Paper-Specific Experiments (Pending)

| Experiment | Paper | Status |
|------------|-------|--------|
| Semantic prime CKA comparisons | Paper 1 | **TODO**: Run `mc geometry primes probe` |
| Null distribution generation | Paper 1 | **TODO**: 200 control samples |
| Modifier entropy matrix | Paper 2 | **TODO**: Run `mc entropy measure` |
| Temperature sweep | Paper 2 | **TODO**: T ∈ {0.0, ..., 1.5} |
| Safety signal AUROC | Paper 2 | **TODO**: Curated prompt suite |
| Intersection maps | Paper 3 | **TODO**: Run `mc model analyze-alignment` |
| Skill retention benchmarks | Paper 3 | **TODO**: HumanEval subset |

## Test Data Requirements

See [TEST_DATA_REQUIREMENTS.md](TEST_DATA_REQUIREMENTS.md) for:
- Exact CLI commands to generate data
- Required model IDs
- Output format specifications
- Human review requirements (safety prompts)

## References

All cited papers are available in `docs/references/arxiv/` (46 PDFs, ~120 MB).

Master bibliography: [KnowledgeasHighDimensionalGeometryInLLMs.md](../docs/research/KnowledgeasHighDimensionalGeometryInLLMs.md)
