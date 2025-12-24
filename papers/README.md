# ModelCypher Research Papers

This directory contains publication-quality research manuscripts.

## Paper Series

| Paper | Title | Status | Focus |
|-------|-------|--------|-------|
| [Paper 0](paper-0-the-shape-of-knowledge.md) | The Shape of Knowledge | Position paper | Framework synthesis |
| [Paper 1](paper-1-manifold-hypothesis-of-agency.md) | The Manifold Hypothesis of Agency | **Methodology complete** | CKA anchor probing |
| [Paper 2](paper-2-linguistic-thermodynamics.md) | Linguistic Thermodynamics | **Methodology complete** | Entropy dynamics |
| [Paper 3](paper-3-unified-manifold-alignment.md) | Cross-Architecture Transfer | **Methodology complete** | Adapter transfer |
| [Paper 4](paper-4-modelcypher-toolkit.md) | ModelCypher Toolkit | Systems paper | Toolkit overview |

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
