# ModelCypher Research Papers

This directory contains publication-quality research manuscripts.

## Publication Status

**Platform**: GitHub Releases + Zenodo (DOI)

| Document | Purpose |
|----------|---------|
| [PUBLICATION_GUIDE.md](PUBLICATION_GUIDE.md) | How to create release and get DOI |
| [RELEASE_NOTES.md](RELEASE_NOTES.md) | Release notes template |
| [NEGATIVE-RESULTS.md](NEGATIVE-RESULTS.md) | Key falsified hypothesis (semantic primes) |

## Paper Series

| Paper | Title | Status | Focus |
|-------|-------|--------|-------|
| [Paper 0](paper-0-the-shape-of-knowledge.md) | The Shape of Knowledge | ✅ Ready | Geometric Knowledge Thesis (Framework) |
| [Paper 1](paper-1-invariant-semantic-structure.md) | Invariant Semantic Structure | ✅ Ready | CKA > 0.9 cross-family (primes AND random) |
| [Paper 2](paper-2-entropy-safety-signal.md) | Entropy Safety Signal | ⚠️ Preliminary | ΔH methodology + initial results |
| [Paper 3](paper-3-cross-architecture-transfer.md) | Cross-Architecture Transfer | ⚠️ Preliminary | Methodology + initial 65-78% retention |
| [Paper 4](paper-4-modelcypher-toolkit.md) | ModelCypher Toolkit | ✅ Ready | 274 modules, 3060 tests (Systems) |
| [Paper 5](paper-5-semantic-highway.md) | The Semantic Highway | ✅ Ready | Early-layer ID cliff (3 models, properly hedged) |

### Key Finding (2025-12-25)

**Semantic primes are NOT geometrically special** compared to random words. Both achieve CKA > 0.9 across model families. This is a *stronger* result than originally hypothesized: invariance is universal, not limited to theoretically-motivated concepts. See [NEGATIVE-RESULTS.md](NEGATIVE-RESULTS.md).

## Quality Standards

All papers follow arXiv/NeurIPS conventions:

- **Abstract**: Single paragraph summarizing contribution
- **Methodology**: Mathematical definitions, algorithms, protocols
- **Falsification**: Explicit criteria for rejecting hypotheses
- **Related Work**: Inline citations to foundational papers
- **Reproducibility**: CLI commands, code pointers, seeds

## Experimental Status

### Validated Research (2025-12-25)

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
| Semantic prime CKA comparisons | Paper 1 | Run `mc geometry primes probe` |
| Null distribution generation | Paper 1 | 200 control samples |
| Modifier entropy matrix | Paper 2 | Run `mc entropy measure` |
| Temperature sweep | Paper 2 | T ∈ {0.0, ..., 1.5} |
| Safety signal AUROC | Paper 2 | Curated prompt suite |
| Intersection maps | Paper 3 | Run `mc model analyze-alignment` |
| Skill retention benchmarks | Paper 3 | HumanEval subset |

## Test Data Requirements

See [TEST_DATA_REQUIREMENTS.md](TEST_DATA_REQUIREMENTS.md) for:
- Exact CLI commands to generate data
- Required model IDs
- Output format specifications
- Human review requirements (safety prompts)

## Quick Publish

```bash
# 1. Commit all changes
git add papers/
git commit -m "docs: prepare paper series for publication"

# 2. Create release tag
git tag -a v0.1.0-papers -m "ModelCypher Paper Series v0.1.0"
git push origin v0.1.0-papers

# 3. Create GitHub release (requires gh CLI)
gh release create v0.1.0-papers \
  --title "ModelCypher Paper Series v0.1.0" \
  --notes-file papers/RELEASE_NOTES.md

# 4. Connect Zenodo (one-time, in browser)
# Visit: https://zenodo.org/account/settings/github/
# Enable: Ethyros-AI/ModelCypher
# DOI will be generated automatically
```

See [PUBLICATION_GUIDE.md](PUBLICATION_GUIDE.md) for detailed instructions.

## References

All cited papers are available in `docs/references/arxiv/` (46 PDFs, ~120 MB).

Master bibliography: [KnowledgeasHighDimensionalGeometryInLLMs.md](../docs/research/KnowledgeasHighDimensionalGeometryInLLMs.md)
