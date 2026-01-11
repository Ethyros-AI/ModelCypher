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

Status labels indicate experimental validation status as of January 2026.

| Paper | Title | Status | Focus |
|-------|-------|--------|-------|
| [Paper 0](paper-0-the-shape-of-knowledge.md) | The Shape of Knowledge | **Supported** | Geometric Knowledge Thesis - alignment invariance verified |
| [Paper 1](paper-1-invariant-semantic-structure.md) | Invariant Semantic Structure | **Supported** | CKA = 1.0 after Procrustes alignment |
| [Paper 2](paper-2-entropy-safety-signal.md) | Entropy Safety Signal | Methodology | ΔH methodology (reproduction pending) |
| [Paper 3](paper-3-cross-architecture-transfer.md) | Cross-Architecture Transfer | Methodology | Cross-architecture transfer (reproduction pending) |
| [Paper 4](paper-4-modelcypher-toolkit.md) | ModelCypher Toolkit | Reference | Toolkit overview |
| [Paper 5](paper-5-semantic-highway.md) | The Semantic Highway | **Supported** | Layer-wise ID compression verified |

**Key experimental result**: Raw CKA = 0.60 → Aligned CKA = 1.00 (see [`experiments/results/geometry_validation.json`](../experiments/results/geometry_validation.json))

### Historical Note (2025-12-25)

A single run suggested semantic primes are not geometrically special compared to random words. Reproduction is pending. See [NEGATIVE-RESULTS.md](NEGATIVE-RESULTS.md).

## Quality Standards

All papers follow arXiv/NeurIPS conventions:

- **Abstract**: Single paragraph summarizing contribution
- **Methodology**: Mathematical definitions, algorithms, protocols
- **Falsification**: Explicit criteria for rejecting hypotheses
- **Related Work**: Inline citations to foundational papers
- **Reproducibility**: CLI commands, code pointers, seeds

## Experimental Status

### Verified (January 2026)

| Experiment | Result | Source |
|------------|--------|--------|
| Alignment Invariance | Raw CKA = 0.60 → Aligned CKA = 1.0 | [`experiments/results/geometry_validation.json`](../experiments/results/geometry_validation.json) |
| Layer-wise ID | 15.8 → 1.8 → 9.6 (compression pattern) | Same file |
| Domain Geometry | Spatial ID=1.5, Moral ID=8.0 | Same file |

### Reproduction Pending

| Experiment | Paper | CLI Command |
|------------|-------|-------------|
| Cross-model CKA | Paper 1 | `mc geometry primes probe-model` |
| Modifier entropy | Paper 2 | `mc thermo measure` |
| Cross-architecture transfer | Paper 3 | `mc model analyze-alignment` |

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

All cited papers are available in `docs/references/arxiv/`.

Master bibliography: [KnowledgeasHighDimensionalGeometryInLLMs.md](../docs/research/KnowledgeasHighDimensionalGeometryInLLMs.md)
