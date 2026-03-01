# ModelCypher Paper Series v0.1.0

> **Note (2026-03-01):** This is the original v0.1.0 release notes template from early development. For current project state, see [MISSION.md](../docs/MISSION.md).

## The Geometric Knowledge Thesis

This release collects draft manuscripts and protocols. Empirical validation is pending and must be reproduced with current code and data.

---

## Papers

### Empirical Results

| Paper | Title | Status | Scope |
|-------|-------|--------|-------|
| **Paper 5** | [The Semantic Highway](paper-5-semantic-highway.md) | [EMPIRICAL] | Early-layer ID cliff exploration |
| **Paper 1** | [Invariant Semantic Structure](paper-1-invariant-semantic-structure.md) | [VALIDATED] intra-model; [CONJECTURAL] cross-model | CKA comparisons |

### Framework & Systems

| Paper | Title | Status | Scope |
|-------|-------|--------|-------|
| **Paper 0** | [The Shape of Knowledge](paper-0-the-shape-of-knowledge.md) | [VALIDATED] | Theoretical framework |
| **Paper 4** | [ModelCypher Toolkit](paper-4-modelcypher-toolkit.md) | [EMPIRICAL] | Toolkit overview |

### Methodology (Draft) [CONJECTURAL]

| Paper | Title | Status | Scope |
|-------|-------|--------|-------|
| **Paper 2** | [Entropy Safety Signal](paper-2-entropy-safety-signal.md) | [CONJECTURAL] | Protocol definition |
| **Paper 3** | [Cross-Architecture Transfer](paper-3-cross-architecture-transfer.md) | [CONJECTURAL] | Protocol definition |

---

## Historical Note

A prior run suggested semantic primes are not geometrically special compared to random words. Reproduction is pending. See [NEGATIVE-RESULTS.md](NEGATIVE-RESULTS.md) for details.

---

## Reproducibility (Raw Metrics)

Use the ModelCypher CLI to reproduce raw measurements:

```bash
# Install
poetry install

# Analyze concept volumes
poetry run mc analyze concept-volume --model /path/to/model

# Cross-model alignment validation
poetry run mc analyze crm-build /path/to/model --output model_crm.json
poetry run mc analyze crm-compare model_crm.json reference_crm.json

# Run intrinsic dimension analysis
poetry run mc analyze dimension-profile --model /path/to/model --prompt "test"
```

---

## Citation

```bibtex
@software{kempf2025modelcypher,
  author = {Kempf, Jason},
  title = {ModelCypher: A Geometric Toolkit for Large Language Model Analysis and Safe Adaptation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Ethyros-AI/ModelCypher},
  version = {0.1.0}
}
```

---

## What's Next

- Full validation of Paper 2 (entropy safety signal with curated prompt suite)
- Full validation of Paper 3 (HumanEval benchmark for adapter transfer)
- Additional model families (Phi, Gemma, Falcon)
