# ModelCypher Paper Series v0.1.0

## The Geometric Knowledge Thesis

This release collects draft manuscripts and protocols. Empirical validation is pending and must be reproduced with current code and data.

---

## Papers

### Draft Manuscripts (Validation Pending)

| Paper | Title | Scope |
|-------|-------|-------|
| **Paper 5** | [The Semantic Highway](paper-5-semantic-highway.md) | Early-layer ID cliff exploration |
| **Paper 1** | [Invariant Semantic Structure](paper-1-invariant-semantic-structure.md) | Cross-model CKA comparisons |

### Framework & Systems

| Paper | Title | Scope |
|-------|-------|-------|
| **Paper 0** | [The Shape of Knowledge](paper-0-the-shape-of-knowledge.md) | Theoretical framework |
| **Paper 4** | [ModelCypher Toolkit](paper-4-modelcypher-toolkit.md) | Toolkit overview |

### Methodology (Draft)

| Paper | Title | Scope |
|-------|-------|-------|
| **Paper 2** | [Entropy Safety Signal](paper-2-entropy-safety-signal.md) | Protocol definition |
| **Paper 3** | [Cross-Architecture Transfer](paper-3-cross-architecture-transfer.md) | Protocol definition |

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

# Cross-model reasoning geometry validation
poetry run mc analyze reasoning-geometry-validation --model /path/to/model --benchmark arithmetic

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
