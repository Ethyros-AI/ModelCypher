# ModelCypher Paper Series v0.1.0

## The Geometric Knowledge Thesis

This release presents empirical evidence that knowledge in large language models has measurable geometric structure that is preserved across architectures.

---

## Papers

### Tier 1: Validated Results

| Paper | Title | Key Finding |
|-------|-------|-------------|
| **Paper 5** | [The Semantic Highway](papers/paper-5-semantic-highway.md) | Intrinsic dimension cliff in layers 0-4 across Qwen, Llama, Mistral |
| **Paper 1** | [Invariant Semantic Structure](papers/paper-1-invariant-semantic-structure.md) | CKA > 0.9 cross-family for BOTH primes AND random words |

### Tier 2: Framework & Systems

| Paper | Title | Key Finding |
|-------|-------|-------------|
| **Paper 0** | [The Shape of Knowledge](papers/paper-0-the-shape-of-knowledge.md) | Theoretical framework for geometric knowledge thesis |
| **Paper 4** | [ModelCypher Toolkit](papers/paper-4-modelcypher-toolkit.md) | 274 modules, 3060 tests, 46 papers implemented |

### Tier 3: Methodology (Preliminary)

| Paper | Title | Status |
|-------|-------|--------|
| **Paper 2** | [Entropy Safety Signal](papers/paper-2-entropy-safety-signal.md) | Protocol defined, preliminary AUROC ~0.85 |
| **Paper 3** | [Cross-Architecture Transfer](papers/paper-3-cross-architecture-transfer.md) | Protocol defined, preliminary 65-78% retention |

---

## Key Scientific Finding

**Semantic primes are NOT geometrically special.**

Original hypothesis: NSM semantic primes would show higher cross-model CKA than random words.

Actual result:
- Semantic primes: CKA = 0.92
- Random words: CKA = 0.94

This FALSIFIES the original hypothesis but reveals a STRONGER result: **universal representation invariance** applies to ALL concepts, not just theoretically-motivated ones.

See [NEGATIVE-RESULTS.md](papers/NEGATIVE-RESULTS.md) for full analysis.

---

## Reproducibility

All results can be reproduced using ModelCypher CLI:

```bash
# Install
poetry install

# Run semantic prime comparison
poetry run mc geometry primes probe --model /path/to/model

# Run CKA comparison
poetry run mc geometry cka compare --model-a /path/to/model-a --model-b /path/to/model-b

# Run intrinsic dimension analysis
poetry run mc geometry atlas dimensionality-study /path/to/model
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
