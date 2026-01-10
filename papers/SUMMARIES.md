# Paper Summaries

One-sentence thesis, scope, and verification command for each paper. Results are draft and must be reproduced with current code.

---

## Paper 0: The Shape of Knowledge

**Thesis:** Knowledge in LLMs has measurable geometric structure—concepts occupy regions, inference follows trajectories, and safety can be enforced as geometric constraints.

**Key Result:** Framework paper synthesizing prior work into the Geometric Knowledge Thesis.

**Status:** Theoretical foundation for Papers 1-3.

→ [Full Paper](paper-0-the-shape-of-knowledge.md)

---

## Paper 1: Invariant Semantic Structure

**Thesis:** Representation geometry is invariant across model families; CKA is used to study cross-model similarity.

**Key Result:** Reports cross-family CKA comparisons across vocab sets (reproduction pending).

**Verify:**
```bash
poetry run mc geometry primes compare ./model-A ./model-B --output text
```

→ [Full Paper](paper-1-invariant-semantic-structure.md)

---

## Paper 2: Entropy as Safety Signal

**Thesis:** Entropy divergence between base and instruction-tuned models (ΔH) is evaluated as a safety signal.

**Key Result:** Defines ΔH measurement protocol and evaluation plan (reproduction pending).

**Verify:**
```bash
poetry run mc entropy dual-path --model ./tuned --base ./base --prompt "your prompt" --output text
```

→ [Full Paper](paper-2-entropy-safety-signal.md)

---

## Paper 3: Cross-Architecture Transfer

**Thesis:** Evaluates whether LoRA adapters can transfer across architectures via geometric alignment.

**Key Result:** Defines transfer protocol and alignment metrics (reproduction pending).

**Verify:**
```bash
poetry run mc geometry interference predict --source ./qwen --target ./llama --output text
```

→ [Full Paper](paper-3-cross-architecture-transfer.md)

---

## Paper 4: ModelCypher Toolkit

**Thesis:** The Geometric Knowledge Thesis can be made operational with reproducible measurement tools.

**Key Result:** Toolkit overview with measurement implementations (CKA, ΔH, Procrustes).

**Verify:**
```bash
poetry run pytest
poetry run mc --help
```

→ [Full Paper](paper-4-modelcypher-toolkit.md)

---

## Paper 5: The Semantic Highway

**Thesis:** Explores whether intrinsic dimension drops sharply in early layers and stabilizes in a low-ID plateau (reproduction pending).

**Key Result:** Reports early-layer ID measurements and trends (reproduction pending).

**Verify:**
```bash
poetry run mc geometry atlas dimensionality-study /path/to/model --layer 0 --layer 1 --layer 2 --output json
```

→ [Full Paper](paper-5-semantic-highway.md)

---

## Citation

```bibtex
@software{ModelCypher2025,
  author = {Kempf, Jason and ModelCypher Contributors},
  title = {ModelCypher: High-Dimensional Geometry for LLM Safety and Merging},
  year = {2025},
  url = {https://github.com/Ethyros-AI/ModelCypher}
}
```
