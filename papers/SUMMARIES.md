# Paper Summaries

One-sentence thesis, scope, and verification command for each paper. Results are draft and must be reproduced with current code.

---

## Paper 0: The Shape of Knowledge [EMPIRICAL]

**Thesis:** Knowledge in LLMs has measurable geometric structure—concepts occupy regions, inference follows trajectories, and safety can be enforced as geometric constraints.

**Key Result:** Framework paper synthesizing prior work into the Geometric Knowledge Thesis. Intra-model alignment invariance verified (CKA = 1.0 after Procrustes).

**Status:** [PROVEN: by construction] alignment invariance (CKA = 1.0 on training probes after Procrustes); [CONJECTURAL] dimensional hierarchy, physics implications, safety-through-geometry.

→ [Full Paper](paper-0-the-shape-of-knowledge.md)

---

## Paper 1: Invariant Semantic Structure [PROVEN: by construction] intra-model; [CONJECTURAL] cross-model

**Thesis:** Representation geometry is invariant across model families; CKA is used to study cross-model similarity.

**Key Result:** Intra-model alignment invariance [PROVEN: by construction] (CKA = 1.0 on training probes after Procrustes). Cross-family CKA comparisons reported but reproduction pending [CONJECTURAL]. ~~Semantic primes are geometrically special~~ [DISPROVEN: see NEGATIVE-RESULTS.md].

**Verify:**
```bash
poetry run mc analyze reasoning-geometry-validation --model ./model-A --benchmark arithmetic
```

→ [Full Paper](paper-1-invariant-semantic-structure.md)

---

## Paper 2: Entropy as Safety Signal [CONJECTURAL]

**Thesis:** Entropy divergence between base and instruction-tuned models (ΔH) is evaluated as a safety signal.

**Key Result:** Defines ΔH measurement protocol and evaluation plan (reproduction pending). No validated results.

**Verify:**
```bash
poetry run mc analyze entropy-trajectory --model ./tuned --prompt "your prompt"
```

→ [Full Paper](paper-2-entropy-safety-signal.md)

---

## Paper 3: Cross-Architecture Transfer [CONJECTURAL]

**Thesis:** Evaluates whether LoRA adapters can transfer across architectures via geometric alignment.

**Key Result:** Defines transfer protocol and alignment metrics (reproduction pending). No validated results.

**Verify:**
```bash
poetry run mc merge run -s ./qwen -t ./llama -o ./merged
```

→ [Full Paper](paper-3-cross-architecture-transfer.md)

---

## Paper 4: ModelCypher Toolkit [EMPIRICAL]

**Thesis:** The Geometric Knowledge Thesis can be made operational with reproducible measurement tools.

**Key Result:** Toolkit overview with measurement implementations (CKA [VALIDATED], null-space transplant [VALIDATED], ΔH [CONJECTURAL]).

**Verify:**
```bash
poetry run pytest
poetry run mc --help
```

→ [Full Paper](paper-4-modelcypher-toolkit.md)

---

## Paper 5: The Semantic Highway [EMPIRICAL]

**Thesis:** Explores whether intrinsic dimension drops sharply in early layers and stabilizes in a low-ID plateau.

**Key Result:** Early-layer ID cliff and mid-layer plateau observed in SmolLM-135M [EMPIRICAL]. Mechanistic interpretation [CONJECTURAL].

**Verify:**
```bash
poetry run mc analyze dimension-profile --model /path/to/model --prompt "test"
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
