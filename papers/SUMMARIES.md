# Paper Summaries

One-sentence thesis, key result, and verification command for each paper.

---

## Paper 0: The Shape of Knowledge

**Thesis:** Knowledge in LLMs has measurable geometric structure—concepts occupy regions, inference follows trajectories, and safety can be enforced as geometric constraints.

**Key Result:** Framework paper synthesizing 14 pillars of prior work into the Geometric Knowledge Thesis.

**Status:** Theoretical foundation for Papers 1-3.

→ [Full Paper](paper-0-the-shape-of-knowledge.md)

---

## Paper 1: Invariant Semantic Structure

**Thesis:** Representation geometry is invariant across model families—all concepts (not just semantic primes) show high cross-model CKA.

**Key Result:** Cross-family CKA = 0.94 ± 0.01 across Qwen/Llama/Mistral. Semantic primes (0.92) ≈ random words (0.94), demonstrating universal geometric convergence.

**Verify:**
```bash
mc geometry primes compare ./model-A ./model-B --output text
# Expected: CKA > 0.7 for semantic primes
```

→ [Full Paper](paper-1-invariant-semantic-structure.md)

---

## Paper 2: Entropy as Safety Signal

**Thesis:** Entropy divergence between base and instruction-tuned models (ΔH) detects harmful prompts before generation.

**Key Result:** AUROC = 0.85 for harmful/benign classification (vs 0.51 for raw entropy).

**Verify:**
```bash
mc entropy dual-path --model ./tuned --base ./base --prompt "your prompt" --output text
# High ΔH = potential safety concern
```

→ [Full Paper](paper-2-entropy-safety-signal.md)

---

## Paper 3: Cross-Architecture Transfer

**Thesis:** LoRA adapters can transfer across model architectures via geometric alignment.

**Key Result:** 65-78% skill retention on Qwen→Llama transfer (vs 0% naive copying), <8% safety drift.

**Verify:**
```bash
mc geometry interference predict --source ./qwen --target ./llama --output text
# CKA > 0.7 predicts successful transfer
```

→ [Full Paper](paper-3-cross-architecture-transfer.md)

---

## Paper 4: ModelCypher Toolkit

**Thesis:** The Geometric Knowledge Thesis can be made operational with reproducible measurement tools.

**Key Result:** 274 domain modules, 3,000+ tests, validated implementations of CKA, ΔH, and Procrustes alignment.

**Verify:**
```bash
poetry run pytest  # 3,000+ tests should pass
mc --help          # 50+ commands available
```

→ [Full Paper](paper-4-modelcypher-toolkit.md)

---

## Paper 5: The Semantic Highway

**Thesis:** All transformer language models compress tokenized input to a universal low-dimensional semantic manifold (~1.4 intrinsic dimension) within the first 1-2 layers, regardless of architecture, size, or training data.

**Key Result:** Cross-architecture validation across Qwen (0.5B), Llama (3B), and Mistral (7B) shows:
- 41-79% intrinsic dimension collapse in layers 0-2
- Universal plateau at ID = 1.40 ± 0.10
- Harder domains compress more (ρ = 0.832)

**Verify:**
```bash
mc geometry atlas dimensionality-study /path/to/model --layer 2 --output json
# Expected: Mean ID ~1.3-1.5 at layer 2
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
