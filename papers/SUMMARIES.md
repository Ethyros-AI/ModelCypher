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

**Thesis:** In three tested transformer LLMs, intrinsic dimension drops sharply in the first 1–2 layers and then stabilizes in a low-ID plateau (~1.3–1.5). We treat this as a preliminary observation and a hypothesis about projection onto a low-dimensional conceptual manifold.

**Key Result:** Across Qwen (0.5B), Llama (3B), and Mistral (7B) we observe:
- 40–79% intrinsic dimension collapse in layers 0–2
- Post-cliff plateau in the 1.3–1.5 range (mean 1.40 ± 0.10 across these models)
- In Qwen, higher initial domain ID compresses more (ρ = 0.832)

**Verify:**
```bash
mc geometry atlas dimensionality-study /path/to/model --layer 0 --layer 1 --layer 2 --output json
# Look for an early-layer cliff and a low-ID plateau in the first few layers
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
