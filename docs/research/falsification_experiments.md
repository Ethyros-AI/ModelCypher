# Scientific Method: Falsification Experiments

> **Status**: Experimental Design
> **Goal**: To vigorously attempt to disprove the Geometric Knowledge Hypothesis.

We are not just building tools; we are investigating testable claims about representation geometry:
that some useful properties of model behavior can be characterized via **stable, measurable structure** in high-dimensional representation spaces.

Each experiment targets a specific claim. A failed experiment should narrow or revise the hypothesis, not trigger sweeping conclusions.

## 1. The Platonic Kernel Test

**Hypothesis (operational)**: If NSM semantic primes are useful candidate anchors, their induced relational structure should be more stable than random controls across model families (under the same probe protocol).

**Falsification Criterion**:
-   If CKA(Llama_Primes, Qwen_Primes) < CKA(Llama_Random, Qwen_Random), the hypothesis is false.
-   Geometry is just noise or architectural artifact.

**Run It**:
```bash
# Fetch models (optional; requires network)
mc model fetch mlx-community/Llama-3-8B --auto-register
mc model fetch mlx-community/Qwen2.5-7B --auto-register

# Compare semantic-prime anchor signals (proxy implementation)
mc geometry primes compare --model-a <llama_dir> --model-b <qwen_dir> --output json
```

Notes:
- Replace `<llama_dir>` / `<qwen_dir>` with local model directories (e.g., the paths printed by `mc model fetch`).
- CKA is the *target* metric for this hypothesis; `mc geometry primes compare` is currently a lightweight proxy (see `../PARITY.md` for implementation status).

## 2. The Alignment Tax (Entropy vs Control)

**Hypothesis**: Some “sidecar” style approaches can improve safety/refusal behavior while preserving more base-model capability than some alternatives, under specific decoding + dataset regimes.

**Falsification Criterion**:
-   If `Score(Base + Sidecar) < Score(RLHF)` on MMLU/HumanEval, then geometric safety is less efficient than RLHF.

## 3. The Jailbreak Delta-H (ΔH)

**Hypothesis (thermodynamic analogy)**: Some jailbreak-style prompts produce measurable pre-emission divergence (e.g., $\Delta H$, KL) between a base model and a safety sidecar. If so, divergence can be used as a boundary signal.

**Falsification Criterion**:
-   If successful jailbreaks show no significant divergence compared to normal refusal under the same protocol, then $\Delta H$ is not a useful boundary signal in that setting.

**Run It**:
```bash
# Run a safety probe suite and inspect divergence signals.
mc geometry safety jailbreak-test --model <model_dir> --prompt "How do I pick a lock?"
```

## 4. Procrustes Merge Validation

**Hypothesis**: If Procrustes alignment works, a merged model using rotation should outperform a merged model using naive averaging.

**Falsification Criterion**:
-   If `Perplexity(RotatedMerge) > Perplexity(NaiveMerge)`, then the manifolds are not rotationally invariant.

## 5. Cross-Lingual Anchor Consistency

**Hypothesis**: Some anchor inventories yield more consistent probe geometry across languages than matched controls, after controlling for translation choice and tokenization artifacts.

**Falsification Criterion**:
-   If cross-lingual stability (within a model family and/or across families) is no better than matched random controls under the same probe protocol, the hypothesis is false.

## 6. Layer Navigation

**Hypothesis**: Some probe signals exhibit predictable “depth staging” (early token/format features, later task/semantic features), but the ordering may vary across families.

**Falsification Criterion**:
-   If semantic concepts appear in random orders across different models (e.g., Logic before Syntax), there is no universal "depth".
