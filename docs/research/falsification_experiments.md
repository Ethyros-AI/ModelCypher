# Scientific Method: Falsification Experiments

> **Status**: Experimental Design
> **Goal**: To vigorously attempt to disprove the Geometric Knowledge Hypothesis.

We are not just building tools; we are investigating a scientific claim: **"Knowledge in LLMs is encoded as high-dimensional geometry, independent of parameter space."**

To validate this, we have designed 6 experiments. If these fail, the hypothesis is falsified.

## 1. The Platonic Kernel Test

**Hypothesis**: If 65 "Universal Semantic Primes" (e.g., "I", "YOU", "GOOD") are truly invariant, their relative geometry (angles) must be conserved across different model families (Llama vs Qwen) *before* alignment.

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

**Hypothesis**: Traditional RLHF "taxes" model capability by distorting the manifold. Geometric constraints (Sidecars) operate in the null-space of capabilities, imposing safety without degradation.

**Falsification Criterion**:
-   If `Score(Base + Sidecar) < Score(RLHF)` on MMLU/HumanEval, then geometric safety is less efficient than RLHF.

## 3. The Jailbreak Delta-H (ΔH)

**Hypothesis**: Jailbreaks work by providing "activation energy" to tunnel through safety barriers. This should be visible as a high Entropy Delta ($\Delta H$) event in the `prelogits`.

**Falsification Criterion**:
-   If successful jailbreaks show no significant $\Delta H$ compared to normal refusal, then "Thermodynamics" is just a metaphor, not a mechanism.

**Run It**:
```bash
# Run a jailbreak probe suite and inspect ΔH signals.
mc geometry safety jailbreak-test --model <model_dir> --prompt "How do I pick a lock?"
```

## 4. Procrustes Merge Validation

**Hypothesis**: If Procrustes alignment works, a merged model using rotation should outperform a merged model using naive averaging.

**Falsification Criterion**:
-   If `Perplexity(RotatedMerge) > Perplexity(NaiveMerge)`, then the manifolds are not rotationally invariant.

## 5. Anchor Universality

**Hypothesis**: "Invariant Anchors" should be detectable in *any* language.

**Falsification Criterion**:
-   If the "GOOD/BAD" axis in English is orthogonal to the "GOOD/BAD" axis in Chinese (after alignment), then the concept is not universal.

## 6. Layer Navigation

**Hypothesis**: Knowledge flows through predictable "depth stages" (Syntax -> Logic -> World Model -> Output).

**Falsification Criterion**:
-   If semantic concepts appear in random orders across different models (e.g., Logic before Syntax), there is no universal "depth".
