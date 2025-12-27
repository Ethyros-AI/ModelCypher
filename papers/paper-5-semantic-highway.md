# Paper 5: The Semantic Highway (Preliminary Observation)

**Author**: Jason Kempf
**Affiliation**: EthyrosAI
**Date**: December 2025

> **Status**: Preliminary empirical observation across three model families; hypothesis for further testing.

## Abstract

Across three transformer language models (Qwen2.5-0.5B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3), we observe a consistent pattern in intrinsic dimension (ID) profiles measured with TwoNN on a 439-probe semantic corpus: (1) a sharp early-layer drop in ID ("dimensionality cliff"), and (2) a mid-layer plateau in the range 1.3–1.5. The magnitude and layer index of the cliff vary by architecture (40–79% drop across the tested models). Using Qwen with broader layer coverage, we also observe that domains with higher initial ID compress more strongly (Spearman ρ = 0.832). We present these results as an observation, not a universal law, and propose a working hypothesis: early transformer layers rapidly project tokenized representations onto a low-dimensional conceptual manifold, after which representations evolve primarily within that manifold. We outline follow-up tests needed to determine how broadly this pattern holds across architectures, scales, languages, and training regimes.

## 1. Introduction

The Platonic Representation Hypothesis (Huh et al., 2024) suggests that independently trained neural networks converge to similar internal representations. Our prior work ([Paper 1](paper-1-invariant-semantic-structure.md)) validates strong cross-family similarity with CKA > 0.9 on several anchor sets. An open question remains: *what dynamics produce this convergence across architectures?*

This paper reports a simple empirical observation about *intrinsic dimension over depth* in three transformer LLMs. When we measure ID using TwoNN across a fixed semantic probe corpus, all three models show an early-layer ID collapse followed by a low-ID plateau. We use "semantic highway" as a shorthand label for this plateau regime, but treat it as a **working hypothesis** rather than a universal property.

### 1.1 Contributions

1. **Observed early-layer cliff (3 models)**: A sharp reduction in intrinsic dimension within the first 1–2 layers for all tested models.

2. **Observed low-ID plateau (3 models)**: A mid-layer ID plateau in the range 1.3–1.5 for all tested models.

3. **Domain-dependent compression (Qwen)**: In Qwen 0.5B, higher initial domain ID correlates with stronger compression (ρ = 0.832).

4. **Hypothesis + test plan**: A concrete mechanism hypothesis (rapid projection to a conceptual manifold) and a set of follow-up experiments to test generality.

## 2. Methods

### 2.1 Intrinsic Dimension Estimation

We use the Two-Nearest Neighbors (TwoNN) method (Facco et al., 2017) to estimate local intrinsic dimension:

For each point $i$, let $r_{i,1}$ and $r_{i,2}$ be the first and second nearest-neighbor distances, and define:

$$\mu_i = \frac{r_{i,2}}{r_{i,1}}$$

Under the TwoNN model assumptions, $\mu$ follows $F(\mu) = 1 - \mu^{-d}$, implying a linear relationship:

$$-\log(1 - F(\mu)) = d \, \log(\mu)$$

Using the empirical CDF, we estimate $d$ as the slope of a regression through the origin between $x_i = \log(\mu_{(i)})$ and $y_i = -\log\left(1 - \frac{i}{N}\right)$ for sorted ratios $\mu_{(i)}$.

Distances $r_{i,1}, r_{i,2}$ are computed via geodesic path lengths on a k-NN graph (k=10).

### 2.2 Semantic Probe Corpus

We analyze 439 probes from the UnifiedAtlas spanning 12 semantic domains:

| Domain | Probe Count | Examples |
|--------|-------------|----------|
| Mathematical | 65 | Fibonacci, primes, Catalan |
| Logical | 41 | Modus ponens, De Morgan |
| Computational | 73 | Gates, algorithms |
| Spatial | 38 | Left/right, near/far |
| Temporal | 33 | Past/future, duration |
| Moral | 23 | Right/wrong, virtue |
| Affective | 28 | Joy, fear, anger |
| Relational | 40 | Kinship, social roles |
| Mental | 20 | Think, know, believe |
| Linguistic | 35 | Syntax, semantics |
| Structural | 10 | Part/whole, containment |
| Philosophical | 33 | Existence, causation |

Each probe has 3-8 support texts. Per-probe ID is computed by:
1. Extracting activations for all support texts at target layer
2. Building k-NN graph on activation vectors
3. Computing geodesic distances via Dijkstra's algorithm
4. Estimating ID via TwoNN regression

### 2.3 Models Under Test

| Model | Family | Layers | Hidden Size | Quantization |
|-------|--------|--------|-------------|--------------|
| Qwen2.5-0.5B-Instruct | Qwen | 24 | 896 | bf16 |
| Llama-3.2-3B-Instruct | Llama | 28 | 3072 | 4-bit |
| Mistral-7B-Instruct-v0.3 | Mistral | 32 | 4096 | 4-bit |

### 2.4 Analysis Protocol

For each model:
1. Extract activations at layers [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 20, L-1]
2. Compute per-probe ID at each layer
3. Aggregate by domain and compute mean ID
4. Identify cliff (maximum layer-over-layer drop)
5. Identify plateau (stable ID region)
6. Identify expansion (final layer ID increase)

## 3. Results

### 3.1 The Dimensionality Cliff

All three tested architectures exhibit dramatic ID collapse in early layers:

**Qwen 2.5-0.5B (24 layers)**
| Layer | Mean ID | Δ from prev |
|-------|---------|-------------|
| 0 | 7.39 | — |
| 1 | 6.89 | -6.8% |
| **2** | **1.42** | **-79.4%** |
| 3 | 1.23 | -13.4% |
| 4-20 | ~1.30 | stable |
| 23 | 8.15 | +527% |

**Llama 3.2-3B (28 layers)**
| Layer | Mean ID | Δ from prev |
|-------|---------|-------------|
| 0 | 3.21 | — |
| **1** | **1.51** | **-53.0%** |
| 2 | 1.53 | +1.3% |
| 3 | 1.50 | -2.0% |

**Mistral 7B (32 layers)**
| Layer | Mean ID | Δ from prev |
|-------|---------|-------------|
| 0 | 2.34 | — |
| **1** | **1.39** | **-40.6%** |
| 2 | 1.39 | 0% |
| 3 | 1.44 | +3.6% |
| 4 | 1.51 | +4.9% |
| 5 | 1.55 | +2.6% |

### 3.2 Observed Plateau in Three Models

Despite different architectures, sizes, and cliff magnitudes, these three models converge to a similar low-ID range after the cliff:

| Model | Cliff Layer | Cliff Magnitude | Plateau ID |
|-------|-------------|-----------------|------------|
| Qwen 0.5B | L1→L2 | 79.4% | **1.3** |
| Llama 3B | L0→L1 | 53.0% | **1.5** |
| Mistral 7B | L0→L1 | 40.6% | **1.4** |

**Mean plateau ID (these three models): 1.40 ± 0.10**

This similarity is observed despite:
- 14× difference in parameter count (0.5B vs 7B)
- Different tokenizers (Qwen vs Llama vs Mistral)
- Different training data
- Different quantization (bf16 vs 4-bit)

### 3.3 Domain Complexity Normalization

Using Qwen 0.5B data with full layer coverage, we analyze how different semantic domains compress:

| Domain | L0 ID | L2 ID | Compression |
|--------|-------|-------|-------------|
| Spatial | 2.86 | 0.72 | 74.8% |
| Linguistic | 3.39 | 0.87 | 74.3% |
| Structural | 5.20 | 1.52 | 70.8% |
| Temporal | 5.54 | 1.26 | 77.3% |
| Mental | 5.92 | 1.11 | 81.2% |
| Mathematical | 6.60 | 1.59 | 75.9% |
| Moral | 6.96 | 1.35 | 80.6% |
| Logical | 7.00 | 1.36 | 80.6% |
| Affective | 8.04 | 1.45 | 82.0% |
| Computational | 8.26 | 1.72 | 79.2% |
| Relational | 10.71 | 1.35 | 87.4% |
| Philosophical | 12.54 | 1.81 | 85.6% |

**Spearman correlation (initial ID vs compression): ρ = 0.832**

Domains with higher initial complexity (e.g., Philosophical at 12.54) compress more aggressively (85.6%) than simpler domains (e.g., Spatial at 2.86, compressing 74.8%). The cliff normalizes semantic complexity to a uniform bottleneck.

### 3.4 Three-Regime Interpretation (Working Hypothesis)

In these three models, the ID profiles suggest three regimes:

1. **Entry Zone (L0)**: Variable dimensionality (2.3 - 7.4 ID)
   - Architecture-specific embedding representation
   - Tokenization artifacts present

2. **Plateau regime (L1/L2 onwards)**: Low-ID plateau (~1.4 ID in these models)
   - Compressed semantic representation
   - Candidate architecture-robust structure (requires broader testing)
   - Stable through middle layers

3. **Exit Zone (final layers)**: Expansion for output
   - Qwen L23: 8.15 ID (6× expansion)
   - Task-specific formatting
   - Vocabulary projection preparation

## 4. Discussion

### 4.1 Possible Explanations for the Cliff

One plausible mechanism is **information bottleneck compression** (Tishby & Zaslavsky, 2015): early layers discard tokenization- and architecture-specific degrees of freedom while retaining semantics needed for downstream behavior. Under this view, the plateau value in the 1.3–1.5 range reflects a stable low-dimensional regime that supports:
- Maintaining semantic distinctions
- Enabling compositionality
- Supporting generalization

### 4.2 Interpreting the Plateau (Hypothesis)

An interpretation consistent with these measurements is that the plateau reflects the **latent shape of conceptual space**: language meaning may be representable on a low-dimensional manifold, and early transformer layers learn a projection from tokenized input into that manifold. However, with only three models we cannot distinguish this explanation from alternatives such as shared training dynamics, tokenizer/frequency effects, or estimator artifacts. The goal of this paper is to surface the pattern and specify how to test it.

### 4.3 Implications for Transfer Learning

If the plateau regime generalizes, it could help explain why cross-architecture transfer can work:
- **Shared highway**: Middle-layer representations are compatible across models
- **Different ramps**: Entry/exit zones are architecture-specific
- **LoRA efficiency**: Adapters modify highway traffic, not the road itself

Our [Paper 3](paper-3-cross-architecture-transfer.md) results (65-78% skill retention on Qwen→Llama transfer) are consistent with this story, but do not by themselves establish causality between a low-ID plateau and transfer success.

### 4.4 Implications for Model Merging

This three-regime interpretation suggests (but does not guarantee) that:
- **Early layer merging** is difficult (different entry ramps)
- **Middle layer merging** is safe (shared highway)
- **Late layer merging** is risky (different exit formatting)

This aligns with empirical observations that middle-layer SLERP merging outperforms early/late layer merging.

### 4.5 Limitations and Follow-Up Experiments

This document reports a pattern observed in **three** models. That is enough to motivate a hypothesis, not enough to claim a universal property.

Key limitations:
- **Model coverage**: Only three instruction-tuned transformer models; broader coverage (base models, multilingual, different training data, more scales) is required.
- **Quantization mismatch**: Two models are 4-bit while one is bf16; quantization can affect distances and therefore ID estimates.
- **Small per-probe sample sizes**: Probes use 3–8 support texts; TwoNN is valid at small N but can be high-variance, especially per-probe. The mean across 439 probes may be stable, but this should be verified with confidence intervals and repeated runs.
- **Estimator + distance sensitivity**: Results may depend on TwoNN configuration (regression vs MLE), k-NN geodesic parameters, and probe construction choices.

Follow-up experiments to test generality:
1. Replicate across a wider model suite (base vs instruct, more families/sizes, multilingual).
2. Cross-check multiple ID estimators and report uncertainty (e.g., TwoNN regression vs MLE + bootstrap).
3. Stress-test probe construction (different corpora, different invariant probe sets, randomized controls).
4. Test whether the plateau regime correlates with cross-model transfer success (Paper 3) and with mid-layer geometric similarity metrics (e.g., CKA).

## 5. Related Work

**Intrinsic Dimension in Neural Networks**: Ansuini et al. (2019) measured ID in vision networks; we extend this to language models with semantic probes.

**Platonic Representation Hypothesis**: Huh et al. (2024) showed cross-model representation similarity; we propose a candidate mechanistic interpretation via early-layer ID collapse and a mid-layer plateau regime.

**Information Bottleneck**: Tishby & Zaslavsky (2015) proposed compression-relevance tradeoffs; we observe this as the cliff.

**Layer-wise Analysis**: Voita et al. (2019) and Jawahar et al. (2019) analyzed layer functions; we quantify this with ID.

## 6. Reproducibility

All experiments can be reproduced with ModelCypher:

```bash
# Single-layer dimensionality analysis
mc geometry atlas dimensionality-study /path/to/model --layer 2 --output json

# Full layer sweep
for layer in 0 1 2 3 4 5 6 8 12 16 20; do
  mc geometry atlas dimensionality-study /path/to/model --layer $layer --output json
done
```

**Reproducibility**: Run the commands above on any compatible model. Results are deterministic given the same model weights and probe corpus.

## 7. Conclusion

Across three tested transformer LLMs, we observe an early-layer intrinsic-dimension collapse followed by a low-ID plateau around ~1.4. Rather than claiming universality, we treat this as a concrete observation and a working hypothesis about representation geometry over depth. If the pattern holds more broadly, it may:

1. **Help explain representation convergence**: architectures may rapidly project onto a shared low-dimensional regime
2. **Support transfer learning**: mid-layer compatibility could be higher than early/late layers
3. **Suggest intervention layers**: geometric interventions may be most stable in the plateau regime

In these three models, the cliff location varies (L0→L1 vs L1→L2), while the plateau range is similar. Determining whether that similarity is a property of language, training, architecture, or the estimator requires broader replication.

## 8. Falsification Criteria

This working hypothesis would be weakened or refuted if broader tests show that:

1. ❌ A transformer model achieves competitive performance without showing a cliff
2. ❌ Plateau ID varies widely across comparable models when measured with multiple ID estimators and probe corpora
3. ❌ The effect disappears under modest changes to probe set construction, distance metric configuration, or quantization
4. ❌ Cross-architecture transfer success does not correlate at all with mid-layer geometric similarity in follow-up studies

## References

Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. *NeurIPS*. [arXiv:1905.12784](https://arxiv.org/abs/1905.12784).

Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports* 7, 12140. [DOI:10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y).

[Huh et al. (2024)](../docs/references/arxiv/Huh_2024_Platonic_Representation.pdf). The Platonic Representation Hypothesis. *ICML 2024*. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987).

Jawahar, G., Sagot, B., & Seddah, D. (2019). What does BERT learn about the structure of language? *ACL*. [arXiv:1905.05950](https://arxiv.org/abs/1905.05950).

Tishby, N., & Zaslavsky, N. (2015). Deep learning and the information bottleneck principle. *IEEE ITW*. [arXiv:1503.02406](https://arxiv.org/abs/1503.02406).

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting. *ACL*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418).
