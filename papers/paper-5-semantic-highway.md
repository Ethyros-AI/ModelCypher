# Paper 5: The Semantic Highway (Preliminary Observation)

**Author**: Jason Kempf
**Affiliation**: EthyrosAI
**Date**: December 2025

> **Status**: Draft observation; reproduction pending.

## Abstract

Pilot runs on three transformer language models (Qwen, Llama, Mistral) suggested a consistent pattern in intrinsic dimension (ID) profiles measured with TwoNN: (1) a sharp early-layer drop in ID ("dimensionality cliff"), and (2) a mid-layer low-ID plateau. The magnitude and layer index of the cliff varied by architecture. We present these results as an observation, not a universal law, and propose a working hypothesis: early transformer layers rapidly project tokenized representations onto a low-dimensional conceptual manifold, after which representations evolve primarily within that manifold. We outline follow-up tests needed to determine how broadly this pattern holds across architectures, scales, languages, and training regimes.

## 1. Introduction

The Platonic Representation Hypothesis (Huh et al., 2024) suggests that independently trained neural networks converge to similar internal representations. Our prior work ([Paper 1](paper-1-invariant-semantic-structure.md)) reported high cross-family similarity on several anchor sets (reproduction pending). An open question remains: *what dynamics produce this convergence across architectures?*

This paper reports a simple empirical observation about *intrinsic dimension over depth* in three transformer LLMs. When we measure ID using TwoNN across a fixed semantic probe corpus, all three models show an early-layer ID collapse followed by a low-ID plateau. We use "semantic highway" as a shorthand label for this plateau regime, but treat it as a **working hypothesis** rather than a universal property.

### 1.1 Contributions

1. **Observed early-layer cliff (3 models)**: A sharp reduction in intrinsic dimension within the first layers for all tested models (reproduction pending).

2. **Observed low-ID plateau (3 models)**: A mid-layer low-ID plateau for all tested models (reproduction pending).

3. **Domain-dependent compression (Qwen)**: Pilot analysis suggested higher initial domain ID correlates with stronger compression (reproduction pending).

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

We analyze probes from the UnifiedAtlas spanning semantic domains:

| Domain | Examples |
|--------|----------|
| Mathematical | Fibonacci, primes, Catalan |
| Logical | Modus ponens, De Morgan |
| Computational | Gates, algorithms |
| Spatial | Left/right, near/far |
| Temporal | Past/future, duration |
| Moral | Right/wrong, virtue |
| Affective | Joy, fear, anger |
| Relational | Kinship, social roles |
| Mental | Think, know, believe |
| Linguistic | Syntax, semantics |
| Structural | Part/whole, containment |
| Philosophical | Existence, causation |

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

> Results are not included in this draft. Pilot runs exist but are not reproduced; rerun using the protocol and record raw data before drawing conclusions.

## 4. Discussion

### 4.1 Possible Explanations for the Cliff

One plausible mechanism is **information bottleneck compression** (Tishby & Zaslavsky, 2015): early layers discard tokenization- and architecture-specific degrees of freedom while retaining semantics needed for downstream behavior. Under this view, the plateau reflects a stable low-dimensional regime that supports:
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

Our [Paper 3](paper-3-cross-architecture-transfer.md) protocol explores transfer; if the plateau generalizes, it may help explain cross-architecture compatibility. This does not establish causality.

### 4.4 Implications for Model Merging

This three-regime interpretation suggests (but does not guarantee) that:
- **Early layer merging** is difficult (different entry ramps)
- **Middle layer merging** is safe (shared highway)
- **Late layer merging** is risky (different exit formatting)

This suggests a hypothesis that mid-layer merges may be more stable than early/late layer merges, but empirical validation is required.

### 4.5 Limitations and Follow-Up Experiments

This document reports a pattern observed in **three** models. That is enough to motivate a hypothesis, not enough to claim a universal property.

Key limitations:
- **Model coverage**: Only three instruction-tuned transformer models; broader coverage (base models, multilingual, different training data, more scales) is required.
- **Quantization mismatch**: Two models are 4-bit while one is bf16; quantization can affect distances and therefore ID estimates.
- **Small per-probe sample sizes**: Probes use few support texts; TwoNN is valid at small N but can be high-variance, especially per-probe. The mean across all probes may be stable, but this should be verified with confidence intervals and repeated runs.
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
# Full dimensionality study (per-layer results)
poetry run mc --output json geometry atlas dimensionality-study /path/to/model --include-results
```

**Reproducibility**: Run the command above on any compatible model. Determinism depends on the same model weights, probe corpus, and backend configuration.

## 7. Conclusion

Pilot runs suggested an early-layer intrinsic-dimension collapse followed by a low-ID plateau. Rather than claiming universality, we treat this as a concrete observation and a working hypothesis about representation geometry over depth. If the pattern holds more broadly, it may:

1. **Help explain representation convergence**: architectures may rapidly project onto a shared low-dimensional regime
2. **Support transfer learning**: mid-layer compatibility could be higher than early/late layers
3. **Suggest intervention layers**: geometric interventions may be most stable in the plateau regime

In these pilot runs, the cliff location varied across models. Determining whether the plateau similarity is a property of language, training, architecture, or the estimator requires broader replication.

## 8. Falsification Criteria

This working hypothesis would be weakened or refuted if broader tests show that:

1. ❌ A transformer model achieves competitive performance without showing a cliff
2. ❌ Plateau ID varies widely across comparable models when measured with multiple ID estimators and probe corpora
3. ❌ The effect disappears under modest changes to probe set construction, distance metric configuration, or quantization
4. ❌ Cross-architecture transfer success does not correlate at all with mid-layer geometric similarity in follow-up studies

## References

[Ansuini et al. (2019)](../docs/references/arxiv/Ansuini_2019_Intrinsic_dimension_data_representations_deep_neural.pdf). Intrinsic dimension of data representations in deep neural networks. *NeurIPS*. [arXiv:1905.12784](https://arxiv.org/abs/1905.12784).

Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports* 7, 12140. [DOI:10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y).

[Huh et al. (2024)](../docs/references/arxiv/Huh_2024_Platonic_Representation.pdf). The Platonic Representation Hypothesis. *ICML 2024*. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987).

[Jawahar et al. (2019)](../docs/references/arxiv/Jawahar_2019_BERT_Structure.pdf). What does BERT learn about the structure of language? *ACL 2019*. [ACL Anthology](https://aclanthology.org/P19-1356/).

[Tishby & Zaslavsky (2015)](../docs/references/arxiv/Tishby_2015_Deep_Learning_Information_Bottleneck_Principle.pdf). Deep learning and the information bottleneck principle. *IEEE ITW*. [arXiv:1503.02406](https://arxiv.org/abs/1503.02406).

[Voita et al. (2019)](../docs/references/arxiv/Voita_2019_Analyzing_MultiHead_SelfAttention_Specialized_Heads_Do.pdf). Analyzing multi-head self-attention: Specialized heads do the heavy lifting. *ACL*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418).
