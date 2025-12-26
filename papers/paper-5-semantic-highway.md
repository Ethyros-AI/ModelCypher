# Paper 5: The Semantic Highway

**Author**: Jason Kempf
**Affiliation**: EthyrosAI
**Date**: December 2025

> **Status**: Empirical validation with cross-architecture replication.

## Abstract

We discover a universal property of transformer language models: **all architectures compress input representations to the same low-dimensional manifold (~1.4 intrinsic dimension) within the first 1-2 layers, regardless of model family, size, or training data**. Using TwoNN intrinsic dimension estimation across 373 semantic probes, we demonstrate this "semantic highway" in three model families—Qwen (0.5B), Llama (3B), and Mistral (7B). The compression magnitude varies (41-79%), but the plateau value is invariant. We further show that harder semantic domains (higher initial ID) compress more aggressively (ρ = 0.832), suggesting the cliff acts as a normalizing bottleneck. These findings explain why cross-architecture transfer learning succeeds and predict optimal layers for geometric intervention.

## 1. Introduction

The Platonic Representation Hypothesis (Huh et al., 2024) observes that neural networks trained on different data converge to similar internal representations. Our prior work (Paper 1) validated this with CKA > 0.9 across model families. But *why* do representations converge? What mechanism forces independently trained models toward the same structure?

We propose the **Semantic Highway Hypothesis**:

> All sufficiently large transformer models compress tokenized input to a low-dimensional semantic manifold within the first few layers, maintain this representation through a "semantic highway" of middle layers, then expand for task-specific output formatting.

This paper provides empirical evidence for this hypothesis through cross-architecture intrinsic dimension measurement.

### 1.1 Contributions

1. **Discovery of the dimensionality cliff**: 41-79% intrinsic dimension collapse in layers 0-2 across all tested architectures.

2. **Universal plateau value**: Post-cliff ID stabilizes at 1.3-1.5 regardless of initial dimensionality or architecture.

3. **Complexity normalization**: Harder semantic domains (higher initial ID) experience proportionally greater compression (ρ = 0.832).

4. **Three-zone architecture**: Entry zone (variable ID) → Semantic highway (plateau) → Exit zone (expansion).

## 2. Methods

### 2.1 Intrinsic Dimension Estimation

We use the Two-Nearest Neighbors (TwoNN) method (Facco et al., 2017) to estimate local intrinsic dimension:

```
ID = log(N-1) / log(μ)
```

where μ is the ratio of second-nearest to first-nearest neighbor distances, computed via geodesic path lengths on a k-NN graph (k=10).

### 2.2 Semantic Probe Corpus

We analyze 373 probes from the UnifiedAtlas spanning 12 semantic domains:

| Domain | Probe Count | Examples |
|--------|-------------|----------|
| Mathematical | 51 | Fibonacci, primes, Catalan |
| Logical | 41 | Modus ponens, De Morgan |
| Computational | 73 | Gates, algorithms |
| Spatial | 15 | Left/right, near/far |
| Temporal | 33 | Past/future, duration |
| Moral | 20 | Right/wrong, virtue |
| Affective | 28 | Joy, fear, anger |
| Relational | 37 | Kinship, social roles |
| Mental | 20 | Think, know, believe |
| Linguistic | 21 | Syntax, semantics |
| Structural | 10 | Part/whole, containment |
| Philosophical | 24 | Existence, causation |

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

All three architectures exhibit dramatic ID collapse in early layers:

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

### 3.2 Universal Plateau Value

Despite different architectures, sizes, and cliff magnitudes, all models converge to the same plateau:

| Model | Cliff Layer | Cliff Magnitude | Plateau ID |
|-------|-------------|-----------------|------------|
| Qwen 0.5B | L1→L2 | 79.4% | **1.3** |
| Llama 3B | L0→L1 | 53.0% | **1.5** |
| Mistral 7B | L0→L1 | 40.6% | **1.4** |

**Mean plateau ID: 1.40 ± 0.10**

This invariance holds despite:
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

### 3.4 Three-Zone Architecture

We identify three distinct zones across all architectures:

1. **Entry Zone (L0)**: Variable dimensionality (2.3 - 7.4 ID)
   - Architecture-specific embedding representation
   - Tokenization artifacts present

2. **Semantic Highway (L1/L2 onwards)**: Universal plateau (~1.4 ID)
   - Compressed semantic representation
   - Architecture-invariant structure
   - Stable through middle layers

3. **Exit Zone (final layers)**: Expansion for output
   - Qwen L23: 8.15 ID (6× expansion)
   - Task-specific formatting
   - Vocabulary projection preparation

## 4. Discussion

### 4.1 Why the Cliff Exists

The cliff appears to implement **information bottleneck compression** (Tishby & Zaslavsky, 2015). Early layers discard architecture-specific tokenization artifacts to reach a universal semantic representation. The plateau value (~1.4 ID) may represent an optimal compression level for:
- Maintaining semantic distinctions
- Enabling compositionality
- Supporting generalization

### 4.2 Why the Plateau is Universal

The invariance of plateau ID across architectures suggests it reflects **structure in language itself**, not learned artifacts. All models compress to the same manifold because:
1. Human language has inherent semantic dimensionality
2. The compression-reconstruction tradeoff has a universal optimum
3. Transformer attention mechanisms converge to the same solution

### 4.3 Implications for Transfer Learning

The Semantic Highway explains why cross-architecture transfer works:
- **Shared highway**: Middle-layer representations are compatible across models
- **Different ramps**: Entry/exit zones are architecture-specific
- **LoRA efficiency**: Adapters modify highway traffic, not the road itself

Our Paper 3 results (65-78% skill retention on Qwen→Llama transfer) are explained by the shared plateau: both models use the same semantic highway, so geometric alignment succeeds.

### 4.4 Implications for Model Merging

The three-zone architecture predicts:
- **Early layer merging** is difficult (different entry ramps)
- **Middle layer merging** is safe (shared highway)
- **Late layer merging** is risky (different exit formatting)

This aligns with empirical observations that middle-layer SLERP merging outperforms early/late layer merging.

## 5. Related Work

**Intrinsic Dimension in Neural Networks**: Ansuini et al. (2019) measured ID in vision networks; we extend this to language models with semantic probes.

**Platonic Representation Hypothesis**: Huh et al. (2024) showed cross-model representation similarity; we provide a mechanistic explanation via the semantic highway.

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

We discover the **Semantic Highway**: a universal low-dimensional manifold (~1.4 ID) to which all transformer language models compress within the first 1-2 layers. This finding:

1. **Explains representation convergence**: All models hit the same bottleneck
2. **Explains transfer learning**: Shared highway enables cross-architecture compatibility
3. **Predicts optimal intervention points**: Highway layers are the target for geometric modification

The cliff location varies (L0→L1 vs L1→L2), but the destination is invariant. Different architectures take different on-ramps, but all merge onto the same semantic highway.

## 8. Falsification Criteria

This hypothesis would be falsified if:

1. ❌ A transformer model achieves competitive performance without showing a cliff
2. ❌ Plateau ID varies by more than 0.5 across architectures of similar scale
3. ❌ Middle-layer interventions are less effective than early-layer interventions
4. ❌ Cross-architecture transfer fails despite matched plateau ID

## References

Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. NeurIPS.

Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports.

Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. arXiv:2405.07987.

Jawahar, G., Sagot, B., & Seddah, D. (2019). What does BERT learn about the structure of language? ACL.

Tishby, N., & Zaslavsky, N. (2015). Deep learning and the information bottleneck principle. ITW.

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting. ACL.
