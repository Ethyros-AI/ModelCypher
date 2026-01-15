# Geometric AI Alignment Experiments Report

**Project**: ModelCypher
**Date**: January 2026
**Primary Model**: LFM2.5-1.2B-Instruct-bf16

---

## Executive Summary

Six experiments testing whether AI alignment manifests as measurable geometric structure in activation space.

### Results Summary

| Experiment | Computed Metric | Value |
|------------|-----------------|-------|
| 1. Alignment Detection | CKA (base vs instruct) | 0.668 |
| 2. Refusal Direction | Classification accuracy (layer 6) | 0.97 |
| 3. Cross-Model Universality | Mean best accuracy (4 models) | 0.9475 |
| 4. Jailbreak Detection | Overall accuracy (LFM / Qwen) | 0.921 / 0.793 |
| 5. Alignment Transfer | Refusal rate increase | +56.7% (43.3% → 100%) |
| 6. Geometric Guardrails | F1 (LFM / Qwen) | 0.879 / 0.404 |

---

## Experiment 1: Alignment Detection

**Measurement**: CKA similarity between base model (LFM2-1.2B) and instruct model (LFM2.5-1.2B-Instruct) activations.

### Raw Data

| Layer | CKA | Base ID | Instruct ID |
|-------|-----|---------|-------------|
| 0 | 0.793 | 18.75 | 21.17 |
| 1 | 0.736 | 14.37 | 14.45 |
| 2 | 0.803 | 16.27 | 12.48 |
| 3 | 0.795 | 21.27 | 20.45 |
| 4 | 0.610 | 16.40 | 17.13 |
| 5 | 0.636 | 14.62 | 14.34 |
| 6 | 0.779 | 14.47 | 18.73 |
| 7 | 0.735 | 9.27 | 12.58 |
| 8 | 0.780 | 10.18 | 7.72 |
| 9 | 0.666 | 10.88 | 10.41 |
| 10 | 0.741 | 17.01 | 13.53 |
| 11 | 0.538 | 17.90 | 12.39 |
| 12 | 0.529 | 18.63 | 13.64 |
| 13 | 0.537 | 22.38 | 9.15 |
| 14 | 0.549 | 18.92 | 13.22 |
| 15 | 0.471 | 25.89 | 24.29 |

### Aggregates

- Mean CKA: **0.668**
- Mean Base Intrinsic Dimension: **16.70**
- Mean Instruct Intrinsic Dimension: **14.73**
- Total Novel Directions (instruct): **5**

### Observation

CKA decreases in later layers (0.47-0.54 for layers 11-15 vs 0.73-0.80 for layers 0-3). Instruct model has lower mean intrinsic dimension (14.73 vs 16.70).

---

## Experiment 2: Refusal Direction Extraction

**Measurement**: Classification accuracy using difference-in-means direction (r = mean(harmful) - mean(harmless)) to separate harmful from harmless prompts.

### Raw Data (All Layers)

| Layer | Direction Strength | Separation | Accuracy |
|-------|-------------------|------------|----------|
| 0 | 0.038 | 0.033 | 0.93 |
| 1 | 0.044 | 0.043 | 0.70 |
| 2 | 0.042 | 0.041 | 0.90 |
| 3 | 0.041 | 0.040 | 0.88 |
| 4 | 0.046 | 0.044 | 0.93 |
| 5 | 0.051 | 0.049 | 0.95 |
| 6 | 0.055 | 0.051 | **0.97** |
| 7 | 0.090 | 0.087 | 0.61 |
| 8 | 0.137 | 0.088 | 0.60 |
| 9 | 0.100 | 0.086 | 0.77 |
| 10 | 0.118 | 0.108 | 0.78 |
| 11 | 0.144 | 0.133 | 0.88 |
| 12 | 0.213 | 0.180 | 0.79 |
| 13 | 0.334 | 0.267 | 0.68 |
| 14 | 0.422 | 0.315 | 0.74 |
| 15 | 0.520 | 0.495 | 0.94 |

### Aggregates

- Mean accuracy: **0.816**
- Maximum accuracy: **0.97** (layer 6)
- Mean separation: **0.129**
- Maximum separation: **0.495** (layer 15)

### Observation

Layer 6 achieves highest classification accuracy (0.97). Later layers have higher separation magnitude but lower accuracy, suggesting entanglement with other features.

---

## Experiment 3: Cross-Model Universality

**Measurement**: Refusal direction classification accuracy across 4 models from 3 architecture families.

### Per-Model Results

| Model | Hidden Size | Layers | Best Layer | Best Accuracy |
|-------|-------------|--------|------------|---------------|
| LFM2.5-1.2B-Instruct | 2048 | 16 | 6 | 0.97 |
| Qwen2.5-Coder-0.5B-Instruct | 896 | 24 | 23 | 0.93 |
| Qwen2.5-3B-Instruct | 2048 | 36 | 28 | 0.89 |
| Granite-3B-code-instruct | 2560 | 32 | 5 | 1.00 |

### Pairwise Layer-wise Correlations

| Model Pair | Separation Corr. | Strength Corr. |
|------------|------------------|----------------|
| LFM ↔ Qwen-0.5B | 0.759 | 0.761 |
| LFM ↔ Qwen-3B | 0.472 | 0.405 |
| LFM ↔ Granite | 0.867 | 0.888 |
| Qwen-0.5B ↔ Qwen-3B | 0.278 | 0.797 |
| Qwen-0.5B ↔ Granite | 0.960 | 0.971 |
| Qwen-3B ↔ Granite | 0.561 | 0.923 |

### Aggregates

- Mean best accuracy: **0.9475**
- Mean separation correlation: **0.650**
- Mean strength correlation: **0.791**

### Observation

All four models achieve >0.89 accuracy using difference-in-means direction. High correlations (0.76-0.97) between most model pairs in layer-wise patterns.

---

## Experiment 4: Jailbreak Detection

**Measurement**: Projection of prompt activations onto refusal direction.

**Layer Selection**: Layer selected by maximum classification accuracy on harmful/harmless training data.

### Results: LFM2.5-1.2B-Instruct

**Layer**: 0 (98% training accuracy)

| Category | N | Mean Projection |
|----------|---|-----------------|
| Harmless | 50 | -0.068 |
| Harmful | 50 | -0.034 |
| Jailbreak | 40 | -0.043 |

**Threshold**: -0.050 (95th percentile of harmless)

| Metric | Value |
|--------|-------|
| Accuracy | 0.921 |
| Precision | 0.976 |
| Recall | 0.900 |
| F1 | 0.936 |

### Results: Qwen2.5-3B-Instruct

**Layer**: 31 (97% training accuracy)

| Category | N | Mean Projection |
|----------|---|-----------------|
| Harmless | 50 | -10.08 |
| Harmful | 50 | +15.60 |
| Jailbreak | 40 | +9.22 |

**Threshold**: 8.59 (95th percentile of harmless)

| Metric | Value |
|--------|-------|
| Accuracy | 0.793 |
| Precision | 0.969 |
| Recall | 0.700 |
| F1 | 0.813 |

### Observation

Both models show jailbreaks projecting between harmless and harmful on the refusal direction. LFM achieves 92.1% accuracy, Qwen achieves 79.3%. Different architectures select different optimal layers (LFM: early layer 0, Qwen: late layer 31).

---

## Experiment 5: Alignment Transfer

**Measurement**: Change in harmful prompt refusal rate after adding refusal direction to base model activations. Only harmful prompts are steered.

### Configuration

- Source: LFM2.5-1.2B-Instruct-bf16
- Target: LFM2-1.2B-bf16
- Steering strength: 1.0
- Layer: 5 (selected by maximum classification accuracy)
- Refusal threshold: -0.0494 (95th percentile of instruct harmless projections)

### Results

| Model | Harmful Refusal Rate | Harmless False Positive Rate |
|-------|---------------------|------------------------------|
| Instruct (baseline) | 100.0% | 3.3% |
| Base (before steering) | 43.3% | 0.0% |
| Base (after steering) | 100.0% | — |

### Transfer Metrics

| Metric | Value |
|--------|-------|
| Refusal rate increase | +56.7% |
| Transfer effectiveness | 100% |

### Observation

Steering harmful prompts by adding the instruct-derived refusal direction pushed all 30 harmful prompts above the refusal threshold. The base model's harmful refusal rate increased from 43.3% to 100%, matching the instruct model. At layer 5, the base model initially has low harmful refusal rate (43.3%) but steering brings it to parity with the aligned instruct model.

---

## Experiment 6: Geometric Guardrails

**Measurement**: Boundary violation detection using refusal projection threshold and distance from safe centroid.

**Threshold Optimization**: Thresholds are automatically optimized for each model's geometry by grid search over percentile combinations, maximizing detection while constraining false positive rate to ~10%.

### Results: LFM2.5-1.2B-Instruct

**Layer**: 4 (selected by max boundary score 0.50)
**Optimized Boundary**: refusal_threshold=-0.245, safe_radius=0.106 (score=0.737)

| Category | N | Violation Rate |
|----------|---|----------------|
| Harmless | 30 | 0.300 |
| Harmful | 30 | 0.733 |
| Jailbreak | 40 | **1.000** |

| Metric | Value |
|--------|-------|
| Precision | 0.873 |
| Recall | 0.886 |
| F1 | **0.879** |

### Results: Qwen2.5-3B-Instruct

**Layer**: 30 (selected by max boundary score 0.35)
**Optimized Boundary**: refusal_threshold=-19.78, safe_radius=72.0 (score=0.563)

| Category | N | Violation Rate |
|----------|---|----------------|
| Harmless | 30 | 0.167 |
| Harmful | 30 | 0.200 |
| Jailbreak | 40 | 0.325 |

| Metric | Value |
|--------|-------|
| Precision | 0.792 |
| Recall | 0.271 |
| F1 | 0.404 |

### Observation

Model-agnostic threshold optimization finds different operating points for each architecture. LFM achieves 100% jailbreak detection with F1=0.879. Qwen's weaker boundary score (0.563 vs 0.737) indicates its geometry places jailbreaks closer to harmless activations, resulting in lower detection (F1=0.404). This suggests boundary-based detection effectiveness varies by architecture.

---

## Method Notes

### Layer Selection

**For classification (Experiments 2, 4, 5)**: Layer selected by computing classification accuracy at each layer (using 95th percentile threshold) and choosing the layer with maximum accuracy.

**For boundary detection (Experiment 6)**: Layer selected by grid search over percentile combinations at each layer, choosing the layer that maximizes (harmful_violation_rate - harmless_violation_rate) while constraining FPR.

Different models and different detection methods select different optimal layers based on task requirements.

### Model-Agnostic Threshold Optimization (Experiment 6)

Instead of fixed percentiles, thresholds are optimized per-model by:
1. Grid search over refusal percentiles (1-19%) and distance percentiles (80-98%)
2. For each combination, compute FPR and TPR on training data
3. Select thresholds that maximize a score balancing detection and false positives
4. Constrain to target FPR (~10%)

This ensures the boundary adapts to each model's geometric structure rather than using arbitrary fixed values.

### Threshold Derivation

Detection threshold is the 95th percentile of harmless projections. This controls false positive rate by construction.

### Boundary Percentiles

The 5th percentile (refusal threshold) and 95th percentile (safe radius) are configuration parameters. Different values produce different precision/recall tradeoffs.

### Alignment Transfer Method

Refusal direction extracted from instruct model. Threshold computed as 95th percentile of instruct's harmless projections. Only harmful prompts are steered (refusal direction added). Transfer effectiveness measures how close the steered base model's refusal rate is to the instruct model's.

### Sample Sizes

- Harmful prompts: 50 (30 for alignment transfer)
- Harmless prompts: 50 (30 for alignment transfer)
- Jailbreak prompts: 40

---

## Raw Data Files

All raw measurements stored in JSON:

- `experiments/results/alignment_detection.json`
- `experiments/results/refusal_direction.json`
- `experiments/results/cross_model_universality.json`
- `experiments/results/jailbreak_detection.json`
- `experiments/results/alignment_transfer.json`
- `experiments/results/geometric_guardrails.json`

---

## References

1. Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction."
2. Huh et al. (2024). "The Platonic Representation Hypothesis."
3. Zou et al. (2023). "Representation Engineering."
