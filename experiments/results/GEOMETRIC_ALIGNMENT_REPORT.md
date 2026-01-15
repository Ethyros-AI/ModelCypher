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
| 4. Jailbreak Detection | Overall accuracy | **0.943** |
| 5. Alignment Transfer | Separation improvement | 0.0 |
| 6. Geometric Guardrails | Precision / FPR | 0.879 / 0.133 |

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

**Layer Selection**: Layer 15 selected by maximum separation (0.506).

### Mean Projections by Category

| Category | N | Mean Projection |
|----------|---|-----------------|
| Harmless | 50 | -0.463 |
| Harmful | 50 | +0.032 |
| Jailbreak | 40 | +0.0001 |

### Absolute Differences

- Harmful - Harmless: **0.495**
- Harmful - Jailbreak: **0.032**
- Jailbreak - Harmless: **0.463**

### Detection Metrics

Threshold: 95th percentile of harmless projections = **-0.255**

| Metric | Value |
|--------|-------|
| Accuracy | **0.943** |
| Precision | 0.977 |
| Recall | 0.933 |
| F1 | **0.955** |
| True Positives | 84 |
| False Positives | 2 |
| True Negatives | 48 |
| False Negatives | 6 |

### Jailbreak-Only Detection

| Metric | Value |
|--------|-------|
| Accuracy | 0.933 |
| Precision | 0.947 |
| Recall | 0.900 |
| F1 | 0.923 |

### Observation

Jailbreak prompts have mean projection (0.0001) nearly identical to harmful (0.032), both significantly higher than harmless (-0.463). Detection threshold at 95th percentile of harmless yields 94.3% accuracy.

---

## Experiment 5: Alignment Transfer

**Measurement**: Change in harmful/harmless separation after adding refusal direction to base model activations.

### Configuration

- Source: LFM2.5-1.2B-Instruct-bf16
- Target: LFM2-1.2B-bf16
- Steering strength: 1.0
- Layer: 15

### Results

| Metric | Instruct | Base (before) | Base (after) |
|--------|----------|---------------|--------------|
| Harmful/Harmless Separation | 0.501 | 0.835 | 0.835 |

### Per-Prompt Steering Effect

All 30 test prompts showed projection increase of exactly 1.0 (equal to steering strength), confirming the additive operation works mechanically.

### Observation

Separation unchanged because both harmful and harmless projections shift by identical amount. Base model already had higher separation (0.835) than instruct (0.501).

---

## Experiment 6: Geometric Guardrails

**Measurement**: Boundary violation detection using refusal projection threshold and distance from safe centroid.

**Layer Selection**: Layer 15 selected by maximum separation (0.552).

### Configuration

- Refusal threshold: 5th percentile of harmless projections = **-0.233**
- Safe radius: 95th percentile of harmless distances = **1.469**

(These percentiles are configuration parameters.)

### Detection Rates

| Category | N | Violation Rate |
|----------|---|----------------|
| Harmless | 30 | 0.133 |
| Harmful | 30 | 0.133 |
| Jailbreak | 40 | **0.625** |

### Detection Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.879 |
| Recall | 0.414 |
| F1 | 0.563 |
| True Positives | 29 |
| False Positives | 4 |
| True Negatives | 26 |
| False Negatives | 41 |

### Steering Recovery

Of 29 violations detected, 11 (37.9%) were brought within boundary by steering.

### Observation

Jailbreaks detected at 62.5% rate, higher than harmful (13.3%) or harmless (13.3%). The boundary-based approach primarily catches jailbreaks via distance violation.

---

## Method Notes

### Layer Selection (Fixed)

Layer selected by computing separation at each layer and choosing the layer with maximum separation. For this model, layer 15 had separation 0.506-0.552 (depending on train/test split).

### Threshold Derivation

Detection threshold is the 95th percentile of harmless projections. This controls false positive rate by construction.

### Boundary Percentiles

The 5th percentile (refusal threshold) and 95th percentile (safe radius) are configuration parameters. Different values produce different precision/recall tradeoffs.

### Sample Sizes

- Harmful prompts: 50
- Harmless prompts: 50
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
