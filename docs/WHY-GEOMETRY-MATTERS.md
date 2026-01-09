# Why Geometry Matters: The Proof

This document shows empirical procedures and output fields that demonstrate why geometric methods outperform naive approaches. Replace example values with your own runs.

---

## The Problem with Naive Model Merging

When you merge two models by averaging their weights, you're assuming knowledge is stored in the same "locations" in both models. It isn't.

```mermaid
graph LR
    subgraph Naive["Naive Merge: Average Weights"]
        A1[Model A Layer 12] -->|0.5| M1[Merged]
        B1[Model B Layer 12] -->|0.5| M1
        M1 -->|?| X1[Collision]
    end

    subgraph Geometric["Geometric Merge: Align First"]
        A2[Model A Layer 12] --> P[Procrustes Align]
        B2[Model B Layer 12] --> P
        P --> M2[Merged]
        M2 -->|preserved| Y[Both Skills Intact]
    end

    style X1 fill:#f99,stroke:#933
    style Y fill:#9f9,stroke:#393
```

**Procrustes alignment** rotates one model's weight space to match the other before merging. This preserves the geometric relationships between concepts.

---

## Empirical Results: Geometry vs. Naive Baselines

### Experiment: Merge Analysis Metrics (example fields; replace with your run)

`mc geometry interference predict` emits `globalMetrics`. Example comparison:

| Pair | meanOverlap | meanCka | meanCurvatureDivergence | meanDistance |
|------|-------------|---------|-------------------------|--------------|
| Source vs Target | ... | ... | ... | ... |
| Merged vs Target | ... | ... | ... | ... |

**How to read the numbers (no vibes):**

- **meanOverlap**: Mean overlap across domain probes.
- **meanCka**: Alignment of domain activations (CKA).
- **meanCurvatureDivergence**: Mean curvature divergence across domains.
- **meanDistance**: Mean distance metric from merge analysis.

Compare against your own baselines; do not use fixed thresholds.

---

## Why Does This Happen?

### The Rotation Problem

Two models trained on the same data can learn identical knowledge, but store it in rotated coordinate systems.

```
Model A: "cat" → [0.8, 0.2, 0.1]
Model B: "cat" → [0.2, 0.8, 0.1]  ← Same concept, rotated representation
```

Averaging these gives `[0.5, 0.5, 0.1]` — which is neither cat. Procrustes finds the rotation matrix that aligns them first.

### The Interference Problem

When concepts overlap in merged weight space, they interfere. ModelCypher predicts this *before* you merge:

```bash
mc geometry interference predict model-A model-B
```

Output (excerpt):
```
SPATIAL:
  Mean Overlap: 0.23
  Domain CKA: 0.94
  Mean Curvature Divergence: 0.15
  Mean Distance: 0.32
```

If high interference is predicted, you can use **null-space projection** to merge only in directions that don't collide.

---

## Safety: Pre-Emission Detection

Traditional safety filters check *after* the model generates a token. ModelCypher detects distress *during* the forward pass.

| Input | baselineEntropy | attackEntropy | deltaH | thresholdExceedance |
|-------|-----------------|--------------|--------|---------------------|
| "Explain math" | ... | ... | ... | ... |
| "Adversarial Jailbreak" | ... | ... | ... | ... |

**What this means:**
- ΔH is a raw instability signal; compare against baselines for the model family.
- Use it to localize geometry stress during inference, not as a causal claim.

---

## The Mathematical Foundation

ModelCypher isn't inventing new math. It applies established theory:

| Concept | Source | Application |
|---------|--------|-------------|
| Riemannian Geometry | Amari (2000) | Measuring curvature of activation manifolds |
| Procrustes Analysis | Gower (1975) | Aligning weight spaces before merging |
| CKA Similarity | Kornblith (2019) | Comparing representations across architectures |
| Persistent Homology | Naitzat (2020) | Topological fingerprints of models |
| Information Geometry | Fefferman (2016) | Manifold hypothesis for neural networks |

See [papers/](../papers/) for the full research foundation.

---

## Recent model merging literature (2024–2025)

- [Nobari et al. (2025)](references/arxiv/AIM_2025_Activation_Informed_Merging.pdf). Activation-Informed Merging of Large Language Models. [arXiv:2502.02421](https://arxiv.org/abs/2502.02421)
- FW-Merging (ICCV 2025). Scaling Model Merging with Frank-Wolfe Optimization. https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_FW-Merging_Scaling_Model_Merging_with_Frank-Wolfe_Optimization_ICCV_2025_paper.pdf
- [Yang et al. (2024)](references/arxiv/SuperMerge_2024_Gradient_Based_Model_Merging.pdf). SuperMerge: An Approach For Gradient-Based Model Merging. [arXiv:2412.10416](https://arxiv.org/abs/2412.10416)
- [Fang et al. (2025)](references/arxiv/GW_Feature_Alignment_2025_Model_Merging.pdf). Efficient Multi-Task Inferencing: Model Merging with Gromov-Wasserstein Feature Alignment. [arXiv:2503.09774](https://arxiv.org/abs/2503.09774)
- NEig-OWM (2025). Null-space orthogonal weight modification to preserve prior tasks. https://doi.org/10.1016/j.eswa.2025.127468

---

## Reproduce These Results

```bash
# Verify geometric invariants
mc geometry waypoint validate

# Run merge comparison
mc merge \
    --source ./model-A \
    --target ./model-B \
    --output-dir ./merged-geometric

# For a naive baseline, run your preferred linear merge tool and save to ./merged-naive.

# Compare results
mc model probe ./merged-geometric --output text
mc model probe ./merged-naive --output text
```

---

## The Bottom Line

> **Benchmarks measure outputs. Geometry measures structure.**
>
> You can game outputs. You can't fake topology.

If these numbers don't match what you see, [file an issue](https://github.com/Ethyros-AI/ModelCypher/issues).
