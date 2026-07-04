# Cross-Domain Geometry Synthesis

**Date:** 2026-02-03

> **DE-ORBIT NOTICE [2026-07-04]:** `plasma/` is unintegrated,
> torch-based, and unbaselined against published disruption predictors. It is
> not a ModelCypher product surface or validation evidence. TODO(owner):
> relocate or delete `plasma/` outside the flagship repo; any external "1s
> disruption lead time" claim requires a DisruptionBench baseline first.

## Abstract [EXPLORATORY]

Universal low-dimensional structure emerges across domains—but with domain-specific compression ratios. Plasma dynamics, LLM embeddings, and RL policies all live on manifolds far smaller than their measurement spaces, but the compression efficiency varies systematically:

| Domain | Typical Compression | Why |
|--------|---------------------|-----|
| **LLMs** | 0.5-2% | Billions of training tokens collapse redundancy |
| **RL policies** | ~10% | 100k episodes, task-focused |
| **Plasma** | 8-10% | Physics-constrained, no learning |

The ~9% ratio observed in plasma and RL is **not** a universal constant. It's what you get without massive training data. LLMs compress further because they've seen enough examples to find near-optimal representations.

---

## Methods

### Intrinsic Dimension Estimation

**TwoNN (Two Nearest Neighbors):**
- For each point, compute ratio μ = r₂/r₁ of distances to two nearest neighbors
- Intrinsic dimension: ID = 1 / mean(log(μ))
- Reference: Facco et al. (2017), "Estimating the intrinsic dimension of datasets"

**PCA Effective Dimension:**
- Compute eigenvalue spectrum λ₁, λ₂, ... λₙ
- Effective dimension: D_eff = (Σλᵢ)² / Σλᵢ² (participation ratio)
- Measures "how many dimensions contribute equally"

**Expansion Ratio:**
- Ratio of k-NN distances between successive timesteps/layers
- Measures local geometric change through processing/time

### Data Sources

| Source | Location | Description |
|--------|----------|-------------|
| LFM2-350M dimension profile | `docs/research/dimensional_hierarchy.md` | Layer-wise ID via TwoNN |
| MAST tokamak | `plasma/results/GEOMETRY_FINDINGS.md` | 5 shots, 44D diagnostics |
| TORAX simulator | `plasma/results/torax_scenarios.json` | 3 scenarios, 12-16D states |
| RL tearing avoidance | `plasma/notebooks/10_rl_policy_geometry.py` | 32D bottleneck analysis |
| Multi-model expansion | `data/experiments/archive/trajectory_analysis_synthesis.md` | DeepSeek-R1, LFM2-1.2B, LFM2-350M |

---

## Results

### Unified Dimension Table

| Domain | System | Measurement Dim | Intrinsic Dim | Ratio | Notes |
|--------|--------|-----------------|---------------|-------|-------|
| **Plasma** | MAST (real tokamak) | 44 | 3.54 ± 0.57 | **8.0%** | AMC diagnostics, 5 shots |
| **Plasma** | TORAX basic | 12 | 1.04 | **8.7%** | DeepMind simulator |
| **Plasma** | TORAX ITER rampup | 16 | 1.09 | **6.8%** | ITER-like scenario |
| **Plasma** | TORAX ITER PC | 15 | 1.14 | **7.6%** | Plasma current control |
| **RL** | Tearing policy bottleneck | 32 | ~3.3 | **~10%** | Nonlinear CNN features |
| **RL** | Tearing policy input (PCA) | 165 | ~8.5 | **~5%** | Linear compression |
| **LLM** | LFM2-350M highway core (L7-9) | 1024 | 4.7 | **0.5%** | Maximum compression |
| **LLM** | LFM2-350M entry ramp (L0-3) | 1024 | 19.3 | **1.9%** | Initial embedding |
| **LLM** | LFM2-350M exit ramp (L12-15) | 1024 | 16.4 | **1.6%** | Task-specific expansion |
| **LLM** | LFM2-350M highway edges (L4-6, 10-11) | 1024 | 21.0 | **2.1%** | Transition zones |
| **LLM** | GPT-2 embeddings | 768 | 10-50 | **1-7%** | Literature baseline |

### Training Volume vs Compression [EXPLORATORY]

| System | Training Data | Compression Ratio | Notes |
|--------|---------------|-------------------|-------|
| LLM (350M params) | ~Billions of tokens | 0.5-2% | Massive redundancy elimination |
| RL policy (tearing) | ~100k episodes | ~10% | Task-focused, limited coverage |
| MAST plasma | 0 (unsupervised) | 8% | Physics constraints only |
| TORAX sim | 0 (unsupervised) | 7-9% | Simulation physics only |

**Observation:** More training data → tighter compression. LLMs compress 5-20× more efficiently than physics-constrained systems without learned representations.

### Specialist vs Base Model Patterns

| Model Type | Example | Expansion Ratio Variance | Dimension Recovery |
|------------|---------|--------------------------|-------------------|
| **Base** | LFM2-350M | σ = 0.316 (high) | Yes (L15-16) |
| **General/Instruct** | LFM2-1.2B | σ = 0.073 (moderate) | Partial |
| **Specialist** | DeepSeek-R1-8B | σ = 0.000 (constant) | No |

Base models differentiate geometry by task type. Specialists collapse to fixed geometry.

---

## Discussion

### Why LLMs Compress More Than Plasma [CONJECTURAL]

**Hypothesis: Training data volume determines compression efficiency.**

1. **Redundancy discovery**: Each training example reveals another redundancy in the representation. With billions of tokens, LLMs discover most of the exploitable structure.

2. **Gradient-based optimization**: Backprop actively pushes representations toward efficient manifolds. Physics has no such pressure.

3. **Task diversity**: LLMs see diverse tasks, so the compression must preserve what's common across all of them. This forces tighter, more universal manifolds.

4. **No physics constraints**: Plasma is constrained by conservation laws, MHD equations, boundary conditions. LLMs only need to predict tokens—fewer hard constraints, more room for compression.

### Layer-wise vs Temporal Structure

LLMs have **spatial structure** (layers 0 → N):
```
Entry Ramp (2%)  →  Highway Core (0.5%)  →  Exit Ramp (1.6%)
     ↓                    ↓                      ↓
 Embed tokens       Compress to             Task-specific
 to semantics       universal core          expansion
```

Plasma has **temporal structure** (time 0 → T):
```
Startup → Steady-state → (Instability?) → Termination
   ↓            ↓               ↓              ↓
 High ID    Low ID (~8%)    ID spike?      Collapse
```

Both show compression then expansion, but the axis differs:
- LLMs: compression is spatial (middle layers)
- Plasma: compression is temporal (steady-state phase)

### Cross-Domain Tool Trial [EXPLORATORY]

The same geometric tools work across domains:

| Tool | LLM Finding | Plasma Finding | RL Finding |
|------|-------------|----------------|------------|
| **Intrinsic dimension** | Highway = 0.5% | Steady-state = 8% | Bottleneck = 10% |
| **Expansion ratio** | Task differentiation | Disruption precursor | Stability indicator |
| **Spectral entropy** | Layer complexity | Plasma state entropy | - |
| **Anomaly detection** | OOD detection | 5/7 disruptions found | - |

The tools are domain-agnostic. The interpretations are domain-specific.

### Prediction: Bigger Models -> Tighter Compression [CONJECTURAL]

If training volume drives compression, then:
- Larger LLMs (more training) should have tighter highway compression
- Models trained on more data should compress more than smaller models
- Specialist training (narrow domain) should maintain fixed geometry

Preliminary evidence from dimension recovery analysis:
- LFM2-350M: Highway ~0.5%
- Larger models tend toward smaller recovery ratios
- DeepSeek-R1 (heavily trained reasoning specialist): Constant geometry, no recovery

---

## Appendix: Raw Data Sources

### MAST Tokamak Shots

| Shot | Channels | Expansion | Local Dim | Notes |
|------|----------|-----------|-----------|-------|
| 30400 | 44 | 1.22 ± 0.82 | 3.31 ± 0.87 | |
| 30420 | 44 | 1.28 ± 0.82 | 3.08 ± 0.74 | Lowest dimension |
| 30440 | 44 | 1.39 ± 2.48 | 3.33 ± 0.95 | High volatility |
| 30460 | 44 | 1.28 ± 1.33 | 3.30 ± 0.99 | |
| 30473 | 44 | 1.08 ± 0.13 | 4.66 ± 0.87 | Smoothest, highest dim |

Source: `plasma/results/GEOMETRY_FINDINGS.md`

### TORAX Scenarios

```json
{
  "basic": {
    "n_diagnostics": 12,
    "mean_dimension": 1.04,
    "pc1_var": 99.96%
  },
  "iter_rampup": {
    "n_diagnostics": 16,
    "mean_dimension": 1.09,
    "pc1_var": 99.70%
  },
  "iter_pc": {
    "n_diagnostics": 15,
    "mean_dimension": 1.14,
    "pc1_var": 99.29%
  }
}
```

Source: `plasma/results/torax_scenarios.json`

### LLM Layer-wise Intrinsic Dimension

| Region | Layers | Mean ID | Compression |
|--------|--------|---------|-------------|
| Entry Ramp | 0-3 | 19.3 | 98.1% |
| Highway Edges | 4-6, 10-11 | 21.0 | 97.9% |
| **Highway Core** | **7-9** | **4.7** | **99.5%** |
| Exit Ramp | 12-15 | 16.4 | 98.4% |

Source: `docs/research/dimensional_hierarchy.md`

### Expansion Ratio by Model Type

| Model | Mean | Std | Classification |
|-------|------|-----|----------------|
| DeepSeek-R1-8B | 1.00 | 0.000 | Specialist |
| LFM2-1.2B | 1.07 | 0.073 | General |
| LFM2-350M | 1.40 | 0.316 | Base |

Source: `data/experiments/archive/trajectory_analysis_synthesis.md`

---

## Conclusion [EXPLORATORY]

1. **Low-dimensional structure is universal**: All measured systems (plasma, LLMs, RL) live on manifolds orders of magnitude smaller than their measurement spaces.

2. **Compression efficiency scales with training**: LLMs achieve 0.5-2% ratios vs 8-10% for physics-only systems. This 5-20× difference reflects billions of training examples discovering redundancy.

3. **The tools transfer**: Expansion ratio, intrinsic dimension, and spectral entropy work identically across domains. The measurements are domain-agnostic; the interpretations require domain knowledge.

4. **Structure differs by axis**: LLMs compress spatially (middle layers). Plasma compresses temporally (steady-state). RL compresses through learned bottlenecks. Same principle, different manifestation.

---

*Synthesis performed 2026-02-03 as part of ModelCypher cross-domain geometry research.*
