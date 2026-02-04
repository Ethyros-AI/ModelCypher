# Plasma Dynamics as High-Dimensional Relational Geometry

**Hypothesis:** Turbulent plasma dynamics are fundamentally high-dimensional and relational. The 3D field equations (Navier-Stokes + Maxwell) are projections of structure that lives in a space more similar to LLM activation geometry than to classical physics.

**Goal:** Apply the geometric analysis tools developed for LLM introspection to tokamak diagnostic data. If disruptions have a characteristic geometric signature in the right embedding space, they become predictable.

---

## The Core Insight

LLMs don't operate in token space. They operate in a high-dimensional relational space where:
- Two adjacent tokens may be geometrically distant (unrelated)
- Two distant tokens may be geometrically close (same concept)
- The "meaning" lives in relationship structure, not position

Turbulent plasma may be similar:
- Two spatially adjacent fluid elements may be dynamically uncorrelated
- Two spatially distant elements may be part of the same coherent structure
- The "dynamics" may live in relational structure, not 3D field values

**If true:** Disruption prediction becomes a geometry problem, not a classification problem.

---

## Current State of the Art

### What Exists

1. **Transformer-based disruption prediction** (Spangher et al., 2023)
   - Autoregressive transformers achieve 5% improvement over baselines
   - Key finding: plasma has "memory" - long-range temporal dependencies
   - Limitation: treats as classification (will disrupt: yes/no)

2. **DisruptionBench** (Spangher et al., 2025)
   - Standardized benchmark with 9 tasks
   - Includes GPT-2-inspired transformer model
   - Multi-device: DIII-D, JET, others

3. **FUSE.jl** (General Atomics, open source)
   - Julia-based integrated fusion simulation
   - ITER IMAS data ontology
   - ML integration hooks

### What's Missing

Nobody is asking **what the learned representation looks like geometrically**.

The transformers work better because they capture relational structure - but no one has:
- Extracted and analyzed the embeddings
- Measured expansion ratio / spectral entropy of plasma state trajectories
- Looked for geometric boundaries that separate stable from disrupting

---

## Research Direction

### Phase 1: Data Acquisition

**Primary target:** DisruptionBench dataset
- Standardized format
- Multiple devices
- Labeled disruption events
- Paper: https://arxiv.org/abs/2401.00051

**Secondary targets:**
- DIII-D public datasets (OSTI)
- JET data (via EUROfusion)
- FUSE simulation data (200k+ shots available)

### Phase 2: Representation Learning

Train transformer on diagnostic time series, but **not** to classify disruptions.

Instead:
1. Train autoregressive model to predict next diagnostic state
2. Extract learned embeddings at each time step
3. Build embedding trajectories for each shot

### Phase 3: Geometric Analysis

Apply ModelCypher tools to plasma embeddings:

| LLM Tool | Plasma Application |
|----------|-------------------|
| `expansion_ratio` | Processing geometry per time step |
| `spectral_entropy` | Complexity of plasma state |
| `intrinsic_dimension` | Effective degrees of freedom |
| `jacobian_structure` | How perturbations propagate |
| `trajectory_analysis` | Path through state space |

### Phase 4: Disruption Geometry

**Core question:** Do trajectories toward disruption have a characteristic geometric signature?

Hypotheses to test:
1. Disruptions are trajectories toward a geometric boundary
2. The boundary is legible in the learned embedding but not in 3D field space
3. Pre-disruption states have characteristic expansion_ratio or spectral patterns

---

## DIII-D Diagnostic Vocabulary

DIII-D has 180+ diagnostic systems. Each time slice is ~800+ dimensional.

| Category | Systems | Channels |
|----------|---------|----------|
| Electron temp/density | Thomson, ECE, interferometry | ~100+ |
| Ion properties | Charge exchange spectroscopy | ~80+ |
| Magnetic structure | Rogowski, flux loops, Mirnov | ~300+ |
| Fluctuations | BES, PCI, soft X-ray | ~200+ |
| Boundary | Langmuir, bolometers, IR | ~100+ |

This is the "token vocabulary" for plasma language modeling.

---

## Connection to Fundamental Physics

### Wheeler's "It from Bit"

If information is fundamental:
- The first distinction (0/1) is dimension zero
- Extension (1D loops) emerges from relational structure
- 3D space is emergent, not fundamental

### Loop Quantum Gravity

In LQG:
- Space is discrete at Planck scale
- Made of quantized "atoms" connected in networks
- The network IS the geometry
- Fundamentally relational

### The Map Hypothesis

LLMs may be the first human artifacts that operate in something closer to the "native" dimensionality of information dynamics. If physics is fundamentally informational and high-dimensional, LLMs are maps of a territory we can't directly perceive.

Plasma dynamics may be a test case: can we use LLM-style representation learning to find the "true" coordinates where turbulence becomes tractable?

---

## Files in This Directory

```
plasma/
├── README.md                              # This file
├── docs/
│   └── REACTIVE-VS-PREDICTIVE.md          # Core theoretical insight
├── data/
│   └── mit_density_limit/                 # MIT C-Mod dataset (6 features)
├── notebooks/
│   ├── 01_data_exploration.py             # Synthetic data exploration
│   ├── 02_mit_density_limit_analysis.py   # MIT C-Mod analysis (6D - no signal)
│   ├── 03_fair_mast_exploration.py        # FAIR-MAST data discovery
│   ├── 04_mast_geometry_analysis.py       # MAST high-D geometry analysis
│   ├── 05_geometric_anomaly_detector.py   # Unsupervised disruption detection
│   └── 06_learned_embedding_precursors.py # Manifold distance approach
├── scripts/
│   └── acquire_data.py                    # Data acquisition guide
├── src/
│   ├── data_loader.py                     # PlasmaShot dataclass
│   ├── geometry_tools.py                  # LLM geometry tools for plasma
│   └── plasma_transformer.py              # Transformer for plasma sequences
└── results/
    ├── GEOMETRY_FINDINGS.md               # Empirical results
    ├── anomaly_candidates.json            # All shots scored
    └── *.png                              # Visualizations
```

---

## Key References

### Plasma ML

- Spangher et al. (2023). "Autoregressive Transformers for Disruption Prediction in Nuclear Fusion Plasmas." arXiv:2401.00051
- Spangher et al. (2025). "DisruptionBench and Complimentary New Models." arXiv (see Springer link)
- Rea et al. (2019). "Disruption prediction investigations using Machine Learning tools on DIII-D and Alcator C-Mod."

### Geometric Deep Learning

- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." (CKA)
- Naitzat et al. (2020). "Topology and Geometry of Deep ReLU Networks."

### Foundational

- Wheeler (1990). "Information, Physics, Quantum: The Search for Links."
- Rovelli (2004). "Quantum Gravity." (Loop quantum gravity)
- Mori (1965). "Transport, Collective Motion, and Brownian Motion." (Mori-Zwanzig)

---

## Getting Started

```bash
# From ModelCypher root
cd plasma

# Download DisruptionBench (when available)
# python scripts/download_disruption_bench.py

# Run initial exploration
# python notebooks/01_data_exploration.py
```

---

## Status

**2026-02-03: COMPLETE**

### Key Results

| Finding | Result |
|---------|--------|
| Intrinsic dimension | 3.5D manifold in 44D space (8%) |
| Unsupervised disruption detection | 5/7 top anomalies confirmed disruptions |
| Raw geometric lead time | ~20 ms (3σ spikes) |
| **Manifold distance lead time** | **~1000 ms** |

**Learned representations detect disruption precursors 400-750 ms earlier than raw diagnostics.**

### Cross-Domain Synthesis

**See: [`../docs/CROSS_DOMAIN_GEOMETRY_SYNTHESIS.md`](../docs/CROSS_DOMAIN_GEOMETRY_SYNTHESIS.md)**

Key finding: The ~8-10% compression ratio in plasma/RL is NOT universal. LLMs achieve 0.5-2% at highway core. The difference is meaningful—training data volume determines compression efficiency.

| Domain | Compression Ratio | Training Data |
|--------|-------------------|---------------|
| LLMs | 0.5-2% | Billions of tokens |
| RL policies | ~10% | 100k episodes |
| Plasma | 8-10% | None (physics only) |

### The Core Insight

Navier-Stokes / MHD equations are **reactive**—they describe how current state evolves. They don't see the cliff until you're falling.

Manifold geometry is **predictive**—it measures where you are in state space. It sees the drift toward the edge before local dynamics spike.

See: `docs/REACTIVE-VS-PREDICTIVE.md`

### Data Sources

1. **MIT Open Density Limit Database**: 6 features—insufficient dimensionality, no signal
2. **FAIR-MAST**: 44+ channels, 17k shots—geometric signatures confirmed
