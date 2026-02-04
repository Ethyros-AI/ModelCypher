# Reactive Equations, Predictive Geometry

## The Core Insight

Stable plasma operation requires continuous real-time adjustment. The control system isn't maintaining a fixed state—it's navigating a high-dimensional manifold, correcting every millisecond to stay within the stable region.

The Navier-Stokes and MHD equations that govern plasma dynamics are **reactive**: they describe how the current state evolves into the next state. Local rules, local causality. They don't predict—they respond.

But disruptions happen when **global constraints** fail: energy balance, current profile stability, pressure limits. These constraints define the topology of the stable manifold. The local equations don't see the cliff until you're falling off it.

This is why learned representations give earlier warning: they measure **where you are** in state space, not just **what's happening locally**.

---

## Empirical Results

Analyzing MAST tokamak data with 44 diagnostic channels:

### Finding 1: Low-Dimensional Manifold

Plasma dynamics live on a ~3.5D manifold within 44D measurement space. Only 8% of the measurement dimensionality captures the actual degrees of freedom.

### Finding 2: Geometric Disruption Signatures

Unsupervised anomaly detection using expansion ratio, spectral entropy, and local dimension identified disruptions without labels. 5 of 7 top geometric anomalies were confirmed disruptions.

### Finding 3: Manifold Distance Gives Earlier Warning

| Method | Lead Time |
|--------|-----------|
| Raw diagnostic spikes (3σ) | ~20 ms |
| Raw expansion ratio (2σ) | ~500 ms |
| **Manifold distance** | **~1000 ms** |

By measuring distance from a learned stable manifold (PCA on stable shots), we detect disruption precursors **400-750 ms earlier** than raw geometric features.

---

## Why This Works

### The Reactive View (Navier-Stokes / MHD)

```
State(t) → Physics → State(t+dt)
```

The equations describe local evolution. Given the current velocity field, pressure, magnetic field, they compute the forces and predict the next instant. This is fundamentally reactive:

- Local causality only
- No awareness of global constraints
- No concept of "distance from instability"
- Sees the cliff when you're already falling

### The Predictive View (Manifold Geometry)

```
State(t) → Embedding → Position in State Space → Distance from Stable Region
```

The manifold approach measures something different:

- Global position in state space
- Distance from the stable attractor
- Proximity to the boundary
- Sees the drift toward the edge before local dynamics spike

The prediction isn't forecasting the trajectory. It's recognizing the topology.

---

## Implications for Turbulence

This suggests a general principle for prediction in turbulent/chaotic systems:

### What Doesn't Work

**Integrating the equations forward** - This is just reactive simulation. You can't outrun the dynamics by computing them faster. By the time the equations show the instability, it's already happening.

### What Does Work

**Learning the manifold structure** - Map the state space. Find the stable attractors. Identify the boundaries. Measure distance to the edge. The "prediction" is positional awareness, not trajectory forecasting.

### The Informational Perspective

A turbulent system is continuously processing information:
- Sensor data → State estimate
- State estimate → Control response
- Control response → System evolution

The high-dimensional diagnostic space isn't just measurement—it's the bandwidth of the informational channel. The 44 diagnostic channels are 44 bits of simultaneous information about the plasma state.

The ~3.5D manifold we found is the **effective information content**. Most of those 44 channels are redundant projections of the same underlying dynamics.

Disruption prediction works by detecting when the effective state leaves the stable region of this informational manifold—before the redundant projections have caught up.

---

## Connection to LLM Geometry

This is exactly analogous to what we observe in language models:

| Plasma | LLM |
|--------|-----|
| 44 diagnostic channels | 768+ embedding dimensions |
| ~3.5D intrinsic manifold | ~10-50D semantic manifold |
| Stable plasma states | Coherent text states |
| Disruption | Hallucination / incoherence |
| Control system | Attention mechanism |

In both cases:
- High-dimensional measurement space
- Low-dimensional intrinsic dynamics
- Stability defined by position on manifold
- Failure modes detectable as geometric divergence

The LLM "knows" when it's generating nonsense the same way the plasma "knows" when it's approaching disruption—by drifting away from the learned manifold of valid states.

---

## Theoretical Interpretation

### Wheeler's "It from Bit"

If information is fundamental, the plasma isn't really a 3D fluid—it's a high-dimensional informational process that we measure through 3D projections. The NS/MHD equations describe the projections. The manifold describes the information.

### The Control Problem

Tokamak operation is navigation in high-dimensional state space:
- The stable region is an attractor basin
- The control system applies forces to stay in the basin
- Disruption = falling out of the basin
- Prediction = measuring distance to the basin boundary

The physics equations tell you how the system responds to forces. They don't tell you where the basin boundaries are. That's what the learned manifold provides.

### Generalization

This approach should apply to any system where:
1. Local dynamics are well-described by differential equations
2. Global stability depends on remaining within some region
3. The stable region has learnable structure

Examples: turbulent flow control, climate tipping points, financial system stability, neural dynamics.

---

## Summary

**Reactive**: NS/MHD equations describe local state evolution. They're always one step behind.

**Predictive**: Manifold geometry measures global position. It sees the drift before the fall.

**Result**: 1 second of warning vs 20 ms, by switching from reactive features to learned manifold distance.

**Implication**: Prediction in turbulent systems isn't about forecasting trajectories. It's about recognizing topology.

---

*Analysis performed 2026-02-03 on FAIR-MAST tokamak data using ModelCypher geometry tools.*

---

## Appendix A: Mathematical Formalization

### A.1 Definition of Stable Manifold

Let $\mathbf{x}(t) \in \mathbb{R}^D$ be the plasma diagnostic state vector at time $t$, where $D$ is the number of diagnostic channels.

**Definition (Stable Manifold):** The stable manifold $\mathcal{M} \subset \mathbb{R}^D$ is the set of states reachable during normal plasma operation without disruption:

$$\mathcal{M} = \{ \mathbf{x} : \exists \text{ stable trajectory } \gamma \text{ such that } \mathbf{x} \in \gamma \}$$

**Approximation (PCA Manifold):** We approximate $\mathcal{M}$ as an affine subspace using principal component analysis on stable trajectories:

$$\hat{\mathcal{M}} = \{ \mathbf{x} : \mathbf{x} = \boldsymbol{\mu} + \mathbf{V}_k \mathbf{z}, \mathbf{z} \in \mathbb{R}^k \}$$

where:
- $\boldsymbol{\mu} \in \mathbb{R}^D$ is the mean state from stable trajectories
- $\mathbf{V}_k \in \mathbb{R}^{D \times k}$ contains the top $k$ principal components
- $\mathbf{z}$ is the low-dimensional coordinate on the manifold

**Distance to Manifold:** For any state $\mathbf{x}$, the distance to the stable manifold is:

$$d(\mathbf{x}, \hat{\mathcal{M}}) = \| \mathbf{x} - \text{Proj}_{\hat{\mathcal{M}}}(\mathbf{x}) \|_2 = \| (\mathbf{I} - \mathbf{V}_k\mathbf{V}_k^T)(\mathbf{x} - \boldsymbol{\mu}) \|_2$$

This is the reconstruction error—the component of $\mathbf{x}$ in the null space of the PCA projection.

### A.2 Information-Theoretic Argument for Earlier Warning

**Claim:** Manifold distance provides earlier warning than individual diagnostic thresholds.

**Argument:**

Let $\mathbf{x}(t)$ evolve toward disruption. Decompose into manifold and residual components:

$$\mathbf{x}(t) = \underbrace{\mathbf{V}_k \mathbf{z}(t)}_{\text{on-manifold}} + \underbrace{\mathbf{r}(t)}_{\text{off-manifold residual}}$$

1. **Individual thresholds** trigger when some $x_i(t) > \theta_i$. This is a 1D projection of the state.

2. **Manifold distance** uses $\|\mathbf{r}(t)\|_2$, which aggregates deviations across all $(D-k)$ null-space directions simultaneously.

**Information aggregation:** The manifold distance is effectively a likelihood ratio test:

$$d(\mathbf{x}, \mathcal{M})^2 \propto -2 \log \frac{P(\mathbf{x} | \text{stable})}{P(\mathbf{x} | \text{uniform})}$$

under Gaussian assumptions. This integrates evidence from all diagnostic channels.

**Why earlier?** Pre-disruption drift typically begins in modes orthogonal to normal operation—modes that individual diagnostics measure weakly but the aggregate residual captures. The manifold sees the drift when it's distributed across many channels; individual thresholds wait until it concentrates in one.

### A.3 Connection to Slow Manifold Theory

The structure we observe relates to **slow manifold theory** in dynamical systems.

**Setup:** Tokamak dynamics have a separation of timescales:
- **Fast** (μs-ms): Alfvén waves, electron dynamics, MHD oscillations
- **Slow** (ms-s): Current diffusion, pressure evolution, position control

**Slow Manifold:** The fast dynamics rapidly relax to a quasi-equilibrium that varies slowly. This quasi-equilibrium defines a **slow manifold** $\mathcal{M}_{slow}$ in the full state space.

**Connection to PCA Manifold:**
- The $k \approx 3.5$ dimensions we observe via PCA likely correspond to the slow variables
- The $(D-k)$ null-space dimensions capture fast transients that average to zero during stable operation
- Disruption precursors appear as growing components in the "fast" directions—energy leaking from slow to fast modes

**Fenichel's Theorem (informal):** Under separation of timescales, the slow manifold is approximately invariant. Trajectories that leave it (growing residual $\|\mathbf{r}(t)\|$) indicate breakdown of the timescale separation—often preceding instability.

### A.4 Generalization Conditions

When does manifold-based prediction work?

**Required:**
1. **Low intrinsic dimension:** The system's stable dynamics live on a manifold with $k \ll D$. Otherwise, the "normal" subspace is the whole space.

2. **Separable failure modes:** Failure states are geometrically distinct from stable states. The manifold boundary corresponds to a physical stability boundary.

3. **Gradual departure:** The system drifts away from stability before catastrophic failure. Instantaneous failures give no warning regardless of method.

4. **Sufficient sampling:** Training trajectories must adequately cover the stable manifold. Blind spots in training → blind spots in detection.

**Empirical tests for generalization:**
- **Cross-validation:** Does held-out stable data have low manifold distance?
- **Failure correlation:** Does manifold distance rank failures correctly?
- **Cross-device transfer:** Does the manifold structure persist across devices with similar physics?

---

## Appendix B: Plasma-LLM Parallel (Detailed)

| Concept | LLM | Plasma |
|---------|-----|--------|
| **State space** | Token embeddings $\mathbb{R}^{768+}$ | Diagnostic vector $\mathbb{R}^{44}$ |
| **Intrinsic dim** | ~10-50D (semantic content) | ~3.5D (equilibrium DOF) |
| **Stable region** | Coherent, factual text | Confined plasma with good confinement |
| **Failure mode** | Hallucination, repetition, incoherence | Disruption (VDE, density limit, locked mode) |
| **Observable precursor** | Entropy spike, attention diffusion | Expansion ratio spike, dimension increase |
| **Control system** | Attention mechanism, temperature | PF coils, heating, fueling |
| **Early warning** | Semantic uncertainty measures | Manifold distance |

**Deeper structural parallels:**

1. **Compression-expansion cycles:**
   - LLM: Attention compresses context → FFN expands → residual
   - Plasma: Confinement compresses energy → instabilities expand → dissipation

2. **Mode competition:**
   - LLM: Multiple candidate completions compete
   - Plasma: MHD modes compete for energy

3. **Cascade failure:**
   - LLM: One wrong token → compounding errors
   - Plasma: One unstable mode → energy cascade → disruption

4. **Geometric signatures:**
   - Both show: low-D manifold during stability, high-D excursion during failure
   - Both show: entropy changes precede catastrophic failure

---

## Appendix C: References

1. Fenichel, N. (1979). "Geometric singular perturbation theory for ordinary differential equations." *Journal of Differential Equations*, 31(1), 53-98.

2. Levina, E., & Bickel, P. J. (2004). "Maximum likelihood estimation of intrinsic dimension." *NIPS*.

3. de Vries, P. C., et al. (2011). "Survey of disruption causes at JET." *Nuclear Fusion*, 51(5), 053018.

4. Rea, C., & Granetz, R. S. (2018). "Exploratory machine learning studies for disruption prediction using large databases on DIII-D." *Fusion Science and Technology*, 74(1-2), 89-100.

5. Kates-Harbeck, J., Svyatkovskiy, A., & Tang, W. (2019). "Predicting disruptive instabilities in controlled fusion plasmas through deep learning." *Nature*, 568(7753), 526-531.

6. Anthropic (2024). "Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet." *Anthropic Research*.

---

*Extended 2026-02-03 with mathematical formalization and LLM parallels.*
