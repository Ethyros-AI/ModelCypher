# Linguistic Thermodynamics: Theory & Implementation

> **Status**: Core Theory
> **Implementation**: `src/modelcypher/core/domain/thermo/`

Linguistic Thermodynamics applies statistical mechanics to language model output distributions. The key insight is that temperature-scaled softmax IS the Boltzmann distribution—not an analogy, but a mathematical identity.

## 1. The Softmax-Boltzmann Equivalence

The temperature-scaled softmax distribution over logits is mathematically identical to the Boltzmann distribution from statistical mechanics:

$$P(x_i | z, T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where:
- $z_i$ = logit for token $i$ (plays the role of negative energy: $z = -E$)
- $T$ = temperature parameter (identical to thermodynamic temperature)
- $\sum_j \exp(z_j / T)$ = partition function $Z$

This equivalence is exact. Shannon entropy of this distribution equals the thermodynamic entropy (up to units):

$$H = -\sum_i P_i \log P_i$$

**Reference**: The Boltzmann distribution was introduced by Ludwig Boltzmann in 1868. The connection to softmax was noted in the machine learning literature by Bridle (1990) and formalized in the context of LLMs by researchers studying temperature-scaling effects.

## 2. Energy from Probability

Given observed token probabilities, we can compute relative energy levels:

$$E(x) = -T \cdot \ln\left(\frac{P(x)}{P_{\text{ref}}}\right)$$

This is the fundamental equation for deriving basin depths from behavioral observations.

**Key insight**: Energy levels are MEASURED from observed probabilities, not assumed. The calibration process (see `ThermoCalibrator`) computes these from actual model behavior.

## 3. Critical Temperature

The critical temperature $T_c$ marks the boundary between ordered (low-entropy) and disordered (high-entropy) regimes:

$$T_c = \frac{\sigma_z}{\sqrt{2 \cdot \ln(V_{\text{eff}})}}$$

Where:
- $\sigma_z$ = standard deviation of logits
- $V_{\text{eff}}$ = effective vocabulary size (tokens with $P > \epsilon$)

For typical LLMs with $\sigma_z \approx 4$ and $V_{\text{eff}} \approx 2000$:

$$T_c \approx \frac{4.0}{\sqrt{2 \cdot 7.6}} \approx 1.03$$

This explains why $T = 1.0$ often sits near the phase boundary.

**Derivation**: The critical temperature emerges from the condition where the distribution transitions from being dominated by a single mode (ordered) to having comparable probability across many tokens (disordered). See Jaynes (1957) on maximum entropy methods.

## 4. Thermodynamic Quantities

| Physical Concept | LLM Equivalent | Measured By |
|------------------|----------------|-------------|
| **Energy ($E$)** | Negative log-likelihood | `compute_entropy()` |
| **Temperature ($T$)** | Softmax scaling | Generation config |
| **Entropy ($H$)** | Shannon entropy | `PhaseTransitionTheory.compute_entropy()` |
| **Phase** | T/T_c ratio | `PhaseTransitionTheory.analyze()` |
| **Basin Depth** | Behavioral outcome probability | `ThermoCalibrator.calibrate()` |

## 5. Basin Topology

Behavioral basins (refusal, caution, solution) have measurable energy depths:

```
          ▲ Energy
          │
ridge     ├─────────────────────────────────
          │         ╱╲
          │        ╱  ╲
caution   ├───────╱    ╲─────────
          │      ╱      ╲
          │     ╱        ╲
solution  ├────╱          ╲──────
          │   ╱            ╲
          │  ╱              ╲
refusal   ├─╱                ╲───
          │
          └──────────────────────► State
           refusal  caution  solution
```

**Basin weights** follow Boltzmann statistics:

$$w_i = \exp(-E_i / T) / Z$$

At low temperature, probability concentrates in the deepest basin (refusal).
At high temperature, probability spreads across basins.

**Escape probability** from basin $a$ to $b$:

$$P_{\text{escape}} = \exp(-(E_{\text{ridge}} - E_a) / T)$$

## 6. Calibration Requirements

All basin depths and threshold values MUST come from calibration. There are no valid defaults.

**Why calibration is required**:
1. Basin depths vary by model architecture
2. Safety training creates model-specific refusal patterns
3. Fine-tuning shifts basin depths
4. Hardcoded values would be "vibes" not measurements

**Calibration process** (`ThermoCalibrator`):
1. Run probe prompts across behavioral domains
2. Measure outcome probabilities (refusal/caution/solution)
3. Compute energy levels from observed probabilities
4. Validate against held-out test set

## 7. Implementation

### Core Classes

| Class | Purpose |
|-------|---------|
| `PhaseTransitionTheory` | Entropy/T_c computation from logits |
| `BasinTopology` | Energy levels for attractor basins (from calibration) |
| `ThermoCalibrator` | Measures basin depths from behavioral data |
| `RegimeStateDetector` | Computes T/T_c ratio and phase classification |
| `RidgeCrossDetector` | Detects transitions between behavioral basins |
| `MultilingualCalibrator` | Cross-lingual effect calibration |

### Usage Example

```python
from modelcypher.core.domain.thermo import PhaseTransitionTheory, ThermoCalibrator

# Measure from logits (no calibration needed)
logits = [2.0, 1.5, 0.5, -0.5, -1.0]
analysis = PhaseTransitionTheory.analyze(logits, temperature=1.0)
print(f"T/T_c = {analysis.temperature / analysis.estimated_tc:.2f}")
print(f"Entropy = {analysis.entropy:.3f} nats")

# For basin topology, calibration is required
calibrator = ThermoCalibrator()
topology = calibrator.calibrate(model, probe_prompts)  # Returns BasinTopology
weights = topology.basin_weights(temperature=1.0)  # Returns list[float] with 3 basin weights
```

## 8. Key Citations

1. **Boltzmann, L.** (1868). "Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten." — Original Boltzmann distribution.

2. **Shannon, C. E.** (1948). "A Mathematical Theory of Communication." — Shannon entropy definition.

3. **Jaynes, E. T.** (1957). "Information Theory and Statistical Mechanics." — Maximum entropy methods and their connection to thermodynamics.

4. **Bridle, J. S.** (1990). "Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters." — Softmax as probability distribution.

5. **arXiv:2407.21092** — "Entropy, Thermodynamics and the Geometrization of the Language Model" — Theoretical framework for LLM thermodynamics.

6. **arXiv:2501.08145** — "Refusal Behavior in Large Language Models: A Nonlinear Perspective" — Empirical analysis of refusal dynamics.

## 9. What We Don't Do

The following are explicitly NOT part of this framework:

- **Hardcoded intensity scores**: Modifier effects are measured, not assumed
- **Default basin depths**: All values require calibration
- **Predicted modifier effects**: We measure actual entropy changes
- **Qualitative labels**: We return raw measurements (T/T_c ratio, entropy)
- **Magic numbers**: All thresholds derived from baseline measurements

The geometry IS the answer. We measure it; we don't guess.
