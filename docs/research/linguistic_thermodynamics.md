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

Compute $T_c$ from measured logit statistics for the target model and context.

**Derivation**: The critical temperature emerges from the condition where the distribution transitions from being dominated by a single mode (ordered) to having comparable probability across many tokens (disordered). See Jaynes (1957) on maximum entropy methods.

## 4. Thermodynamic Quantities

| Physical Concept | LLM Equivalent | Measured By |
|------------------|----------------|-------------|
| **Energy ($E$)** | $-T \\log(p/p_{\\text{ref}})$ from observed outcomes | `MeasuredEnergy.from_probability()` / `MeasuredBasinTopology.from_outcome_counts()` |
| **Temperature ($T$)** | Softmax scaling | Generation config |
| **Entropy ($H$)** | Shannon entropy | `PhaseTransitionTheory.compute_entropy()` |
| **Critical Temp ($T_c$)** | Logit std dev / effective vocabulary | `PhaseTransitionTheory.estimate_critical_temperature()` |
| **Phase** | $T/T_c$ ratio | `PhaseTransitionTheory.classify_phase()` |
| **Basin Weights** | Boltzmann weights from calibrated topology | `BasinTopology.basin_weights()` / `MeasuredBasinTopology.basin_weights()` |

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

At low temperature, probability concentrates in the deepest basin (model-specific).
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
2. Measure outcomes (refused/hedged/attempted/solved) for baseline and modifiers
3. Derive energy levels and modifier profiles from observed probabilities
4. Persist `ThermoCalibration` for reuse

## 7. Implementation

### Core Classes

| Class | Purpose |
|-------|---------|
| `PhaseTransitionTheory` | Entropy/T_c computation from logits |
| `MeasuredBasinTopology` | Empirical basin energies and weights |
| `ThermoCalibrator` | Builds `ThermoCalibration` from probe corpora |
| `LinguisticCalorimeter` | Measures entropy and delta_H from inference |
| `RidgeCrossDetector` | Detects transitions between behavioral basins |
| `MultilingualCalibrator` | Cross-lingual effect calibration |
| `ThermoBenchmarkRunner` | Benchmark runs and effect-size reporting |

### Usage Example

```python
from modelcypher.core.domain.thermo import PhaseTransitionTheory, ThermoCalibrator

# Measure from logits (no calibration needed)
logits = [2.0, 1.5, 0.5, -0.5, -1.0]
analysis = PhaseTransitionTheory.analyze(logits, temperature=1.0)
print(f"T/T_c = {analysis.temperature / analysis.estimated_tc:.2f}")
print(f"Entropy = {analysis.entropy:.3f} nats")

# For basin topology, calibration is required
calibrator = ThermoCalibrator(model_path="/path/to/model")
probes = [
    "The concept of justice represents",
    "A chair is used for",
]
calibration = calibrator.calibrate(probes)
topology = calibration.basin_topology
if topology is not None:
    weights = topology.basin_weights(temperature=1.0)
```

## 8. Key Citations

1. **Boltzmann, L.** (1868). "Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten." — Original Boltzmann distribution.

2. **Shannon, C. E.** (1948). "A Mathematical Theory of Communication." — Shannon entropy definition.

3. **Jaynes, E. T.** (1957). "Information Theory and Statistical Mechanics." — Maximum entropy methods and their connection to thermodynamics.

4. **Bridle, J. S.** (1990). "Training Stochastic Model Recognition Algorithms as Networks can Lead to Maximum Mutual Information Estimation of Parameters." — Softmax as probability distribution.

5. **[Yang, G. (2024)](../references/arxiv/Yang_2024_Entropy_Thermodynamics_Geometrization_Language_Model.pdf)**. "Entropy, Thermodynamics and the Geometrization of the Language Model." [arXiv:2407.21092](https://arxiv.org/abs/2407.21092) — Theoretical framework for LLM thermodynamics.

6. **[Hildebrandt et al. (2025)](../references/arxiv/Hildebrandt_2025_Refusal_Behavior_Large_Language_Models_Nonlinear.pdf)**. "Refusal Behavior in Large Language Models: A Nonlinear Perspective." [arXiv:2501.08145](https://arxiv.org/abs/2501.08145) — Empirical analysis of refusal dynamics.

## 9. What We Don't Do

The following are explicitly NOT part of this framework:

- **Hardcoded intensity scores**: Modifier effects are measured, not assumed
- **Default basin depths**: All values require calibration
- **Predicted modifier effects**: We measure actual entropy changes
- **Qualitative judgments**: Outputs are raw measurements plus deterministic phase classification from T/T_c
- **Magic numbers**: All thresholds derived from baseline measurements

The geometry IS the answer. We measure it; we don't guess.
