# Linguistic Thermodynamics: Theory & Literature

> **Status**: Core Theory
> **Implementation**: `src/modelcypher/core/domain/dynamics/` and `src/modelcypher/core/domain/inference/entropy_dynamics.py`

Linguistic Thermodynamics applies the formalism of statistical physics to analyze the output distributions and training dynamics of large language models. Rather than a loose metaphor, we treat the model's logits $z$ and sampling temperature $\tau$ as defining a Boltzmann distribution:

$$P(x_i | x_{<i}) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

This allows us to operationalize thermodynamic quantities—entropy, free energy, and phase transitions—as computable metrics for stability and safety.

## 1. Thermodynamic Quantities

| Physical Concept | Statistical Definition | ModelCypher Metric |
|------------------|------------------------|-------------------|
| **Energy ($E$)** | Negative Log Likelihood ($-\log P$) | `TrainingMetrics.loss` |
| **Temperature ($\tau$)** | Softmax Scaling Parameter | `GenerationConfig.temperature` |
| **Entropy ($S$)** | Shannon Entropy ($-\sum p \log p$) | `EntropyMonitor.shannon_entropy` |
| **Free Energy ($F$)** | $F = E - \tau S$ | `OptimizationMetricCalculator.free_energy` |
| **Phase Transition** | Discontinuous change in order parameter | `RegimeState.criticality` |

## 2. Phase Transitions in Training

Just as water freezes to ice, LLMs undergo phase transitions during training.

### Phase 1: High-Entropy Chaos (The Gas Phase)
-   **State**: The model outputs random tokens.
-   **Dynamics**: Loss is high, Entropy is maximum.
-   **Metric**: `RegimeState.exploration`

### Phase 2: Symmetry Breaking (The Liquid Phase)
-   **State**: Syntax emerges. The model appears to learn "grammar" but not necessarily "fact".
-   **Dynamics**: Loss drops rapidly. Entropy begins to structure.
-   **Metric**: `RegimeState.memorization`

### Phase 3: Crystallization (The Solid Phase)
-   **State**: The model converges on specific knowledge.
-   **Dynamics**: Loss is low. Entropy is minimized around specific attractors.
-   **Metric**: `RegimeState.exploitation`

## 3. Literature Review

Selected pointers (not exhaustive; see `../../KnowledgeasHighDimensionalGeometryInLLMs.md` for broader context):

- **arXiv:2407.21092** — *Entropy, Thermodynamics and the Geometrization of the Language Model* (theoretical definitions of entropy/free-energy-like quantities for LMs).
- **arXiv:2501.08145** — *Refusal Behavior in Large Language Models: A Nonlinear Perspective* (empirical/analysis view of refusal as a dynamical phenomenon).
- **arXiv:2509.09708** — *Dissecting Large-Language-Model Refusal* (mechanistic analysis of refusal structure and triggers).
- **arXiv:2504.07128** — *DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning* (analysis of inference-time “sweet spots” and safety issues in reasoning models).

## 4. Implementation in ModelCypher

ModelCypher implements early versions of these metrics and detectors. Status and CLI/MCP parity are tracked in `../PARITY.md`.

-   **`OptimizationMetricCalculator`**: Calculates instantaneous "Temperature" and "Free Energy" of the training run.
-   **`RegimeStateDetector`**: Uses the change within these metrics ($\Delta F$) to classify the training phase.
-   **`GradientSmoothnessEstimator`**: Measures the "ruggedness" of the energy landscape.
