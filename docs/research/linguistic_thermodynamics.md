# Linguistic Thermodynamics: Theory & Literature

> **Status**: Core Theory
> **Implementation**: `src/modelcypher/core/domain/dynamics/` and `src/modelcypher/core/domain/inference/entropy_dynamics.py`

Linguistic Thermodynamics is a **working analogy** for describing measurable properties of language models (loss, entropy, temperature-dependent behavior) using terms borrowed from statistical physics. We do **not** claim that an LLM is literally a thermodynamic system. The goal is to define quantities we can compute and then test whether they correlate with useful regimes (e.g., instability under modifiers, refusal dynamics, or training-phase shifts).

## 1. The Core Analogy

| Thermodynamic Concept | LLM Equivalent | ModelCypher Metric |
|-----------------------|----------------|-------------------|
| **Energy ($E$)** | Loss ($\mathcal{L}$) | `TrainingMetrics.loss` |
| **Temperature ($T$)** | Sampling Temp / Noise | `GenerationConfig.temperature` |
| **Entropy ($S$)** | Token Distribution Entropy | `EntropyMonitor.shannon_entropy` |
| **Free Energy ($F$)** | $F = E - TS$ | `OptimizationMetricCalculator.free_energy` |
| **Quenching** | RLHF / Fine-tuning | `RegimeState.generalization` |

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
