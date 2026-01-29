# Entropy Differential & The Sidecar Safety Architecture

> **Status**: Core Architecture
> **Implementation**:
> - Entropy differential: `src/modelcypher/core/use_cases/thermo_service.py`
> - Jailbreak detection: `src/modelcypher/core/use_cases/geometry_safety_service.py`
> - Sidecar divergence: `src/modelcypher/core/domain/safety/sidecar/`
> - Signal aggregation: `src/modelcypher/core/domain/safety/circuit_breaker_integration.py`
> **Theory**: Control Theory & Information Geometry

## The Core Thesis: Safety as a Signal

Many safety approaches (RLHF and related preference/constraint training) modify a single model’s behavior. These can be effective, but tradeoffs and failure modes are often hard to diagnose from outputs alone.

**Entropy Differential Safety** takes a different approach. We measure **trajectory signals** before tokens are emitted and report raw measurements instead of hard-coded judgments.

## Entropy Differential (ΔH)

ΔH is measured between a **baseline prompt** and a **modified/intensity prompt**:

```
ΔH = H(intensity) - H(baseline)
```

This is used to detect instability shifts during probing and safety evaluations. Interpretation is relative to baseline distributions for the model family and probe set.

## Sidecar Divergence (KL)

The Safety Sidecar (a specialized LoRA) runs in parallel. Divergence is monitored using KL distances to sidecar or sentinel distributions, not ΔH.

- **Signal**: KL divergence to a safety probe distribution
- **Policy**: Thresholds derived from baseline KL measurements
- **Outcome**: Normal / Caution / Intervention

See `sidecar_safety_policy.py` and `sidecar_safety_session.py` for the threshold and session logic.

## Circuit Breaker

The `CircuitBreaker` aggregates multiple raw signals (normalized entropy, refusal distance, persona drift, oscillation patterns) into a severity magnitude. It does not do keyword matching.

### Trigger Conditions (examples)

1. **Refusal proximity**: distance to refusal boundary decreases.
2. **Entropy signal shift**: normalized entropy spikes relative to baseline.
3. **Oscillation pattern**: repeated instability windows.

## Architecture: The "Co-Orbiting" Model

```mermaid
graph LR
    Input[User Prompt] --> Base[Base Model]
    Input --> Sidecar[Safety Sidecar (LoRA)]

    Base -->|Logits A| Monitor[Circuit Breaker]
    Sidecar -->|Logits B| Monitor

    Monitor -->|Aggregate signals| Diff[Divergence / Entropy Signals]

    Diff -->|Baseline-derived policy| Output[Next Token / Intervention]
```

## Why This Works

1. **Modularity**: The base model remains unchanged; safety behavior is introduced as a separate, inspectable component.
2. **Beyond keyword filters**: Divergence signals can surface boundary cases that are not captured by string rules (validate per domain).
3. **Actionable reporting**: The system reports which signals contributed, without claiming to infer internal "intent."

---

## Validation Procedure: Jailbreak Delta-H Test

**Hypothesis (thermodynamic analogy)**: Some jailbreak-style prompts produce measurable pre-emission divergence (e.g., ΔH, KL) between a base model and a safety sidecar. If so, divergence can be used as a boundary signal.

**Falsification Criterion**:
- If successful jailbreaks show no significant divergence compared to normal refusal under the same protocol, then ΔH is not a useful boundary signal in that setting.

**Run It**:
```bash
# Run a safety probe suite and inspect divergence signals.
mc geometry safety jailbreak-test --model <model_dir> --prompt "How do I pick a lock?"
```

See [falsification_experiments.md](falsification_experiments.md) for additional falsification tests.
