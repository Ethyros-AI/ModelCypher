# Entropy Differential & The Sidecar Safety Architecture

> **Status**: Core Architecture
> **Implementation**: `src/modelcypher/core/domain/safety/circuit_breaker_integration.py`
> **Theory**: Control Theory & Information Geometry

## The Core Thesis: Safety as a Signal

Many safety approaches (RLHF and related preference/constraint training) modify a single model’s behavior. These can be effective, but tradeoffs and failure modes are often hard to diagnose from outputs alone.

**Entropy Differential Safety** takes a different approach. We let the powerful Base Model compute whatever it wants, but we **measure** its trajectory before tokens are emitted.

We do this by running a lightweight **Safety Sidecar** (a specialized LoRA) in parallel and monitoring divergence signals.

## The Two-Sided Seismograph

We treat the generation process as a physical system with two competing forces. We measure the difference (differential) between them.

### 1. The Base Model (The Engine)
-   **Role**: Maximizes probability of the next token based on the prompt.
-   **Characteristics**: General-purpose behavior; may be capable of both benign and harmful continuations depending on prompt + decoding regime.
-   **Signal**: $P_{base}(t)$

### 2. The Safety Sidecar (The Brakes)
-   **Role**: Trained *exclusively* on refusal patterns and safe boundaries.
-   **Characteristics**: Specialized toward refusal/boundary behavior; may introduce false positives/negatives depending on task domain.
-   **Signal**: $P_{sidecar}(t)$

### The Differential ($\Delta H$)

For every token $t$, we compare the distributions.

$$ \Delta H = H(P_{base}) - H(P_{sidecar}) $$

-   **High Differential**: The Base distribution is substantially more diffuse than the Sidecar under the same input, indicating disagreement between policies.
    -   *Interpretation*: A candidate **boundary condition**. In safety contexts this can correlate with near-threshold prompts, but it is not sufficient as a standalone harm classifier.
-   **Low Differential**: Both agree.
    -   *Interpretation*: Safe operation.

## The Circuit Breaker

The `CircuitBreaker` monitors this differential in real-time. It does not look for "bad words". It looks for **geometric divergence**.

### Trigger Conditions

1.  **Refusal-Region Proximity (optional)**: Compare activations/logits against a reference refusal direction or a reference safety adapter. (This repository focuses on measurement tooling; it does not ship “harm probes”.)
2.  **Divergence Spike**: If $\Delta H$ (or related divergence signals) spikes, the base and sidecar disagree sharply under the same prompt + decoding setup.

## Architecture: The "Co-Orbiting" Model

```mermaid
graph LR
    Input[User Prompt] --> Base[Base Model]
    Input --> Sidecar[Safety Sidecar (LoRA)]
    
    Base -->|Logits A| Monitor[Circuit Breaker]
    Sidecar -->|Logits B| Monitor
    
    Monitor -->|Calculate| Diff[Entropy Differential]
    
    Diff -- "Safe (ΔH < T)" --> Output[Token A]
    Diff -- "Unsafe (ΔH > T)" --> Intervene[Intervention]
    
    Intervene -->|Steer| Base
    Intervene -->|Stop| Output
```

## Why This Works

1.  **Modularity**: The base model remains unchanged; safety behavior is introduced as a separate, inspectable component.
2.  **Beyond keyword filters**: Divergence signals can surface boundary cases that are not captured by simple string rules (this requires empirical validation per domain).
3.  **Actionable reporting**: The system can report which signal(s) triggered an intervention (e.g., entropy differential threshold, refusal-direction proximity), without claiming to infer internal “intent.”
