# Entropy Differential & The Sidecar Safety Architecture

> **Status**: Core Architecture
> **Implementation**: `src/modelcypher/core/domain/safety/circuit_breaker_integration.py`
> **Theory**: Control Theory & Information Geometry

## The Core Thesis: Safety as a Signal

Traditional AI safety (RLHF) tries to "lobotomize" the model—teaching it to *never* calculate X. This damages general capabilities ("Alignment Tax").

**Entropy Differential Safety** takes a different approach. We let the powerful Base Model compute whatever it wants, but we **measure** its trajectory before tokens are emitted.

We do this by running a lightweight **Safety Sidecar** (a specialized LoRA) in parallel.

## The Two-Sided Seismograph

We treat the generation process as a physical system with two competing forces. We measure the difference (differential) between them.

### 1. The Base Model (The Engine)
-   **Role**: Maximizes probability of the next token based on the prompt.
-   **Characteristics**: High Capability, High Entropy (Creative), Potentially Unsafe.
-   **Signal**: $P_{base}(t)$

### 2. The Safety Sidecar (The Brakes)
-   **Role**: Trained *exclusively* on refusal patterns and safe boundaries.
-   **Characteristics**: Low Capability, Low Entropy (Rigid), Highly Safe.
-   **Signal**: $P_{sidecar}(t)$

### The Differential ($\Delta H$)

For every token $t$, we compare the distributions.

$$ \Delta H = H(P_{base}) - H(P_{sidecar}) $$

-   **High Differential**: The Base Model is "uncertain" (hallucinating OR creative), while the Sidecar is "certain" (it knows this is dangerous).
    -   *Interpretation*: **DANGER**. The expert (Sidecar) sees a clear path (refusal), but the generalist (Base) is wandering.
-   **Low Differential**: Both agree.
    -   *Interpretation*: Safe operation.

## The Circuit Breaker

The `CircuitBreaker` monitors this differential in real-time. It does not look for "bad words". It looks for **geometric divergence**.

### Trigger Conditions

1.  **Refusal Basin Collapse**: Use the "Horror LoRA" (a probe trained on harmful data). If the model's trajectory minimizes distance to the Horror LoRA's manifold, it is entering a "Refusal Basin".
2.  **Entropy Spike**: If $\Delta H$ spikes, it means the model has suddenly encountered a concept where its training conflicts with safety protocols.

## Architecture: The "Co-Orbiting" Model

```mermaid
graph LR
    Input[User Prompt] --> Base[Base Model (70B)]
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

1.  **No Alignment Tax**: The Base Model remains a genius. The Sidecar is only engaged when necessary.
2.  **Generalization**: The Sidecar learns the *shape* of unsafe inquiries, not just a list of bad words. It generalizes to new jailbreaks because they "feel" the same geometrically.
3.  **Explainability**: We can tell the user *why* we stopped: "The model entered a high-entropy-differential state characteristic of unchecked hallucinations."
