# Training Parameter Geometry

> **Status**: Reference
> **Domain**: Training Dynamics

This document outlines the geometric interpretation of training parameters, specifically focusing on Low-Rank Adaptation (LoRA) as a geometric constraint.

## The LoRA Geometry

When we train an adapter, we are not updating the full weight matrix $W \in \mathbb{R}^{d \times k}$. We are updating a low-rank decomposition $BA$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$.

$$ W' = W + \frac{\alpha}{r} BA $$

### Geometric Interpretation

1.  **Rank ($r$) = Subspace Dimensionality**:
    -   $r$ defines the **degrees of freedom** of the update.
    -   Small $r$ (4-8): Constrains the model to move only along a few specific "semantic directions" (e.g., "become more polite"). This works like a **railgun**—hard to deviate from the target trajectory.
    -   Large $r$ (64+): Allows complex, wiggly trajectories. Good for learning new facts, but prone to "forgetting" (moving off the manifold).

2.  **Alpha ($\alpha$) = Vector Magnitude (Loudness)**:
    -   $\alpha/r$ is a scalar multiplier.
    -   Geometrically, it scales the length of the update vector $\Delta W$.
    -   High $\alpha$: "Loud" updates. The model jumps far in the direction of the gradient.
    -   Low $\alpha$: "Quiet" precision updates.

### Subspace Analysis

Research (Aghajanyan et al., 2021) shows that the "Intrinsic Dimensionality" of LLM fine-tuning is extremely low (often < 100). This explains why LoRA works: we don't *need* the full billion-parameter space to change behavior. We just need to find the right 100-dimensional subspace.

## Gradient Smoothness & Loss Landscapes

ModelCypher includes `GradientSmoothnessEstimator` (`src/modelcypher/core/domain/training/gradient_smoothness_estimator.py`) to measure the local geometry of the loss landscape during training.

-   **High Variance (Rugged)**: The model is in a chaotic region. Updates are unstable.
-   **Low Variance (Smooth)**: The model is in a convex basin (or "wide valley"). Generalization is likely better here.
-   **Signal-to-Noise Ratio (SNR)**: Measures whether the gradient vector $g$ points in a consistent direction over time (High SNR) or flails randomly (Low SNR).

$$ \text{SNR} = \frac{\| \mu_g \|^2}{\sigma_g^2} $$

We use these metrics to dynamically adjust the learning rate or trigger "Idle Training" pauses.
