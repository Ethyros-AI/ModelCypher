# Geometric Continual Learning: SOTA Analysis & The Subspace Collision Thesis

## Introduction

Continual learning in deep neural networks has historically been framed as a problem of catastrophic forgetting, managed through probabilistic heuristics (e.g., EWC), memory replay buffers, or rigid parameter isolation. In the ModelCypher framework, we discard probabilistic framing entirely. The forward pass is a deterministic geometric map; therefore, forgetting is not a loss of "memory" or "probability," but a deterministic interference in weight space.

This document formalizes the geometric thesis for continual learning, maps existing State-of-the-Art (SOTA) methods to geometric truths, and outlines the pre-registered falsification hypotheses that govern our experimental pipeline.

---

## Part 1: The Forgetting Problem Is a Subspace Collision

In high-dimensional geometry, a neural network layer transforms input activations $A$ into outputs via its weight matrix $W$. When a model learns Task A, it organizes its weights such that a specific subspace of $W$ reliably projects $A_A$ into the correct latent representation. 

Catastrophic forgetting occurs precisely when a subsequent weight update $\Delta W_B$ for Task B possesses a non-zero projection onto the subspace relied upon by Task A. 

$$\text{Forgetting occurs iff: } ||\text{Proj}_{\text{span}(A_A)}(\Delta W_B)|| > \epsilon$$

Every practically effective SOTA method in continual learning is, at its core, attempting to solve one of the following: 
1. **Null-Space Projection:** Forcing $\Delta W_B$ into the orthogonal complement of $\text{span}(A_A)$.
2. **Manifold Constraint:** Restricting optimization trajectories to functionally invariant paths.
3. **Replay:** Continually re-asserting the geometric constraints of Task A during Task B's optimization.
4. **Parameter Isolation:** Allocating orthogonal parameters to subsequent tasks.

From first principles, **null-space projection is the geometrically correct solution** for the NB-LoRA framework, as it allows parameter-efficient updates that are guaranteed, mathematically, not to perturb the activation manifolds of previous tasks.

---

## Part 2: What ModelCypher Already Solves By Construction

ModelCypher's strict adherence to geometric primitives (Weyl bounds, Cayley parameterization, SVD telemetry) maps perfectly onto a rigorous continual learning solution:

*   **NB-LoRA (Cayley Parameterization) $\rightarrow$ Weyl Perturbation Bounds on Existing Knowledge:** 
    By construction, our non-binary LoRA operates on the Stiefel manifold via the Cayley transform. This ensures that the spectral norm of any perturbation $||\Delta W||_2$ is strictly bounded, guaranteeing the spectral safety of existing representations.
*   **Tikhonov Null-Space Projection $\rightarrow$ New Learning in the Complement:**
    Instead of probabilistic regularization, we project subsequent weight deltas strictly into the null space of the existing activation covariance matrix. New learning occupies untouched dimensions.
*   **Weyl Budget Monitoring $\rightarrow$ Capacity Tracking:**
    The ratio $||\Delta W||_2 / \sigma_{k}$ gives us a deterministic metric for how much "room" is left in a layer before the null space is exhausted.
*   **CKA Verification $\rightarrow$ Capability Preservation Measurement:**
    Centered Kernel Alignment provides an exact representational similarity score between pre- and post-update activation manifolds.
*   **EFE Curiosity Policy $\rightarrow$ Consolidation Trigger:**
    Expected Free Energy signals (geometric surprise) tell the system exactly when a new task's manifold geometry deviates sufficiently to warrant attention, or when null-space capacity dictates that adapter consolidation must occur.

---

## Part 3: SOTA Mapping — Where Geometry Agrees and Where It Corrects

For each major SOTA family, we analyze their empirical findings against ModelCypher's geometric predictions:

| SOTA Method / Family | ModelCypher Geometric Mapping | Geometric Correction / Unification |
| :--- | :--- | :--- |
| **O-LoRA / InfLoRA (Orthogonal LoRA)** | Null-space projection. | ModelCypher's null-space projection is the continuous (Tikhonov) generalization of their strictly orthogonal binary subspace separation. They isolate subspaces discretely; ModelCypher maintains continuous spectral capacity. |
| **TITANS / SuRe (Surprise-driven)** | Eigenscore / Expected Free Energy. | Eigenscore is the fundamental geometric surprise signal. Our framework converges with their information-theoretic selection criteria but provides a deterministic spectral upper bound for novelty absorption. |
| **OLieRA (Lie Group LoRA)** | Cayley-Stiefel parameterization. | OLieRA empirically discovered the utility of Lie group optimization; ModelCypher enforces it fundamentally via the Cayley transform for all adapter matrices to preserve isometry constraints. |
| **Spectral Regularization (ICLR 2025)** | Weyl bounds on $\Delta W$. | Spectral regularization treats preservation as a soft empirical penalty in the loss function; ModelCypher establishes Weyl bounds as strict, verifiable limits on symmetric parameter perturbations. |
| **NeuroDream / Sleep-time Compute** | Consolidation Service + Manifold Completion. | Sleep/dreaming abstracts the offline re-entanglement process; this structurally maps to iterative adapter merging (consolidation) to recover underlying base model null-space capacity. |
| **FIP (Functionally Invariant Paths)** | Functional Equivalence Class Trajectories. | Both use Riemannian structure to constrain weight trajectories. However, FIP constructs geodesics on the level set of the loss function (the functional equivalence class), whereas ModelCypher's Cayley retraction strictly operates on the Stiefel manifold. |

---

## Part 4: The Geometric Continual Learning Thesis

Derived from ModelCypher's geometry, a robust continual learning mechanism is not a training regime, but a lifecycle of geometric accounting:

1.  **New Knowledge:** Defined strictly as a weight delta $\Delta W$ derived from Cayley-parameterized training on the target task.
2.  **Preservation:** Ensured by enforcing that $\Delta W$ lies entirely within the null space of the existing activation subspace (Tikhonov projection).
3.  **Capacity:** The rank of the null space determines the absolute absorption limit per layer.
4.  **Saturation:** When the null-space rank approaches zero, the layer can no longer absorb representation without subspace collision (forgetting).
5.  **Consolidation:** The process of adapter merging re-distributes representations, re-expanding the available null-space capacity.

**The Continual Cycle:** `Train` $\rightarrow$ `Saturate` $\rightarrow$ `Consolidate` $\rightarrow$ `Train`

---

## Part 5: Falsification Hypotheses

ModelCypher operates on empirical proof. These pre-registered pass/fail hypotheses govern our G7 experiments.

*   **H1: Null-Space Rank Depletion is Monotonic.** 
    Across sequential tasks (without consolidation), the rank of the available null space per layer must strictly decrease monotonically.
*   **H2: CKA Retention Correlates with Capacity.**
    The CKA similarity (task retention) after Task N will correlate with the remaining null-space capacity before Task N was ingested. We will pre-register the measurement, report the observed $r$, and evaluate whether the correlation is statistically significant ($p$-value derived from measured data) rather than relying on arbitrary bounds.
*   **H3: Consolidation Restores Capacity.**
    An adapter merge (consolidation phase) into the base weights will safely re-entangle representations and restore capacity to the point where $\text{null\_rank} > 0$ for all layers, without catastrophic loss of CKA on early tasks.
*   **H4: Scale Invariance of Geometric Ratios.**
    The geometric relationships observed (e.g., null-space depletion rate per $10^6$ parameters) will transfer across structural scales. We evaluate scale invariance by measuring whether the ratio of depletion rates across scales (350M vs 1.2B) remains within a $2\times$ tolerance.

---

## References

1. **Chaudhry, A., et al. (2018).** *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence.* (Forgetting Measure Definitions)
2. **Lopez-Paz, D. & Ranzato, M. (2017).** *Gradient Episodic Memory for Continual Learning.* (FWT / BWT formalisms)
3. **Liu, Y., et al. (2024).** *O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning.* (Binary subspace separation logic)
4. **Gao, Y., et al. (2024).** *OLieRA: Orthogonal Lie Group Low-Rank Adaptation.* (Lie-group structures in parameter efficient tuning)
5. **Kornblith, S., et al. (2019).** *Similarity of Neural Network Representations Revisited.* (CKA basis)
6. **Anonymous (2025).** *Spectral Regularization for Task Interference.* ICLR 2025.
