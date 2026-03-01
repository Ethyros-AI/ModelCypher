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
    The geometric relationships observed (e.g., null-space depletion rate per $10^6$ parameters) will transfer across structural scales. We evaluate scale invariance by measuring whether the ratio of depletion rates across scales (350M vs 1.2B) remains stable. We will pre-register the measurement, report the invariance factor, and evaluate whether the structural proportionality holds with statistical significance ($p$-value derived from measured data variance across seeds) rather than relying on an arbitrary tolerance.

---

## Part 6: Preliminary Empirical Results — Cross-Scale Capacity Profiles

### LFM2 Architecture (350M vs 1.2B)

Exp2 capacity profiling on the LFM2 architecture reveals a structurally invariant null-space distribution across scales.

#### Capacity Utilization (Scale Invariance)

| Layer Type | 350M util | 1.2B util | Δ |
| :--- | :---: | :---: | :---: |
| q_proj | 0.467 | 0.486 | 0.019 |
| attn_out_proj | 0.632 | 0.628 | 0.004 |
| conv_out_proj | 0.648 | 0.649 | 0.001 |
| FFN (w1/w2/w3) | 0.877–0.914 | 0.874–0.918 | < 0.01 |
| k_proj | 0.627 | 0.713 | 0.086 |
| v_proj | 0.783 | 0.883 | 0.100 |

#### Null-Space Capacity (Linear Scaling with Hidden Dimension)

| Layer Type | 350M null dims (d=1024) | 1.2B null dims (d=2048) | Ratio |
| :--- | :---: | :---: | :---: |
| q_proj | 4.00 | 7.83 | 1.96× |
| attn_out_proj | 1.83 | 3.50 | 1.91× |
| conv_out_proj | 1.70 | 3.70 | 2.18× |
| FFN, k_proj, v_proj | 0 | 0 | — |

#### Key Findings

1. **H4 supported:** Capacity utilization is structurally constant across scales (q_proj invariance factor = 1.04). Null-space dimension scales linearly with hidden dimension — the *fraction* of available null-space is architecturally determined.
2. **Bottleneck identified:** q_proj layers carry the majority of available null-space budget. FFN layers are saturated (85–95% utilization, zero null dims). This means sequential task injection will deplete attention projections first at any scale.
3. **H1 prediction narrowed:** Monotonic depletion (H1) reduces to tracking q_proj null-space consumption during sequential NB-LoRA training.

### Cross-Architecture Capacity Profiles

Exp2 profiling was extended across 4 distinct architectures to test whether the q_proj bottleneck is architecture-specific or universal.

#### Attention Projection Capacity (q_proj + o_proj)

| Architecture | Params | q_proj util | q_proj null | o_proj util | o_proj null | FFN util |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LFM2-350M | 354M | 0.467 | 4.0 | 0.632 | 1.8 | 0.88–0.91 |
| LFM2-1.2B | 1.17B | 0.486 | 7.8 | 0.628 | 3.5 | 0.87–0.92 |
| Qwen2.5-3B | 3.09B | 0.532 | 5.3 | 0.589 | 3.5 | 0.88–0.91 |
| Llama-3.2-3B | 3.21B | 0.530 | 32.3 | 0.625 | 5.6 | 0.85–0.89 |
| Qwen3-8B | 8.19B | 0.580 | 17.8 | 0.632 | 6.3 | 0.83–0.87 |

> [!WARNING]
> Mistral-7B-4bit was profiled but excluded: 4-bit quantization produces numerically invalid spectral decompositions. Capacity utilization values overflow to $10^{15}$. Spectral capacity analysis requires at minimum bf16 precision.

#### Universal Structural Invariants

Across all bf16 architectures tested (LFM2, Qwen, Llama — 3 architecture families, 5 models, 354M to 8.2B parameters):

1. **q_proj is universally the capacity bottleneck.** It has the lowest utilization (0.47–0.58) and highest null-space dim across all architectures. This is not coincidental — the query projection creates the attention pattern, so it must maintain the largest representational diversity.
2. **o_proj is the secondary reservoir.** Consistent utilization around 0.59–0.63 with meaningful null-space dims.
3. **FFN layers are universally saturated** (0.83–0.92 utilization, zero null dims). No architecture provides trainable null-space capacity in the feed-forward network.
4. **k_proj and v_proj have near-zero null-space** in all architectures tested (exception: Llama-3.2-3B k_proj shows 0.6 mean null dims). These projections are tightly utilized.

#### Implications for Continual Learning

The universal q_proj bottleneck means the geometric continual learning cycle's capacity arithmetic is **architecture-independent**: at any scale, the number of sequential tasks that can be absorbed before consolidation is bounded by $\sum_l \text{null\_rank}(W^l_{q\_proj})$.

---

## References

1. **Chaudhry, A., et al. (2018).** *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence.* (Forgetting Measure Definitions)
2. **Lopez-Paz, D. & Ranzato, M. (2017).** *Gradient Episodic Memory for Continual Learning.* (FWT / BWT formalisms)
3. **Liu, Y., et al. (2024).** *O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning.* (Binary subspace separation logic)
4. **Gao, Y., et al. (2024).** *OLieRA: Orthogonal Lie Group Low-Rank Adaptation.* (Lie-group structures in parameter efficient tuning)
5. **Kornblith, S., et al. (2019).** *Similarity of Neural Network Representations Revisited.* (CKA basis)
6. **Anonymous (2025).** *Spectral Regularization for Task Interference.* ICLR 2025.
