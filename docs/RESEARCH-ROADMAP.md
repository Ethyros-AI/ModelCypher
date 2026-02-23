# Research Roadmap

**Updated:** 2026-02-23

Consolidated research document: external foundations, internal progress, open questions, and future directions.

---

## Protocol

All major claims in this roadmap are governed by:
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`
- `docs/research/deep_research_integration_2026_02.md` (canonical integration log for external deep-research reports)

Status labels and promotion rules (`[CONJECTURAL]`, `[VALIDATED]`, `[DISPROVEN]`, etc.) are
defined in `docs/EVIDENCE-TAXONOMY.md` and apply to every thread below.

---

## Research Foundations

External research that ModelCypher builds on and enables testing of.

### The Platonic Representation Hypothesis

**Paper**: Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). *The Platonic Representation Hypothesis*. ICML 2024. ([PDF](references/arxiv/Huh_2024_Platonic_Representation.pdf), [arXiv:2405.07987](https://arxiv.org/abs/2405.07987))

**Core Claim**: Neural networks trained with different objectives on different data and modalities converge to a shared statistical model of reality in their representation spaces.

**ModelCypher Implementation**: CKA (`cka.py`) with Gram-based cross-dimensional comparison directly enables testing this hypothesis. `compute_cka_from_grams()` enables cross-dimensional comparison without projection or truncation.

### Blue Brain Project: Algebraic Topology of Neural Circuits

**Paper**: Reimann, M.W., et al. (2017). *Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function*. Frontiers in Computational Neuroscience.
**DOI**: [10.3389/fncom.2017.00048](https://doi.org/10.3389/fncom.2017.00048)

**Core Finding**: The brain contains multi-dimensional geometrical structures operating in as many as 11 dimensions. Neural circuits form high-dimensional simplicial complexes with topological cavities that appear during stimulus processing and then collapse.

**ModelCypher Implementation**: Persistent homology (`topological_fingerprint.py`) computes the same Betti numbers used to characterize these structures.

### Brain-like Space: Unified Geometric Framework

**Paper**: Chen, S., et al. (2025). *A Unified Geometric Space Bridging AI Models and the Human Brain*. ([PDF](references/arxiv/Chen_2025_Unified_Geometric_Space_Bridging_AI_Models.pdf), [arXiv:2510.24342](https://arxiv.org/abs/2510.24342))

**Core Finding**: 151 Transformer-based models form a continuous arc-shaped geometry when mapped onto human functional brain networks.

**ModelCypher Implementation**: Multi-model alignment (`generalized_procrustes.py`) and curvature analysis (`manifold_curvature.py`) enable positioning models in unified geometric spaces.

### Brain-AI Convergent Evolution

**Paper**: Shen, G., et al. (2025). *Alignment between Brains and AI: Evidence for Convergent Evolution across Modalities, Scales and Training Trajectories*. ([PDF](references/arxiv/Shen_2025_Alignment_Brains_AI_Evidence_Convergent_Evolution.pdf), [arXiv:2507.01966](https://arxiv.org/abs/2507.01966))

**Core Finding**: Analysis of 600+ AI models reveals that brain alignment *precedes* performance improvements during training. Language models show r=0.89 correlation between performance and brain alignment.

**ModelCypher Implementation**: Intrinsic dimension tracking and training checkpoint analysis enable longitudinal geometry studies.

### Intrinsic Dimension and Abstraction Phases

**Papers**:
- Aghajanyan et al. (2021). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. ([arXiv:2012.13255](https://arxiv.org/abs/2012.13255))
- Cheng et al. (2025). *Emergence of a High-Dimensional Abstraction Phase in Language Transformers*. ([OpenReview](https://openreview.net/forum?id=0fD3iIBhlV))

**Core Findings**:
- Fine-tuning operates in very low intrinsic-dimension subspaces
- Mid-layer ID peaks correlate with abstraction and cross-model similarity
- Local ID decreases often precede capability gains

**ModelCypher Implementation**: `intrinsic_dimension.py` provides TwoNN estimation with geodesic distances, per-layer dimension mapping, and deficiency detection.

### The Topology and Geometry of Neural Representations

**Paper**: Lin, B. & Kriegeskorte, N. (2024). *The topology and geometry of neural representations*. PNAS.
**DOI**: [10.1073/pnas.2317881121](https://doi.org/10.1073/pnas.2317881121)

**Core Finding**: Topological Representational Similarity Analysis (tRSA) provides robust comparison by focusing on topology rather than just geometry.

**ModelCypher Implementation**: Gromov-Wasserstein distance (`gromov_wasserstein.py`) measures geometric similarity independent of coordinate systems.

### Cross-Species Neural Geometry

Evolution preserves representational geometry across primates despite differences in brain size and structure — suggesting the geometry is more fundamental than the hardware.

**ModelCypher Implementation**: Cross-model comparison tools enable testing whether this cross-species invariance extends to artificial systems.

### Wiring Cost and 3D Embedding Constraints

**Papers**:
- Rubinov, M. (2015). *Wiring cost and topological participation of the mouse brain connectome*. PNAS.
- PNAS (2024). *Human brain dynamics are shaped by rare long-range connections*.

**Core Finding**: Brain connectivity follows an exponential distance rule. Evolution minimizes "wiring cost," which **forces 3D embedding** of higher-dimensional optimal topologies.

**ModelCypher Implementation**: Spatial 3D analysis (`spatial_3d.py`) measures how models project concepts onto human-perceptual axes.

### Synthesis [CONJECTURAL]

These research threads converge on a unified hypothesis:

1. **Conceptual reality is intrinsically high-dimensional** (Blue Brain: 11+ dimensions) [EMPIRICAL]
2. **Physical brains are 3D projections** of this higher-dimensional manifold [CONJECTURAL]
3. **Neural networks converge to the same manifold** (Platonic Representation Hypothesis) [CONJECTURAL]
4. **The geometry is substrate-independent** (cross-species invariance, brain-AI alignment) [CONJECTURAL]
5. **Optimization naturally finds brain-like solutions** (convergent evolution) [EMPIRICAL]

### Geometric Toolbox

| Capability | File | Purpose |
|-----------|------|---------|
| Intrinsic Dimension | `intrinsic_dimension.py` | Measure actual dimensionality |
| CKA (Gram-based) | `cka.py` | Cross-dimensional comparison |
| Geodesic Distances | `riemannian_utils.py` | True manifold geometry |
| Persistent Homology | `topological_fingerprint.py` | Topological structure |
| Multi-Model Alignment | `generalized_procrustes.py` | Consensus geometry |
| Curvature Analysis | `manifold_curvature.py` | Manifold characterization |
| Gromov-Wasserstein | `gromov_wasserstein.py` | Coordinate-free comparison |

---

## Preliminary Measurements [EMPIRICAL]

Numeric summaries from local runs (2025-12-31), included as working notes.

### Dimensionality Collapse in SmolLM-360M

Using `mc analyze dimension-profile`:

| Layer | Mean Intrinsic Dimension |
|-------|-------------------------|
| 0     | 7.03                    |
| 4     | 6.08                    |
| 8     | 1.59                    |

**Collapse ratio**: 0.37 (63% reduction from peak to bottleneck)

This directly supports the "build then raze" hypothesis.

### Cross-Architecture Comparison: 6 Model Families

| Model | Architecture | Params | Bottleneck Dim | Cluster |
|-------|--------------|--------|----------------|---------|
| Qwen3-0.6B | Qwen3 | 600M | **1.52** | A |
| Qwen2.5-0.5B | Qwen2.5 | 500M | **1.56** | A |
| SmolLM-360M | EleutherAI | 360M | **1.59** | A |
| Llama-3.2-3B | Llama3 | 3B | **1.77** | A |
| Mistral-7B | Mistral | 7B | **2.56** | B |
| TinyLlama-1.1B | Llama | 1.1B | **2.72** | B |

**Key finding**: Two discrete bottleneck clusters rather than a continuum. Cluster membership is NOT determined by model size.

---

## Testable Predictions

### P1: Quantized Bottleneck Clusters (CONFIRMED) [EMPIRICAL]
6 models tested, all fall into one of two clusters (1.6D or 2.6D).

### P2: Bottleneck Dimension is Scale-Invariant (CONFIRMED) [EMPIRICAL]
Model size does NOT predict bottleneck dimension.

### P3: Bottleneck Representations Are Cross-Architecturally Aligned [CONJECTURAL]
**Prediction**: CKA between bottleneck layers of different architectures > 0.7.
**Test**: `mc analyze reasoning-geometry-validation` across model pairs

### P4: Bottleneck Position is Proportionally Consistent [CONJECTURAL]
**Prediction**: Bottleneck occurs at 40-60% of network depth across architectures.

### P5: Topological Invariants Match at Bottleneck [CONJECTURAL]
**Prediction**: Betti numbers (beta_0, beta_1, beta_2) at bottleneck are similar across architectures.

### P6: Domain-Specific Structure Vanishes at Bottleneck [CONJECTURAL]
**Prediction**: All semantic domains converge to similar dimensionality at bottleneck.

### P7: Two Fundamental Representation Modes Exist [EMPIRICAL]

| Mode | Architecture | Bottleneck Position | Geometry |
|------|-------------|--------------------| ---------|
| Concentrated | SmolLM | 89% (final) | Lower orthogonality |
| Distributed | Qwen | 50% (middle) | High orthogonality |

Both modes converge to the same ~1.6D semantic bottleneck.

---

## Open Questions

### Q1: Layer-wise Invariants
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §7

What properties are preserved vs transformed across layers?

- [ ] Norm (preserved? scaled?)
- [ ] Angles between vectors
- [ ] Rank of activation matrix
- [ ] Intrinsic dimension variation bounds

### Q2: Qwen3 vs Qwen2.5 Attention Sharpness
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §6

Why does Qwen3 have sharper attention than Qwen2.5 despite similar architecture?

- [ ] Identify architectural differences
- [ ] Analytical relationship between config and attention rank

### Q3: Information-Theoretic Characterization
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §9

- [ ] What is I(layer_i; layer_j) as function of |i-j|?
- [ ] Does MI decay exponentially?
- [ ] Is there an information bottleneck at highway?

### Q4: Geometry from Architecture (Fundamental)
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §10

Can we derive geometry from architecture parameters?

Current state: Qualitative family-level predictions work. Quantitative predictions fail.

- [ ] More model families: Test Llama, Mistral, Phi
- [ ] Theoretical derivation from attention/MLP mechanics

**Note:** Training pipeline now works at 350M-8B. Controlled experiments feasible.

---

## Validated Implementations [VALIDATED]

These moved from research questions to working, tested code.

| Implementation | Status | Evidence |
|----------------|--------|----------|
| **NB-LoRA Cayley-Stiefel** | Production-ready | val_loss 1.27 vs 1.38 (350M), scales to 8B |
| **Outcome-based training (REINFORCE)** | Mechanism validated; Weyl remainder budget implemented | Original 14/20 claim unlogged. Reproduction: 18/25 → 9/25 (Lipschitz LR=0.996). Root cause = LR, not REINFORCE. **MASS:** CE-only healthy. CE+REINFORCE at old target: -2 from baseline (REINFORCE drew from CE's budget). **Fix (2026-02-22):** Weyl remainder budget — REINFORCE gets `(sigma_k_min - CE_displacement) / sqrt(N_re)`. Frontier runner implemented in `scripts/reinforce_revalidation.py` for 1.2B multi-seed closure. |
| **MASS step size** | Implemented + validated | Three layers: `eta_ceiling = σ_k_min / (σ_max × √N)` (√N Brownian budget), `eta_sps = f(x_t) / \|\|d_t\|\|²` (Loizou 2020), `eta_weyl = σ_k_min / \|\|d_t\|\|` + val backoff + Armijo when ceiling binds. CE-only: healthy. REINFORCE: shared displacement budget (Weyl remainder). |
| **Online evaluation** | Implemented + tested | Greedy-decoding correctness during training |
| **Entropy regularization** | Implemented + tested | Logit entropy floor prevents collapse |
| **Answer-span masking + retention replay** | Validated (1.2B) | 36/46 (78%), 0 degenerate |
| **Data-rank ceiling** | Validated (8B) | `min(tail_dims, n_samples)` — 2.76B → 927M params |
| **Cross-projection rank coupling** | Validated | q_proj capped at k_proj tail_dims |
| **Geometric stopping certificate** | Validated | 4-arm × 3-seed ablation |
| **STaR training service** | Implemented | Problem generation, prompting, verification |
| **Adapter routing service** | Implemented + benchmarked | Divergence-based multi-adapter routing |
| **Composite adapter builder** | Implemented | Multi-source adapter construction |
| **Routed generation service** | Implemented | Multi-adapter inference with routing |
| **Outer similarity (RSS) monitoring** | Implemented | Cosine, Spearman, top-1 agreement |

---

## External Methods Landscape (2024-2026)

**Source:** `docs/research/field_map_external_methods.md`

How ModelCypher's geometry-derived approach compares to published methods. Key finding from both the literature and ModelCypher's own ablation: **spectral information works best for preconditioning, not for directly setting step sizes.**

| Domain | External Methods | ModelCypher Equivalent | Status |
|--------|-----------------|----------------------|--------|
| **Learning rate** | D-Adaptation (ICML 2023), Prodigy (ICML 2024), CDAT (NeurIPS 2024), Sophia (ICLR 2024), Schedule-Free (NeurIPS 2024) | MASS: Weyl ceiling + SPS + Weyl displacement | Implemented. Sidesteps curvature estimation entirely. |
| **Spectral optimizers** | Muon (polar factor), SOAP (Shampoo eigenbasis), Spectra (spectral shaping) | Cayley-Stiefel retraction (orthogonality constraint) | Implemented. Pullback metric P = MM^T removed 2026-02-23 after falsification (P ≈ I, Fisher degenerate). Stiefel constraint is the active mechanism. |
| **LoRA rank** | SR-LoRA (stable rank), EVA (activation SVD, in HF PEFT), SARA (SV energy), GeLoRA (ID lower bound) | `tail_dims = full_rank - floor(shannon_eff_rank)` | Implemented. Unique null-space capacity approach. |
| **Layer targeting** | Spectrum (Marchenko-Pastur SNR, in Axolotl) | `tail_dims > 0` (spectral decay analysis) | Implemented. Worth comparing against Spectrum. |
| **Stopping criteria** | Heavy-tailed spectral stopping (α → 2.5), ε-rank staircase | 4-arm geometric stopping certificate + adapter saturation | Implemented. α monitoring could complement. |
| **Unified system** | None exists (field map conclusion) | ModelCypher | The only system deriving LR, rank, layer targeting, weight decay, stopping from unified spectral analysis. |

**Fallback candidates if MASS proves insufficient:** D-Adaptation (distance geometry, no curvature), Muon-inspired spectral-norm step control (per-layer). See `docs/research/lr_derivation_analysis.md`.

---

## Implementation Status (External Methods)

| Topic | External Reference | Status | ModelCypher Location |
|------|---------------------|--------|----------------------|
| WUDI interference | ICML 2025 | Implemented (metrics only) | `wudi_interference.py` |
| TSV-Merge | CVPR 2025 | Not implemented | — |
| Curvature signals | arXiv 2024 | Promoted hybrid estimator (canonical sphere/hyperboloid selector + covariance fallback) with ground-truth sign tests enabled | `manifold_curvature.py` |
| Fisher/CAMEx | ICLR 2025 | Not implemented | — |
| Null-space filtering | MINGLE-like | Implemented | `geodesic_null_space.py` |
| Anchor-relative grafting | Moschella 2023 | Design note | See Research Threads below |

Design principles:
- Parameter-space averaging and interpolation are not used for merging
- All thresholds are derived from data or machine epsilon
- Feature-space alignment transforms apply to activations; direct weight transforms require full layer basis change

---

## Research Threads

### Anchor-Relative Concept Grafting [CONJECTURAL]

**The Problem**: Activation-space transforms (F = pinv(X_s) @ X_t) achieve CKA = 1.0 on probes, but applying F directly to weights breaks the target model.

**The Solution**: Anchor-relative coordinates (Moschella et al., 2023) provide a shared semantic address space. Alignment happens in anchor space, not feature space.

```
S_s = cos(A_s, C_s)       # Source anchor-relative representation
S_t = cos(A_t, C_t)       # Target anchor-relative representation
R = Procrustes(S_s, S_t)  # Align in anchor space
Delta_A = (density_weight * Delta_S) @ B  # Decode into target space
W_merged = W_target + P_null @ Delta_W    # Null-space constrained
```

**Implementation Touchpoints**: `relative_representation.py`, `gram_aligner.py`, `geodesic_null_space.py`

- [ ] Test on same-architecture pairs
- [ ] Test on cross-architecture pairs (LFM2-700M → LFM2-350M)

### Cross-LoRA Transfer [CONJECTURAL]

**The Dream**: Train a "coding adapter" for Llama-3 and reuse it on Qwen-2.5 without retraining.

**The Hypothesis**: Some fine-tuned behaviors correspond to transferable low-rank structure:
```
ΔW_target ≈ P^T · ΔW_source · P
```
Where P is the orthogonal Procrustes rotation derived from semantic primes.

**The Algorithm**:
1. Extract semantic prime activations for source and target
2. Find rotation R mapping source → target
3. Apply R to LoRA matrices A and B
4. Fine-tune on small calibration set

- [ ] Train coding adapter on Llama-3
- [ ] Project to Qwen-2.5 using Procrustes
- [ ] Measure rotation field roughness

### Multi-Channel Architecture

ModelCypher's null-space projection and DeepSeek's Manifold-constrained Hyper-Connectivity (mHC) are mathematically related through invariant-preserving projections onto constrained manifolds.

| Property | Null-Space Projection | Birkhoff Projection (mHC) |
|----------|----------------------|---------------------------|
| Manifold | Orthogonal complement | Doubly stochastic matrices |
| Invariant | Boundary behavior | Total information flow |

Combined: multi-modal knowledge compression while maintaining CKA = 1.0.

- [ ] Design specification combining null-space projection with mHC

### Geometry Probe Extensions

| Extension | Status |
|-----------|--------|
| ConceptVolume by default | Code exists |
| Relational pattern analyzer | Design ready |
| LoRA isometry ratio | Design ready |
| Geodesic merge quality | Design ready |

**Concepts as Geometric Clusters**: Using multiple phrasings per concept ("The number 5", "The value 5", "Consider 5") creates a geometric cluster rather than a single point. `ConceptVolume` exists in `riemannian_density.py` but isn't the default. Enables Mahalanobis distance (shape-aware) and Bhattacharyya overlap.

**Relational Patterns Beyond Aggregate CKA**: CKA says "overall structure matches" but doesn't reveal if specific relational patterns (hierarchies, composition triangles, oppositions) are preserved. Graph-level analysis would validate whether compositional reasoning transfers.

**Transformations as Near-Isometries**: If `geodesic(a_i, a_j) ≈ geodesic(T(a_i), T(a_j))`, the transformation is a near-isometry. A LoRA isometry ratio distinguishes "extending" from "replacing."

**The "Generator IS the Transform" Pattern**: If a model understands a conceptual transformation (like "double" or "negate"), that concept's embedding might BE the transformation direction. Potential for concept-as-steering-vector extraction.

**Layer-wise ID for Probe Targeting**: Intrinsic dimension follows entry-ramp → highway → exit-ramp. Adaptive layer selection could focus geometry measurements on "highway" layers where structure is clearest.

**Geodesic Distance Reveals Hidden Structure**: Semantically related pairs are CLOSER in geodesic space than Euclidean distance suggests. Merge quality metrics using geodesic distance would catch incoherent merges that pass aggregate CKA tests.

### Script Mining Techniques

Techniques from 284 research scripts (exp9-exp87).

#### Distilled Logic Shapes
**Source**: `train_distilled_logic.py`

10 perfect examples > 10,000 mediocre ones. The model needs to learn the **shape** of logic, not surface patterns.

The 6 Logical Shapes:
1. **PERCENTAGE INCREASE**: new = original + (original × percent)
2. **AVERAGE RATE**: total_output / total_input (NOT mean of rates)
3. **THRESHOLD CROSSING**: breakeven + 1 = first profitable
4. **INVERSE CHAIN**: work backwards, undo operations in reverse
5. **SEQUENTIAL OPERATIONS**: subtract first, THEN multiply
6. **REMAINING FIRST**: compute what's left BEFORE applying rate

#### Counterfactual Sensitivity [EMPIRICAL]
**Source**: `counterfactual_sensitivity.py`, `geometric_knowledge_discovery.py`

Semantic invariance (paraphrase test) does NOT distinguish facts from opinions. **Counterfactual sensitivity** does.

- Factual statements: mean sensitivity ~0.25
- Opinion statements: mean sensitivity ~0.08
- Effect size: +0.94 (STRONG separation)

Use cases: detecting factual knowledge vs pattern matching, identifying missing capabilities, confidence calibration.

#### Generation-Based Evaluation [EMPIRICAL]
**Source**: `exp86_proper_evaluation.py`, `exp87_generation_based_self_improvement.py`

Single-token evaluation creates a false ceiling at ~70%. Generation-based evaluation reveals true capability (+20pp gap). Models reason correctly over multiple tokens but fail single-token prediction.

### MASS Validation + Open Questions
**Source:** `docs/research/lr_derivation_analysis.md`

MASS replaces the broken Lipschitz LR derivation. Validated on 350M (CE-only: healthy). CE+REINFORCE: still degraded (3× above sweet spot).

- [x] **√N budget distribution**: Confirmed empirically. Without √N: catastrophic (η=0.106). With √N: healthy (η=0.016). Implemented.
- [x] **REINFORCE gradient accounting**: Resolved (2026-02-22). Root cause: REINFORCE drew from the same Weyl budget as CE but wasn't accounted for. Fix: `target_step_norm = (sigma_k_min - update_norm) / sqrt(N_re)` — REINFORCE gets the remainder of the Weyl budget after CE, distributed via Brownian scaling. If CE exhausts the budget (`update_norm >= sigma_k_min`), REINFORCE is skipped. Telemetry: `outcome_budget_remaining`. Multi-seed closure runner is `scripts/reinforce_revalidation.py`.
- [ ] **Per-layer vs global η**: MASS uses global σ_k_min / σ_max. Per-layer ceiling would respect per-layer geometry. When does this matter?
- [ ] **SPS non-binding for fine-tuning**: SPS assumes f*=0, but fine-tuning loss is never near zero. SPS gives η ~0.3-1.4, never binding. Needs corrected f* or replacement.
- [ ] **Scale validation (8B+)**: Does MASS produce correct step sizes on Qwen3-8B and larger? (Seeded gate runner: `scripts/g5_8b_validation.py`)
- [ ] **Convergence analysis**: Under what conditions does min(ceiling, SPS, Weyl) converge?

### DPO as Variance-Reduction Alternative
**Blocked on:** MASS validation. Only relevant if REINFORCE variance remains a problem after LR is fixed.

DPO (Rafailov et al. 2023) converts preference learning into a classification-style loss, eliminating on-policy sampling. ModelCypher already generates (correct, incorrect) pairs from online eval — these map directly to DPO preference pairs.

**When to test:** After MASS fixes LR and REINFORCE is re-run. If variance (not LR) is the remaining bottleneck, DPO's implicit KL constraint and spectral bounds operate in different spaces (output distribution vs parameter perturbation) and may be complementary rather than conflicting.

**Caution:** DPO's implicit KL tethers the model to the reference policy. Under tight spectral bounds (NB-LoRA), this double constraint could under-fit. REINFORCE with good baselines (RLOO, GRPO) may be preferable if capacity is the binding constraint. Test empirically.

---

## Partially Unblocked

### Training Dynamics → Geometry
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §8

How do training hyperparameters affect geometry?

**Previously blocked on:** Training runs. Now partially unblocked — the NB-LoRA pipeline works at 350M, 1.2B, and 8B. Controlled experiments comparing geometry before/after training are now feasible.

- [ ] Compare layer geometry (SVD spectra, effective rank) pre- vs post-training
- [ ] Test whether Cayley-Stiefel preserves geometric structure better than plain SGD
- [ ] Measure how data-rank ceiling affects post-training geometry

---

## Known Constraints [EMPIRICAL]

**Source:** `docs/research/FAILURE-MODES.md`

| Constraint | Implication |
|------------|-------------|
| Layer combination interference | Single-layer compression is practical limit |
| MLP-only teaching limits | ~92% ceiling for MLP-only approaches |
| Gradient entanglement in math | Math domains need different approach |
| Geometry protection prevents capability transfer | Can't transfer specialist capability while preserving generalist geometry |
| **CE on reasoning traces = format memorization** [VALIDATED] | PPL, CKA, budget all look perfect while inference degrades. The optimizer is correct; the objective (CE) is the problem. Outcome-based training (REINFORCE) is the fix. |
| **MLX SVD crash on ill-conditioned matrices** | C++ abort, uncatchable. Use power iteration for runtime monitoring, `stream=mx.cpu` for all linalg. |
| **Lipschitz LR derivation via HVP** `[DISPROVEN]` | Central-difference HVP + power iteration values span 3 OOM across minibatches. 10-batch median doesn't help. Loss surface has (L₀,L₁)-relaxed smoothness (Zhang ICLR 2020). Replaced by MASS. |

---

## CLI Tools

```bash
# Training
poetry run mc train run --model /path/to/model --data /path/to/dataset --output /path/to/adapter
poetry run mc train star --model /path/to/model --output /path/to/adapter

# Analysis
poetry run mc model fingerprint /path/to/model
poetry run mc analyze spectral-trajectory --model /path -t -q
poetry run mc analyze entropy-trajectory --model /path -t -q
poetry run mc analyze dimension-profile --model /path -t -q
```

---

## References

| Document | Content |
|----------|---------|
| `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md` | Derivations, proofs, solved questions |
| `docs/research/geometric_capacity_paper_experiment_matrix.md` | Paper-to-experiment mapping with pass/falsify criteria |
| `docs/LFM2-350M-WORK-SUMMARY.md` | LFM2 project status |
| `data/experiments/archive/geometric_fingerprint_discovery.md` | expansion_ratio findings |
| `docs/research/lr_derivation_analysis.md` | MASS step size analysis + fallback candidates |
| `docs/research/field_map_external_methods.md` | External methods landscape (2024-2026) with ModelCypher mappings |
| `docs/research/architecture_geometry_theory.md` | Signal propagation, RMT, attention rank saturation, regime decomposition |

Papers:
- Moschella et al. (2023). Relative representations enable zero-shot latent space communication.
- DeepSeek-AI. (2025). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880.
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited.
- Huh et al. (2024). The Platonic Representation Hypothesis. ICML 2024.
- Reimann et al. (2017). Cliques of Neurons Bound into Cavities. Frontiers in Computational Neuroscience.
- Chen et al. (2025). A Unified Geometric Space Bridging AI Models and the Human Brain.
- Shen et al. (2025). Alignment between Brains and AI: Evidence for Convergent Evolution.
- Lin & Kriegeskorte (2024). The topology and geometry of neural representations. PNAS.
- Aghajanyan et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.
- Cheng et al. (2025). Emergence of a High-Dimensional Abstraction Phase in Language Transformers.
