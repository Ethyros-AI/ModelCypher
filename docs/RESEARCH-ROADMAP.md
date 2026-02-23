# Research Roadmap

**Updated:** 2026-02-22

---

## Protocol

All major claims in this roadmap are governed by:
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`
- `docs/research/deep_research_integration_2026_02.md` (canonical integration log for external deep-research reports)

Status labels and promotion rules (`[CONJECTURAL]`, `[VALIDATED]`, `[DISPROVEN]`, etc.) are
defined in `docs/EVIDENCE-TAXONOMY.md` and apply to every thread below.

---

## Open Questions

### Q1: Layer-wise Invariants
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §7

What properties are preserved vs transformed across layers?

- [ ] Norm (preserved? scaled?)
- [ ] Angles between vectors
- [ ] Rank of activation matrix
- [ ] Intrinsic dimension variation bounds

---

### Q2: Qwen3 vs Qwen2.5 Attention Sharpness
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §6

Why does Qwen3 have sharper attention than Qwen2.5 despite similar architecture?

- [ ] Identify architectural differences
- [ ] Analytical relationship between config and attention rank

---

### Q3: Information-Theoretic Characterization
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §9

- [ ] What is I(layer_i; layer_j) as function of |i-j|?
- [ ] Does MI decay exponentially?
- [ ] Is there an information bottleneck at highway?

---

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
| **Outcome-based training (REINFORCE)** | Mechanism validated; Weyl remainder budget implemented | Original 14/20 claim unlogged. Reproduction: 18/25 → 9/25 (Lipschitz LR=0.996). Root cause = LR, not REINFORCE. **MASS:** CE-only healthy. CE+REINFORCE at old target: -2 from baseline (REINFORCE drew from CE's budget). **Fix (2026-02-22):** Weyl remainder budget — REINFORCE gets `(sigma_k_min - CE_displacement) / sqrt(N_re)`. Awaiting re-validation. |
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
| **Spectral optimizers** | Muon (polar factor), SOAP (Shampoo eigenbasis), Spectra (spectral shaping) | Cayley-Stiefel preconditioner (P = M M^T) | Implemented. Constraint-driven pullback metric (P ≈ I in practice; Stiefel constraint is the active mechanism). |
| **LoRA rank** | SR-LoRA (stable rank), EVA (activation SVD, in HF PEFT), SARA (SV energy), GeLoRA (ID lower bound) | `tail_dims = full_rank - floor(shannon_eff_rank)` | Implemented. Unique null-space capacity approach. |
| **Layer targeting** | Spectrum (Marchenko-Pastur SNR, in Axolotl) | `tail_dims > 0` (spectral decay analysis) | Implemented. Worth comparing against Spectrum. |
| **Stopping criteria** | Heavy-tailed spectral stopping (α → 2.5), ε-rank staircase | 4-arm geometric stopping certificate + adapter saturation | Implemented. α monitoring could complement. |
| **Unified system** | None exists (field map conclusion) | ModelCypher | The only system deriving LR, rank, layer targeting, weight decay, stopping from unified spectral analysis. |

**Fallback candidates if MASS proves insufficient:** D-Adaptation (distance geometry, no curvature), Muon-inspired spectral-norm step control (per-layer). See `docs/research/lr_derivation_analysis.md`.

---

## Research Threads

**Source:** `RESEARCH-MAP.md` Part VI

### Anchor-Relative Concept Grafting
- [ ] Test on same-architecture pairs
- [ ] Test on cross-architecture pairs (LFM2-700M → LFM2-350M)

### Cross-LoRA Transfer
- [ ] Train coding adapter on Llama-3
- [ ] Project to Qwen-2.5 using Procrustes
- [ ] Measure rotation field roughness

### Multi-Channel Architecture
- [ ] Design specification combining null-space projection with mHC

### Geometry Probe Extensions
| Extension | Status |
|-----------|--------|
| ConceptVolume by default | Code exists |
| Relational pattern analyzer | Design ready |
| LoRA isometry ratio | Design ready |
| Geodesic merge quality | Design ready |

### Script Mining Techniques
| Technique | Archive Location |
|-----------|------------------|
| Distilled Logic Shapes | `train_distilled_logic.py` |
| Counterfactual Sensitivity | `counterfactual_sensitivity.py` |
| Generation-Based Evaluation | `exp86_proper_evaluation.py` |

### MASS Validation + Open Questions
**Source:** `docs/research/lr_derivation_analysis.md`

MASS replaces the broken Lipschitz LR derivation. Validated on 350M (CE-only: healthy). CE+REINFORCE: still degraded (3× above sweet spot).

- [x] **√N budget distribution**: Confirmed empirically. Without √N: catastrophic (η=0.106). With √N: healthy (η=0.016). Implemented.
- [x] **REINFORCE gradient accounting**: Resolved (2026-02-22). Root cause: REINFORCE drew from the same Weyl budget as CE but wasn't accounted for. Fix: `target_step_norm = (sigma_k_min - update_norm) / sqrt(N_re)` — REINFORCE gets the remainder of the Weyl budget after CE, distributed via Brownian scaling. If CE exhausts the budget (`update_norm >= sigma_k_min`), REINFORCE is skipped. Telemetry: `outcome_budget_remaining`. Awaiting re-validation run.
- [ ] **Per-layer vs global η**: MASS uses global σ_k_min / σ_max. Per-layer ceiling would respect per-layer geometry. When does this matter?
- [ ] **SPS non-binding for fine-tuning**: SPS assumes f*=0, but fine-tuning loss is never near zero. SPS gives η ~0.3-1.4, never binding. Needs corrected f* or replacement.
- [ ] **Scale validation (8B+)**: Does MASS produce correct step sizes on Qwen3-8B and larger?
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
| **Lipschitz LR derivation via HVP is broken** `[VALIDATED]` | Central-difference HVP + power iteration values span 3 OOM across minibatches. 10-batch median doesn't help. Loss surface has (L₀,L₁)-relaxed smoothness (Zhang ICLR 2020). Replaced by MASS. |

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

## Reference Documents

| Document | Content |
|----------|---------|
| `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md` | Derivations, proofs, solved questions |
| `docs/research/geometric_capacity_paper_experiment_matrix.md` | Paper-to-experiment mapping with pass/falsify criteria |
| `docs/LFM2-350M-WORK-SUMMARY.md` | LFM2 project status |
| `docs/PHI_FINDINGS.md` | φ numerology analysis |
| `data/experiments/geometric_fingerprint_discovery.md` | expansion_ratio findings |
| `data/experiments/phi_distribution_analysis.md` | Task-type distribution data |
| `docs/research/lr_derivation_analysis.md` | MASS step size analysis + fallback candidates |
| `docs/research/field_map_external_methods.md` | External methods landscape (2024-2026) with ModelCypher mappings |
| `docs/research/architecture_geometry_theory.md` | Signal propagation, RMT, attention rank saturation, regime decomposition |
