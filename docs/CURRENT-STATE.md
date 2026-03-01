# Current State

> **This document is archived.** For current project state, see [MISSION.md](MISSION.md).
> For the research roadmap, see [RESEARCH-ROADMAP.md](RESEARCH-ROADMAP.md).
>
> The content below is a point-in-time snapshot from 2026-02-23.

---

## Training Pipeline: NB-LoRA via Cayley-Stiefel Retraction + MASS [VALIDATED]

<!-- evidence: VALIDATED | scope: 350M, 700M, 1.2B | date: 2026-02-20 | method: 4-arm x 3-seed ablation, multi-model -->

**Status: Production-ready on 350M-1.2B. 8B validation in progress.**

One command, zero configuration:

```bash
mc train run --model /path/to/model --data /path/to/dataset --output /path/to/adapter
```

Every parameter derived from geometry. See `docs/MISSION.md` for the 15 hyperparameters and their geometric replacements.

### Validated Results

| Model | Scale | Training | Outcome |
|-------|-------|----------|---------|
| LFM2-350M | 350M | Cayley-Stiefel + CE | val_loss 1.27 (vs 1.38 plain SGD) |
| LFM2-350M | 350M | + REINFORCE interleaved | [DISPROVEN] Revalidation (2026-02-23): REINFORCE gradient orthogonal to CE on 350M. Degradation monotonic with steps. Model lacks latent capability for RLVR. |
| LFM2-1.2B | 1.2B | Answer-mask + retention | 36/46 (78%), 0 degenerate |
| Qwen3-8B | 8B | Geometry + injection + training start | IN PROGRESS; G5 seeded gate runner implemented (`scripts/g5_8b_validation.py`), full validation pending |

### Key Architecture Decisions (Validated)

- **Optimizer:** Cayley-Stiefel retraction (orthogonality constraint on NB-LoRA factors). Pullback metric P = MM^T was removed 2026-02-23 after falsification showed P ≈ I throughout training (||P-I||/√r median 0.001, Fisher condition number 1.95×10⁸ with 99.96% eigenvalues degenerate — Karakida 2021). The Stiefel constraint is the active mechanism. Step-size control: MASS `eta_step = min(eta_ceiling, eta_sps, eta_weyl)` where `eta_ceiling = σ_k_min / (σ_max × √N)` (Weyl 1912).
- **Rank:** Per-layer from tail_dims = full_rank - floor(shannon_eff_rank), capped by data-rank ceiling min(tail_dims, n_train_samples).
- **Cross-projection coupling:** q_proj rank capped at k_proj tail_dims per attention layer.
- **Stopping:** 4 criteria (val loss stable, val loss increasing, adapter saturation exhausted, max iterations circuit breaker).
- **Verification:** CKA alignment to base model, spectral bounds by construction.

### What Failed (Don't Repeat)

- **Constrained training (--paired):** Ablation showed constraints monotonically hurt. Disabled.
- **SFT CE on reasoning traces:** Format memorization. PPL drops, inference degrades. The objective is the problem, not the optimizer.
- **Cross-projection rank coupling alone:** Improved knowledge but amplified repetition. Root cause is objective (CE), not attention rank.
- **ScaledGD on Stiefel manifold:** Wrong — ScaledGD is for unconstrained low-rank, not orthogonality-constrained.

---

## Mission Guardrail Status

| Guardrail | Status | Evidence |
|-----------|--------|----------|
| **G1: Zero magic numbers** | CLOSED | All thresholds from SVD, IEEE 754, or measured data. LR backoff = sqrt(eps_f32), bootstrap CI = data-derived. |
| **G2: Spectral safety** | CLOSED | NB-LoRA Cayley parameterization bounds by construction. |
| **G3: Data-derived convergence** | CLOSED | 4-arm x 3-seed ablation (2026-02-17). |
| **G4: Capability preservation** | CLOSED | CKA verification post-training. |
| **G5: Reproducible across models** | IN PROGRESS | 350M, 700M, 1.2B validated. 8B: geometry analysis complete, full training validation pending (`scripts/g5_8b_validation.py`). |
| **G6: Verifiable quality** | CLOSED | Spectral bounds, CKA, concept volume, mode connectivity all implemented. |
| **G7: Falsifiability** | CLOSED | Protocol at `GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`. |

---

## New Services (Experimental, Not Yet CLI)

| Service | Purpose | Status |
|---------|---------|--------|
| **STaR training** | Self-Taught Reasoner orchestration | Implemented, `mc train star` CLI exists |
| **Adapter routing** | Divergence-based multi-adapter routing | Implemented + benchmarked, no CLI |
| **Composite adapter builder** | Build adapters from multiple sources | Implemented, no CLI |
| **Routed generation** | Multi-adapter inference with routing | Implemented, no CLI |
| **Outcome training** | REINFORCE objective for reasoning | Implemented + validated (350M); 1.2B frontier matrix runner implemented (`scripts/reinforce_revalidation.py`) |
| **Online evaluation** | Greedy-decoding correctness during training | Implemented, service param only |
| **Outer similarity (RSS)** | Adapter interference monitoring | Implemented, service param only |

---

## Test Suite

**6,809 tests passing** (2026-03-01)

---

## Key Files

| File | Purpose |
|------|---------|
| `backends/mlx_training_adapter.py` | NBLoRALinear, train_loop, MASS step control, outcome/entropy |
| `core/use_cases/dataset_training_service.py` | Training orchestration + CKA verification |
| `core/use_cases/star_training_service.py` | STaR orchestration |
| `core/domain/training/geometric_lora.py` | Weight analysis, rank derivation, data-rank ceiling |
| `core/domain/training/geometric_optimizer.py` | Per-layer optimizer config from SVD |
| `core/domain/training/outcome_objective.py` | REINFORCE objective |
| `core/domain/training/online_eval.py` | Online evaluation during training |
| `cli/commands/train.py` | `mc train run` and `mc train star` |

---

*"The solve was never parameters. The solve was understanding the geometry."*
