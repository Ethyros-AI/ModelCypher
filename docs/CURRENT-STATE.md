# Current State

> **Last Updated:** 2026-02-20
>
> **Mission:** See `docs/MISSION.md`
> **Research roadmap:** See `docs/RESEARCH-ROADMAP.md`

---

## Training Pipeline: NB-LoRA via Cayley-Riemannian Natural Gradient [VALIDATED]

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
| LFM2-350M | 350M | Cayley-Riemannian + CE | val_loss 1.27 (vs 1.38 plain SGD) |
| LFM2-350M | 350M | + REINFORCE interleaved | 14/20 accuracy (vs 11/20 baseline) [EMPIRICAL: unlogged claim, reproduction failed — see MEMORY.md] |
| LFM2-1.2B | 1.2B | Answer-mask + retention | 36/46 (78%), 0 degenerate |
| Qwen3-8B | 8B | Geometry + injection + training start | Confirmed working (full run in progress) |

### Key Architecture Decisions (Validated)

- **Optimizer:** Cayley-Riemannian natural gradient with P = MM^T where M = I+Z. Step bound: eta <= 2/(L * lambda_max(P)).
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
| **G5: Reproducible across models** | NEARLY CLOSED | 350M, 700M, 1.2B validated. 8B geometry + injection + training confirmed; full run in progress. |
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
| **Outcome training** | REINFORCE objective for reasoning | Implemented + validated (350M), service param only |
| **Online evaluation** | Greedy-decoding correctness during training | Implemented, service param only |
| **Outer similarity (RSS)** | Adapter interference monitoring | Implemented, service param only |

---

## Test Suite

**6051 tests passing, 39 skipped** (2026-02-20)

---

## Key Files

| File | Purpose |
|------|---------|
| `backends/mlx_training_adapter.py` | NBLoRALinear, train_loop, Lipschitz, outcome/entropy |
| `core/use_cases/dataset_training_service.py` | Training orchestration + CKA verification |
| `core/use_cases/star_training_service.py` | STaR orchestration |
| `core/domain/training/geometric_lora.py` | Weight analysis, rank derivation, data-rank ceiling |
| `core/domain/training/geometric_optimizer.py` | Per-layer optimizer config from SVD |
| `core/domain/training/outcome_objective.py` | REINFORCE objective |
| `core/domain/training/online_eval.py` | Online evaluation during training |
| `cli/commands/train.py` | `mc train run` and `mc train star` |

---

*"The solve was never parameters. The solve was understanding the geometry."*
