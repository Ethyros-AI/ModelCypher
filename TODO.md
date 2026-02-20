# ModelCypher TODO

**Updated:** 2026-02-20

---

## Code Projects

| Project | Location | Purpose |
|---------|----------|---------|
| **ModelCypher** | `/` | Main geometric analysis toolkit for neural networks |
| **Plasma** | `/plasma/` | Tokamak plasma geometry analysis (fusion research application) |

---

## Code Tasks

### ModelCypher Core

**Open:**
- [ ] G5: Complete full 8B training run (Qwen3-8B) — geometry + injection confirmed, training enters at 8.1 tok/sec, full run not yet completed
- [ ] Validate outcome-based training (REINFORCE) on 1.2B — proven on 350M (14/20 vs 11/20), needs scale validation
- [ ] Entropy reg + answer-mask mutual exclusivity fix — currently in `else` branch (line ~2092 of `mlx_training_adapter.py`), needs refactor to apply independently

**Recently completed (2026-02-20):**
- Data-rank ceiling: `min(tail_dims, n_train_samples)` — 8B params 2.76B → 927M (2.91x reduction)
- Duplicate SVD elimination: `derive_optimizer_geometry_config()` accepts precomputed geometries
- Streaming B_crit estimation: two-pass constant-memory gradient noise estimation
- SVD `compute_uv=False` optimization: ~3x faster geometry analysis
- G1 magic numbers resolved: LR backoff floor from `sqrt(eps_f32)`, bootstrap CI from data
- Outcome-based training validated on 350M: 14/20 (70%) vs 11/20 (55%) baseline
- STaR training service + `mc train star` CLI
- Adapter routing service + benchmarking
- Composite adapter builder + routed generation service
- Outer similarity (RSS) monitoring
- Newton-Schulz orthogonalization for gradient preconditioning
- Budget cap and max epochs envelope parameters
- Experimental CLI flags removed per Research vs Production Policy
- Scripts for cluster-swap ablation and fast attractor testing

**Previously completed (2026-02-13):**
- Training domain test coverage: 15 new test files (257 training domain tests)
- Fixed 3 bugs in `training_notifications.py` (missing logger, wrong class references)

### Plasma Subproject

- [ ] Complete TODO/FIXME items in `plasma/src/diiid_loader.py`
- [ ] Complete TODO/FIXME items in `plasma/src/data_loader.py`

---

## Implementation Backlog

Techniques from research that could become CLI features (per Research vs Production Policy):

| Technique | Origin | Status |
|-----------|--------|--------|
| Adapter routing | Implemented + benchmarked | Needs real-model validation for CLI promotion |
| Composite adapter builder | Implemented | Needs real-model validation for CLI promotion |
| Routed generation | Implemented | Needs real-model validation for CLI promotion |
| Concepts as Geometric Clusters | Research design doc | Design ready |
| Counterfactual Sensitivity | Archived code | Code exists in archive |
| Generation-Based Evaluation | Archived code | Code exists in archive |
| LoRA Isometry Ratio | Research design doc | Design ready |
| Geodesic Merge Quality | Research design doc | Design ready |

---

## Research

**All research tracking consolidated in: `docs/RESEARCH-ROADMAP.md`**

Quick links:
- Open mathematical questions: `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md`
- LFM2-350M project: `docs/LFM2-350M-WORK-SUMMARY.md`
- Failure modes: `docs/research/FAILURE-MODES.md`

---

*Source files: docs/research/*.md, plasma/src/*
