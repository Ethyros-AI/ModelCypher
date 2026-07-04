# ModelCypher TODO

**Updated:** 2026-07-04

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
- [ ] G5: Complete credibility proof (Qwen3-8B) — owner rerun required on Apple Silicon. The 2026-03-01 CLI credibility run did not complete: `results/cli_credibility_2026-03-01/train_result.json.tmp` is a 0-byte tmp file. Do not report G5 as complete until a non-tmp `train_result.json`, pre/post benchmark JSON, degeneration trace, and gates are present.
- [ ] Entropy reg + answer-mask mutual exclusivity fix — currently in `else` branch, needs refactor to apply independently

**Recently completed (2026-02-25 through 2026-03-01):**
- Gradient accumulation for 8B training (OOM fix): memory-safe micro-batch probe + grad accumulation [VALIDATED]
- Pre/post benchmark evaluation via `--benchmark quick` flag (GSM8K, ARC-Easy, BoolQ)
- Degeneration measurement alignment: per-epoch check using few-shot prompts, 512 tokens, 20 samples
- `degeneration_exceeded` stopping criterion
- MP-weighted Tikhonov null-space projector (A/B validation: won all 5 metrics vs binary projector)
- Binary projector mode removed from codebase
- ActivationProviderAdapter delegation fix (4 methods)
- `mc quantize correct` CLI promotion
- bf16 SVD guard for `compute_per_layer_signal_ranks`
- Delegation contract tests for ActivationProviderAdapter. Current test collection count is generated in `README.md` by `scripts/update_test_count.py`.
- K-FAC removed after its validation path failed to justify product complexity; see `docs/research/REFUTATION-LEDGER.md`

**Previously completed (2026-02-20):**
- Data-rank ceiling: `min(tail_dims, n_train_samples)` — 8B params 2.76B → 927M (2.91x reduction) [VALIDATED]
- Duplicate SVD elimination: `derive_optimizer_geometry_config()` accepts precomputed geometries
- Streaming B_crit estimation: two-pass constant-memory gradient noise estimation
- SVD `compute_uv=False` optimization: ~3x faster geometry analysis
- G1 magic numbers resolved: LR backoff floor from `sqrt(eps_f32)`, bootstrap CI from data
- STaR training service + `mc train star` CLI
- Adapter routing service + benchmarking
- Newton-Schulz orthogonalization for gradient preconditioning

**Closed (no longer pursuing):**
- ~~Validate REINFORCE on 1.2B~~ — REINFORCE on 350M closed 2026-02-23 (gradient orthogonal to CE, degradation monotonic). 1.2B attempt deferred until CE-based pipeline fully validated.

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
