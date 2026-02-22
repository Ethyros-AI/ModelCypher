# Research Roadmap

**Updated:** 2026-02-20

---

## Protocol

All major claims in this roadmap are governed by:
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`

Status labels and promotion rules (`OPEN`, `SUPPORTED`, `FALSIFIED`) are
defined there and apply to every thread below.

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
| **NB-LoRA Cayley-Riemannian** | Production-ready | val_loss 1.27 vs 1.38 (350M), scales to 8B |
| **Outcome-based training (REINFORCE)** | Validated (350M) | 14/20 (70%) vs 11/20 (55%) baseline |
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

---

## Partially Unblocked

### Training Dynamics → Geometry
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §8

How do training hyperparameters affect geometry?

**Previously blocked on:** Training runs. Now partially unblocked — the NB-LoRA pipeline works at 350M, 1.2B, and 8B. Controlled experiments comparing geometry before/after training are now feasible.

- [ ] Compare layer geometry (SVD spectra, effective rank) pre- vs post-training
- [ ] Test whether Cayley-Riemannian preserves geometric structure better than plain SGD
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
