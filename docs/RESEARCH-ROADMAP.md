# Research Roadmap

**Updated:** 2026-02-04

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

**Note:** Controlled training experiments blocked on training runs.

---

## Research Threads

**Source:** `RESEARCH-MAP.md` Part VI

### Thread 1: Anchor-Relative Concept Grafting
- [ ] Test on same-architecture pairs
- [ ] Test on cross-architecture pairs (LFM2-700M → LFM2-350M)

### Thread 2: Cross-LoRA Transfer
- [ ] Train coding adapter on Llama-3
- [ ] Project to Qwen-2.5 using Procrustes
- [ ] Measure rotation field roughness

### Thread 3: Multi-Channel Architecture
- [ ] Design specification combining null-space projection with mHC

### Thread 4: Geometry Probe Extensions
| Extension | Status |
|-----------|--------|
| ConceptVolume by default | Code exists |
| Relational pattern analyzer | Design ready |
| LoRA isometry ratio | Design ready |
| Geodesic merge quality | Design ready |

### Thread 5: Script Mining Techniques
| Technique | Archive Location |
|-----------|------------------|
| Distilled Logic Shapes | `train_distilled_logic.py` |
| Counterfactual Sensitivity | `counterfactual_sensitivity.py` |
| Generation-Based Evaluation | `exp86_proper_evaluation.py` |

---

## Blocked

### Training Dynamics → Geometry
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §8

How do training hyperparameters affect geometry?

**Blocked on:** Training runs (need to train same arch with varied params)

---

## Known Constraints

**Source:** `docs/research/FAILURE-MODES.md`

| Constraint | Implication |
|------------|-------------|
| Layer combination interference | Single-layer compression is practical limit |
| MLP-only teaching limits | ~92% ceiling for MLP-only approaches |
| Gradient entanglement in math | Math domains need different approach |
| Geometry protection prevents capability transfer | Can't transfer specialist capability while preserving generalist geometry |

---

## CLI Tools

```bash
poetry run mc model fingerprint /path/to/model
poetry run mc safety spectral-trajectory --model /path -t -q
poetry run mc safety entropy-trajectory --model /path -t -q
poetry run mc safety dimension-profile --model /path -t -q
```

---

## Reference Documents

| Document | Content |
|----------|---------|
| `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md` | Derivations, proofs, solved questions |
| `docs/LFM2-350M-WORK-SUMMARY.md` | LFM2 project status |
| `docs/PHI_FINDINGS.md` | φ numerology analysis |
| `data/experiments/geometric_fingerprint_discovery.md` | expansion_ratio findings |
| `data/experiments/phi_distribution_analysis.md` | Task-type distribution data |
