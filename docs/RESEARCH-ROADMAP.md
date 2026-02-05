# Research Roadmap

**Updated:** 2026-02-04

Consolidated from: `OPEN-MATHEMATICAL-QUESTIONS.md`, `RESEARCH-MAP.md`, `LFM2-350M-WORK-SUMMARY.md`, `.claude/plans/geometric-research-plan.md`

---

## Status Summary

| Category | Solved | Partial | Open | Blocked |
|----------|--------|---------|------|---------|
| Mathematical Questions | 7 | 3 | 4 | 1 |
| Research Threads | 0 | 0 | 5 | 0 |
| Known Failure Modes | 0 | 0 | 3 | 0 |

---

## Priority 1: Open Mathematical Questions

### Q1: Training Hyperparameters → Subspace Allocation
**Status:** OPEN
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §2

What training hyperparameters determine Q/K subspace allocation?
- [ ] Can we predict subspace overlap from training recipe?
- [ ] What causes Granite vs Qwen to allocate subspaces differently despite same GQA?

**Approach:** Controlled training experiments varying single parameters.

---

### Q2: RLHF Geometry Flattening
**Status:** OPEN
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §3

Why do specialist models (instruct, code, reasoning) have expansion_ratio variance ≈ 0?

Experiments needed:
- [ ] Compare base vs instruct checkpoints of same model
- [ ] Measure geometry during RLHF training (if checkpoints available)
- [ ] Test if flat geometry is necessary or sufficient for instruction following

---

### Q3: Attention Eigenvalue Distribution
**Status:** PARTIAL
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §6

Remaining questions:
- [ ] Why does Qwen3 have sharper attention than Qwen2.5?
- [ ] Analytical relationship between GQA and attention sharpness?

---

### Q4: Layer-wise Invariants
**Status:** NOT STARTED
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §7

What properties are preserved vs transformed across layers?

Candidates to investigate:
- [ ] Norm (preserved? scaled?)
- [ ] Angles between vectors
- [ ] Rank of activation matrix
- [ ] Intrinsic dimension variation bounds

---

### Q5: Information-Theoretic Characterization
**Status:** NOT STARTED
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §9

- [ ] What is I(layer_i; layer_j) as function of |i-j|?
- [ ] Does MI decay exponentially?
- [ ] Is there an information bottleneck at highway?

---

### Q6: The Fundamental Question
**Status:** OPEN
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §10

**Can we write down an equation that predicts geometry from architecture?**

Current state: Qualitative predictions work (family-level). Quantitative predictions fail.

Path forward:
1. [ ] Controlled experiments: Train same architecture with varied single parameters
2. [ ] More model families: Test Llama, Mistral, Phi
3. [ ] Theoretical derivation: Why early vs late compression from attention/MLP mechanics

---

## Priority 2: LFM2-350M Geometric Self-Awareness

**Source:** `LFM2-350M-WORK-SUMMARY.md`

### Blockers (Must answer before training)

1. **Natural expansion_ratio distribution by task type**
   - [ ] Simple facts - expected range?
   - [ ] Complex reasoning (CRT) - expected range?
   - [ ] Creative tasks - expected range?
   - [ ] Code generation - expected range?

2. **Model size/architecture effects**
   - [ ] Does optimal expansion_ratio vary by model size?
   - [ ] Is LFM2-350M's φ=0.618 typical or architecture-specific?

3. **Attractor structure**
   - [ ] Single attractor or multiple basins?
   - [ ] Different optimal geometries for different task types?

### Next Steps (After blockers resolved)

- [ ] Inference-time geometric feedback: Route expansion_ratio back into forward pass
- [ ] Factual verification: Geometry misses confident hallucination - why?
- [ ] New adapter training with spectral scale bounds (`apply_lora_geometric()`)

---

## Priority 3: Research Threads

**Source:** `RESEARCH-MAP.md` Part VI

### Thread 1: Anchor-Relative Concept Grafting
**Status:** OPEN (design complete)

- [ ] Test on same-architecture pairs (known to work)
- [ ] Test on cross-architecture pairs (LFM2-700M → LFM2-350M)
- [ ] Implement `relative_representation.py` updates

### Thread 2: Cross-LoRA Transfer
**Status:** INVESTIGATIVE

- [ ] Train coding adapter on Llama-3
- [ ] Project to Qwen-2.5 using Procrustes
- [ ] Measure rotation field roughness

### Thread 3: Multi-Channel Architecture
**Status:** DESIGN PHASE

Combine null-space projection with DeepSeek's mHC.
- [ ] Design specification
- [ ] Prototype implementation

### Thread 4: Geometry Probe Extensions
**Status:** READY FOR INTEGRATION

| Extension | File | Status |
|-----------|------|--------|
| ConceptVolume by default | `riemannian_density.py` | Code exists |
| Relational pattern analyzer | new | Design ready |
| LoRA isometry ratio | new | Design ready |
| Adaptive layer selection | new | Design ready |
| Geodesic merge quality | new | Design ready |

### Thread 5: Script Mining Techniques
**Status:** READY FOR INTEGRATION

| Technique | Archive Location | Notes |
|-----------|------------------|-------|
| Distilled Logic Shapes | `train_distilled_logic.py` | 6 patterns > 10K examples |
| Counterfactual Sensitivity | `counterfactual_sensitivity.py` | Effect size 1.44 |
| Generation-Based Evaluation | `exp86_proper_evaluation.py` | Breaks 70% ceiling |
| Geometry-Derived Training | `geometry_derived_training.py` | LR = 1/(κ×scale) |

---

## Priority 4: Experiments (Geometric Research Plan)

**Source:** `.claude/plans/geometric-research-plan.md`

### Phase 1: Verify Measurements
- [ ] EXPERIMENT 1: Check SVD ratio directions (verify inverses exist)
- [ ] EXPERIMENT 10: Null hypothesis (random matrices)
- [ ] EXPERIMENT 11: Untrained vs trained models

### Phase 2: Understand Structure
- [ ] EXPERIMENT 6: Pre vs post nonlinearity geometry
- [ ] EXPERIMENT 7: Gram matrix eigenvalues
- [ ] EXPERIMENT 9: Residual stream tracking

### Phase 3: Test Manipulations
- [ ] EXPERIMENT 2: Orthogonal rotation
- [ ] EXPERIMENT 3: Uniform scaling
- [ ] EXPERIMENT 4: Surgical SVD modification

### Phase 4: Information Flow
- [ ] EXPERIMENT 8: Jacobian analysis
- [ ] EXPERIMENT 5: Rank-1 perturbation toward missing constant

---

## Known Failure Modes (Unsolved)

**Source:** `docs/research/FAILURE-MODES.md`

| Failure | Root Cause | Implication |
|---------|------------|-------------|
| Layer Combination Interference | "Compression quantum" - manifold shift between layers | Single-layer compression is practical limit |
| MLP-Only Teaching Limits | MLPs encode knowledge, not reasoning (needs attention) | ~92% ceiling for MLP-only approaches |
| Gradient Entanglement in Math | Math gradients more entangled (42% vs 78% survival) | Math domains need different approach |

---

## Blocked

### Training Dynamics → Geometry
**Source:** `OPEN-MATHEMATICAL-QUESTIONS.md` §8

How do training hyperparameters affect final geometry?

**Status:** BLOCKED - needs training runs

Hyperparameters to vary:
- Learning rate
- Batch size
- Weight decay
- Warmup schedule
- Dropout

---

## Solved (Reference)

These questions have been answered. See `OPEN-MATHEMATICAL-QUESTIONS.md` for details.

| Question | Key Finding |
|----------|-------------|
| Highway location determination | GQA → K capacity → Q/K alignment → attention selectivity |
| MLP nonlinearity geometry | Gate × Up multiplication is key, SiLU has minimal effect |
| Manifold topology preservation | β₀ constant (no tears), β₁ = reasoning signature |
| Layer Jacobian structure | Full-rank near-identity, not rank-1 |
| Decay formula | Norm-weighted average of component decays (no arbitrary constants) |
| Attention eigenvalue (LFM2) | Rank-1 attention explains uniform distribution |
| Exit convergence | Reasoning training reduces mean norm (not deviation) |

---

## Quick Reference: CLI Tools

```bash
# Geometric fingerprint (expansion_ratio by task)
poetry run mc model fingerprint /path/to/model

# Spectral trajectory
poetry run mc safety spectral-trajectory --model /path -t -q

# Entropy trajectory
poetry run mc safety entropy-trajectory --model /path -t -q

# Intrinsic dimension profile
poetry run mc safety dimension-profile --model /path -t -q

# Cognitive reflection test
poetry run mc safety cognitive-reflection-test --model /path
```

---

## Detailed Documents

| Document | Content |
|----------|---------|
| `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md` | Full derivations and proofs |
| `docs/LFM2-350M-WORK-SUMMARY.md` | LFM2 project status and adapters |
| `docs/RESEARCH-MAP.md` | External research connections |
| `docs/research/FAILURE-MODES.md` | Documented failure categories |
| `.claude/plans/geometric-research-plan.md` | Experimental protocol |
