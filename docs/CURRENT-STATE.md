# Current Research State

> **Last Updated:** 2026-01-29

---

## Focus: Geometric Self-Alignment

Training small models to recognize and leverage their own geometric structure for capability improvement.

### Core Hypothesis

The alignment problem reframed: instead of teaching models what humans want, give models access to their own manifold geometry. The knowledge is already in the weights - what's missing is the ability to observe and self-correct.

---

## Active Model

**LFM2-350M** (Liquid Foundation Model 2, 350M parameters)

- Location: `/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16`
- Hidden dimension: 960
- Layers: 16
- Architecture: SwiGLU MLP, grouped-query attention

Why this model:
- Small enough for rapid iteration
- Large enough to exhibit meaningful geometry
- Same architecture family as larger LFM2 variants (scalable research)

---

## Training Progression

### Phase 1: Atomic Inference Rules

**Adapter:** `data/adapters/phase1_inference_rules/`

| Metric | Value |
|--------|-------|
| Examples | 64 (8 rules × 8 each) |
| Loss | 0.9961 → 0.0436 |
| Peak Layer | 7 |
| Null Space Activation | 87% |

Rules trained: Modus Ponens, Modus Tollens, Disjunctive Syllogism, Hypothetical Syllogism, Reductio ad Absurdum, Addition, Simplification, Conjunction

### Phase 2: Rule Compositions

**Adapter:** `data/adapters/phase2_rule_compositions/`

| Metric | Value |
|--------|-------|
| Examples | 53 |
| Loss | 0.8759 → 0.0303 |
| Peak Layer | 12 |
| Null Space Activation | 39% |

Compositions: HS+MP chains, DS+MP chains, triple-step reasoning

### Phase 3: Rule Recognition

**Adapter:** `data/adapters/phase3_rule_recognition/`

| Metric | Value |
|--------|-------|
| Examples | 48 |
| Loss | 1.9094 → 0.0563 |
| Peak Layer | 14 |
| Null Space Activation | 39% |

Meta-cognitive task: identify which rule applies to a given argument

### Phase 4: Conciseness

**Adapter:** `data/adapters/phase4_conciseness/`

Training on producing concise, confident outputs without hedging

### Cumulative Adapter

**Adapter:** `data/adapters/phases_1_4_cumulative/`

Combined training on all phases for unified capability

---

## Key Findings

### 1. Domain Fingerprints: Structure vs Facts

Cross-scale analysis (350M, 700M, 1.2B) reveals geometric structure by domain:

| Domain | Rank | Status |
|--------|------|--------|
| **Linguistic** | 126 | Rich, stable geometry |
| **Computational** | 211 | Richest geometry |
| Math | 1 | Collapsed to single dimension |
| Affective | 1 | Collapsed |
| Temporal | 1 | Collapsed |
| Moral | 1 | Collapsed |
| Safety | 1 | Collapsed |
| Physical | 1 | Collapsed |
| **Factual** | 255 | Full rank but 99.2% zeros |

**Insight:** Language and computation are "native" domains with real geometric structure. Everything else is projected onto single dimensions. The models don't lack parameters - they lack **activated geometry**.

### 2. Null Space Activation Pattern

LoRA training primarily activates **null space** in expansion layers (w1, w3):
- Phase 1: 87% null space (adding without overwriting)
- Phases 2-3: 39% null space (some modification)

This suggests we're activating latent structure, not creating new capabilities.

### 3. Peak Layer Progression

As tasks become more abstract, the peak change layer moves deeper:
- Pattern detection (Phase 1): Layer 7
- Composition (Phase 2): Layer 12
- Classification (Phase 3): Layer 14

### 4. Layer 7: The Computational Singularity

Consistently shows entropy peak across diverse prompts - maximum uncertainty before resolution. This is the transition from feature detection to abstract reasoning.

---

## Verified Merge (2026-01-22)

**Same-architecture merge works:**

- Pipeline: LFM2-700M → LFM2-350M
- ID: `pipeline-d43443bb`
- Output: `/Volumes/CodeCypher/models/merged/test-merge-2026-01-22`
- Inference test: "2+2 = 4" with coherent explanation
- Speed: 478 tokens/second

**Cross-architecture merge fails:**
- DeepSeek-R1-Qwen3-8B → LFM2-1.2B produces orthogonal outputs
- Safety mechanisms implemented to revert to target weights

---

## Next Steps

### Immediate

1. **Expand domain coverage** - Train logical rules for more domains beyond inference
2. **Test generalization** - Does rule training transfer to novel problems?
3. **Implement real-time geometry observation** - Model can query its own entropy during generation

### Near-term

4. **Grassmannian signature diagnostics** - Compute positive minor ratios for concept pairs
5. **LoRA generation from geometry** - Given diagnostic, generate corrective adapter
6. **Self-alignment loop prototype** - Model detects and fixes its own geometric misalignment

### Long-term

7. **Autonomous alignment** - Continuous self-monitoring and intervention
8. **Alignment transfer** - Share geometric corrections across model instances

---

## Repository Structure

### Active Adapters

```
data/adapters/
├── phase1_inference_rules/      # Atomic rules
├── phase2_rule_compositions/    # Chaining
├── phase3_rule_recognition/     # Meta-cognition
├── phase4_conciseness/          # Output quality
└── phases_1_4_cumulative/       # Combined training
```

### Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/research/GEOMETRIC-SELF-ALIGNMENT.md` | Philosophical foundation |
| `docs/research/positive_geometry_scale_comparison.md` | Domain fingerprint analysis |
| `docs/research/COMPRESSION-RESEARCH-SYNTHESIS.md` | T-matrix compression findings |
| `docs/WHY-GEOMETRY-MATTERS.md` | High-level motivation |

### Key Code

| Path | Purpose |
|------|---------|
| `src/modelcypher/core/domain/geometry/` | Geometric analysis tools |
| `src/modelcypher/core/use_cases/merge/` | Model merging pipeline |
| `src/modelcypher/core/use_cases/train/` | Training orchestration |

---

## Commands

### Run geometry analysis

```bash
mc geometry report model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16
```

### Train adapter

```bash
mc train run \
  --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  --data data/training/phase1_inference_rules.jsonl \
  --output data/adapters/phase1_inference_rules
```

### Test inference

```bash
mc infer run \
  --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  --adapter data/adapters/phases_1_4_cumulative \
  --prompt "If P then Q. P. Therefore?"
```

---

*"The solve was never parameters. The solve was understanding the geometry."*
