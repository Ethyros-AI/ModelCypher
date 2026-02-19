# Current Research State

> **Last Updated:** 2026-02-04
>
> **Next steps & research roadmap:** See `docs/RESEARCH-ROADMAP.md`

---

## Focus: Geometric Self-Alignment

Training small models to recognize and leverage their own geometric structure for capability improvement.

### Core Hypothesis

The alignment problem reframed: instead of teaching models what humans want, give models access to their own manifold geometry. The knowledge is already in the weights - what's missing is the ability to observe and self-correct.

---

## Active Model

**LFM2-350M** (Liquid Foundation Model 2, 350M parameters)

- Location: `/path/to/models/example-model`
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

### Script Mining Findings (2026-01-29)

Mining 284 research scripts (exp9-exp87, ~105K lines) revealed a coherent research arc:

| Phase | Scripts | Finding |
|-------|---------|---------|
| 1: Compression | exp9-exp33 | Marchenko-Pastur filtering works; gate layers at 85% SVD energy |
| 2: Golden Layer | exp38-exp44 | Optimal layer at ~67% depth across architectures |
| 3: Cross-Arch | exp45-exp55 | CKA=0.9255 via F=pinv(src)@tgt; single direction achieves 91.7% |
| 4: Failures | exp56-exp65 | Pattern interference; entropy-gated teaching |
| 5: Self-Improvement | exp66-exp75 | Models can observe own manifold; geometric self-play works |
| 6: Scaling | exp76-exp82 | 70% ceiling; teacher bridging breaks through |
| 7: Limits | exp83-exp87 | Generation-based eval shows 20pp gap over single-token |

**Documented Failure Modes:** See `docs/research/FAILURE-MODES.md`

**Novel Techniques:** See `docs/RESEARCH-ROADMAP.md` (Threads 4-5)

---

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
| `docs/GEOMETRY-GUIDE.md` | Geometry motivation + reporting guide |

### Key Code

| Path | Purpose |
|------|---------|
| `src/modelcypher/core/domain/geometry/` | Geometric analysis tools |
| `src/modelcypher/core/use_cases/merge/` | Model merging pipeline |
| `src/modelcypher/core/use_cases/train/` | Training orchestration |

---

## Commands

### Inspect model

```bash
mc model info /path/to/models/example-model
```

### Train adapter

```bash
mc train run \
  --model /path/to/models/example-model \
  --data data/training/phase1_inference_rules.jsonl \
  --output data/adapters/phase1_inference_rules
```

### Test inference

```bash
mc infer run \
  --model /path/to/models/example-model \
  --adapter data/adapters/phases_1_4_cumulative \
  --prompt "If P then Q. P. Therefore?"
```

---

*"The solve was never parameters. The solve was understanding the geometry."*
