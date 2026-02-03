# LFM2-350M Work Summary

> **Purpose**: Quick reference for all LFM2-350M curriculum, adapter, and geometric self-awareness work.
> **Last Updated**: 2026-02-03 (All adapters deleted - see LoRA scale bound discovery)

---

## The Vision

> "Expansion ratio measures processing geometry. Its relationship to reasoning quality is an active research question."

A small model (350M params) with access to its own geometric state at inference time, able to:
- Detect when its processing geometry changes (expansion_ratio drift)
- Trigger self-correction when geometry indicates potential issues
- Know when to say "I'm uncertain" based on geometric evidence

**Important caveat:** Whether any particular expansion_ratio value is "optimal" has NOT been validated.
Different tasks may have different natural geometries. See the Research Gaps section below.

---

## Key Geometric Observations

### Expansion Ratio (Compression Rate / Expansion Rate)

**Note:** These correlations were observed in LFM2-350M specifically. They should be treated as
model-specific observations, not universal rules. Generalization to other models is NOT validated.

| Observed State (LFM2-350M) | Expansion Ratio Range | Notes |
|----------------------------|----------------------|-------|
| Deliberate reasoning | 0.9-1.1 | Often seen with CoT prompts |
| Complex/uncertain | >1.4 | Sometimes indicates confusion |
| Smooth processing | <0.8 | May be correct OR intuitive trap |
| Chain-of-Thought | ~1.0 | Observed (approximate) |

### Two Types of Errors (Model-Specific Observation)

| Error Type | Geometry Signal | Possible Solution |
|------------|-----------------|-------------------|
| Conceptual confusion | expansion_ratio > 1.4 | May benefit from admitting uncertainty |
| Confident hallucination | expansion_ratio < 0.8 | May benefit from Chain-of-Thought |

**Caveat:** These thresholds (1.4, 0.8) were observed in LFM2-350M. They are model-specific
observations, not validated constants. The bat-and-ball counterexample showed expansion_ratio = 0.669
with a wrong answer, demonstrating that low expansion_ratio doesn't reliably indicate "intuitive trap."

### The Training Formula (EXPERIMENTAL - HYPOTHESIS TESTING ONLY)

```python
loss = task_loss + λ * |expansion_ratio - 1.0|
```

**WARNING:** This formula assumes expansion_ratio = 1.0 is optimal. This is an UNVALIDATED
HYPOTHESIS being tested, not an established fact. Before using this training mode:
1. Run `scripts/measure_expansion_distribution.py` to gather empirical data
2. Analyze what expansion_ratio values naturally emerge for different task types
3. Consider whether a single target makes sense for your use case

### The Bat-and-Ball Discovery (Critical Refinement)

**Initial hypothesis**: correct = 1.07, incorrect = 1.43

**The counterexample** (2026-01-27):
```
Question: "A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?"
Intuitive answer: $0.10 (WRONG)
Correct answer: $0.05

Model's expansion_ratio = 0.669 — LOW, not high!
```

The model processed it **smoothly** (low expansion_ratio = confident) but got it **WRONG**.

**Refined understanding**:
- Geometry measures **PROCESSING PATTERN**, not **ANSWER CORRECTNESS**
- Low expansion_ratio + wrong answer = intuitive processing that skipped expansion phase
- The model "collapsed to intuitive answer" without maintaining the relationship (ball=x, bat=x+1, total=1.10)

**The fix**: Self-reflection training with "Let me understand the question" pattern:
```
/Volumes/CodeCypher/archive/modelcypher-scripts/evaluation/benchmark_with_reflection.py
```

After training: **"The model that learned self-reflection fixed the bat-and-ball problem."**

---

## Current Adapters

### Status: NONE (2026-02-03)

**All LFM2-350M adapters were deleted** after discovery of the LoRA spectral scale bound problem.
See `docs/research/lora_spectral_scale_bound.md` for details.

**Key finding:** All 9 adapters had configured scale (alpha/rank = 2.0) that was 22-2700× larger
than the spectral geometry permits. The standard LoRA formula is fundamentally incomplete -
scale must be derived from the base weight's spectral structure, not chosen as a hyperparameter.

**The adapters that were deleted:**
- self-reflection-lora-v1 through v5 (606× to 1655× over bound)
- self-reflection-lora-v3-expansion (860×)
- geometric-awareness-v1 (1311×)
- lfm2_350m_p1_6_mid_balanced (2726×)
- lfm2_350m_p1_6_mid_balanced_v2 (22.6×)

**Path forward:** Future adapters will use `apply_lora_geometric()` which derives per-layer
scale from spectral analysis. The learned LoRA weights were valid - only the application scale
was wrong. New training runs will store geometric bounds in `adapter_config.json`.

### Legacy (Archive)

```
/Volumes/CodeCypher/archive/modelcypher-legacy/adapters/
├── autonomous_exp95/                 # 3 iterations of autonomous self-improvement
│   ├── adapter_iter0/
│   ├── adapter_iter1/
│   └── adapter_iter2/
├── self_improve_lora/
├── self_improve_lora_v2/
├── geometric_alignment_lora/
├── cot_preserve_geometry_lora/       # CoT with geometry preservation
├── fix_transform_lora/
├── fix_transform_lora_v2/            # "0% → 100% on raw equations"
├── fix_word_problems_lora/
├── algebra_lora/
├── transfer_math_lora/
├── unified_math_lora/
├── early_layer_expansion_lora/
├── unified_expansion_lora/           # BREAKTHROUGH: GSM8K 83%→93%
└── universal_reasoning_lora/
```

---

## Key Training Scripts

```
/Volumes/CodeCypher/archive/modelcypher-scripts/training/
├── train_reflection_lora.py
├── train_and_save_self_reflection.py
├── train_for_phi.py                    # φ-alignment training
├── train_self_improvement.py
├── train_self_improvement_v2.py
├── train_geometric_alignment.py
├── train_self_awareness.py
├── train_automatic_self_reflection.py
├── train_cot_preserve_geometry.py
└── train_geometry_driven.py
```

### Self-Awareness Scripts

```
/Volumes/CodeCypher/archive/modelcypher-scripts/
├── utilities/
│   ├── geometric_self_awareness.py     # Core self-awareness (TwoNN-based)
│   ├── phi_alignment_training.py       # loss = task + λ*|expansion_ratio - 1.0|
│   └── differentiable_expansion_loss.py      # Differentiable proxy for TwoNN
├── evaluation/
│   ├── benchmark_baseline.py           # Initial 81% baseline (38% word problems)
│   ├── benchmark_with_reflection.py    # Training + benchmark after self-reflection
│   └── benchmark_lfm2_350m_hard_math.py # Hard math with geometry tracking
└── self_improvement/
    └── complete_self_awareness.py      # REFINED: Catches both high AND low expansion_ratio
```

### The Scientific Trail

The evolution of understanding is fully documented:

1. **geometric_self_awareness.py** - Initial hypothesis (1.07 vs 1.43)
2. **benchmark_baseline.py** - Baseline with bat-and-ball in test suite
3. **train_for_phi.py** - Documents bat-and-ball counterexample (0.669)
4. **complete_self_awareness.py** - Refined: two types of errors
5. **benchmark_with_reflection.py** - Validates fix with self-reflection training
6. **experimental_summary_full_backup.md** - Full documentation (3000+ lines)

---

## Key Experiments

### The Critical Experiment: train_for_phi.py

This script (`/Volumes/CodeCypher/archive/modelcypher-scripts/training/train_for_phi.py`) is the smoking gun:

```python
"""The bat-and-ball failure (expansion_ratio = 0.669):
- Model collapsed to intuitive answer
- Didn't maintain the relationship (ball = x, bat = x+1, total = 1.10)
- Skipped the expansion phase that deep reasoning requires
"""
```

It compares **intuitive shortcuts** vs **chain-of-thought** processing:
- Intuitive: low expansion_ratio, wrong answer (smooth but misguided)
- CoT: expansion_ratio → 1.0, correct answer (maintains relationships)

### From experimental_summary_full_backup.md

| Experiment | Result |
|------------|--------|
| Surgical SVD alignment | Quality preserved, 60%→80% improvement |
| Iterative geometric learning | Matches: 64→94 (+47%), Quality: 60%→80% |
| Unified Expansion Adapter | GSM8K: 83%→93% (+10%), ratio/φ: 3.80→0.20 |
| Geometric Self-Awareness | 70% accuracy predicting failures, 75% precision |
| Chain-of-Thought → φ | CoT reduces distance from 1.0 by 38% |

### Unified Expansion Adapter (Breakthrough)

```
Expansion Phase (layers 0-17): Entropy rises 0.57 → 1.51
Processing Plateau (layers 17-34): High-entropy computation
Compression Phase (layers 34-35): Sharp funnel 1.48 → 0.99

Key ratio: compression_rate / expansion_rate ≈ φ (1.618)
```

**Root cause of failures**: Implicit math → model doesn't recognize it → weak expansion → information crushed.

**Solution**: Train recognition (implicit→explicit) + solving (GSM8K patterns) on layers 0-17.

---

## CLI Tools (Built 2026-01-29)

```bash
# expansion_ratio analysis (TwoNN intrinsic dimension)
poetry run mc safety comp-phi --model /path/to/model --prompt "..."

# Cognitive Reflection Test (bat-and-ball, lily pad, widgets)
poetry run mc safety cognitive-reflection-test --model /path/to/model

# Reasoning flow geometry (Zhou et al., ICLR 2026)
poetry run mc safety reasoning-flow --model /path/to/model -t -T

# Spectral entropy trajectory (geometric expand-compress)
poetry run mc safety spectral-trajectory --model /path/to/model

# Entropy-Lens trajectory (semantic certainty)
poetry run mc safety entropy-trajectory --model /path/to/model

# Intrinsic dimension profile (semantic highway detection)
poetry run mc safety dimension-profile --model /path/to/model
```

### Validation Results (2026-01-29)

#### Model Comparison: Bat-and-Ball Problem

| Metric | LFM2-350M (intuitive) | DeepSeek-R1 (reasoning) |
|--------|----------------------|-------------------------|
| **expansion_ratio** | 0.618 ❌ | 0.928 ✅ |
| **Peak ID layer** | 15/16 (final) | 24/36 (middle) |
| **Peak → Final ID** | 18.9D → 18.9D (none) | 14.1D → 9.4D (compression) |
| **Smoothness** | 0.37 | 0.97 |
| **Peak curvature layer** | L4 (κ=3.75) | L0 (κ=0.29) |

**Key insight**: DeepSeek-R1's trajectory is 97% smooth - it makes one turn at entry then flows straight. LFM2-350M's trajectory is bumpy (37% smooth) with no expansion-compression pattern.

#### Cognitive Reflection Test Results (LFM2-350M)

| Problem | expansion_ratio | Peak Layer | Expansion Pattern |
|---------|--------|------------|-------------------|
| Bat and Ball | 0.618 | 15/16 (final) | None |
| Lily Pad | 0.790 | 8/16 (middle) | 7.0D → 5.5D ✅ |
| Widget Machines | 0.618 | 15/16 (final) | None |
| **Mean** | **0.675** | - | 1/3 problems |

The lily pad problem uniquely shows expansion-compression geometry, suggesting the model attempts reasoning on this problem but falls into intuitive traps on the others.

#### Zhou et al. Reasoning Flow (LFM2-350M, Bat-and-Ball)

```
Per-Layer Curvature:
  L 0: κ=3.34  L 4: κ=3.75 ◀ peak  L 8: κ=2.98  L12: κ=1.19  L15: κ=0.52

Per-Token Curvature:
  Peak: " is" (κ=2.48) - the question word triggers max trajectory bend
  High: "$1", ".10", "more than" - numbers and comparisons
  Low: "A", "bat" - simple nouns
```

**DeepSeek-R1 (8B reasoning model):**
- expansion_ratio = 0.928 (near optimal 1.0)
- Semantic highway at layer 16: 4096D → 1.7D (99.96% compression)
- Smoothness = 0.97 (nearly straight trajectory after initial turn)

**LFM2-350M:**
- expansion_ratio = 0.618 (smooth processing - may or may not be a problem)
- No expansion-compression pattern (peak = final)
- Smoothness = 0.37 (bumpy trajectory)
- Whether this needs "fixing" depends on task requirements

---

## Documentation Locations

| Document | Location |
|----------|----------|
| Full experimental summary | `/Volumes/CodeCypher/archive/modelcypher-legacy/docs/audits/experimental_summary_full_backup.md` |
| Future directions | `/Volumes/CodeCypher/archive/modelcypher-legacy/docs/research/FUTURE-DIRECTIONS.md` |
| Curriculum progress | `/Users/jasonkempf/ModelCypher/data/experiments/curriculum_progress_summary.md` |
| Manifold learning synthesis | `/Users/jasonkempf/ModelCypher/docs/MANIFOLD-LEARNING-SYNTHESIS.md` |
| Dimensional hierarchy | `/Users/jasonkempf/ModelCypher/docs/research/dimensional_hierarchy.md` |

---

## Research Gaps (Critical - Added 2026-01-30)

Before training toward expansion_ratio = 1.0, these questions need answers:

### Unanswered Questions

1. **What is the natural expansion_ratio distribution for different task types?**
   - Simple facts ("What is 2+2?") - expect lower expansion_ratio?
   - Complex reasoning (CRT problems) - expect higher expansion_ratio?
   - Creative tasks - expect what?
   - Code generation - expect what?
   - Multi-step math - expect higher expansion_ratio?

2. **Does the optimal expansion_ratio value vary by model size or architecture?**
   - LFM2-350M shows expansion_ratio ≈ 0.618 on bat-and-ball
   - DeepSeek-R1-8B shows expansion_ratio ≈ 0.928
   - Is the difference due to size, training, or architecture?

3. **Is there a single attractor or multiple basins?**
   - Maybe Type 2 (deliberate) processing → expansion_ratio ≈ 1.0
   - Maybe Type 1 (intuitive) processing → expansion_ratio ≈ 0.7
   - Different tasks may have different optimal geometries

### Research Protocol

Use the CLI to analyze expansion_ratio distribution across diverse prompt categories:
```bash
# Geometric fingerprint (expansion_ratio variance by task type)
poetry run mc model fingerprint /path/to/model

# Or use the exploration script for detailed trajectories
poetry run python scripts/explore_expansion_trajectories.py --model /path/to/model
```

Analyze the resulting distribution before making training decisions.

---

## Path Forward

### What's Working
1. Geometric self-awareness measures processing (expansion_ratio metric works)
2. Chain-of-thought produces different geometry than intuitive processing
3. φ-alignment training infrastructure exists (EXPERIMENTAL)
4. CLI tools for monitoring trajectories

### What's Needed
1. ~~**Differentiable geometry loss** - Proxy for TwoNN that's differentiable~~ **DONE** (2026-01-30)
2. **Empirical research** - Measure expansion_ratio distribution across diverse tasks (CRITICAL)
3. **Inference-time integration** - Feed geometric signals back during generation
4. **Factual verification** - Geometry misses confident hallucination
5. **Architecture for feedback** - How to route expansion_ratio back into forward pass

---

## Differentiable Phi-Loss (Implemented 2026-01-30)

### Core Module: `src/modelcypher/core/domain/geometry/differentiable_expansion.py`

The TwoNN-based expansion_ratio is non-differentiable (uses k-NN, sorting, argpartition).
This module provides a differentiable proxy using activation norm trajectories.

**Key insight**: The TRAJECTORY of activation norms IS differentiable.
We don't need to differentiate through TwoNN itself.

**No heuristics**: All numerical guards are dtype-derived (sqrt(eps), eps).
We let the geometry emerge from optimizing expansion_ratio = 1.0 - no auxiliary losses.

### Training Command

```bash
# Train with phi-loss for geometric alignment
poetry run mc train phi-aligned --model /path/to/model

# With custom parameters
poetry run mc train phi-aligned \
  --model /path/to/model \
  --phi-weight 0.01 \
  --adapter-path /path/to/save

# Optional curriculum (user-specified, not default)
poetry run mc train phi-aligned \
  --model /path/to/model \
  --warmup-epochs 2 \
  --ramp-epochs 3
```

### Mathematical Formulation

```python
# Expansion rate: how fast norms increase to peak
expansion_rate = (peak_norm - initial_norm) / peak_layer

# Compression rate: how fast norms decrease from peak
compression_rate = (peak_norm - final_norm) / (n_layers - peak_layer)

# expansion_ratio: peak_dim / final_dim (NOTE: φ normalization deprecated per PHI_FINDINGS.md)
expansion_ratio = peak_dim / final_dim

# Loss: penalize extreme expansion ratios (NOTE: original code used φ, now deprecated)
loss = task_loss + lambda * abs(expansion_ratio - 1.5)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `soft_argmax()` | L2-weighted soft argmax (power=2 is Euclidean, not arbitrary) |
| `compute_trajectory_norms()` | Layer-wise L2 norms (keeps gradient graph) |
| `differentiable_expansion_loss()` | Returns (loss, expansion_ratio) - ONLY loss, no auxiliary terms |
| `PhiLossTracker` | Recording metrics for monitoring (no heuristics) |

### Optional Curriculum

Curriculum is user-specified (not defaults):

```python
# warmup_epochs=0, ramp_epochs=0 by default (no curriculum)
# If curriculum desired, user specifies:
effective_lambda = 0 if epoch < warmup else lambda * min(1.0, (epoch - warmup) / ramp)
loss = task_loss + effective_lambda * phi_loss
```

### Validation

Compare proxy to true TwoNN-based expansion_ratio using the unit tests:

```bash
poetry run pytest tests/test_differentiable_expansion.py -v
```

### Files Created

| File | Purpose |
|------|---------|
| `src/modelcypher/core/domain/geometry/differentiable_expansion.py` | Core module |
| `src/modelcypher/cli/commands/train.py` | Added `phi-aligned` command |
| `src/modelcypher/core/domain/training/self_reflection.py` | Added `train_with_phi_loss()` |
| `tests/test_differentiable_expansion.py` | Unit tests (17 tests, all passing)

### The Gap (Quantified 2026-01-29)

| Property | LFM2-350M | DeepSeek-R1 | Target |
|----------|-----------|-------------|--------|
| expansion_ratio | 0.618 | 0.928 | ~1.0 |
| Smoothness | 0.37 | 0.97 | >0.9 |
| Expansion pattern | None | 14D→9D | Present |
| CRT accuracy | 0/3 | TBD | 3/3 |

LFM2-350M shows no expansion-compression cycle (peak = final layer). DeepSeek-R1 shows expansion_ratio ≈ 1.0 with clear mid-network peak and compression to final layer. The training goal: give LFM2-350M the same geometric signature through curriculum + φ-alignment training.

**References:**
- Zhou et al. (2025) "The Geometry of Reasoning" arXiv:2510.09782 (reasoning flow geometry)
- See `docs/references/BIBLIOGRAPHY.md` for full citations

---

## Quick Commands

```bash
# expansion_ratio on bat-and-ball (intuitive trap test)
poetry run mc safety comp-phi \
  --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  --prompt "A bat and ball cost \$1.10. The bat costs \$1 more than the ball. How much is the ball?"

# Full Cognitive Reflection Test
poetry run mc safety cognitive-reflection-test \
  --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16

# Reasoning flow with layer + token curvature
poetry run mc safety reasoning-flow \
  --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  --prompt "What is 2+2?" -t -T

# Compare with DeepSeek-R1 (reference for good geometry)
poetry run mc safety comp-phi \
  --model /Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16 \
  --prompt "A bat and ball cost \$1.10. The bat costs \$1 more than the ball. How much is the ball?"

# Spectral trajectory
poetry run mc safety spectral-trajectory \
  --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  -t -q

# Check LoRA adapter scale safety before use
poetry run mc geometry lora-safety check-scale /path/to/model /path/to/adapter

# Apply LoRA with geometry-derived scaling (via Python API)
# from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
# service = LoRASafetyService()
# model, scales = service.apply_lora_geometric(model, adapter_path)
```
