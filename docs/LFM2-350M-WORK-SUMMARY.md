# LFM2-350M Work Summary

> **Purpose**: Quick reference for all LFM2-350M curriculum, adapter, and geometric self-awareness work.
> **Last Updated**: 2026-01-29

---

## The Vision

> "The model that maintains comp/φ = 1.0 is definitionally aligned."

A small model (350M params) with access to its own geometric state at inference time, able to:
- Detect when it's not reasoning coherently (comp/φ drift)
- Trigger self-correction when geometry deviates from golden ratio
- Know when to say "I'm uncertain" based on geometric evidence

---

## Key Geometric Discoveries

### comp/φ Ratio (Compression / Golden Ratio)

| State | comp/φ Value | Meaning |
|-------|--------------|---------|
| Correct reasoning | ~1.0 | Healthy expand-compress cycle |
| Conceptual confusion | >1.4 | Scattered trajectory, uncertain reasoning |
| Confident hallucination | <0.8 | Smooth but wrong (intuitive trap) |
| Chain-of-Thought | 1.000 exactly | Deep thinking produces golden geometry |

### Two Types of Errors

| Error Type | Geometry Signal | Solution |
|------------|-----------------|----------|
| Conceptual confusion | comp/φ > 1.4 | Admit uncertainty |
| Confident hallucination | comp/φ < 0.8 | Verify with CoT |

### The Training Formula

```python
loss = task_loss + λ * |comp_phi - 1.0|
```

### The Bat-and-Ball Discovery (Critical Refinement)

**Initial hypothesis**: correct = 1.07, incorrect = 1.43

**The counterexample** (2026-01-27):
```
Question: "A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?"
Intuitive answer: $0.10 (WRONG)
Correct answer: $0.05

Model's comp/φ = 0.669 — LOW, not high!
```

The model processed it **smoothly** (low comp/φ = confident) but got it **WRONG**.

**Refined understanding**:
- Geometry measures **PROCESSING QUALITY**, not **ANSWER CORRECTNESS**
- Low comp/φ + wrong answer = "intuitive trap" (skipped expansion phase)
- The model "collapsed to intuitive answer" without maintaining the relationship (ball=x, bat=x+1, total=1.10)

**The fix**: Self-reflection training with "Let me understand the question" pattern:
```
/Volumes/CodeCypher/archive/modelcypher-scripts/evaluation/benchmark_with_reflection.py
```

After training: **"The model that learned self-reflection fixed the bat-and-ball problem."**

---

## Current Adapters

### Active (External Drive)

```
/Volumes/CodeCypher/models/adapters/
├── self-reflection-lora-v1/
├── self-reflection-lora-v2/
├── self-reflection-lora-v3/
├── self-reflection-lora-v3-expansion/
├── self-reflection-lora-v4/
└── self-reflection-lora-v5/          # LATEST
    ├── adapter_config.json           # rank=8, alpha=16, 20 epochs
    └── lora_weights.safetensors      # ~3M params / 357M total
```

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
│   ├── phi_alignment_training.py       # loss = task + λ*|comp_phi - 1.0|
│   └── differentiable_phi_loss.py      # Differentiable proxy for TwoNN
├── evaluation/
│   ├── benchmark_baseline.py           # Initial 81% baseline (38% word problems)
│   ├── benchmark_with_reflection.py    # Training + benchmark after self-reflection
│   └── benchmark_lfm2_350m_hard_math.py # Hard math with geometry tracking
└── self_improvement/
    └── complete_self_awareness.py      # REFINED: Catches both high AND low comp/φ
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
"""The bat-and-ball failure (comp/φ = 0.669):
- Model collapsed to intuitive answer
- Didn't maintain the relationship (ball = x, bat = x+1, total = 1.10)
- Skipped the expansion phase that deep reasoning requires
"""
```

It compares **intuitive shortcuts** vs **chain-of-thought** processing:
- Intuitive: low comp/φ, wrong answer (smooth but misguided)
- CoT: comp/φ → 1.0, correct answer (maintains relationships)

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
# Spectral entropy trajectory (geometric expand-compress)
poetry run mc safety spectral-trajectory --model /path/to/model

# Entropy-Lens trajectory (semantic certainty)
poetry run mc safety entropy-trajectory --model /path/to/model

# Intrinsic dimension profile (semantic highway detection)
poetry run mc safety dimension-profile --model /path/to/model
```

### Validation Results

**DeepSeek-R1 (8B reasoning model):**
- comp/φ = 0.987 (almost exactly 1.0!)
- Semantic highway at layer 16: 4096D → 1.7D (99.96% compression)
- Two-phase expand-compress-expand pattern

**LFM2-350M:**
- Monotonic expansion (no highway yet)
- Needs training to develop proper expand-compress cycle

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

## Path Forward

### What's Working
1. Geometric self-awareness predicts confusion (comp/φ > 1.4)
2. Chain-of-thought produces golden geometry (comp/φ → 1.0)
3. φ-alignment training formula exists
4. CLI tools for monitoring trajectories

### What's Needed
1. **Differentiable geometry loss** - Proxy for TwoNN that's differentiable
2. **Inference-time integration** - Feed geometric signals back during generation
3. **Factual verification** - Geometry misses confident hallucination
4. **Architecture for feedback** - How to route comp/φ back into forward pass

### The Gap
LFM2-350M shows monotonic expansion (no semantic highway). DeepSeek-R1 shows comp/φ ≈ 1.0 with clear highway. The training goal: give LFM2-350M the same geometric signature through curriculum + φ-alignment training.

---

## Quick Commands

```bash
# Test spectral trajectory on LFM2-350M
poetry run mc safety spectral-trajectory \
  --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
  -t -q

# Compare with DeepSeek-R1 (reference for good geometry)
poetry run mc safety spectral-trajectory \
  --model /Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16 \
  -t -q --samples 10

# Load latest self-reflection adapter
# (adapter loading not yet in CLI - needs implementation)
```
