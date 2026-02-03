# Expansion Ratio Distribution Analysis

## Date: 2026-01-30 (Updated: 2026-02-03)

## Key Finding: DeepSeek-R1's Constant Expansion Ratio

DeepSeek-R1 (8B reasoning model) shows **constant expansion_ratio ≈ 1.0** (flat trajectory) for ALL prompts.
- Zero variance across 35 prompts in 9 categories
- Peak always at final layer (36/36)
- This is fundamentally different from base models

## Historical Note on Metrics

**The raw data below uses the OLD metric:** `(peak_dim / final_dim) / φ`

When this analysis was conducted, we divided by φ (1.618). PHI_FINDINGS.md later showed this φ normalization has no theoretical justification.

- Old metric value 0.618 = raw expansion_ratio 1.0 (flat trajectory)
- Old metric value 1.0 = raw expansion_ratio 1.618
- Old metric value 1.268 = raw expansion_ratio 2.05

The key finding (specialist models have flat trajectories) remains valid. The raw data values should be multiplied by 1.618 to get the actual expansion_ratio.

## Raw Results

### LFM2-350M (Base Model)
```
Overall expansion_ratio range: [0.618, 1.268]
Overall expansion_ratio mean: 0.866 +/- 0.195
Overall expansion_ratio median: 0.880
Between-category variance: 0.0368

Per-category means:
- simple_facts:     1.123 (easy retrieval → high comp/φ)
- math_simple:      1.086 (pattern matching → high comp/φ)
- code:             0.978 (structured → near 1.0)
- creative:         0.900 (generation → moderate)
- logic_complex:    0.731 (reasoning → lower)
- math_reasoning:   0.746 (multi-step → lower)
- crt_reasoning:    0.669 (intuitive traps → lower)
- logic_simple:     0.658 (syllogisms → lower)
- chain_of_thought: 0.618 (explicit CoT → exactly 1/φ!)
```

### LFM2-1.2B (Larger Base Model)
```
Overall expansion_ratio range: [0.618, 0.788]
Overall expansion_ratio mean: 0.659 +/- 0.046
Overall expansion_ratio median: 0.640
Between-category variance: 0.0024

Per-category means:
- simple_facts:     0.738
- code:             0.692
- creative:         0.654
- math_simple:      0.664
- logic_complex:    0.640
- logic_simple:     0.623
- crt_reasoning:    0.626
- math_reasoning:   0.621
- chain_of_thought: 0.618 (exactly 1/φ)
```

### DeepSeek-R1 (RL-Tuned Reasoning Model)
```
Overall expansion_ratio range: [0.618, 0.618]
Overall expansion_ratio mean: 0.618 +/- 0.000
Overall expansion_ratio median: 0.618
Between-category variance: 0.0000

ALL categories: exactly 0.618
Peak layer: always 36/36 (final layer)
```

## Interpretation

### Pattern 1: Reasoning Tasks Converge to 1/φ
Across all models, reasoning tasks (CoT, CRT, logic) trend toward 0.618.
- LFM2-350M: CoT = 0.618, CRT = 0.669, logic = 0.658-0.731
- LFM2-1.2B: CoT = 0.618, nearly all tasks approach 0.62-0.64
- DeepSeek-R1: ALL tasks = 0.618

### Pattern 2: Model Size → Lower Variance
- LFM2-350M: high variance (0.195 std), strong task-type effects
- LFM2-1.2B: low variance (0.046 std), tasks converging
- DeepSeek-R1: zero variance, complete convergence

### Pattern 3: RL Training Produces Stable Geometry
DeepSeek-R1 was trained with RLHF on reasoning tasks. The result:
- Compression geometry is frozen at a constant value
- This suggests RL optimization found a stable attractor
- The mechanism is unknown; whether this value is "optimal" is an open question

## Hypotheses

### H1: Reasoning Models Converge to Stable Geometry
When models are optimized for coherent reasoning (either through scale or RL),
they converge to stable expansion_ratio values.

Evidence:
- CoT prompts show consistent expansion_ratio in base models
- Larger models show lower variance overall
- RL-trained reasoning model has near-zero variance

### H2: Different Processing Modes Have Different Ratios
expansion_ratio may vary by processing mode:
- Simple facts: higher ratio (retrieval-heavy)
- Creative: lower ratio (generation-heavy)
- Code: moderate ratio (balanced)

### H3: Task Type Determines Natural Geometry
Different tasks may have different natural geometries:
- Retrieval tasks: higher expansion_ratio (more aggressive compression)
- Reasoning tasks: lower expansion_ratio (more balanced processing)
- The "optimal" value is likely task-dependent, not universal

## Implications for Training

1. **Don't assume a universal optimal target**
   - Different tasks may have different natural geometries
   - A single target value may be inappropriate

2. **RL produces stable attractors**
   - DeepSeek-R1's constant geometry suggests RL found a stable point
   - The mechanism is unknown

3. **Training should be empirically informed**
   - Measure natural distribution before choosing targets
   - Consider task-specific training if appropriate

## Answer to Research Questions

Q1: Is expansion_ratio signal or noise?
**SIGNAL**. Clear task-type dependence across models. Not random.

Q2: Why does DeepSeek-R1 show constant expansion_ratio?
**RL training converged to a stable attractor**. Whether this value is
"optimal" or just a local minimum is an open question.

Q3: Should we train toward expansion_ratio = 1.0?
**UNKNOWN** - requires more research. The assumption that 1.0 is optimal
for all tasks has not been validated. Measure empirical distributions first.

## Next Steps

1. Test other RL-tuned models to see if constant-geometry pattern holds
2. Characterize natural expansion_ratio distributions across diverse tasks
3. Investigate why peak always at final layer for DeepSeek-R1
4. Test whether training toward measured natural values improves performance
