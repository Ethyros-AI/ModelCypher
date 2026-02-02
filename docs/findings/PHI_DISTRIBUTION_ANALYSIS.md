# Expansion Ratio Distribution Analysis: Model Classification via Variance

**Date:** 2026-01-31 (Updated: 2026-02-02)
**Status:** Validated

## Summary

The expansion_ratio variance across task types is a reliable discriminator for model classification:

| Model | Mean Expansion Ratio | Std | Classification |
|-------|----------------------|-----|----------------|
| DeepSeek-R1-8B | 1.00 | 0.000 | **SPECIALIST** |
| LFM2-1.2B | 1.07 | 0.073 | GENERAL |
| LFM2-350M | 1.40 | 0.316 | **BASE** |

## Key Finding: DeepSeek-R1 Shows Constant Expansion Ratio

DeepSeek-R1 (reasoning specialist) shows **approximately constant** expansion_ratio across ALL 35 measurements and ALL 9 task categories with **near-zero variance**.

The consistency suggests the model maintains stable processing geometry regardless of task type. The specific value observed (approximately 1.0) may reflect RL training convergence, but whether this is "optimal" is an open question.

### Interpretation

- **Specialist models** (DeepSeek-R1, Qwen-Coder): Constant geometry because they're optimized for a single domain
- **Base models** (LFM2-350M): High variance because they differentiate between task types
- **General/instruct models** (LFM2-1.2B): Moderate variance - some task differentiation but more constrained

## Data Location

```
data/experiments/phi_distribution_deepseek_r1.json  # Historical - uses old naming
data/experiments/phi_distribution_lfm2_1p2b.json    # Historical - uses old naming
data/experiments/phi_distribution_lfm2_350m.json    # Historical - uses old naming
```

## Category Breakdown (LFM2-350M)

Shows how base models vary by task:

| Category | Mean Expansion Ratio | Note |
|----------|---------------------|------|
| simple_facts | 1.81 | Higher ratio (retrieval) |
| math_simple | 1.76 | Higher ratio |
| creative | 1.46 | Moderate |
| code | 1.59 | Moderate |
| crt_reasoning | 1.08 | Lower ratio |
| math_reasoning | 1.21 | Lower ratio |
| logic_simple | 1.07 | Lower ratio |
| logic_complex | 1.18 | Lower ratio |
| chain_of_thought | 1.00 | Consistent low ratio |

### Pattern

- **Retrieval/simple tasks**: Higher expansion_ratio - model expands then compresses aggressively
- **Reasoning tasks**: Lower expansion_ratio - more balanced processing
- **Chain-of-thought**: Consistent low ratio - stable processing geometry

## CLI Tool

Use `mc model fingerprint` to classify any model:

```bash
mc model fingerprint /path/to/model

# Output:
# MODEL FINGERPRINT: BASE
# Expansion Ratio Statistics:
#   Variance: 0.122309
# Classification: High geometric variation
```

## Implications for Merging

1. **Specialist → Base merging** may fail because specialists lack task differentiation capability
2. **Base → Specialist merging** can transfer capabilities via null-space projection
3. **Geometric fingerprint** should be checked before merge to predict success

## Next Steps

- [ ] Validate pattern on more models (Llama, Mistral, Phi)
- [ ] Correlate expansion_ratio variance with downstream benchmark performance
- [ ] Test if dimension recovery correlates with expansion_ratio variance
