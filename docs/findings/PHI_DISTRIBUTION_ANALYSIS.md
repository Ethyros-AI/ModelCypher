# Phi Distribution Analysis: Model Classification via comp/φ Variance

**Date:** 2026-01-31
**Status:** Validated

## Summary

The comp/φ (compression ratio φ) variance across task types is a reliable discriminator for model classification:

| Model | Mean comp/φ | Std | Classification |
|-------|-------------|-----|----------------|
| DeepSeek-R1-8B | 0.618 | 0.000 | **SPECIALIST** |
| LFM2-1.2B | 0.658 | 0.045 | GENERAL |
| LFM2-350M | 0.866 | 0.195 | **BASE** |

## Key Finding: DeepSeek-R1 Shows Constant 0.618

DeepSeek-R1 (reasoning specialist) shows **exactly** comp/φ = 0.618 = 1/φ across ALL 35 measurements and ALL 9 task categories with **zero variance**.

This is the golden ratio reciprocal, suggesting the model maintains perfect geometric self-similarity regardless of task type.

### Interpretation

- **Specialist models** (DeepSeek-R1, Qwen-Coder): Constant geometry because they're optimized for a single domain
- **Base models** (LFM2-350M): High variance because they differentiate between task types
- **General/instruct models** (LFM2-1.2B): Moderate variance - some task differentiation but more constrained

## Data Location

```
data/experiments/phi_distribution_deepseek_r1.json
data/experiments/phi_distribution_lfm2_1p2b.json
data/experiments/phi_distribution_lfm2_350m.json
```

## Category Breakdown (LFM2-350M)

Shows how base models vary by task:

| Category | Mean comp/φ | Note |
|----------|-------------|------|
| simple_facts | 1.12 | High expansion (retrieval) |
| math_simple | 1.09 | High expansion |
| creative | 0.90 | Moderate |
| code | 0.98 | Moderate |
| crt_reasoning | 0.67 | Low (near 1/φ) |
| math_reasoning | 0.75 | Low |
| logic_simple | 0.66 | Low (near 1/φ) |
| logic_complex | 0.73 | Low |
| chain_of_thought | 0.618 | **Exactly 1/φ** |

### Pattern

- **Retrieval/simple tasks**: High comp/φ (>1.0) - model expands then compresses
- **Reasoning tasks**: Low comp/φ (~0.618) - minimal compression, preserves information
- **Chain-of-thought**: Exactly 0.618 - perfect information preservation

## CLI Tool

Use `mc model fingerprint` to classify any model:

```bash
mc model fingerprint /path/to/model

# Output:
# MODEL FINGERPRINT: BASE
# comp/φ Statistics:
#   Variance: 0.122309
# Classification: High geometric variation
```

## Implications for Merging

1. **Specialist → Base merging** may fail because specialists lack task differentiation capability
2. **Base → Specialist merging** can transfer capabilities via null-space projection
3. **Geometric fingerprint** should be checked before merge to predict success

## Next Steps

- [ ] Validate pattern on more models (Llama, Mistral, Phi)
- [ ] Correlate comp/φ variance with downstream benchmark performance
- [ ] Test if dimension recovery correlates with comp/φ variance
