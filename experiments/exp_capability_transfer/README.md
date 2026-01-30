# Capability Transfer Validation Experiment

## Purpose

Validates whether null-space merging actually transfers capabilities from a source model to a target model without destroying what the target already knows.

## Protocol

1. **Benchmark source model** - Measure coding capability (source is a coding model)
2. **Benchmark target model** - Measure general reasoning capability
3. **Execute null-space merge** - Transfer source capabilities to target
4. **Benchmark merged model** - Measure BOTH capabilities

## Success Criteria

- **Capability transferred**: `merged_code_score > target_code_score`
- **Capability preserved**: `merged_reasoning_score >= 0.95 × target_reasoning_score`

## Usage

```bash
# Full experiment (uses default models)
python experiments/exp_capability_transfer/run_experiment.py

# Quick smoke test
python experiments/exp_capability_transfer/run_experiment.py --quick

# Custom models
python experiments/exp_capability_transfer/run_experiment.py \
  --source /path/to/coding/model \
  --target /path/to/general/model \
  --output /path/to/merged/model

# Skip merge (evaluate existing merged model)
python experiments/exp_capability_transfer/run_experiment.py --skip-merge
```

## Default Models

- **Source**: Qwen2.5-Coder-0.5B-Instruct (coding specialist)
- **Target**: LFM2-350M (general language model)

## Output

Results are saved to `data/experiments/capability_transfer_result.json`:

```json
{
  "timestamp": "2026-01-30T...",
  "capability_transferred": true,
  "capability_preserved": true,
  "experiment_success": true,
  "code_transfer_delta": 0.15,
  "reasoning_preservation": 0.97,
  ...
}
```

## Benchmarks Used

### Code Prompts (10 samples)
- Simple function completion (sum, is_even, etc.)
- List comprehension completion
- Basic Python syntax

### Reasoning Prompts (10 samples)
- Factual questions (capital of France, etc.)
- Simple logic (if-then reasoning)
- Basic arithmetic

## Interpretation

| Result | Meaning |
|--------|---------|
| Both pass | Null-space merge successfully transfers capability |
| Transfer fails | Source capability did not transfer (check layer matching) |
| Preservation fails | Target capability was damaged (check null-space projection) |
| Both fail | Merge produced non-functional model |
