# Challenge Card: QLoRA

## Assumption Challenged
Full-precision optimizer state is required for useful adapter training at scale.

## Measurable Prediction (ModelCypher Terms)
For matched datasets and bounded decode settings, 4-bit base + LoRA should reduce:
- `train_probe.train_step_peak_gb`
- `memory_stages[*].active_gb`
while maintaining comparable geometric outcome metrics (e.g., CKA/retention probes).

## Minimum Falsifiable Experiment
1. Profile bf16 baseline on the same Qwen checkpoint.
2. Profile 4-bit QLoRA variant with identical prompt/train-probe settings.
3. Compare memory deltas and geometric quality deltas.
4. Falsify if memory savings fail or quality degradation exceeds baseline drift envelope.

## Integration Effort
Medium.

## Source
- [QLoRA (arXiv:2305.14314)](https://arxiv.org/abs/2305.14314)

