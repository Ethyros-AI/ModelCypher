# Challenge Card: GaLore / Q-GaLore / Lotus

## Assumption Challenged
Optimizer-state and gradient memory must scale linearly with full parameter dimensionality.

## Measurable Prediction (ModelCypher Terms)
Low-rank gradient projection families should reduce step memory while preserving useful update geometry:
- lower `train_probe.train_step_peak_gb`
- comparable or improved geometric progress per wall-clock step
- improved feasible parameter ceiling under fixed unified memory

## Minimum Falsifiable Experiment
1. Establish one-step and short-run memory baselines under current training path.
2. Integrate one low-rank gradient method behind a controlled flag.
3. Compare memory, throughput, and geometric quality metrics.
4. Falsify if memory reduction does not translate to equivalent geometric progress.

## Integration Effort
High.

## Sources
- [GaLore (arXiv:2403.03507)](https://arxiv.org/abs/2403.03507)
- [Q-GaLore (arXiv:2407.08296)](https://arxiv.org/abs/2407.08296)
- [Lotus (arXiv:2602.01233)](https://arxiv.org/abs/2602.01233)

