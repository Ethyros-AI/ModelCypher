# Challenge Card: DoRA

## Assumption Challenged
Low-rank adaptation should treat weight magnitude and direction as a single coupled update.

## Measurable Prediction (ModelCypher Terms)
DoRA-style decomposition should yield:
- improved behavior-per-parameter transfer efficiency
- better `behavioral_norm` retention at equal trainable parameter count
- lower interference on target manifold under null-space projection

## Minimum Falsifiable Experiment
1. Match rank/parameter budget between LoRA and DoRA variants.
2. Run identical data and bounded decode protocol.
3. Compare geometric transfer metrics and behavioral retention.
4. Falsify if decomposition fails to improve transfer efficiency.

## Integration Effort
Medium.

## Source
- [DoRA (arXiv:2402.09353)](https://arxiv.org/abs/2402.09353)

