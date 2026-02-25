# Challenge Card: rsLoRA

## Assumption Challenged
LoRA scaling can stay fixed (`alpha/r`) across rank without destabilizing geometry.

## Measurable Prediction (ModelCypher Terms)
Using rsLoRA scaling (`alpha/sqrt(r)`) should reduce rank-induced instability:
- lower variance in `spectral_bounds_ok` outcomes across ranks
- smoother rank-to-quality curves in geometric retention metrics

## Minimum Falsifiable Experiment
1. Choose one Qwen base model and fixed data slice.
2. Sweep ranks with classic LoRA scaling and rsLoRA scaling.
3. Compare spectral diagnostics and geometric retention consistency.
4. Falsify if rsLoRA does not improve stability under the same rank sweep.

## Integration Effort
Low to Medium.

## Source
- [rsLoRA (arXiv:2312.03732)](https://arxiv.org/abs/2312.03732)

