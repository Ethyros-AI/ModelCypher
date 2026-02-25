# Challenge Card: PiSSA + EVA Initialization

## Assumption Challenged
Random LoRA initialization is sufficient; data/spectrum-aware initialization adds little.

## Measurable Prediction (ModelCypher Terms)
PiSSA/EVA initialization should improve early-step geometry:
- better `preserved_fraction` after projection steps
- faster approach to stable spectral diagnostics
- lower iterations to reach equivalent validation geometry

## Minimum Falsifiable Experiment
1. Hold model, dataset, rank, and optimizer settings fixed.
2. Compare random init vs PiSSA vs EVA init.
3. Measure first-N-step geometric trajectories and final retention.
4. Falsify if initialization-aware methods show no measurable early-geometry advantage.

## Integration Effort
Medium.

## Sources
- [PiSSA (arXiv:2404.02948)](https://arxiv.org/abs/2404.02948)
- [EVA (arXiv:2410.07170)](https://arxiv.org/abs/2410.07170)

