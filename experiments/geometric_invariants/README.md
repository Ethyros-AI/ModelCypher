# Geometric Invariants Experiment

## Hypothesis

All LLMs trained on language encode the same invariant geometric shape. Different models are different projections/compressions of this universal geometry.

## Measurements

1. **Intrinsic Dimension**: True dimensionality of the representation manifold
2. **Sectional Curvature**: How geodesics converge/diverge (spherical vs hyperbolic)
3. **CKA Alignment**: Coordinate system alignment between models (should be 1.0)

## Models

### Small (fast experiments)
- LFM2-350M-MLX-bf16
- Qwen2.5-Coder-0.5B-Instruct-bf16

### Medium (1-3B)
- Qwen2.5-Math-1.5B-bf16
- Qwen3-1.7B-MLX-bf16
- Qwen2.5-3B-Instruct-bf16
- granite-3b-code-instruct-128k-mlx

### Large (7-8B)
- Qwen3-8B-bf16
- Qwen2.5-Coder-7B-Instruct-bf16
- granite-8b-code-instruct-128k-mlx
- BioMistral-7B (medical domain)
- Saul-7B-Instruct-v1 (legal domain)

## Experiment Design

### Experiment 1: Intra-family scaling
Do larger models have the same geometric shape as smaller ones?
- Compare Qwen2.5-Coder-0.5B vs 3B vs 7B
- Measure: ID, curvature, inter-model CKA

### Experiment 2: Cross-family comparison
Do different architectures encode the same shape?
- Compare Qwen3 vs Granite vs BioMistral at similar sizes
- Measure: ID, curvature, cross-family CKA

### Experiment 3: Domain specialization
Does domain training change the geometric shape?
- Compare BioMistral (medical) vs Saul (legal) vs Qwen (general)
- Measure: per-domain activation geometry

## Output

Results saved to `results/` as JSON with full reproducibility metadata.
Scientific report generated in `reports/`.
