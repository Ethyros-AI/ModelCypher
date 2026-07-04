#!/usr/bin/env python3
"""Validate NB-LoRA per-direction spectral bounds on real model weights.

This script tests the mathematical framework:
1. Load LFM2-350M model weights
2. Create NB-LoRA layers with geometry-derived bounds
3. Verify per-direction bounds: |[U^T @ Δ @ V]_ii| / σ_i(W) for each direction
4. Compare with what standard LoRA scaling would produce

The key insight: the bound is relational (per-direction ratio), not a fixed number.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LayerValidationResult:
    """Result of validating NB-LoRA bounds on one layer."""
    layer_name: str
    sigma_k: float
    scale_bound: float  # max(S) = σ_k / 2
    max_delta_spectral: float  # 2 * scale_bound
    actual_delta_spectral: float
    per_direction_max_ratio: float
    per_direction_violations: int
    is_safe: bool


def load_model_weights(model_path: str, backend):
    """Load model weights from MLX format."""
    from pathlib import Path

    model_dir = Path(model_path)

    # Use mlx.core to load safetensors (handles bfloat16)
    import mlx.core as mx

    weights = {}
    for sf_file in model_dir.glob("*.safetensors"):
        loaded = mx.load(str(sf_file))
        for k, v in loaded.items():
            # Convert to float32 for computation
            weights[k] = backend.astype(backend.array(v), "float32")
            backend.eval(weights[k])

    logger.info(f"Loaded {len(weights)} weight tensors")
    return weights


def validate_layer(
    W,
    layer_name: str,
    rank: int,
    backend,
) -> LayerValidationResult:
    """Validate NB-LoRA bounds on a single layer."""
    from modelcypher.core.domain.geometry.cayley_lora import (
        create_nb_lora_from_base_weight,
    )

    # Create NB-LoRA layer with geometry-derived bounds
    layer = create_nb_lora_from_base_weight(W, rank=rank, backend=backend)

    actual_spectral = layer.get_spectral_norm()

    # Verify per-direction bounds
    result = layer.verify_per_direction_bounds(W)

    # Extract sigma_k from the layer config
    # scale_bound = (σ_k / 2) * 0.9 → σ_k = scale_bound * 2 / 0.9
    sigma_k = layer.scale_bound * 2.0 / 0.9

    return LayerValidationResult(
        layer_name=layer_name,
        sigma_k=sigma_k,
        scale_bound=layer.scale_bound,
        max_delta_spectral=2.0 * layer.scale_bound,
        actual_delta_spectral=actual_spectral,
        per_direction_max_ratio=result.max_ratio,
        per_direction_violations=len(result.violations),
        is_safe=result.is_safe,
    )


def main():
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    rank = 8

    # Check volume is mounted
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Is the external volume mounted?")
        return 1

    logger.info("=" * 70)
    logger.info("NB-LoRA Per-Direction Bound Validation")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Rank: {rank}")
    logger.info("")

    # Initialize and get backend
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()
    from modelcypher.core.domain._backend import get_default_backend
    backend = get_default_backend()
    logger.info(f"Backend: {backend.__class__.__name__}")

    # Load weights
    logger.info("Loading model weights...")
    weights = load_model_weights(model_path, backend)

    # Find projection layers to validate
    proj_layers = [k for k in weights.keys() if any(p in k for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]
    proj_layers = sorted(proj_layers)[:12]  # First 12 projection layers

    logger.info(f"Found {len(proj_layers)} projection layers to validate")
    logger.info("")

    # Validate each layer
    results = []
    for layer_name in proj_layers:
        W = weights[layer_name]
        if len(W.shape) != 2:
            continue

        logger.info(f"Validating {layer_name} [{W.shape}]...")

        try:
            result = validate_layer(W, layer_name, rank, backend)
            results.append(result)
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info("")

    # Summary table
    logger.info(f"{'Layer':<35} {'σ_k':>8} {'max(S)':>8} {'||Δ||':>8} {'max_ratio':>10} {'safe':>6}")
    logger.info("-" * 80)

    safe_count = 0
    for r in results:
        safe_str = "✓" if r.is_safe else f"✗({r.per_direction_violations})"
        logger.info(
            f"{r.layer_name:<35} {r.sigma_k:>8.4f} {r.scale_bound:>8.4f} "
            f"{r.actual_delta_spectral:>8.4f} {r.per_direction_max_ratio:>10.4f} {safe_str:>6}"
        )
        if r.is_safe:
            safe_count += 1

    logger.info("-" * 80)
    logger.info(f"Safe layers: {safe_count}/{len(results)}")
    logger.info("")

    # Analysis
    logger.info("ANALYSIS")
    logger.info("-" * 70)

    if results:
        avg_sigma_k = sum(r.sigma_k for r in results) / len(results)
        avg_max_ratio = sum(r.per_direction_max_ratio for r in results) / len(results)
        max_max_ratio = max(r.per_direction_max_ratio for r in results)

        logger.info(f"Average σ_k: {avg_sigma_k:.4f}")
        logger.info(f"Average per-direction max ratio: {avg_max_ratio:.4f}")
        logger.info(f"Maximum per-direction ratio: {max_max_ratio:.4f}")
        logger.info("")

        # Compare with standard LoRA
        logger.info("COMPARISON WITH STANDARD LORA")
        logger.info("-" * 70)
        standard_alpha = 16
        standard_rank = 8
        standard_scale = standard_alpha / standard_rank  # = 2.0

        logger.info(f"Standard LoRA scale (alpha={standard_alpha}, rank={standard_rank}): {standard_scale}")
        logger.info("")

        # If standard LoRA used scale=2.0 on ||B@A||≈1.0, the delta would be ~2.0
        # Compare to our geometry-derived bounds
        for r in results[:3]:  # First 3 layers
            # What would standard LoRA do?
            # Delta_standard = scale × ||B@A|| ≈ 2.0 × 1.0 = 2.0
            # Our bound says delta should be ≤ σ_k
            ratio_over = standard_scale / r.sigma_k if r.sigma_k > 0 else float('inf')
            logger.info(f"{r.layer_name}:")
            logger.info(f"  σ_k = {r.sigma_k:.4f}")
            logger.info(f"  Standard LoRA scale (2.0) would be {ratio_over:.1f}× over geometric bound")

    return 0


if __name__ == "__main__":
    sys.exit(main())
