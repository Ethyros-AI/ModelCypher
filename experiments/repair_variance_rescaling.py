#!/usr/bin/env python3
"""REPAIR-style variance rescaling for merged models.

Based on Jordan et al. 2023 "REPAIR: REnormalizing Permuted Activations
for Interpolation Repair"

The idea: After merging, measure activation variance at each layer and
rescale weights so merged model matches target variance.

Key insight from our diagnostic:
- Target layer 11 creates variance explosion: 20K -> 128K
- Merged layer 11 stays normal: 24K -> 1.9K
- The MLP weights are ~86% similar but don't create same extreme outliers
- Solution: Rescale the MLP output to match target variance
"""

import json
import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.adapters.mlx_model_loader import MLXModelLoader

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def collect_layer_variances(model, input_ids: mx.array) -> dict[int, float]:
    """Collect hidden state variance after each transformer layer."""
    variances = {}

    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    if len(input_ids.shape) == 1:
        input_ids = input_ids.reshape(1, -1)

    # Embed
    if hasattr(base, "embed_tokens"):
        h = base.embed_tokens(input_ids)
    else:
        h = base.wte(input_ids)
    mx.eval(h)
    variances[-1] = float(mx.var(h.reshape(-1)))

    layers = base.layers if hasattr(base, "layers") else base.h

    seq_len = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(layers):
        try:
            result = layer(h, mask=mask, cache=None)
        except TypeError:
            try:
                result = layer(h, mask, None)
            except TypeError:
                result = layer(h, mask)

        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)
        variances[i] = float(mx.var(h.reshape(-1)))

    return variances


def compute_rescale_factors(target_vars: dict, merged_vars: dict) -> dict[int, float]:
    """Compute variance rescaling factors per layer."""
    factors = {}
    for layer_idx in sorted(target_vars.keys()):
        if layer_idx not in merged_vars:
            continue
        tgt_var = target_vars[layer_idx]
        mrg_var = merged_vars[layer_idx]
        if mrg_var > 0:
            # Scale factor to match target variance
            # merged * factor should have target variance
            # factor = sqrt(target_var / merged_var)
            factor = (tgt_var / mrg_var) ** 0.5
        else:
            factor = 1.0
        factors[layer_idx] = factor
    return factors


def apply_mlp_rescaling(model, rescale_factors: dict[int, float]) -> None:
    """Apply rescaling factors to MLP down_proj weights."""

    if hasattr(model, "model"):
        base = model.model
    else:
        base = model

    layers = base.layers if hasattr(base, "layers") else base.h

    for layer_idx, factor in rescale_factors.items():
        if layer_idx < 0:
            continue  # Skip embedding layer
        if layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]
        if not hasattr(layer, "mlp"):
            continue

        mlp = layer.mlp
        if hasattr(mlp, "down_proj"):
            # Scale down_proj weight by factor
            # This scales the MLP output, affecting layer output variance
            old_weight = mlp.down_proj.weight
            new_weight = old_weight * factor
            mx.eval(new_weight)
            mlp.down_proj.weight = new_weight
            logger.info(f"Layer {layer_idx}: Rescaled down_proj by {factor:.4f}")


def repair_merged_model(
    target_path: str,
    merged_path: str,
    output_path: str,
    test_prompt: str = "The capital of France is Paris, which is known for",
) -> dict:
    """Apply REPAIR-style variance rescaling to merged model."""

    loader = MLXModelLoader()

    logger.info("=" * 80)
    logger.info("REPAIR VARIANCE RESCALING")
    logger.info("=" * 80)

    # Load target and collect variances
    logger.info(f"Loading target: {target_path}")
    target_model, target_tok = loader.load_model_for_training(target_path)
    input_ids = mx.array(target_tok.encode(test_prompt))

    logger.info("Collecting target variances...")
    target_vars = collect_layer_variances(target_model, input_ids)

    del target_model
    mx.eval([])

    # Load merged and collect variances
    logger.info(f"Loading merged: {merged_path}")
    merged_model, merged_tok = loader.load_model_for_training(merged_path)

    logger.info("Collecting merged variances...")
    merged_vars = collect_layer_variances(merged_model, input_ids)

    # Compute rescaling factors
    logger.info("\nComputing rescaling factors...")
    factors = compute_rescale_factors(target_vars, merged_vars)

    # Log factors
    logger.info("\nRescale factors per layer:")
    for layer_idx in sorted(factors.keys()):
        factor = factors[layer_idx]
        tgt_var = target_vars.get(layer_idx, 0)
        mrg_var = merged_vars.get(layer_idx, 0)
        layer_name = "Emb" if layer_idx == -1 else f"L{layer_idx:02d}"
        logger.info(
            f"  {layer_name}: tgt_var={tgt_var:12.2e}, mrg_var={mrg_var:12.2e}, "
            f"factor={factor:8.4f}"
        )

    # Apply rescaling (only to the first layer with major variance mismatch)
    # This is the ROOT CAUSE - fixing it should cascade correctly
    logger.info("\nApplying rescaling to problematic layers...")
    # Find first layer with major mismatch (factor > 2 or < 0.5)
    first_problem_layer = None
    for k in sorted(factors.keys()):
        if k >= 0 and (factors[k] > 2.0 or factors[k] < 0.5):
            first_problem_layer = k
            break

    if first_problem_layer is not None:
        layers_to_fix = {first_problem_layer: factors[first_problem_layer]}
    else:
        layers_to_fix = {}
    logger.info(f"Layers needing rescaling: {sorted(layers_to_fix.keys())}")

    apply_mlp_rescaling(merged_model, layers_to_fix)

    # Verify the fix
    logger.info("\nCollecting post-repair variances...")
    repaired_vars = collect_layer_variances(merged_model, input_ids)

    logger.info("\nPost-repair comparison:")
    for layer_idx in sorted(target_vars.keys()):
        if layer_idx not in repaired_vars:
            continue
        tgt_var = target_vars[layer_idx]
        rep_var = repaired_vars[layer_idx]
        ratio = rep_var / (tgt_var + 1e-12)
        layer_name = "Emb" if layer_idx == -1 else f"L{layer_idx:02d}"
        status = "✓" if 0.5 < ratio < 2.0 else "✗"
        logger.info(f"  {layer_name}: target={tgt_var:12.2e}, repaired={rep_var:12.2e}, ratio={ratio:.4f} {status}")

    # Test generation with repaired model
    logger.info(f"\nTesting generation with repaired model...")
    from mlx_lm import generate

    test_output = generate(
        merged_model,
        merged_tok,
        prompt="The capital of France is",
        max_tokens=30,
        verbose=False,
    )
    logger.info(f"Repaired model output: {test_output}")

    # Also save if output path provided
    if output_path and output_path != "SKIP":
        logger.info(f"\nSaving repaired model to {output_path}")
        import shutil

        merged_dir = Path(merged_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                            "special_tokens_map.json", "vocab.json", "merges.txt"]:
            src = merged_dir / config_file
            if src.exists():
                shutil.copy(src, output_dir / config_file)

        # Save weights using mlx.save_safetensors
        weights = {}
        for k, v in merged_model.parameters().items():
            if hasattr(v, 'shape'):
                weights[k] = v
        mx.save_safetensors(str(output_dir / "model.safetensors"), weights)
        logger.info(f"Model saved to {output_path}")

    results = {
        "target_vars": {str(k): v for k, v in target_vars.items()},
        "merged_vars": {str(k): v for k, v in merged_vars.items()},
        "repaired_vars": {str(k): v for k, v in repaired_vars.items()},
        "rescale_factors": {str(k): v for k, v in factors.items()},
        "layers_rescaled": list(layers_to_fix.keys()),
    }

    logger.info("\nDone!")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="The capital of France is Paris, which is known for")

    args = parser.parse_args()

    results = repair_merged_model(args.target, args.merged, args.output, args.prompt)

    # Save results
    results_path = Path(args.output) / "repair_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
