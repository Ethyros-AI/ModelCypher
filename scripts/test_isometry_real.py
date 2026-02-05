#!/usr/bin/env python3
"""Test isometry metrics on real LoRA adapter.

Tests the experimental isometry metrics on the michael_kl_v12 adapter
against the Qwen3-8B base model.

Usage:
    poetry run python scripts/test_isometry_real.py
"""

from __future__ import annotations

import json
from pathlib import Path

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.adapters.adapter_weights_loader import AutoAdapterWeightsLoader
from modelcypher.experimental.lora_isometry import (
    IsometryMetrics,
    compute_isometry_metrics,
)


def main():
    # Initialize backend first
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    # Paths
    adapter_path = Path("/Volumes/codecypher/adapters/ensemble-characters/michael_kl_v12")
    model_path = Path("/Volumes/codecypher/models/mlx-community/Qwen3-8B-bf16")

    print(f"Testing isometry metrics")
    print(f"  Adapter: {adapter_path}")
    print(f"  Model:   {model_path}")
    print()

    # Load backend
    backend = get_default_backend()
    print(f"Backend: {type(backend).__name__}")

    # Load adapter weights
    loader = AutoAdapterWeightsLoader()
    adapter_file = adapter_path / "adapters.safetensors"
    adapter_weights = loader.load(adapter_file, backend)
    backend.eval(*adapter_weights.values())
    print(f"Loaded {len(adapter_weights)} adapter weight tensors")

    # Load sharded model index
    from safetensors import safe_open

    index_file = model_path / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    print(f"Loaded weight index with {len(weight_map)} weights across shards")

    def load_base_weight(key: str):
        """Load a single weight from the correct shard."""
        import mlx.core as mx
        shard_name = weight_map.get(key)
        if not shard_name:
            return None
        shard_path = model_path / shard_name
        # Use mlx.core.load for native bfloat16 support
        weights = mx.load(str(shard_path))
        return weights.get(key)

    # Test on sample layers
    results: list[dict] = []
    test_layers = [0, 1, 2, 17, 35]  # First, middle, last
    test_projections = ["q_proj", "v_proj"]

    for layer_idx in test_layers:
        for proj in test_projections:
            # Adapter weight keys (format: model.layers.X.self_attn.proj.lora_a/b)
            lora_a_key = f"model.layers.{layer_idx}.self_attn.{proj}.lora_a"
            lora_b_key = f"model.layers.{layer_idx}.self_attn.{proj}.lora_b"

            # Base weight key
            base_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"

            if lora_a_key not in adapter_weights:
                continue

            # Load weights
            lora_a = adapter_weights[lora_a_key]
            lora_b = adapter_weights[lora_b_key]
            base_w = load_base_weight(base_key)

            if base_w is None:
                print(f"Skipping {base_key} - not found in model")
                continue

            # Compute delta: ΔW = A @ B for MLX format
            # MLX stores: lora_a as (in_features, rank), lora_b as (rank, out_features)
            delta_w = backend.matmul(lora_a, lora_b)

            # Match delta shape to base weight (some layers may be transposed)
            if delta_w.shape != base_w.shape:
                if delta_w.shape == (base_w.shape[1], base_w.shape[0]):
                    delta_w = backend.transpose(delta_w)
                else:
                    print(f"Skipping layer {layer_idx} {proj}: shape mismatch delta={delta_w.shape} base={base_w.shape}")
                    continue

            # Compute W' = W + ΔW
            w_modified = backend.add(base_w, delta_w)
            backend.eval(delta_w, base_w, w_modified)

            # Cast to float32 for SVD (bfloat16 not supported)
            import mlx.core as mx
            base_w_f32 = base_w.astype(mx.float32)
            w_mod_f32 = w_modified.astype(mx.float32)

            # Compute isometry metrics
            metrics = compute_isometry_metrics(base_w_f32, w_mod_f32, backend)

            result = {
                "layer": layer_idx,
                "projection": proj,
                "spr": round(metrics.spectral_preservation_ratio, 4),
                "angle": round(metrics.spectral_angle, 2),
                "cond_ratio": round(metrics.condition_ratio, 4),
                "rfd": round(metrics.relative_frobenius_deviation, 4),
                "rank_orig": metrics.rank_original,
                "rank_mod": metrics.rank_modified,
            }
            results.append(result)

            print(
                f"Layer {layer_idx:2d} {proj:6s}: "
                f"angle={metrics.spectral_angle:5.1f}° "
                f"SPR={metrics.spectral_preservation_ratio:.3f} "
                f"cond_ratio={metrics.condition_ratio:.3f} "
                f"RFD={metrics.relative_frobenius_deviation:.4f}"
            )

    # Summary statistics
    if results:
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        avg_angle = sum(r["angle"] for r in results) / len(results)
        avg_spr = sum(r["spr"] for r in results) / len(results)
        avg_cond = sum(r["cond_ratio"] for r in results) / len(results)
        min_angle = min(r["angle"] for r in results)
        max_angle = max(r["angle"] for r in results)

        print(f"  Layers tested: {len(results)}")
        print(f"  Avg Spectral Angle: {avg_angle:.2f}°")
        print(f"  Avg SPR: {avg_spr:.4f}")
        print(f"  Avg Condition Ratio: {avg_cond:.4f}")
        print(f"  Angle range: [{min_angle:.2f}°, {max_angle:.2f}°]")

        # Interpretation
        print()
        if avg_angle < 5.0:
            print("✓ Adapter preserves geometry well (angle < 5°)")
        elif avg_angle < 30.0:
            print("⚠ Moderate geometric drift (5° < angle < 30°)")
        else:
            print("✗ Significant geometric deviation (angle > 30°)")
    else:
        print("No layers tested!")


if __name__ == "__main__":
    main()
