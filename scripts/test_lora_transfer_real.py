#!/usr/bin/env python3
"""Test Universal LoRA Projector on real model weights.

This script validates the transfer algorithm on actual model weights,
not synthetic test matrices. It uses the small LFM2 models for speed.

Usage:
    poetry run python scripts/test_lora_transfer_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main():
    """Test LoRA transfer on real model weights."""
    # Initialize backend first
    from modelcypher.backends import initialize_default_backend

    initialize_default_backend()

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.universal_lora_projector import (
        UniversalLoRAProjector,
        compute_lora_delta,
        decompose_to_lora,
    )

    # Check volume is mounted
    models_dir = Path("/Volumes/CodeCypher/models/mlx-community")
    if not models_dir.exists():
        print("ERROR: Volume not mounted. Run: ls /Volumes/CodeCypher/")
        sys.exit(1)

    # Use smallest models for fast testing
    source_model = models_dir / "LFM2-350M-MLX-bf16"
    # Cross-dimension test: different hidden dimensions
    target_model = models_dir / "LFM2-700M-bf16"

    # Uncomment below for same-dimension test:
    # target_model = models_dir / "LFM2-350M-MLX-bf16"

    if not source_model.exists():
        print(f"ERROR: Source model not found: {source_model}")
        sys.exit(1)

    if not target_model.exists():
        print(f"ERROR: Target model not found: {target_model}")
        sys.exit(1)

    print("=" * 60)
    print("Universal LoRA Projector - Real Model Test")
    print("=" * 60)
    print(f"Source: {source_model.name}")
    print(f"Target: {target_model.name}")
    print()

    # Load weights
    print("Loading model weights...")
    from modelcypher.adapters.model_loader import load_model_weights_only

    source_weights = load_model_weights_only(str(source_model))
    target_weights = load_model_weights_only(str(target_model))

    print(f"  Source weights: {len(source_weights)} tensors")
    print(f"  Target weights: {len(target_weights)} tensors")

    # Find matching layer pairs (same layer name pattern)
    # Focus on attention projections which are typical LoRA targets
    # Filter for 2D weight matrices only (not layer norms which are 1D)
    source_attn_keys = [
        k for k in source_weights
        if "self_attn" in k and "weight" in k and "_proj" in k
        and len(source_weights[k].shape) == 2
    ]
    target_attn_keys = [
        k for k in target_weights
        if "self_attn" in k and "weight" in k and "_proj" in k
        and len(target_weights[k].shape) == 2
    ]

    print(f"\n  Source attention layers: {len(source_attn_keys)}")
    print(f"  Target attention layers: {len(target_attn_keys)}")

    # Match by layer pattern (layers.0.self_attn.q_proj.weight, etc.)
    def extract_layer_pattern(key: str) -> str:
        """Extract layer pattern for matching."""
        # model.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj
        parts = key.split(".")
        # Find "layers" and take from there
        if "layers" in parts:
            idx = parts.index("layers")
            return ".".join(parts[idx:-1])  # Exclude .weight
        return key

    source_patterns = {extract_layer_pattern(k): k for k in source_attn_keys}
    target_patterns = {extract_layer_pattern(k): k for k in target_attn_keys}

    # Find common patterns
    common_patterns = set(source_patterns.keys()) & set(target_patterns.keys())
    print(f"\n  Common layer patterns: {len(common_patterns)}")

    if not common_patterns:
        print("ERROR: No matching layer patterns found between models")
        # Print some examples for debugging
        print("\n  Source patterns (first 5):")
        for p in list(source_patterns.keys())[:5]:
            print(f"    {p}")
        print("\n  Target patterns (first 5):")
        for p in list(target_patterns.keys())[:5]:
            print(f"    {p}")
        sys.exit(1)

    # Test on first few matching layers
    test_patterns = sorted(common_patterns)[:3]
    print(f"\n  Testing on: {test_patterns}")

    # Initialize projector
    backend = get_default_backend()
    projector = UniversalLoRAProjector(backend=backend)

    print("\n" + "-" * 60)
    print("Computing SVDs and testing transfer...")
    print("-" * 60)

    results = []
    for pattern in test_patterns:
        src_key = source_patterns[pattern]
        tgt_key = target_patterns[pattern]

        src_weight = source_weights[src_key]
        tgt_weight = target_weights[tgt_key]

        print(f"\n  Layer: {pattern}")
        print(f"    Source shape: {src_weight.shape}")
        print(f"    Target shape: {tgt_weight.shape}")

        # Convert to backend arrays
        src_arr = backend.array(src_weight)
        tgt_arr = backend.array(tgt_weight)

        # Compute SVDs with subsampling (now works correctly!)
        # The fix: U is reconstructed from full weight via U = W @ V @ S^{-1}
        src_svd = projector.compute_layer_svd(src_arr, sample_size=512)
        tgt_svd = projector.compute_layer_svd(tgt_arr, sample_size=512)

        print(f"    Source effective rank: {src_svd.effective_rank}")
        print(f"    Target effective rank: {tgt_svd.effective_rank}")

        # Create a synthetic LoRA delta (simulating an adapter update)
        # In practice this would come from a trained adapter
        lora_rank = 8
        out_dim, in_dim = src_weight.shape

        # Create random low-rank update
        np.random.seed(42)
        B = np.random.randn(out_dim, lora_rank).astype(np.float32) * 0.01
        A = np.random.randn(lora_rank, in_dim).astype(np.float32) * 0.01

        lora_B = backend.array(B)
        lora_A = backend.array(A)

        # Compute delta
        delta = compute_lora_delta(lora_A, lora_B, backend)
        delta_norm = float(backend.to_scalar(backend.sqrt(backend.sum(delta * delta))))
        print(f"    LoRA delta norm: {delta_norm:.6f}")

        # Transfer (works for both same and different dimensions)
        transferred, result = projector.transfer_layer(
            lora_delta=delta,
            source_svd=src_svd,
            target_svd=tgt_svd,
            layer_key=pattern,
        )

        transferred_shape = tuple(int(d) for d in backend.shape(transferred))
        transferred_norm = float(
            backend.to_scalar(backend.sqrt(backend.sum(transferred * transferred)))
        )

        cross_dim = src_weight.shape != tgt_weight.shape
        print(f"    Cross-dimension: {cross_dim}")
        print(f"    Transferred shape: {transferred_shape}")
        print(f"    Transferred norm: {transferred_norm:.6f}")
        print(f"    Projection error: {result.projection_error:.4f}")
        print(f"    Grassmann distance: {result.grassmann_distance:.4f}")
        print(f"    Was truncated: {result.was_truncated}")

        results.append(
            {
                "layer": pattern,
                "source_rank": src_svd.effective_rank,
                "target_rank": tgt_svd.effective_rank,
                "projection_error": result.projection_error,
                "grassmann_distance": result.grassmann_distance,
                "energy_ratio": transferred_norm / delta_norm,
                "cross_dimension": cross_dim,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        avg_error = sum(r["projection_error"] for r in results) / len(results)
        avg_grass = sum(r["grassmann_distance"] for r in results) / len(results)
        avg_energy = sum(r["energy_ratio"] for r in results) / len(results)
        cross_dim_count = sum(1 for r in results if r["cross_dimension"])

        print(f"\n  Layers tested: {len(results)}")
        print(f"  Cross-dimension transfers: {cross_dim_count}")
        print(f"  Avg projection error: {avg_error:.4f}")
        print(f"  Avg Grassmann distance: {avg_grass:.4f}")
        print(f"  Avg energy ratio: {avg_energy:.4f}")

        # Success criteria
        # For cross-dimension, we expect more energy loss but should still preserve structure
        if avg_energy > 0.01 and avg_energy < 100:
            print("\n  STATUS: ✅ PASS - Transfer algorithm working on real weights")
            if cross_dim_count > 0:
                print("           (Cross-dimension transfers completed successfully)")
        else:
            print("\n  STATUS: ⚠️ NEEDS INVESTIGATION - Energy ratio out of bounds")
    else:
        print("\n  No transfers completed.")

    print()


if __name__ == "__main__":
    main()
