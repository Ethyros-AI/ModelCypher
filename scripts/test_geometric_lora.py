#!/usr/bin/env python3
"""Test geometric LoRA configuration derivation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import initialize_default_backend
initialize_default_backend()


def main():
    from mlx_lm import load

    from modelcypher.core.domain.training.geometric_lora import (
        analyze_model_geometry,
        compute_geometric_rank,
        select_target_modules,
    )

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    print("\nAnalyzing geometry (first 3 layers only for speed)...")

    # For testing, only analyze first few layers
    base_model = getattr(model, "model", model)
    original_layers = base_model.layers
    base_model.layers = original_layers[:3]

    geometries = analyze_model_geometry(model)

    # Restore
    base_model.layers = original_layers

    print(f"\nAnalyzed {len(geometries)} layers")
    print("=" * 70)

    # Show results
    print(f"\n{'Layer':<45} {'Decay':>10} {'σ_k':>10} {'Tail':>8} {'Target':>8}")
    print("-" * 70)

    for key, geom in sorted(geometries.items()):
        target = "YES" if geom.is_targetable else "no"
        print(f"{key:<45} {geom.decay_ratio:>10.1f} {geom.sigma_k:>10.4f} {geom.tail_dims:>8} {target:>8}")

    # Derive config
    targets = select_target_modules(geometries)
    print(f"\nTargetable modules: {len(targets)}")

    if targets:
        rank = compute_geometric_rank(geometries, targets)
        print(f"Geometric rank: {rank}")

        # Show σ_k for targets
        print("\nTarget σ_k values (these are the per-layer scales):")
        for t in targets:
            print(f"  {t}: σ_k = {geometries[t].sigma_k:.4f}")


if __name__ == "__main__":
    main()
