#!/usr/bin/env python3
"""
Model Merge Example

This example demonstrates how to merge two models using geometric alignment.
The geometric merge uses Procrustes analysis and spectral penalties to
preserve important representations from both models.

Usage:
    python examples/05_model_merge.py base_model model_a model_b -o merged_output

Requirements:
    - Compatible model weights (same architecture)
    - Sufficient disk space for output
    - macOS with Apple Silicon for MLX backend
"""
import argparse
from pathlib import Path

from modelcypher.core.use_cases.model_merge_service import (
    ModelMergeService,
    GeometricMergeConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Merge models using geometric alignment"
    )
    parser.add_argument(
        "base",
        help="Path to base model (provides architecture)",
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Paths to models to merge",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend alpha (0=first model, 1=second model). Default: 0.5",
    )
    parser.add_argument(
        "--method",
        choices=["linear", "geometric"],
        default="geometric",
        help="Merge method. Default: geometric",
    )
    args = parser.parse_args()

    # Validate paths
    base_path = Path(args.base)
    if not base_path.exists():
        print(f"Error: Base model not found: {base_path}")
        return 1

    model_paths = [Path(p) for p in args.models]
    for path in model_paths:
        if not path.exists():
            print(f"Error: Model not found: {path}")
            return 1

    print("Model Merge")
    print("=" * 60)
    print(f"Base model: {base_path}")
    print(f"Models to merge: {len(model_paths)}")
    for p in model_paths:
        print(f"  - {p}")
    print(f"Method: {args.method}")
    print(f"Alpha: {args.alpha}")
    print(f"Output: {args.output}")
    print()

    # Initialize service
    service = ModelMergeService()

    if args.method == "geometric":
        # Configure geometric merge
        config = GeometricMergeConfig(
            alpha=args.alpha,
            gaussian_sigma=2.0,  # Smooth alpha across layers
            spectral_penalty=0.01,  # Penalize ill-conditioned weights
            svd_blend=True,  # Use SVD-aware blending
        )

        print("Running geometric merge...")
        print("  This uses Procrustes alignment and spectral regularization.")
        result = service.geometric_merge(
            base_model=str(base_path),
            models=[str(p) for p in model_paths],
            output=args.output,
            config=config,
        )
    else:
        # Linear interpolation (simpler, faster)
        print("Running linear merge...")
        result = service.linear_merge(
            base_model=str(base_path),
            models=[str(p) for p in model_paths],
            output=args.output,
            alpha=args.alpha,
        )

    print("\nMerge complete!")
    print("-" * 40)
    print(f"Output path: {result.output_path}")
    print(f"Total parameters: {result.parameter_count:,}")
    print(f"Layers merged: {result.layer_count}")

    if hasattr(result, "alignment_score"):
        print(f"Alignment score: {result.alignment_score:.4f}")
    if hasattr(result, "spectral_condition"):
        print(f"Spectral condition: {result.spectral_condition:.4f}")

    print("\nNext steps:")
    print(f"  1. Test the merged model: mc model probe {args.output}")
    print(f"  2. Run inference: mc infer run {args.output} --prompt 'Hello'")


if __name__ == "__main__":
    exit(main() or 0)
