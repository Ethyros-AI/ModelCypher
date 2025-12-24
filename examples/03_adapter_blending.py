#!/usr/bin/env python3
"""
Adapter Blending Example

This example demonstrates how to blend two or more LoRA adapters
into a single adapter using geometric interpolation.

Usage:
    python examples/03_adapter_blending.py adapter1.safetensors adapter2.safetensors -o blended.safetensors

Requirements:
    - Two or more compatible LoRA adapters (same base model, same rank)
"""
import argparse
from pathlib import Path

from modelcypher.core.use_cases.adapter_service import AdapterService


def main():
    parser = argparse.ArgumentParser(
        description="Blend multiple LoRA adapters into one"
    )
    parser.add_argument(
        "adapters",
        nargs="+",
        help="Paths to adapter files (.safetensors)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for blended adapter",
    )
    parser.add_argument(
        "-w", "--weights",
        help="Comma-separated blend weights (e.g., '0.6,0.4'). Defaults to equal weights.",
    )
    args = parser.parse_args()

    # Validate inputs
    adapter_paths = [Path(p) for p in args.adapters]
    for path in adapter_paths:
        if not path.exists():
            print(f"Error: Adapter not found: {path}")
            return 1

    # Parse weights
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(adapter_paths):
            print(f"Error: Number of weights ({len(weights)}) must match adapters ({len(adapter_paths)})")
            return 1
    else:
        weights = [1.0 / len(adapter_paths)] * len(adapter_paths)

    print("Adapter Blending")
    print("=" * 60)
    print("\nInput adapters:")
    for path, weight in zip(adapter_paths, weights):
        print(f"  {path.name}: weight={weight:.2f}")

    # Initialize service
    service = AdapterService()

    # First, inspect the adapters
    print("\nInspecting adapters...")
    for path in adapter_paths:
        info = service.inspect(str(path))
        print(f"  {path.name}: rank={info.rank}, alpha={info.alpha}, layers={info.layer_count}")

    # Blend the adapters
    print(f"\nBlending with weights: {weights}")
    result = service.blend(
        adapters=[str(p) for p in adapter_paths],
        weights=weights,
        output=args.output,
    )

    print("\nBlend complete!")
    print(f"  Output: {result.output_path}")
    print(f"  Combined rank: {result.rank}")
    print(f"  Total parameters: {result.parameter_count:,}")

    # Verify the output
    output_info = service.inspect(result.output_path)
    print(f"\nVerification:")
    print(f"  Output file valid: {Path(result.output_path).exists()}")
    print(f"  Layer count: {output_info.layer_count}")


if __name__ == "__main__":
    exit(main() or 0)
