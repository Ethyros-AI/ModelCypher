#!/usr/bin/env python3
"""
Entropy Analysis Example

This example demonstrates how to analyze entropy patterns in model outputs.
Entropy metrics can reveal training dynamics, model confidence, and potential issues.

Usage:
    python examples/04_entropy_analysis.py /path/to/model --prompt "Explain quantum computing"

Requirements:
    - A local MLX-compatible model
    - macOS with Apple Silicon
"""
import argparse
from pathlib import Path

from modelcypher.core.use_cases.thermo_service import ThermoService


def main():
    parser = argparse.ArgumentParser(
        description="Analyze entropy patterns in model outputs"
    )
    parser.add_argument(
        "model",
        help="Path to local model directory",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the concept of entropy in information theory.",
        help="Prompt to analyze",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    print("Entropy Analysis")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Temperature: {args.temperature}")
    print()

    # Initialize service
    service = ThermoService()

    # Run thermodynamic measurement
    print("Running thermodynamic analysis...")
    result = service.measure(
        model_path=str(model_path),
        prompt=args.prompt,
        temperature=args.temperature,
    )

    print("\nResults:")
    print("-" * 40)
    print(f"Entropy: {result.entropy:.4f}")
    print(f"Temperature: {result.temperature:.4f}")
    print(f"Free Energy: {result.free_energy:.4f}")
    print(f"\nInterpretation:")
    print(f"  {result.interpretation}")

    # Additional analysis: ridge detection
    print("\n" + "-" * 40)
    print("Phase transition analysis...")
    try:
        ridge_result = service.detect_ridge(
            model_path=str(model_path),
            prompt=args.prompt,
        )
        print(f"Ridge detected: {ridge_result.ridge_detected}")
        if ridge_result.ridge_detected:
            print(f"Ridge location: {ridge_result.ridge_position}")
            print(f"Transition type: {ridge_result.transition_type}")
    except Exception as e:
        print(f"Ridge detection skipped: {e}")

    print("\n" + "=" * 60)
    print("Analysis complete.")


if __name__ == "__main__":
    exit(main() or 0)
