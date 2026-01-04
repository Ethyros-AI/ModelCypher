#!/usr/bin/env python3

# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""
Entropy Analysis Example

This example demonstrates how to measure entropy across linguistic modifiers
and compute raw thermodynamic metrics for a prompt.

Usage:
    python examples/04_entropy_analysis.py /path/to/model --prompt "Explain quantum computing"

Requirements:
    - A local MLX-compatible model
    - macOS with Apple Silicon
"""
import argparse
from pathlib import Path

from modelcypher.core.use_cases.thermo_service import ThermoService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure entropy across linguistic modifiers"
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
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    print("Entropy Analysis (Raw Metrics)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Prompt: {args.prompt[:50]}...")
    print()

    # Initialize service
    service = ThermoService()

    # Run thermodynamic measurement
    print("Running entropy measurements...")
    result = service.measure(
        prompt=args.prompt,
        model_path=str(model_path),
    )

    print("\nMeasurements:")
    print("-" * 40)
    for measurement in result.measurements:
        delta_h = "baseline" if measurement.delta_h is None else f"{measurement.delta_h:.6f}"
        print(
            f"{measurement.modifier:>10}: entropy={measurement.mean_entropy:.6f}, "
            f"delta_h={delta_h}, ridge_crossed={measurement.ridge_crossed}"
        )

    stats = result.statistics
    print("\nStatistics:")
    print("-" * 40)
    print(f"Mean entropy: {stats.mean_entropy:.6f}")
    print(f"Std entropy: {stats.std_entropy:.6f}")
    print(f"Min entropy: {stats.min_entropy:.6f}")
    print(f"Max entropy: {stats.max_entropy:.6f}")
    if stats.mean_delta_h is not None:
        print(f"Mean delta_h: {stats.mean_delta_h:.6f}")

    # Differential measurement (baseline vs intensity)
    detect_result = service.detect(
        prompt=args.prompt,
        model_path=str(model_path),
    )
    print("\nDifferential measurement:")
    print("-" * 40)
    print(f"Baseline entropy: {detect_result.baseline_entropy:.6f}")
    print(f"Intensity entropy: {detect_result.intensity_entropy:.6f}")
    print(f"Delta H: {detect_result.delta_h:.6f}")
    print(f"Processing time: {detect_result.processing_time:.3f}s")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
