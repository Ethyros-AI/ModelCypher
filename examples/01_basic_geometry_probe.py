#!/usr/bin/env python3
"""
Basic Geometry Probe Example

This example demonstrates how to probe a model and inspect its geometric properties.
Geometric probing reveals how a model represents concepts internally.

Usage:
    python examples/01_basic_geometry_probe.py /path/to/model

Requirements:
    - A local MLX-compatible model (e.g., from mlx-community on Hugging Face)
    - macOS with Apple Silicon for MLX backend
"""
import sys
from pathlib import Path

from modelcypher.core.use_cases.model_probe_service import ModelProbeService


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_basic_geometry_probe.py /path/to/model")
        print("\nExample models:")
        print("  /Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-bf16")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    print(f"Probing model: {model_path}")
    print("-" * 60)

    # Initialize the probe service
    service = ModelProbeService()

    # Run a basic probe to get model structure
    result = service.probe(str(model_path))

    # Display results
    print(f"Model: {result.model_id}")
    print(f"Architecture: {result.architecture}")
    print(f"Parameter count: {result.parameter_count:,}")
    print(f"Hidden size: {result.hidden_size}")
    print(f"Number of layers: {result.num_layers}")
    print(f"Vocabulary size: {result.vocab_size}")

    print("\nLayer breakdown:")
    for layer in result.layers[:5]:  # Show first 5 layers
        print(f"  {layer.name}: {layer.parameter_count:,} params")

    if len(result.layers) > 5:
        print(f"  ... and {len(result.layers) - 5} more layers")

    print("\nGeometric metrics:")
    if hasattr(result, "geometric_metrics") and result.geometric_metrics:
        metrics = result.geometric_metrics
        print(f"  Intrinsic dimension: {metrics.get('intrinsic_dimension', 'N/A')}")
        print(f"  Effective rank: {metrics.get('effective_rank', 'N/A')}")


if __name__ == "__main__":
    main()
