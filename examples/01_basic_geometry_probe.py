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
Basic Geometry Probe Example

This example demonstrates how to probe a model directory and inspect basic
architecture + weight metadata.

Usage:
    poetry run python examples/01_basic_geometry_probe.py /path/to/model

Requirements:
    - A local model directory containing `config.json` and one or more
      `*.safetensors` files.
"""
import sys
from pathlib import Path

from modelcypher.core.use_cases.model_probe_service import ModelProbeService


def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run python examples/01_basic_geometry_probe.py /path/to/model")
        print("\nTip:")
        print("  Fetch a model: poetry run mc model fetch <repo_id>")
        print("  Then probe the returned localPath.")
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
    print(f"Model: {model_path.name}")
    print(f"Architecture: {result.architecture}")
    print(f"Parameter count: {result.parameter_count:,}")
    print(f"Hidden size: {result.hidden_size}")
    print(f"Number of layers: {len(result.layers)}")
    print(f"Vocabulary size: {result.vocab_size}")
    print(f"Attention heads: {result.num_attention_heads}")
    if result.quantization:
        print(f"Quantization: {result.quantization}")

    print("\nLayer breakdown:")
    for layer in result.layers[:5]:  # Show first 5 layers
        print(f"  {layer.name}: {layer.parameters:,} params")

    if len(result.layers) > 5:
        print(f"  ... and {len(result.layers) - 5} more layers")

    print("\nGeometric metrics: N/A (use geometry CLI or probes for metrics)")


if __name__ == "__main__":
    main()
