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
Adapter Inspection Example

This example demonstrates how to inspect one or more LoRA adapters to
report geometric properties (rank, sparsity) without blending.

Usage:
    python examples/03_adapter_inspection.py /path/to/adapter_dir
    python examples/03_adapter_inspection.py /path/to/a /path/to/b --json

Requirements:
    - One or more adapter directories containing safetensors weights
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelcypher.core.use_cases.adapter_service import AdapterService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect LoRA adapters (geometry-first, no blending)"
    )
    parser.add_argument(
        "adapters",
        nargs="+",
        help="Paths to adapter directories",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output",
    )
    args = parser.parse_args()

    service = AdapterService()
    results = []

    for adapter_path in args.adapters:
        path = Path(adapter_path)
        if not path.exists():
            print(f"Error: Adapter not found: {path}")
            return 1

        info = service.inspect(str(path))
        results.append((path, info))

    if args.json:
        payload = [
            {
                "path": str(path),
                "rank": info.rank,
                "alpha": info.alpha,
                "sparsity": info.sparsity,
                "parameterCount": info.parameter_count,
                "layerCount": len(info.layer_analysis),
                "targetModules": info.target_modules,
            }
            for path, info in results
        ]
        print(json.dumps(payload, indent=2))
        return 0

    print("Adapter Inspection (Geometry-First)")
    print("=" * 60)
    for path, info in results:
        print(f"\n{path.name}")
        print(f"  Path: {path}")
        print(f"  Rank: {info.rank}")
        print(f"  Alpha: {info.alpha}")
        print(f"  Sparsity: {info.sparsity:.2%}")
        print(f"  Parameters: {info.parameter_count:,}")
        print(f"  Layers: {len(info.layer_analysis)}")
        if info.target_modules:
            print(f"  Target Modules: {', '.join(info.target_modules)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
