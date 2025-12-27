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
Model Merge Example

This example demonstrates how to merge two models using geometric alignment.
The geometric merge uses Procrustes analysis to align representations and
Fréchet mean to blend singular values along the geodesic.

Pipeline: VOCAB → PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE → VALIDATE

Usage:
    python examples/05_model_merge.py source_model target_model -o merged_output

Requirements:
    - Two model directories with weights
    - Sufficient disk space for output
    - macOS with Apple Silicon for MLX backend
"""
import argparse
from pathlib import Path

from modelcypher.cli.composition import get_geometric_merger
from modelcypher.core.use_cases.unified_geometric_merge import UnifiedMergeConfig


def main():
    parser = argparse.ArgumentParser(
        description="Merge models using geometric alignment"
    )
    parser.add_argument(
        "source",
        help="Path to source model (capabilities to transfer)",
    )
    parser.add_argument(
        "target",
        help="Path to target model (base architecture)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without saving (shows what would happen)",
    )
    parser.add_argument(
        "--probe-mode",
        choices=["precise", "fast"],
        default="precise",
        help="Probe mode: precise (CKA on activations) or fast (weight-level). Default: precise",
    )
    parser.add_argument(
        "--transport-guided",
        action="store_true",
        help="Use Gromov-Wasserstein instead of Procrustes for alignment",
    )
    args = parser.parse_args()

    # Validate paths
    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists():
        print(f"Error: Source model not found: {source_path}")
        return 1

    if not target_path.exists():
        print(f"Error: Target model not found: {target_path}")
        return 1

    print("Model Merge - Pure Geometry")
    print("=" * 60)
    print(f"Source model: {source_path}")
    print(f"Target model: {target_path}")
    print(f"Output: {args.output}")
    print(f"Probe mode: {args.probe_mode}")
    print(f"Transport-guided: {args.transport_guided}")
    print(f"Dry run: {args.dry_run}")
    print()
    print("Pipeline: VOCAB → PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE → VALIDATE")
    print()

    # Initialize merger with dependency injection
    merger = get_geometric_merger()

    # Configure merge (optional - defaults are sensible)
    merger.config = UnifiedMergeConfig(
        probe_mode=args.probe_mode,
        use_transport_guided=args.transport_guided,
    )

    print("Running geometric merge...")
    print("  Using Procrustes alignment and Fréchet mean blending.")
    print()

    result = merger.merge(
        source_path=str(source_path),
        target_path=str(target_path),
        output_dir=args.output if not args.dry_run else None,
        dry_run=args.dry_run,
    )

    print("\nMerge complete!")
    print("-" * 40)
    print(f"Layers merged: {result.layer_count}")
    print(f"Weights merged: {result.weight_count}")
    print(f"Mean confidence: {result.mean_confidence:.4f}")
    print(f"Mean Procrustes error: {result.mean_procrustes_error:.6f}")

    if result.vocab_aligned:
        print(f"Vocabulary aligned: Yes")
    if result.validation_metrics:
        print(f"Safety verdict: {result.safety_verdict}")
        print(f"Refusal preserved: {result.refusal_preserved}")

    if result.output_path:
        print(f"\nOutput path: {result.output_path}")
        print("\nNext steps:")
        print(f"  1. Test the merged model: mc model probe {result.output_path}")
        print(f"  2. Run inference: mc infer run {result.output_path} --prompt 'Hello'")
    elif args.dry_run:
        print("\n[Dry run - no output saved]")


if __name__ == "__main__":
    exit(main() or 0)
