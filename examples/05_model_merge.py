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
The merge uses Gram alignment and null-space constrained transplant to preserve
target boundaries while grafting dense source regions.

Pipeline: PROBE → DENSITY → TRANSPLANT

Usage:
    poetry run python examples/05_model_merge.py source_model target_model -o merged_output --dry-run

To run an actual merge, omit `--dry-run` (requires a working GPU backend + real
model weights).

Requirements:
    - Two model directories with weights
    - Sufficient disk space for output
    - A working GPU backend for the full merge pipeline
"""
import argparse
from pathlib import Path

from modelcypher.cli.composition import get_merge_pipeline_service
from modelcypher.core.use_cases.model_probe_service import ModelProbeService


def main() -> int:
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
        help="Validate paths and merge compatibility without running the merge pipeline",
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
    print()
    print("Pipeline: PROBE → DENSITY → TRANSPLANT")
    print()

    if args.dry_run:
        print("Dry run: validating model compatibility (no model loading/inference).")
        print()
        probe = ModelProbeService()
        try:
            source_probe = probe.probe(str(source_path))
            target_probe = probe.probe(str(target_path))
            validation = probe.validate_merge(str(source_path), str(target_path))
        except Exception as exc:
            print(f"Validation failed: {exc}")
            return 1

        print("Model details:")
        print(f"  Source architecture: {source_probe.architecture}")
        print(f"  Target architecture: {target_probe.architecture}")
        print(f"  Source hidden size: {source_probe.hidden_size}")
        print(f"  Target hidden size: {target_probe.hidden_size}")
        print(f"  Source vocab size: {source_probe.vocab_size}")
        print(f"  Target vocab size: {target_probe.vocab_size}")
        print()
        print("Merge compatibility:")
        print(f"  Low-effort: {validation.low_effort}")
        print(f"  Architecture match: {validation.architecture_match}")
        print(f"  Vocab match: {validation.vocab_match}")
        print(f"  Dimension match: {validation.dimension_match}")
        if validation.warnings:
            print("  Warnings:")
            for warning in validation.warnings:
                print(f"    - {warning}")
        print("\nDry run complete.")
        return 0

    service = get_merge_pipeline_service()

    print("Running geometric merge pipeline...")
    print("  Using exact CKA alignment and null-space transplant.")
    print()

    try:
        result = service.run(
            source_path=str(source_path),
            target_path=str(target_path),
            output_dir=args.output,
        )
    except RuntimeError as exc:
        print(f"Merge failed: {exc}")
        return 1

    print("\nMerge complete!")
    print("-" * 40)
    print(f"Pipeline ID: {result.pipeline_id}")
    print(f"Output directory: {result.output_dir}")

    pre = result.pre_merge
    print("\nPre-merge metrics:")
    print(f"  Mean overlap: {pre.mean_overlap:.6f}")
    print(f"  Mean subspace alignment: {pre.mean_subspace_alignment:.6f}")
    print(f"  Mean curvature divergence: {pre.mean_curvature_divergence:.6f}")
    print(f"  Mean distance: {pre.mean_distance:.6f}")
    print(f"  Aligned pairs: {pre.aligned_pairs}")

    merge_result = result.merge_result
    print("\nMerge results:")
    print(f"  Layers merged: {merge_result.get('layer_count', 0)}")
    print(f"  Weights merged: {merge_result.get('weight_count', 0)}")
    print(f"  Mean preserved fraction: {merge_result.get('mean_preserved_fraction', 0.0):.6f}")
    print(f"  Mean Procrustes error: {merge_result.get('mean_procrustes_error', 0.0):.6f}")

    post = result.post_merge
    print("\nPost-merge metrics:")
    print(f"  Layers transplanted: {post.layers_transplanted}")
    print(f"  Weights transplanted: {post.weights_transplanted}")
    print(f"  Mean preserved fraction: {post.mean_preserved_fraction:.6f}")
    print(f"  Mean CKA after: {post.mean_cka_after:.6f}")

    output_path = merge_result.get("output_path") or result.output_dir
    print(f"\nOutput path: {output_path}")
    print("\nNext steps:")
    print(f"  1. Test the merged model: poetry run mc model probe {output_path}")
    print(f"  2. Run inference: poetry run mc infer run --model {output_path} --prompt 'Hello'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
