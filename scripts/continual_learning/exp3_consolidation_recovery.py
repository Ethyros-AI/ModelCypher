#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

import argparse
import json
from pathlib import Path

from modelcypher.backends import initialize_default_backend
from modelcypher.cli.composition import get_capacity_analysis_service, get_model_loader

MODEL_PATH_DEFAULT = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
OUTPUT_ROOT_DEFAULT = Path("results/continual_learning/exp3")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 3: Consolidation Recovery")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--adapter-path", required=True, help="Path to saturated adapter to consolidate")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    seed_dir = output_root / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    backend = initialize_default_backend()
    capacity_service = get_capacity_analysis_service()
    
    print(f"=== Starting Experiment 3 (Consolidation Recovery) | Seed: {args.seed} ===")
    
    # 1. Measure Pre-Consolidation Capacity
    print(f"Measuring capacity of base model pre-merge: {args.model_path}")
    pre_report = capacity_service.analyze(model_path=str(args.model_path))
    pre_merge_capacity = (
        sum(r.null_space_dim_f32 for r in pre_report.layer_metrics.values()) 
        / max(1, len(pre_report.layer_metrics)) if pre_report.layer_metrics else 0
    )
    
    # 2. Consolidate (Merge) Adapter into Base Weights
    print(f"Consolidating adapter {args.adapter_path} into {args.model_path} ...")
    merged_model_path = seed_dir / "merged_model"
    
    loader = get_model_loader()
    loader.merge_adapter_to_base(
        base_model_path=str(args.model_path),
        adapter_path=str(args.adapter_path),
        output_path=str(merged_model_path)
    )
    
    # 3. Measure Post-Consolidation Capacity
    print(f"Measuring capacity of consolidated model: {merged_model_path}")
    post_report = capacity_service.analyze(model_path=str(merged_model_path))
    post_merge_capacity = (
        sum(r.null_space_dim_f32 for r in post_report.layer_metrics.values()) 
        / max(1, len(post_report.layer_metrics)) if post_report.layer_metrics else 0
    )
    
    # H3 Threshold Validation
    # We use min_dim as the proxy for max rank (e.g. 1024)
    first_layer_dim = list(post_report.layer_metrics.values())[0].weight_shape[0] if post_report.layer_metrics else 1024
    recovery_ratio = post_merge_capacity / float(first_layer_dim)
    
    output = {
        "seed": args.seed,
        "target_model": args.model_path,
        "adapter_path": args.adapter_path,
        "pre_merge_capacity_rank": pre_merge_capacity,
        "post_merge_capacity_rank": post_merge_capacity,
        "recovery_ratio": recovery_ratio,
        "h3_passed": recovery_ratio > 0.0 # Testing strictly > 0 per the updated H3 document
    }
    
    out_file = seed_dir / "exp3_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
