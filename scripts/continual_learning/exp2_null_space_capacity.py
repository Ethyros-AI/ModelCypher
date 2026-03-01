#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

import argparse
import json
from pathlib import Path

from modelcypher.backends import initialize_default_backend
from modelcypher.cli.composition import get_capacity_analysis_service

MODEL_PATH_DEFAULT = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
OUTPUT_ROOT_DEFAULT = Path("results/continual_learning/exp2")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 2: Null-Space Capacity Dynamics")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
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
    
    print(f"=== Starting Experiment 2 (Null-Space Capacity) | Seed: {args.seed} ===")
    
    checkpoint_path = seed_dir / "capacity_checkpoint.json"
    
    # 1. Profile Base Model Capacity
    capacity_report = capacity_service.analyze(
        model_path=str(args.model_path),
        checkpoint_path=checkpoint_path,
        resume=False,
    )
    
    # Exposing the rank dictionary to verify SOTA hypotheses
    mean_rank = (
        sum(report.null_space_dim_f32 for report in capacity_report.layer_metrics.values()) 
        / max(1, len(capacity_report.layer_metrics)) 
        if capacity_report.layer_metrics else 0.0
    )
    
    saturation_summary = {
        "analyzed_layers": capacity_report.analyzed_layers,
        "analyzed_parameters": capacity_report.analyzed_parameters,
        "mean_null_rank": mean_rank,
        "layers": {k: v.to_dict() for k, v in capacity_report.layer_metrics.items()}
    }
    
    output = {
        "seed": args.seed,
        "model_path": args.model_path,
        "capacity_report": saturation_summary
    }
    
    out_file = seed_dir / "exp2_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
