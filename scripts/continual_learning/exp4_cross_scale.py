#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

import argparse
import json
from pathlib import Path

from modelcypher.backends import initialize_default_backend
from modelcypher.cli.composition import get_capacity_analysis_service

OUTPUT_ROOT_DEFAULT = Path("results/continual_learning/exp4")

# Using paths as stubs to show the orchestration.
MODELS_TO_TEST = [
    {"name": "350M", "path": "/Volumes/CodeCypher/models/mlx-community/Qwen3-350M-bf16"},
    {"name": "1.2B", "path": "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.2B-bf16"}
]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 4: Cross-Scale Validation")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    return parser.parse_args()

def run_scale_experiment(model_info: dict, capacity_service) -> dict:
    """Run capacity geometry analysis for a specific scale."""
    print(f"Running geometric capacity telemetry for {model_info['name']}...")
    
    try:
        report = capacity_service.analyze(model_path=model_info["path"])
        hidden_dim = report.layer_reports[0].weight_shape[0] if report.layer_reports else 1024

        # Use pre-computed mean from the report
        mean_capacity = report.mean_capacity_utilization
    except Exception as e:
        print(f"Failed to analyze true model, using fallback. Error: {e}")
        hidden_dim = 1024 if model_info["name"] == "350M" else 2048
        mean_capacity = None  # Explicitly null — no real measurement available

    return {
        "scale": model_info["name"],
        "hidden_dim": hidden_dim,
        "mean_capacity_utilization": mean_capacity,
        "normalized_depletion_rate": mean_capacity,  # null when simulation mode
        "simulation_mode": mean_capacity is None,
    }

def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    initialize_default_backend()
    capacity_service = get_capacity_analysis_service()
    
    print(f"=== Starting Experiment 4 (Cross-Scale Validation) ===")
    
    results = []
    for model in MODELS_TO_TEST:
        try:
            res = run_scale_experiment(model, capacity_service)
            results.append(res)
        except Exception as e:
            print(f"Failed to run scale {model['name']}: {e}")
            
    if len(results) == 2:
        ratio_350m = results[0]["normalized_depletion_rate"]
        ratio_1_2b = results[1]["normalized_depletion_rate"]
        
        if ratio_350m is not None and ratio_1_2b is not None and min(ratio_350m, ratio_1_2b) > 0:
            invariance_factor = max(ratio_350m, ratio_1_2b) / min(ratio_350m, ratio_1_2b)
        else:
            invariance_factor = None
        
        # H4 judgment is deferred: statistical significance requires variance
        # across seeds.  We record the invariance factor for downstream analysis.
        summary = {
            "scale_results": results,
            "invariance_factor": invariance_factor,
            "h4_judgment": "deferred — requires multi-seed variance for significance",
        }
    else:
        summary = {"scale_results": results, "error": "Could not compare scales."}
        
    out_file = output_root / "exp4_cross_scale_results.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
