# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Falsifier verdict renderer for LKM rank vs optimization experiment.

Reads fact_geometry.json from each of the 4 arms and determines which
hypothesis (rank bottleneck, optimization ceiling, interference, confound)
best explains the ~11% failure rate.

Usage:
    poetry run python scripts/lkm/evaluate_falsifier.py \\
        --results-dir results/lora_memory_capacity_validation/Qwen3.5-0.8B-bf16/falsifier/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Arm naming convention
ARM_NAMES = {
    "B0-r4-s1500": {"rank": 4, "steps": 1500},
    "B0-r4-s4500": {"rank": 4, "steps": 4500},
    "B0-r16-s1500": {"rank": 16, "steps": 1500},
    "B0-r16-s4500": {"rank": 16, "steps": 4500},
}

# 5% absolute EM gain threshold.
# Derived from: 11% failure ceiling × 50% = meaningful change.
EM_GAIN_THRESHOLD = 0.05

# Interference threshold for "high interference" pair classification.
# A pair with cosine > 0.5 means the two facts' projected gradients share
# more than half their direction in the LoRA subspace.
HIGH_INTERFERENCE_THRESHOLD = 0.5


def load_arm_results(results_dir: str) -> dict[str, dict]:
    """Load fact_geometry.json and raw_scores.jsonl from each arm directory.

    Args:
        results_dir: Base directory containing arm subdirectories.

    Returns:
        Dict mapping arm name to {"em": float, "geometry": dict}.
    """
    base = Path(results_dir)
    arms = {}

    for arm_name in ARM_NAMES:
        arm_dir = base / arm_name
        geom_path = arm_dir / "fact_geometry.json"
        scores_path = arm_dir / "raw_scores.jsonl"

        if not geom_path.exists():
            raise FileNotFoundError(f"Missing {geom_path}")

        with open(geom_path) as f:
            geometry = json.load(f)

        # Compute EM from raw_scores if available, else from geometry
        if scores_path.exists():
            with open(scores_path) as f:
                scores = [json.loads(line) for line in f if line.strip()]
            em = sum(1 for s in scores if s["exact_match"]) / len(scores)
        else:
            # Fall back to counting from retained_fractions
            rf_items = geometry["retained_fractions"]
            n_match = sum(1 for r in rf_items if r.get("exact_match") is True)
            n_total = sum(
                1 for r in rf_items if r.get("exact_match") is not None
            )
            em = n_match / n_total if n_total > 0 else 0.0

        arms[arm_name] = {
            "em": em,
            "geometry": geometry,
        }

    return arms


def compute_verdict(arms: dict[str, dict]) -> dict:
    """Evaluate hypotheses from arm results.

    Args:
        arms: Dict mapping arm name to {"em": float, "geometry": dict}.

    Returns:
        Verdict dict with arms summary, hypothesis tests, and verdict string.
    """
    # Extract key quantities
    em_r4_s1500 = arms["B0-r4-s1500"]["em"]
    em_r4_s4500 = arms["B0-r4-s4500"]["em"]
    em_r16_s1500 = arms["B0-r16-s1500"]["em"]
    delta_em_rank = em_r16_s1500 - em_r4_s1500
    delta_em_opt = em_r4_s4500 - em_r4_s1500

    # Geometry summaries
    geom_r4_s1500 = arms["B0-r4-s1500"]["geometry"]["summary"]
    geom_r16_s1500 = arms["B0-r16-s1500"]["geometry"]["summary"]

    mean_rf_failed_r4 = geom_r4_s1500.get("mean_rf_failed")
    mean_rf_passed_r4 = geom_r4_s1500.get("mean_rf_passed")
    mean_rf_failed_r16 = geom_r16_s1500.get("mean_rf_failed")
    mean_rf_passed_r16 = geom_r16_s1500.get("mean_rf_passed")

    # RF separation: how much do failed facts differ from passed in RF?
    rf_sep_r4 = (
        (mean_rf_passed_r4 - mean_rf_failed_r4)
        if mean_rf_failed_r4 is not None and mean_rf_passed_r4 is not None
        else None
    )
    rf_sep_r16 = (
        (mean_rf_passed_r16 - mean_rf_failed_r16)
        if mean_rf_failed_r16 is not None and mean_rf_passed_r16 is not None
        else None
    )

    # Interference at r4
    mean_interference_r4 = geom_r4_s1500.get("mean_interference", 0.0)
    n_high_r4 = geom_r4_s1500.get("n_high_interference_pairs", 0)

    # Build arm summary
    arm_summary = {}
    for name, data in arms.items():
        s = data["geometry"]["summary"]
        arm_summary[name] = {
            "em": round(data["em"], 4),
            "mean_rf": s.get("mean_rf"),
            "mean_rf_failed": s.get("mean_rf_failed"),
            "mean_rf_passed": s.get("mean_rf_passed"),
            "mean_interference": s.get("mean_interference"),
        }

    rank_gain_large = delta_em_rank > EM_GAIN_THRESHOLD
    opt_gain_large = delta_em_opt > EM_GAIN_THRESHOLD

    # Hypothesis tests
    h_rank = {
        "rank_em_gain": round(delta_em_rank, 4),
        "opt_em_gain": round(delta_em_opt, 4),
        "rf_separation_r4": round(rf_sep_r4, 4) if rf_sep_r4 is not None else None,
        "rf_separation_r16": round(rf_sep_r16, 4) if rf_sep_r16 is not None else None,
        "pass": rank_gain_large and not opt_gain_large and rf_sep_r4 is not None and rf_sep_r4 > 0.05,
    }

    h_opt = {
        "rank_em_gain": round(delta_em_rank, 4),
        "opt_em_gain": round(delta_em_opt, 4),
        "pass": opt_gain_large and not rank_gain_large,
    }

    h_intrf = {
        "rank_em_gain": round(delta_em_rank, 4),
        "opt_em_gain": round(delta_em_opt, 4),
        "mean_interference_r4": round(mean_interference_r4, 4),
        "n_high_interference_pairs_r4": n_high_r4,
        "pass": rank_gain_large and n_high_r4 > 0,
    }

    h_confound = {
        "rank_em_gain": round(delta_em_rank, 4),
        "opt_em_gain": round(delta_em_opt, 4),
        "rf_separation_r4": round(rf_sep_r4, 4) if rf_sep_r4 is not None else None,
        "pass": (
            rank_gain_large
            and opt_gain_large
            and (rf_sep_r4 is None or rf_sep_r4 < 0.05)
        ),
    }

    # Verdict
    if h_rank["pass"] and not h_intrf["pass"]:
        verdict = "RANK_BOTTLENECK"
    elif h_intrf["pass"]:
        verdict = "INTERFERENCE_CLUSTERING"
    elif h_opt["pass"]:
        verdict = "OPTIMIZATION_CEILING"
    elif h_confound["pass"]:
        verdict = "CONFOUND_RANK_HELPS_OPT"
    elif rank_gain_large and opt_gain_large:
        verdict = "MIXED"
    elif not rank_gain_large and not opt_gain_large:
        verdict = "NEITHER"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "arms": arm_summary,
        "hypothesis_tests": {
            "H-RANK": h_rank,
            "H-OPT": h_opt,
            "H-INTRF": h_intrf,
            "H-CONFOUND": h_confound,
        },
        "verdict": verdict,
    }


def main() -> None:
    """CLI entry point for falsifier verdict rendering."""
    parser = argparse.ArgumentParser(
        description="Evaluate LKM falsifier arms and render verdict."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Base directory containing arm subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for verdict JSON (default: results_dir/falsifier_verdict.json).",
    )

    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.results_dir) / "falsifier_verdict.json"
    )

    print(f"Loading results from: {args.results_dir}")
    arm_results = load_arm_results(args.results_dir)

    for name, data in arm_results.items():
        print(f"  {name}: EM={data['em']:.4f}")

    verdict = compute_verdict(arm_results)

    print(f"\nVerdict: {verdict['verdict']}")
    print(f"  ΔEM_rank = {verdict['hypothesis_tests']['H-RANK']['rank_em_gain']:.4f}")
    print(f"  ΔEM_opt  = {verdict['hypothesis_tests']['H-RANK']['opt_em_gain']:.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nWrote verdict to: {output_path}")


if __name__ == "__main__":
    main()
