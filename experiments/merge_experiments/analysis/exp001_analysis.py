#!/usr/bin/env python3
"""
Analysis script for EXP001: Reasoning Transfer via Geometric Merge.

Compares baseline benchmarks with merged model results.
"""

import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_results(
    target_baseline: dict,
    source_baseline: dict,
    merged: dict | None = None,
) -> dict:
    """Compare benchmark results between models."""
    comparison = {
        "target_model": target_baseline.get("model_path", "unknown"),
        "source_model": source_baseline.get("model_path", "unknown"),
        "benchmarks": {},
    }

    target_benchmarks = target_baseline.get("benchmarks", {})
    source_benchmarks = source_baseline.get("benchmarks", {})
    merged_benchmarks = merged.get("benchmarks", {}) if merged else {}

    all_benchmarks = set(target_benchmarks.keys()) | set(source_benchmarks.keys())

    for name in sorted(all_benchmarks):
        target_acc = target_benchmarks.get(name, {}).get("accuracy", 0)
        source_acc = source_benchmarks.get(name, {}).get("accuracy", 0)

        entry = {
            "target_baseline": target_acc,
            "source_baseline": source_acc,
            "source_vs_target": source_acc - target_acc,
        }

        if merged and name in merged_benchmarks:
            merged_acc = merged_benchmarks[name].get("accuracy", 0)
            entry["merged"] = merged_acc
            entry["improvement_vs_target"] = merged_acc - target_acc
            entry["improvement_pct"] = (
                100 * (merged_acc - target_acc) / target_acc if target_acc > 0 else 0
            )
            # How much of the gap did we close?
            gap = source_acc - target_acc
            if gap > 0:
                entry["gap_closed_pct"] = 100 * (merged_acc - target_acc) / gap
            else:
                entry["gap_closed_pct"] = None

        comparison["benchmarks"][name] = entry

    # Speed comparison
    target_speed = target_baseline.get("inference_speed", {}).get("mean_tokens_per_sec", 0)
    source_speed = source_baseline.get("inference_speed", {}).get("mean_tokens_per_sec", 0)

    comparison["speed"] = {
        "target_baseline": target_speed,
        "source_baseline": source_speed,
    }

    if merged:
        merged_speed = merged.get("inference_speed", {}).get("mean_tokens_per_sec", 0)
        comparison["speed"]["merged"] = merged_speed
        comparison["speed"]["speed_retention_pct"] = (
            100 * merged_speed / target_speed if target_speed > 0 else 0
        )

    return comparison


def print_report(comparison: dict) -> None:
    """Print formatted comparison report."""
    print("=" * 70)
    print("EXP001: REASONING TRANSFER ANALYSIS")
    print("=" * 70)
    print()
    print(f"Target (base): {Path(comparison['target_model']).name}")
    print(f"Source (donor): {Path(comparison['source_model']).name}")
    print()

    print("-" * 70)
    print("BENCHMARK COMPARISON")
    print("-" * 70)
    print(f"{'Benchmark':<15} {'Target':>10} {'Source':>10} {'Gap':>10}", end="")

    has_merged = any("merged" in b for b in comparison["benchmarks"].values())
    if has_merged:
        print(f" {'Merged':>10} {'Improv':>10} {'Gap%':>10}")
    else:
        print()

    for name, data in comparison["benchmarks"].items():
        target = f"{data['target_baseline']*100:.1f}%"
        source = f"{data['source_baseline']*100:.1f}%"
        gap = f"{data['source_vs_target']*100:+.1f}%"

        print(f"{name:<15} {target:>10} {source:>10} {gap:>10}", end="")

        if "merged" in data:
            merged = f"{data['merged']*100:.1f}%"
            improv = f"{data['improvement_vs_target']*100:+.1f}%"
            gap_closed = data.get("gap_closed_pct")
            gap_str = f"{gap_closed:.0f}%" if gap_closed is not None else "N/A"
            print(f" {merged:>10} {improv:>10} {gap_str:>10}")
        else:
            print()

    print()
    print("-" * 70)
    print("INFERENCE SPEED")
    print("-" * 70)
    speed = comparison["speed"]
    print(f"Target baseline: {speed['target_baseline']:.1f} tokens/sec")
    print(f"Source baseline: {speed['source_baseline']:.1f} tokens/sec")

    if "merged" in speed:
        print(f"Merged model:    {speed['merged']:.1f} tokens/sec")
        print(f"Speed retention: {speed['speed_retention_pct']:.1f}%")

    print()
    print("=" * 70)

    # Success criteria check
    if has_merged:
        print("SUCCESS CRITERIA CHECK")
        print("-" * 70)

        # Check reasoning improvement
        reasoning_benchmarks = ["gpqa", "mmlu_pro", "gsm8k", "arc_challenge"]
        improvements = []
        for name in reasoning_benchmarks:
            if name in comparison["benchmarks"] and "improvement_pct" in comparison["benchmarks"][name]:
                improvements.append(comparison["benchmarks"][name]["improvement_pct"])

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            criterion_met = avg_improvement >= 10
            print(f"Reasoning improvement: {avg_improvement:.1f}% (target: >= 10%) "
                  f"{'PASS' if criterion_met else 'FAIL'}")

        # Check speed
        if "speed_retention_pct" in speed:
            speed_ok = speed["speed_retention_pct"] >= 90
            print(f"Speed retention: {speed['speed_retention_pct']:.1f}% (target: >= 90%) "
                  f"{'PASS' if speed_ok else 'FAIL'}")

        print("=" * 70)


def main():
    results_dir = Path("experiments/merge_experiments/results")

    # Load baseline results
    target_path = results_dir / "lfm25_baseline.json"
    source_path = results_dir / "deepseek_r1_baseline.json"
    merged_path = results_dir / "exp001_merged.json"

    if not target_path.exists():
        print(f"ERROR: Target baseline not found: {target_path}")
        sys.exit(1)

    if not source_path.exists():
        print(f"ERROR: Source baseline not found: {source_path}")
        sys.exit(1)

    target = load_results(target_path)
    source = load_results(source_path)
    merged = load_results(merged_path) if merged_path.exists() else None

    comparison = compare_results(target, source, merged)

    # Print report
    print_report(comparison)

    # Save comparison to JSON
    output_path = results_dir / "exp001_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
