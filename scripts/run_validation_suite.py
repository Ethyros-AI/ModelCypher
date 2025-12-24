#!/usr/bin/env python3
"""
ModelCypher Comprehensive Validation Suite

Runs all CLI/MCP commands against 4 test models and generates
reproducible validation report.

Usage:
    poetry run python scripts/run_validation_suite.py --output-dir /path/to/results
    poetry run python scripts/run_validation_suite.py --category A  # Run only Category A
    poetry run python scripts/run_validation_suite.py --model M1    # Run only on M1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Model definitions
MODELS = {
    "M1": "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    "M2": "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
    "M3": "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-3B-Instruct-bf16",
    "M4": "/Volumes/CodeCypher/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

MODEL_INFO = {
    "M1": {"name": "Qwen2.5-0.5B-Instruct-bf16", "type": "General", "size": "0.5B", "family": "Qwen"},
    "M2": {"name": "Qwen2.5-3B-Instruct-bf16", "type": "General", "size": "3B", "family": "Qwen"},
    "M3": {"name": "Qwen2.5-Coder-3B-Instruct-bf16", "type": "Coder", "size": "3B", "family": "Qwen"},
    "M4": {"name": "Mistral-7B-Instruct-v0.3-4bit", "type": "General", "size": "7B", "family": "Mistral"},
}


@dataclass
class TestResult:
    test_id: str
    category: str
    command: str
    model: str | None
    status: str  # "pass", "fail", "error", "skip"
    output: dict[str, Any] | None
    error: str | None
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "testId": self.test_id,
            "category": self.category,
            "command": self.command,
            "model": self.model,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "durationSeconds": round(self.duration_seconds, 2),
            "timestamp": self.timestamp,
        }


def run_cli_command(command: str, timeout: int = 300) -> tuple[dict | None, str | None, float]:
    """Execute CLI command and return (output, error, duration)."""
    start = time.time()
    try:
        # Add --ai flag for JSON output
        if "--output" not in command and "--ai" not in command:
            command = command + " --ai"

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                return output, None, duration
            except json.JSONDecodeError:
                return {"raw": result.stdout}, None, duration
        else:
            return None, result.stderr or result.stdout, duration
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout}s", time.time() - start
    except Exception as e:
        return None, str(e), time.time() - start


def validate_result(test_id: str, output: dict | None, criteria: dict) -> bool:
    """Validate output against criteria."""
    if output is None:
        return False

    for key, check in criteria.items():
        if key not in output:
            if check.get("required", True):
                return False
            continue

        value = output[key]

        if "min" in check and value < check["min"]:
            return False
        if "max" in check and value > check["max"]:
            return False
        if "in" in check and value not in check["in"]:
            return False
        if "type" in check and not isinstance(value, check["type"]):
            return False

    return True


# Test definitions by category
TESTS = {
    "A": {  # Model Introspection
        "A1": {
            "name": "Model Probe",
            "command": "poetry run mc model probe {model}",
            "per_model": True,
            "criteria": {"layers": {"min": 1}, "hiddenSize": {"min": 1}},
        },
        "A2": {
            "name": "Vocab Compare Same Family",
            "command": "poetry run mc model vocab-compare {M1} {M2}",
            "per_model": False,
            "criteria": {"overlapRatio": {"min": 0, "max": 1}},
        },
        "A3": {
            "name": "Vocab Compare Cross Family",
            "command": "poetry run mc model vocab-compare {M1} {M4}",
            "per_model": False,
            "criteria": {"overlapRatio": {"min": 0, "max": 1}},
        },
        "A4": {
            "name": "Validate Merge Same Family",
            "command": "poetry run mc model validate-merge --source {M1} --target {M2}",
            "per_model": False,
            "criteria": {},
        },
        "A5": {
            "name": "Validate Merge Cross Family",
            "command": "poetry run mc model validate-merge --source {M1} --target {M4}",
            "per_model": False,
            "criteria": {},
        },
    },
    "B": {  # Geometry Metrics
        "B1": {
            "name": "Gromov-Wasserstein Distance",
            "command": "poetry run mc geometry metrics gromov-wasserstein {M1} {M2}",
            "per_model": False,
            "criteria": {"distance": {"min": 0, "max": 1}},
        },
        "B2": {
            "name": "Intrinsic Dimension",
            "command": "poetry run mc geometry metrics intrinsic-dimension {model}",
            "per_model": True,
            "criteria": {"dimension": {"min": 1}},
        },
        "B4": {
            "name": "Spatial Probe Model",
            "command": "poetry run mc geometry spatial probe-model {model}",
            "per_model": True,
            "criteria": {},
        },
        "B5": {
            "name": "Spatial Euclidean",
            "command": "poetry run mc geometry spatial euclidean {model}",
            "per_model": True,
            "criteria": {},
        },
    },
    "D": {  # Safety & Entropy
        "D1": {
            "name": "Circuit Breaker",
            "command": 'poetry run mc geometry safety circuit-breaker --model {model} --prompt "Hello, how are you?"',
            "per_model": True,
            "criteria": {},
        },
        "D2": {
            "name": "Thermo Measure",
            "command": 'poetry run mc thermo measure --model {model} --prompt "Hello"',
            "per_model": True,
            "criteria": {},
        },
    },
    "G": {  # Inference
        "G1": {
            "name": "Basic Math",
            "command": 'poetry run mc infer run --model {model} --prompt "What is 2+2? Answer with just the number."',
            "per_model": True,
            "criteria": {},
        },
        "G2": {
            "name": "Code Completion",
            "command": 'poetry run mc infer run --model {M3} --prompt "Complete this Python function:\\ndef hello():\\n    "',
            "per_model": False,
            "criteria": {},
        },
    },
}


def run_category(category: str, output_dir: Path, models_filter: list[str] | None = None) -> list[TestResult]:
    """Run all tests in a category."""
    results = []
    tests = TESTS.get(category, {})

    for test_id, test_def in tests.items():
        print(f"\n{'='*60}")
        print(f"Running {test_id}: {test_def['name']}")
        print(f"{'='*60}")

        if test_def.get("per_model", False):
            # Run for each model
            for model_id, model_path in MODELS.items():
                if models_filter and model_id not in models_filter:
                    continue

                cmd = test_def["command"].format(model=model_path)
                print(f"\n  Model: {model_id}")
                print(f"  Command: {cmd}")

                output, error, duration = run_cli_command(cmd)

                if output:
                    status = "pass" if validate_result(test_id, output, test_def.get("criteria", {})) else "fail"
                else:
                    status = "error"

                result = TestResult(
                    test_id=f"{test_id}_{model_id}",
                    category=category,
                    command=cmd,
                    model=model_id,
                    status=status,
                    output=output,
                    error=error,
                    duration_seconds=duration,
                )
                results.append(result)

                # Save individual result
                result_file = output_dir / f"{test_id}_{model_id}.json"
                with open(result_file, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

                print(f"  Status: {status} ({duration:.2f}s)")
        else:
            # Single test (not per-model)
            cmd = test_def["command"]
            for key, path in MODELS.items():
                cmd = cmd.replace(f"{{{key}}}", path)

            print(f"  Command: {cmd}")

            output, error, duration = run_cli_command(cmd)

            if output:
                status = "pass" if validate_result(test_id, output, test_def.get("criteria", {})) else "fail"
            else:
                status = "error"

            result = TestResult(
                test_id=test_id,
                category=category,
                command=cmd,
                model=None,
                status=status,
                output=output,
                error=error,
                duration_seconds=duration,
            )
            results.append(result)

            # Save result
            result_file = output_dir / f"{test_id}.json"
            with open(result_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            print(f"  Status: {status} ({duration:.2f}s)")

    return results


def generate_summary(results: list[TestResult]) -> dict:
    """Generate summary statistics."""
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    errors = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status == "skip")

    total_duration = sum(r.duration_seconds for r in results)

    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}
        by_category[r.category]["total"] += 1
        if r.status == "pass":
            by_category[r.category]["passed"] += 1
        elif r.status == "fail":
            by_category[r.category]["failed"] += 1
        elif r.status == "error":
            by_category[r.category]["errors"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "passRate": round(passed / total * 100, 1) if total > 0 else 0,
        "totalDurationSeconds": round(total_duration, 1),
        "byCategory": by_category,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="ModelCypher Validation Suite")
    parser.add_argument("--output-dir", type=str, default="/Volumes/CodeCypher/experiments/validation-20251223")
    parser.add_argument("--category", type=str, help="Run only specific category (A, B, C, D, E, F, G)")
    parser.add_argument("--model", type=str, help="Run only on specific model (M1, M2, M3, M4)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ModelCypher Comprehensive Validation Suite")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Verify models exist
    print("Verifying models...")
    for model_id, model_path in MODELS.items():
        if Path(model_path).exists():
            print(f"  ✓ {model_id}: {model_path}")
        else:
            print(f"  ✗ {model_id}: {model_path} (NOT FOUND)")
    print()

    all_results = []

    categories = [args.category] if args.category else list(TESTS.keys())
    models_filter = [args.model] if args.model else None

    for category in categories:
        if category not in TESTS:
            print(f"Unknown category: {category}")
            continue

        print(f"\n{'#'*60}")
        print(f"# CATEGORY {category}")
        print(f"{'#'*60}")

        category_dir = output_dir / category.lower()
        category_dir.mkdir(exist_ok=True)

        results = run_category(category, category_dir, models_filter)
        all_results.extend(results)

    # Generate summary
    summary = generate_summary(all_results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Pass rate: {summary['passRate']}%")
    print(f"Total duration: {summary['totalDurationSeconds']}s")

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Save all results
    all_results_file = output_dir / "all_results.json"
    with open(all_results_file, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return 0 if summary["errors"] == 0 and summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
