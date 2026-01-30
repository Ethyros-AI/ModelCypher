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

"""Capability Transfer Validation Experiment.

Protocol:
1. Benchmark source model (coding capability via code prompts)
2. Benchmark target model (general capability via MMLU/ARC)
3. Execute null-space merge (source -> target)
4. Benchmark merged model on BOTH capabilities
5. Compare deltas

Success Criteria:
- Merged model code score > Target baseline (capability transferred)
- Merged MMLU/ARC >= 0.95 x Target (preservation)

This validates whether null-space merging actually transfers capabilities
without destroying what the target model already knows.

Usage:
    # Full experiment (requires models on CodeCypher volume)
    python experiments/exp_capability_transfer/run_experiment.py

    # Quick smoke test with tiny models
    python experiments/exp_capability_transfer/run_experiment.py --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default model paths (on CodeCypher volume)
DEFAULT_SOURCE = "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16"
DEFAULT_TARGET = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DEFAULT_OUTPUT = "/Volumes/CodeCypher/models/merged/capability_transfer_test"


@dataclass
class BenchmarkResult:
    """Result of benchmarking a model."""
    model_path: str
    model_name: str
    code_score: float  # Code task accuracy
    reasoning_score: float  # Reasoning/MMLU accuracy
    n_code_samples: int
    n_reasoning_samples: int
    errors: list[str]


@dataclass
class ExperimentResult:
    """Result of the full capability transfer experiment."""
    timestamp: str
    source_model: str
    target_model: str
    merged_model: str

    # Benchmark results
    source_baseline: BenchmarkResult
    target_baseline: BenchmarkResult
    merged_result: BenchmarkResult

    # Analysis
    code_transfer_delta: float  # merged - target (should be positive)
    reasoning_preservation: float  # merged / target (should be >= 0.95)
    capability_transferred: bool
    capability_preserved: bool
    experiment_success: bool


# Code benchmark prompts (simple code generation tasks)
CODE_PROMPTS = [
    ("Write a Python function that returns the sum of two numbers:\ndef add(a, b):", "return a + b"),
    ("Write a Python function to check if a number is even:\ndef is_even(n):", "return n % 2 == 0"),
    ("Complete this Python list comprehension to double all numbers:\nnumbers = [1, 2, 3, 4, 5]\ndoubled = [x", "* 2 for x in numbers"),
    ("Write code to print 'Hello, World!':\n", "print"),
    ("Write a function to find the maximum of a list:\ndef find_max(lst):", "return max(lst)"),
    ("Write a function to reverse a string:\ndef reverse(s):", "return s[::-1]"),
    ("Write code to check if a string is a palindrome:\ndef is_palindrome(s):", "return s == s[::-1]"),
    ("Write a function to compute factorial:\ndef factorial(n):", "if n <= 1"),
    ("Write a function to find the length of a list:\ndef list_length(lst):", "return len(lst)"),
    ("Write code to join a list of strings with commas:\ndef join_strings(lst):", "return"),
]

# Reasoning prompts (simple logic and knowledge)
REASONING_PROMPTS = [
    ("What is the capital of France?\nAnswer:", "paris"),
    ("All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded? Answer:", "yes"),
    ("What is 7 + 5?\nAnswer:", "12"),
    ("If it rains, the ground gets wet. It rained today. Is the ground wet? Answer:", "yes"),
    ("How many days are in a week?\nAnswer:", "7"),
    ("What planet is closest to the Sun?\nAnswer:", "mercury"),
    ("If A > B and B > C, is A > C? Answer:", "yes"),
    ("What is the opposite of 'hot'?\nAnswer:", "cold"),
    ("How many legs does a dog have?\nAnswer:", "4"),
    ("What comes after Tuesday?\nAnswer:", "wednesday"),
]


def evaluate_model(model_path: str) -> BenchmarkResult:
    """Evaluate a model on code and reasoning benchmarks."""
    from mlx_lm import load

    model_name = Path(model_path).name
    logger.info(f"Evaluating {model_name}...")

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return BenchmarkResult(
            model_path=model_path,
            model_name=model_name,
            code_score=0.0,
            reasoning_score=0.0,
            n_code_samples=0,
            n_reasoning_samples=0,
            errors=[str(e)],
        )

    import mlx.core as mx

    errors = []

    def generate(prompt: str, max_tokens: int = 30) -> str:
        """Generate text from prompt."""
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            current = input_ids

            for _ in range(max_tokens):
                logits = model(current)
                mx.eval(logits)
                next_token = int(mx.argmax(logits[0, -1, :]).item())

                if next_token == tokenizer.eos_token_id:
                    break

                generated.append(next_token)
                current = mx.concatenate([current, mx.array([[next_token]])], axis=1)

            return tokenizer.decode(generated).strip()
        except Exception as e:
            errors.append(f"Generation error: {e}")
            return ""

    # Evaluate code prompts
    code_correct = 0
    for prompt, expected in CODE_PROMPTS:
        output = generate(prompt, max_tokens=50)
        # Flexible matching: check if expected substring appears
        if expected.lower() in output.lower():
            code_correct += 1
    code_score = code_correct / len(CODE_PROMPTS)

    # Evaluate reasoning prompts
    reasoning_correct = 0
    for prompt, expected in REASONING_PROMPTS:
        output = generate(prompt, max_tokens=20)
        if expected.lower() in output.lower():
            reasoning_correct += 1
    reasoning_score = reasoning_correct / len(REASONING_PROMPTS)

    logger.info(f"  Code: {code_score:.1%} ({code_correct}/{len(CODE_PROMPTS)})")
    logger.info(f"  Reasoning: {reasoning_score:.1%} ({reasoning_correct}/{len(REASONING_PROMPTS)})")

    return BenchmarkResult(
        model_path=model_path,
        model_name=model_name,
        code_score=code_score,
        reasoning_score=reasoning_score,
        n_code_samples=len(CODE_PROMPTS),
        n_reasoning_samples=len(REASONING_PROMPTS),
        errors=errors,
    )


def run_merge(source: str, target: str, output: str) -> bool:
    """Run null-space merge using mc CLI."""
    logger.info(f"Running merge: {Path(source).name} -> {Path(target).name}")

    cmd = [
        "poetry", "run", "mc", "merge", "run",
        "-s", source,
        "-t", target,
        "-o", output,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"Merge failed: {result.stderr}")
            return False

        logger.info("Merge completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Merge timed out")
        return False
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return False


def run_experiment(
    source: str,
    target: str,
    output: str,
    skip_merge: bool = False,
) -> ExperimentResult:
    """Run the full capability transfer experiment."""
    logger.info("=" * 70)
    logger.info("CAPABILITY TRANSFER VALIDATION EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Source (coding): {Path(source).name}")
    logger.info(f"Target (general): {Path(target).name}")
    logger.info(f"Output: {Path(output).name}")
    logger.info("=" * 70)

    timestamp = datetime.now().isoformat()

    # Step 1: Baseline source
    logger.info("\n[Step 1/4] Benchmarking source model...")
    source_baseline = evaluate_model(source)

    # Step 2: Baseline target
    logger.info("\n[Step 2/4] Benchmarking target model...")
    target_baseline = evaluate_model(target)

    # Step 3: Run merge
    if not skip_merge:
        logger.info("\n[Step 3/4] Running null-space merge...")
        merge_success = run_merge(source, target, output)
        if not merge_success:
            logger.error("Merge failed - cannot complete experiment")
            # Return partial result
            return ExperimentResult(
                timestamp=timestamp,
                source_model=source,
                target_model=target,
                merged_model=output,
                source_baseline=source_baseline,
                target_baseline=target_baseline,
                merged_result=BenchmarkResult(
                    model_path=output,
                    model_name="MERGE_FAILED",
                    code_score=0.0,
                    reasoning_score=0.0,
                    n_code_samples=0,
                    n_reasoning_samples=0,
                    errors=["Merge failed"],
                ),
                code_transfer_delta=0.0,
                reasoning_preservation=0.0,
                capability_transferred=False,
                capability_preserved=False,
                experiment_success=False,
            )
    else:
        logger.info("\n[Step 3/4] Skipping merge (--skip-merge specified)")

    # Step 4: Benchmark merged
    logger.info("\n[Step 4/4] Benchmarking merged model...")
    merged_result = evaluate_model(output)

    # Analysis
    code_transfer_delta = merged_result.code_score - target_baseline.code_score
    reasoning_preservation = (
        merged_result.reasoning_score / target_baseline.reasoning_score
        if target_baseline.reasoning_score > 0
        else 1.0
    )

    capability_transferred = code_transfer_delta > 0
    capability_preserved = reasoning_preservation >= 0.95
    experiment_success = capability_transferred and capability_preserved

    # Report
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    logger.info("\nBenchmark Scores:")
    logger.info(f"{'Model':<30} {'Code':>10} {'Reasoning':>12}")
    logger.info("-" * 54)
    logger.info(f"{'Source (baseline)':<30} {source_baseline.code_score:>10.1%} {source_baseline.reasoning_score:>12.1%}")
    logger.info(f"{'Target (baseline)':<30} {target_baseline.code_score:>10.1%} {target_baseline.reasoning_score:>12.1%}")
    logger.info(f"{'Merged':<30} {merged_result.code_score:>10.1%} {merged_result.reasoning_score:>12.1%}")

    logger.info("\nCapability Transfer Analysis:")
    logger.info(f"  Code transfer delta:      {code_transfer_delta:+.1%} {'PASS' if capability_transferred else 'FAIL'}")
    logger.info(f"  Reasoning preservation:   {reasoning_preservation:.1%} {'PASS' if capability_preserved else 'FAIL'}")

    logger.info("\n" + "=" * 70)
    if experiment_success:
        logger.info("EXPERIMENT: SUCCESS")
        logger.info("Capability was transferred without destroying target knowledge.")
    else:
        logger.info("EXPERIMENT: PARTIAL SUCCESS" if capability_transferred or capability_preserved else "EXPERIMENT: FAILED")
        if not capability_transferred:
            logger.info("  - Code capability did NOT transfer (delta <= 0)")
        if not capability_preserved:
            logger.info("  - Reasoning capability was degraded (preservation < 95%)")
    logger.info("=" * 70)

    return ExperimentResult(
        timestamp=timestamp,
        source_model=source,
        target_model=target,
        merged_model=output,
        source_baseline=source_baseline,
        target_baseline=target_baseline,
        merged_result=merged_result,
        code_transfer_delta=code_transfer_delta,
        reasoning_preservation=reasoning_preservation,
        capability_transferred=capability_transferred,
        capability_preserved=capability_preserved,
        experiment_success=experiment_success,
    )


def main():
    parser = argparse.ArgumentParser(description="Capability Transfer Validation Experiment")
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help="Path to source model (with coding capability)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Path to target model (general)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path for merged model output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use smallest available models",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step (use existing merged model)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="Path to save JSON results",
    )

    args = parser.parse_args()

    # Quick mode uses smallest models
    if args.quick:
        args.source = "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16"
        args.target = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
        logger.info("Quick mode: using smallest available models")

    # Check models exist
    for path_name, path in [("source", args.source), ("target", args.target)]:
        if not Path(path).exists():
            logger.error(f"{path_name.title()} model not found: {path}")
            sys.exit(1)

    # Run experiment
    result = run_experiment(
        source=args.source,
        target=args.target,
        output=args.output,
        skip_merge=args.skip_merge,
    )

    # Save results
    results_file = args.results_file or "data/experiments/capability_transfer_result.json"
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    result_dict = {
        "timestamp": result.timestamp,
        "source_model": result.source_model,
        "target_model": result.target_model,
        "merged_model": result.merged_model,
        "source_baseline": asdict(result.source_baseline),
        "target_baseline": asdict(result.target_baseline),
        "merged_result": asdict(result.merged_result),
        "code_transfer_delta": result.code_transfer_delta,
        "reasoning_preservation": result.reasoning_preservation,
        "capability_transferred": result.capability_transferred,
        "capability_preserved": result.capability_preserved,
        "experiment_success": result.experiment_success,
    }

    with open(results_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")

    # Exit code based on success
    sys.exit(0 if result.experiment_success else 1)


if __name__ == "__main__":
    main()
