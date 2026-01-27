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

"""BenchmarkService: Run benchmarks with geometric metrics.

This service combines benchmark loading with evaluation and geometric
alignment measurement. It supports running multiple benchmarks and
aggregating results.

Usage:
    service = BenchmarkService()
    results = service.run_suite(model, tokenizer, "comprehensive")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from modelcypher.core.use_cases.curriculum.benchmark_loader import (
    BenchmarkLoader,
    Benchmark,
    BenchmarkSample,
)

logger = logging.getLogger(__name__)


# Benchmark suite definitions
SUITES = {
    "quick": ["gsm8k", "arc_easy", "boolq"],
    "reasoning": ["gsm8k", "arc_challenge", "hellaswag"],
    "factual": ["mmlu", "arc_easy", "boolq"],
    "comprehensive": [
        "gsm8k", "arc_easy", "arc_challenge",
        "hellaswag", "boolq",
    ],
}


@dataclass
class FailureCase:
    """A single failure case for analysis."""
    benchmark: str
    prompt: str
    expected: str
    actual: str
    e_pi_matches: int = 0
    comp_phi: float = 0.0


@dataclass
class GeometricMetrics:
    """Geometric alignment metrics for a benchmark run."""
    avg_e_pi_matches: float = 0.0
    avg_comp_phi: float = 0.0
    strong_alignment_pct: float = 0.0  # % with e/π ratio >= 0.40


@dataclass
class BenchmarkResult:
    """Result of a single benchmark evaluation."""
    benchmark: str
    accuracy: float
    correct: int
    total: int
    failures: list[FailureCase] = field(default_factory=list)
    geometric: GeometricMetrics = field(default_factory=GeometricMetrics)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class SuiteResult:
    """Result of running a benchmark suite."""
    suite: str
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    overall_accuracy: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite": self.suite,
            "overall_accuracy": self.overall_accuracy,
            "timestamp": self.timestamp,
            "benchmarks": [
                {
                    "benchmark": r.benchmark,
                    "accuracy": r.accuracy,
                    "correct": r.correct,
                    "total": r.total,
                    "geometric": asdict(r.geometric),
                    "failure_count": len(r.failures),
                }
                for r in self.benchmarks
            ],
        }


class BenchmarkService:
    """Service for running benchmarks with geometric metrics."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.loader = BenchmarkLoader(cache_dir)

    def run_benchmark(
        self,
        model,
        tokenizer,
        benchmark_name: str,
        generate_fn: Callable,
        limit: Optional[int] = None,
        compute_geometry: bool = True,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            benchmark_name: Name of the benchmark (gsm8k, arc_easy, etc.)
            generate_fn: Function to generate text (model, tokenizer, prompt, max_tokens) -> str
            limit: Maximum samples to evaluate
            compute_geometry: Whether to compute geometric metrics

        Returns:
            BenchmarkResult with accuracy and failures
        """
        # Load benchmark
        benchmark = self.loader.load(benchmark_name, split="test", limit=limit)
        logger.info(f"Loaded {len(benchmark.samples)} samples from {benchmark_name}")

        correct = 0
        failures = []
        e_pi_matches_list = []
        comp_phi_list = []

        for sample in benchmark.samples:
            # Generate response
            response = generate_fn(
                model, tokenizer,
                prompt=sample.prompt,
                max_tokens=50,
                verbose=False,
            )

            # Check correctness
            is_correct = self._check_answer(response, sample)

            if is_correct:
                correct += 1
            else:
                failures.append(FailureCase(
                    benchmark=benchmark_name,
                    prompt=sample.prompt[:100],
                    expected=sample.answer,
                    actual=response[:100],
                ))

            # Compute geometry if requested
            if compute_geometry:
                metrics = self._compute_geometry(model, tokenizer, response)
                e_pi_matches_list.append(metrics["e_pi_matches"])
                comp_phi_list.append(metrics["comp_phi"])

        # Calculate geometric aggregates
        geometric = GeometricMetrics()
        if e_pi_matches_list:
            geometric.avg_e_pi_matches = sum(e_pi_matches_list) / len(e_pi_matches_list)
            geometric.avg_comp_phi = sum(comp_phi_list) / len(comp_phi_list)
            total_layers = 16  # LFM2-350M has 16 layers
            strong_count = sum(1 for m in e_pi_matches_list if m / total_layers >= 0.40)
            geometric.strong_alignment_pct = strong_count / len(e_pi_matches_list)

        return BenchmarkResult(
            benchmark=benchmark_name,
            accuracy=correct / len(benchmark.samples),
            correct=correct,
            total=len(benchmark.samples),
            failures=failures[:10],  # Keep top 10 failures
            geometric=geometric,
        )

    def run_suite(
        self,
        model,
        tokenizer,
        suite_name: str,
        generate_fn: Callable,
        limit_per_benchmark: Optional[int] = None,
    ) -> SuiteResult:
        """Run a suite of benchmarks.

        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            suite_name: Name of the suite (quick, comprehensive, etc.)
            generate_fn: Function to generate text
            limit_per_benchmark: Maximum samples per benchmark

        Returns:
            SuiteResult with all benchmark results
        """
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite: {suite_name}. Available: {list(SUITES.keys())}")

        benchmarks = SUITES[suite_name]
        results = []

        for benchmark_name in benchmarks:
            logger.info(f"Running {benchmark_name}...")
            try:
                result = self.run_benchmark(
                    model, tokenizer, benchmark_name, generate_fn,
                    limit=limit_per_benchmark,
                )
                results.append(result)
                logger.info(f"  {benchmark_name}: {result.accuracy:.1%} ({result.correct}/{result.total})")
            except Exception as e:
                logger.error(f"  {benchmark_name}: FAILED - {e}")

        # Calculate overall accuracy
        total_correct = sum(r.correct for r in results)
        total_questions = sum(r.total for r in results)
        overall = total_correct / total_questions if total_questions > 0 else 0.0

        return SuiteResult(
            suite=suite_name,
            benchmarks=results,
            overall_accuracy=overall,
        )

    def _check_answer(self, response: str, sample: BenchmarkSample) -> bool:
        """Check if response contains the expected answer."""
        response_lower = response.lower().strip()
        expected_lower = sample.answer.lower().strip()

        # Direct containment
        if expected_lower in response_lower:
            return True

        # For multiple choice, check if the letter is selected
        if sample.choices:
            for i, choice in enumerate(sample.choices):
                if choice.lower() == expected_lower:
                    letter = chr(65 + i)  # A, B, C, D
                    if letter.lower() in response_lower[:5]:
                        return True

        return False

    def _compute_geometry(self, model, tokenizer, text: str) -> dict:
        """Compute geometric metrics for a response."""
        try:
            from modelcypher.core.domain.inference.self_align import compute_alignment_metrics
            metrics = compute_alignment_metrics(model, tokenizer, text)
            return {
                "e_pi_matches": metrics.e_pi_matches,
                "comp_phi": metrics.comp_phi,
            }
        except Exception:
            return {"e_pi_matches": 0, "comp_phi": 0.0}

    def save_results(self, result: SuiteResult, output_path: Path) -> None:
        """Save benchmark results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")
