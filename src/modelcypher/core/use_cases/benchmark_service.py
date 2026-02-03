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
    "local_smoke": ["local:tests/fixtures/benchmark_smoke.json"],
}


@dataclass
class FailureCase:
    """A single failure case for analysis."""
    benchmark: str
    prompt: str
    expected: str
    actual: str
    e_pi_matches: int = 0
    expansion_ratio: float = 0.0


@dataclass
class GeometricMetrics:
    """Geometric alignment metrics for a benchmark run."""
    avg_e_pi_matches: float = 0.0
    avg_expansion_ratio: float = 0.0
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
    entropy_profile: dict | None = None
    intrinsic_dimension_profile: dict | None = None
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
            "entropy_profile": self.entropy_profile,
            "intrinsic_dimension_profile": self.intrinsic_dimension_profile,
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
        max_failures: Optional[int] = 10,
        max_tokens: int = 512,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            benchmark_name: Name of the benchmark (gsm8k, arc_easy, etc.)
            generate_fn: Function to generate text (model, tokenizer, prompt, max_tokens) -> str
            limit: Maximum samples to evaluate
            compute_geometry: Whether to compute geometric metrics
            max_tokens: Maximum tokens for generation (default 512 - let the model think)

        Returns:
            BenchmarkResult with accuracy and failures
        """
        # Load benchmark
        benchmark = self.loader.load(benchmark_name, split="test", limit=limit)
        logger.info(f"Loaded {len(benchmark.samples)} samples from {benchmark_name}")

        correct = 0
        failures = []
        e_pi_matches_list = []
        expansion_ratio_list = []

        for sample in benchmark.samples:
            # Generate response - let the model take whatever journey it needs
            response = generate_fn(
                model, tokenizer,
                prompt=sample.prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            # Compute geometry if requested
            metrics = {"e_pi_matches": 0, "expansion_ratio": 0.0}
            if compute_geometry:
                metrics = self._compute_geometry(model, tokenizer, response)
                e_pi_matches_list.append(metrics["e_pi_matches"])
                expansion_ratio_list.append(metrics["expansion_ratio"])

            # Check correctness
            is_correct = self._check_answer(response, sample)

            if is_correct:
                correct += 1
            else:
                failures.append(FailureCase(
                    benchmark=benchmark_name,
                    prompt=sample.prompt,
                    expected=sample.answer,
                    actual=response,
                    e_pi_matches=metrics["e_pi_matches"],
                    expansion_ratio=metrics["expansion_ratio"],
                ))

        # Calculate geometric aggregates
        geometric = GeometricMetrics()
        if e_pi_matches_list:
            geometric.avg_e_pi_matches = sum(e_pi_matches_list) / len(e_pi_matches_list)
            geometric.avg_expansion_ratio = sum(expansion_ratio_list) / len(expansion_ratio_list)
            total_layers = 16  # LFM2-350M has 16 layers
            strong_count = sum(1 for m in e_pi_matches_list if m / total_layers >= 0.40)
            geometric.strong_alignment_pct = strong_count / len(e_pi_matches_list)

        if max_failures is not None:
            failures = failures[:max_failures]

        return BenchmarkResult(
            benchmark=benchmark_name,
            accuracy=correct / len(benchmark.samples),
            correct=correct,
            total=len(benchmark.samples),
            failures=failures,
            geometric=geometric,
        )

    def run_suite(
        self,
        model,
        tokenizer,
        suite_name: str,
        generate_fn: Callable,
        limit_per_benchmark: Optional[int] = None,
        max_failures: Optional[int] = 10,
        max_tokens: int = 512,
        entropy_probe_path: str | None = None,
    ) -> SuiteResult:
        """Run a suite of benchmarks.

        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            suite_name: Name of the suite (quick, comprehensive, etc.)
            generate_fn: Function to generate text
            limit_per_benchmark: Maximum samples per benchmark
            max_tokens: Maximum tokens for generation (let the model think)

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
                    max_failures=max_failures,
                    max_tokens=max_tokens,
                )
                results.append(result)
                logger.info(f"  {benchmark_name}: {result.accuracy:.1%} ({result.correct}/{result.total})")
            except Exception as e:
                logger.error(f"  {benchmark_name}: FAILED - {e}")

        # Calculate overall accuracy
        total_correct = sum(r.correct for r in results)
        total_questions = sum(r.total for r in results)
        overall = total_correct / total_questions if total_questions > 0 else 0.0

        entropy_profile = None
        intrinsic_dimension_profile = None
        if entropy_probe_path:
            probe_prompts = self._load_probe_prompts(entropy_probe_path)
            entropy_profile = self._compute_entropy_profile(model, tokenizer, probe_prompts)
            intrinsic_dimension_profile = self._compute_intrinsic_dimension_profile(
                model, tokenizer, probe_prompts
            )

        return SuiteResult(
            suite=suite_name,
            benchmarks=results,
            overall_accuracy=overall,
            entropy_profile=entropy_profile,
            intrinsic_dimension_profile=intrinsic_dimension_profile,
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
                "expansion_ratio": metrics.expansion_ratio,
            }
        except Exception:
            return {"e_pi_matches": 0, "expansion_ratio": 0.0}

    def _load_probe_prompts(self, path: str) -> list[str]:
        probe_path = Path(path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe prompts not found: {probe_path}")
        prompts: list[str] = []
        if probe_path.suffix == ".jsonl":
            import json
            for line in probe_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("prompt") or record.get("text")
                if prompt:
                    prompts.append(prompt)
        else:
            prompts = [line.strip() for line in probe_path.read_text().splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts loaded from {probe_path}")
        return prompts

    def _compute_entropy_profile(self, model, tokenizer, prompts: list[str]) -> dict:
        from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector

        projector = LayerEntropyProjector()
        profile = projector.profile_model(model, tokenizer, prompts)
        trajectory = profile.entropy_trajectory()
        if not trajectory:
            return {"trajectory": []}
        peak_idx = max(range(len(trajectory)), key=lambda i: trajectory[i])
        peak = trajectory[peak_idx]
        initial = trajectory[0]
        final = trajectory[-1]
        expansion_rate = (peak - initial) / float(max(1, peak_idx))
        compression_rate = (peak - final) / float(max(1, (len(trajectory) - 1 - peak_idx)))
        # expansion_ratio = compression_rate / expansion_rate (target: 1.0)
        expansion_ratio = (
            compression_rate / expansion_rate
            if expansion_rate != 0.0
            else 0.0
        )
        return {
            "model_name": profile.model_name,
            "created_at": profile.created_at.isoformat(),
            "trajectory": trajectory,
            "peak_layer": peak_idx,
            "initial_entropy": initial,
            "peak_entropy": peak,
            "final_entropy": final,
            "expansion_rate": expansion_rate,
            "compression_rate": compression_rate,
            "expansion_ratio": expansion_ratio,
            "layer_stats": {
                idx: {
                    "mean_entropy": result.mean_entropy,
                    "entropy_variance": result.entropy_variance,
                    "min_entropy": result.min_entropy,
                    "max_entropy": result.max_entropy,
                    "sample_count": result.sample_count,
                }
                for idx, result in profile.layer_results.items()
            },
        }

    def _compute_intrinsic_dimension_profile(self, model, tokenizer, prompts: list[str]) -> dict:
        """Compute intrinsic dimension profile across layers.

        Uses compute_with_convergence() which finds the minimum sample size
        needed for a stable ID estimate. The convergence criterion is derived
        from machine epsilon (sqrt(eps)), not guessed. This automatically
        handles memory by only using as many points as the geometry requires.
        """
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

        backend = get_default_backend()
        projector = LayerEntropyProjector(backend)
        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model.layers or model.model.layers")

        num_layers = len(layers)
        target_layers = set(range(num_layers))
        layer_points: dict[int, list] = {i: [] for i in target_layers}

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            if isinstance(tokens, list):
                input_ids = backend.array([tokens])
            else:
                input_ids = tokens
                if input_ids.ndim == 1:
                    input_ids = backend.reshape(input_ids, (1, -1))

            captured = projector._capture_layer_states(base_model, layers, input_ids, target_layers)
            for layer_idx, hidden_state in captured.items():
                if hidden_state.ndim == 3:
                    pts = hidden_state[0, :, :]
                elif hidden_state.ndim == 2:
                    pts = hidden_state
                else:
                    pts = backend.reshape(hidden_state, (1, -1))
                layer_points[layer_idx].append(pts)

            # Explicit cleanup to prevent memory accumulation
            del captured

        estimator = IntrinsicDimension(backend)
        min_samples = IntrinsicDimension.local_dimension_min_samples()
        id_results: dict[int, dict] = {}

        for layer_idx, pts_list in layer_points.items():
            if not pts_list:
                continue
            all_pts = pts_list[0] if len(pts_list) == 1 else backend.concatenate(pts_list, axis=0)
            sample_count = int(all_pts.shape[0])

            # Free the list now that we have concatenated
            layer_points[layer_idx] = []

            if sample_count < min_samples:
                id_results[layer_idx] = {
                    "intrinsic_dimension": None,
                    "sample_count": sample_count,
                    "usable_count": 0,
                    "ci_lower": None,
                    "ci_upper": None,
                    "ci_resamples": None,
                }
                continue

            # Use convergence-based estimation: automatically finds minimum
            # sample size needed for stable estimate. Convergence threshold
            # is sqrt(machine_epsilon) - derived from numerical precision,
            # not guessed. This handles memory implicitly by only computing
            # geodesics on the subsample that achieves convergence.
            estimate = estimator.compute_with_convergence(all_pts, with_ci=True)
            id_results[layer_idx] = {
                "intrinsic_dimension": estimate.intrinsic_dimension,
                "sample_count": sample_count,
                "usable_count": estimate.usable_count,  # Actual samples used for convergence
                "ci_lower": estimate.ci.lower if estimate.ci else None,
                "ci_upper": estimate.ci.upper if estimate.ci else None,
                "ci_resamples": estimate.ci.resamples if estimate.ci else None,
            }

            # Free the points array
            del all_pts

        return {
            "model_name": getattr(model, "name", None) or model.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "layers": id_results,
        }

    def save_results(self, result: SuiteResult, output_path: Path) -> None:
        """Save benchmark results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")
