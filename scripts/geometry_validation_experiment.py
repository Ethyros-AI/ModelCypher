#!/usr/bin/env python3
"""Geometry Validation Experiment: Testing whether geometric metrics predict reasoning quality.

This script collects per-sample geometric metrics alongside benchmark accuracy
to test whether there's a deterministic relationship between geometry and correctness.

Research Questions:
1. Do geometric metrics have distinguishable distributions for correct vs incorrect samples?
2. If yes, is it because correct reasoning has a specific geometric structure?
3. Can we predict correctness BEFORE seeing the answer using geometry alone?

The goal is to find MECHANISM, not just correlation. We want to understand WHAT
geometric transformation distinguishes correct from incorrect processing.

Usage:
    poetry run python scripts/geometry_validation_experiment.py \
        --model /path/to/model \
        --output results/geometry_validation/

Output:
    - Per-sample JSONL with all metrics
    - Summary statistics
    - Distribution analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GeometricSample:
    """Per-sample geometric metrics alongside benchmark result."""

    # Identification
    sample_id: str
    benchmark: str
    prompt: str
    expected_answer: str
    model_response: str

    # Correctness (the dependent variable)
    is_correct: bool

    # Geometric metrics (independent variables)
    # All computed BEFORE seeing the answer (on the prompt, not the response)

    # Entropy trajectory metrics
    entropy_trajectory: list[float] = field(default_factory=list)
    initial_entropy: float = 0.0
    peak_entropy: float = 0.0
    final_entropy: float = 0.0
    peak_layer: int = 0
    expansion_rate: float = 0.0
    compression_rate: float = 0.0
    expansion_ratio: float = 0.0

    # Intrinsic dimension metrics (per-layer)
    intrinsic_dimension_trajectory: list[float] = field(default_factory=list)
    mean_intrinsic_dimension: float = 0.0
    final_intrinsic_dimension: float = 0.0
    id_expansion_ratio: float = 0.0  # peak_id / final_id

    # Spectral entropy metrics (per-layer)
    spectral_entropy_trajectory: list[float] = field(default_factory=list)
    mean_spectral_entropy: float = 0.0
    final_spectral_entropy: float = 0.0

    # Reasoning flow metrics (on response trajectory)
    arc_length: float = 0.0
    mean_curvature: float = 0.0
    max_curvature: float = 0.0
    smoothness: float = 0.0
    directness: float = 0.0
    velocity_variance: float = 0.0

    # Error tracking
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for the validation experiment."""

    model_path: str
    output_dir: Path
    benchmarks: list[str] = field(
        default_factory=lambda: ["gsm8k", "arc_challenge", "hellaswag", "boolq"]
    )
    samples_per_benchmark: int = 100
    max_tokens: int = 512
    seed: int = 42
    temperature: float = 0.0  # Greedy for reproducibility


class GeometryValidationExperiment:
    """Orchestrates the geometry validation experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.results: list[GeometricSample] = []

    def setup(self) -> None:
        """Load model and initialize services."""
        from modelcypher.backends import initialize_default_backend

        logger.info(f"Loading model from {self.config.model_path}")
        self.backend = initialize_default_backend()

        # Load model using backend
        model_path = Path(self.config.model_path)
        self.model, self.tokenizer = self.backend.load_model(str(model_path))

        logger.info(f"Model loaded: {type(self.model).__name__}")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate(self, prompt: str) -> str:
        """Generate response using greedy decoding."""
        # Note: mlx_lm uses default greedy decoding, temperature isn't passed directly
        return self.backend.generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.config.max_tokens,
        )

    def _compute_entropy_trajectory(self, text: str) -> dict[str, Any]:
        """Compute layer-wise entropy trajectory for text."""
        from modelcypher.core.domain.entropy.layer_entropy_projector import (
            LayerEntropyProjector,
        )

        try:
            projector = LayerEntropyProjector(self.backend)
            profile = projector.profile_model(self.model, self.tokenizer, [text])
            trajectory = profile.entropy_trajectory()

            if not trajectory:
                return {"error": "Empty entropy trajectory"}

            peak_idx = max(range(len(trajectory)), key=lambda i: trajectory[i])
            peak = trajectory[peak_idx]
            initial = trajectory[0]
            final = trajectory[-1]

            expansion_rate = (peak - initial) / float(max(1, peak_idx))
            compression_rate = (peak - final) / float(
                max(1, (len(trajectory) - 1 - peak_idx))
            )
            expansion_ratio = (
                compression_rate / expansion_rate if expansion_rate != 0.0 else 0.0
            )

            return {
                "entropy_trajectory": trajectory,
                "initial_entropy": initial,
                "peak_entropy": peak,
                "final_entropy": final,
                "peak_layer": peak_idx,
                "expansion_rate": expansion_rate,
                "compression_rate": compression_rate,
                "expansion_ratio": expansion_ratio,
            }
        except Exception as e:
            return {"error": str(e)}

    def _compute_intrinsic_dimension_trajectory(self, text: str) -> dict[str, Any]:
        """Compute layer-wise intrinsic dimension."""
        from modelcypher.core.domain.entropy.layer_entropy_projector import (
            LayerEntropyProjector,
        )
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
        )

        try:
            projector = LayerEntropyProjector(self.backend)

            # Get model layers
            base_model = getattr(self.model, "model", self.model)
            layers = getattr(base_model, "layers", None)
            if layers is None:
                return {"error": "Could not find model layers"}

            num_layers = len(layers)
            target_layers = set(range(num_layers))

            # Encode text
            tokens = self.tokenizer.encode(text)
            if isinstance(tokens, list):
                input_ids = self.backend.array([tokens])
            else:
                input_ids = tokens
                if input_ids.ndim == 1:
                    input_ids = self.backend.reshape(input_ids, (1, -1))

            # Capture hidden states
            captured = projector._capture_layer_states(
                base_model, layers, input_ids, target_layers
            )

            # Compute ID per layer
            estimator = IntrinsicDimension(self.backend)
            id_trajectory = []

            for layer_idx in sorted(captured.keys()):
                hidden = captured[layer_idx]
                if hidden.ndim == 3:
                    pts = hidden[0, :, :]
                elif hidden.ndim == 2:
                    pts = hidden
                else:
                    pts = self.backend.reshape(hidden, (1, -1))

                # Need at least 4 points for TwoNN
                if pts.shape[0] < 4:
                    id_trajectory.append(float("nan"))
                    continue

                try:
                    estimate = estimator.compute(pts)
                    id_trajectory.append(estimate.intrinsic_dimension)
                except Exception:
                    id_trajectory.append(float("nan"))

            valid_ids = [d for d in id_trajectory if d == d]  # Filter NaN
            if not valid_ids:
                return {"error": "All ID estimates failed"}

            mean_id = sum(valid_ids) / len(valid_ids)
            final_id = id_trajectory[-1] if id_trajectory[-1] == id_trajectory[-1] else mean_id
            peak_id = max(valid_ids)
            id_expansion_ratio = peak_id / final_id if final_id > 0 else 0.0

            return {
                "intrinsic_dimension_trajectory": id_trajectory,
                "mean_intrinsic_dimension": mean_id,
                "final_intrinsic_dimension": final_id,
                "id_expansion_ratio": id_expansion_ratio,
            }
        except Exception as e:
            return {"error": str(e)}

    def _compute_spectral_entropy_trajectory(self, text: str) -> dict[str, Any]:
        """Compute layer-wise spectral entropy."""
        from modelcypher.core.domain.entropy.layer_entropy_projector import (
            LayerEntropyProjector,
        )
        from modelcypher.core.domain.geometry.effective_rank import EffectiveRank

        try:
            projector = LayerEntropyProjector(self.backend)

            base_model = getattr(self.model, "model", self.model)
            layers = getattr(base_model, "layers", None)
            if layers is None:
                return {"error": "Could not find model layers"}

            num_layers = len(layers)
            target_layers = set(range(num_layers))

            tokens = self.tokenizer.encode(text)
            if isinstance(tokens, list):
                input_ids = self.backend.array([tokens])
            else:
                input_ids = tokens
                if input_ids.ndim == 1:
                    input_ids = self.backend.reshape(input_ids, (1, -1))

            captured = projector._capture_layer_states(
                base_model, layers, input_ids, target_layers
            )

            ranker = EffectiveRank(self.backend)
            spectral_trajectory = []

            for layer_idx in sorted(captured.keys()):
                hidden = captured[layer_idx]
                if hidden.ndim == 3:
                    pts = hidden[0, :, :]
                elif hidden.ndim == 2:
                    pts = hidden
                else:
                    pts = self.backend.reshape(hidden, (1, -1))

                try:
                    result = ranker.compute(pts)
                    spectral_trajectory.append(result.spectral_entropy)
                except Exception:
                    spectral_trajectory.append(float("nan"))

            valid = [s for s in spectral_trajectory if s == s]
            if not valid:
                return {"error": "All spectral entropy estimates failed"}

            return {
                "spectral_entropy_trajectory": spectral_trajectory,
                "mean_spectral_entropy": sum(valid) / len(valid),
                "final_spectral_entropy": spectral_trajectory[-1]
                if spectral_trajectory[-1] == spectral_trajectory[-1]
                else sum(valid) / len(valid),
            }
        except Exception as e:
            return {"error": str(e)}

    def _compute_reasoning_flow(self, text: str) -> dict[str, Any]:
        """Compute reasoning flow metrics (curvature, arc length, etc.)."""
        from modelcypher.core.domain.geometry.reasoning_flow import (
            ReasoningFlowAnalyzer,
        )

        try:
            # Get final layer embeddings for token trajectory
            base_model = getattr(self.model, "model", self.model)
            layers = getattr(base_model, "layers", None)
            if layers is None:
                return {"error": "Could not find model layers"}

            tokens = self.tokenizer.encode(text)
            if isinstance(tokens, list):
                input_ids = self.backend.array([tokens])
            else:
                input_ids = tokens
                if input_ids.ndim == 1:
                    input_ids = self.backend.reshape(input_ids, (1, -1))

            # Get final layer hidden states
            # This requires a forward pass - use the projector's capture method
            from modelcypher.core.domain.entropy.layer_entropy_projector import (
                LayerEntropyProjector,
            )

            projector = LayerEntropyProjector(self.backend)
            final_layer = len(layers) - 1
            captured = projector._capture_layer_states(
                base_model, layers, input_ids, {final_layer}
            )

            hidden = captured[final_layer]
            if hidden.ndim == 3:
                positions = hidden[0, :, :]  # [T, D]
            else:
                positions = hidden

            if positions.shape[0] < 3:
                return {"error": "Not enough tokens for flow analysis"}

            analyzer = ReasoningFlowAnalyzer(self.backend)
            metrics = analyzer.analyze_flow(positions)

            return {
                "arc_length": metrics.arc_length,
                "mean_curvature": metrics.mean_curvature,
                "max_curvature": metrics.max_curvature,
                "smoothness": metrics.smoothness,
                "directness": metrics.directness,
                "velocity_variance": metrics.velocity_variance,
            }
        except Exception as e:
            return {"error": str(e)}

    def _check_answer(self, response: str, expected: str, choices: list[str] | None) -> bool:
        """Check if response contains expected answer."""
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        # Direct containment
        if expected_lower in response_lower:
            return True

        # Multiple choice: check if letter is selected
        if choices:
            for i, choice in enumerate(choices):
                if choice.lower() == expected_lower:
                    letter = chr(65 + i)
                    if letter.lower() in response_lower[:5]:
                        return True

        return False

    def run_sample(
        self, sample_id: str, benchmark: str, prompt: str, expected: str, choices: list[str] | None
    ) -> GeometricSample:
        """Run a single sample, collecting all geometric metrics."""
        logger.debug(f"Processing sample {sample_id}")

        # Generate response
        response = self._generate(prompt)
        is_correct = self._check_answer(response, expected, choices)

        # Collect geometric metrics (on prompt, before answer)
        # This is critical: we want to predict correctness BEFORE seeing the answer
        entropy_metrics = self._compute_entropy_trajectory(prompt)
        id_metrics = self._compute_intrinsic_dimension_trajectory(prompt)
        spectral_metrics = self._compute_spectral_entropy_trajectory(prompt)

        # Reasoning flow is computed on the full response trajectory
        full_text = prompt + response
        flow_metrics = self._compute_reasoning_flow(full_text)

        # Aggregate errors
        errors = []
        for metrics in [entropy_metrics, id_metrics, spectral_metrics, flow_metrics]:
            if "error" in metrics:
                errors.append(metrics["error"])

        # Build result
        result = GeometricSample(
            sample_id=sample_id,
            benchmark=benchmark,
            prompt=prompt,
            expected_answer=expected,
            model_response=response,
            is_correct=is_correct,
            error="; ".join(errors) if errors else None,
        )

        # Populate metrics
        if "entropy_trajectory" in entropy_metrics:
            result.entropy_trajectory = entropy_metrics["entropy_trajectory"]
            result.initial_entropy = entropy_metrics["initial_entropy"]
            result.peak_entropy = entropy_metrics["peak_entropy"]
            result.final_entropy = entropy_metrics["final_entropy"]
            result.peak_layer = entropy_metrics["peak_layer"]
            result.expansion_rate = entropy_metrics["expansion_rate"]
            result.compression_rate = entropy_metrics["compression_rate"]
            result.expansion_ratio = entropy_metrics["expansion_ratio"]

        if "intrinsic_dimension_trajectory" in id_metrics:
            result.intrinsic_dimension_trajectory = id_metrics["intrinsic_dimension_trajectory"]
            result.mean_intrinsic_dimension = id_metrics["mean_intrinsic_dimension"]
            result.final_intrinsic_dimension = id_metrics["final_intrinsic_dimension"]
            result.id_expansion_ratio = id_metrics["id_expansion_ratio"]

        if "spectral_entropy_trajectory" in spectral_metrics:
            result.spectral_entropy_trajectory = spectral_metrics["spectral_entropy_trajectory"]
            result.mean_spectral_entropy = spectral_metrics["mean_spectral_entropy"]
            result.final_spectral_entropy = spectral_metrics["final_spectral_entropy"]

        if "arc_length" in flow_metrics:
            result.arc_length = flow_metrics["arc_length"]
            result.mean_curvature = flow_metrics["mean_curvature"]
            result.max_curvature = flow_metrics["max_curvature"]
            result.smoothness = flow_metrics["smoothness"]
            result.directness = flow_metrics["directness"]
            result.velocity_variance = flow_metrics["velocity_variance"]

        return result

    def run_benchmark(self, benchmark_name: str) -> list[GeometricSample]:
        """Run experiment on a single benchmark."""
        from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader

        logger.info(f"Running benchmark: {benchmark_name}")

        loader = BenchmarkLoader()
        benchmark = loader.load(
            benchmark_name, split="test", limit=self.config.samples_per_benchmark
        )

        results = []
        for i, sample in enumerate(benchmark.samples):
            sample_id = f"{benchmark_name}_{i:04d}"
            result = self.run_sample(
                sample_id=sample_id,
                benchmark=benchmark_name,
                prompt=sample.prompt,
                expected=sample.answer,
                choices=sample.choices,
            )
            results.append(result)

            # Progress logging
            if (i + 1) % 10 == 0:
                correct_so_far = sum(1 for r in results if r.is_correct)
                logger.info(
                    f"  {benchmark_name}: {i+1}/{len(benchmark.samples)} "
                    f"(accuracy so far: {correct_so_far}/{i+1} = {correct_so_far/(i+1):.1%})"
                )

        return results

    def run(self) -> None:
        """Run the full experiment."""
        logger.info("Starting geometry validation experiment")
        logger.info(f"Config: {self.config}")

        self.setup()

        for benchmark_name in self.config.benchmarks:
            try:
                results = self.run_benchmark(benchmark_name)
                self.results.extend(results)
            except Exception as e:
                logger.error(f"Failed on {benchmark_name}: {e}")

        self.save_results()
        self.compute_summary()

    def save_results(self) -> None:
        """Save all results to JSONL."""
        output_path = self.config.output_dir / "samples.jsonl"
        with open(output_path, "w") as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict()) + "\n")
        logger.info(f"Saved {len(self.results)} samples to {output_path}")

    def compute_summary(self) -> None:
        """Compute and save summary statistics."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        correct = [r for r in self.results if r.is_correct]
        incorrect = [r for r in self.results if not r.is_correct]

        # Per-metric statistics for correct vs incorrect
        metrics = [
            "expansion_ratio",
            "mean_intrinsic_dimension",
            "id_expansion_ratio",
            "mean_spectral_entropy",
            "mean_curvature",
            "smoothness",
            "directness",
        ]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.config.model_path,
            "total_samples": len(self.results),
            "correct_count": len(correct),
            "incorrect_count": len(incorrect),
            "overall_accuracy": len(correct) / len(self.results) if self.results else 0,
            "per_benchmark": {},
            "metric_distributions": {},
        }

        # Per-benchmark accuracy
        benchmarks = set(r.benchmark for r in self.results)
        for benchmark in benchmarks:
            bench_results = [r for r in self.results if r.benchmark == benchmark]
            bench_correct = sum(1 for r in bench_results if r.is_correct)
            summary["per_benchmark"][benchmark] = {
                "total": len(bench_results),
                "correct": bench_correct,
                "accuracy": bench_correct / len(bench_results) if bench_results else 0,
            }

        # Metric distributions
        for metric in metrics:
            correct_vals = [getattr(r, metric, 0.0) for r in correct if hasattr(r, metric)]
            incorrect_vals = [getattr(r, metric, 0.0) for r in incorrect if hasattr(r, metric)]

            # Filter NaN
            correct_vals = [v for v in correct_vals if v == v]
            incorrect_vals = [v for v in incorrect_vals if v == v]

            if correct_vals and incorrect_vals:
                summary["metric_distributions"][metric] = {
                    "correct": {
                        "mean": sum(correct_vals) / len(correct_vals),
                        "min": min(correct_vals),
                        "max": max(correct_vals),
                        "n": len(correct_vals),
                    },
                    "incorrect": {
                        "mean": sum(incorrect_vals) / len(incorrect_vals),
                        "min": min(incorrect_vals),
                        "max": max(incorrect_vals),
                        "n": len(incorrect_vals),
                    },
                }

        # Save summary
        summary_path = self.config.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total samples: {summary['total_samples']}")
        print(f"Overall accuracy: {summary['overall_accuracy']:.1%}")
        print()
        print("Per-benchmark accuracy:")
        for bench, stats in summary["per_benchmark"].items():
            print(f"  {bench}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
        print()
        print("Metric distributions (correct vs incorrect):")
        for metric, dist in summary["metric_distributions"].items():
            c = dist["correct"]
            i = dist["incorrect"]
            diff = c["mean"] - i["mean"]
            print(f"  {metric}:")
            print(f"    Correct:   mean={c['mean']:.4f}, range=[{c['min']:.4f}, {c['max']:.4f}]")
            print(f"    Incorrect: mean={i['mean']:.4f}, range=[{i['min']:.4f}, {i['max']:.4f}]")
            print(f"    Diff: {diff:+.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run geometry validation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--output",
        default="results/geometry_validation/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k", "arc_challenge", "hellaswag", "boolq"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples per benchmark",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        model_path=args.model,
        output_dir=Path(args.output),
        benchmarks=args.benchmarks,
        samples_per_benchmark=args.samples,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    experiment = GeometryValidationExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
