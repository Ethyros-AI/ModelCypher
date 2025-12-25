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

"""Training performance benchmarking.

Provides metrics collection and comparison for training optimization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkResults:
    """Results from a training benchmark."""

    total_duration: float
    """Total training duration (seconds)."""

    total_steps: int
    """Total number of training steps."""

    total_tokens: int
    """Total tokens processed."""

    tokens_per_second: float
    """Average throughput (tokens/second)."""

    average_step_latency: float
    """Average time per step (seconds)."""

    peak_memory_usage_gb: float
    """Peak memory usage (GB)."""

    throughput_score: float
    """Composite throughput score (accounts for memory efficiency)."""

    def compare(self, other: BenchmarkResults) -> BenchmarkComparison:
        """Compare this result to another (for A/B testing optimizations).

        Args:
            other: Baseline results to compare against.

        Returns:
            Performance comparison.
        """
        speedup_factor = self.tokens_per_second / max(other.tokens_per_second, 0.001)
        memory_reduction = 1.0 - (
            self.peak_memory_usage_gb / max(other.peak_memory_usage_gb, 0.001)
        )
        latency_reduction = 1.0 - (
            self.average_step_latency / max(other.average_step_latency, 0.001)
        )

        return BenchmarkComparison(
            speedup_factor=speedup_factor,
            memory_reduction=memory_reduction,
            latency_reduction=latency_reduction,
            baseline=other,
            optimized=self,
        )


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison between baseline and optimized training jobs."""

    speedup_factor: float
    """Speed improvement factor (e.g., 2.0 = 2x faster)."""

    memory_reduction: float
    """Memory reduction (e.g., 0.3 = 30% less memory)."""

    latency_reduction: float
    """Step latency reduction (e.g., 0.2 = 20% faster steps)."""

    baseline: BenchmarkResults
    """Baseline results."""

    optimized: BenchmarkResults
    """Optimized results."""

    @property
    def speedup_percent(self) -> int:
        """Speedup as percentage."""
        return int((self.speedup_factor - 1.0) * 100)

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Performance Comparison (Optimized vs. Baseline)
───────────────────────────────────────────────────────────
Speedup:            {self.speedup_factor:.2f}x ({self.speedup_percent}% faster)
Memory Reduction:   {self.memory_reduction * 100:.1f}%
Latency Reduction:  {self.latency_reduction * 100:.1f}%
───────────────────────────────────────────────────────────
Baseline:  {self.baseline.tokens_per_second:.2f} tok/s, {self.baseline.peak_memory_usage_gb:.2f} GB
Optimized: {self.optimized.tokens_per_second:.2f} tok/s, {self.optimized.peak_memory_usage_gb:.2f} GB
"""


class TrainingBenchmark:
    """Training performance benchmarking.

    Collects metrics during training for performance analysis and optimization.
    """

    def __init__(self):
        """Initialize benchmark."""
        self._start_time: float | None = None
        self._total_tokens: int = 0
        self._total_steps: int = 0
        self._step_latencies: list[float] = []
        self._peak_memory_usage: int = 0

    def start(self) -> None:
        """Start benchmarking."""
        self._start_time = time.time()
        self._total_tokens = 0
        self._total_steps = 0
        self._step_latencies = []
        self._peak_memory_usage = 0
        logger.info("Benchmark started")

    def record_step(
        self,
        tokens: int,
        latency: float,
        memory_usage: int,
    ) -> None:
        """Record a training step.

        Args:
            tokens: Number of tokens processed in this step.
            latency: Time taken for this step (seconds).
            memory_usage: Current memory usage (bytes).
        """
        self._total_tokens += tokens
        self._total_steps += 1
        self._step_latencies.append(latency)
        self._peak_memory_usage = max(self._peak_memory_usage, memory_usage)

    def results(self) -> BenchmarkResults:
        """Get benchmark results.

        Returns:
            Benchmark results with computed metrics.
        """
        if self._start_time is None:
            return BenchmarkResults(
                total_duration=0,
                total_steps=0,
                total_tokens=0,
                tokens_per_second=0,
                average_step_latency=0,
                peak_memory_usage_gb=0,
                throughput_score=0,
            )

        duration = time.time() - self._start_time
        tokens_per_second = self._total_tokens / duration if duration > 0 else 0

        avg_latency = 0.0
        if self._step_latencies:
            avg_latency = sum(self._step_latencies) / len(self._step_latencies)

        # Use decimal GB (SI units)
        peak_memory_gb = self._peak_memory_usage / 1_000_000_000.0

        # Throughput score: normalized metric for comparisons
        # Score = tokens/second * (1 - memory_penalty)
        memory_penalty = min(peak_memory_gb / 128.0, 1.0) * 0.2  # 20% penalty at 128GB
        throughput_score = tokens_per_second * (1.0 - memory_penalty)

        return BenchmarkResults(
            total_duration=duration,
            total_steps=self._total_steps,
            total_tokens=self._total_tokens,
            tokens_per_second=tokens_per_second,
            average_step_latency=avg_latency,
            peak_memory_usage_gb=peak_memory_gb,
            throughput_score=throughput_score,
        )

    def formatted_summary(self) -> str:
        """Get a formatted summary string of benchmark results.

        Returns:
            Formatted summary string.
        """
        results = self.results()
        return f"""
═══════════════════════════════════════════════════════════
ModelCypher Training Benchmark Results
═══════════════════════════════════════════════════════════
Duration:           {results.total_duration:.2f}s
Steps:              {results.total_steps}
Tokens:             {results.total_tokens}
Throughput:         {results.tokens_per_second:.2f} tok/s
Avg Step Latency:   {results.average_step_latency * 1000:.3f}ms
Peak Memory:        {results.peak_memory_usage_gb:.2f} GB
Throughput Score:   {results.throughput_score:.2f}
═══════════════════════════════════════════════════════════
"""

    def log_summary(self) -> None:
        """Log a summary of benchmark results."""
        summary = self.formatted_summary()
        logger.info(summary)
