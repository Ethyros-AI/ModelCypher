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

"""Tests for training_benchmark module."""

import pytest

from modelcypher.core.domain.training.training_benchmark import (
    BenchmarkResults,
    TrainingBenchmark,
)


class TestBenchmarkResults:
    """Tests for BenchmarkResults dataclass."""

    def test_compare_with_faster_optimized(self):
        """Test comparison when optimized is faster."""
        baseline = BenchmarkResults(
            total_duration=100.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=100.0,
            average_step_latency=1.0,
            peak_memory_usage_gb=16.0,
            throughput_score=95.0,
        )

        optimized = BenchmarkResults(
            total_duration=50.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=200.0,
            average_step_latency=0.5,
            peak_memory_usage_gb=12.0,
            throughput_score=196.0,
        )

        comparison = optimized.compare(baseline)

        assert comparison.speedup_factor == 2.0
        assert comparison.memory_reduction == 0.25  # 25% reduction
        assert comparison.latency_reduction == 0.5  # 50% faster
        assert comparison.baseline is baseline
        assert comparison.optimized is optimized

    def test_compare_with_slower_result(self):
        """Test comparison when 'optimized' is actually slower."""
        baseline = BenchmarkResults(
            total_duration=50.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=200.0,
            average_step_latency=0.5,
            peak_memory_usage_gb=12.0,
            throughput_score=196.0,
        )

        slower = BenchmarkResults(
            total_duration=100.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=100.0,
            average_step_latency=1.0,
            peak_memory_usage_gb=16.0,
            throughput_score=95.0,
        )

        comparison = slower.compare(baseline)

        assert comparison.speedup_factor == 0.5  # Half the speed
        assert comparison.memory_reduction < 0  # Memory increased
        assert comparison.latency_reduction < 0  # Latency increased

    def test_compare_handles_zero_baseline_uses_floor_value(self):
        """Test that comparison uses floor value (0.001) to prevent division by zero."""
        baseline = BenchmarkResults(
            total_duration=0.001,
            total_steps=0,
            total_tokens=0,
            tokens_per_second=0.0,  # Zero throughput - would cause div by zero
            average_step_latency=0.0,  # Zero latency - would cause div by zero
            peak_memory_usage_gb=0.0,  # Zero memory - would cause div by zero
            throughput_score=0.0,
        )

        optimized = BenchmarkResults(
            total_duration=50.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=200.0,
            average_step_latency=0.5,
            peak_memory_usage_gb=12.0,
            throughput_score=196.0,
        )

        comparison = optimized.compare(baseline)

        # Implementation uses max(x, 0.001) for division protection
        # speedup = 200 / max(0, 0.001) = 200 / 0.001 = 200000
        assert comparison.speedup_factor == pytest.approx(200000.0, rel=0.01)
        # memory_reduction = 1 - (12 / max(0, 0.001)) = 1 - 12000 = -11999
        assert comparison.memory_reduction == pytest.approx(-11999.0, rel=0.01)
        # latency_reduction = 1 - (0.5 / max(0, 0.001)) = 1 - 500 = -499
        assert comparison.latency_reduction == pytest.approx(-499.0, rel=0.01)


class TestBenchmarkComparison:
    """Tests for BenchmarkComparison dataclass."""

    def test_summary_contains_computed_values(self):
        """Test that summary contains the actual computed metric values."""
        baseline = BenchmarkResults(
            total_duration=100.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=100.0,
            average_step_latency=1.0,
            peak_memory_usage_gb=16.0,
            throughput_score=95.0,
        )

        optimized = BenchmarkResults(
            total_duration=50.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=200.0,
            average_step_latency=0.5,
            peak_memory_usage_gb=12.0,
            throughput_score=196.0,
        )

        comparison = optimized.compare(baseline)
        summary = comparison.summary

        # Verify actual computed values appear in summary, not just labels
        assert "2.00x" in summary  # speedup_factor
        assert "100%" in summary  # speedup_percent
        assert "25.0%" in summary  # memory_reduction (25%)
        assert "50.0%" in summary  # latency_reduction (50%)
        assert "100.00 tok/s" in summary  # baseline throughput
        assert "200.00 tok/s" in summary  # optimized throughput
        assert "16.00 GB" in summary  # baseline memory
        assert "12.00 GB" in summary  # optimized memory


class TestTrainingBenchmark:
    """Tests for TrainingBenchmark class."""

    def test_start_resets_state(self):
        """Test that start() resets all counters."""
        benchmark = TrainingBenchmark()

        # Record some steps
        benchmark.start()
        benchmark.record_step(tokens=100, latency=0.1, memory_usage=1_000_000_000)
        benchmark.record_step(tokens=100, latency=0.1, memory_usage=1_000_000_000)

        # Reset
        benchmark.start()
        results = benchmark.results()

        # Duration will be non-zero since we just started
        assert results.total_steps == 0
        assert results.total_tokens == 0

    def test_record_step_accumulates(self):
        """Test that record_step accumulates metrics correctly."""
        benchmark = TrainingBenchmark()
        benchmark.start()

        benchmark.record_step(tokens=100, latency=0.1, memory_usage=1_000_000_000)
        benchmark.record_step(tokens=200, latency=0.2, memory_usage=2_000_000_000)
        benchmark.record_step(tokens=300, latency=0.3, memory_usage=1_500_000_000)

        results = benchmark.results()

        assert results.total_steps == 3
        assert results.total_tokens == 600
        assert results.peak_memory_usage_gb == 2.0

    def test_average_step_latency(self):
        """Test average step latency calculation."""
        benchmark = TrainingBenchmark()
        benchmark.start()

        benchmark.record_step(tokens=100, latency=0.1, memory_usage=1_000_000_000)
        benchmark.record_step(tokens=100, latency=0.2, memory_usage=1_000_000_000)
        benchmark.record_step(tokens=100, latency=0.3, memory_usage=1_000_000_000)

        results = benchmark.results()

        assert abs(results.average_step_latency - 0.2) < 0.001  # (0.1+0.2+0.3)/3

    def test_throughput_score_memory_penalty_formula_at_128gb(self):
        """Test that 128GB memory applies exactly 20% penalty as per formula."""
        benchmark = TrainingBenchmark()
        benchmark.start()

        # Record a step with exactly 128GB memory usage
        benchmark.record_step(
            tokens=10000,
            latency=0.1,
            memory_usage=128_000_000_000,  # 128GB exactly
        )

        results = benchmark.results()

        # Formula: penalty = min(peak_gb/128, 1.0) * 0.2
        # At 128GB: penalty = min(128/128, 1.0) * 0.2 = 0.2
        # Score = tokens_per_second * (1 - 0.2) = tokens_per_second * 0.8
        expected_ratio = 0.8
        actual_ratio = results.throughput_score / results.tokens_per_second
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.001)

    def test_throughput_score_with_low_memory(self):
        """Test throughput score with low memory usage."""
        benchmark = TrainingBenchmark()
        benchmark.start()

        # Record a step with low memory usage
        benchmark.record_step(
            tokens=10000,
            latency=0.1,
            memory_usage=1_000_000_000,  # 1GB
        )

        results = benchmark.results()

        # Score should be close to tokens_per_second with low memory
        # Memory penalty: (1/128) * 0.2 = 0.00156 -> 99.84% of throughput
        ratio = results.throughput_score / results.tokens_per_second
        assert ratio > 0.99

    def test_formatted_summary_contains_recorded_values(self):
        """Test formatted summary contains the actual recorded metrics."""
        benchmark = TrainingBenchmark()
        benchmark.start()
        benchmark.record_step(tokens=1000, latency=0.1, memory_usage=8_000_000_000)

        summary = benchmark.formatted_summary()

        # Verify recorded data appears in summary
        assert "Steps:              1" in summary
        assert "Tokens:             1000" in summary
        assert "8.00 GB" in summary  # Peak memory

    def test_peak_memory_tracking(self):
        """Test that peak memory is correctly tracked across steps."""
        benchmark = TrainingBenchmark()
        benchmark.start()

        benchmark.record_step(tokens=100, latency=0.1, memory_usage=2_000_000_000)
        benchmark.record_step(tokens=100, latency=0.1, memory_usage=4_000_000_000)
        benchmark.record_step(tokens=100, latency=0.1, memory_usage=3_000_000_000)
        benchmark.record_step(tokens=100, latency=0.1, memory_usage=1_000_000_000)

        results = benchmark.results()

        assert results.peak_memory_usage_gb == 4.0
