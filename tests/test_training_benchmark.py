"""Tests for training_benchmark module."""

import time
import pytest

from modelcypher.core.domain.training.training_benchmark import (
    BenchmarkResults,
    BenchmarkComparison,
    TrainingBenchmark,
)


class TestBenchmarkResults:
    """Tests for BenchmarkResults dataclass."""

    def test_basic_initialization(self):
        """Test that BenchmarkResults initializes correctly."""
        results = BenchmarkResults(
            total_duration=10.0,
            total_steps=100,
            total_tokens=50000,
            tokens_per_second=5000.0,
            average_step_latency=0.1,
            peak_memory_usage_gb=8.0,
            throughput_score=4800.0,
        )

        assert results.total_duration == 10.0
        assert results.total_steps == 100
        assert results.total_tokens == 50000
        assert results.tokens_per_second == 5000.0
        assert results.average_step_latency == 0.1
        assert results.peak_memory_usage_gb == 8.0
        assert results.throughput_score == 4800.0

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

    def test_compare_handles_zero_baseline(self):
        """Test that comparison handles near-zero baseline values gracefully."""
        baseline = BenchmarkResults(
            total_duration=0.001,
            total_steps=0,
            total_tokens=0,
            tokens_per_second=0.0,  # Zero throughput
            average_step_latency=0.0,
            peak_memory_usage_gb=0.0,
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

        # Should not raise division by zero
        comparison = optimized.compare(baseline)
        assert comparison.speedup_factor > 0


class TestBenchmarkComparison:
    """Tests for BenchmarkComparison dataclass."""

    def test_speedup_percent_calculation(self):
        """Test speedup_percent property."""
        baseline = BenchmarkResults(
            total_duration=100.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=100.0,
            average_step_latency=1.0,
            peak_memory_usage_gb=16.0,
            throughput_score=95.0,
        )

        comparison = BenchmarkComparison(
            speedup_factor=1.5,  # 50% faster
            memory_reduction=0.2,
            latency_reduction=0.3,
            baseline=baseline,
            optimized=baseline,  # Placeholder
        )

        assert comparison.speedup_percent == 50

    def test_speedup_percent_with_slowdown(self):
        """Test speedup_percent when there's actually a slowdown."""
        baseline = BenchmarkResults(
            total_duration=100.0,
            total_steps=100,
            total_tokens=10000,
            tokens_per_second=100.0,
            average_step_latency=1.0,
            peak_memory_usage_gb=16.0,
            throughput_score=95.0,
        )

        comparison = BenchmarkComparison(
            speedup_factor=0.8,  # 20% slower
            memory_reduction=-0.1,
            latency_reduction=-0.2,
            baseline=baseline,
            optimized=baseline,
        )

        assert comparison.speedup_percent == -20

    def test_summary_format(self):
        """Test that summary property produces readable output."""
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

        assert "Performance Comparison" in summary
        assert "Speedup" in summary
        assert "Memory Reduction" in summary
        assert "Latency Reduction" in summary
        assert "Baseline" in summary
        assert "Optimized" in summary


class TestTrainingBenchmark:
    """Tests for TrainingBenchmark class."""

    def test_initial_state(self):
        """Test initial benchmark state before start."""
        benchmark = TrainingBenchmark()
        results = benchmark.results()

        assert results.total_duration == 0
        assert results.total_steps == 0
        assert results.total_tokens == 0

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

    def test_tokens_per_second_calculation(self):
        """Test throughput calculation based on duration."""
        benchmark = TrainingBenchmark()
        benchmark.start()

        # Simulate some work
        time.sleep(0.1)
        benchmark.record_step(tokens=1000, latency=0.1, memory_usage=1_000_000_000)

        results = benchmark.results()

        # Tokens per second should be roughly 1000/0.1 = 10000
        # But actual duration is slightly more than 0.1s
        assert results.tokens_per_second > 0
        assert results.tokens_per_second < 20000  # Sanity check

    def test_throughput_score_with_memory_penalty(self):
        """Test that throughput score applies memory penalty."""
        benchmark = TrainingBenchmark()
        benchmark.start()

        # Record a step with high memory usage
        benchmark.record_step(
            tokens=10000,
            latency=0.1,
            memory_usage=128_000_000_000,  # 128GB
        )

        results = benchmark.results()

        # Score should be less than tokens_per_second due to memory penalty
        assert results.throughput_score < results.tokens_per_second

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

    def test_formatted_summary(self):
        """Test formatted summary output."""
        benchmark = TrainingBenchmark()
        benchmark.start()
        benchmark.record_step(tokens=1000, latency=0.1, memory_usage=8_000_000_000)

        summary = benchmark.formatted_summary()

        assert "ModelCypher Training Benchmark" in summary
        assert "Duration" in summary
        assert "Steps" in summary
        assert "Tokens" in summary
        assert "Throughput" in summary
        assert "Peak Memory" in summary

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
