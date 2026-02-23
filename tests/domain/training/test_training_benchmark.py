# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.training_benchmark import (
    BenchmarkResults,
    TrainingBenchmark,
)


class TestTrainingBenchmarkNotStarted:
    """Results before start() — all zeros, no crash."""

    def test_results_before_start(self):
        """Calling results() without start() returns zero state."""
        bench = TrainingBenchmark()
        r = bench.results()
        assert r.total_duration == 0
        assert r.total_steps == 0
        assert r.total_tokens == 0
        assert r.tokens_per_second == 0
        assert r.average_step_latency == 0
        assert r.peak_memory_usage_gb == 0
        assert r.throughput_score == 0


class TestTrainingBenchmarkRecording:
    """Tests for record_step and results computation."""

    def test_accumulates_tokens_and_steps(self):
        """record_step adds tokens and increments step count."""
        bench = TrainingBenchmark()
        bench.start()
        bench.record_step(tokens=100, latency=0.1, memory_usage=1_000_000_000)
        bench.record_step(tokens=200, latency=0.2, memory_usage=2_000_000_000)
        bench.record_step(tokens=150, latency=0.15, memory_usage=1_500_000_000)

        r = bench.results()
        assert r.total_steps == 3
        assert r.total_tokens == 450

    def test_peak_memory_tracks_max(self):
        """peak_memory_usage_gb is the maximum across all steps."""
        bench = TrainingBenchmark()
        bench.start()
        bench.record_step(tokens=100, latency=0.1, memory_usage=1_000_000_000)
        bench.record_step(tokens=100, latency=0.1, memory_usage=4_000_000_000)
        bench.record_step(tokens=100, latency=0.1, memory_usage=2_000_000_000)

        r = bench.results()
        assert r.peak_memory_usage_gb == pytest.approx(4.0)

    def test_memory_bytes_to_gb_si_units(self):
        """Memory uses SI units: 1 GB = 1,000,000,000 bytes."""
        bench = TrainingBenchmark()
        bench.start()
        bench.record_step(tokens=100, latency=0.1, memory_usage=500_000_000)

        r = bench.results()
        assert r.peak_memory_usage_gb == pytest.approx(0.5)

    def test_average_step_latency(self):
        """Average latency is mean of recorded latencies."""
        bench = TrainingBenchmark()
        bench.start()
        bench.record_step(tokens=100, latency=0.1, memory_usage=0)
        bench.record_step(tokens=100, latency=0.3, memory_usage=0)

        r = bench.results()
        assert r.average_step_latency == pytest.approx(0.2)

    def test_tokens_per_second_positive(self):
        """tokens_per_second is positive after recording tokens."""
        bench = TrainingBenchmark()
        bench.start()
        bench.record_step(tokens=1000, latency=0.1, memory_usage=0)

        r = bench.results()
        assert r.tokens_per_second > 0

    def test_start_resets_state(self):
        """Calling start() again resets all accumulated state."""
        bench = TrainingBenchmark()
        bench.start()
        bench.record_step(tokens=500, latency=0.1, memory_usage=10_000_000_000)

        bench.start()  # Reset
        r = bench.results()
        assert r.total_tokens == 0
        assert r.total_steps == 0
        assert r.peak_memory_usage_gb == 0


class TestBenchmarkComparison:
    """Tests for BenchmarkResults.compare() and BenchmarkComparison."""

    def _make_results(
        self, tps: float = 100.0, mem_gb: float = 4.0, latency: float = 0.01,
    ) -> BenchmarkResults:
        return BenchmarkResults(
            total_duration=10.0,
            total_steps=100,
            total_tokens=1000,
            tokens_per_second=tps,
            average_step_latency=latency,
            peak_memory_usage_gb=mem_gb,
            throughput_score=tps * 0.9,
        )

    def test_speedup_factor_2x(self):
        """Optimized 2x faster -> speedup_factor=2.0."""
        baseline = self._make_results(tps=100.0)
        optimized = self._make_results(tps=200.0)
        cmp = optimized.compare(baseline)
        assert cmp.speedup_factor == pytest.approx(2.0)

    def test_speedup_percent(self):
        """2x faster -> 100% speedup."""
        baseline = self._make_results(tps=100.0)
        optimized = self._make_results(tps=200.0)
        cmp = optimized.compare(baseline)
        assert cmp.speedup_percent == 100

    def test_memory_reduction(self):
        """Half memory -> 50% reduction."""
        baseline = self._make_results(mem_gb=8.0)
        optimized = self._make_results(mem_gb=4.0)
        cmp = optimized.compare(baseline)
        assert cmp.memory_reduction == pytest.approx(0.5)

    def test_latency_reduction(self):
        """Half latency -> 50% reduction."""
        baseline = self._make_results(latency=0.02)
        optimized = self._make_results(latency=0.01)
        cmp = optimized.compare(baseline)
        assert cmp.latency_reduction == pytest.approx(0.5)

    def test_zero_baseline_protected(self):
        """Zero baseline → inf speedup, 0.0 for undefined reductions."""
        baseline = self._make_results(tps=0.0, mem_gb=0.0, latency=0.0)
        optimized = self._make_results(tps=100.0, mem_gb=4.0, latency=0.01)
        cmp = optimized.compare(baseline)
        assert cmp.speedup_factor == float("inf")
        assert cmp.memory_reduction == 0.0
        assert cmp.latency_reduction == 0.0

    def test_summary_contains_key_metrics(self):
        """Summary string includes speedup factor and percent."""
        baseline = self._make_results(tps=100.0, mem_gb=8.0)
        optimized = self._make_results(tps=150.0, mem_gb=6.0)
        cmp = optimized.compare(baseline)
        s = cmp.summary
        assert "1.50x" in s
        assert "50%" in s
