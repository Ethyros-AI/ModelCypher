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

"""Tests for dual-path inference dataclasses and entropy calculations.

These tests verify the dataclass configurations and entropy calculation logic
for the dual-path generator.
"""

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


class TestSecurityScanMetrics:
    """Tests for SecurityScanMetrics dataclass."""

    def test_metrics_creation(self):
        """SecurityScanMetrics can be created with all fields."""
        from modelcypher.infrastructure.dual_path_mlx import SecurityScanMetrics

        metrics = SecurityScanMetrics(
            token_count=100,
            time_to_first_token_ms=50.5,
            total_time_ms=1000.0,
            tokens_per_second=100.0,
        )

        assert metrics.token_count == 100
        assert abs(metrics.time_to_first_token_ms - 50.5) <= _eps()
        assert abs(metrics.total_time_ms - 1000.0) <= _eps()
        assert abs(metrics.tokens_per_second - 100.0) <= _eps()


class TestEntropyDeltaSample:
    """Tests for EntropyDeltaSample dataclass."""

    def test_sample_creation(self):
        """EntropyDeltaSample can be created with all fields."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaSample

        sample = EntropyDeltaSample(
            token_index=0,
            generated_token=12345,
            base_entropy=2.5,
            base_logit_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.8,
            adapter_logit_variance=0.15,
            adapter_top_token=200,
            latency_ms=5.0,
        )

        assert sample.token_index == 0
        assert sample.generated_token == 12345
        assert abs(sample.base_entropy - 2.5) <= _eps()
        assert abs(sample.adapter_entropy - 2.8) <= _eps()
        assert sample.base_top_token == 100
        assert sample.adapter_top_token == 200

    def test_sample_delta_property(self):
        """EntropyDeltaSample computes delta correctly."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaSample

        sample = EntropyDeltaSample(
            token_index=0,
            generated_token=100,
            base_entropy=3.0,
            base_logit_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.0,
            adapter_logit_variance=0.15,
            adapter_top_token=100,
            latency_ms=5.0,
        )

        # delta = base - adapter
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        assert abs(sample.delta - 1.0) <= eps * max(1.0, abs(sample.delta))

    def test_sample_top_token_disagreement(self):
        """EntropyDeltaSample detects top token disagreement."""
        from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaSample

        # Same top token
        same = EntropyDeltaSample(
            token_index=0,
            generated_token=100,
            base_entropy=2.0,
            base_logit_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.0,
            adapter_logit_variance=0.1,
            adapter_top_token=100,
            latency_ms=5.0,
        )
        assert same.top_token_disagreement is False

        # Different top tokens
        different = EntropyDeltaSample(
            token_index=0,
            generated_token=100,
            base_entropy=2.0,
            base_logit_variance=0.1,
            base_top_token=100,
            adapter_entropy=2.0,
            adapter_logit_variance=0.1,
            adapter_top_token=200,
            latency_ms=5.0,
        )
        assert different.top_token_disagreement is True


class TestLogitEntropyCalculator:
    """Tests for LogitEntropyCalculator."""

    def test_calculator_creation(self):
        """LogitEntropyCalculator can be created without configuration knobs."""
        from modelcypher.core.domain.inference.entropy_dynamics import LogitEntropyCalculator

        calc = LogitEntropyCalculator()
        assert calc is not None

    def test_compute_returns_entropy_tuple(self):
        """Compute returns a tuple of (entropy, variance)."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitEntropyCalculator

        backend = get_default_backend()
        calc = LogitEntropyCalculator()

        # Create sample logits
        backend.random_seed(42)
        logits = backend.random_normal((100,))
        entropy, variance = calc.compute(logits)

        assert isinstance(entropy, float)
        assert isinstance(variance, float)
        eps = _eps()
        assert entropy >= -eps
        assert variance >= -eps

    def test_compute_skip_variance(self):
        """Compute can skip variance calculation."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitEntropyCalculator

        backend = get_default_backend()
        calc = LogitEntropyCalculator()

        backend.random_seed(42)
        logits = backend.random_normal((100,))
        entropy, variance = calc.compute(logits, skip_variance=True)

        assert isinstance(entropy, float)
        assert abs(variance) <= _eps()


class TestLogitDivergenceCalculator:
    """Tests for LogitDivergenceCalculator."""

    def test_kl_divergence_same_distribution(self):
        """KL divergence of identical distributions is zero."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitDivergenceCalculator

        backend = get_default_backend()
        calc = LogitDivergenceCalculator()

        backend.random_seed(42)
        logits = backend.random_normal((100,))
        kl = calc.kl_divergence(logits, logits)

        eps = division_epsilon(backend, backend.array([0.0]))
        assert kl >= -eps
        assert kl <= eps

    def test_kl_divergence_different_distributions(self):
        """KL divergence of different distributions is positive."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitDivergenceCalculator

        backend = get_default_backend()
        calc = LogitDivergenceCalculator()

        backend.random_seed(42)
        logits_p = backend.random_normal((100,))
        backend.random_seed(123)
        logits_q = backend.random_normal((100,))
        kl = calc.kl_divergence(logits_p, logits_q)

        assert kl >= -_eps()

    def test_stable_softmax(self):
        """Stable softmax doesn't overflow on large logits."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.inference.entropy_dynamics import LogitDivergenceCalculator

        backend = get_default_backend()
        calc = LogitDivergenceCalculator()

        # Create large logits that could cause overflow without stability
        large_logits = backend.array([1000.0, 1001.0, 1002.0])
        probs = calc.stable_softmax(large_logits)

        # Check probabilities sum to 1
        prob_sum_arr = backend.sum(probs)
        backend.eval(prob_sum_arr)
        prob_sum = float(backend.to_scalar(prob_sum_arr))
        eps = division_epsilon(backend, backend.array([0.0]))
        assert abs(prob_sum - 1.0) <= eps * max(1.0, abs(prob_sum))
