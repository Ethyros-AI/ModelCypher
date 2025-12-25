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

"""Entropy dynamics tests requiring MLX (Apple Silicon)."""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain.inference.entropy_dynamics import (
    EntropyDeltaTracker,
    LogitDivergenceCalculator,
    LogitEntropyCalculator,
)


def _test_tracker_config() -> EntropyDeltaTracker.Configuration:
    """Create test config with explicit values."""
    return EntropyDeltaTracker.Configuration(
        top_k=10,
        anomaly_threshold=0.6,
        consecutive_anomaly_count=3,
        compute_variance=True,
        source="EntropyDeltaTracker",
    )


def test_logit_entropy_calculator():
    calc = LogitEntropyCalculator(top_k=2)

    # High entropy: uniform distribution [10, 10, 10]
    # Softmax([10,10,10]) = [0.33, 0.33, 0.33]
    # Entropy = - sum(0.33 * log(0.33)) = ln(3) approx 1.098
    logits = mx.array([10.0, 10.0, 10.0])
    ent, var = calc.compute(logits)

    assert abs(ent - 1.0986) < 0.01

    # Low entropy: peaked distribution [100, 0, 0]
    # Softmax approx [1, 0, 0]
    # Entropy approx 0
    logits_peaked = mx.array([100.0, 0.0, 0.0])
    ent_peak, var_peak = calc.compute(logits_peaked)

    assert ent_peak < 0.01


def test_logit_divergence_calculator():
    calc = LogitDivergenceCalculator()

    # Same distribution -> 0 KL
    logits = mx.array([1.0, 2.0, 3.0])
    kl = calc.kl_divergence(logits, logits)
    assert abs(kl) < 1e-6

    # Different distribution
    p = mx.array([10.0, 0.0])  # approx [1, 0]
    q = mx.array([0.0, 10.0])  # approx [0, 1]
    # KL(p||q) should be large
    kl_large = calc.kl_divergence(p, q)
    assert kl_large > 5.0


def test_entropy_delta_tracker_anomaly():
    tracker = EntropyDeltaTracker(_test_tracker_config())
    tracker.start_session()

    # Base uncertain (high entropy), Adapter confident (low entropy) -> Anomaly
    # Base: Uniform
    base_logits = mx.array([1.0, 1.0, 1.0])
    # Adapter: Peaked
    adapter_logits = mx.array([100.0, 0.0, 0.0])

    sample = tracker.record_dual_entropy(
        base_logits, adapter_logits, token_index=0, generated_token=0
    )

    assert sample.base_entropy > 1.0
    assert sample.adapter_entropy < 0.1
    assert sample.delta > 0.9

    # Anomaly score should be high
    assert sample.anomaly_score > 0.6

    # Check if anomaly was recorded
    assert tracker.consecutive_anomalies == 1

    # End session
    result = tracker.end_session()
    assert result.anomaly_count == 1
