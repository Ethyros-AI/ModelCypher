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
from modelcypher.core.domain.entropy import (
    EntropyDeltaCalibration,
    EntropyDeltaTracker,
    LogitDivergenceCalculator,
    LogitEntropyCalculator,
)
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    log_scalar,
)


def test_logit_entropy_calculator():
    # High entropy: uniform distribution [10, 10, 10]
    # Softmax([10,10,10]) = [0.33, 0.33, 0.33]
    # Entropy = - sum(0.33 * log(0.33)) = ln(3)
    logits = mx.array([10.0, 10.0, 10.0])
    calc = LogitEntropyCalculator()
    ent, var = calc.compute(logits)

    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([0.0]))
    expected_uniform_entropy = log_scalar(float(logits.shape[-1]), backend)
    assert abs(ent - expected_uniform_entropy) <= eps * max(1.0, abs(expected_uniform_entropy))

    # Low entropy: peaked distribution [100, 0, 0]
    # Softmax approx [1, 0, 0]
    # Entropy approx 0
    logits_peaked = mx.array([100.0, 0.0, 0.0])
    ent_peak, var_peak = calc.compute(logits_peaked)

    backend = get_default_backend()
    assert ent_peak <= division_epsilon(backend, backend.array([0.0]))


def test_logit_divergence_calculator():
    calc = LogitDivergenceCalculator()

    # Same distribution -> 0 KL
    logits = mx.array([1.0, 2.0, 3.0])
    kl = calc.kl_divergence(logits, logits)
    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([0.0]))
    assert abs(kl) <= eps

    # Different distribution
    p = mx.array([10.0, 0.0])  # approx [1, 0]
    q = mx.array([0.0, 10.0])  # approx [0, 1]
    # KL(p||q) should be large
    kl_large = calc.kl_divergence(p, q)
    backend = get_default_backend()
    assert kl_large > division_epsilon(backend, backend.array([0.0]))


async def test_entropy_delta_tracker_anomaly():
    base_logits = mx.array([1.0, 1.0, 1.0])
    calibration = EntropyDeltaCalibration(anomaly_threshold=1.0, source="test")
    tracker = EntropyDeltaTracker(calibration)
    tracker.start_session()

    # Base uncertain (high entropy), Adapter confident (low entropy) -> Anomaly
    # Base: Uniform
    # Adapter: Peaked
    adapter_logits = mx.array([100.0, 0.0, 0.0])

    sample = await tracker.record_dual_entropy(
        base_logits, adapter_logits, token_index=0, generated_token=0
    )

    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([0.0]))
    expected_uniform_entropy = log_scalar(float(base_logits.shape[-1]), backend)
    assert abs(sample.base_entropy - expected_uniform_entropy) <= eps * max(
        1.0, abs(expected_uniform_entropy)
    )
    assert sample.adapter_entropy <= eps
    assert sample.base_entropy >= sample.adapter_entropy + eps

    # Anomaly score is the entropy ratio for positive deltas.
    expected_ratio = sample.delta / sample.base_entropy
    expected_delta = sample.base_entropy - sample.adapter_entropy
    assert abs(sample.delta - expected_delta) <= eps * max(1.0, abs(expected_delta))
    assert abs(sample.anomaly_score - expected_ratio) <= eps * max(1.0, abs(expected_ratio))

    # End session
    result = tracker.end_session()
    assert result.anomaly_count == 1
