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

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.spectral_analysis import (
    SpectralConfig,
    compute_spectral_metrics,
)


def test_layer_norm_spectral_norm():
    """LayerNorm (1D weight) should use vector norm, and condition number should be 1.0."""
    backend = get_default_backend()
    # 1D weights (bias or LN scale)
    source_ln = backend.array([1.0, 1.0, 1.0])
    target_ln = backend.array([1.0, 1.0, 1.1])
    backend.eval(source_ln, target_ln)

    # Convert to numpy for compute_spectral_metrics (expects numpy arrays)
    source_np = backend.to_numpy(source_ln)
    target_np = backend.to_numpy(target_ln)

    metrics = compute_spectral_metrics(source_np, target_np)

    assert metrics.condition_number == 1.0
    # sqrt(3) for source, sqrt(1 + 1 + 1.21) for target
    expected_source_norm = float(backend.to_numpy(backend.sqrt(backend.array(3.0))))
    expected_target_norm = float(
        backend.to_numpy(backend.sqrt(backend.array(1.0**2 + 1.0**2 + 1.1**2)))
    )
    assert metrics.source_spectral_norm == pytest.approx(expected_source_norm)
    assert metrics.target_spectral_norm == pytest.approx(expected_target_norm)


def test_layer_norm_mismatch_confidence():
    """Test spectral confidence for LayerNorm mismatch."""
    backend = get_default_backend()
    source_ln = backend.array([1.0, 0.0])
    target_ln = backend.array([10.0, 0.0])
    backend.eval(source_ln, target_ln)

    source_np = backend.to_numpy(source_ln)
    target_np = backend.to_numpy(target_ln)

    metrics = compute_spectral_metrics(source_np, target_np)

    # ratio = 1/10 = 0.1
    # confidence = min(0.1, 10.0) = 0.1
    assert metrics.spectral_ratio == pytest.approx(0.1)
    assert metrics.spectral_confidence == pytest.approx(0.1)


def test_layer_norm_zero_norm_stability():
    """Test spectral metrics when target LayerNorm is zero."""
    backend = get_default_backend()
    source_ln = backend.array([1.0, 2.0])
    target_ln = backend.zeros((2,))
    backend.eval(source_ln, target_ln)

    source_np = backend.to_numpy(source_ln)
    target_np = backend.to_numpy(target_ln)

    config = SpectralConfig(epsilon=1e-6)
    metrics = compute_spectral_metrics(source_np, target_np, config=config)

    # target_spectral_norm should be clamped to epsilon
    assert metrics.target_spectral_norm == config.epsilon
    # sqrt(5) / epsilon
    expected_ratio = float(backend.to_numpy(backend.sqrt(backend.array(5.0)))) / config.epsilon
    assert metrics.spectral_ratio == pytest.approx(expected_ratio)


def test_layer_norm_identical_confidence():
    """Identical LayerNorms should have 1.0 confidence."""
    backend = get_default_backend()
    backend.random_seed(42)
    ln = backend.random_randn((128,))
    backend.eval(ln)

    ln_np = backend.to_numpy(ln)
    metrics = compute_spectral_metrics(ln_np, ln_np)

    assert metrics.spectral_confidence == pytest.approx(1.0)
    assert metrics.delta_frobenius == pytest.approx(0.0)


def test_layer_norm_influence_on_penalty():
    """Test how LayerNorm mismatch influences spectral penalty."""
    from modelcypher.core.domain.geometry.spectral_analysis import apply_spectral_penalty

    # Low confidence (0.2) should significantly increase alpha
    alpha = 0.3
    confidence = 0.2
    strength = 0.5

    adjusted = apply_spectral_penalty(alpha, confidence, strength)

    # penalty = (1 - 0.2) * 0.5 = 0.4
    # adjusted = 0.3 + (1 - 0.3) * 0.4 = 0.3 + 0.28 = 0.58
    assert adjusted == pytest.approx(0.58)
