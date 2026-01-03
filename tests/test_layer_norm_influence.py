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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.spectral_analysis import compute_spectral_metrics


def test_layer_norm_spectral_norm():
    """LayerNorm (1D weight) should use vector norm, and condition number should be 1.0."""
    backend = get_default_backend()
    # 1D weights (bias or LN scale)
    source_ln = backend.array([1.0, 1.0, 1.0])
    target_ln = backend.array([1.0, 1.0, 1.1])
    backend.eval(source_ln, target_ln)

    # All parameters derived from data - no config needed
    metrics = compute_spectral_metrics(source_ln, target_ln)

    assert metrics.condition_number == 1.0
    # sqrt(3) for source, sqrt(1 + 1 + 1.21) for target
    expected_source_norm = float(backend.to_scalar(backend.sqrt(backend.array(3.0))))
    expected_target_norm = float(
        backend.to_scalar(backend.sqrt(backend.array(1.0**2 + 1.0**2 + 1.1**2)))
    )
    eps = machine_epsilon(backend, source_ln)
    assert abs(metrics.source_spectral_norm - expected_source_norm) <= eps
    assert abs(metrics.target_spectral_norm - expected_target_norm) <= eps


def test_layer_norm_mismatch_ratio_symmetry():
    """Test spectral ratio symmetry for LayerNorm mismatch."""
    backend = get_default_backend()
    source_ln = backend.array([1.0, 0.0])
    target_ln = backend.array([10.0, 0.0])
    backend.eval(source_ln, target_ln)

    metrics = compute_spectral_metrics(source_ln, target_ln)

    # ratio = 1/10 = 0.1
    # ratio symmetry = min(0.1, 10.0) = 0.1
    eps = machine_epsilon(backend, source_ln)
    assert abs(metrics.spectral_ratio - 0.1) <= eps
    assert abs(metrics.spectral_ratio_symmetry - 0.1) <= eps


def test_layer_norm_zero_norm_stability():
    """Test spectral metrics when target LayerNorm is zero."""
    backend = get_default_backend()
    source_ln = backend.array([1.0, 2.0])
    target_ln = backend.zeros((2,))
    backend.eval(source_ln, target_ln)

    metrics = compute_spectral_metrics(source_ln, target_ln)

    # target_spectral_norm should be clamped to epsilon
    eps = division_epsilon(backend, target_ln)
    assert metrics.target_spectral_norm == eps
    # sqrt(5) / epsilon
    expected_ratio = float(backend.to_scalar(backend.sqrt(backend.array(5.0)))) / eps
    eps_ratio = machine_epsilon(backend, source_ln) * max(1.0, abs(expected_ratio))
    assert abs(metrics.spectral_ratio - expected_ratio) <= eps_ratio


def test_layer_norm_identical_ratio_symmetry():
    """Identical LayerNorms should have ratio symmetry of 1.0."""
    backend = get_default_backend()
    backend.random_seed(42)
    ln = backend.random_normal((128,))
    backend.eval(ln)

    metrics = compute_spectral_metrics(ln, ln)

    eps = machine_epsilon(backend, ln)
    assert abs(metrics.spectral_ratio_symmetry - 1.0) <= eps
    assert abs(metrics.delta_frobenius) <= eps
