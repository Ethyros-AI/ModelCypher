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

import numpy as np
import pytest
from modelcypher.core.domain.geometry.spectral_analysis import compute_spectral_metrics, SpectralConfig


def test_layer_norm_spectral_norm():
    """LayerNorm (1D weight) should use vector norm, and condition number should be 1.0."""
    # 1D weights (bias or LN scale)
    source_ln = np.array([1.0, 1.0, 1.0])
    target_ln = np.array([1.0, 1.0, 1.1])
    
    metrics = compute_spectral_metrics(source_ln, target_ln)
    
    assert metrics.condition_number == 1.0
    assert metrics.source_spectral_norm == pytest.approx(np.sqrt(3))
    assert metrics.target_spectral_norm == pytest.approx(np.sqrt(1.0**2 + 1.0**2 + 1.1**2))


def test_layer_norm_mismatch_confidence():
    """Test spectral confidence for LayerNorm mismatch."""
    source_ln = np.array([1.0, 0.0])
    target_ln = np.array([10.0, 0.0])
    
    metrics = compute_spectral_metrics(source_ln, target_ln)
    
    # ratio = 1/10 = 0.1
    # confidence = min(0.1, 10.0) = 0.1
    assert metrics.spectral_ratio == pytest.approx(0.1)
    assert metrics.spectral_confidence == pytest.approx(0.1)


def test_layer_norm_zero_norm_stability():
    """Test spectral metrics when target LayerNorm is zero."""
    source_ln = np.array([1.0, 2.0])
    target_ln = np.zeros(2)
    
    config = SpectralConfig(epsilon=1e-6)
    metrics = compute_spectral_metrics(source_ln, target_ln, config=config)
    
    # target_spectral_norm should be clamped to epsilon
    assert metrics.target_spectral_norm == config.epsilon
    assert metrics.spectral_ratio == pytest.approx(np.sqrt(5) / config.epsilon)


def test_layer_norm_identical_confidence():
    """Identical LayerNorms should have 1.0 confidence."""
    ln = np.random.randn(128)
    metrics = compute_spectral_metrics(ln, ln)
    
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
