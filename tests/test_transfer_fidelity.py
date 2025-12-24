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

from __future__ import annotations

import math

from modelcypher.core.domain.geometry.transfer_fidelity import TransferFidelityPrediction


def test_transfer_fidelity_identical():
    gram = [
        1.0, 0.2, 0.3,
        0.2, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    result = TransferFidelityPrediction.predict(gram, gram, n=3)
    assert result is not None
    assert abs(result.expected_fidelity - 1.0) < 1e-6
    assert result.sample_size == 3


def test_transfer_fidelity_orthogonal():
    """Different Gram matrices should have different fidelities."""
    # Use 3x3 matrices for meaningful correlation (3 off-diagonal pairs)
    gram_a = [
        1.0, 0.1, 0.2,
        0.1, 1.0, 0.3,
        0.2, 0.3, 1.0,
    ]
    gram_b = [
        1.0, 0.8, 0.7,
        0.8, 1.0, 0.6,
        0.7, 0.6, 1.0,
    ]
    result = TransferFidelityPrediction.predict(gram_a, gram_b, n=3)
    assert result is not None
    assert result.sample_size == 3
    # Different matrices should have fidelity < 1.0
    assert result.expected_fidelity < 1.0


def test_transfer_fidelity_invalid_size():
    """Invalid gram matrix size returns None."""
    gram_a = [1.0, 0.2, 0.2, 1.0]  # 2x2
    gram_b = [1.0, 0.2, 0.3, 0.2, 1.0, 0.4, 0.3, 0.4, 1.0]  # 3x3
    result = TransferFidelityPrediction.predict(gram_a, gram_b, n=3)
    assert result is None


def test_transfer_fidelity_size_one():
    """n=1 returns None (no off-diagonal elements)."""
    gram = [1.0]
    result = TransferFidelityPrediction.predict(gram, gram, n=1)
    assert result is None


def test_transfer_fidelity_qualitative_assessment():
    """Test qualitative assessment thresholds."""
    gram_perfect = [
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    result = TransferFidelityPrediction.predict(gram_perfect, gram_perfect, n=3)
    assert result is not None
    assert result.qualitative_assessment == "excellent"


def test_transfer_fidelity_fisher_z_confidence_interval():
    """Fisher z-transform produces valid 95% CI."""
    gram = [
        1.0, 0.5, 0.3, 0.2, 0.1,
        0.5, 1.0, 0.4, 0.3, 0.2,
        0.3, 0.4, 1.0, 0.5, 0.3,
        0.2, 0.3, 0.5, 1.0, 0.4,
        0.1, 0.2, 0.3, 0.4, 1.0,
    ]
    result = TransferFidelityPrediction.predict(gram, gram, n=5)
    assert result is not None
    # For identical matrices, expected_fidelity should be 1.0
    assert abs(result.expected_fidelity - 1.0) < 1e-6
    # CI bounds should be finite and reasonable (close to 1.0)
    lower, upper = result.correlation_ci95
    assert math.isfinite(lower)
    assert math.isfinite(upper)
    assert lower > 0.9  # Lower bound should be high for perfect correlation
    assert upper <= 1.0  # Upper bound is capped by Fisher Z transform


def test_transfer_fidelity_with_null_distribution():
    """Null distribution comparison adjusts confidence."""
    gram_a = [
        1.0, 0.6, 0.3,
        0.6, 1.0, 0.5,
        0.3, 0.5, 1.0,
    ]
    gram_b = [
        1.0, 0.5, 0.4,
        0.5, 1.0, 0.6,
        0.4, 0.6, 1.0,
    ]
    null_samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = TransferFidelityPrediction.predict_with_null_distribution(
        gram_a, gram_b, n=3, null_samples=null_samples
    )
    assert result is not None
    # Confidence reflects percentile in null distribution
    assert 0.0 <= result.confidence <= 1.0


def test_transfer_fidelity_empty_null_distribution():
    """Empty null distribution returns base prediction."""
    # Use 3x3 matrix for meaningful correlation (3 off-diagonal pairs)
    gram = [
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    result = TransferFidelityPrediction.predict_with_null_distribution(
        gram, gram, n=3, null_samples=[]
    )
    assert result is not None
    assert result.sample_size == 3
    # With identical matrices and empty null dist, should get base prediction
    assert abs(result.expected_fidelity - 1.0) < 1e-6
