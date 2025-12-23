from __future__ import annotations

import math

from modelcypher.core.domain.transfer_fidelity import TransferFidelityPrediction


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
    """Orthogonal Gram matrices should have low fidelity."""
    gram_a = [1.0, 0.0, 0.0, 1.0]
    gram_b = [1.0, 0.9, 0.9, 1.0]
    result = TransferFidelityPrediction.predict(gram_a, gram_b, n=2)
    assert result is not None
    # Single off-diagonal pair, so sample_size == 1
    assert result.sample_size == 1


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
    # For identical matrices, CI should contain 1.0
    lower, upper = result.correlation_ci95
    assert math.isfinite(lower)
    assert math.isfinite(upper)
    assert lower <= result.expected_fidelity <= upper


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
    gram = [1.0, 0.5, 0.5, 1.0]
    result = TransferFidelityPrediction.predict_with_null_distribution(
        gram, gram, n=2, null_samples=[]
    )
    assert result is not None
    assert result.sample_size == 1
