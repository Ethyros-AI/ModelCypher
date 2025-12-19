from __future__ import annotations

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
