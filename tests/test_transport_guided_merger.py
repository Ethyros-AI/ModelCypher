from __future__ import annotations

from modelcypher.core.domain.geometry.transport_guided_merger import TransportGuidedMerger


def test_transport_guided_merger_synthesize_identity() -> None:
    source_weights = [[1.0, 0.0], [0.0, 1.0]]
    target_weights = [[0.5, 0.5], [0.5, 0.5]]
    transport_plan = [[1.0, 0.0], [0.0, 1.0]]
    result = TransportGuidedMerger.synthesize(
        source_weights=source_weights,
        target_weights=target_weights,
        transport_plan=transport_plan,
        config=TransportGuidedMerger.Config(
            coupling_threshold=0.0,
            normalize_rows=False,
            blend_alpha=0.0,
        ),
    )
    assert result is not None
    assert result == source_weights


def test_transport_guided_merger_dimension_confidence() -> None:
    plan = [[0.9, 0.1], [0.5, 0.5]]
    confidences = TransportGuidedMerger._compute_dimension_confidences(plan)
    assert confidences[0] > confidences[1]


def test_transport_guided_merger_synthesize_with_blend() -> None:
    """Blend alpha mixes source and target weights."""
    source_weights = [[1.0, 0.0], [0.0, 1.0]]
    target_weights = [[0.0, 1.0], [1.0, 0.0]]
    transport_plan = [[1.0, 0.0], [0.0, 1.0]]
    result = TransportGuidedMerger.synthesize(
        source_weights=source_weights,
        target_weights=target_weights,
        transport_plan=transport_plan,
        config=TransportGuidedMerger.Config(
            coupling_threshold=0.0,
            normalize_rows=False,
            blend_alpha=0.5,
        ),
    )
    assert result is not None
    # With alpha=0.5, result should be midpoint between source and target
    assert abs(result[0][0] - 0.5) < 1e-6
    assert abs(result[0][1] - 0.5) < 1e-6


def test_transport_guided_merger_empty_weights() -> None:
    """Empty weights return None."""
    result = TransportGuidedMerger.synthesize(
        source_weights=[],
        target_weights=[[1.0]],
        transport_plan=[],
    )
    assert result is None


def test_transport_guided_merger_threshold_application() -> None:
    """Coupling threshold filters small values."""
    plan = [[0.01, 0.99], [0.5, 0.5]]
    thresholded = TransportGuidedMerger._apply_threshold(plan, threshold=0.1)
    assert thresholded[0][0] == 0.0  # Below threshold
    assert thresholded[0][1] == 0.99  # Above threshold


def test_transport_guided_merger_row_normalization() -> None:
    """Row normalization makes rows sum to 1."""
    plan = [[0.2, 0.8], [0.3, 0.3]]
    normalized = TransportGuidedMerger._normalize_rows(plan)
    assert abs(sum(normalized[0]) - 1.0) < 1e-6
    assert abs(sum(normalized[1]) - 1.0) < 1e-6


def test_transport_guided_merger_marginal_error() -> None:
    """Marginal error computation works correctly."""
    # Uniform coupling should have low error
    uniform_plan = [[0.25, 0.25], [0.25, 0.25]]
    row_err, col_err = TransportGuidedMerger._compute_marginal_error(uniform_plan)
    assert row_err < 0.1
    assert col_err < 0.1


def test_transport_guided_merger_effective_rank() -> None:
    """Effective rank counts couplings above threshold."""
    plan = [[0.9, 0.01], [0.01, 0.9]]
    rank = TransportGuidedMerger._compute_effective_rank(plan, threshold=0.1)
    assert rank == 2  # Only diagonal elements above threshold
