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
    confidences = TransportGuidedMerger.compute_dimension_confidences(plan)
    assert confidences[0] > confidences[1]
