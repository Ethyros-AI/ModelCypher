from __future__ import annotations

import pytest

from modelcypher.core.domain.entropy.entropy_delta_sample import EntropyDeltaSample
from modelcypher.core.domain.entropy.model_state import ModelState
from modelcypher.core.domain.geometry.manifold_dimensionality import ManifoldDimensionality


def test_entropy_trace_features() -> None:
    features = ManifoldDimensionality.entropy_trace_features([1.0, 2.0, 3.0])
    assert features is not None
    assert features.token_count == 3
    assert features.mean == pytest.approx(2.0)
    assert features.std_dev == pytest.approx(1.0)
    assert features.max == pytest.approx(3.0)
    assert features.feature_vector == [3.0, 2.0, 1.0]


def test_feature_stats() -> None:
    stats = ManifoldDimensionality.feature_stats([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])
    assert len(stats) == 2
    assert stats[0].name == "a"
    assert stats[0].mean == pytest.approx(2.0)
    assert stats[1].name == "b"
    assert stats[1].mean == pytest.approx(3.0)


def test_summarize_prior_tension() -> None:
    samples = [
        EntropyDeltaSample.create(
            token_index=0,
            generated_token=1,
            base_entropy=1.0,
            base_top_k_variance=0.1,
            base_state=ModelState.nominal,
            base_top_token=1,
            adapter_entropy=1.2,
            adapter_top_k_variance=0.2,
            adapter_state=ModelState.nominal,
            adapter_top_token=1,
            base_surprisal=2.0,
            base_approval_probability=0.1,
            normalized_approval_score=0.2,
        ),
        EntropyDeltaSample.create(
            token_index=1,
            generated_token=2,
            base_entropy=1.5,
            base_top_k_variance=0.2,
            base_state=ModelState.nominal,
            base_top_token=2,
            adapter_entropy=1.6,
            adapter_top_k_variance=0.2,
            adapter_state=ModelState.nominal,
            adapter_top_token=3,
            base_surprisal=4.0,
            base_approval_probability=0.05,
            normalized_approval_score=0.1,
        ),
    ]
    summary = ManifoldDimensionality.summarize_prior_tension(samples)
    assert summary is not None
    assert summary.token_count == 2
    assert summary.mean_base_surprisal == pytest.approx(3.0)
    assert summary.top_token_disagreement_rate == pytest.approx(0.5)


def test_estimate_id() -> None:
    points = [[float(i), 0.0] for i in range(6)]
    summary = ManifoldDimensionality.estimate_id(points, bootstrap_resamples=None, use_regression=False)
    assert summary.sample_count == 6
    assert summary.intrinsic_dimension > 0
