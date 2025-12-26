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

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_delta_sample import EntropyDeltaSample
from modelcypher.core.domain.geometry.manifold_dimensionality import (
    BackendManifoldDimensionality,
    ManifoldDimensionality,
    get_manifold_dimensionality,
)


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
            base_top_token=1,
            adapter_entropy=1.2,
            adapter_top_k_variance=0.2,
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
            base_top_token=2,
            adapter_entropy=1.6,
            adapter_top_k_variance=0.2,
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
    summary = ManifoldDimensionality.estimate_id(
        points, bootstrap_resamples=None, use_regression=False
    )
    assert summary.sample_count == 6
    assert summary.intrinsic_dimension > 0


class TestBackendManifoldDimensionality:
    """Tests for the GPU-accelerated BackendManifoldDimensionality."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def md(self, backend):
        return BackendManifoldDimensionality(backend)

    def test_entropy_trace_features_matches_pure_python(self, md) -> None:
        """Backend entropy trace features should match pure Python."""
        entropies = [1.0, 2.0, 3.0, 4.0, 5.0]

        pure = ManifoldDimensionality.entropy_trace_features(entropies)
        backend = md.entropy_trace_features(entropies)

        assert pure is not None
        assert backend is not None
        assert pure.token_count == backend.token_count
        assert pure.mean == pytest.approx(backend.mean, abs=1e-6)
        assert pure.std_dev == pytest.approx(backend.std_dev, abs=1e-6)
        assert pure.max == pytest.approx(backend.max, abs=1e-6)

    def test_entropy_trace_features_empty_returns_none(self, md) -> None:
        """Empty input should return None."""
        assert md.entropy_trace_features([]) is None

    def test_entropy_trace_features_with_nan(self, md) -> None:
        """NaN values should be filtered out."""
        entropies = [1.0, float("nan"), 3.0]
        features = md.entropy_trace_features(entropies)
        assert features is not None
        assert features.token_count == 2
        assert features.mean == pytest.approx(2.0, abs=1e-6)

    def test_feature_stats_matches_pure_python(self, md) -> None:
        """Backend feature stats should match pure Python."""
        points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        names = ["a", "b"]

        pure = ManifoldDimensionality.feature_stats(points, names)
        backend = md.feature_stats(points, names)

        assert len(pure) == len(backend)
        for p, b in zip(pure, backend):
            assert p.index == b.index
            assert p.name == b.name
            assert p.mean == pytest.approx(b.mean, abs=1e-6)
            assert p.std_dev == pytest.approx(b.std_dev, abs=1e-6)

    def test_feature_stats_empty_returns_empty(self, md) -> None:
        """Empty input should return empty list."""
        assert md.feature_stats([], []) == []

    def test_summarize_prior_tension_matches_pure_python(self, md) -> None:
        """Backend prior tension summary should match pure Python."""
        samples = [
            EntropyDeltaSample.create(
                token_index=0,
                generated_token=1,
                base_entropy=1.0,
                base_top_k_variance=0.1,
                base_top_token=1,
                adapter_entropy=1.2,
                adapter_top_k_variance=0.2,
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
                base_top_token=2,
                adapter_entropy=1.6,
                adapter_top_k_variance=0.2,
                adapter_top_token=3,
                base_surprisal=4.0,
                base_approval_probability=0.05,
                normalized_approval_score=0.1,
            ),
        ]

        pure = ManifoldDimensionality.summarize_prior_tension(samples)
        backend = md.summarize_prior_tension(samples)

        assert pure is not None
        assert backend is not None
        assert pure.token_count == backend.token_count
        assert pure.mean_base_surprisal == pytest.approx(
            backend.mean_base_surprisal, abs=1e-6
        )
        assert pure.top_token_disagreement_rate == pytest.approx(
            backend.top_token_disagreement_rate, abs=1e-6
        )

    def test_summarize_prior_tension_empty_returns_none(self, md) -> None:
        """Empty samples should return None."""
        assert md.summarize_prior_tension([]) is None

    def test_estimate_id_matches_pure_python(self, md) -> None:
        """Backend ID estimate should match pure Python."""
        points = [[float(i), 0.0] for i in range(6)]

        pure = ManifoldDimensionality.estimate_id(
            points, bootstrap_resamples=None, use_regression=False
        )
        backend = md.estimate_id(
            points, bootstrap_resamples=None, use_regression=False
        )

        assert pure.sample_count == backend.sample_count
        assert pure.intrinsic_dimension == pytest.approx(
            backend.intrinsic_dimension, abs=1e-6
        )


class TestGetManifoldDimensionality:
    """Tests for the factory function."""

    def test_returns_class_without_backend(self) -> None:
        """Factory should return ManifoldDimensionality class without backend."""
        result = get_manifold_dimensionality()
        assert result is ManifoldDimensionality

    def test_returns_instance_with_backend(self) -> None:
        """Factory should return BackendManifoldDimensionality instance with backend."""
        backend = get_default_backend()
        result = get_manifold_dimensionality(backend)
        assert isinstance(result, BackendManifoldDimensionality)
