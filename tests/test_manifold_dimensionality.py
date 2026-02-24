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
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values)))


def test_entropy_trace_features() -> None:
    features = ManifoldDimensionality.entropy_trace_features([1.0, 2.0, 3.0])
    assert features is not None
    assert features.token_count == 3
    assert abs(features.mean - 2.0) < _eps(features.mean, 2.0)
    assert abs(features.std_dev - 1.0) < _eps(features.std_dev, 1.0)
    assert abs(features.max - 3.0) < _eps(features.max, 3.0)
    assert features.feature_vector == [3.0, 2.0, 1.0]


def test_feature_stats() -> None:
    stats = ManifoldDimensionality.feature_stats([[1.0, 2.0], [3.0, 4.0]], ["a", "b"])
    assert len(stats) == 2
    assert stats[0].name == "a"
    assert abs(stats[0].mean - 2.0) < _eps(stats[0].mean, 2.0)
    assert stats[1].name == "b"
    assert abs(stats[1].mean - 3.0) < _eps(stats[1].mean, 3.0)


def test_summarize_prior_tension() -> None:
    samples = [
        EntropyDeltaSample.create(
            token_index=0,
            generated_token=1,
            base_entropy=1.0,
            base_logit_variance=0.1,
            base_top_token=1,
            adapter_entropy=1.2,
            adapter_logit_variance=0.2,
            adapter_top_token=1,
            base_logit_margin=2.0,
            base_token_logit=0.1,
            base_rank_fraction=0.2,
        ),
        EntropyDeltaSample.create(
            token_index=1,
            generated_token=2,
            base_entropy=1.5,
            base_logit_variance=0.2,
            base_top_token=2,
            adapter_entropy=1.6,
            adapter_logit_variance=0.2,
            adapter_top_token=3,
            base_logit_margin=4.0,
            base_token_logit=0.05,
            base_rank_fraction=0.1,
        ),
    ]
    summary = ManifoldDimensionality.summarize_prior_tension(samples)
    assert summary is not None
    assert summary.token_count == 2
    assert abs(summary.mean_base_logit_margin - 3.0) < _eps(
        summary.mean_base_logit_margin, 3.0
    )
    assert abs(summary.min_base_rank_fraction - 0.1) < _eps(
        summary.min_base_rank_fraction, 0.1
    )
    assert abs(summary.top_token_disagreement_rate - 0.5) < _eps(
        summary.top_token_disagreement_rate, 0.5
    )


def test_estimate_id() -> None:
    points = [[float(i), 0.0] for i in range(6)]
    summary = ManifoldDimensionality.estimate_id(points)
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
        assert abs(pure.mean - backend.mean) < _eps(pure.mean, backend.mean)
        assert abs(pure.std_dev - backend.std_dev) < _eps(pure.std_dev, backend.std_dev)
        assert abs(pure.max - backend.max) < _eps(pure.max, backend.max)

    def test_entropy_trace_features_empty_returns_none(self, md) -> None:
        """Empty input should return None."""
        assert md.entropy_trace_features([]) is None

    def test_entropy_trace_features_with_nan(self, md) -> None:
        """NaN values should be filtered out."""
        entropies = [1.0, float("nan"), 3.0]
        features = md.entropy_trace_features(entropies)
        assert features is not None
        assert features.token_count == 2
        assert abs(features.mean - 2.0) < _eps(features.mean, 2.0)

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
            assert abs(p.mean - b.mean) < _eps(p.mean, b.mean)
            assert abs(p.std_dev - b.std_dev) < _eps(p.std_dev, b.std_dev)

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
                base_logit_variance=0.1,
                base_top_token=1,
                adapter_entropy=1.2,
                adapter_logit_variance=0.2,
                adapter_top_token=1,
                base_logit_margin=2.0,
                base_token_logit=0.1,
                base_rank_fraction=0.2,
            ),
            EntropyDeltaSample.create(
                token_index=1,
                generated_token=2,
                base_entropy=1.5,
                base_logit_variance=0.2,
                base_top_token=2,
                adapter_entropy=1.6,
                adapter_logit_variance=0.2,
                adapter_top_token=3,
                base_logit_margin=4.0,
                base_token_logit=0.05,
                base_rank_fraction=0.1,
            ),
        ]

        pure = ManifoldDimensionality.summarize_prior_tension(samples)
        backend = md.summarize_prior_tension(samples)

        assert pure is not None
        assert backend is not None
        assert pure.token_count == backend.token_count
        assert abs(pure.mean_base_logit_margin - backend.mean_base_logit_margin) < _eps(
            pure.mean_base_logit_margin, backend.mean_base_logit_margin
        )
        assert abs(
            pure.top_token_disagreement_rate - backend.top_token_disagreement_rate
        ) < _eps(pure.top_token_disagreement_rate, backend.top_token_disagreement_rate)

    def test_summarize_prior_tension_empty_returns_none(self, md) -> None:
        """Empty samples should return None."""
        assert md.summarize_prior_tension([]) is None

    def test_estimate_id_matches_pure_python(self, md) -> None:
        """Backend ID estimate should match pure Python."""
        points = [[float(i), 0.0] for i in range(6)]

        pure = ManifoldDimensionality.estimate_id(points)
        backend = md.estimate_id(points)

        assert pure.sample_count == backend.sample_count
        assert abs(pure.intrinsic_dimension - backend.intrinsic_dimension) < _eps(
            pure.intrinsic_dimension, backend.intrinsic_dimension
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
