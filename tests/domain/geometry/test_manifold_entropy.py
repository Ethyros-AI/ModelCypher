# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelcypher.core.domain.geometry.effective_rank import EffectiveRankResult
from modelcypher.core.domain.geometry.manifold_entropy import (
    LayerEntropyResult,
    ManifoldEntropy,
    ManifoldEntropyResult,
    compute_manifold_entropy,
)


class _StubIntrinsicDimension:
    def __init__(self, value: float) -> None:
        self._value = value

    def compute(self, _activations):
        return SimpleNamespace(intrinsic_dimension=self._value)


class _StubEffectiveRank:
    def __init__(self, shannon_rank: float, spectral_entropy: float) -> None:
        self._rank = shannon_rank
        self._entropy = spectral_entropy

    def compute(self, _activations):
        return EffectiveRankResult(
            renyi_effective_rank=self._rank,
            shannon_effective_rank=self._rank,
            spectral_entropy=self._entropy,
            sample_count=4,
            feature_dim=3,
            n_singular_values=3,
        )


def test_layer_entropy_ratio_property() -> None:
    result = LayerEntropyResult(
        layer_idx=0,
        intrinsic_dimension=3.0,
        effective_rank=6.0,
        spectral_entropy=1.2,
        sample_count=10,
    )
    zero_rank = LayerEntropyResult(
        layer_idx=1,
        intrinsic_dimension=3.0,
        effective_rank=0.0,
        spectral_entropy=1.2,
        sample_count=10,
    )

    assert result.dimension_ratio == pytest.approx(0.5)
    assert zero_rank.dimension_ratio == 0.0


def test_compute_layer_entropy_returns_zero_for_too_few_samples(any_backend) -> None:
    entropy = ManifoldEntropy(any_backend)
    activations = any_backend.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # n=3

    result = entropy.compute_layer_entropy(activations, layer_idx=4)

    assert result.layer_idx == 4
    assert result.sample_count == 3
    assert result.intrinsic_dimension == 0.0
    assert result.effective_rank == 0.0
    assert result.spectral_entropy == 0.0


def test_compute_layer_entropy_uses_estimators(any_backend) -> None:
    entropy = ManifoldEntropy(any_backend)
    entropy._id_estimator = _StubIntrinsicDimension(2.5)
    entropy._eff_rank = _StubEffectiveRank(shannon_rank=3.5, spectral_entropy=1.1)

    activations = any_backend.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    result = entropy.compute_layer_entropy(activations, layer_idx=7)

    assert result.layer_idx == 7
    assert result.sample_count == 4
    assert result.intrinsic_dimension == pytest.approx(2.5)
    assert result.effective_rank == pytest.approx(3.5)
    assert result.spectral_entropy == pytest.approx(1.1)


def test_compute_from_activations_aggregates(monkeypatch, any_backend) -> None:
    entropy = ManifoldEntropy(any_backend)

    def _fake_compute(_activations, layer_idx: int) -> LayerEntropyResult:
        return LayerEntropyResult(
            layer_idx=layer_idx,
            intrinsic_dimension=float(layer_idx + 1),
            effective_rank=float(layer_idx + 2),
            spectral_entropy=float(layer_idx + 3),
            sample_count=4,
        )

    monkeypatch.setattr(entropy, "compute_layer_entropy", _fake_compute)
    result = entropy.compute_from_activations(
        {
            0: any_backend.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]),
            1: any_backend.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
        }
    )

    assert isinstance(result, ManifoldEntropyResult)
    assert result.total_entropy == pytest.approx(7.0)  # 3 + 4
    assert result.mean_intrinsic_dimension == pytest.approx(1.5)  # (1 + 2) / 2
    assert result.mean_effective_rank == pytest.approx(2.5)  # (2 + 3) / 2
    assert sorted(result.layer_entropies.keys()) == [0, 1]


def test_entropy_delta_and_convenience_function(monkeypatch, any_backend) -> None:
    entropy = ManifoldEntropy(any_backend)
    before = ManifoldEntropyResult(total_entropy=5.0, layer_entropies={})
    after = ManifoldEntropyResult(total_entropy=3.0, layer_entropies={})
    assert entropy.compute_entropy_delta(before, after) == pytest.approx(2.0)

    expected = ManifoldEntropyResult(total_entropy=9.0, layer_entropies={})

    def _fake_compute(self, layer_activations):
        return expected

    monkeypatch.setattr(ManifoldEntropy, "compute_from_activations", _fake_compute)
    result = compute_manifold_entropy({0: any_backend.array([[1.0, 2.0]])}, backend=any_backend)
    assert result is expected

