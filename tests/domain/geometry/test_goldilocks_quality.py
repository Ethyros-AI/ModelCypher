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

import modelcypher.core.domain.geometry.goldilocks_quality as goldilocks
from modelcypher.core.domain.geometry.goldilocks_quality import (
    GoldilocksQualityResult,
    _compute_cka_and_barrier,
    classify_problems_by_quality,
    compute_goldilocks_quality,
)


def test_compute_cka_and_barrier_mismatched_samples(any_backend) -> None:
    b = any_backend
    activations = b.array([[1.0, 0.0], [1.0, 0.0]])
    reference = b.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    cka_similarity, barrier_height = _compute_cka_and_barrier(activations, reference, b)

    assert cka_similarity == pytest.approx(1.0)
    assert barrier_height == pytest.approx(0.0)


def test_compute_goldilocks_quality_with_stubbed_dependencies(monkeypatch, any_backend) -> None:
    def _fake_fisher(_activations, _backend):
        return SimpleNamespace(mean_fim=0.002)

    monkeypatch.setattr(goldilocks, "compute_empirical_fisher_diagonal", _fake_fisher)
    monkeypatch.setattr(goldilocks, "_compute_cka_and_barrier", lambda *_: (0.9, 0.05))

    b = any_backend
    activations = b.array([[1.0, 0.0], [0.0, 1.0]])
    reference = b.array([[1.0, 0.0], [0.0, 1.0]])

    result = compute_goldilocks_quality(activations, reference, backend=b)

    assert result.cka_similarity == pytest.approx(0.9)
    assert result.barrier_height == pytest.approx(0.05)
    assert result.fisher_mean == pytest.approx(0.002)


def test_classify_problems_by_quality_groups_indices() -> None:
    results = [
        GoldilocksQualityResult(cka_similarity=0.9, barrier_height=0.05, fisher_mean=0.01),
        GoldilocksQualityResult(cka_similarity=0.85, barrier_height=0.05, fisher_mean=0.01),
        GoldilocksQualityResult(cka_similarity=0.8, barrier_height=0.03, fisher_mean=0.01),
        GoldilocksQualityResult(cka_similarity=0.6, barrier_height=0.2, fisher_mean=0.01),
    ]

    groups = classify_problems_by_quality(results, n_groups=3)

    flat = [idx for group in groups for idx in group]
    assert sorted(flat) == [0, 1, 2, 3]
    assert 0 in groups[0]
    assert 3 in groups[-1]
