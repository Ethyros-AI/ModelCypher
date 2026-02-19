# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for topological phase monitoring in MLXTrainingAdapter."""

from __future__ import annotations

import math
from typing import Any

from modelcypher.backends.mlx_training_adapter import EpochMetrics


def test_epoch_metrics_topo_fields_default_none():
    """EpochMetrics topo fields default to None."""
    em = EpochMetrics(
        epoch=0,
        train_loss=1.0,
        val_loss=0.9,
        lipschitz_L=10.0,
        eta=0.01,
        update_norm=0.1,
        max_spectral_ratio=0.5,
        mean_token_entropy=3.0,
        repetition_rate=0.05,
        elapsed_seconds=1.0,
    )
    assert em.topo_betti_0 is None
    assert em.topo_betti_1 is None
    assert em.topo_persistence_entropy is None
    assert em.topo_mean_ricci_curvature is None
    assert em.topo_ricci_curvature_std is None


def test_epoch_metrics_topo_fields_settable():
    """EpochMetrics topo fields can be set after construction."""
    em = EpochMetrics(
        epoch=1,
        train_loss=0.8,
        val_loss=0.7,
        lipschitz_L=8.0,
        eta=0.005,
        update_norm=0.05,
        max_spectral_ratio=0.4,
        mean_token_entropy=2.5,
        repetition_rate=0.02,
        elapsed_seconds=2.0,
    )
    em.topo_betti_0 = 3
    em.topo_betti_1 = 1
    em.topo_persistence_entropy = 0.72
    em.topo_mean_ricci_curvature = -0.15
    em.topo_ricci_curvature_std = 0.08

    d = em.to_dict()
    assert d["topo_betti_0"] == 3
    assert d["topo_betti_1"] == 1
    assert abs(d["topo_persistence_entropy"] - 0.72) < 1e-6
    assert abs(d["topo_mean_ricci_curvature"] - (-0.15)) < 1e-6
    assert abs(d["topo_ricci_curvature_std"] - 0.08) < 1e-6


def test_epoch_metrics_to_dict_includes_topo_keys():
    """to_dict() always includes topo keys (even when None)."""
    em = EpochMetrics(
        epoch=0,
        train_loss=1.0,
        val_loss=None,
        lipschitz_L=None,
        eta=0.01,
        update_norm=None,
        max_spectral_ratio=None,
        mean_token_entropy=None,
        repetition_rate=None,
        elapsed_seconds=0.5,
    )
    d = em.to_dict()
    assert "topo_betti_0" in d
    assert "topo_betti_1" in d
    assert "topo_persistence_entropy" in d
    assert "topo_mean_ricci_curvature" in d
    assert "topo_ricci_curvature_std" in d


def test_topological_fingerprint_on_synthetic_points(any_backend):
    """BackendTopologicalFingerprint.compute() returns valid Betti numbers."""
    from modelcypher.core.domain.geometry.topological_fingerprint import (
        BackendTopologicalFingerprint,
    )

    # 10 points on a circle (should have B0=1 connected component)
    points = []
    for i in range(10):
        angle = 2 * math.pi * i / 10
        points.append([math.cos(angle), math.sin(angle)])

    topo = BackendTopologicalFingerprint(any_backend)
    fingerprint = topo.compute(points)

    assert fingerprint.betti_numbers.get(0, 0) >= 1
    assert fingerprint.summary.persistence_entropy >= 0.0


def test_ollivier_ricci_on_synthetic_points(any_backend):
    """OllivierRicciCurvature.compute() returns valid result."""
    from modelcypher.core.domain.geometry.ollivier_ricci import (
        OllivierRicciCurvature,
    )

    b = any_backend
    # 10 points on a circle in 4D
    rows = []
    for i in range(10):
        angle = 2 * math.pi * i / 10
        rows.append([math.cos(angle), math.sin(angle), 0.0, 0.0])

    points = b.array(rows)
    b.eval(points)

    orc = OllivierRicciCurvature(b)
    result = orc.compute(points)

    assert result.n_points == 10
    assert isinstance(result.mean_edge_curvature, float)
    assert isinstance(result.std_edge_curvature, float)
