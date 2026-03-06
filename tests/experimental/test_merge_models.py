# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

from __future__ import annotations

import pytest

from modelcypher.experimental.merge.models import LayerSemanticProfile


def test_transfer_safety_requires_boundary_measurement() -> None:
    profile = LayerSemanticProfile()

    with pytest.raises(ValueError, match="boundary_radii"):
        profile.get_transfer_safety(0)


def test_transfer_safety_requires_requested_layer_radius() -> None:
    profile = LayerSemanticProfile(boundary_radii={1: 2.0})

    with pytest.raises(KeyError, match="Layer 0"):
        profile.get_transfer_safety(0)


def test_transfer_safety_zero_when_all_boundary_radii_are_zero() -> None:
    profile = LayerSemanticProfile(boundary_radii={0: 0.0})

    assert profile.get_transfer_safety(0) == 0.0


def test_transfer_safety_normalizes_against_measured_maximum() -> None:
    profile = LayerSemanticProfile(boundary_radii={0: 2.0, 1: 4.0})

    assert profile.get_transfer_safety(0) == pytest.approx(0.5)
    assert profile.get_transfer_safety(1) == pytest.approx(1.0)
