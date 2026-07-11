"""Synthetic fixed-basis feature-survival tests."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.fixed_basis_survival import (
    measure_fixed_basis_survival,
)


def test_identical_state_has_zero_coefficient_change(any_backend) -> None:
    activations = any_backend.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    basis = any_backend.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = measure_fixed_basis_survival(
        activations,
        activations,
        basis,
        backend=any_backend,
    )

    tolerance = math.sqrt(any_backend.finfo().eps)
    assert result.reference_residual_ratio == pytest.approx(0.0, abs=tolerance)
    assert result.candidate_residual_ratio == pytest.approx(0.0, abs=tolerance)
    assert result.coefficient_relative_change == pytest.approx(0.0, abs=tolerance)
    assert result.coefficient_cosine == pytest.approx(1.0, abs=tolerance)
    assert any_backend.tolist(result.feature_energy_ratio) == pytest.approx(
        [1.0, 1.0],
        abs=tolerance,
    )


def test_joint_rotation_preserves_fixed_basis_measurements(any_backend) -> None:
    reference = any_backend.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    candidate = reference * 0.75
    basis = any_backend.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    angle = math.pi / 5.0
    rotation = any_backend.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    original = measure_fixed_basis_survival(
        reference,
        candidate,
        basis,
        backend=any_backend,
    )
    rotated = measure_fixed_basis_survival(
        reference @ rotation,
        candidate @ rotation,
        basis @ rotation,
        backend=any_backend,
    )

    tolerance = math.sqrt(any_backend.finfo().eps)
    assert rotated.coefficient_relative_change == pytest.approx(
        original.coefficient_relative_change,
        abs=tolerance,
    )
    assert rotated.coefficient_cosine == pytest.approx(
        original.coefficient_cosine,
        abs=tolerance,
    )
    assert any_backend.tolist(rotated.feature_energy_ratio) == pytest.approx(
        any_backend.tolist(original.feature_energy_ratio),
        abs=tolerance,
    )
