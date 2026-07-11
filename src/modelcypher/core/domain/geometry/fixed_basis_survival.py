"""Fixed-basis feature survival across precision or checkpoint states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class FixedBasisSurvival:
    """Raw reconstruction and coefficient-change measurements in one frozen basis."""

    reference_residual_ratio: float
    candidate_residual_ratio: float
    coefficient_relative_change: float
    coefficient_cosine: float
    reference_feature_energy: "Array"
    candidate_feature_energy: "Array"
    feature_energy_ratio: "Array"


def measure_fixed_basis_survival(
    reference_activations: "Array",
    candidate_activations: "Array",
    frozen_basis: "Array",
    *,
    backend: "Backend",
) -> FixedBasisSurvival:
    """Measure candidate changes using a basis fitted only on the reference state.

    Activations must contain identical observations in identical order. Basis
    rows are feature vectors; the Moore-Penrose pseudoinverse supplies the
    least-squares coefficients without a fitted threshold.
    """
    if len(reference_activations.shape) != 2:
        raise ValueError("Reference activations must have shape [samples, features]")
    if tuple(reference_activations.shape) != tuple(candidate_activations.shape):
        raise ValueError("Reference and candidate activations must have identical shape")
    if len(frozen_basis.shape) != 2:
        raise ValueError("Frozen basis must have shape [basis_features, features]")
    if int(frozen_basis.shape[1]) != int(reference_activations.shape[1]):
        raise ValueError("Frozen basis width must match activation feature dimension")

    basis_pinv = backend.pinv(frozen_basis)
    reference_coefficients = backend.matmul(reference_activations, basis_pinv)
    candidate_coefficients = backend.matmul(candidate_activations, basis_pinv)
    reference_reconstruction = backend.matmul(reference_coefficients, frozen_basis)
    candidate_reconstruction = backend.matmul(candidate_coefficients, frozen_basis)

    reference_residual = reference_activations - reference_reconstruction
    candidate_residual = candidate_activations - candidate_reconstruction
    coefficient_delta = candidate_coefficients - reference_coefficients

    reference_energy = backend.sum(reference_activations * reference_activations)
    candidate_energy = backend.sum(candidate_activations * candidate_activations)
    reference_residual_energy = backend.sum(reference_residual * reference_residual)
    candidate_residual_energy = backend.sum(candidate_residual * candidate_residual)
    coefficient_energy = backend.sum(reference_coefficients * reference_coefficients)
    coefficient_delta_energy = backend.sum(coefficient_delta * coefficient_delta)
    coefficient_dot = backend.sum(reference_coefficients * candidate_coefficients)
    candidate_coefficient_energy = backend.sum(
        candidate_coefficients * candidate_coefficients
    )

    reference_feature_energy = backend.sum(
        reference_coefficients * reference_coefficients,
        axis=0,
    )
    candidate_feature_energy = backend.sum(
        candidate_coefficients * candidate_coefficients,
        axis=0,
    )
    feature_eps = division_epsilon(backend, reference_feature_energy)
    feature_energy_ratio = candidate_feature_energy / backend.maximum(
        reference_feature_energy,
        feature_eps,
    )

    scalar_tensors = (
        reference_energy,
        candidate_energy,
        reference_residual_energy,
        candidate_residual_energy,
        coefficient_energy,
        coefficient_delta_energy,
        coefficient_dot,
        candidate_coefficient_energy,
    )
    backend.eval(
        *scalar_tensors,
        reference_feature_energy,
        candidate_feature_energy,
        feature_energy_ratio,
    )
    values = [float(backend.to_scalar(value)) for value in scalar_tensors]
    (
        reference_energy_value,
        candidate_energy_value,
        reference_residual_energy_value,
        candidate_residual_energy_value,
        coefficient_energy_value,
        coefficient_delta_energy_value,
        coefficient_dot_value,
        candidate_coefficient_energy_value,
    ) = values

    reference_eps = division_epsilon(backend, reference_energy)
    candidate_eps = division_epsilon(backend, candidate_energy)
    coefficient_eps = division_epsilon(backend, coefficient_energy)
    cosine_denominator = (
        coefficient_energy_value * candidate_coefficient_energy_value
    ) ** 0.5

    return FixedBasisSurvival(
        reference_residual_ratio=(
            reference_residual_energy_value
            / max(reference_energy_value, reference_eps)
        )
        ** 0.5,
        candidate_residual_ratio=(
            candidate_residual_energy_value
            / max(candidate_energy_value, candidate_eps)
        )
        ** 0.5,
        coefficient_relative_change=(
            coefficient_delta_energy_value
            / max(coefficient_energy_value, coefficient_eps)
        )
        ** 0.5,
        coefficient_cosine=coefficient_dot_value
        / max(cosine_denominator, coefficient_eps),
        reference_feature_energy=reference_feature_energy,
        candidate_feature_energy=candidate_feature_energy,
        feature_energy_ratio=feature_energy_ratio,
    )
