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

"""Evidence suite that quantifies alignment, geodesic accuracy, and causality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.alignment_validation import (
    AlignmentGeneralizationReport,
    DomainAlignmentReport,
    alignment_generalization_by_domain,
    alignment_generalization_report,
)
from modelcypher.core.domain.geometry.analytic_manifolds import (
    analytic_geodesic_distances,
    sample_circle_points,
    sample_sphere_points,
)
from modelcypher.core.domain.geometry.constrained_transplant import (
    CausalInterventionReport,
    causal_intervention_report,
)
from modelcypher.core.domain.geometry.manifold_accuracy import (
    CurvatureAccuracyReport,
    GeodesicAccuracyReport,
    curvature_accuracy_report,
    geodesic_accuracy_report,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.atlas_protocols import AtlasProbeProtocol
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class GeodesicConvergenceEvidence:
    small: GeodesicAccuracyReport
    large: GeodesicAccuracyReport
    mean_abs_error_ratio: float
    max_abs_error_ratio: float
    mean_relative_error_ratio: float


@dataclass(frozen=True)
class CurvatureConvergenceEvidence:
    small: CurvatureAccuracyReport
    large: CurvatureAccuracyReport
    mean_abs_error_ratio: float
    max_abs_error_ratio: float


@dataclass(frozen=True)
class EvidenceReport:
    alignment_generalization: AlignmentGeneralizationReport
    geodesic_convergence: GeodesicConvergenceEvidence
    curvature_convergence: CurvatureConvergenceEvidence
    causal_intervention: CausalInterventionReport
    domain_alignment: DomainAlignmentReport | None


def run_synthetic_evidence(
    alignment_dim: int = 16,
    alignment_samples: int | None = None,
    circle_samples: tuple[int, int] = (32, 64),
    sphere_grid: tuple[tuple[int, int], tuple[int, int]] = ((6, 8), (8, 10)),
    radius: float = 1.0,
    seed: int | None = 0,
    domain_source: "Array | None" = None,
    domain_target: "Array | None" = None,
    domain_probes: "list[AtlasProbeProtocol] | None" = None,
    backend: "Backend | None" = None,
) -> EvidenceReport:
    """Run synthetic evidence checks with optional domain alignment."""
    b = backend or get_default_backend()

    if alignment_dim <= 0:
        raise ValueError("alignment_dim must be positive.")

    if alignment_samples is None:
        alignment_samples = max(4, alignment_dim * 2)

    if alignment_samples < 4:
        raise ValueError("alignment_samples must be at least 4.")

    # Alignment generalization (linear mapping, held-out evaluation)
    if seed is not None:
        b.random_seed(seed)
    source = b.random_normal((alignment_samples, alignment_dim))
    transform = b.random_normal((alignment_dim, alignment_dim))
    target = b.matmul(source, transform)
    b.eval(source, transform, target)

    indices = list(range(alignment_samples))
    train_idx = indices[::2]
    holdout_idx = indices[1::2]
    if len(train_idx) < 2 or len(holdout_idx) < 2:
        raise ValueError("Need at least 2 samples in train and holdout splits.")

    alignment_report = alignment_generalization_report(
        source=source,
        target=target,
        train_indices=train_idx,
        holdout_indices=holdout_idx,
        backend=b,
    )

    # Geodesic convergence on a circle
    circle_small = sample_circle_points(circle_samples[0], radius=radius, backend=b)
    circle_large = sample_circle_points(circle_samples[1], radius=radius, backend=b)
    analytic_small = analytic_geodesic_distances(circle_small, radius=radius, backend=b)
    analytic_large = analytic_geodesic_distances(circle_large, radius=radius, backend=b)
    geo_small = geodesic_accuracy_report(
        circle_small, analytic_small, backend=b
    )
    geo_large = geodesic_accuracy_report(
        circle_large, analytic_large, backend=b
    )
    mean_abs_ratio = (
        geo_large.mean_abs_error / geo_small.mean_abs_error
        if geo_small.mean_abs_error > 0
        else 0.0
    )
    max_abs_ratio = (
        geo_large.max_abs_error / geo_small.max_abs_error
        if geo_small.max_abs_error > 0
        else 0.0
    )
    mean_rel_ratio = (
        geo_large.mean_relative_error / geo_small.mean_relative_error
        if geo_small.mean_relative_error > 0
        else 0.0
    )
    geodesic_evidence = GeodesicConvergenceEvidence(
        small=geo_small,
        large=geo_large,
        mean_abs_error_ratio=float(mean_abs_ratio),
        max_abs_error_ratio=float(max_abs_ratio),
        mean_relative_error_ratio=float(mean_rel_ratio),
    )

    # Curvature convergence on a sphere
    (lat_small, lon_small), (lat_large, lon_large) = sphere_grid
    sphere_small = sample_sphere_points(lat_small, lon_small, radius=radius, backend=b)
    sphere_large = sample_sphere_points(lat_large, lon_large, radius=radius, backend=b)
    analytic_curvature = 1.0 / (radius * radius)
    curv_small = curvature_accuracy_report(
        sphere_small,
        analytic_curvature=analytic_curvature,
        backend=b,
    )
    curv_large = curvature_accuracy_report(
        sphere_large,
        analytic_curvature=analytic_curvature,
        backend=b,
    )
    curv_mean_ratio = (
        curv_large.mean_abs_error / curv_small.mean_abs_error
        if curv_small.mean_abs_error > 0
        else 0.0
    )
    curv_max_ratio = (
        curv_large.max_abs_error / curv_small.max_abs_error
        if curv_small.max_abs_error > 0
        else 0.0
    )
    curvature_evidence = CurvatureConvergenceEvidence(
        small=curv_small,
        large=curv_large,
        mean_abs_error_ratio=float(curv_mean_ratio),
        max_abs_error_ratio=float(curv_max_ratio),
    )

    # Causal intervention evidence
    in_dim = alignment_dim
    out_dim = max(2, alignment_dim // 2)
    core_samples = max(4, alignment_dim)
    boundary_samples = max(2, alignment_dim // 4)

    if seed is not None:
        b.random_seed(seed + 1)
    weight_target = b.random_normal((out_dim, in_dim))
    activations_core = b.random_normal((core_samples, in_dim))
    delta_activations = b.random_normal((core_samples, out_dim))
    boundary_activations = b.random_normal((boundary_samples, in_dim))
    b.eval(weight_target, activations_core, delta_activations, boundary_activations)

    causal_report = causal_intervention_report(
        target_weights=weight_target,
        activations_core=activations_core,
        delta_activations=delta_activations,
        boundary_activations=boundary_activations,
        backend=b,
    )

    domain_alignment = None
    if domain_source is not None and domain_target is not None and domain_probes is not None:
        domain_alignment = alignment_generalization_by_domain(
            source=domain_source,
            target=domain_target,
            probes=domain_probes,
            backend=b,
        )

    return EvidenceReport(
        alignment_generalization=alignment_report,
        geodesic_convergence=geodesic_evidence,
        curvature_convergence=curvature_evidence,
        causal_intervention=causal_report,
        domain_alignment=domain_alignment,
    )


__all__ = [
    "CurvatureConvergenceEvidence",
    "EvidenceReport",
    "GeodesicConvergenceEvidence",
    "run_synthetic_evidence",
]
