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

"""Manifold evidence metrics for activation point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank, EffectiveRankResult
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.manifold_curvature import SectionalCurvatureEstimator
from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry
from modelcypher.core.domain.geometry.riemannian_validation import derive_k_neighbors

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.manifold_curvature import ManifoldCurvatureProfile
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class CurvatureSummary:
    mean_sectional: float
    variance_sectional: float
    min_sectional: float
    max_sectional: float
    dominant_sign: str
    sign_distribution: dict[str, float]
    estimated_dimension: float | None


@dataclass(frozen=True)
class SupportManifoldDiagnostics:
    renyi_support_ratio: float
    shannon_support_ratio: float
    renyi_null_ratio: float
    shannon_null_ratio: float
    renyi_id_gap: float | None
    shannon_id_gap: float | None


@dataclass(frozen=True)
class ManifoldEvidenceResult:
    sample_count: int
    feature_dim: int
    k_neighbors: int
    intrinsic_dimension: float | None
    intrinsic_dimension_usable: int
    effective_rank: EffectiveRankResult
    support_diagnostics: SupportManifoldDiagnostics
    tangent_rank: EffectiveRankResult | None
    curvature: CurvatureSummary | None
    frechet_variance: float | None


def _summarize_curvature(profile: "ManifoldCurvatureProfile") -> CurvatureSummary:
    sign_distribution = {
        sign.value: float(value) for sign, value in profile.sign_distribution.items()
    }
    return CurvatureSummary(
        mean_sectional=float(profile.global_mean),
        variance_sectional=float(profile.global_variance),
        min_sectional=min(lc.min_sectional for lc in profile.local_curvatures)
        if profile.local_curvatures
        else 0.0,
        max_sectional=max(lc.max_sectional for lc in profile.local_curvatures)
        if profile.local_curvatures
        else 0.0,
        dominant_sign=profile.dominant_sign.value,
        sign_distribution=sign_distribution,
        estimated_dimension=profile.estimated_dimension,
    )


def _compute_support_diagnostics(
    feature_dim: int,
    effective_rank: EffectiveRankResult,
    intrinsic_dimension: float | None,
) -> SupportManifoldDiagnostics:
    if feature_dim > 0:
        renyi_support_ratio = effective_rank.renyi_effective_rank / feature_dim
        shannon_support_ratio = effective_rank.shannon_effective_rank / feature_dim
        renyi_null_ratio = 1.0 - renyi_support_ratio
        shannon_null_ratio = 1.0 - shannon_support_ratio
    else:
        renyi_support_ratio = 0.0
        shannon_support_ratio = 0.0
        renyi_null_ratio = 0.0
        shannon_null_ratio = 0.0

    if intrinsic_dimension is None:
        renyi_id_gap = None
        shannon_id_gap = None
    else:
        renyi_id_gap = effective_rank.renyi_effective_rank - intrinsic_dimension
        shannon_id_gap = effective_rank.shannon_effective_rank - intrinsic_dimension

    return SupportManifoldDiagnostics(
        renyi_support_ratio=renyi_support_ratio,
        shannon_support_ratio=shannon_support_ratio,
        renyi_null_ratio=renyi_null_ratio,
        shannon_null_ratio=shannon_null_ratio,
        renyi_id_gap=renyi_id_gap,
        shannon_id_gap=shannon_id_gap,
    )


def compute_manifold_evidence(
    points: "Array",
    backend: "Backend | None" = None,
) -> ManifoldEvidenceResult:
    """Compute manifold evidence metrics for activation point clouds."""
    b = backend or get_default_backend()
    pts = b.array(points) if not hasattr(points, "shape") else points
    b.eval(pts)

    if len(pts.shape) < 2:
        empty_effective_rank = EffectiveRankResult(
            renyi_effective_rank=0.0,
            shannon_effective_rank=0.0,
            spectral_entropy=0.0,
            sample_count=0,
            feature_dim=0,
            n_singular_values=0,
        )
        return ManifoldEvidenceResult(
            sample_count=0,
            feature_dim=0,
            k_neighbors=0,
            intrinsic_dimension=None,
            intrinsic_dimension_usable=0,
            effective_rank=empty_effective_rank,
            support_diagnostics=_compute_support_diagnostics(
                0, empty_effective_rank, None
            ),
            tangent_rank=None,
            curvature=None,
            frechet_variance=None,
        )

    sample_count = int(pts.shape[0])
    feature_dim = int(pts.shape[1])
    if sample_count == 0 or feature_dim == 0:
        empty_effective_rank = EffectiveRankResult(
            renyi_effective_rank=0.0,
            shannon_effective_rank=0.0,
            spectral_entropy=0.0,
            sample_count=sample_count,
            feature_dim=feature_dim,
            n_singular_values=0,
        )
        return ManifoldEvidenceResult(
            sample_count=sample_count,
            feature_dim=feature_dim,
            k_neighbors=0,
            intrinsic_dimension=None,
            intrinsic_dimension_usable=0,
            effective_rank=empty_effective_rank,
            support_diagnostics=_compute_support_diagnostics(
                feature_dim, empty_effective_rank, None
            ),
            tangent_rank=None,
            curvature=None,
            frechet_variance=None,
        )

    k_neighbors = derive_k_neighbors(pts, b)

    id_computer = IntrinsicDimension(b)
    try:
        id_result = id_computer.compute(pts, with_ci=False)
        intrinsic_dimension = float(id_result.intrinsic_dimension)
        intrinsic_usable = int(id_result.usable_count)
    except Exception:
        intrinsic_dimension = None
        intrinsic_usable = 0

    er_computer = EffectiveRank(b)
    effective_rank = er_computer.compute(pts)
    support_diagnostics = _compute_support_diagnostics(
        feature_dim, effective_rank, intrinsic_dimension
    )

    rg = RiemannianGeometry(b)
    tangent_rank: EffectiveRankResult | None = None
    frechet_variance: float | None = None
    try:
        frechet = rg.frechet_mean(pts)
        frechet_variance = float(frechet.final_variance)
        tangent = rg.log_map(pts, frechet.mean)
        b.eval(tangent)
        tangent_rank = er_computer.compute(tangent)
    except Exception:
        tangent_rank = None

    curvature_summary: CurvatureSummary | None = None
    try:
        estimator = SectionalCurvatureEstimator()
        profile = estimator.estimate_manifold_profile(pts)
        curvature_summary = _summarize_curvature(profile)
    except Exception:
        curvature_summary = None

    return ManifoldEvidenceResult(
        sample_count=sample_count,
        feature_dim=feature_dim,
        k_neighbors=int(k_neighbors),
        intrinsic_dimension=intrinsic_dimension,
        intrinsic_dimension_usable=intrinsic_usable,
        effective_rank=effective_rank,
        support_diagnostics=support_diagnostics,
        tangent_rank=tangent_rank,
        curvature=curvature_summary,
        frechet_variance=frechet_variance,
    )


__all__ = [
    "CurvatureSummary",
    "SupportManifoldDiagnostics",
    "ManifoldEvidenceResult",
    "compute_manifold_evidence",
]
