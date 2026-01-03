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

"""Embedding projection utilities for cross-vocabulary alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    svd_via_eigh,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class ProjectionStrategy(str, Enum):
    TRUNCATE = "truncate"
    PCA = "pca"
    PROCRUSTES = "procrustes"
    CCA = "cca"
    OPTIMAL_TRANSPORT = "optimal_transport"


@dataclass(frozen=True)
class ProjectionConfig:
    strategy: ProjectionStrategy = ProjectionStrategy.PROCRUSTES


@dataclass
class ProjectionResult:
    projected_embeddings: "Array"
    projection_matrix: "Array | None"
    reconstruction_error: float
    alignment_score: float
    strategy_used: ProjectionStrategy
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        output_shape = list(self.projected_embeddings.shape)
        payload = {
            "reconstruction_error": self.reconstruction_error,
            "alignment_score": self.alignment_score,
            "strategy_used": self.strategy_used.value,
            "output_shape": output_shape,
            "has_projection_matrix": self.projection_matrix is not None,
        }
        payload.update(self.metadata)
        return payload


class EmbeddingProjector:
    """Project embeddings into a shared geometric space."""

    def __init__(
        self,
        config: ProjectionConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = config or ProjectionConfig()
        self._backend = backend or get_default_backend()

    def project(
        self,
        source: "Array",
        target: "Array",
        shared_token_indices: tuple[list[int], list[int]] | None = None,
    ) -> ProjectionResult:
        if self.config.strategy == ProjectionStrategy.TRUNCATE:
            projected, meta = self._project_truncate(source, target)
            projection_matrix = None
        elif self.config.strategy == ProjectionStrategy.PCA:
            projected, projection_matrix, meta = self._project_pca(source, target)
        elif self.config.strategy == ProjectionStrategy.PROCRUSTES:
            projected, projection_matrix, meta = self._project_procrustes(
                source, target, shared_token_indices
            )
        elif self.config.strategy == ProjectionStrategy.CCA:
            projected, projection_matrix, meta = self._project_cca(
                source, target, shared_token_indices
            )
        elif self.config.strategy == ProjectionStrategy.OPTIMAL_TRANSPORT:
            projected, meta = self._project_optimal_transport(source, target)
            projection_matrix = meta.get("coupling")
        else:
            raise ValueError(f"Unknown projection strategy: {self.config.strategy}")

        quality = self.compute_alignment_quality(
            source, projected, target, shared_indices=shared_token_indices
        )
        alignment_score = quality["mean_cosine_similarity"]
        reconstruction_error = quality["mse"]

        return ProjectionResult(
            projected_embeddings=projected,
            projection_matrix=projection_matrix,
            reconstruction_error=reconstruction_error,
            alignment_score=alignment_score,
            strategy_used=self.config.strategy,
            metadata=meta,
        )

    def compute_alignment_quality(
        self,
        source: "Array",
        projected: "Array",
        target: "Array",
        shared_indices: tuple[list[int], list[int]] | None = None,
    ) -> dict[str, float]:
        backend = self._backend
        source_arr = backend.array(source)
        projected_arr = backend.array(projected)
        target_arr = backend.array(target)
        backend.eval(source_arr, projected_arr, target_arr)

        if shared_indices is None:
            sample_count = min(
                source_arr.shape[0],
                projected_arr.shape[0],
                target_arr.shape[0],
            )
            source_sel = source_arr[:sample_count]
            projected_sel = projected_arr[:sample_count]
            target_sel = target_arr[:sample_count]
        else:
            source_idx, target_idx = shared_indices
            source_sel = source_arr[source_idx]
            projected_sel = projected_arr[source_idx]
            target_sel = target_arr[target_idx]
            sample_count = len(source_idx)

        diff = projected_sel - target_sel
        mse_arr = backend.mean(diff * diff)

        proj_norms = backend.norm(projected_sel, axis=1)
        target_norms = backend.norm(target_sel, axis=1)
        denom = proj_norms * target_norms
        eps = division_epsilon(backend, denom)
        denom = backend.clip(denom, eps, None)
        cos = backend.sum(projected_sel * target_sel, axis=1) / denom
        mean_cos = backend.mean(cos)

        source_norms = backend.norm(source_sel, axis=1)
        norm_denom = backend.clip(source_norms, eps, None)
        norm_ratio = backend.mean(proj_norms / norm_denom)

        backend.eval(mse_arr, mean_cos, norm_ratio)

        return {
            "mse": float(backend.to_scalar(mse_arr)),
            "mean_cosine_similarity": float(backend.to_scalar(mean_cos)),
            "norm_preservation_ratio": float(backend.to_scalar(norm_ratio)),
            "n_samples_evaluated": int(sample_count),
        }

    def _project_truncate(
        self,
        source: "Array",
        target: "Array",
    ) -> tuple["Array", dict[str, Any]]:
        backend = self._backend
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        backend.eval(source_arr, target_arr)

        projected = self._resize_features(source_arr, int(target_arr.shape[1]))
        backend.eval(projected)

        meta = {
            "source_dim": int(source_arr.shape[1]),
            "target_dim": int(target_arr.shape[1]),
        }
        return projected, meta

    def _project_pca(
        self,
        source: "Array",
        target: "Array",
    ) -> tuple["Array", "Array", dict[str, Any]]:
        backend = self._backend
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        backend.eval(source_arr, target_arr)

        target_dim = int(target_arr.shape[1])
        source_dim = int(source_arr.shape[1])
        n_components = min(target_dim, source_dim)

        mean = backend.mean(source_arr, axis=0, keepdims=True)
        centered = source_arr - mean
        cov = backend.matmul(backend.transpose(centered), centered)
        U, S, _ = svd_via_eigh(backend, cov, full_matrices=False)
        components = U[:, :n_components]
        projected = backend.matmul(centered, components)

        projected = self._resize_features(projected, target_dim)
        backend.eval(projected)

        total_variance = backend.sum(S)
        explained = backend.sum(S[:n_components])
        backend.eval(total_variance, explained)
        eps = division_epsilon(backend, S)
        explained_val = float(backend.to_scalar(explained))
        total_variance_val = float(backend.to_scalar(total_variance))
        ratio = explained_val / max(total_variance_val, eps)

        meta = {
            "n_components": n_components,
            "explained_variance_ratio": ratio,
        }
        return projected, components, meta

    def _project_procrustes(
        self,
        source: "Array",
        target: "Array",
        shared_indices: tuple[list[int], list[int]] | None,
    ) -> tuple["Array", "Array", dict[str, Any]]:
        backend = self._backend
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        backend.eval(source_arr, target_arr)

        target_dim = int(target_arr.shape[1])
        source_resized = self._resize_features(source_arr, target_dim)

        source_anchor, target_anchor = self._select_anchor_pairs(
            source_resized, target_arr, shared_indices
        )
        rotation, source_mean, target_mean = self._procrustes_params(
            source_anchor, target_anchor
        )
        centered = source_resized - source_mean
        projected = backend.matmul(centered, rotation) + target_mean
        backend.eval(projected)

        meta = {
            "n_anchors": int(source_anchor.shape[0]),
        }
        return projected, rotation, meta

    def _project_cca(
        self,
        source: "Array",
        target: "Array",
        shared_indices: tuple[list[int], list[int]] | None,
    ) -> tuple["Array", "Array", dict[str, Any]]:
        backend = self._backend
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        backend.eval(source_arr, target_arr)

        target_dim = int(target_arr.shape[1])
        source_resized = self._resize_features(source_arr, target_dim)

        source_anchor, target_anchor = self._select_anchor_pairs(
            source_resized, target_arr, shared_indices
        )
        source_mean = backend.mean(source_anchor, axis=0, keepdims=True)
        target_mean = backend.mean(target_anchor, axis=0, keepdims=True)
        source_centered = source_anchor - source_mean
        target_centered = target_anchor - target_mean

        cross_cov = backend.matmul(backend.transpose(source_centered), target_centered)
        U, _, Vt = svd_via_eigh(backend, cross_cov, full_matrices=False)
        rotation = backend.matmul(U, Vt)

        centered = source_resized - source_mean
        projected = backend.matmul(centered, rotation) + target_mean
        backend.eval(projected)

        canonical = self._compute_canonical_correlations(
            source_centered,
            target_centered,
            U,
            Vt,
        )

        meta = {
            "canonical_correlations": canonical,
            "n_components": len(canonical),
        }
        return projected, rotation, meta

    def _project_optimal_transport(
        self,
        source: "Array",
        target: "Array",
    ) -> tuple["Array", dict[str, Any]]:
        from modelcypher.core.domain.geometry.gromov_wasserstein import (
            GromovWassersteinDistance,
        )

        backend = self._backend
        source_arr = backend.array(source)
        target_arr = backend.array(target)
        backend.eval(source_arr, target_arr)

        source_dim = int(source_arr.shape[1])
        target_dim = int(target_arr.shape[1])

        if source_dim == target_dim:
            projected = source_arr
            coupling = None
            transport_cost = 0.0
        else:
            source_gram = backend.matmul(backend.transpose(source_arr), source_arr)
            target_gram = backend.matmul(backend.transpose(target_arr), target_arr)
            backend.eval(source_gram, target_gram)

            gw = GromovWassersteinDistance(backend)
            result = gw.compute(source_gram, target_gram)
            coupling = result.coupling
            projected = backend.matmul(source_arr, coupling)
            backend.eval(projected)
            transport_cost = float(result.distance)

        meta = {
            "n_source_samples": int(source_arr.shape[0]),
            "n_target_samples": int(target_arr.shape[0]),
            "shared_dim": min(source_dim, target_dim),
            "transport_cost": transport_cost,
        }
        if coupling is not None:
            meta["coupling"] = coupling
        return projected, meta

    def _resize_features(self, array: "Array", target_dim: int) -> "Array":
        backend = self._backend
        current_dim = int(array.shape[1])
        if current_dim == target_dim:
            return array
        if current_dim > target_dim:
            return array[:, :target_dim]
        pad_dim = target_dim - current_dim
        padding = backend.zeros((int(array.shape[0]), pad_dim))
        return backend.concatenate([array, padding], axis=1)

    def _select_anchor_pairs(
        self,
        source: "Array",
        target: "Array",
        shared_indices: tuple[list[int], list[int]] | None,
    ) -> tuple["Array", "Array"]:
        backend = self._backend
        if shared_indices is None:
            count = min(int(source.shape[0]), int(target.shape[0]))
            source_sel = source[:count]
            target_sel = target[:count]
            return source_sel, target_sel
        source_idx, target_idx = shared_indices
        return source[source_idx], target[target_idx]

    def _procrustes_params(
        self, source: "Array", target: "Array"
    ) -> tuple["Array", "Array", "Array"]:
        backend = self._backend
        source_mean = backend.mean(source, axis=0, keepdims=True)
        target_mean = backend.mean(target, axis=0, keepdims=True)
        source_centered = source - source_mean
        target_centered = target - target_mean

        cross_cov = backend.matmul(backend.transpose(source_centered), target_centered)
        U, _, Vt = svd_via_eigh(backend, cross_cov, full_matrices=False)
        # MLX det() has unstable behavior for some sizes; use the raw orthogonal solution.
        rotation = backend.matmul(U, Vt)
        backend.eval(rotation)
        return rotation, source_mean, target_mean

    def _compute_canonical_correlations(
        self,
        source: "Array",
        target: "Array",
        U: "Array",
        Vt: "Array",
    ) -> list[float]:
        backend = self._backend
        source_proj = backend.matmul(source, U)
        target_proj = backend.matmul(target, backend.transpose(Vt))
        backend.eval(source_proj, target_proj)

        n_components = int(min(source_proj.shape[1], target_proj.shape[1]))
        eps = division_epsilon(backend, source_proj)

        # Vectorized: compute all correlations at once
        # Column-wise dot products
        dot_products = backend.sum(source_proj[:, :n_components] * target_proj[:, :n_components], axis=0)
        # Column norms
        s_norms = backend.sqrt(backend.sum(source_proj[:, :n_components] ** 2, axis=0))
        t_norms = backend.sqrt(backend.sum(target_proj[:, :n_components] ** 2, axis=0))
        # Safe division
        denom = backend.maximum(s_norms * t_norms, eps)
        corr_arr = dot_products / denom
        backend.eval(corr_arr)

        # O(1) extraction via tolist() instead of O(n) to_scalar() loop
        return [float(x) for x in backend.tolist(corr_arr)]
