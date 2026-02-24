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
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.backend_matrix_utils import BackendMatrixUtils
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class ProjectionResult:
    projected_embeddings: "Array"
    projection_matrix: "Array | None"
    reconstruction_error: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        output_shape = list(self.projected_embeddings.shape)
        payload = {
            "reconstruction_error": self.reconstruction_error,
            "output_shape": output_shape,
            "has_projection_matrix": self.projection_matrix is not None,
        }
        payload.update(self.metadata)
        return payload


class EmbeddingProjector:
    """Project embeddings into a shared geometric space."""

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        self._backend = backend or get_default_backend()

    def project(
        self,
        source: "Array",
        target: "Array",
        shared_token_indices: tuple[list[int], list[int]] | None = None,
    ) -> ProjectionResult:
        projected, projection_matrix, meta = self._project_procrustes(
            source, target, shared_token_indices
        )

        metrics = self.compute_projection_metrics(
            source, projected, target, shared_indices=shared_token_indices
        )
        reconstruction_error = metrics["mse"]
        meta = dict(meta)
        meta.update(metrics)

        return ProjectionResult(
            projected_embeddings=projected,
            projection_matrix=projection_matrix,
            reconstruction_error=reconstruction_error,
            metadata=meta,
        )

    def compute_projection_metrics(
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

        cos_vals, dist_vals = geodesic_pairwise_metrics(
            projected_sel, target_sel, backend
        )
        mse_arr = backend.mean(dist_vals * dist_vals)
        mean_cos = backend.mean(cos_vals)

        proj_norms = geodesic_norms(projected_sel, backend)
        source_norms = geodesic_norms(source_sel, backend)
        eps = division_epsilon(backend, source_norms)
        norm_denom = backend.clip(source_norms, eps, None)
        norm_ratio = backend.mean(proj_norms / norm_denom)

        backend.eval(mse_arr, mean_cos, norm_ratio)

        return {
            "mse": float(backend.to_scalar(mse_arr)),
            "mean_cosine_similarity": float(backend.to_scalar(mean_cos)),
            "norm_preservation_ratio": float(backend.to_scalar(norm_ratio)),
            "n_samples_evaluated": int(sample_count),
        }

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
        # Use BackendMatrixUtils for Procrustes alignment
        utils = BackendMatrixUtils(backend)
        _, result = utils.procrustes_align(source_anchor, target_anchor, center=True)
        rotation = result.rotation

        # Apply rotation to full source (not just anchors)
        source_mean = backend.mean(source_resized, axis=0, keepdims=True)
        target_mean = backend.mean(target_arr, axis=0, keepdims=True)
        centered = source_resized - source_mean
        projected = backend.matmul(centered, rotation) + target_mean
        backend.eval(projected)

        meta = {
            "n_anchors": int(source_anchor.shape[0]),
        }
        return projected, rotation, meta

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
        if shared_indices is None:
            count = min(int(source.shape[0]), int(target.shape[0]))
            source_sel = source[:count]
            target_sel = target[:count]
            return source_sel, target_sel
        source_idx, target_idx = shared_indices
        return source[source_idx], target[target_idx]

