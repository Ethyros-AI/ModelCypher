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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_stitcher import ModelFingerprints
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class ProjectionMethod(str, Enum):
    pca = "pca"
    tsne = "tsne"
    umap = "umap"


from modelcypher.core.domain.geometry.exceptions import ProjectionError


@dataclass(frozen=True)
class ProjectionFeature:
    layer: int
    dimension: int
    frequency: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.dimension}"


@dataclass(frozen=True)
class ProjectionPoint:
    prime_id: str
    prime_text: str
    x: float
    y: float


@dataclass(frozen=True)
class Projection:
    model_id: str
    method: ProjectionMethod
    max_features: int
    included_layers: list[int] | None
    features: list[ProjectionFeature]
    points: list[ProjectionPoint]


class ModelFingerprintsProjection:
    @staticmethod
    def _project_geodesic_mds(
        points: "Array",
        backend: "Backend",
        target_dim: int,
    ) -> "Array":
        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=None)
        geo_dist = geo_result.distances
        backend.eval(geo_dist)

        max_val = float(backend.finfo(geo_dist.dtype).max)
        finite = backend.where(geo_dist < max_val, geo_dist, backend.zeros_like(geo_dist))
        max_finite = backend.max(finite)
        backend.eval(max_finite)
        geo_dist = backend.where(geo_dist < max_val, geo_dist, max_finite)
        backend.eval(geo_dist)

        D_sq = geo_dist * geo_dist
        backend.eval(D_sq)

        row_mean = backend.mean(D_sq, axis=1, keepdims=True)
        col_mean = backend.mean(D_sq, axis=0, keepdims=True)
        grand_mean = backend.mean(D_sq)
        backend.eval(row_mean, col_mean, grand_mean)

        B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)
        B = 0.5 * (B + backend.transpose(B))
        backend.eval(B)

        B_frob_sq = backend.sum(B * B)
        backend.eval(B_frob_sq)
        B_frob = backend.sqrt(B_frob_sq)
        backend.eval(B_frob)

        eps = division_epsilon(backend, B)
        reg_lambda = eps * float(backend.to_scalar(B_frob))
        B = B + reg_lambda * backend.eye(int(B.shape[0]))
        backend.eval(B)

        eigenvalues, eigenvectors = backend.eigh(B)
        backend.eval(eigenvalues, eigenvectors)

        n_eig = eigenvalues.shape[0]
        rev_idx = backend.arange(n_eig - 1, -1, -1)
        backend.eval(rev_idx)
        eigenvalues = backend.take(eigenvalues, rev_idx, axis=0)
        eigenvectors = backend.take(eigenvectors, rev_idx, axis=1)
        backend.eval(eigenvalues, eigenvectors)

        pos_mask = eigenvalues > eps
        backend.eval(pos_mask)
        n_positive_arr = backend.sum(
            backend.where(pos_mask, backend.ones_like(eigenvalues), backend.zeros_like(eigenvalues))
        )
        backend.eval(n_positive_arr)
        n_positive = int(backend.to_scalar(n_positive_arr))
        if n_positive < target_dim:
            raise ProjectionError(
                f"Only {n_positive} positive eigenvalues, need {target_dim}. "
                "Distance matrix is non-metric."
            )

        U_k = eigenvectors[:, :target_dim]
        S_k = eigenvalues[:target_dim]
        backend.eval(U_k, S_k)

        eps_arr = backend.zeros_like(S_k) + eps
        S_k = backend.maximum(S_k, eps_arr)
        backend.eval(S_k)

        sqrt_S_k = backend.sqrt(S_k)
        backend.eval(sqrt_S_k)
        projected = U_k * sqrt_S_k[None, :]
        backend.eval(projected)

        return projected

    @staticmethod
    def project_2d(
        fingerprints: ModelFingerprints,
        method: ProjectionMethod = ProjectionMethod.pca,
        max_features: int = 1200,
        layers: set[int] | None = None,
        seed: int = 42,
    ) -> Projection:
        if not fingerprints.fingerprints:
            raise ProjectionError("No fingerprints available for projection.")
        if len(fingerprints.fingerprints) < 2:
            raise ProjectionError(
                f"Projection requires at least 2 fingerprints (got {len(fingerprints.fingerprints)})."
            )

        if method in {ProjectionMethod.tsne, ProjectionMethod.umap}:
            raise ProjectionError(
                f"Projection method {method.value.upper()} is not available in-app yet."
            )
        # Geodesic MDS is the only supported projection; PCA is an alias.

        feature_list = ModelFingerprintsProjection._select_features(
            fingerprints=fingerprints,
            max_features=max_features,
            layers=layers,
        )
        if len(feature_list) < 2:
            raise ProjectionError(
                f"Projection requires at least 2 features (got {len(feature_list)})."
            )

        included_layers = sorted(layers) if layers else None
        n = len(fingerprints.fingerprints)
        d = len(feature_list)

        feature_index = {
            (feature.layer, feature.dimension): idx for idx, feature in enumerate(feature_list)
        }

        matrix = [[0.0] * d for _ in range(n)]
        for row, fingerprint in enumerate(fingerprints.fingerprints):
            for layer, dims in fingerprint.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    col = feature_index.get((layer, dim.index))
                    if col is None:
                        continue
                    matrix[row][col] = float(dim.activation)

        backend = get_default_backend()
        points_arr = backend.array(matrix, dtype="float32")
        backend.eval(points_arr)

        projected = ModelFingerprintsProjection._project_geodesic_mds(
            points_arr, backend, target_dim=2
        )
        backend.eval(projected)

        coords_list = backend.tolist(projected)
        points: list[ProjectionPoint] = []
        for row, fingerprint in enumerate(fingerprints.fingerprints):
            points.append(
                ProjectionPoint(
                    prime_id=fingerprint.prime_id,
                    prime_text=fingerprint.prime_text,
                    x=float(coords_list[row][0]),
                    y=float(coords_list[row][1]),
                )
            )

        return Projection(
            model_id=fingerprints.model_id,
            method=method,
            max_features=max_features,
            included_layers=included_layers,
            features=feature_list,
            points=points,
        )

    @staticmethod
    def _select_features(
        fingerprints: ModelFingerprints,
        max_features: int,
        layers: set[int] | None,
    ) -> list[ProjectionFeature]:
        frequency: dict[tuple[int, int], int] = {}
        for fingerprint in fingerprints.fingerprints:
            for layer, dims in fingerprint.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    key = (layer, dim.index)
                    frequency[key] = frequency.get(key, 0) + 1

        limit = max(1, max_features)
        sorted_items = sorted(
            frequency.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        features = []
        for (layer, dimension), count in sorted_items[:limit]:
            features.append(ProjectionFeature(layer=layer, dimension=dimension, frequency=count))
        return features
