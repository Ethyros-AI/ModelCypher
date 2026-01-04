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

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.ports.backend import Backend

# Data Structures mimicking Swift ManifoldStitcher.ModelFingerprints


@dataclass(frozen=True)
class ActivatedDimension:
    index: int
    activation: float


@dataclass(frozen=True)
class Fingerprint:
    prime_id: str
    prime_text: str
    # layer_index -> list of ActivatedDimension
    activated_dimensions: dict[int, list[ActivatedDimension]]


@dataclass(frozen=True)
class ModelFingerprints:
    model_id: str
    fingerprints: list[Fingerprint]


# Projection Logic


class ProjectionMethod(Enum):
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"


from modelcypher.core.domain.geometry.exceptions import ProjectionError


@dataclass
class ProjectionFeature:
    layer: int
    dimension: int
    frequency: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.dimension}"


@dataclass
class ProjectionPoint:
    id: str
    prime_id: str
    prime_text: str
    x: float
    y: float


@dataclass
class Projection:
    model_id: str
    method: ProjectionMethod
    max_features: int
    included_layers: list[list[int]] | None
    features: list[ProjectionFeature]
    points: list[ProjectionPoint]


class ModelFingerprintsProjection:
    """
    Project model fingerprints to 2D for visualization.
    Ported from ModelFingerprintsProjection.swift.
    """

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or get_default_backend()

    def project_2d(
        self,
        fingerprints: ModelFingerprints,
        method: ProjectionMethod = ProjectionMethod.PCA,
        max_features: int = 1200,
        layers: set[int] | None = None,
        seed: int = 42,
    ) -> Projection:
        if len(fingerprints.fingerprints) < 2:
            raise ProjectionError(f"Insufficient samples: {len(fingerprints.fingerprints)}")

        if method != ProjectionMethod.PCA:
            raise ProjectionError(f"Unsupported method: {method}")

        # 1. Feature Selection
        feature_list = self._select_features(fingerprints, max_features, layers)

        if len(feature_list) < 2:
            raise ProjectionError(f"Insufficient features: {len(feature_list)}")

        # 2. Build Matrix
        n = len(fingerprints.fingerprints)
        d = len(feature_list)

        # Mapping (layer, dim) -> col_index
        feature_index = {(f.layer, f.dimension): i for i, f in enumerate(feature_list)}

        # Build matrix as list of lists then convert to backend array
        matrix_data = [[0.0] * d for _ in range(n)]

        for row, fp in enumerate(fingerprints.fingerprints):
            for layer, dims in fp.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    key = (layer, dim.index)
                    if key in feature_index:
                        col = feature_index[key]
                        matrix_data[row][col] = dim.activation

        X = self._backend.array(matrix_data, dtype="float32")
        self._backend.eval(X)

        rg = RiemannianGeometry(self._backend)
        geo_result = rg.geodesic_distances(X, k_neighbors=None)
        geo_dist = geo_result.distances
        self._backend.eval(geo_dist)

        max_val = float(self._backend.finfo(geo_dist.dtype).max)
        finite = self._backend.where(
            geo_dist < max_val, geo_dist, self._backend.zeros_like(geo_dist)
        )
        max_finite = self._backend.max(finite)
        self._backend.eval(max_finite)
        geo_dist = self._backend.where(geo_dist < max_val, geo_dist, max_finite)
        self._backend.eval(geo_dist)

        D_sq = geo_dist * geo_dist
        self._backend.eval(D_sq)

        row_mean = self._backend.mean(D_sq, axis=1, keepdims=True)
        col_mean = self._backend.mean(D_sq, axis=0, keepdims=True)
        grand_mean = self._backend.mean(D_sq)
        self._backend.eval(row_mean, col_mean, grand_mean)

        B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)
        B = 0.5 * (B + self._backend.transpose(B))
        self._backend.eval(B)

        B_frob_sq = self._backend.sum(B * B)
        self._backend.eval(B_frob_sq)
        B_frob = self._backend.sqrt(B_frob_sq)
        self._backend.eval(B_frob)

        eps = division_epsilon(self._backend, B)
        reg_lambda = eps * float(self._backend.to_scalar(B_frob))
        B = B + reg_lambda * self._backend.eye(int(B.shape[0]))
        self._backend.eval(B)

        eigenvalues, eigenvectors = self._backend.eigh(B)
        self._backend.eval(eigenvalues, eigenvectors)

        n_eig = eigenvalues.shape[0]
        rev_idx = self._backend.arange(n_eig - 1, -1, -1)
        self._backend.eval(rev_idx)
        eigenvalues = self._backend.take(eigenvalues, rev_idx, axis=0)
        eigenvectors = self._backend.take(eigenvectors, rev_idx, axis=1)
        self._backend.eval(eigenvalues, eigenvectors)

        pos_mask = eigenvalues > eps
        self._backend.eval(pos_mask)
        n_positive_arr = self._backend.sum(
            self._backend.where(
                pos_mask, self._backend.ones_like(eigenvalues), self._backend.zeros_like(eigenvalues)
            )
        )
        self._backend.eval(n_positive_arr)
        n_positive = int(self._backend.to_scalar(n_positive_arr))
        if n_positive < 2:
            raise ProjectionError(
                f"Only {n_positive} positive eigenvalues, need 2. "
                "Distance matrix is non-metric."
            )

        U_k = eigenvectors[:, :2]
        S_k = eigenvalues[:2]
        self._backend.eval(U_k, S_k)

        eps_arr = self._backend.zeros_like(S_k) + eps
        S_k = self._backend.maximum(S_k, eps_arr)
        self._backend.eval(S_k)

        sqrt_S_k = self._backend.sqrt(S_k)
        self._backend.eval(sqrt_S_k)
        coords = U_k * sqrt_S_k[None, :]
        self._backend.eval(coords)

        points = []
        coords_list = self._backend.tolist(coords)
        for i, fp in enumerate(fingerprints.fingerprints):
            points.append(
                ProjectionPoint(
                    id=fp.prime_id,
                    prime_id=fp.prime_id,
                    prime_text=fp.prime_text,
                    x=coords_list[i][0],
                    y=coords_list[i][1],
                )
            )

        included_layers = [sorted(list(layers))] if layers else None

        return Projection(
            model_id=fingerprints.model_id,
            method=method,
            max_features=max_features,
            included_layers=included_layers,
            features=feature_list,
            points=points,
        )

    def _select_features(
        self,
        fingerprints: ModelFingerprints,
        max_features: int,
        layers: set[int] | None,
    ) -> list[ProjectionFeature]:
        freq_map: dict[tuple[int, int], int] = {}

        for fp in fingerprints.fingerprints:
            for layer, dims in fp.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    key = (layer, dim.index)
                    freq_map[key] = freq_map.get(key, 0) + 1

        # Sort by frequency desc, then layer asc, then dim asc
        sorted_items = sorted(freq_map.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))

        limit = max(1, max_features)
        selected = sorted_items[:limit]

        return [ProjectionFeature(layer=k[0], dimension=k[1], frequency=v) for k, v in selected]
