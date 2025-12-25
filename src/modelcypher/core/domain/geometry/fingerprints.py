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

import numpy as np

from modelcypher.core.domain._backend import get_default_backend
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

        # Create dense matrix on CPU first
        matrix_np = np.zeros((n, d), dtype=np.float32)

        for row, fp in enumerate(fingerprints.fingerprints):
            for layer, dims in fp.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    key = (layer, dim.index)
                    if key in feature_index:
                        col = feature_index[key]
                        matrix_np[row, col] = dim.activation

        # Move to backend
        X = self._backend.array(matrix_np)

        # 3. Normalize & Center
        # L2 Normalize rows
        norms = self._backend.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        self._backend.ones_like(X)
        mask = norms > 1e-9
        X = self._backend.where(mask, X / norms, X)

        # Center columns (Mean centering)
        means = self._backend.mean(X, axis=0, keepdims=True)
        X = X - means

        # 4. PCA via SVD
        U, S, Vt = self._backend.svd(X)

        # Take top 2
        # Coordinates = U[:, :2] * S[:2]
        coords = U[:, :2] * S[:2]

        # Extract points
        points = []
        coords_np = self._backend.to_numpy(coords)

        for i, fp in enumerate(fingerprints.fingerprints):
            pt = coords_np[i]
            points.append(
                ProjectionPoint(
                    id=fp.prime_id,
                    prime_id=fp.prime_id,
                    prime_text=fp.prime_text,
                    x=float(pt[0]),
                    y=float(pt[1]),
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
