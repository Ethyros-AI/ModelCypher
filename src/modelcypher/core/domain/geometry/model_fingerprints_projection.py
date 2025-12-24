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
import math


from modelcypher.core.domain.geometry.manifold_stitcher import ModelFingerprints


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
            raise ProjectionError(f"Projection requires at least 2 fingerprints (got {len(fingerprints.fingerprints)}).")

        if method in {ProjectionMethod.tsne, ProjectionMethod.umap}:
            raise ProjectionError(f"Projection method {method.value.upper()} is not available in-app yet.")

        feature_list = ModelFingerprintsProjection._select_features(
            fingerprints=fingerprints,
            max_features=max_features,
            layers=layers,
        )
        if len(feature_list) < 2:
            raise ProjectionError(f"Projection requires at least 2 features (got {len(feature_list)}).")

        included_layers = sorted(layers) if layers else None
        n = len(fingerprints.fingerprints)
        d = len(feature_list)

        feature_index = {
            (feature.layer, feature.dimension): idx for idx, feature in enumerate(feature_list)
        }

        matrix = [0.0] * (n * d)
        for row, fingerprint in enumerate(fingerprints.fingerprints):
            row_offset = row * d
            for layer, dims in fingerprint.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    col = feature_index.get((layer, dim.index))
                    if col is None:
                        continue
                    matrix[row_offset + col] = float(dim.activation)

        ModelFingerprintsProjection._normalize_rows(matrix, rows=n, cols=d)
        ModelFingerprintsProjection._center_columns(matrix, rows=n, cols=d)

        rng = _LCG(seed=seed)
        v1 = ModelFingerprintsProjection._power_iteration_top_eigenvector(
            matrix=matrix,
            rows=n,
            cols=d,
            rng=rng,
        )
        if v1 is None:
            raise ProjectionError(f"Projection requires at least 2 features (got {d}).")
        v2 = ModelFingerprintsProjection._power_iteration_top_eigenvector(
            matrix=matrix,
            rows=n,
            cols=d,
            rng=rng,
            orthogonal_to=v1,
        )
        if v2 is None:
            raise ProjectionError(f"Projection requires at least 2 features (got {d}).")

        points: list[ProjectionPoint] = []
        for row, fingerprint in enumerate(fingerprints.fingerprints):
            row_offset = row * d
            x = 0.0
            y = 0.0
            for col in range(d):
                value = matrix[row_offset + col]
                x += value * v1[col]
                y += value * v2[col]
            points.append(
                ProjectionPoint(
                    prime_id=fingerprint.prime_id,
                    prime_text=fingerprint.prime_text,
                    x=x,
                    y=y,
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

    @staticmethod
    def _normalize_rows(matrix: list[float], rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            return
        for row in range(rows):
            offset = row * cols
            sum_squares = 0.0
            for col in range(cols):
                value = matrix[offset + col]
                sum_squares += value * value
            if sum_squares <= 0:
                continue
            inv_norm = 1.0 / math.sqrt(sum_squares)
            for col in range(cols):
                matrix[offset + col] = matrix[offset + col] * inv_norm

    @staticmethod
    def _center_columns(matrix: list[float], rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            return
        sums = [0.0] * cols
        for row in range(rows):
            offset = row * cols
            for col in range(cols):
                sums[col] += matrix[offset + col]
        means = [value / float(rows) for value in sums]
        for row in range(rows):
            offset = row * cols
            for col in range(cols):
                matrix[offset + col] = matrix[offset + col] - means[col]

    @staticmethod
    def _power_iteration_top_eigenvector(
        matrix: list[float],
        rows: int,
        cols: int,
        rng: "_LCG",
        orthogonal_to: list[float] | None = None,
        max_iterations: int = 64,
        tolerance: float = 1e-6,
    ) -> list[float] | None:
        if rows <= 0 or cols <= 0:
            return None
        v = [rng.next_double() - 0.5 for _ in range(cols)]
        if orthogonal_to is not None:
            ModelFingerprintsProjection._orthogonalize(v, orthogonal_to)
        if not ModelFingerprintsProjection._normalize(v):
            return None

        temp = [0.0] * rows
        w = [0.0] * cols

        for _ in range(max_iterations):
            for row in range(rows):
                offset = row * cols
                total = 0.0
                for col in range(cols):
                    total += matrix[offset + col] * v[col]
                temp[row] = total

            for col in range(cols):
                w[col] = 0.0
            for row in range(rows):
                offset = row * cols
                t = temp[row]
                for col in range(cols):
                    w[col] += matrix[offset + col] * t

            if orthogonal_to is not None:
                ModelFingerprintsProjection._orthogonalize(w, orthogonal_to)
            if not ModelFingerprintsProjection._normalize(w):
                break

            delta = 0.0
            for i in range(cols):
                diff = w[i] - v[i]
                delta += diff * diff
            if delta < tolerance * tolerance:
                v = list(w)
                break
            v = list(w)

        return v

    @staticmethod
    def _normalize(vector: list[float]) -> bool:
        sum_squares = 0.0
        for value in vector:
            sum_squares += value * value
        if sum_squares <= 0:
            return False
        inv_norm = 1.0 / math.sqrt(sum_squares)
        for i in range(len(vector)):
            vector[i] *= inv_norm
        return True

    @staticmethod
    def _orthogonalize(vector: list[float], basis: list[float]) -> None:
        if len(vector) != len(basis):
            return
        dot = 0.0
        for i in range(len(vector)):
            dot += vector[i] * basis[i]
        for i in range(len(vector)):
            vector[i] -= dot * basis[i]


class _LCG:
    def __init__(self, seed: int) -> None:
        self.state = seed if seed != 0 else 0xDEAD_BEEF

    def next_uint64(self) -> int:
        self.state = (self.state * 6_364_136_223_846_793_005 + 1) & 0xFFFFFFFFFFFFFFFF
        return self.state

    def next_double(self) -> float:
        value = self.next_uint64() >> 11
        return float(value) / float(1 << 53)
