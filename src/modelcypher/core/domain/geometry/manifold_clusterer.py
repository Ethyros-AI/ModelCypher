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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
)
from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldRegion,
    RegionQueryResult,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.core.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
    epsilon: float = 0.3
    min_points: int = 5
    compute_intrinsic_dimension: bool = True
    max_clusters: int = 50


@dataclass(frozen=True)
class ClusteringResult:
    regions: list[ManifoldRegion]
    noise_points: list[ManifoldPoint]
    new_clusters_formed: int
    clusters_merged: int
    points_assigned_to_existing: int


class ManifoldClusterer:
    Configuration = Configuration

    def __init__(self, configuration: Configuration = Configuration()) -> None:
        self.config = configuration

    def cluster(self, points: list[ManifoldPoint]) -> ClusteringResult:
        if not points:
            return ClusteringResult(
                regions=[],
                noise_points=[],
                new_clusters_formed=0,
                clusters_merged=0,
                points_assigned_to_existing=0,
            )

        # Compute geodesic distance matrix over the full point set.
        # Geodesic distance accounts for manifold curvature - Euclidean is
        # only an approximation that fails in high-dimensional curved spaces.
        geodesic_matrix = self._compute_geodesic_matrix(points)

        labels = [-1 for _ in points]
        cluster_id = 0

        for i in range(len(points)):
            if labels[i] != -1:
                continue
            neighbors = self._region_query_geodesic(geodesic_matrix, i)
            if len(neighbors) < self.config.min_points:
                labels[i] = -2
            else:
                self._expand_cluster_geodesic(geodesic_matrix, labels, i, neighbors, cluster_id)
                cluster_id += 1

        regions: list[ManifoldRegion] = []
        noise_points: list[ManifoldPoint] = []

        for cluster in range(cluster_id):
            member_indices = [idx for idx, label in enumerate(labels) if label == cluster]
            member_points = [points[idx] for idx in member_indices]
            # Extract the geodesic submatrix for this cluster's points
            cluster_geodesic = geodesic_matrix[np.ix_(member_indices, member_indices)]
            region = self._build_region_geodesic(member_points, cluster_geodesic)
            if region is not None:
                regions.append(region)

        for idx, label in enumerate(labels):
            if label == -2:
                noise_points.append(points[idx])

        logger.info("Full clustering: %s regions, %s noise points", len(regions), len(noise_points))

        return ClusteringResult(
            regions=regions,
            noise_points=noise_points,
            new_clusters_formed=len(regions),
            clusters_merged=0,
            points_assigned_to_existing=0,
        )

    def cluster_incremental(
        self,
        new_points: list[ManifoldPoint],
        existing_regions: list[ManifoldRegion],
        existing_noise: list[ManifoldPoint],
    ) -> ClusteringResult:
        if not new_points:
            return ClusteringResult(
                regions=existing_regions,
                noise_points=existing_noise,
                new_clusters_formed=0,
                clusters_merged=0,
                points_assigned_to_existing=0,
            )

        updated_regions = list(existing_regions)
        noise_points = list(existing_noise)
        assigned_to_existing = 0
        new_clusters_formed = 0

        # For incremental assignment, compute geodesic distance between each new point
        # and region centroids. When comparing a single point to a single centroid,
        # the k-NN graph has exactly one edge, so geodesic = Euclidean by construction.
        # We still use the geodesic code path for correctness and consistency.
        region_point_additions: dict[str, list[ManifoldPoint]] = {}
        for point in new_points:
            nearest_region = None
            nearest_distance = float("inf")
            for region in updated_regions:
                # Geodesic distance between point and centroid
                distance = self._geodesic_distance_pair(point, region.centroid)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_region = region
            if nearest_region is not None and nearest_distance <= self.config.epsilon:
                region_point_additions.setdefault(str(nearest_region.id), []).append(point)
                assigned_to_existing += 1
            else:
                noise_points.append(point)

        for region_id, additions in region_point_additions.items():
            idx = next(
                (i for i, region in enumerate(updated_regions) if str(region.id) == region_id), None
            )
            if idx is None:
                continue
            region = updated_regions[idx]
            all_members = [region.centroid] + additions
            # Compute geodesic matrix for region update
            geodesic_matrix = self._compute_geodesic_matrix(all_members)
            updated_region = self._build_region_geodesic(
                all_members,
                geodesic_matrix,
                existing_id=region.id,
                existing_member_ids=region.member_ids + [pt.id for pt in additions],
            )
            if updated_region is not None:
                updated_regions[idx] = updated_region

        if len(noise_points) >= self.config.min_points:
            noise_cluster_result = self.cluster(noise_points)
            updated_regions.extend(noise_cluster_result.regions)
            noise_points = noise_cluster_result.noise_points
            new_clusters_formed = noise_cluster_result.new_clusters_formed

        merged_regions, merge_count = self._merge_overlapping_regions_geodesic(updated_regions)
        final_regions = self._enforce_max_clusters(merged_regions)

        logger.info(
            "Incremental clustering: %s assigned, %s new, %s merged",
            assigned_to_existing,
            new_clusters_formed,
            merge_count,
        )

        return ClusteringResult(
            regions=final_regions,
            noise_points=noise_points,
            new_clusters_formed=new_clusters_formed,
            clusters_merged=merge_count,
            points_assigned_to_existing=assigned_to_existing,
        )

    def _region_query(self, points: list[ManifoldPoint], point_index: int) -> list[int]:
        point = points[point_index]
        neighbors: list[int] = []
        for j in range(len(points)):
            if point.distance(points[j]) <= self.config.epsilon:
                neighbors.append(j)
        return neighbors

    def _expand_cluster(
        self,
        points: list[ManifoldPoint],
        labels: list[int],
        point_index: int,
        neighbors: list[int],
        cluster_id: int,
    ) -> None:
        labels[point_index] = cluster_id
        seed_set = list(neighbors)
        i = 0
        while i < len(seed_set):
            neighbor_index = seed_set[i]
            if labels[neighbor_index] == -2:
                labels[neighbor_index] = cluster_id
            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id
                neighbor_neighbors = self._region_query(points, neighbor_index)
                if len(neighbor_neighbors) >= self.config.min_points:
                    for nn in neighbor_neighbors:
                        if nn not in seed_set:
                            seed_set.append(nn)
            i += 1

    def _compute_geodesic_matrix(self, points: list[ManifoldPoint]) -> np.ndarray:
        """Compute pairwise geodesic distances via k-NN graph.

        Geodesic distance is the correct metric on curved manifolds.
        Euclidean distance is only valid in flat spaces and systematically
        underestimates (positive curvature) or overestimates (negative curvature)
        true manifold distances.
        """
        if len(points) <= 1:
            return np.zeros((len(points), len(points)))

        # Build feature matrix
        features = np.array([p.feature_vector for p in points], dtype=np.float64)

        # Use RiemannianGeometry for geodesic computation
        from modelcypher.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()
        rg = RiemannianGeometry(backend)

        # k_neighbors scales with sqrt(n) for good graph connectivity
        k_neighbors = max(2, min(len(points) - 1, int(np.sqrt(len(points)))))
        result = rg.geodesic_distances(features, k_neighbors=k_neighbors)
        return np.asarray(result.distances)

    def _geodesic_distance_pair(self, p1: ManifoldPoint, p2: ManifoldPoint) -> float:
        """Compute geodesic distance between two points.

        For exactly 2 points, the k-NN graph has a single edge, so
        geodesic = Euclidean by construction. We use the geodesic code
        path for consistency and correctness.
        """
        matrix = self._compute_geodesic_matrix([p1, p2])
        return float(matrix[0, 1])

    def _region_query_geodesic(
        self, geodesic_matrix: np.ndarray, point_index: int
    ) -> list[int]:
        """Find epsilon-neighborhood using precomputed geodesic distances."""
        distances = geodesic_matrix[point_index, :]
        neighbors: list[int] = []
        for j, dist in enumerate(distances):
            if dist <= self.config.epsilon:
                neighbors.append(j)
        return neighbors

    def _expand_cluster_geodesic(
        self,
        geodesic_matrix: np.ndarray,
        labels: list[int],
        point_index: int,
        neighbors: list[int],
        cluster_id: int,
    ) -> None:
        """DBSCAN cluster expansion using geodesic distances."""
        labels[point_index] = cluster_id
        seed_set = list(neighbors)
        i = 0
        while i < len(seed_set):
            neighbor_index = seed_set[i]
            if labels[neighbor_index] == -2:
                labels[neighbor_index] = cluster_id
            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id
                neighbor_neighbors = self._region_query_geodesic(geodesic_matrix, neighbor_index)
                if len(neighbor_neighbors) >= self.config.min_points:
                    for nn in neighbor_neighbors:
                        if nn not in seed_set:
                            seed_set.append(nn)
            i += 1

    def _build_region_geodesic(
        self,
        points: list[ManifoldPoint],
        geodesic_matrix: np.ndarray,
        existing_id: object | None = None,
        existing_member_ids: list[object] | None = None,
    ) -> ManifoldRegion | None:
        """Build region using geodesic distances and Fréchet mean."""
        if not points:
            return None

        # Use Fréchet mean (manifold-aware center) instead of arithmetic mean
        centroid, centroid_idx = self._compute_centroid_geodesic(points, geodesic_matrix)

        # Radius is max geodesic distance from centroid to any member
        radius = float(np.max(geodesic_matrix[centroid_idx, :]))
        dominant_gates = self._compute_dominant_gates(points)

        intrinsic_dimension = None
        if self.config.compute_intrinsic_dimension and len(points) >= 3:
            intrinsic_dimension = self._estimate_intrinsic_dimension(points)

        region_type = ManifoldRegion.classify(centroid)

        return ManifoldRegion(
            id=existing_id or uuid4(),
            region_type=region_type,
            centroid=centroid,
            member_count=len(points),
            member_ids=existing_member_ids or [pt.id for pt in points],
            dominant_gates=dominant_gates,
            intrinsic_dimension=intrinsic_dimension,
            radius=radius,
        )

    def _compute_centroid_geodesic(
        self, points: list[ManifoldPoint], geodesic_matrix: np.ndarray
    ) -> tuple[ManifoldPoint, int]:
        """Compute Fréchet mean (geodesic medoid) as manifold center.

        The Fréchet mean minimizes sum of squared geodesic distances.
        For discrete point sets, this is the geodesic medoid (the point
        that minimizes total geodesic distance to all others).

        Returns the centroid point and its index in the points list.
        """
        if not points:
            return (
                ManifoldPoint(
                    id=uuid4(),
                    mean_entropy=0.0,
                    entropy_variance=0.0,
                    first_token_entropy=0.0,
                    gate_count=0,
                    mean_gate_confidence=0.0,
                    dominant_gate_category=0.0,
                    entropy_path_correlation=0.0,
                    assessment_strength=0.0,
                    prompt_hash="centroid",
                ),
                0,
            )

        if len(points) == 1:
            return points[0], 0

        # Find geodesic medoid: point minimizing sum of geodesic distances
        # Use squared geodesic distances for Fréchet mean
        sum_squared_distances = np.sum(geodesic_matrix**2, axis=1)
        medoid_idx = int(np.argmin(sum_squared_distances))

        return points[medoid_idx], medoid_idx

    def _merge_overlapping_regions_geodesic(
        self, regions: list[ManifoldRegion]
    ) -> tuple[list[ManifoldRegion], int]:
        """Merge overlapping regions using geodesic distance between centroids."""
        if len(regions) <= 1:
            return regions, 0

        # Build geodesic matrix for all region centroids
        centroids = [r.centroid for r in regions]
        centroid_geodesic = self._compute_geodesic_matrix(centroids)

        merged_regions: list[ManifoldRegion] = []
        merged: set[str] = set()
        merge_count = 0

        for i, region in enumerate(regions):
            if str(region.id) in merged:
                continue
            current_region = region
            merged_points = [current_region.centroid]
            merged_ids = list(current_region.member_ids)

            for j in range(i + 1, len(regions)):
                other = regions[j]
                if str(other.id) in merged:
                    continue
                # Use geodesic distance between centroids
                distance = float(centroid_geodesic[i, j])
                overlap_threshold = current_region.radius + other.radius
                if distance < overlap_threshold:
                    merged.add(str(other.id))
                    merged_points.append(other.centroid)
                    merged_ids.extend(other.member_ids)
                    merge_count += 1

            if len(merged_points) > 1:
                # Recompute geodesic matrix for merged points
                merge_geodesic = self._compute_geodesic_matrix(merged_points)
                rebuilt = self._build_region_geodesic(
                    merged_points,
                    merge_geodesic,
                    existing_id=current_region.id,
                    existing_member_ids=merged_ids,
                )
                if rebuilt is not None:
                    current_region = rebuilt
            merged_regions.append(current_region)

        return merged_regions, merge_count

    def _build_region(
        self,
        points: list[ManifoldPoint],
        existing_id: object | None = None,
        existing_member_ids: list[object] | None = None,
    ) -> ManifoldRegion | None:
        if not points:
            return None

        centroid = self._compute_centroid(points)
        radius = max((point.distance(centroid) for point in points), default=0.0)
        dominant_gates = self._compute_dominant_gates(points)

        intrinsic_dimension = None
        if self.config.compute_intrinsic_dimension and len(points) >= 3:
            intrinsic_dimension = self._estimate_intrinsic_dimension(points)

        region_type = ManifoldRegion.classify(centroid)

        return ManifoldRegion(
            id=existing_id or uuid4(),
            region_type=region_type,
            centroid=centroid,
            member_count=len(points),
            member_ids=existing_member_ids or [pt.id for pt in points],
            dominant_gates=dominant_gates,
            intrinsic_dimension=intrinsic_dimension,
            radius=radius,
        )

    def _compute_centroid(self, points: list[ManifoldPoint]) -> ManifoldPoint:
        if not points:
            return ManifoldPoint(
                id=uuid4(),
                mean_entropy=0.0,
                entropy_variance=0.0,
                first_token_entropy=0.0,
                gate_count=0,
                mean_gate_confidence=0.0,
                dominant_gate_category=0.0,
                entropy_path_correlation=0.0,
                assessment_strength=0.0,
                prompt_hash="centroid",
            )

        count = float(len(points))
        mean_entropy = sum(point.mean_entropy for point in points) / count
        entropy_variance = sum(point.entropy_variance for point in points) / count
        first_token_entropy = sum(point.first_token_entropy for point in points) / count
        gate_count = int(sum(float(point.gate_count) for point in points) / count)
        mean_gate_confidence = sum(point.mean_gate_confidence for point in points) / count
        dominant_gate_category = sum(point.dominant_gate_category for point in points) / count
        entropy_path_correlation = sum(point.entropy_path_correlation for point in points) / count
        assessment_strength = sum(point.assessment_strength for point in points) / count

        return ManifoldPoint(
            id=uuid4(),
            mean_entropy=mean_entropy,
            entropy_variance=entropy_variance,
            first_token_entropy=first_token_entropy,
            gate_count=gate_count,
            mean_gate_confidence=mean_gate_confidence,
            dominant_gate_category=dominant_gate_category,
            entropy_path_correlation=entropy_path_correlation,
            assessment_strength=assessment_strength,
            prompt_hash="centroid",
        )

    def _compute_dominant_gates(self, points: list[ManifoldPoint]) -> list[str]:
        category_counts: dict[int, int] = {}
        known_gates = [
            "INIT",
            "REASON",
            "BRANCH",
            "LOOP",
            "CONCLUDE",
            "RECALL",
            "COMPARE",
            "SYNTHESIZE",
            "EVALUATE",
            "OUTPUT",
        ]

        for point in points:
            raw_index = point.dominant_gate_category * float(len(known_gates) - 1)
            index = min(max(int(round(raw_index)), 0), len(known_gates) - 1)
            category_counts[index] = category_counts.get(index, 0) + 1

        sorted_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
        dominant_gates: list[str] = []
        for index, _ in sorted_categories[:3]:
            if index < len(known_gates):
                dominant_gates.append(known_gates[index])
        return dominant_gates

    def _estimate_intrinsic_dimension(self, points: list[ManifoldPoint]) -> float | None:
        if len(points) < 3:
            return None
        double_points = [[float(value) for value in point.feature_vector] for point in points]
        try:
            estimate = IntrinsicDimension.compute_two_nn(double_points)
            return estimate.intrinsic_dimension
        except EstimatorError as exc:
            logger.debug("Failed to estimate intrinsic dimension: %s", exc)
            return None

    def _merge_overlapping_regions(
        self, regions: list[ManifoldRegion]
    ) -> tuple[list[ManifoldRegion], int]:
        if len(regions) <= 1:
            return regions, 0

        merged_regions: list[ManifoldRegion] = []
        merged: set[str] = set()
        merge_count = 0

        for i, region in enumerate(regions):
            if str(region.id) in merged:
                continue
            current_region = region
            merged_points = [current_region.centroid]
            merged_ids = list(current_region.member_ids)

            for j in range(i + 1, len(regions)):
                other = regions[j]
                if str(other.id) in merged:
                    continue
                distance = current_region.centroid.distance(other.centroid)
                overlap_threshold = current_region.radius + other.radius
                if distance < overlap_threshold:
                    merged.add(str(other.id))
                    merged_points.append(other.centroid)
                    merged_ids.extend(other.member_ids)
                    merge_count += 1

            if len(merged_points) > 1:
                rebuilt = self._build_region(
                    merged_points,
                    existing_id=current_region.id,
                    existing_member_ids=merged_ids,
                )
                if rebuilt is not None:
                    current_region = rebuilt
            merged_regions.append(current_region)

        return merged_regions, merge_count

    def _enforce_max_clusters(self, regions: list[ManifoldRegion]) -> list[ManifoldRegion]:
        if len(regions) <= self.config.max_clusters:
            return regions
        sorted_regions = sorted(
            regions,
            key=lambda region: (region.member_count, region.updated_at),
        )
        return list(sorted_regions[-self.config.max_clusters :])

    def find_nearest_region(
        self, point: ManifoldPoint, regions: list[ManifoldRegion]
    ) -> RegionQueryResult:
        if not regions:
            return RegionQueryResult(
                nearest_region=None,
                distance=float("inf"),
                is_within_region=False,
                suggested_type=ManifoldRegion.classify(point),
                confidence=0.0,
            )

        # Use geodesic distance for each point-centroid comparison.
        # For 2 points, geodesic = Euclidean by construction (k-NN has one edge).
        nearest_region: ManifoldRegion | None = None
        nearest_distance = float("inf")
        for region in regions:
            distance = self._geodesic_distance_pair(point, region.centroid)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_region = region

        is_within = nearest_region is not None and nearest_distance <= nearest_region.radius
        if nearest_region is not None:
            confidence = max(
                0.0, 1.0 - (nearest_distance / (nearest_region.radius + self.config.epsilon))
            )
        else:
            confidence = 0.0

        return RegionQueryResult(
            nearest_region=nearest_region,
            distance=nearest_distance,
            is_within_region=is_within,
            suggested_type=ManifoldRegion.classify(point),
            confidence=confidence,
        )
