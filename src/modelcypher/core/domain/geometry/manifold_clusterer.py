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
from uuid import uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.exceptions import EstimatorError
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldRegion,
    RegionThresholds,
    RegionQueryResult,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

logger = logging.getLogger(__name__)


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All clustering parameters are derived from data:
# - epsilon: derived from data distribution (no hardcoded value)
# - intrinsic dimension: ALWAYS computed (it's a measurement, not optional)
# - max_clusters: no artificial limit (let data determine cluster count)
# =============================================================================


@dataclass(frozen=True)
class ClusteringResult:
    regions: tuple[ManifoldRegion, ...]
    noise_points: tuple[ManifoldPoint, ...]
    new_clusters_formed: int
    clusters_merged: int
    points_assigned_to_existing: int


class ManifoldClusterer:
    def __init__(self) -> None:
        pass

    def cluster(self, points: list[ManifoldPoint]) -> ClusteringResult:
        if not points:
            return ClusteringResult(
                regions=(),
                noise_points=(),
                new_clusters_formed=0,
                clusters_merged=0,
                points_assigned_to_existing=0,
            )

        # Compute geodesic distance matrix over the full point set.
        # Geodesic distance accounts for manifold curvature - chord distance is
        # only an approximation that fails in high-dimensional curved spaces.
        geodesic_matrix = self._compute_geodesic_matrix(points)

        epsilon = self._resolve_epsilon(geodesic_matrix)
        min_cluster_size = 2

        labels = [-1 for _ in points]
        cluster_id = 0

        for i in range(len(points)):
            if labels[i] != -1:
                continue
            neighbors = self._region_query_geodesic(geodesic_matrix, i, epsilon)
            if len(neighbors) < min_cluster_size:
                labels[i] = -2
            else:
                self._expand_cluster_geodesic(
                    geodesic_matrix,
                    labels,
                    i,
                    neighbors,
                    cluster_id,
                    epsilon,
                    min_cluster_size,
                )
                cluster_id += 1

        regions: list[ManifoldRegion] = []
        noise_points: list[ManifoldPoint] = []

        for cluster in range(cluster_id):
            member_indices = [idx for idx, label in enumerate(labels) if label == cluster]
            member_points = [points[idx] for idx in member_indices]
            # Recompute geodesic matrix for cluster points only.
            # This is more correct than extracting a submatrix because the k-NN
            # graph structure of the cluster subset may differ from the full set.
            cluster_geodesic = self._compute_geodesic_matrix(member_points)
            region = self._build_region_geodesic(member_points, cluster_geodesic)
            if region is not None:
                regions.append(region)

        for idx, label in enumerate(labels):
            if label == -2:
                noise_points.append(points[idx])

        logger.info("Full clustering: %s regions, %s noise points", len(regions), len(noise_points))

        return ClusteringResult(
            regions=tuple(regions),
            noise_points=tuple(noise_points),
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
                regions=tuple(existing_regions),
                noise_points=tuple(existing_noise),
                new_clusters_formed=0,
                clusters_merged=0,
                points_assigned_to_existing=0,
            )

        updated_regions = list(existing_regions)
        noise_points = list(existing_noise)
        assigned_to_existing = 0
        new_clusters_formed = 0
        epsilon = self._resolve_epsilon(self._compute_geodesic_matrix(new_points))
        min_cluster_size = 2

        # For incremental assignment, compute geodesic distance between each new point
        # and region centroids. When comparing a single point to a single centroid,
        # the k-NN graph has exactly one edge, so geodesic = chord by construction.
        # We still use the geodesic code path for consistency.
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
            if nearest_region is not None and nearest_distance <= epsilon:
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

        if len(noise_points) >= min_cluster_size:
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
            regions=tuple(final_regions),
            noise_points=tuple(noise_points),
            new_clusters_formed=new_clusters_formed,
            clusters_merged=merge_count,
            points_assigned_to_existing=assigned_to_existing,
        )

    def _compute_geodesic_matrix(self, points: list[ManifoldPoint]):
        """Compute pairwise geodesic distances via k-NN graph.

        Geodesic distance accounts for curvature. Chord distance can
        under- or overestimate distances in curved spaces.

        Returns a Backend array (not numpy).
        """
        backend = get_default_backend()

        if len(points) <= 1:
            return backend.zeros((len(points), len(points)))

        # Build feature matrix using Backend
        rows = [backend.array(p.feature_vector) for p in points]
        features = backend.stack(rows, axis=0)

        rg = RiemannianGeometry(backend)

        # Use connectivity-derived k to avoid arbitrary neighborhood choices.
        result = rg.geodesic_distances(features, k_neighbors=None, refine_iterations=0)
        return result.distances

    def _resolve_epsilon(self, geodesic_matrix) -> float:
        """Epsilon is always derived from data - no configuration."""
        return self._derive_epsilon(geodesic_matrix)

    def _derive_epsilon(self, geodesic_matrix) -> float:
        """Derive epsilon from the median nearest-neighbor geodesic distance."""
        backend = get_default_backend()
        backend.eval(geodesic_matrix)
        n = int(backend.shape(geodesic_matrix)[0])
        if n <= 1:
            return 0.0
        inf = float("inf")
        # Use where() to avoid 0 * inf = nan
        eye_mask = backend.eye(n) > 0  # Diagonal is 1.0, off-diagonal is 0.0
        inf_mask = backend.where(eye_mask, backend.full((n, n), inf), backend.zeros((n, n)))
        masked = geodesic_matrix + inf_mask
        nearest = backend.min(masked, axis=1)
        backend.eval(nearest)
        mid = n // 2
        if n % 2 == 1:
            part = backend.argpartition(nearest, mid)
            prefix = backend.take(part, backend.arange(mid + 1), axis=0)
            median = backend.max(backend.take(nearest, prefix, axis=0))
            backend.eval(median)
            median_val = float(backend.to_scalar(backend.squeeze(median)))
            eps_floor = division_epsilon(backend, geodesic_matrix)
            if median_val <= eps_floor:
                max_dist_arr = backend.max(geodesic_matrix)
                backend.eval(max_dist_arr)
                max_dist = float(backend.to_scalar(max_dist_arr))
                scale = max(1.0, max_dist)
                eps_floor = eps_floor * scale
            return max(median_val, eps_floor)
        low_part = backend.argpartition(nearest, mid - 1)
        low_prefix = backend.take(low_part, backend.arange(mid), axis=0)
        lower = backend.max(backend.take(nearest, low_prefix, axis=0))
        high_part = backend.argpartition(nearest, mid)
        high_prefix = backend.take(high_part, backend.arange(mid + 1), axis=0)
        upper = backend.max(backend.take(nearest, high_prefix, axis=0))
        backend.eval(lower, upper)
        median = (backend.squeeze(lower) + backend.squeeze(upper)) / 2.0
        backend.eval(median)
        median_val = float(backend.to_scalar(median))
        eps_floor = division_epsilon(backend, geodesic_matrix)
        if median_val <= eps_floor:
            max_dist_arr = backend.max(geodesic_matrix)
            backend.eval(max_dist_arr)
            max_dist = float(backend.to_scalar(max_dist_arr))
            scale = max(1.0, max_dist)
            eps_floor = eps_floor * scale
        return max(median_val, eps_floor)

    def _geodesic_distance_pair(self, p1: ManifoldPoint, p2: ManifoldPoint) -> float:
        """Compute geodesic distance between two points.

        For exactly 2 points, the k-NN graph has a single edge, so
        geodesic = chord by construction. We use the geodesic code
        path for consistency.
        """
        backend = get_default_backend()
        matrix = self._compute_geodesic_matrix([p1, p2])
        backend.eval(matrix)
        row0 = backend.take(matrix, backend.array([0]), axis=0)
        row0 = backend.squeeze(row0, axis=0)
        val = backend.take(row0, backend.array([1]), axis=0)
        backend.eval(val)
        return float(backend.to_scalar(backend.squeeze(val)))

    def _region_query_geodesic(
        self, geodesic_matrix, point_index: int, epsilon: float
    ) -> list[int]:
        """Find epsilon-neighborhood using precomputed geodesic distances.

        Args:
            geodesic_matrix: Backend array of pairwise geodesic distances.
            point_index: Index of the query point.

        Returns:
            List of neighbor indices within epsilon distance.
        """
        backend = get_default_backend()
        backend.eval(geodesic_matrix)
        row = backend.take(geodesic_matrix, backend.array([point_index]), axis=0)
        row = backend.squeeze(row, axis=0)
        backend.eval(row)
        mask = row <= epsilon
        mask_int = backend.astype(mask, "int32")
        count_arr = backend.sum(mask_int)
        backend.eval(mask_int, count_arr)
        count = int(backend.to_scalar(count_arr))
        if count <= 0:
            return []
        neg_mask = -mask_int
        kth = max(0, count - 1)
        partitioned = backend.argpartition(neg_mask, kth)
        neighbor_idx = backend.take(partitioned, backend.arange(count), axis=0)
        backend.eval(neighbor_idx)
        neighbors = backend.tolist(neighbor_idx)
        return [int(x) for x in neighbors]

    def _expand_cluster_geodesic(
        self,
        geodesic_matrix,
        labels: list[int],
        point_index: int,
        neighbors: list[int],
        cluster_id: int,
        epsilon: float,
        min_cluster_size: int,
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
                neighbor_neighbors = self._region_query_geodesic(
                    geodesic_matrix, neighbor_index, epsilon
                )
                if len(neighbor_neighbors) >= min_cluster_size:
                    for nn in neighbor_neighbors:
                        if nn not in seed_set:
                            seed_set.append(nn)
            i += 1

    def _build_region_geodesic(
        self,
        points: list[ManifoldPoint],
        geodesic_matrix,
        existing_id: object | None = None,
        existing_member_ids: list[object] | None = None,
    ) -> ManifoldRegion | None:
        """Build region using geodesic distances and Fréchet mean."""
        if not points:
            return None

        backend = get_default_backend()

        # Use Fréchet mean (manifold-aware center) instead of arithmetic mean
        centroid, centroid_idx = self._compute_centroid_geodesic(points, geodesic_matrix)

        # Radius is max geodesic distance from centroid to any member
        row = backend.take(geodesic_matrix, backend.array([centroid_idx]), axis=0)
        row = backend.squeeze(row, axis=0)
        backend.eval(row)
        radius_arr = backend.max(row)
        backend.eval(radius_arr)
        radius = float(backend.to_scalar(radius_arr))
        dominant_gates = self._compute_dominant_gates(points)

        # ALWAYS compute intrinsic dimension - it's a measurement, not optional
        intrinsic_dimension = None
        if len(points) >= 3:
            intrinsic_dimension = self._estimate_intrinsic_dimension(points)

        # Derive classification thresholds from actual point data
        entropies = [pt.mean_entropy for pt in points]
        variances = [pt.entropy_variance for pt in points]
        coherences = [pt.mean_gate_similarity for pt in points]
        thresholds = RegionThresholds.from_percentiles(
            entropies, variances, coherences
        )
        region_type = ManifoldRegion.classify(centroid, thresholds)

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
        self, points: list[ManifoldPoint], geodesic_matrix
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
                    mean_gate_similarity=0.0,
                    dominant_gate_category=0.0,
                    entropy_path_correlation=0.0,
                    assessment_strength=0.0,
                    prompt_hash="centroid",
                ),
                0,
            )

        if len(points) == 1:
            return points[0], 0

        # Find geodesic medoid: point minimizing sum of squared geodesic distances
        backend = get_default_backend()
        squared = geodesic_matrix * geodesic_matrix
        sum_squared = backend.sum(squared, axis=1)
        backend.eval(sum_squared)
        medoid_idx_arr = backend.argmin(sum_squared)
        backend.eval(medoid_idx_arr)
        medoid_idx = int(backend.to_scalar(medoid_idx_arr))

        return points[medoid_idx], medoid_idx

    def _merge_overlapping_regions_geodesic(
        self, regions: list[ManifoldRegion]
    ) -> tuple[list[ManifoldRegion], int]:
        """Merge overlapping regions using geodesic distance between centroids."""
        if len(regions) <= 1:
            return regions, 0

        backend = get_default_backend()

        # Build geodesic matrix for all region centroids
        centroids = [r.centroid for r in regions]
        centroid_geodesic = self._compute_geodesic_matrix(centroids)
        backend.eval(centroid_geodesic)

        merged_regions: list[ManifoldRegion] = []
        merged: set[str] = set()
        merge_count = 0

        for i, region in enumerate(regions):
            if str(region.id) in merged:
                continue
            current_region = region
            merged_points = [current_region.centroid]
            merged_ids = list(current_region.member_ids)

            row = backend.take(centroid_geodesic, backend.array([i]), axis=0)
            row = backend.squeeze(row, axis=0)
            backend.eval(row)
            row_list = backend.tolist(row)
            for j in range(i + 1, len(regions)):
                other = regions[j]
                if str(other.id) in merged:
                    continue
                # Use geodesic distance between centroids
                distance = float(row_list[j])
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

    def _enforce_max_clusters(self, regions: list[ManifoldRegion]) -> list[ManifoldRegion]:
        """No artificial cluster limit - let data determine cluster count."""
        return regions

    def find_nearest_region(
        self, point: ManifoldPoint, regions: list[ManifoldRegion]
    ) -> RegionQueryResult:
        # Derive classification thresholds from region centroids (or point if no regions)
        if regions:
            centroids = [r.centroid for r in regions]
            entropies = [c.mean_entropy for c in centroids]
            variances = [c.entropy_variance for c in centroids]
            coherences = [c.mean_gate_similarity for c in centroids]
        else:
            entropies = [point.mean_entropy]
            variances = [point.entropy_variance]
            coherences = [point.mean_gate_similarity]
        thresholds = RegionThresholds.from_percentiles(
            entropies, variances, coherences
        )

        if not regions:
            return RegionQueryResult(
                nearest_region=None,
                distance=float("inf"),
                is_within_region=False,
                suggested_character=ManifoldRegion.classify(point, thresholds),
                confidence=0.0,
            )

        # Use geodesic distance for each point-centroid comparison.
        # For 2 points, geodesic = chord by construction (k-NN has one edge).
        nearest_region: ManifoldRegion | None = None
        nearest_distance = float("inf")
        for region in regions:
            distance = self._geodesic_distance_pair(point, region.centroid)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_region = region

        is_within = nearest_region is not None and nearest_distance <= nearest_region.radius
        if nearest_region is not None:
            backend = get_default_backend()
            eps = division_epsilon(backend, backend.array([nearest_region.radius]))
            confidence = max(0.0, 1.0 - (nearest_distance / (nearest_region.radius + eps)))
        else:
            confidence = 0.0

        return RegionQueryResult(
            nearest_region=nearest_region,
            distance=nearest_distance,
            is_within_region=is_within,
            suggested_character=ManifoldRegion.classify(point, thresholds),
            confidence=confidence,
        )
