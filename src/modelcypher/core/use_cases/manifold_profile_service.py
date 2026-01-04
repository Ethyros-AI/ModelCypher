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
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_clusterer import ManifoldClusterer
from modelcypher.core.domain.geometry.manifold_profile import (
    InterventionSuggestion,
    ManifoldPoint,
    ManifoldProfile,
    ManifoldRegion,
    RegionQueryResult,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.thermo_path_integration import CombinedMeasurement
from modelcypher.ports.storage import ManifoldProfileStore

logger = logging.getLogger(__name__)


class ManifoldProfileService:
    # ==========================================================================
    # NO CONFIGURATION CLASSES
    # ==========================================================================
    # All parameters are derived from data:
    # - Clustering: always attempted, algorithm determines if enough structure exists
    # - Distance thresholds: derived from data distribution (percentiles)
    # - All clustering parameters: derived from data by ManifoldClusterer
    # ==========================================================================

    def __init__(
        self,
        store: ManifoldProfileStore,
    ) -> None:
        self.store = store
        self.clusterer = ManifoldClusterer()
        self._pending_points: dict[str, list[ManifoldPoint]] = {}

    def record_observation(
        self,
        measurement: CombinedMeasurement,
        model_id: str,
        model_name: str,
        prompt_hash: str,
        intervention_level: int | None = None,
    ) -> None:
        point = ManifoldPoint.from_measurement(
            measurement=measurement,
            prompt_hash=prompt_hash,
            intervention_level=intervention_level,
        )
        self.record_point(point, model_id=model_id, model_name=model_name)

    def record_point(self, point: ManifoldPoint, model_id: str, model_name: str) -> None:
        self._pending_points.setdefault(model_id, []).append(point)
        pending_count = len(self._pending_points.get(model_id, []))
        if pending_count >= 10:
            self.flush_pending_points(model_id=model_id, model_name=model_name)
        logger.debug("Recorded point for %s (pending=%s)", model_id, pending_count)

    def flush_pending_points(self, model_id: str, model_name: str) -> None:
        points = self._pending_points.get(model_id) or []
        if not points:
            return

        profile = self.store.load(model_id)
        if profile is None:
            profile = ManifoldProfile(
                id=uuid4(),
                model_id=model_id,
                model_name=model_name,
            )

        updated_profile = ManifoldProfile(
            id=profile.id,
            model_id=profile.model_id,
            model_name=profile.model_name,
            regions=list(profile.regions),
            recent_points=list(profile.recent_points) + points,
            total_point_count=profile.total_point_count + len(points),
            created_at=profile.created_at,
            updated_at=datetime.utcnow(),
            version=profile.version,
        )

        # Always attempt clustering - the algorithm determines if enough structure exists
        # ManifoldClusterer derives all parameters from data (epsilon, min_samples, etc.)
        # If insufficient points, clustering returns everything as noise - correct behavior
        if updated_profile.recent_points:
            updated_profile = self._perform_clustering(updated_profile)

        self.store.save(updated_profile)
        self._pending_points[model_id] = []

        logger.info(
            "Flushed %s points for %s (total=%s)",
            len(points),
            model_id,
            updated_profile.total_point_count,
        )

    def flush_all_pending(self) -> None:
        for model_id, points in list(self._pending_points.items()):
            if not points:
                continue
            profile = self.store.load(model_id)
            model_name = profile.model_name if profile else model_id
            self.flush_pending_points(model_id=model_id, model_name=model_name)

    def trigger_clustering(self, model_id: str) -> ManifoldProfile | None:
        profile = self.store.load(model_id)
        if profile is None:
            logger.warning("Cannot cluster: profile not found for %s", model_id)
            return None
        clustered = self._perform_clustering(profile)
        self.store.save(clustered)
        logger.info("Clustering complete for %s: %s regions", model_id, len(clustered.regions))
        return clustered

    def query_region(self, point: ManifoldPoint, model_id: str) -> RegionQueryResult:
        from modelcypher.core.domain.geometry.manifold_profile import RegionThresholds

        profile = self.store.load(model_id)
        if profile is None:
            # Derive thresholds from the single point (degenerate case)
            thresholds = RegionThresholds.from_percentiles(
                [point.mean_entropy], [point.entropy_variance], [point.mean_gate_similarity]
            )
            return RegionQueryResult(
                nearest_region=None,
                distance=float("inf"),
                is_within_region=False,
                suggested_character=ManifoldRegion.classify(point, thresholds),
                confidence=0.0,
            )
        return self.clusterer.find_nearest_region(point=point, regions=profile.regions)

    def query_region_for_measurement(
        self,
        measurement: CombinedMeasurement,
        model_id: str,
        prompt_hash: str,
    ) -> RegionQueryResult:
        point = ManifoldPoint.from_measurement(measurement=measurement, prompt_hash=prompt_hash)
        return self.query_region(point=point, model_id=model_id)

    def suggest_intervention(self, point: ManifoldPoint, model_id: str) -> InterventionSuggestion:
        from modelcypher.core.domain.geometry.manifold_profile import RegionThresholds

        profile = self.store.load(model_id)
        if profile is None:
            return InterventionSuggestion.no_history()

        query_result = self.clusterer.find_nearest_region(point=point, regions=profile.regions)
        similar_points = self._find_similar_points(
            point=point,
            points=profile.recent_points + [region.centroid for region in profile.regions],
            max_results=10,
        )
        historical_levels = [
            item.intervention_level
            for item in similar_points
            if item.intervention_level is not None
        ]

        if query_result.nearest_region:
            region = query_result.nearest_region
            if region.region_type == ManifoldRegion.RegionCharacter.DENSE:
                suggested_level = 0
                reason = f"Point falls within dense region '{str(region.id)[:8]}'"
            elif region.region_type == ManifoldRegion.RegionCharacter.SPARSE:
                suggested_level = max(historical_levels) if historical_levels else 1
                reason = "Point falls within sparse region - historically uncertain behavior"
            else:
                suggested_level = max(2, max(historical_levels) if historical_levels else 2)
                reason = "Point falls within transitional region - transition zone"
        else:
            if historical_levels:
                suggested_level = max(historical_levels)
                reason = f"No matching region, but similar points triggered level {suggested_level}"
            else:
                # Derive thresholds from profile regions (or point if no regions)
                if profile.regions:
                    centroids = [r.centroid for r in profile.regions]
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
                suggested_level = (
                    0 if ManifoldRegion.classify(point, thresholds) == ManifoldRegion.RegionCharacter.DENSE else 1
                )
                reason = "No historical data - suggestion based on point features"

        if query_result.is_within_region:
            confidence = min(1.0, query_result.confidence + 0.2)
        elif similar_points:
            confidence = float(len(similar_points)) / 10.0
        else:
            confidence = 0.2

        return InterventionSuggestion(
            level=suggested_level,
            reason=reason,
            confidence=confidence,
            based_on_history=bool(historical_levels),
            similar_point_count=len(similar_points),
        )

    def suggest_intervention_for_measurement(
        self,
        measurement: CombinedMeasurement,
        model_id: str,
        prompt_hash: str,
    ) -> InterventionSuggestion:
        point = ManifoldPoint.from_measurement(measurement=measurement, prompt_hash=prompt_hash)
        return self.suggest_intervention(point=point, model_id=model_id)

    def get_profile(self, model_id: str) -> ManifoldProfile | None:
        return self.store.load(model_id)

    def list_profiles(self, limit: int | None = None) -> list[ManifoldProfile]:
        return self.store.list(limit=limit)

    def get_statistics(self, model_id: str) -> ManifoldProfile.Statistics | None:
        return self.store.get_statistics(model_id)

    def delete_profile(self, model_id: str) -> None:
        self.store.delete(model_id)
        self._pending_points.pop(model_id, None)
        logger.info("Deleted profile for %s", model_id)

    def generate_report(self, model_id: str) -> str:
        profile = self.store.load(model_id)
        if profile is None:
            return f"No profile found for model: {model_id}"

        stats = profile.compute_statistics()
        report = [
            f"# Manifold Profile Report: {profile.model_name}",
            "",
            "## Overview",
            f"- Model ID: {profile.model_id}",
            f"- Total Observations: {stats.total_points}",
            f"- Regions: {stats.region_count}",
            f"- Unclustered Points: {stats.recent_point_count}",
            f"- Last Updated: {profile.updated_at}",
            "",
            "## Region Distribution",
            f"- Dense Regions: {stats.dense_region_count}",
            f"- Sparse Regions: {stats.sparse_region_count}",
            f"- Transitional Regions: {stats.transitional_region_count}",
            "",
        ]

        if stats.mean_intrinsic_dimension is not None:
            report.extend(
                [
                    "## Geometry",
                    f"- Mean Intrinsic Dimension: {stats.mean_intrinsic_dimension:.2f}",
                    "",
                ]
            )

        if profile.regions:
            report.append("## Region Details")
            report.append("")
            for region in profile.regions[:10]:
                report.append(
                    f"### {region.region_type.value.capitalize()} Region ({str(region.id)[:8]})"
                )
                report.append(f"- Members: {region.member_count}")
                report.append(f"- Radius: {region.radius:.3f}")
                if region.intrinsic_dimension is not None:
                    report.append(f"- Intrinsic Dimension: {region.intrinsic_dimension:.2f}")
                if region.dominant_gates:
                    report.append(f"- Dominant Gates: {', '.join(region.dominant_gates)}")
                report.append("")

        return "\n".join(report)

    def _perform_clustering(self, profile: ManifoldProfile) -> ManifoldProfile:
        if not profile.regions:
            result = self.clusterer.cluster(profile.recent_points)
            regions = result.regions
            noise = result.noise_points
        else:
            result = self.clusterer.cluster_incremental(
                new_points=profile.recent_points,
                existing_regions=profile.regions,
                existing_noise=[],
            )
            regions = result.regions
            noise = result.noise_points

        return ManifoldProfile(
            id=profile.id,
            model_id=profile.model_id,
            model_name=profile.model_name,
            regions=list(regions),
            recent_points=list(noise),
            total_point_count=profile.total_point_count,
            created_at=profile.created_at,
            updated_at=datetime.utcnow(),
            version=profile.version,
        )

    def _find_similar_points(
        self,
        point: ManifoldPoint,
        points: list[ManifoldPoint],
        max_results: int = 10,
    ) -> list[ManifoldPoint]:
        """Find similar points using geodesic distance.

        Computes geodesic distances from the query point to all candidates
        via k-NN graph. Geodesic is the correct metric on curved manifolds.

        Threshold is derived from data: 25th percentile of the distance distribution.
        This captures the "nearby" points relative to the actual data geometry.
        """
        if not points:
            return []

        backend = get_default_backend()

        # Build matrix: query point at index 0, candidates at indices 1..n
        all_points = [point] + list(points)
        rows = [backend.array(p.feature_vector) for p in all_points]
        features = backend.stack(rows, axis=0)

        rg = RiemannianGeometry(backend)
        n = len(all_points)
        k_neighbors = max(2, min(n - 1, int(n**0.5)))
        result = rg.geodesic_distances(features, k_neighbors=k_neighbors)

        # Extract distances from query (row 0) to each candidate (rows 1..n)
        row0 = result.distances[0, 1:]
        backend.eval(row0)
        count = int(backend.shape(row0)[0])
        if count == 0:
            return []

        # Derive threshold from data: 25th percentile of distance distribution
        percentile_idx = max(0, int(count * 0.25) - 1)
        part = backend.argpartition(row0, percentile_idx)
        prefix = backend.take(part, backend.arange(percentile_idx + 1), axis=0)
        threshold_arr = backend.max(backend.take(row0, prefix, axis=0))
        backend.eval(threshold_arr)

        # Count candidates within threshold
        within = row0 <= threshold_arr
        within_count_arr = backend.sum(backend.astype(within, "int32"))
        backend.eval(within_count_arr)
        within_count = int(backend.to_scalar(within_count_arr))
        if within_count == 0:
            return []
        take_count = min(within_count, max_results)

        # Take closest results via partial sort
        kth = max(0, take_count - 1)
        part_top = backend.argpartition(row0, kth)
        selected_idx = backend.take(part_top, backend.arange(take_count), axis=0)
        selected_vals = backend.take(row0, selected_idx, axis=0)
        order = backend.argsort(selected_vals)
        selected_idx = backend.take(selected_idx, order, axis=0)
        backend.eval(selected_idx)
        selected_list = backend.tolist(selected_idx)
        return [points[int(i)] for i in selected_list]
