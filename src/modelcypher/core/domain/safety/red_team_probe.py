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

"""Red Team Probe.

Static metadata probe that detects geometric outliers in adapter metadata
embeddings. Returns only raw distance measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from modelcypher.core.domain.safety.behavioral_probes import (
    AdapterSafetyProbe,
    ProbeContext,
    ProbeResult,
)
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    find_magnitude_gap_threshold,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.ports.embedding import EmbeddingProvider


class RedTeamProbe(AdapterSafetyProbe):
    """
    Probe that checks adapter metadata for geometric outliers.

    This is a static analysis probe that examines metadata embeddings for
    outlier geometry relative to other metadata fields.
    """

    @property
    def name(self) -> str:
        return "red-team-static"

    @property
    def version(self) -> str:
        return "probe-rt-v1.0"

    def evaluate(self, context: ProbeContext) -> ProbeResult:
        """Evaluate adapter metadata for geometric outliers."""
        if context.embedder is None:
            return ProbeResult(
                probe_name=self.name,
                probe_version=self.version,
                finding_counts={
                    "metadata_items": 0,
                    "missing_embedder": 1,
                },
            )

        items = _collect_metadata_items(context)
        if len(items) < 2:
            return ProbeResult(
                probe_name=self.name,
                probe_version=self.version,
                finding_counts={
                    "metadata_items": len(items),
                    "insufficient_metadata": 1,
                },
            )

        distances, outliers, threshold, mean_distance, max_distance = _metadata_outliers(
            items, context.embedder
        )

        findings = tuple(
            f"{item.field}: mean_distance={item.mean_distance}" for item in outliers
        )

        finding_counts = {
            "metadata_items": len(items),
            "outlier_items": len(outliers),
            "distance_threshold": threshold,
            "mean_distance": mean_distance,
            "max_distance": max_distance,
        }

        return ProbeResult(
            probe_name=self.name,
            probe_version=self.version,
            findings=findings,
            finding_counts=finding_counts,
        )


@dataclass(frozen=True)
class MetadataDistance:
    field: str
    text: str
    mean_distance: float


def _collect_metadata_items(context: ProbeContext) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if context.adapter_name:
        items.append(("adapter_name", context.adapter_name))
    if context.adapter_description:
        items.append(("adapter_description", context.adapter_description))
    for tag in context.skill_tags:
        items.append(("skill_tag", tag))
    if context.creator:
        items.append(("creator", context.creator))
    if context.base_model_id:
        items.append(("base_model_id", context.base_model_id))
    for module in context.target_modules:
        items.append(("target_module", module))
    for dataset in context.training_datasets:
        items.append(("training_dataset", dataset))
    return items


def _distance_threshold(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    max_gap = 0.0
    for i in range(len(sorted_vals) - 1):
        curr = sorted_vals[i]
        next_val = sorted_vals[i + 1]
        if curr > 0.0:
            relative_gap = (next_val - curr) / curr
            if relative_gap > max_gap:
                max_gap = relative_gap
    eps = max(math.ulp(max(sorted_vals)), math.ulp(1.0))
    if max_gap <= eps:
        return float("inf")
    return float(find_magnitude_gap_threshold(sorted_vals))


def _metadata_distances(
    items: list[tuple[str, str]],
    embedder: EmbeddingProvider,
) -> tuple[list[MetadataDistance], float]:
    backend = get_default_backend()
    texts = [text for _, text in items]
    embeddings = embedder.embed(texts)
    points = backend.array(embeddings)
    rg = RiemannianGeometry(backend)
    geo = rg.geodesic_distances(points)
    backend.eval(geo.distances)
    dist_matrix = backend.to_numpy(geo.distances).tolist()
    n = len(items)

    mean_distances: list[MetadataDistance] = []
    for idx, (field, text) in enumerate(items):
        row = dist_matrix[idx]
        total = sum(float(val) for j, val in enumerate(row) if j != idx)
        denom = float(n - 1) if n > 1 else 1.0
        mean_dist = total / denom if denom > 0 else 0.0
        mean_distances.append(
            MetadataDistance(field=field, text=text, mean_distance=mean_dist)
        )

    threshold = _distance_threshold([d.mean_distance for d in mean_distances])
    return mean_distances, threshold


def _metadata_outliers(
    items: list[tuple[str, str]],
    embedder: EmbeddingProvider,
) -> tuple[list[MetadataDistance], list[MetadataDistance], float, float, float]:
    distances, threshold = _metadata_distances(items, embedder)
    outliers = [item for item in distances if item.mean_distance > threshold]
    mean_distance = (
        sum(item.mean_distance for item in distances) / len(distances)
        if distances
        else 0.0
    )
    max_distance = max((item.mean_distance for item in distances), default=0.0)
    return distances, outliers, threshold, mean_distance, max_distance


@dataclass(frozen=True)
class ThreatIndicator:
    """A metadata outlier indicator derived from geometry."""

    field: str
    text: str
    mean_distance: float


class RedTeamScanner:
    """
    Scanner for comprehensive red team analysis.

    Provides a static analysis interface for scanning adapter metadata
    without requiring an async context.
    """

    def __init__(self, embedder: EmbeddingProvider | None = None):
        """Initialize with optional embedder."""
        self._embedder = embedder
        self._probe = RedTeamProbe()

    def scan_adapter(
        self,
        name: str,
        description: str | None = None,
        skill_tags: list[str] | None = None,
        creator: str | None = None,
        base_model_id: str | None = None,
        target_modules: list[str] | None = None,
        training_datasets: list[str] | None = None,
    ) -> list[ThreatIndicator]:
        """
        Scan adapter metadata for threat indicators.

        This is a synchronous interface for quick scanning.

        Args:
            name: Adapter name
            description: Adapter description
            skill_tags: List of skill tags
            creator: Creator identifier
            base_model_id: Base model reference
            target_modules: List of target modules
            training_datasets: List of training dataset references

        Returns:
            List of detected threat indicators
        """
        if self._embedder is None:
            return []

        context = ProbeContext(
            adapter_name=name,
            adapter_description=description,
            skill_tags=tuple(skill_tags or ()),
            creator=creator,
            base_model_id=base_model_id,
            target_modules=tuple(target_modules or ()),
            training_datasets=tuple(training_datasets or ()),
            embedder=self._embedder,
        )
        items = _collect_metadata_items(context)
        if len(items) < 2:
            return []
        _, outliers, _, _, _ = _metadata_outliers(items, self._embedder)
        return [
            ThreatIndicator(
                field=item.field,
                text=item.text,
                mean_distance=item.mean_distance,
            )
            for item in outliers
        ]
