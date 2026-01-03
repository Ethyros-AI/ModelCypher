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

"""Extracts statistical features from LoRA adapter weights for geometric profiling.

This implements lightweight PEFTGuard-style analysis by computing:
- Geodesic spread per target module
- Sparsity ratios (fraction of near-zero elements)
- Outlier detection (layers with unusual statistics)

The extractor does NOT require loading the full model, just the adapter weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    find_magnitude_gap_threshold,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.safety.adapter_safety_models import AdapterSafetyTier
from modelcypher.core.domain.safety.adapter_safety_probe import (
    AdapterSafetyProbe,
    ProbeContext,
    ProbeResult,
)
from modelcypher.core.domain.safety.delta_feature_set import DeltaFeatureSet

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class DeltaFeatureExtractor:
    """Extracts statistical features from LoRA adapter weights for profiling.

    Computes geodesic spreads, sparsity ratios, and gap-derived outlier detection
    without requiring the full base model.
    """

    VERSION = "delta-v1.0"
    """Version string for this extractor."""

    def __init__(
        self,
        backend: "Backend | None" = None,
    ):
        """Create a feature extractor.
        """
        self._backend = backend or get_default_backend()

    async def extract(self, adapter_path: Path) -> DeltaFeatureSet:
        """Extract delta features from adapter weights at the given path.

        Args:
            adapter_path: Directory containing safetensors files.

        Returns:
            Feature set with computed statistics.
        """
        safetensors_files = self._find_safetensors_files(adapter_path)

        if not safetensors_files:
            logger.warning("No safetensors files found in adapter path")
            return DeltaFeatureSet(feature_version=self.VERSION)

        all_geodesic_spreads: list[float] = []
        all_sparsity: list[float] = []

        for file_path in safetensors_files:
            spreads, sparsities = await self._extract_from_file(file_path)
            all_geodesic_spreads.extend(spreads)
            all_sparsity.extend(sparsities)

        # Find outlier layers via gap detection on geodesic spreads
        outlier_indices = self._find_outlier_indices(all_geodesic_spreads)

        logger.info(
            "Extracted delta features: %d layers, %d suspect",
            len(all_geodesic_spreads),
            len(outlier_indices),
        )

        return DeltaFeatureSet(
            geodesic_spreads=tuple(all_geodesic_spreads),
            sparsity=tuple(all_sparsity),
            cosine_to_aligned=(),  # Requires aligned baseline (future)
            outlier_layer_indices=tuple(outlier_indices),
            feature_version=self.VERSION,
        )

    def _find_safetensors_files(self, directory: Path) -> list[Path]:
        """Find safetensors files in a directory."""
        if not directory.exists():
            return []
        return list(directory.glob("*.safetensors"))

    async def _extract_from_file(self, file_path: Path) -> tuple[list[float], list[float]]:
        """Extract features from a single safetensors file.

        Args:
            file_path: Path to the safetensors file.

        Returns:
            Tuple of (geodesic_spreads, sparsities).
        """
        geodesic_spreads: list[float] = []
        sparsities: list[float] = []

        try:
            backend = self._backend
            weights = backend.load_safetensors(str(file_path))
            for tensor in weights.values():
                tensor_backend = backend.array(tensor)

                points = self._tensor_to_points(tensor_backend)
                points = self._sample_points(points)
                spread = self._geodesic_spread(points)

                # Compute sparsity (fraction of near-zero elements)
                # Derive threshold from tensor dtype using machine epsilon
                sparsity_threshold = machine_epsilon(backend, tensor_backend)
                abs_tensor = backend.abs(tensor_backend)
                near_zero_count = backend.sum(abs_tensor < sparsity_threshold)
                backend.eval(near_zero_count)
                geodesic_spreads.append(spread)
                shape = backend.shape(tensor_backend)
                total_elements = 1
                for dim in shape:
                    total_elements *= int(dim)
                sparsity = (
                    float(backend.to_scalar(near_zero_count)) / total_elements
                    if total_elements > 0
                    else 0.0
                )
                sparsities.append(sparsity)

        except ImportError:
            logger.warning("safetensors library not available, returning empty features")
        except Exception as e:
            logger.error("Error extracting features from %s: %s", file_path, e)

        return geodesic_spreads, sparsities

    def _tensor_to_points(self, tensor_backend: "Array") -> "Array":
        """Reshape tensor into a 2D point cloud for geodesic analysis."""
        backend = self._backend
        shape = backend.shape(tensor_backend)
        if not shape:
            return backend.reshape(tensor_backend, (1, 1))
        if len(shape) == 1:
            return backend.reshape(tensor_backend, (shape[0], 1))
        return backend.reshape(tensor_backend, (shape[0], -1))

    def _sample_points(self, points: "Array") -> "Array":
        """Downsample points deterministically to keep geodesic costs derived from data size."""
        backend = self._backend
        n = int(points.shape[0])
        if n <= 1:
            return points
        target = int(sqrt_scalar(float(n), backend))
        min_points = 2 if n >= 2 else 1
        target = max(min_points, min(target, n))
        if target == n:
            return points
        step = max(1, n // target)
        indices = backend.arange(0, n, step)
        sampled = backend.take(points, indices, axis=0)
        backend.eval(sampled)
        return sampled

    def _geodesic_spread(self, points: "Array") -> float:
        """Compute RMS geodesic spread for a point cloud."""
        backend = self._backend
        n = int(points.shape[0])
        if n <= 1:
            return 0.0
        rg = RiemannianGeometry(backend)
        mean_result = rg.frechet_mean(points)
        mean_sq = mean_result.final_variance / max(n, 1)
        return sqrt_scalar(mean_sq, backend)

    def _find_outlier_indices(self, values: list[float]) -> list[int]:
        """Find indices of values that are statistical outliers.

        Args:
            values: List of values to analyze.

        Returns:
            Indices of outlier values.
        """
        if len(values) <= 2:
            return []

        sorted_values = sorted(values)
        threshold = find_magnitude_gap_threshold(sorted_values)
        return [i for i, v in enumerate(values) if v > threshold]


class DeltaFeatureProbe(AdapterSafetyProbe):
    """Probe that evaluates adapter weight statistics."""

    NAME = "delta-features"
    VERSION = "probe-delta-v1.0"

    def __init__(
        self,
        extractor: DeltaFeatureExtractor | None = None,
    ):
        """Create a delta feature probe.

        Args:
            extractor: Feature extractor to use. Defaults to new instance.
        """
        self._extractor = extractor or DeltaFeatureExtractor()

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def version(self) -> str:
        return self.VERSION

    @property
    def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
        return frozenset(
            [AdapterSafetyTier.QUICK, AdapterSafetyTier.STANDARD, AdapterSafetyTier.FULL]
        )

    async def evaluate(self, context: ProbeContext) -> ProbeResult:
        """Evaluate adapter weight statistics.

        Args:
            context: Probe context with adapter path.

        Returns:
            Probe result with raw finding counts.
        """
        features = await self._extractor.extract(context.adapter_path)

        findings: list[str] = []

        # Collect raw counts - no arbitrary thresholds
        finding_counts: dict[str, int] = {
            "total_layers": features.layer_count,
            "outlier_layers": len(features.outlier_layer_indices),
            "zero_spread_layers": sum(1 for n in features.geodesic_spreads if n == 0),
        }

        # Check for outlier layers (using data-derived outlier detection)
        if features.has_outlier_layers:
            findings.append(
                f"{len(features.outlier_layer_indices)}/{features.layer_count} "
                "layers have outlier geodesic spread"
            )

        # Check for zero-spread layers (exact zeros are data artifacts)
        if finding_counts["zero_spread_layers"] > 0:
            findings.append(
                f"{finding_counts['zero_spread_layers']} layers have zero geodesic spread"
            )

        # Add statistics to finding_counts
        if features.geodesic_spreads:
            finding_counts["max_geodesic_spread"] = int(max(features.geodesic_spreads))
            finding_counts["min_geodesic_spread"] = int(min(features.geodesic_spreads))

        logger.info(
            "Delta probe: %d layers, outliers=%d, findings=%d",
            features.layer_count,
            len(features.outlier_layer_indices),
            len(findings),
        )

        return ProbeResult(
            probe_name=self.name,
            details=(
                f"outlier_layers={finding_counts['outlier_layers']} "
                f"total_layers={finding_counts['total_layers']}"
            ),
            findings=tuple(findings),
            probe_version=self.version,
            finding_counts=finding_counts,
        )
