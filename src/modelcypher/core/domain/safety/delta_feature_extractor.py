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
- L2 norms per target module
- Sparsity ratios (fraction of near-zero elements)
- Outlier detection (layers with unusual statistics)

The extractor does NOT require loading the full model, just the adapter weights.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    find_magnitude_gap_threshold,
    machine_epsilon,
)
from modelcypher.core.domain.safety.adapter_safety_models import AdapterSafetyTier
from modelcypher.core.domain.safety.adapter_safety_probe import (
    AdapterSafetyProbe,
    ProbeContext,
    ProbeResult,
)
from modelcypher.core.domain.safety.delta_feature_set import DeltaFeatureSet

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


class DeltaFeatureExtractor:
    """Extracts statistical features from LoRA adapter weights for profiling.

    Computes L2 norms, sparsity ratios, and gap-derived outlier detection
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

        all_l2_norms: list[float] = []
        all_sparsity: list[float] = []

        for file_path in safetensors_files:
            norms, sparsities = await self._extract_from_file(file_path)
            all_l2_norms.extend(norms)
            all_sparsity.extend(sparsities)

        # Find outlier layers via gap detection on L2 norms
        outlier_indices = self._find_outlier_indices(all_l2_norms)

        logger.info(
            "Extracted delta features: %d layers, %d suspect",
            len(all_l2_norms),
            len(outlier_indices),
        )

        return DeltaFeatureSet(
            l2_norms=tuple(all_l2_norms),
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
            Tuple of (l2_norms, sparsities).
        """
        l2_norms: list[float] = []
        sparsities: list[float] = []

        try:
            backend = self._backend
            weights = backend.load_safetensors(str(file_path))
            for tensor in weights.values():
                tensor_backend = backend.array(tensor)

                # Compute L2 norm
                sum_sq = backend.sum(tensor_backend * tensor_backend)
                backend.eval(sum_sq)
                l2_norm = float(backend.to_scalar(backend.sqrt(sum_sq)))
                l2_norms.append(l2_norm)

                # Compute sparsity (fraction of near-zero elements)
                # Derive threshold from tensor dtype using machine epsilon
                sparsity_threshold = machine_epsilon(backend, tensor_backend)
                abs_tensor = backend.abs(tensor_backend)
                near_zero_count = backend.sum(abs_tensor < sparsity_threshold)
                backend.eval(near_zero_count)
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

        return l2_norms, sparsities

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
            "zero_norm_layers": sum(1 for n in features.l2_norms if n == 0),
        }

        # Check for outlier layers (using data-derived outlier detection)
        if features.has_outlier_layers:
            findings.append(
                f"{len(features.outlier_layer_indices)}/{features.layer_count} "
                "layers have outlier L2 norms"
            )

        # Check for zero-norm layers (exact zeros are data artifacts)
        if finding_counts["zero_norm_layers"] > 0:
            findings.append(
                f"{finding_counts['zero_norm_layers']} layers have zero L2 norm"
            )

        # Add statistics to finding_counts
        if features.l2_norms:
            finding_counts["max_l2_norm"] = int(max(features.l2_norms))
            finding_counts["min_l2_norm"] = int(min(features.l2_norms))

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
