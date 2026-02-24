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

"""Model probe service - orchestrates backend-specific probe implementations.

This service delegates to an injected ModelProbePort. Platform selection
is handled by the composition root.
"""

from __future__ import annotations

import logging

from modelcypher.adapters.model_probe import (
    AlignmentAnalysisResult,
    MergeValidationResult,
    ModelProbe,
    ModelProbeResult,
)

logger = logging.getLogger(__name__)

__all__ = ["ModelProbeService"]


class ModelProbeService:
    """
    Service for probing and analyzing model architecture and compatibility.

    This is a facade that delegates to the appropriate backend-specific probe.
    Use the composition root to select the appropriate probe implementation.
    """

    def __init__(self, probe: ModelProbe) -> None:
        """
        Initialize the service.

        Args:
            probe: Probe implementation (required).
        """
        self._probe = probe

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details.

        Args:
            model_path: Path to the model directory containing config.json and weight files.

        Returns:
            ModelProbeResult with architecture details.

        Raises:
            ValueError: If model path is invalid or config.json is missing.
        """
        result = self._probe.probe(model_path)
        try:
            from modelcypher.core.domain.geometry.model_profile import ModelProfileStore
            from modelcypher.core.use_cases.model_profiler_service import ModelProfilerService

            store = ModelProfileStore()
            profile, identity = store.ensure(model_path)
            profiler = ModelProfilerService()
            updated = profiler.update_identity(
                profile,
                model_path,
                probe_result=result,
                identity=identity,
            )
            store.save(updated, identity)
        except Exception as exc:
            logger.warning("Failed to persist model profile for %s: %s", model_path, exc)
        return result

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models.

        Args:
            source: Path to the source model directory.
            target: Path to the target model directory.

        Returns:
            MergeValidationResult with compatibility assessment.
        """
        return self._probe.validate_merge(source, target)

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models.

        Computes layer-wise drift using weight comparison.

        Args:
            model_a: Path to the first model directory.
            model_b: Path to the second model directory.

        Returns:
            AlignmentAnalysisResult with drift metrics bounded in [0.0, 1.0].
        """
        return self._probe.analyze_alignment(model_a, model_b)
