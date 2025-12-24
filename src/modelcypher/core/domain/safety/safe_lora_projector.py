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

"""Safe LoRA projector for safety subspace projection.

Safe LoRA projects adapter weights onto a "safe" subspace to remove
potentially harmful directions while preserving useful capabilities.

This scaffolds the projection workflow by checking for cached projection
payloads. If no cache is found, projection is skipped with a warning.
Projection math is intentionally deferred until cache assets are provided.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


class SafeLoRAProjectionStatus(str, Enum):
    """Status of a Safe LoRA projection attempt."""

    APPLIED = "applied"
    """Projection was successfully applied to the adapter."""

    SKIPPED = "skipped"
    """Projection was skipped (cache found but math deferred)."""

    UNAVAILABLE = "unavailable"
    """No cached projection matrix available for this base model."""


@dataclass(frozen=True)
class SafeLoRAProjectionResult:
    """Result of Safe LoRA projection attempt."""

    status: SafeLoRAProjectionStatus
    """Status of the projection attempt."""

    warnings: tuple[str, ...] = ()
    """Warnings generated during projection."""

    details: str | None = None
    """Additional details about the projection."""

    @property
    def was_applied(self) -> bool:
        """Whether the projection was successfully applied."""
        return self.status == SafeLoRAProjectionStatus.APPLIED

    @property
    def is_available(self) -> bool:
        """Whether a projection matrix was available."""
        return self.status != SafeLoRAProjectionStatus.UNAVAILABLE

    @classmethod
    def applied(cls, details: str | None = None) -> SafeLoRAProjectionResult:
        """Create a result for successful application."""
        return cls(status=SafeLoRAProjectionStatus.APPLIED, details=details)

    @classmethod
    def skipped(
        cls, warnings: tuple[str, ...] = (), details: str | None = None
    ) -> SafeLoRAProjectionResult:
        """Create a result for skipped projection."""
        return cls(
            status=SafeLoRAProjectionStatus.SKIPPED, warnings=warnings, details=details
        )

    @classmethod
    def unavailable(cls, warning: str) -> SafeLoRAProjectionResult:
        """Create a result for unavailable projection matrix."""
        return cls(status=SafeLoRAProjectionStatus.UNAVAILABLE, warnings=(warning,))


class SafeLoRAProjector:
    """Safe LoRA projector that uses cached projection matrices when available.

    This scaffolds the projection workflow by checking for cached projection
    payloads under `resources/safety/projections/<baseModelID>/projection.safetensors`.
    If no cache is found, projection is skipped with a warning. Projection math
    is intentionally deferred until cache assets are provided.

    Safe LoRA (from the paper "Safe LoRA: the Silver Lining of Reducing Safety
    Risks when Fine-tuning Large Language Models") projects adapter weights
    onto a subspace that is orthogonal to safety-critical directions, thereby
    reducing the risk of the adapter degrading the model's safety alignment.
    """

    def __init__(self, resources_path: Path | None = None):
        """Create a Safe LoRA projector.

        Args:
            resources_path: Path to resources directory containing projection
                matrices. Defaults to package resources.
        """
        self._resources_path = resources_path

    async def project(
        self,
        base_model_id: str,
        adapter_path: Path,
    ) -> SafeLoRAProjectionResult:
        """Attempt to project an adapter using a cached safety subspace.

        Args:
            base_model_id: Base model identifier (e.g., "mlx-community/Llama-3.2-3B").
            adapter_path: Path to adapter directory.

        Returns:
            Projection result indicating whether projection was applied or skipped.
        """
        sanitized_base = self._sanitize(base_model_id)
        subdir = f"safety/projections/{sanitized_base}"

        projection_path = self._find_projection_file(subdir)

        if projection_path is None:
            warning = (
                f"Safe LoRA projection skipped: no cached projection matrix "
                f"for base model {base_model_id}"
            )
            logger.info(warning)
            return SafeLoRAProjectionResult.unavailable(warning)

        # Projection math requires aligned/base subspace; intentionally deferred
        # until assets exist.
        logger.info(
            "Safe LoRA projection placeholder: found cache at %s but math is deferred",
            projection_path.name,
        )
        warning = (
            "Safe LoRA projection cache found but application is deferred "
            "pending subspace assets"
        )
        return SafeLoRAProjectionResult.skipped(
            warnings=(warning,), details=projection_path.name
        )

    def _find_projection_file(self, subdir: str) -> Path | None:
        """Find a projection file in the resources directory.

        Args:
            subdir: Subdirectory to search in.

        Returns:
            Path to projection file, or None if not found.
        """
        if self._resources_path is None:
            # Try to find package resources
            try:
                import importlib.resources as pkg_resources

                # This is a placeholder - actual implementation would use
                # proper resource discovery
                return None
            except ImportError:
                return None

        projection_dir = self._resources_path / subdir
        if not projection_dir.exists():
            return None

        projection_file = projection_dir / "projection.safetensors"
        if projection_file.exists():
            return projection_file

        return None

    @staticmethod
    def _sanitize(model_id: str) -> str:
        """Sanitize a model ID for use in file paths.

        Args:
            model_id: Model identifier to sanitize.

        Returns:
            Sanitized identifier safe for file paths.
        """
        # Replace problematic characters with underscores
        sanitized = re.sub(r"[/: ]", "_", model_id)
        return sanitized


@dataclass
class SafeLoRAConfiguration:
    """Configuration for Safe LoRA projection."""

    enabled: bool = True
    """Whether Safe LoRA projection is enabled."""

    resources_path: Path | None = None
    """Path to resources directory containing projection matrices."""

    skip_if_unavailable: bool = True
    """Whether to skip projection if no cache is available (vs. failing)."""

    @classmethod
    def default(cls) -> SafeLoRAConfiguration:
        """Default configuration with projection enabled."""
        return cls()

    @classmethod
    def disabled(cls) -> SafeLoRAConfiguration:
        """Configuration with projection disabled."""
        return cls(enabled=False)
