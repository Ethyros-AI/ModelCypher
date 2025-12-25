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

Based on the paper: "Safe LoRA: the Silver Lining of Reducing Safety Risks
when Fine-tuning Large Language Models" (Hsu et al., 2024).

The projection removes the component of adapter weights that lies in the
safety-critical subspace, preserving the orthogonal complement which
captures useful capabilities without degrading safety alignment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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
        return cls(status=SafeLoRAProjectionStatus.SKIPPED, warnings=warnings, details=details)

    @classmethod
    def unavailable(cls, warning: str) -> SafeLoRAProjectionResult:
        """Create a result for unavailable projection matrix."""
        return cls(status=SafeLoRAProjectionStatus.UNAVAILABLE, warnings=(warning,))


class SafeLoRAProjector:
    """Safe LoRA projector that uses cached projection matrices.

    Safe LoRA projects adapter weights onto a subspace orthogonal to safety-
    critical directions, reducing the risk of the adapter degrading safety.

    The projection matrix P should be computed from the difference between
    aligned and unaligned model weights. Given adapter weight W, the projected
    weight is: W_safe = W - P @ W (removing the safety-critical component).

    Projection matrices are stored under:
    `resources/safety/projections/<baseModelID>/projection.safetensors`
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
        """Project adapter weights onto the safe subspace.

        Loads the cached projection matrix and applies it to all adapter
        weight matrices, saving the projected adapter in-place.

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

        # Apply the projection
        try:
            projected_count = await self._apply_projection(adapter_path, projection_path)
            details = f"Projected {projected_count} weight matrices using {projection_path.name}"
            logger.info("Safe LoRA projection applied: %s", details)
            return SafeLoRAProjectionResult.applied(details=details)
        except Exception as e:
            warning = f"Safe LoRA projection failed: {e}"
            logger.warning(warning)
            return SafeLoRAProjectionResult.skipped(warnings=(warning,))

    async def _apply_projection(self, adapter_path: Path, projection_path: Path) -> int:
        """Apply projection matrix to adapter weights.

        The projection removes the safety-critical component:
        W_safe = W - P @ W

        where P is the projection matrix onto the safety subspace.

        Args:
            adapter_path: Path to adapter directory.
            projection_path: Path to projection matrix file.

        Returns:
            Number of weight matrices projected.
        """
        try:
            import mlx.core as mx
        except ImportError:
            raise ImportError("MLX required for Safe LoRA projection")

        # Load projection matrix
        projection_weights = mx.load(str(projection_path))

        # The projection file should contain projection matrices for each layer
        # Format: {"layers.N.proj": mx.array} where proj is the projection matrix
        if not projection_weights:
            raise ValueError(f"Empty projection file: {projection_path}")

        # Load adapter weights
        adapter_file = adapter_path / "adapters.safetensors"
        if not adapter_file.exists():
            adapter_file = adapter_path / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise ValueError(f"No adapter weights found in {adapter_path}")

        adapter_weights = mx.load(str(adapter_file))
        projected_count = 0

        # Apply projection to each LoRA weight matrix
        new_weights = {}
        for key, weight in adapter_weights.items():
            # Check if we have a projection matrix for this layer
            # Extract layer index from key like "layers.5.lora_a.weight"
            proj_key = self._get_projection_key(key, projection_weights)

            if proj_key is not None and proj_key in projection_weights:
                P = projection_weights[proj_key]
                # W_safe = W - P @ W (remove safety-critical component)
                # For LoRA, we project the output direction (lora_b weights)
                if "lora_b" in key or "lora_B" in key:
                    if P.shape[0] == weight.shape[0]:
                        projected_weight = weight - P @ weight
                        new_weights[key] = projected_weight
                        projected_count += 1
                        continue

            # Keep original weight if no projection available
            new_weights[key] = weight

        # Save projected weights
        if projected_count > 0:
            mx.save_safetensors(str(adapter_file), new_weights)

        return projected_count

    def _get_projection_key(self, adapter_key: str, projection_weights: dict) -> str | None:
        """Map adapter weight key to projection matrix key.

        Args:
            adapter_key: Key from adapter weights (e.g., "layers.5.lora_b.weight").
            projection_weights: Available projection matrices.

        Returns:
            Matching projection key, or None if no match.
        """
        import re

        # Extract layer number
        match = re.search(r"layers\.(\d+)", adapter_key)
        if not match:
            return None

        layer_idx = match.group(1)

        # Try common projection key formats
        candidates = [
            f"layers.{layer_idx}.proj",
            f"layer_{layer_idx}_proj",
            f"projection.{layer_idx}",
        ]

        for candidate in candidates:
            if candidate in projection_weights:
                return candidate

        return None

    def _find_projection_file(self, subdir: str) -> Path | None:
        """Find a projection file in the resources directory.

        Searches in order:
        1. Explicitly provided resources_path
        2. Package data directory (modelcypher/data/)
        3. User cache directory (~/.cache/modelcypher/)

        Args:
            subdir: Subdirectory to search in.

        Returns:
            Path to projection file, or None if not found.
        """
        search_paths: list[Path] = []

        # 1. Explicitly provided path
        if self._resources_path is not None:
            search_paths.append(self._resources_path)

        # 2. Package data directory
        try:
            import modelcypher

            pkg_dir = Path(modelcypher.__file__).parent / "data"
            if pkg_dir.exists():
                search_paths.append(pkg_dir)
        except (ImportError, AttributeError):
            pass

        # 3. User cache directory
        cache_dir = Path.home() / ".cache" / "modelcypher"
        if cache_dir.exists():
            search_paths.append(cache_dir)

        # Search all paths for projection file
        for base_path in search_paths:
            projection_dir = base_path / subdir
            if not projection_dir.exists():
                continue

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
