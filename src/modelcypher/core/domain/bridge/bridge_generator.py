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

"""Bridge generator for cross-modal knowledge transfer.

Creates affine bridges between encoder spaces using GramAlign and reports
CKA diagnostics.

References:
    - docs/research/multi_modal_cka_validation.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.cka import compute_geodesic_cka

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class BridgeGeneratorResult:
    """Result of generating a cross-modal bridge.

    The bridge transform F maps from source space to target space. Geodesic
    CKA(source @ F, target) reports overlap on the shared manifold.

    For the reverse direction (target → source), use F_inv.
    """

    # Forward transform: source → target [d_source, d_target]
    transform: Any

    # Inverse transform: target → source [d_target, d_source]
    transform_inv: Any

    # Scale ratio for magnitude normalization
    scale_ratio: float

    # Source and target dimensions
    source_dim: int
    target_dim: int

    # Geodesic CKA achieved (overlap diagnostic)
    cka_achieved: float

    # 1.0 - geodesic CKA
    numerical_deviation: float

    # Raw CKA before alignment (for diagnostics)
    raw_cka: float

    # Number of samples used to compute alignment
    n_samples: int

    # Metadata
    source_name: str = "source"
    target_name: str = "target"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrossModalBridge:
    """A bridge for transforming embeddings between encoder spaces.

    This class wraps the transform matrix and provides convenient methods
    for applying the bridge to embeddings.

    Usage:
        bridge = bridge_service.load("bridge.safetensors")
        transformed = bridge.apply(source_embeddings)  # source → target
        reversed = bridge.apply_inverse(target_embeddings)  # target → source
    """

    # Forward transform: source → target
    transform: Any

    # Inverse transform: target → source
    transform_inv: Any

    # Scale ratio for magnitude normalization
    scale_ratio: float

    # Dimensions
    source_dim: int
    target_dim: int

    # Backend reference
    backend: Any

    # Metadata
    source_name: str = "source"
    target_name: str = "target"

    def apply(
        self,
        embeddings: "Array",
        normalize_scale: bool = True,
    ) -> "Array":
        """Apply bridge transform to embeddings (source → target).

        Args:
            embeddings: Source embeddings [..., source_dim]
            normalize_scale: Whether to apply scale normalization

        Returns:
            Transformed embeddings [..., target_dim]
        """
        backend = self.backend
        embeddings = backend.array(embeddings)
        backend.eval(embeddings)

        # Apply transform
        # embeddings: [..., source_dim]
        # transform: [source_dim, target_dim]
        # result: [..., target_dim]
        result = backend.matmul(embeddings, self.transform)

        if normalize_scale:
            result = result * self.scale_ratio

        backend.eval(result)
        return result

    def apply_inverse(
        self,
        embeddings: "Array",
        normalize_scale: bool = True,
    ) -> "Array":
        """Apply inverse bridge transform (target → source).

        Args:
            embeddings: Target embeddings [..., target_dim]
            normalize_scale: Whether to apply inverse scale normalization

        Returns:
            Transformed embeddings [..., source_dim]
        """
        backend = self.backend
        embeddings = backend.array(embeddings)
        backend.eval(embeddings)

        # Apply inverse transform
        result = backend.matmul(embeddings, self.transform_inv)

        if normalize_scale and self.scale_ratio != 0:
            result = result / self.scale_ratio

        backend.eval(result)
        return result

class BridgeGenerator:
    """Generates affine bridges between encoder spaces.

    Uses GramAlign to find a linear transform and reports CKA diagnostics.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize the bridge generator.

        Args:
            backend: Backend for tensor operations
        """
        self._backend = backend or get_default_backend()
        self._aligner = GramAligner(self._backend)

    def generate(
        self,
        source_activations: "Array",
        target_activations: "Array",
        *,
        source_name: str = "source",
        target_name: str = "target",
    ) -> BridgeGeneratorResult:
        """Generate a bridge between two encoder spaces.

        Args:
            source_activations: Activations from source encoder [n_samples, d_source]
            target_activations: Activations from target encoder [n_samples, d_target]
            source_name: Name of source encoder (for metadata)
            target_name: Name of target encoder (for metadata)

        Returns:
            BridgeGeneratorResult with the transform and diagnostics

        Raises:
            ValueError: If sample counts don't match
        """
        backend = self._backend

        source_acts = backend.array(source_activations)
        target_acts = backend.array(target_activations)
        backend.eval(source_acts, target_acts)

        n_source = int(source_acts.shape[0])
        n_target = int(target_acts.shape[0])

        if n_source != n_target:
            raise ValueError(
                f"Sample counts must match: source has {n_source}, target has {n_target}"
            )

        source_dim = int(source_acts.shape[1])
        target_dim = int(target_acts.shape[1])

        logger.info(
            "BRIDGE GENERATOR: Computing alignment %s [%dD] → %s [%dD]",
            source_name, source_dim, target_name, target_dim
        )

        # Compute raw CKA before alignment
        raw_cka = compute_geodesic_cka(source_acts, target_acts, backend=backend)
        logger.info("Raw geodesic CKA (before alignment): %.4f", raw_cka)

        # Find the alignment transform
        alignment = self._aligner.find_perfect_alignment(source_acts, target_acts)

        cka_achieved = alignment.achieved_cka
        numerical_deviation = alignment.numerical_deviation
        scale_ratio = alignment.scale_ratio
        F = alignment.feature_transform  # [d_source, d_target]

        logger.info("Aligned geodesic CKA: %.4f (deviation: %.2e)", cka_achieved, numerical_deviation)
        logger.info("Scale ratio: %.4f", scale_ratio)
        logger.info("Transform shape: %s", F.shape)

        # Compute inverse transform for reverse direction
        F_inv = backend.pinv(F)
        backend.eval(F_inv)

        logger.info("Inverse transform shape: %s", F_inv.shape)

        return BridgeGeneratorResult(
            transform=F,
            transform_inv=F_inv,
            scale_ratio=scale_ratio,
            source_dim=source_dim,
            target_dim=target_dim,
            cka_achieved=cka_achieved,
            numerical_deviation=numerical_deviation,
            raw_cka=raw_cka,
            n_samples=n_source,
            source_name=source_name,
            target_name=target_name,
        )


    def to_bridge(self, result: BridgeGeneratorResult) -> CrossModalBridge:
        """Convert a BridgeGeneratorResult to a CrossModalBridge.

        Args:
            result: The generation result

        Returns:
            CrossModalBridge instance ready for use
        """
        return CrossModalBridge(
            transform=result.transform,
            transform_inv=result.transform_inv,
            scale_ratio=result.scale_ratio,
            source_dim=result.source_dim,
            target_dim=result.target_dim,
            backend=self._backend,
            source_name=result.source_name,
            target_name=result.target_name,
        )


__all__ = [
    "BridgeGenerator",
    "BridgeGeneratorResult",
    "CrossModalBridge",
]
