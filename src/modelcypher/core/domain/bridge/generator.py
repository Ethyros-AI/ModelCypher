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

"""
Bridge Generator for cross-modal knowledge transfer.

Creates affine bridges between encoder spaces using GramAlign. The key insight
is that CKA = 1.0 is achievable across ALL modalities because neural networks
discover the same invariant geometry.

Validated Experimentally (January 2026):
    | Modality Pair     | Raw CKA | Aligned CKA |
    |-------------------|---------|-------------|
    | Text ↔ Vision     | 0.7842  | 1.0000      |
    | Text ↔ Audio      | 0.5469  | 1.0000      |
    | Text ↔ Diffusion  | 0.7230  | 1.0000      |
    | Vision ↔ Audio    | 0.6653  | 1.0000      |
    | Vision ↔ Diffusion| 0.8647  | 1.0000      |
    | Audio ↔ Diffusion | 0.7099  | 1.0000      |

Usage:
    from modelcypher.core.domain.bridge import BridgeGenerator

    generator = BridgeGenerator(backend)
    result = generator.generate(source_acts, target_acts)
    generator.save_bridge(result, Path("bridge.safetensors"))

    # Later, apply the bridge
    bridge = generator.load_bridge(Path("bridge.safetensors"))
    transformed = bridge.apply(source_embeddings)

References:
    - docs/research/multi_modal_cka_validation.md
    - /Volumes/CodeCypher/experiments/multi-modal-compression-2026-01-09/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.cka import compute_linear_cka
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class BridgeGeneratorResult:
    """Result of generating a cross-modal bridge.

    The bridge transform F maps from source space to target space such that
    CKA(source @ F, target) = 1.0 (invariant).

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

    # CKA achieved (should be 1.0, invariant)
    cka_achieved: float

    # Numerical deviation from CKA = 1.0
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
        bridge = CrossModalBridge.load(Path("bridge.safetensors"), backend)
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

    @classmethod
    def load(
        cls,
        path: Path,
        backend: "Backend | None" = None,
    ) -> "CrossModalBridge":
        """Load a bridge from safetensors file.

        Args:
            path: Path to the bridge file
            backend: Backend for tensor operations

        Returns:
            CrossModalBridge instance
        """
        backend = backend or get_default_backend()

        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError(
                "safetensors required for bridge loading. "
                "Install with: pip install safetensors"
            )

        with safe_open(str(path), framework="numpy") as f:
            transform_np = f.get_tensor("transform")
            transform_inv_np = f.get_tensor("transform_inv")

            # Load metadata
            metadata = f.metadata() or {}
            scale_ratio = float(metadata.get("scale_ratio", "1.0"))
            source_dim = int(metadata.get("source_dim", transform_np.shape[0]))
            target_dim = int(metadata.get("target_dim", transform_np.shape[1]))
            source_name = metadata.get("source_name", "source")
            target_name = metadata.get("target_name", "target")

        transform = backend.array(transform_np)
        transform_inv = backend.array(transform_inv_np)
        backend.eval(transform, transform_inv)

        return cls(
            transform=transform,
            transform_inv=transform_inv,
            scale_ratio=scale_ratio,
            source_dim=source_dim,
            target_dim=target_dim,
            backend=backend,
            source_name=source_name,
            target_name=target_name,
        )


class BridgeGenerator:
    """Generates affine bridges between encoder spaces.

    Uses GramAlign to find the transform F such that CKA(source @ F, target) = 1.0.
    The bridge is a linear transform - no neural network training required.

    The key insight: neural networks discover the same invariant geometry.
    Different encoders are just different coordinate systems for the same shape.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        fast_mode: bool = False,
    ) -> None:
        """Initialize the bridge generator.

        Args:
            backend: Backend for tensor operations
            fast_mode: Skip CKA precision checks (faster but less diagnostic)
        """
        self._backend = backend or get_default_backend()
        self._aligner = GramAligner(self._backend, fast_mode=fast_mode)
        self._fast_mode = fast_mode

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
        raw_cka = compute_linear_cka(source_acts, target_acts, backend=backend)
        logger.info("Raw CKA (before alignment): %.4f", raw_cka)

        # Find the alignment transform
        alignment = self._aligner.find_perfect_alignment(source_acts, target_acts)

        cka_achieved = alignment.achieved_cka
        numerical_deviation = alignment.numerical_deviation
        scale_ratio = alignment.scale_ratio
        F = alignment.feature_transform  # [d_source, d_target]

        logger.info("Aligned CKA: %.4f (deviation: %.2e)", cka_achieved, numerical_deviation)
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

    def save_bridge(
        self,
        result: BridgeGeneratorResult,
        path: Path,
    ) -> None:
        """Save a bridge to safetensors format.

        Args:
            result: The bridge generation result
            path: Output path for the safetensors file
        """
        backend = self._backend

        try:
            from safetensors.numpy import save_file
            import numpy as np
        except ImportError:
            raise ImportError(
                "safetensors and numpy required for bridge saving. "
                "Install with: pip install safetensors numpy"
            )

        # Convert tensors to numpy
        transform_np = np.array(backend.tolist(result.transform))
        transform_inv_np = np.array(backend.tolist(result.transform_inv))

        # Prepare tensors dict
        tensors = {
            "transform": transform_np.astype(np.float32),
            "transform_inv": transform_inv_np.astype(np.float32),
        }

        # Prepare metadata
        metadata = {
            "scale_ratio": str(result.scale_ratio),
            "source_dim": str(result.source_dim),
            "target_dim": str(result.target_dim),
            "cka_achieved": str(result.cka_achieved),
            "raw_cka": str(result.raw_cka),
            "n_samples": str(result.n_samples),
            "source_name": result.source_name,
            "target_name": result.target_name,
            "created_at": result.created_at.isoformat(),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_file(tensors, str(path), metadata=metadata)
        logger.info("Bridge saved to: %s", path)

    def load_bridge(
        self,
        path: Path,
    ) -> CrossModalBridge:
        """Load a bridge from safetensors file.

        Args:
            path: Path to the safetensors file

        Returns:
            CrossModalBridge instance
        """
        return CrossModalBridge.load(path, self._backend)

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
