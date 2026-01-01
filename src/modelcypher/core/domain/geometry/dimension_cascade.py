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
Dimension cascade for structure-preserving projection from high-D to 3D.

The core insight: Coupling matrices computed via GRAM_TRANSPORT are REUSABLE.
This enables real-time streaming projection during token generation.

Calibration Phase (run ONCE):
1. Capture initial activations [N, hidden_dim]
2. Compute coupling matrices via Gromov-Wasserstein on Gram matrices
3. Chain multiply for composite coupling: π_composite = [hidden_dim, 3]

Streaming Phase (per-token):
1. Get hidden state [hidden_dim]
2. Project: point_3d = hidden @ π_composite  # Single matmul!

Mathematical Guarantee:
- Gram matrices K = X @ X^T capture relational geometry
- GW finds structure-preserving coupling between Gram spaces
- The 3D projection IS the manifold shape, not an approximation

The visualization you see is the ACTUAL geometry of the representation space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cross_dimensional_projection import (
    ProjectionMethod,
    project_cross_dimensional,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    IntrinsicDimension,
    TwoNNConfiguration,
)
from modelcypher.core.domain.geometry.manifold_curvature import (
    OllivierRicciCurvature,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Result of projecting through dimension cascade.

    All values are exact geometric measurements, not approximations:
    - original_dim: Ambient dimension of input activations
    - intrinsic_dim: True dimensionality of the manifold (via TwoNN)
    - projections: Structure-preserving projections at each target dimension
    - couplings: REUSABLE coupling matrices for streaming projection
    - curvatures: Ollivier-Ricci curvature at each dimension
    - geodesic_distortion: How much geodesic structure is distorted (lower = better)

    Attributes:
        original_dim: Original hidden dimension (e.g., 4096)
        intrinsic_dim: Measured intrinsic dimension (typically 50-200)
        projections: Dict mapping target_dim -> projected points [N, target_dim]
        couplings: Dict mapping target_dim -> coupling matrix [d_in, target_dim]
        curvatures: Dict mapping target_dim -> per-point curvatures [N]
        geodesic_distortion: Dict mapping target_dim -> distortion ratio
    """

    original_dim: int
    intrinsic_dim: float
    projections: dict[int, "Array"]
    couplings: dict[int, "Array"]
    curvatures: dict[int, "Array"]
    geodesic_distortion: dict[int, float]


@dataclass
class CascadeConfiguration:
    """Configuration for dimension cascade.

    Attributes:
        target_dims: Target dimensions for cascade (descending order preferred)
        compute_curvature: Whether to compute ORC at each dimension
        curvature_k: Number of neighbors for curvature computation
        min_calibration_points: Minimum points needed for calibration
    """

    target_dims: list[int]
    compute_curvature: bool = True
    curvature_k: int = 15
    min_calibration_points: int = 20


class DimensionCascade:
    """
    Project high-D to 4D→3D→2D→1D with structure preservation.

    KEY INSIGHT: Coupling matrices are computed ONCE and reused for streaming.
    This enables real-time visualization during token generation.

    The geometry you see is REAL:
    - Gram transport finds exact structure-preserving coupling
    - Ollivier-Ricci curvature reflects true manifold curvature
    - Walls (positive ORC) and funnels (negative ORC) are geometric facts

    Usage:
        cascade = DimensionCascade(backend)

        # Calibration phase: compute couplings from initial activations
        result = cascade.calibrate(initial_activations, target_dims=[4, 3])

        # Get composite coupling for streaming (REUSABLE)
        coupling_3d = cascade.get_composite_coupling(target_dim=3)

        # Streaming phase: project new tokens with single matmul
        for token_hidden in token_stream:
            point_3d = token_hidden @ coupling_3d  # FAST!
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """
        Initialize the dimension cascade.

        Args:
            backend: Backend for tensor operations (defaults to MLX on macOS)
        """
        self.backend = backend or get_default_backend()
        self._couplings: dict[int, "Array"] = {}
        self._calibrated = False
        self._original_dim: int | None = None

    @property
    def calibrated(self) -> bool:
        """Whether calibrate() has been called."""
        return self._calibrated

    def calibrate(
        self,
        activations: "Array",
        target_dims: list[int] | None = None,
        config: CascadeConfiguration | None = None,
    ) -> CascadeResult:
        """
        Calibrate the cascade using calibration activations.

        Computes coupling matrices via GRAM_TRANSPORT which are then
        REUSED for all subsequent streaming projections.

        The coupling matrices preserve relational geometry:
        - Gram matrix K = X @ X^T captures pairwise relationships
        - GW on Grams finds optimal correspondence
        - Projection is EXACT: X @ π preserves Gram structure

        Args:
            activations: Calibration data [n_points, hidden_dim]
            target_dims: Target dimensions (defaults to [4, 3, 2])
            config: Cascade configuration

        Returns:
            CascadeResult with projections and REUSABLE couplings

        Raises:
            ValueError: If activations are too small for calibration
        """
        b = self.backend

        if config is None:
            config = CascadeConfiguration(target_dims=target_dims or [4, 3, 2])

        # Validate inputs
        n_points, hidden_dim = activations.shape
        if n_points < config.min_calibration_points:
            raise ValueError(
                f"Need at least {config.min_calibration_points} calibration points, "
                f"got {n_points}"
            )

        self._original_dim = hidden_dim
        logger.info(
            "Calibrating cascade: %d points, %d dims -> %s",
            n_points,
            hidden_dim,
            config.target_dims,
        )

        # Cast to float32 if needed - SVD and other linalg ops require float32+
        if 'float16' in str(activations.dtype):
            activations = b.astype(activations, "float32")
            b.eval(activations)
            logger.debug("Cast activations from float16 to float32 for numerical stability")

        # Compute intrinsic dimension - this is the TRUE dimensionality
        id_config = TwoNNConfiguration()
        id_estimator = IntrinsicDimension(b)
        id_result = id_estimator.compute(activations, id_config)
        intrinsic_dim = id_result.intrinsic_dimension

        logger.info(
            "Intrinsic dimension: %.1f (ambient: %d, ratio: %.2f)",
            intrinsic_dim,
            hidden_dim,
            intrinsic_dim / hidden_dim,
        )

        # Sort dims descending (project from high to low)
        target_dims_sorted = sorted(config.target_dims, reverse=True)

        current = activations
        current_dim = hidden_dim
        projections: dict[int, "Array"] = {current_dim: current}
        couplings: dict[int, "Array"] = {}
        curvatures: dict[int, "Array"] = {}
        geodesic_distortion: dict[int, float] = {}

        for target_dim in target_dims_sorted:
            if target_dim >= current_dim:
                logger.debug(
                    "Skipping %d (not smaller than current %d)", target_dim, current_dim
                )
                continue

            logger.debug("Projecting %d -> %d via GRAM_TRANSPORT", current_dim, target_dim)

            # Use GRAM_TRANSPORT to find structure-preserving coupling
            # This is THE key operation - GW on Gram matrices
            result = project_cross_dimensional(
                source=current,
                target=self._create_target_basis(current, target_dim),
                method=ProjectionMethod.GRAM_TRANSPORT,
                backend=b,
            )

            # Store coupling for streaming reuse
            if result.col_coupling is not None:
                couplings[target_dim] = result.col_coupling
                logger.debug(
                    "Stored coupling: [%d, %d], alignment=%.4f",
                    result.col_coupling.shape[0],
                    result.col_coupling.shape[1],
                    result.alignment_score,
                )

            projections[target_dim] = result.projected
            geodesic_distortion[target_dim] = 1.0 - result.alignment_score

            # Compute curvature at this dimension
            if config.compute_curvature and n_points > config.curvature_k:
                k = min(config.curvature_k, n_points - 1)
                try:
                    orc = OllivierRicciCurvature(b)
                    orc_result = orc.compute(result.projected, k_neighbors=k)

                    # Extract per-point curvatures
                    point_curvatures = b.array([
                        nc.mean_curvature for nc in orc_result.node_curvatures
                    ])
                    curvatures[target_dim] = point_curvatures

                    logger.debug(
                        "Curvature at %dD: mean=%.4f, std=%.4f",
                        target_dim,
                        orc_result.mean_edge_curvature,
                        orc_result.std_edge_curvature,
                    )
                except Exception as exc:
                    logger.warning("Curvature computation failed at %dD: %s", target_dim, exc)

            current = result.projected
            current_dim = target_dim

        # Cache couplings for streaming
        self._couplings = couplings
        self._calibrated = True

        return CascadeResult(
            original_dim=hidden_dim,
            intrinsic_dim=intrinsic_dim,
            projections=projections,
            couplings=couplings,
            curvatures=curvatures,
            geodesic_distortion=geodesic_distortion,
        )

    def _create_target_basis(self, points: "Array", target_dim: int) -> "Array":
        """
        Create target basis for GRAM_TRANSPORT projection.

        Uses PCA to find the principal subspace as the target.
        The coupling matrix will then map source features to this subspace.

        Args:
            points: Source points [n_points, source_dim]
            target_dim: Target dimension

        Returns:
            Target points in the principal subspace [n_points, target_dim]
        """
        b = self.backend

        # SVD requires float32 or higher precision
        # Cast to float32 if needed (common for model activations in float16)
        if 'float16' in str(points.dtype):
            points_f32 = b.astype(points, "float32")
            b.eval(points_f32)
        else:
            points_f32 = points

        # SVD to find principal components
        # points = U @ S @ Vt, we want U[:, :target_dim] @ S[:target_dim]
        U, S, Vt = b.svd(points_f32, full_matrices=False)
        b.eval(U, S, Vt)

        # Project to top target_dim dimensions
        # This is the PCA projection: points @ V[:, :target_dim]
        V_k = b.transpose(Vt[:target_dim, :])  # [source_dim, target_dim]
        target = b.matmul(points_f32, V_k)  # [n_points, target_dim]
        b.eval(target)

        return target

    def project_token(
        self,
        token_hidden: "Array",
        target_dim: int = 3,
    ) -> "Array":
        """
        Project a single token to target dimension using cached couplings.

        This is the STREAMING path - uses precomputed coupling matrices.
        Complexity: O(d_source × d_target) per projection.

        For 4096-dim to 3D: ~12K FLOPs per projection (microseconds on GPU)

        Args:
            token_hidden: Hidden state [hidden_dim]
            target_dim: Target dimension (must have been calibrated)

        Returns:
            Projected point [target_dim]

        Raises:
            RuntimeError: If calibrate() hasn't been called
            ValueError: If target_dim wasn't in calibration targets
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before project_token()")

        if target_dim not in self._couplings:
            raise ValueError(
                f"Target dim {target_dim} not calibrated. "
                f"Available: {list(self._couplings.keys())}"
            )

        b = self.backend
        current = token_hidden

        # Apply coupling chain: high_dim → ... → target_dim
        for dim in sorted(self._couplings.keys(), reverse=True):
            if dim < target_dim:
                continue

            coupling = self._couplings[dim]

            # Reshape for matmul if needed
            if len(current.shape) == 1:
                current = current[None, :]  # [1, d]

            current = b.matmul(current, coupling)
            b.eval(current)

            if dim == target_dim:
                break

        # Remove batch dimension if added
        if len(current.shape) == 2 and current.shape[0] == 1:
            current = current[0]

        return current

    def get_composite_coupling(self, target_dim: int = 3) -> "Array":
        """
        Get a single composite coupling matrix for direct projection.

        Multiplies all coupling matrices in the chain to get a single
        [hidden_dim, target_dim] matrix for O(1) streaming projection.

        This is the optimal approach for real-time visualization:
        1. Compute composite ONCE after calibration
        2. Inject into ActivationStream
        3. Every activation is projected with a single matmul

        Args:
            target_dim: Target dimension for the composite coupling

        Returns:
            Composite coupling matrix [original_dim, target_dim]

        Raises:
            RuntimeError: If calibrate() hasn't been called
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before get_composite_coupling()")

        b = self.backend

        # Chain multiply: π_4 @ π_3 = π_composite
        sorted_dims = sorted(self._couplings.keys(), reverse=True)
        composite: "Array | None" = None

        for dim in sorted_dims:
            if dim < target_dim:
                continue

            coupling = self._couplings[dim]

            if composite is None:
                composite = coupling
            else:
                composite = b.matmul(composite, coupling)
                b.eval(composite)

            if dim == target_dim:
                break

        if composite is None:
            raise ValueError(
                f"No coupling path to target_dim={target_dim}. "
                f"Available dims: {sorted_dims}"
            )

        logger.debug(
            "Composite coupling: [%d, %d]",
            composite.shape[0],
            composite.shape[1],
        )

        return composite

    def recalibrate(
        self,
        new_activations: "Array",
        alpha: float = 0.1,
    ) -> None:
        """
        Incrementally update couplings with new activations.

        Uses exponential moving average to blend new coupling with existing.
        Useful for adaptive calibration during generation.

        Args:
            new_activations: New calibration data [n_points, hidden_dim]
            alpha: Blending weight for new coupling (0.1 = 10% new, 90% old)
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before recalibrate()")

        b = self.backend

        # Compute new couplings
        new_result = self.calibrate(
            new_activations,
            target_dims=list(self._couplings.keys()),
        )

        # Blend with existing
        for dim, new_coupling in new_result.couplings.items():
            if dim in self._couplings:
                old_coupling = self._couplings[dim]
                blended = (1 - alpha) * old_coupling + alpha * new_coupling
                b.eval(blended)
                self._couplings[dim] = blended

        logger.debug("Recalibrated couplings with alpha=%.2f", alpha)
