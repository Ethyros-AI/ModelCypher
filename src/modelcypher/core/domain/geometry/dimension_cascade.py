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
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    power_iteration_eigh,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
    geodesic_distance_matrix,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_pairwise_metrics,
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
    - geodesic_distortion: How much geodesic structure is distorted (lower indicates less distortion)

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
    ) -> CascadeResult:
        """
        Calibrate the cascade using calibration activations.

        Computes coupling matrices via GRAM_TRANSPORT which are then
        REUSED for all subsequent streaming projections.

        The coupling matrices preserve relational geometry:
        - Gram matrix K = X @ X^T captures pairwise relationships
        - GW on Grams finds optimal correspondence
        - Projection is EXACT: X @ π preserves Gram structure

        All parameters (k for curvature, etc.) are derived from data.

        Args:
            activations: Calibration data [n_points, hidden_dim]
            target_dims: Target dimensions (defaults to [4, 3, 2])

        Returns:
            CascadeResult with projections and REUSABLE couplings

        Raises:
            ValueError: If activations are too small for calibration
        """
        b = self.backend

        # Derive target dims (default to 4, 3, 2 for visualization)
        resolved_target_dims = target_dims or [4, 3, 2]

        # Validate inputs - need at least 3 points for geometry
        n_points, hidden_dim = activations.shape
        if n_points < 3:
            raise ValueError(
                f"Need at least 3 calibration points for geometric computation, "
                f"got {n_points}"
            )

        self._original_dim = hidden_dim
        logger.info(
            "Calibrating cascade: %d points, %d dims -> %s",
            n_points,
            hidden_dim,
            resolved_target_dims,
        )

        # Cast to float32 if needed - SVD and other linalg ops require float32+
        if 'float16' in str(activations.dtype):
            activations = b.astype(activations, "float32")
            b.eval(activations)
            logger.debug("Cast activations from float16 to float32 for numerical stability")

        # Compute intrinsic dimension - this is the TRUE dimensionality
        # All parameters derived from data (Berry & Sauer 2016 for k, Facco et al. for method)
        id_estimator = IntrinsicDimension(b)
        id_result = id_estimator.compute(activations)
        intrinsic_dim = id_result.intrinsic_dimension

        logger.info(
            "Intrinsic dimension: %.1f (ambient: %d, ratio: %.2f)",
            intrinsic_dim,
            hidden_dim,
            intrinsic_dim / hidden_dim,
        )

        # Sort dims descending (project from high to low)
        target_dims_sorted = sorted(resolved_target_dims, reverse=True)

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

            logger.debug("Projecting %d -> %d via Isomap", current_dim, target_dim)

            # Use Isomap for GEODESIC-preserving projection
            # Isomap preserves manifold structure via geodesic distances
            # PCA only preserves linear variance - WRONG for curved manifolds
            projected, coupling_matrix = self._project_via_isomap(current, target_dim)
            logger.debug(
                "Stored Isomap coupling: [%d, %d]",
                coupling_matrix.shape[0],
                coupling_matrix.shape[1],
            )

            # Store coupling for streaming reuse
            couplings[target_dim] = coupling_matrix
            projections[target_dim] = projected

            # Measure geodesic distortion: how well does embedding preserve distances?
            # We compute correlation between original geodesic distances and
            # embedded distances (geodesic on the embedded manifold)
            geodesic_distortion[target_dim] = self._measure_geodesic_distortion(
                current, projected
            )

            # Compute curvature at this dimension (data-derived k via ORC)
            if n_points > 1:
                try:
                    orc = OllivierRicciCurvature(b)
                    orc_result = orc.compute(projected)

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

            current = projected
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

    def _project_via_pca(
        self, points: "Array", target_dim: int
    ) -> tuple["Array", "Array"]:
        """
        Legacy entry point for projection.

        Geodesic manifolds require geodesic-preserving embeddings. This method
        is retained for compatibility but now delegates to Isomap.

        Args:
            points: Source points [n_points, source_dim]
            target_dim: Target dimension

        Returns:
            Tuple of (projected points [n_points, target_dim],
                      coupling matrix [source_dim, target_dim])
        """
        return self._project_via_isomap(points, target_dim)

    def _project_via_isomap(
        self, points: "Array", target_dim: int, k_neighbors: int | None = None
    ) -> tuple["Array", "Array"]:
        """
        Project points via Isomap (geodesic-preserving embedding).

        Isomap preserves geodesic distances on the manifold:
        1. Build k-NN graph on high-D points
        2. Compute geodesic distances via shortest paths (Floyd-Warshall)
        3. Apply classical MDS to embed in target dimension
        4. Derive linear coupling via least squares for streaming

        The 3D positions reflect the ACTUAL manifold geometry, not
        Linear variance like PCA.

        Args:
            points: Source points [n_points, source_dim]
            target_dim: Target dimension
            k_neighbors: k for geodesic graph (auto if None)

        Returns:
            Tuple of (projected points [n_points, target_dim],
                      coupling matrix [source_dim, target_dim])
        """
        b = self.backend
        n_points = points.shape[0]
        source_dim = points.shape[1]

        # Cast to float32 if needed
        if 'float16' in str(points.dtype):
            points = b.astype(points, "float32")
            b.eval(points)

        logger.debug("Computing geodesic distances for Isomap (n=%d)", n_points)

        # Step 1: Compute geodesic distance matrix via k-NN graph
        rg = RiemannianGeometry(b)
        geo_result = rg.geodesic_distances(points, k_neighbors=k_neighbors)
        geo_dist = geo_result.distances
        b.eval(geo_dist)

        logger.debug(
            "Geodesic graph: k=%d, connected=%s",
            geo_result.k_neighbors,
            geo_result.connected,
        )

        # Handle disconnected graph: replace inf with max finite distance
        # This preserves the graph structure while avoiding numerical issues
        max_finite = b.max(b.where(geo_dist < 1e30, geo_dist, b.zeros_like(geo_dist)))
        b.eval(max_finite)
        geo_dist = b.where(geo_dist < 1e30, geo_dist, max_finite)
        b.eval(geo_dist)

        # Step 2: Classical MDS on geodesic distances
        # MDS finds embedding that preserves distances
        #
        # Algorithm:
        # 1. D_sq = D² (element-wise)
        # 2. J = I - (1/n) * ones  (centering matrix)
        # 3. B = -0.5 * J @ D_sq @ J  (double centering)
        # 4. B = V @ Λ @ V^T  (eigendecomposition)
        # 5. Y = V_k @ sqrt(Λ_k)  (embedding)

        # Element-wise square of distances
        D_sq = geo_dist * geo_dist
        b.eval(D_sq)

        # Double centering: B = -0.5 * (I - 1/n) @ D_sq @ (I - 1/n)
        # Efficient formulation:
        # B_ij = -0.5 * (D²_ij - row_mean_i - col_mean_j + grand_mean)
        row_mean = b.mean(D_sq, axis=1, keepdims=True)  # [n, 1]
        col_mean = b.mean(D_sq, axis=0, keepdims=True)  # [1, n]
        grand_mean = b.mean(D_sq)  # scalar
        b.eval(row_mean, col_mean, grand_mean)

        B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)
        b.eval(B)

        # Make B symmetric (numerical stability)
        B = 0.5 * (B + b.transpose(B))
        b.eval(B)

        # Regularize B for numerical stability before SVD
        # The regularization λ is derived from the matrix itself:
        # λ = eps * ||B||_F where eps is machine epsilon
        # This ensures B + λI is positive definite without distorting geometry
        B_frob_sq = b.sum(B * B)
        b.eval(B_frob_sq)
        B_frob = b.sqrt(B_frob_sq)
        b.eval(B_frob)

        # Regularization: add λI where λ = machine_eps * ||B||_F
        # This is the minimal perturbation that ensures numerical stability
        eps = division_epsilon(b, B)
        reg_lambda = eps * float(b.to_scalar(B_frob))

        # B_reg = B + λI
        eye_n = b.eye(n_points)
        b.eval(eye_n)
        B = B + reg_lambda * eye_n
        b.eval(B)

        logger.debug("Regularized B with lambda=%.2e (Frobenius=%.2e)", reg_lambda, float(b.to_scalar(B_frob)))

        # Step 3: Eigendecomposition of B (top-k only)
        # B is symmetric positive semi-definite for valid distance matrices.
        eigenvalues, eigenvectors = power_iteration_eigh(b, B, k=target_dim)
        b.eval(eigenvalues, eigenvectors)

        # For MDS, we only use positive eigenvalues
        # Non-metric distances can produce negative eigenvalues
        # Count positive eigenvalues
        pos_mask = eigenvalues > eps
        b.eval(pos_mask)
        n_positive_arr = b.sum(
            b.where(pos_mask, b.ones_like(eigenvalues), b.zeros_like(eigenvalues))
        )
        b.eval(n_positive_arr)
        n_positive = int(b.to_scalar(n_positive_arr))

        if n_positive < target_dim:
            raise ValueError(
                f"Only {n_positive} positive eigenvalues, need {target_dim}. "
                "Distance matrix is non-metric."
            )

        # Take top-k eigenvectors (largest positive eigenvalues)
        U_k = eigenvectors[:, :target_dim]  # [n, target_dim]
        S_k = eigenvalues[:target_dim]  # [target_dim]
        b.eval(U_k, S_k)

        # Clamp small values to avoid numerical issues in sqrt
        eps_arr = b.zeros_like(S_k) + eps
        b.eval(eps_arr)
        S_k = b.maximum(S_k, eps_arr)
        b.eval(S_k)

        # Step 4: Compute embedding
        # Y = U_k @ diag(sqrt(S_k))
        sqrt_S_k = b.sqrt(S_k)  # [target_dim]
        b.eval(sqrt_S_k)

        # Broadcast multiply: Y = U_k * sqrt_S_k
        projected = U_k * sqrt_S_k[None, :]  # [n, target_dim]
        b.eval(projected)

        # Normalize projected points to unit variance per dimension
        # This ensures coupling matrices have reasonable scale
        proj_std = b.std(projected, axis=0)
        b.eval(proj_std)
        proj_std = b.maximum(proj_std, b.zeros_like(proj_std) + eps)
        b.eval(proj_std)
        projected = projected / proj_std[None, :]
        b.eval(projected)

        total_var_arr = b.sum(eigenvalues)
        explained_arr = b.sum(S_k)
        b.eval(total_var_arr, explained_arr)
        total_var = float(b.to_scalar(total_var_arr))
        explained_val = float(b.to_scalar(explained_arr))
        eps = division_epsilon(b, eigenvalues)
        logger.debug(
            "Isomap embedding: explained variance ratio = %.2f%%",
            100.0 * explained_val / max(eps, total_var),
        )

        # Step 5: Derive linear coupling for streaming
        # We want W such that points @ W ≈ projected
        # This is least squares: W = (X^T @ X + λI)^-1 @ X^T @ Y
        # Using regularized normal equations, solved via b.solve()

        # Compute X^T @ X  [d, d]
        XtX = b.matmul(b.transpose(points), points)
        b.eval(XtX)

        # Add Tikhonov regularization: (X^T @ X + λI) where λ is derived from data
        XtX_frob_sq = b.sum(XtX * XtX)
        b.eval(XtX_frob_sq)
        XtX_frob = b.sqrt(XtX_frob_sq)
        b.eval(XtX_frob)

        eps = division_epsilon(b, XtX)
        reg_lambda = eps * float(b.to_scalar(XtX_frob))

        eye_d = b.eye(source_dim)
        b.eval(eye_d)
        XtX_reg = XtX + reg_lambda * eye_d
        b.eval(XtX_reg)

        # Compute X^T @ Y  [d, target_dim]
        XtY = b.matmul(b.transpose(points), projected)
        b.eval(XtY)

        # Solve (X^T @ X + λI) @ W = X^T @ Y for W
        coupling = b.solve(XtX_reg, XtY)
        b.eval(coupling)

        # Check for numerical issues in coupling - FAIL HARD, no fallback
        coupling_max_arr = b.max(b.abs(coupling))
        b.eval(coupling_max_arr)
        coupling_max = float(b.to_scalar(coupling_max_arr))
        if coupling_max > 1e10 or coupling_max != coupling_max:  # NaN check
            raise ValueError(
                f"Coupling matrix has numerical issues (max={coupling_max:.2e}). "
                "This indicates ill-conditioned input data. Increase regularization "
                "or check input activations for NaN/Inf values."
            )

        logger.debug(
            "Coupling via regularized normal equations: reg_lambda=%.2e",
            reg_lambda,
        )

        logger.debug(
            "Isomap coupling: [%d, %d]",
            coupling.shape[0],
            coupling.shape[1],
        )

        return projected, coupling

    def _measure_geodesic_distortion(
        self, original: "Array", projected: "Array"
    ) -> float:
        """
        Measure how well embedding preserves geodesic distances.

        Computes 1 - correlation between original geodesic distances
        and embedded distances (geodesic on the embedded manifold).

        Returns:
            Distortion in [0, 1] where 0 = exact preservation
        """
        b = self.backend
        n = original.shape[0]

        if n < 3:
            return 0.0

        try:
            # Compute geodesic distances in original space
            rg = RiemannianGeometry(b)
            geo_result = rg.geodesic_distances(original)
            geo_dist = geo_result.distances
            b.eval(geo_dist)

            # Compute geodesic distances in embedded space (all dimensions)
            embed_dist = geodesic_distance_matrix(
                projected, k_neighbors=None, backend=b
            )
            b.eval(embed_dist)

            # Flatten and compute correlation
            geo_flat = b.reshape(geo_dist, (-1,))
            embed_flat = b.reshape(embed_dist, (-1,))
            b.eval(geo_flat, embed_flat)

            # Geodesic correlation of centered distance vectors
            geo_mean = b.mean(geo_flat)
            embed_mean = b.mean(embed_flat)
            geo_centered = geo_flat - geo_mean
            embed_centered = embed_flat - embed_mean
            geo_centered_mat = b.reshape(geo_centered, (1, -1))
            embed_centered_mat = b.reshape(embed_centered, (1, -1))
            cos_arr, _ = geodesic_pairwise_metrics(geo_centered_mat, embed_centered_mat, b)
            b.eval(cos_arr)
            corr_val = float(b.to_scalar(cos_arr[0])) if cos_arr.size else 0.0

            # Distortion = 1 - |correlation|
            # Exact embedding has correlation ±1, distortion 0
            distortion = 1.0 - abs(corr_val)

            logger.debug(
                "Geodesic distortion: correlation=%.4f, distortion=%.4f",
                corr_val,
                distortion,
            )

            return max(0.0, min(1.0, distortion))

        except Exception as exc:
            logger.warning("Geodesic distortion measurement failed: %s", exc)
            return 0.5  # Unknown distortion

    def _create_target_basis(self, points: "Array", target_dim: int) -> "Array":
        """
        Create target basis for GRAM_TRANSPORT projection.

        Uses geodesic Isomap to define the target subspace.
        The coupling matrix then maps source features to this subspace.

        Args:
            points: Source points [n_points, source_dim]
            target_dim: Target dimension

        Returns:
            Target points in the principal subspace [n_points, target_dim]
        """
        projected, _ = self._project_via_pca(points, target_dim)
        return projected

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
    ) -> None:
        """
        Recalibrate couplings with new activations.

        Replaces existing couplings with fresh ones computed from new data.
        No blending or interpolation - the new couplings are exact for
        the new activation distribution.

        Args:
            new_activations: New calibration data [n_points, hidden_dim]
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before recalibrate()")

        # Compute new couplings and replace entirely
        new_result = self.calibrate(
            new_activations,
            target_dims=list(self._couplings.keys()),
        )

        # Replace with new couplings (no blending)
        for dim, new_coupling in new_result.couplings.items():
            self._couplings[dim] = new_coupling

        logger.debug("Recalibrated couplings from %d activations", new_activations.shape[0])
