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

"""Generalized Procrustes Analysis (GPA) for multi-model alignment.

Aligns multiple representation matrices to a consensus space using orthogonal
Procrustes transforms and a consensus update step.

References:
    - Gower, J. C. (1975). "Generalized Procrustes Analysis."
      Psychometrika 40(1):33-51. https://doi.org/10.1007/BF02291478
    - Schönemann, P. H. (1966). "A Generalized Solution of the Orthogonal
      Procrustes Problem." Psychometrika 31(1):1-10.
      https://doi.org/10.1007/BF02289451
    - Karcher, H. (1977). "Riemannian Center of Mass and Mollifier Smoothing."
      Communications on Pure and Applied Mathematics 30(5):509-541.
      https://doi.org/10.1002/cpa.3160300502
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.lie_rotation import so_geodesic_distance
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# Parameters are derived from data; Fréchet mean is used, reflections and scaling are disabled.


@dataclass(frozen=True)
class Result:
    consensus: list[list[float]]  # Kept as list for compatibility, could be mx.array in future
    rotations: list[list[list[float]]]
    scales: list[float]
    residuals: list[list[list[float]]]
    converged: bool
    iterations: int
    alignment_error: float
    per_model_errors: list[float]
    consensus_variance_ratio: float
    sample_count: int
    dimension: int
    model_count: int

    @property
    def summary(self) -> str:
        return (
            "Generalized Procrustes Analysis (MLX Accelerated)\n"
            f"- Models: {self.model_count}\n"
            f"- Samples: {self.sample_count} x {self.dimension}\n"
            f"- Converged: {self.converged} (iterations: {self.iterations})\n"
            f"- Alignment Error: {self.alignment_error:.4f}\n"
            f"- Consensus Variance: {self.consensus_variance_ratio * 100:.1f}%"
        )


class GeneralizedProcrustes:
    """Generalized Procrustes Analysis using backend acceleration.

    Supports Fréchet mean (Riemannian) for consensus computation. Use the
    Fréchet mean for curved embedding spaces.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._riemannian = None  # Lazy init for Fréchet mean

    def _array_to_list(self, array: "Array") -> list[float]:
        """Convert 1D array to Python list using native tolist() - O(1) vs O(n)."""
        flat = self._backend.reshape(array, (-1,))
        return self._backend.tolist(flat)

    def _array_to_2d_list(self, array: "Array") -> list[list[float]]:
        """Convert 2D array to nested Python list using native tolist() - O(1) vs O(n*m)."""
        return self._backend.tolist(array)

    def _array_to_3d_list(self, array: "Array") -> list[list[list[float]]]:
        """Convert 3D array to nested Python list using native tolist() - O(1) vs O(n*m*k)."""
        return self._backend.tolist(array)

    def _compute_consensus(
        self,
        aligned_X: "Array",
    ) -> "Array":
        """Compute consensus using a Fréchet mean for curved embedding spaces.

        Args:
            aligned_X: [M, N, K] aligned activation tensor

        Returns:
            [N, K] consensus matrix

        References:
            - Karcher, H. (1977). "Riemannian Center of Mass and Mollifier
              Smoothing." Communications on Pure and Applied Mathematics.
        """
        # With two models, each sample has two points. The k-NN graph collapses
        # to a single edge, so the Fréchet mean is the midpoint along that edge.
        if aligned_X.shape[0] <= 2:
            return self._backend.mean(aligned_X, axis=0)

        # Fréchet mean for curvature-aware consensus
        # For each sample point (row), compute Fréchet mean across models
        # aligned_X: [M, N, K] -> iterate over N samples
        if self._riemannian is None:
            from modelcypher.core.domain.geometry.riemannian_utils import (
                RiemannianGeometry,
            )

            self._riemannian = RiemannianGeometry(backend=self._backend)

        backend = self._backend
        model_count = int(aligned_X.shape[0])
        _M, N, K = aligned_X.shape[0], aligned_X.shape[1], aligned_X.shape[2]

        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        tol = sqrt_scalar(machine_epsilon(backend, aligned_X), backend)
        frechet_max_iter = max(50, 10 * K)

        points_batch = backend.transpose(aligned_X, axes=(1, 0, 2))
        return self._riemannian.frechet_mean_batch(
            points_batch,
            max_iterations=frechet_max_iter,
            tolerance=tol,
            k_neighbors=max(1, model_count - 1),
        )

    def align(
        self,
        activations: list[list[list[float]]],
    ) -> Result | None:
        """Align multiple model activations using Generalized Procrustes Analysis.

        All parameters are derived from data; no configuration needed.
        """
        model_count = len(activations)
        if model_count < 2:  # Need at least 2 models
            return None

        # Verify dims
        n = len(activations[0])
        if n == 0:
            return None
        k = len(activations[0][0])
        if k == 0:
            return None

        # Check all match
        for act in activations:
            if len(act) != n or len(act[0]) != k:
                return None

        # Build tensor stack [M, N, K]
        X = self._backend.array(activations)

        # Derive convergence threshold from machine epsilon
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
        eps = machine_epsilon(self._backend, X)
        convergence_threshold = sqrt_scalar(eps, self._backend)  # sqrt(eps) for relative error

        # 1. Centering
        means = self._backend.mean(X, axis=1, keepdims=True)
        X = X - means

        # 2. No scaling - preserves magnitudes (scaling distorts geometry)
        scales = self._backend.ones((model_count,))

        # Two-model alignment has a closed-form Procrustes solution.
        # Avoid iterative updates for M=2 to keep precision high and runtime low.
        if model_count == 2:
            b = self._backend
            base_eye = b.eye(k)

            X0 = X[0]
            X1 = X[1]

            # Check for identical matrices (self-alignment case).
            # When X0 = X1, the optimal rotation is identity and error is exactly 0.
            # Skip SVD to avoid numerical errors in the null space.
            diff = X0 - X1
            # Use geodesic norms for matrix distance (flatten to row vector)
            # Batch all norm computations in single eval for efficiency
            diff_norm_arr = geodesic_norms(b.reshape(diff, (1, -1)), b)
            x_norm_arr = geodesic_norms(b.reshape(X0, (1, -1)), b)
            x1_norm_arr = geodesic_norms(b.reshape(X1, (1, -1)), b)
            b.eval(diff_norm_arr, x_norm_arr, x1_norm_arr)  # Single eval for all norms
            diff_norm = float(b.to_scalar(diff_norm_arr))
            x_norm = float(b.to_scalar(x_norm_arr))
            x1_norm = float(b.to_scalar(x1_norm_arr))
            eps = float(division_epsilon(b, X0))

            if diff_norm <= eps * max(x_norm, 1.0):
                # Matrices are identical - return exact zero alignment error
                Rs = b.stack([base_eye, base_eye], axis=0)
                zero_residuals = b.zeros((2, n, k))
                zero_errors = b.zeros((2,))
                b.eval(Rs, zero_residuals, zero_errors)

                return Result(
                    consensus=self._array_to_2d_list(X0),
                    rotations=self._array_to_3d_list(Rs),
                    scales=self._array_to_list(scales),
                    residuals=self._array_to_3d_list(zero_residuals),
                    converged=True,
                    iterations=1,
                    alignment_error=0.0,
                    per_model_errors=self._array_to_list(zero_errors),
                    consensus_variance_ratio=1.0,
                    sample_count=n,
                    dimension=k,
                    model_count=model_count,
                )

            # Check for degenerate (zero/near-zero) matrices that would crash SVD/det
            # x1_norm already computed above in batched eval

            if x_norm <= eps or x1_norm <= eps:
                # One or both matrices are degenerate - use identity rotation
                # This avoids SVD on zero matrices which crashes Metal
                Rs = b.stack([base_eye, base_eye], axis=0)
                aligned_X = b.stack([X0, X1], axis=0)
                consensus = self._compute_consensus(aligned_X)
                residuals = aligned_X - consensus
                residuals_flat = b.reshape(residuals, (model_count, -1))
                per_model_norms = geodesic_norms(residuals_flat, b)
                per_model_errors = per_model_norms * per_model_norms
                total_error = float(b.to_scalar(b.sum(per_model_errors)))
                b.eval(Rs, residuals, per_model_errors)

                return Result(
                    consensus=self._array_to_2d_list(consensus),
                    rotations=self._array_to_3d_list(Rs),
                    scales=self._array_to_list(scales),
                    residuals=self._array_to_3d_list(residuals),
                    converged=True,
                    iterations=1,
                    alignment_error=total_error,
                    per_model_errors=self._array_to_list(per_model_errors),
                    consensus_variance_ratio=1.0,
                    sample_count=n,
                    dimension=k,
                    model_count=model_count,
                )

            M = b.matmul(b.transpose(X1), X0)
            U, _, Vt = geodesic_svd(b, M)
            R1_raw = b.matmul(U, Vt)

            # Never allow reflections - preserves orientation
            # Use lazy where-based approach to avoid eval in hot path
            det_val = b.det(R1_raw)
            sign = b.where(det_val < 0, b.array(-1.0), b.array(1.0))
            # Create scale vector for last column: [1, 1, ..., 1, sign]
            k_dim = int(U.shape[1])
            if k_dim > 1:
                scale = b.concatenate([b.ones((k_dim - 1,)), b.reshape(sign, (1,))], axis=0)
                U_scaled = U * b.reshape(scale, (1, k_dim))
                R1 = b.matmul(U_scaled, Vt)
            else:
                # k=1 edge case: just apply sign directly
                R1 = R1_raw * sign
            # No eval needed here - all operations are lazy

            Rs = b.stack([base_eye, R1], axis=0)
            aligned_X = b.stack([X0, b.matmul(X1, R1)], axis=0)
            consensus = self._compute_consensus(aligned_X)

            residuals = aligned_X - consensus
            # Compute geodesic norms for each model's residuals
            residuals_flat = b.reshape(residuals, (model_count, -1))  # [M, n*k]
            per_model_norms = geodesic_norms(residuals_flat, b)  # [M]
            per_model_errors = per_model_norms * per_model_norms  # squared geodesic norms
            # Total error is sum of squared geodesic norms
            current_error_arr = b.sum(per_model_errors)
            # Total variance via geodesic norm
            aligned_flat = b.reshape(aligned_X, (1, -1))
            total_var_arr = geodesic_norms(aligned_flat, b)
            total_var_arr = total_var_arr * total_var_arr  # squared
            b.eval(current_error_arr, total_var_arr, per_model_errors)
            current_error = float(b.to_scalar(current_error_arr))
            total_var = float(b.to_scalar(total_var_arr))
            var_eps = float(division_epsilon(b, aligned_X))
            ratio = 1.0 - (current_error / total_var) if total_var > var_eps else 0.0

            return Result(
                consensus=self._array_to_2d_list(consensus),
                rotations=self._array_to_3d_list(Rs),
                scales=self._array_to_list(scales),
                residuals=self._array_to_3d_list(residuals),
                converged=True,
                iterations=1,
                alignment_error=current_error,
                per_model_errors=self._array_to_list(per_model_errors),
                consensus_variance_ratio=ratio,
                sample_count=n,
                dimension=k,
                model_count=model_count,
            )

        # Initialize Rotations (Identity)
        base_eye = self._backend.eye(k)
        Rs = self._backend.stack([base_eye] * model_count)  # [M, K, K]

        # Initial consensus (Fréchet mean for curvature-aware initialization)
        consensus = self._compute_consensus(X)  # [N, K]

        aligned_X = X  # Initially aligned is just centered X
        X_t = self._backend.transpose(X, axes=(0, 2, 1))

        # Derive max_iterations from number of models
        # GPA typically converges in O(k) iterations; use 10*M as safety limit
        gpa_max_iterations = max(100, 10 * model_count)
        converged = False
        iterations = 0
        current_error = 0.0

        for iter_idx in range(gpa_max_iterations):
            iterations = iter_idx + 1

            M_matrices = self._backend.matmul(X_t, consensus)

            b = self._backend
            U_batch, _, Vt_batch = b.svd(M_matrices)
            Rs = b.matmul(U_batch, Vt_batch)

            # Never allow reflections - preserves orientation (batched)
            # Note: det and where are kept lazy - no eval needed here
            dets = b.det(Rs)
            signs = b.where(dets < 0, b.full(dets.shape, -1.0), b.full(dets.shape, 1.0))
            k = int(U_batch.shape[2])
            if k > 0:
                ones = b.ones((model_count, k - 1), dtype=U_batch.dtype)
                last = b.reshape(signs, (-1, 1))
                scale = b.concatenate([ones, last], axis=1)
                U_batch = U_batch * b.reshape(scale, (model_count, 1, k))
                Rs = b.matmul(U_batch, Vt_batch)

            # Update Aligned X
            aligned_X = self._backend.matmul(X, Rs)

            # New Consensus (always uses Fréchet mean for curvature-awareness)
            new_consensus = self._compute_consensus(aligned_X)

            # Error - normalize by total data energy for scale-invariant convergence
            diffs = aligned_X - new_consensus
            # Use geodesic norms for error computation
            diffs_flat = self._backend.reshape(diffs, (1, -1))
            diffs_norm = geodesic_norms(diffs_flat, self._backend)
            current_error_arr = diffs_norm * diffs_norm  # squared geodesic norm
            aligned_flat = self._backend.reshape(aligned_X, (1, -1))
            aligned_norm = geodesic_norms(aligned_flat, self._backend)
            total_energy_arr = aligned_norm * aligned_norm  # squared
            self._backend.eval(current_error_arr, total_energy_arr)
            current_error = float(self._backend.to_scalar(current_error_arr))
            total_energy = float(self._backend.to_scalar(total_energy_arr))

            # Use relative residual error instead of relative change in error
            # This is scale-invariant: same behavior regardless of input magnitude
            eps = float(division_epsilon(self._backend, aligned_X))
            rel_change = current_error / max(total_energy, eps)
            if rel_change < convergence_threshold:
                converged = True
                consensus = new_consensus
                break

            consensus = new_consensus

        # Final outputs
        residuals = aligned_X - consensus
        # Compute geodesic norms for each model's residuals
        residuals_flat = self._backend.reshape(residuals, (model_count, -1))  # [M, n*k]
        per_model_norms = geodesic_norms(residuals_flat, self._backend)  # [M]
        per_model_errors = per_model_norms * per_model_norms  # squared geodesic norms

        # Variance calc using geodesic norm
        aligned_flat = self._backend.reshape(aligned_X, (1, -1))
        total_var_arr = geodesic_norms(aligned_flat, self._backend)
        total_var_arr = total_var_arr * total_var_arr  # squared
        self._backend.eval(total_var_arr)
        total_var = float(self._backend.to_scalar(total_var_arr))
        residual_var = current_error
        var_eps = float(division_epsilon(self._backend, aligned_X))
        ratio = 1.0 - (residual_var / total_var) if total_var > var_eps else 0.0

        return Result(
            consensus=self._array_to_2d_list(consensus),
            rotations=self._array_to_3d_list(Rs),
            scales=self._array_to_list(scales),
            residuals=self._array_to_3d_list(residuals),
            converged=converged,
            iterations=iterations,
            alignment_error=current_error,
            per_model_errors=self._array_to_list(per_model_errors),
            consensus_variance_ratio=ratio,
            sample_count=n,
            dimension=k,
            model_count=model_count,
        )

    def align_crms(
        self,
        crms: list[ConceptResponseMatrix],
        layer: int,
    ) -> Result | None:
        extracted: list[list[list[float]]] = []
        min_dim = None
        max_dim = None
        for crm in crms:
            if layer not in crm.activations:
                return None
            acts = crm.activations[layer]
            anchors = sorted(acts.keys())
            if not anchors:
                return None
            mat = [acts[k].activation for k in anchors]
            if not mat or not mat[0]:
                return None
            dim = len(mat[0])
            min_dim = dim if min_dim is None else min(min_dim, dim)
            max_dim = dim if max_dim is None else max(max_dim, dim)
            extracted.append(mat)

        if min_dim is None or min_dim <= 0:
            return None

        # Log warning if significant dimension truncation occurs
        if max_dim is not None and max_dim > min_dim:
            loss_pct = (1 - min_dim / max_dim) * 100
            if loss_pct > 25:
                logger.warning(
                    f"GPA dimension truncation at layer {layer}: {max_dim} -> {min_dim} "
                    f"({loss_pct:.1f}% dimension loss). Consider using projection-based "
                    f"alignment (CKA/Gram) to preserve more geometry."
                )

        # Truncate to the shared minimum dimension to align overlapping subspaces.
        trimmed = [[vec[:min_dim] for vec in mat] for mat in extracted]

        return self.align(trimmed)


# =============================================================================
# Per-Layer Rotation Continuity Analysis
# =============================================================================


@dataclass(frozen=True)
class LayerRotationResult:
    """Result of Procrustes alignment at a single layer."""

    layer_index: int
    rotation: list[list[float]] | None  # [k × k] orthogonal rotation matrix (optional)
    error: float  # Frobenius alignment error after rotation
    angular_deviation: float | None = None  # SO(n) geodesic distance from previous rotation
    rotation_delta: float | None = None  # ||log(R_L^T R_{L-1})||_F (Lie algebra norm)


@dataclass
class RotationContinuityResult:
    """Analysis of how rotation requirements change across layers."""

    source_model: str
    target_model: str
    layers: list[LayerRotationResult]
    global_rotation_error: float
    smoothness_ratio: float
    rotation_roughness: float
    mean_angular_velocity: float
    requires_per_layer_alignment: bool  # Renamed from h5_null_rejected
    source_dimension: int
    target_dimension: int
    anchor_count: int

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        verdict = (
            "Per-layer alignment REQUIRED: rotations change significantly across layers"
            if self.requires_per_layer_alignment
            else "Global rotation SUFFICIENT: single rotation works for all layers"
        )
        mean_layer_error = (
            sum(layer_r.error for layer_r in self.layers) / len(self.layers) if self.layers else 0.0
        )
        return (
            "Rotation Continuity Analysis\n"
            "============================\n"
            f"Source: {self.source_model}\n"
            f"Target: {self.target_model}\n"
            f"Dimensions: {self.source_dimension} → {self.target_dimension}\n"
            f"Anchors: {self.anchor_count}\n"
            f"Layers: {len(self.layers)}\n\n"
            "Results:\n"
            f"- Global rotation error: {self.global_rotation_error:.4f}\n"
            f"- Mean per-layer error: {mean_layer_error:.4f}\n"
            f"- Smoothness ratio: {self.smoothness_ratio:.3f}\n"
            f"- Rotation roughness: {self.rotation_roughness:.4f}\n"
            f"- Mean angular velocity: {self.mean_angular_velocity:.4f} rad\n\n"
            f"Conclusion: {verdict}"
        )


class RotationContinuityAnalyzer:
    """Analyze rotation continuity across layers."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def _array_to_2d_list(self, array: "Array") -> list[list[float]]:
        """Convert 2D array to nested Python list using native tolist()."""
        return self._backend.tolist(array)

    def _stack_layer_activations(self, activations: Any) -> "Array | None":
        """Stack activations into a 2D array, handling list or array formats."""
        backend = self._backend
        if activations is None:
            return None
        if hasattr(activations, "shape"):
            arr = backend.array(activations)
            shape = backend.shape(arr)
            if len(shape) == 1:
                arr = backend.reshape(arr, (1, shape[0]))
            backend.eval(arr)
            return arr
        if not activations:
            return None
        arr = backend.stack([backend.array(a) for a in activations], axis=0)
        backend.eval(arr)
        return arr

    def compute_per_layer_alignments(
        self,
        source_activations: dict[int, dict[str, list[float]]],  # layer -> anchor -> activation
        target_activations: dict[int, dict[str, list[float]]],
        source_model: str,
        target_model: str,
        smoothness_ratios: list[float] | None = None,
    ) -> RotationContinuityResult | None:
        """
        Analyze rotation continuity across layers.

        All parameters derived from data - no configuration needed.
        - Reflections: never allowed
        - Smoothness threshold: derived from provided smoothness_ratios distribution

        Args:
            source_activations: Source model activations [layer: [anchor: activation]].
            target_activations: Target model activations [layer: [anchor: activation]].
            source_model: Source model identifier.
            target_model: Target model identifier.
            smoothness_ratios: Historical smoothness ratios for threshold derivation.
                If None, uses 0.7 as threshold (based on empirical data from prior analyses).

        Returns:
            RotationContinuityResult, or None if alignment failed.
        """
        backend = self._backend

        # Get common layers
        common_layers = sorted(set(source_activations.keys()) & set(target_activations.keys()))
        if not common_layers:
            return None

        # Get common anchors from first layer
        first_layer = common_layers[0]
        source_first = source_activations.get(first_layer, {})
        target_first = target_activations.get(first_layer, {})

        common_anchors = sorted(set(source_first.keys()) & set(target_first.keys()))
        if len(common_anchors) < 3:
            return None  # Need at least 3 anchors

        # Get dimensions
        first_source_act = source_first.get(common_anchors[0], [])
        first_target_act = target_first.get(common_anchors[0], [])
        if not first_source_act or not first_target_act:
            return None

        source_dim = len(first_source_act)
        target_dim = len(first_target_act)
        shared_dim = min(source_dim, target_dim)

        # Compute per-layer alignments (batched SVD)
        layer_results: list[LayerRotationResult] = []
        prev_rotation: "Array | None" = None
        layer_payloads: list[tuple[int, "Array", "Array"]] = []
        m_matrices: list["Array"] = []

        for layer_idx in common_layers:
            source_layer = source_activations.get(layer_idx, {})
            target_layer = target_activations.get(layer_idx, {})

            # Build matrices from common anchors
            source_mat = []
            target_mat = []
            for anchor in common_anchors:
                s_act = source_layer.get(anchor)
                t_act = target_layer.get(anchor)
                if s_act is None or t_act is None:
                    continue
                source_mat.append(s_act[:shared_dim])
                target_mat.append(t_act[:shared_dim])

            if len(source_mat) < 3:
                continue

            # Compute Procrustes rotation inputs using backend
            source_arr = backend.array(source_mat)  # [n_anchors, shared_dim]
            target_arr = backend.array(target_mat)

            # Center
            source_arr = source_arr - backend.mean(source_arr, axis=0)
            target_arr = target_arr - backend.mean(target_arr, axis=0)

            # M = source^T @ target
            M = backend.matmul(backend.transpose(source_arr), target_arr)  # [d, d]

            layer_payloads.append((layer_idx, source_arr, target_arr))
            m_matrices.append(M)

        if not layer_payloads:
            return None

        m_batch = backend.stack(m_matrices, axis=0)
        U_batch, _, Vt_batch = backend.svd(m_batch)
        rotations_batch = backend.matmul(U_batch, Vt_batch)

        for idx, (layer_idx, source_arr, target_arr) in enumerate(layer_payloads):
            rotation = rotations_batch[idx]

            # Never allow reflections - preserves orientation
            det_val = backend.det(rotation)
            backend.eval(det_val)
            if float(backend.to_scalar(det_val)) < 0:
                U_i = U_batch[idx]
                Vt_i = Vt_batch[idx]
                U_fixed = backend.concatenate([U_i[:, :-1], -U_i[:, -1:]], axis=1)
                rotation = backend.matmul(U_fixed, Vt_i)

            # Compute error using geodesic distances
            aligned_source = backend.matmul(source_arr, rotation)
            _, error_dist = geodesic_pairwise_metrics(aligned_source, target_arr, backend)
            error_arr = backend.sum(error_dist * error_dist)
            backend.eval(error_arr)
            error = float(backend.to_scalar(error_arr))

            # Compute angular deviation from previous layer
            angular_deviation = None
            rotation_delta = None
            if prev_rotation is not None:
                angular_deviation = so_geodesic_distance(prev_rotation, rotation, backend=backend)
                rotation_delta = angular_deviation * sqrt_scalar(2.0, backend)

            prev_rotation = rotation

            # Convert rotation to list for result
            backend.eval(rotation)
            rotation_list = self._array_to_2d_list(rotation)

            layer_results.append(
                LayerRotationResult(
                    layer_index=layer_idx,
                    rotation=rotation_list,
                    error=error,
                    angular_deviation=angular_deviation,
                    rotation_delta=rotation_delta,
                )
            )

        if not layer_results:
            return None

        # Compute global rotation (using all layers concatenated)
        all_source = []
        all_target = []
        for layer_idx in common_layers:
            source_layer = source_activations.get(layer_idx, {})
            target_layer = target_activations.get(layer_idx, {})
            for anchor in common_anchors:
                s_act = source_layer.get(anchor)
                t_act = target_layer.get(anchor)
                if s_act and t_act:
                    all_source.append(s_act[:shared_dim])
                    all_target.append(t_act[:shared_dim])

        global_source = backend.array(all_source)
        global_target = backend.array(all_target)
        global_source = global_source - backend.mean(global_source, axis=0)
        global_target = global_target - backend.mean(global_target, axis=0)

        M_global = backend.matmul(backend.transpose(global_source), global_target)
        U_g, _, Vt_g = geodesic_svd(backend, M_global)
        global_rotation = backend.matmul(U_g, Vt_g)

        # Never allow reflections - preserves orientation
        det_val = backend.det(global_rotation)
        backend.eval(det_val)
        if float(backend.to_scalar(det_val)) < 0:
            U_g_fixed = backend.concatenate([U_g[:, :-1], -U_g[:, -1:]], axis=1)
            global_rotation = backend.matmul(U_g_fixed, Vt_g)

        aligned_global = backend.matmul(global_source, global_rotation)
        _, global_dist = geodesic_pairwise_metrics(aligned_global, global_target, backend)
        global_error_arr = backend.sum(global_dist * global_dist)
        backend.eval(global_error_arr)
        global_error = float(backend.to_scalar(global_error_arr))

        # Compute metrics
        mean_layer_error = sum(layer_r.error for layer_r in layer_results) / len(layer_results)
        error_eps = float(division_epsilon(backend, global_error_arr))
        smoothness_ratio = mean_layer_error / max(global_error, error_eps)

        # Rotation roughness
        rotation_roughness = sum(
            layer_r.rotation_delta**2 for layer_r in layer_results if layer_r.rotation_delta is not None
        )

        # Mean angular velocity
        angular_devs = [
            layer_r.angular_deviation for layer_r in layer_results if layer_r.angular_deviation is not None
        ]
        mean_angular_velocity = sum(angular_devs) / max(len(angular_devs), 1)

        # Derive smoothness threshold from provided distribution or local ratios
        if smoothness_ratios and len(smoothness_ratios) >= 2:
            # Derive from distribution: mean - 1σ
            mean_sr = sum(smoothness_ratios) / len(smoothness_ratios)
            variance_sr = sum((r - mean_sr) ** 2 for r in smoothness_ratios) / len(smoothness_ratios)
            std_sr = variance_sr ** 0.5
            smoothness_threshold = max(0.0, mean_sr - std_sr)
        else:
            # Derive from current layer ratios to avoid hardcoded heuristics
            ratios = []
            for layer_r in layer_results:
                ratio = layer_r.error / max(global_error, error_eps)
                ratios.append(ratio)
            if len(ratios) >= 2:
                mean_sr = sum(ratios) / len(ratios)
                variance_sr = sum((r - mean_sr) ** 2 for r in ratios) / len(ratios)
                std_sr = variance_sr ** 0.5
                smoothness_threshold = max(0.0, mean_sr - std_sr)
            else:
                smoothness_threshold = smoothness_ratio

        # Requires per-layer alignment if smoothness_ratio < threshold
        requires_per_layer = smoothness_ratio < smoothness_threshold

        return RotationContinuityResult(
            source_model=source_model,
            target_model=target_model,
            layers=layer_results,
            global_rotation_error=global_error,
            smoothness_ratio=smoothness_ratio,
            rotation_roughness=rotation_roughness,
            mean_angular_velocity=mean_angular_velocity,
            requires_per_layer_alignment=requires_per_layer,
            source_dimension=source_dim,
            target_dimension=target_dim,
            anchor_count=len(common_anchors),
        )

    def compute_per_layer_alignments_from_arrays(
        self,
        source_layer_activations: dict[int, "Array | Any"],
        target_layer_activations: dict[int, "Array | Any"],
        source_model: str,
        target_model: str,
        smoothness_ratios: list[float] | None = None,
    ) -> RotationContinuityResult | None:
        """Analyze rotation continuity using per-layer activation arrays."""
        backend = self._backend

        common_layers = sorted(
            set(source_layer_activations.keys()) & set(target_layer_activations.keys())
        )
        if not common_layers:
            return None

        layer_results: list[LayerRotationResult] = []
        prev_rotation: "Array | None" = None
        layer_payloads: list[tuple[int, "Array", "Array"]] = []
        m_matrices: list["Array"] = []
        source_dim = 0
        target_dim = 0
        anchor_count = 0

        for layer_idx in common_layers:
            src_raw = self._stack_layer_activations(source_layer_activations.get(layer_idx))
            tgt_raw = self._stack_layer_activations(target_layer_activations.get(layer_idx))
            if src_raw is None or tgt_raw is None:
                continue

            n_samples = min(int(src_raw.shape[0]), int(tgt_raw.shape[0]))
            if n_samples < 3:
                continue
            if anchor_count == 0:
                anchor_count = n_samples

            src = src_raw[:n_samples, :]
            tgt = tgt_raw[:n_samples, :]
            backend.eval(src, tgt)

            src_dim = int(src.shape[1])
            tgt_dim = int(tgt.shape[1])
            if source_dim == 0:
                source_dim = src_dim
                target_dim = tgt_dim

            shared_dim = min(src_dim, tgt_dim)
            src = src[:, :shared_dim]
            tgt = tgt[:, :shared_dim]

            src = src - backend.mean(src, axis=0)
            tgt = tgt - backend.mean(tgt, axis=0)

            M = backend.matmul(backend.transpose(src), tgt)
            layer_payloads.append((layer_idx, src, tgt))
            m_matrices.append(M)

        if not layer_payloads:
            return None

        m_batch = backend.stack(m_matrices, axis=0)
        U_batch, _, Vt_batch = geodesic_svd(backend, m_batch)
        rotations_batch = backend.matmul(U_batch, Vt_batch)

        for idx, (layer_idx, source_arr, target_arr) in enumerate(layer_payloads):
            rotation = rotations_batch[idx]

            det_val = backend.det(rotation)
            backend.eval(det_val)
            if float(backend.to_scalar(det_val)) < 0:
                U_i = U_batch[idx]
                Vt_i = Vt_batch[idx]
                U_fixed = backend.concatenate([U_i[:, :-1], -U_i[:, -1:]], axis=1)
                rotation = backend.matmul(U_fixed, Vt_i)

            aligned_source = backend.matmul(source_arr, rotation)
            _, error_dist = geodesic_pairwise_metrics(aligned_source, target_arr, backend)
            error_arr = backend.sum(error_dist * error_dist)
            backend.eval(error_arr)
            error = float(backend.to_scalar(error_arr))

            angular_deviation = None
            rotation_delta = None
            if prev_rotation is not None:
                angular_deviation = so_geodesic_distance(prev_rotation, rotation, backend=backend)
                rotation_delta = angular_deviation * sqrt_scalar(2.0, backend)

            prev_rotation = rotation

            layer_results.append(
                LayerRotationResult(
                    layer_index=layer_idx,
                    rotation=None,
                    error=error,
                    angular_deviation=angular_deviation,
                    rotation_delta=rotation_delta,
                )
            )

        if not layer_results:
            return None

        all_source = [payload[1] for payload in layer_payloads]
        all_target = [payload[2] for payload in layer_payloads]
        global_source = backend.concatenate(all_source, axis=0)
        global_target = backend.concatenate(all_target, axis=0)
        global_source = global_source - backend.mean(global_source, axis=0)
        global_target = global_target - backend.mean(global_target, axis=0)

        M_global = backend.matmul(backend.transpose(global_source), global_target)
        U_g, _, Vt_g = geodesic_svd(backend, M_global)
        global_rotation = backend.matmul(U_g, Vt_g)

        det_val = backend.det(global_rotation)
        backend.eval(det_val)
        if float(backend.to_scalar(det_val)) < 0:
            U_g_fixed = backend.concatenate([U_g[:, :-1], -U_g[:, -1:]], axis=1)
            global_rotation = backend.matmul(U_g_fixed, Vt_g)

        aligned_global = backend.matmul(global_source, global_rotation)
        _, global_dist = geodesic_pairwise_metrics(aligned_global, global_target, backend)
        global_error_arr = backend.sum(global_dist * global_dist)
        backend.eval(global_error_arr)
        global_error = float(backend.to_scalar(global_error_arr))

        mean_layer_error = sum(layer_r.error for layer_r in layer_results) / len(layer_results)
        error_eps = float(division_epsilon(backend, global_error_arr))
        smoothness_ratio = mean_layer_error / max(global_error, error_eps)

        rotation_roughness = sum(
            layer_r.rotation_delta**2 for layer_r in layer_results if layer_r.rotation_delta is not None
        )

        angular_devs = [
            layer_r.angular_deviation for layer_r in layer_results if layer_r.angular_deviation is not None
        ]
        mean_angular_velocity = sum(angular_devs) / max(len(angular_devs), 1)

        if smoothness_ratios and len(smoothness_ratios) >= 2:
            mean_sr = sum(smoothness_ratios) / len(smoothness_ratios)
            variance_sr = sum((r - mean_sr) ** 2 for r in smoothness_ratios) / len(smoothness_ratios)
            std_sr = variance_sr ** 0.5
            smoothness_threshold = max(0.0, mean_sr - std_sr)
        else:
            ratios = []
            for layer_r in layer_results:
                ratio = layer_r.error / max(global_error, error_eps)
                ratios.append(ratio)
            if len(ratios) >= 2:
                mean_sr = sum(ratios) / len(ratios)
                variance_sr = sum((r - mean_sr) ** 2 for r in ratios) / len(ratios)
                std_sr = variance_sr ** 0.5
                smoothness_threshold = max(0.0, mean_sr - std_sr)
            else:
                smoothness_threshold = smoothness_ratio

        requires_per_layer = smoothness_ratio < smoothness_threshold

        return RotationContinuityResult(
            source_model=source_model,
            target_model=target_model,
            layers=layer_results,
            global_rotation_error=global_error,
            smoothness_ratio=smoothness_ratio,
            rotation_roughness=rotation_roughness,
            mean_angular_velocity=mean_angular_velocity,
            requires_per_layer_alignment=requires_per_layer,
            source_dimension=source_dim,
            target_dimension=target_dim,
            anchor_count=anchor_count,
        )


# =============================================================================
# Rotation Flow Analysis (Lie Algebra Path)
# =============================================================================


@dataclass(frozen=True)
class RotationFlowResult:
    """Result of rotation flow analysis across layers.

    Tracks how alignment rotations evolve across layers using Lie algebra.
    Smooth flow indicates coherent alignment; jumps indicate layer-specific
    coordinate changes that may require per-layer handling.

    Attributes:
        layer_indices: Indices of analyzed layers.
        rotation_speeds: ||ω_L||_F at each layer transition [n_layers-1].
        rotation_accelerations: ||ω_{L+1} - ω_L||_F [n_layers-2].
        cumulative_rotation: Cumulative angle from layer 0 [n_layers-1].
        lie_algebra_norms: Raw ||log(R_L)||_F [n_layers].
        max_jump_layer: Layer with largest rotation discontinuity.
        max_jump_magnitude: Magnitude of the largest jump.
    """

    layer_indices: tuple[int, ...]
    rotation_speeds: "Array"
    rotation_accelerations: "Array"
    cumulative_rotation: "Array"
    lie_algebra_norms: "Array"
    max_jump_layer: int
    max_jump_magnitude: float


class RotationFlowAnalyzer:
    """Analyze rotation flow across layers using Lie algebra.

    Tracks how Procrustes alignment rotations change layer by layer.
    Uses so(n) logarithm to measure angular velocity and acceleration.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute_rotation_flow(
        self,
        rotations: list["Array"],
        layer_indices: list[int] | None = None,
    ) -> RotationFlowResult:
        """Compute rotation flow metrics from per-layer rotation matrices.

        Given a sequence of Procrustes rotation matrices R_0, R_1, ..., R_{L-1},
        computes:
        - Angular velocity: ω_L = log(R_{L+1} @ R_L^T) ∈ so(d)
        - Rotation speed: ||ω_L||_F
        - Acceleration: ||ω_{L+1} - ω_L||_F

        Args:
            rotations: List of rotation matrices [d, d] per layer.
            layer_indices: Optional layer indices (defaults to 0..L-1).

        Returns:
            RotationFlowResult with flow metrics.
        """
        b = self._backend

        n_layers = len(rotations)
        if n_layers == 0:
            return RotationFlowResult(
                layer_indices=(),
                rotation_speeds=b.array([]),
                rotation_accelerations=b.array([]),
                cumulative_rotation=b.array([]),
                lie_algebra_norms=b.array([]),
                max_jump_layer=0,
                max_jump_magnitude=0.0,
            )

        if layer_indices is None:
            layer_indices = list(range(n_layers))

        # Convert to backend arrays
        R_list = [b.array(R) for R in rotations]
        for R in R_list:
            b.eval(R)

        d = int(b.shape(R_list[0])[0])

        # Import Lie algebra functions
        from modelcypher.core.domain.geometry.lie_rotation import so_log

        # Compute Lie algebra norms for each rotation
        lie_norms = []
        for R in R_list:
            log_R = so_log(R, backend=b)
            b.eval(log_R)
            norm = b.sqrt(b.sum(log_R * log_R))
            b.eval(norm)
            lie_norms.append(float(b.to_scalar(norm)))

        lie_algebra_norms = b.array(lie_norms)
        b.eval(lie_algebra_norms)

        if n_layers < 2:
            return RotationFlowResult(
                layer_indices=tuple(layer_indices),
                rotation_speeds=b.array([]),
                rotation_accelerations=b.array([]),
                cumulative_rotation=b.array([]),
                lie_algebra_norms=lie_algebra_norms,
                max_jump_layer=layer_indices[0],
                max_jump_magnitude=0.0,
            )

        # Compute relative rotations: R_rel[L] = R_{L+1} @ R_L^T
        # This gives the rotation between consecutive layers
        speeds = []
        omega_list = []  # Store Lie algebra elements for acceleration
        cumulative = []
        cumsum = 0.0

        for i in range(n_layers - 1):
            R_curr = R_list[i]
            R_next = R_list[i + 1]

            # Relative rotation from layer i to layer i+1
            R_rel = b.matmul(R_next, b.transpose(R_curr))
            b.eval(R_rel)

            # Angular velocity in Lie algebra
            omega = so_log(R_rel, backend=b)
            b.eval(omega)
            omega_list.append(omega)

            # Speed = Frobenius norm of omega
            speed = b.sqrt(b.sum(omega * omega))
            b.eval(speed)
            speed_val = float(b.to_scalar(speed))
            speeds.append(speed_val)

            # Cumulative rotation
            cumsum += speed_val
            cumulative.append(cumsum)

        rotation_speeds = b.array(speeds)
        cumulative_rotation = b.array(cumulative)
        b.eval(rotation_speeds, cumulative_rotation)

        # Compute accelerations: ||ω_{L+1} - ω_L||_F
        if n_layers < 3:
            rotation_accelerations = b.array([])
            b.eval(rotation_accelerations)
        else:
            accels = []
            for i in range(len(omega_list) - 1):
                diff = omega_list[i + 1] - omega_list[i]
                accel = b.sqrt(b.sum(diff * diff))
                b.eval(accel)
                accels.append(float(b.to_scalar(accel)))

            rotation_accelerations = b.array(accels)
            b.eval(rotation_accelerations)

        # Find max jump
        if len(speeds) > 0:
            max_idx = int(b.to_scalar(b.argmax(rotation_speeds)))
            max_jump_layer = layer_indices[max_idx]
            max_jump_magnitude = speeds[max_idx]
        else:
            max_jump_layer = layer_indices[0]
            max_jump_magnitude = 0.0

        return RotationFlowResult(
            layer_indices=tuple(layer_indices),
            rotation_speeds=rotation_speeds,
            rotation_accelerations=rotation_accelerations,
            cumulative_rotation=cumulative_rotation,
            lie_algebra_norms=lie_algebra_norms,
            max_jump_layer=max_jump_layer,
            max_jump_magnitude=max_jump_magnitude,
        )

    def analyze_layer_alignments(
        self,
        source_layer_activations: dict[int, "Array"],
        target_layer_activations: dict[int, "Array"],
    ) -> RotationFlowResult | None:
        """Compute rotation flow from layer activations.

        Computes Procrustes alignment at each layer, then analyzes how
        the rotations evolve.

        Args:
            source_layer_activations: Dict of layer_idx -> activations [n, d_s].
            target_layer_activations: Dict of layer_idx -> activations [n, d_t].

        Returns:
            RotationFlowResult, or None if alignment failed.
        """
        b = self._backend

        common_layers = sorted(
            set(source_layer_activations.keys()) & set(target_layer_activations.keys())
        )
        if not common_layers:
            return None

        rotations = []
        valid_layers = []

        for layer_idx in common_layers:
            src = source_layer_activations[layer_idx]
            tgt = target_layer_activations[layer_idx]

            src = b.array(src) if not hasattr(src, "shape") else src
            tgt = b.array(tgt) if not hasattr(tgt, "shape") else tgt
            b.eval(src, tgt)

            n_src = int(b.shape(src)[0])
            n_tgt = int(b.shape(tgt)[0])
            n = min(n_src, n_tgt)

            if n < 3:
                continue

            src = src[:n, :]
            tgt = tgt[:n, :]

            d_src = int(b.shape(src)[1])
            d_tgt = int(b.shape(tgt)[1])
            d = min(d_src, d_tgt)

            src = src[:, :d]
            tgt = tgt[:, :d]

            # Center
            src = src - b.mean(src, axis=0)
            tgt = tgt - b.mean(tgt, axis=0)
            b.eval(src, tgt)

            # Procrustes: R = argmin ||src @ R - tgt||_F
            M = b.matmul(b.transpose(src), tgt)
            U, _, Vt = b.svd(M)
            R = b.matmul(U, Vt)
            b.eval(R)

            # Ensure proper rotation (det = +1)
            det_val = b.det(R)
            b.eval(det_val)
            if float(b.to_scalar(det_val)) < 0:
                U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                R = b.matmul(U_fixed, Vt)
                b.eval(R)

            rotations.append(R)
            valid_layers.append(layer_idx)

        if not rotations:
            return None

        return self.compute_rotation_flow(rotations, valid_layers)


def compute_rotation_flow(
    rotations: list["Array"],
    layer_indices: list[int] | None = None,
    backend: "Backend | None" = None,
) -> RotationFlowResult:
    """Compute rotation flow from a list of rotation matrices (convenience).

    Args:
        rotations: List of rotation matrices [d, d] per layer.
        layer_indices: Optional layer indices.
        backend: Backend for tensor operations.

    Returns:
        RotationFlowResult with flow analysis.
    """
    analyzer = RotationFlowAnalyzer(backend)
    return analyzer.compute_rotation_flow(rotations, layer_indices)
