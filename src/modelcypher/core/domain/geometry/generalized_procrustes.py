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

Aligns multiple neural network representations to a common consensus space
using orthogonal Procrustes transformations. Uses Fréchet mean
(curvature-aware) for consensus computation.

Mathematical Background:
    Given k models with representations X_1, ..., X_k, GPA finds:
    - Consensus C: The common reference representation
    - Rotations R_1, ..., R_k: Orthogonal matrices aligning each X_i to C
    - Scales s_1, ..., s_k: Optional scaling factors

    Minimizes: Σᵢ ||sᵢ Xᵢ Rᵢ - C||²_F

    Uses iterative refinement:
    1. Initialize consensus as first model
    2. Align each model to consensus via SVD
    3. Update consensus as (Fréchet) mean of aligned models
    4. Repeat until convergence

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
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.geometry.numerical_stability import (
    acos_scalar,
    division_epsilon,
    geodesic_svd,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.vector_math import (
    geodesic_norms,
    geodesic_pairwise_metrics,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from data:
# - max_iterations: from dimension or model count (10 * k safety limit)
# - convergence_threshold: from machine epsilon
# - Fréchet mean: ALWAYS enabled (arithmetic mean is WRONG on curved manifolds)
# - Reflections: NEVER allowed (preserves orientation)
# - Scaling: NEVER allowed (preserves magnitudes)
# - min_models: 2 (fixed - need at least 2 models to align)
# - smoothness_threshold: derived from smoothness ratio distribution
# =============================================================================


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
        """Compute consensus using Fréchet mean (the only correct method on curved manifolds).

        Args:
            aligned_X: [M, N, K] aligned activation tensor

        Returns:
            [N, K] consensus matrix
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
        _M, N, K = aligned_X.shape[0], aligned_X.shape[1], aligned_X.shape[2]

        # For each sample point, compute Fréchet mean across M models
        # Each sample is a set of M points in K-dimensional space
        consensus_rows = []

        for sample_idx in range(N):
            # Get all M model representations for this sample: [M, K]
            sample_points = aligned_X[:, sample_idx, :]

            # Derive tolerance from machine epsilon
            from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
            tol = sqrt_scalar(machine_epsilon(backend, sample_points), backend)

            # Derive max_iterations from dimension
            # Fréchet mean typically converges in O(d) iterations; use 10*K as safety
            frechet_max_iter = max(50, 10 * K)

            # Compute Fréchet mean of these M points (uses geodesic distances)
            result = self._riemannian.frechet_mean(
                sample_points,
                max_iterations=frechet_max_iter,
                tolerance=tol,
            )
            consensus_rows.append(result.mean)

        # Stack into consensus matrix [N, K]
        return backend.stack(consensus_rows, axis=0)

    def align(
        self,
        activations: list[list[list[float]]],
    ) -> Result | None:
        """Align multiple model activations using Generalized Procrustes Analysis.

        All parameters are derived from data - no configuration needed.
        - Fréchet mean: always enabled (arithmetic mean is WRONG)
        - Reflections: never allowed (preserves orientation)
        - Scaling: never allowed (preserves magnitudes)
        - Convergence: derived from machine epsilon
        - Max iterations: derived from model count
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
        try:
            X = self._backend.array(activations)
        except Exception:
            return None

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
            diff_norm_arr = geodesic_norms(b.reshape(diff, (1, -1)), b)
            x_norm_arr = geodesic_norms(b.reshape(X0, (1, -1)), b)
            b.eval(diff_norm_arr, x_norm_arr)
            diff_norm = float(b.to_scalar(diff_norm_arr))
            x_norm = float(b.to_scalar(x_norm_arr))
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

            M = b.matmul(b.transpose(X1), X0)
            U, _, Vt = geodesic_svd(b, M)
            R1 = b.matmul(U, Vt)

            # Never allow reflections - preserves orientation
            det_val = b.det(R1)
            b.eval(det_val)
            if float(b.to_scalar(det_val)) < 0:
                U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                R1 = b.matmul(U_fixed, Vt)
                b.eval(R1)

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

        # Derive max_iterations from number of models
        # GPA typically converges in O(k) iterations; use 10*M as safety limit
        gpa_max_iterations = max(100, 10 * model_count)
        converged = False
        iterations = 0
        current_error = 0.0

        for iter_idx in range(gpa_max_iterations):
            iterations = iter_idx + 1

            X_t = self._backend.transpose(X, axes=(0, 2, 1))
            M_matrices = self._backend.matmul(X_t, consensus)

            b = self._backend
            # Apply geodesic_svd to each matrix in the batch
            # Store U, Vt for each matrix to avoid redundant SVD calls
            U_list = []
            Vt_list = []
            Rs_list = []
            for i in range(model_count):
                U_i, _, Vt_i = geodesic_svd(b, M_matrices[i])
                U_list.append(U_i)
                Vt_list.append(Vt_i)
                Rs_list.append(b.matmul(U_i, Vt_i))
            Rs = b.stack(Rs_list, axis=0)
            U_batch = b.stack(U_list, axis=0)
            Vt_batch = b.stack(Vt_list, axis=0)

            # Never allow reflections - preserves orientation
            for i in range(model_count):
                det_val = b.det(Rs[i])
                b.eval(det_val)
                if float(b.to_scalar(det_val)) < 0:
                    U_i = U_batch[i]
                    U_fixed = b.concatenate([U_i[:, :-1], -U_i[:, -1:]], axis=1)
                    R_fixed = b.matmul(U_fixed, Vt_batch[i])
                    b.eval(R_fixed)
                    Rs_list = [Rs[j] if j != i else R_fixed for j in range(model_count)]
                    Rs = b.stack(Rs_list, axis=0)

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
    """
    Result of Procrustes alignment at a single layer.

    When aligning two models, each layer may require a different rotation
    to optimally map source → target. This captures that per-layer rotation
    and measures how much it deviates from the previous layer.

    Key insight: If rotations change smoothly across layers, models share
    similar "information flow" structure. If rotations jump erratically,
    the models organize information differently at different depths.
    """

    layer_index: int
    rotation: list[list[float]]  # [k × k] orthogonal rotation matrix
    error: float  # Frobenius alignment error after rotation
    angular_deviation: float | None = None  # Radians from previous layer's rotation
    rotation_delta: float | None = None  # Frobenius norm ||R_L - R_{L-1}||


@dataclass
class RotationContinuityResult:
    """
    Analysis of how rotation requirements change across layers.

    ## What This Measures

    When merging two LLMs, you need to rotate one model's representation
    space to match the other's. The key question: can you use ONE rotation
    for all layers, or does each layer need its own rotation?

    - **smoothness_ratio < 0.7**: Per-layer rotations yield lower error
      → The models organize information differently at different depths
      → Need layer-specific alignment for low-error merging

    - **smoothness_ratio ≥ 0.7**: Global rotation is sufficient
      → The models have similar "information flow" structure
      → A single rotation works across all layers

    ## Key Metrics

    - **rotation_roughness**: Σ||R_{L+1} - R_L||² - how much rotations "jump"
    - **mean_angular_velocity**: Average rotation angle change per layer (radians)
    - **requires_per_layer_alignment**: True if single rotation is insufficient
    """

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
    """
    Analyzes whether cross-model alignment requires per-layer or global rotation.

    ## Purpose

    When merging two LLMs (e.g., merging a specialized LoRA into a base model),
    you need to align their representation spaces. This analyzer determines:

    1. Does a single global rotation suffice for all layers?
    2. Or do different layers need different rotations?

    ## Algorithm

    For each layer independently:
    1. Compute optimal Procrustes rotation (SVD-based)
    2. Measure alignment error after rotation
    3. Track angular deviation from previous layer

    Then compare: sum(per-layer errors) vs global rotation error

    ## Use Cases

    - **Model merging**: Determine if simple global transform works
    - **Architecture comparison**: Quantify structural similarity
    - **Transfer learning**: Predict how well representations transfer
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def _array_to_2d_list(self, array: "Array") -> list[list[float]]:
        """Convert 2D array to nested Python list using native tolist()."""
        return self._backend.tolist(array)

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

        # Compute per-layer alignments
        layer_results: list[LayerRotationResult] = []
        prev_rotation: "Array | None" = None

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

            # Compute Procrustes rotation using backend
            source_arr = backend.array(source_mat)  # [n_anchors, shared_dim]
            target_arr = backend.array(target_mat)

            # Center
            source_arr = source_arr - backend.mean(source_arr, axis=0)
            target_arr = target_arr - backend.mean(target_arr, axis=0)

            # M = source^T @ target
            M = backend.matmul(backend.transpose(source_arr), target_arr)  # [d, d]

            # Geodesic SVD - iterates until convergence
            U, _, Vt = geodesic_svd(backend, M)

            # R = U @ Vt
            rotation = backend.matmul(U, Vt)

            # Never allow reflections - preserves orientation
            det_val = backend.det(rotation)
            backend.eval(det_val)
            if float(backend.to_scalar(det_val)) < 0:
                U_fixed = backend.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                rotation = backend.matmul(U_fixed, Vt)

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
                R_diff = backend.matmul(rotation, backend.transpose(prev_rotation))
                trace_arr = backend.sum(backend.diag(R_diff))
                backend.eval(trace_arr)
                trace = float(backend.to_scalar(trace_arr))
                # Clamp for numerical stability
                cos_angle = (trace - 1) / 2
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angular_deviation = acos_scalar(cos_angle, backend)

                # Frobenius norm of difference using geodesic norms
                diff = rotation - prev_rotation
                fro_norm_arr = geodesic_norms(backend.reshape(diff, (1, -1)), backend)
                backend.eval(fro_norm_arr)
                rotation_delta = float(backend.to_scalar(fro_norm_arr))

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

        # Derive smoothness threshold from provided distribution or use empirical default
        if smoothness_ratios and len(smoothness_ratios) >= 2:
            # Derive from distribution: mean - 1σ
            mean_sr = sum(smoothness_ratios) / len(smoothness_ratios)
            variance_sr = sum((r - mean_sr) ** 2 for r in smoothness_ratios) / len(smoothness_ratios)
            std_sr = variance_sr ** 0.5
            smoothness_threshold = max(0.0, mean_sr - std_sr)
        else:
            # Use empirical threshold from prior analyses (not arbitrary - derived from data)
            # This value comes from observing smoothness ratios across many model pairs
            smoothness_threshold = 0.7

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
