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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class FrechetMeanConfig:
    """Configuration for Fréchet mean computation.

    On curved manifolds, the arithmetic mean doesn't minimize the sum of squared
    geodesic distances. The Fréchet mean (Riemannian center of mass) is the
    proper generalization that respects manifold geometry.

    Attributes:
        enabled: Whether to use Fréchet mean instead of arithmetic mean.
        k_neighbors: Number of neighbors for geodesic distance estimation.
        max_iterations: Maximum Fréchet mean iterations.
        tolerance: Convergence tolerance for Fréchet mean.
    """

    enabled: bool = False
    k_neighbors: int = 10
    max_iterations: int = 50
    tolerance: float = 1e-5


@dataclass(frozen=True)
class Config:
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    allow_reflections: bool = False
    min_models: int = 2
    allow_scaling: bool = False
    frechet_mean: FrechetMeanConfig | None = None

    @staticmethod
    def default() -> "Config":
        return Config()

    @staticmethod
    def with_frechet_mean() -> "Config":
        """Create config that uses Fréchet mean for curvature-aware consensus."""
        return Config(frechet_mean=FrechetMeanConfig(enabled=True))


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

    Supports both arithmetic mean (Euclidean) and Fréchet mean (Riemannian)
    for consensus computation. Use Fréchet mean for curved embedding spaces.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._riemannian = None  # Lazy init for Fréchet mean

    def _compute_consensus(
        self,
        aligned_X: "Array",
        config: Config,
    ) -> "Array":
        """Compute consensus using arithmetic or Fréchet mean.

        Args:
            aligned_X: [M, N, K] aligned activation tensor
            config: GPA configuration

        Returns:
            [N, K] consensus matrix
        """
        if config.frechet_mean is None or not config.frechet_mean.enabled:
            # Standard arithmetic mean
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
        M, N, K = aligned_X.shape[0], aligned_X.shape[1], aligned_X.shape[2]

        # For each sample point, compute Fréchet mean across M models
        # Each sample is a set of M points in K-dimensional space
        consensus_rows = []

        for sample_idx in range(N):
            # Get all M model representations for this sample: [M, K]
            sample_points = aligned_X[:, sample_idx, :]

            # Compute Fréchet mean of these M points (uses geodesic distances)
            result = self._riemannian.frechet_mean(
                sample_points,
                max_iterations=config.frechet_mean.max_iterations,
                tolerance=config.frechet_mean.tolerance,
            )
            consensus_rows.append(result.mean)

        # Stack into consensus matrix [N, K]
        return backend.stack(consensus_rows, axis=0)

    def align(
        self,
        activations: list[list[list[float]]],
        config: Config = Config(),
    ) -> Result | None:
        model_count = len(activations)
        if model_count < config.min_models:
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

        # 1. Centering
        means = self._backend.mean(X, axis=1, keepdims=True)
        X = X - means

        # 2. Scaling (Optional)
        scales = self._backend.ones((model_count,))
        if config.allow_scaling:
            norms = self._backend.sqrt(self._backend.sum(X**2, axis=(1, 2)))
            ones_arr = self._backend.ones((1,))
            scale_factors = self._backend.where(norms > 1e-12, 1.0 / norms, ones_arr)
            X = X * scale_factors[:, None, None]
            scales = norms

        # Initialize Rotations (Identity)
        base_eye = self._backend.eye(k)
        Rs = self._backend.stack([base_eye] * model_count)  # [M, K, K]

        # Initial Consensus (use arithmetic mean for first iteration)
        consensus = self._backend.mean(X, axis=0)  # [N, K]
        # Note: Initial consensus uses arithmetic mean; iterative updates
        # will use Fréchet mean if configured

        aligned_X = X  # Initially aligned is just centered X

        prev_error = float("inf")
        converged = False
        iterations = 0
        current_error = 0.0

        for iter_idx in range(config.max_iterations):
            iterations = iter_idx + 1

            X_t = self._backend.transpose(X, axes=(0, 2, 1))
            M_matrices = self._backend.matmul(X_t, consensus)

            b = self._backend
            U_batch, _, Vt_batch = b.svd(M_matrices)
            Rs = b.matmul(U_batch, Vt_batch)

            if not config.allow_reflections:
                for i in range(model_count):
                    det_val = b.det(Rs[i])
                    b.eval(det_val)
                    if float(b.to_numpy(det_val).item()) < 0:
                        U_i = U_batch[i]
                        U_fixed = b.concatenate([U_i[:, :-1], -U_i[:, -1:]], axis=1)
                        R_fixed = b.matmul(U_fixed, Vt_batch[i])
                        b.eval(R_fixed)
                        Rs_list = [Rs[j] if j != i else R_fixed for j in range(model_count)]
                        Rs = b.stack(Rs_list, axis=0)

            # Update Aligned X
            aligned_X = self._backend.matmul(X, Rs)

            # New Consensus (uses Fréchet mean if configured for curvature-awareness)
            new_consensus = self._compute_consensus(aligned_X, config)

            # Error
            diffs = aligned_X - new_consensus
            current_error = float(self._backend.to_numpy(self._backend.sum(diffs**2)))

            rel_change = abs(prev_error - current_error) / max(prev_error, 1e-12)
            if rel_change < config.convergence_threshold:
                converged = True
                consensus = new_consensus
                break

            prev_error = current_error
            consensus = new_consensus

        # Final outputs
        residuals = aligned_X - consensus
        per_model_errors = self._backend.sum(residuals**2, axis=(1, 2))

        # Variance calc
        total_var = float(self._backend.to_numpy(self._backend.sum(aligned_X**2)))
        residual_var = current_error
        ratio = 1.0 - (residual_var / total_var) if total_var > 1e-12 else 0.0

        return Result(
            consensus=self._backend.to_numpy(consensus).tolist(),
            rotations=self._backend.to_numpy(Rs).tolist(),
            scales=self._backend.to_numpy(scales).tolist(),
            residuals=self._backend.to_numpy(residuals).tolist(),
            converged=converged,
            iterations=iterations,
            alignment_error=current_error,
            per_model_errors=self._backend.to_numpy(per_model_errors).tolist(),
            consensus_variance_ratio=ratio,
            sample_count=n,
            dimension=k,
            model_count=model_count,
        )

    def align_crms(
        self,
        crms: list[ConceptResponseMatrix],
        layer: int,
        config: Config = Config(),
    ) -> Result | None:
        extracted: list[list[list[float]]] = []
        min_dim = None
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
            extracted.append(mat)

        if min_dim is None or min_dim <= 0:
            return None

        # Truncate to the shared minimum dimension to align overlapping subspaces.
        trimmed = [[vec[:min_dim] for vec in mat] for mat in extracted]

        return self.align(trimmed, config)


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

    - **smoothness_ratio < 0.7**: Per-layer rotations are significantly better
      → The models organize information differently at different depths
      → Need layer-specific alignment for good merging

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

    def compute_per_layer_alignments(
        self,
        source_activations: dict[int, dict[str, list[float]]],  # layer -> anchor -> activation
        target_activations: dict[int, dict[str, list[float]]],
        source_model: str,
        target_model: str,
        config: Config = Config(),
    ) -> RotationContinuityResult | None:
        """
        Analyze rotation continuity across layers.

        Args:
            source_activations: Source model activations [layer: [anchor: activation]].
            target_activations: Target model activations [layer: [anchor: activation]].
            source_model: Source model identifier.
            target_model: Target model identifier.
            config: GPA configuration.

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

            # SVD
            U, _, Vt = backend.svd(M)

            # R = U @ Vt
            rotation = backend.matmul(U, Vt)

            # Fix reflection if needed
            if not config.allow_reflections:
                det_val = backend.det(rotation)
                backend.eval(det_val)
                if float(backend.to_numpy(det_val).item()) < 0:
                    U_fixed = backend.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                    rotation = backend.matmul(U_fixed, Vt)

            # Compute error
            aligned_source = backend.matmul(source_arr, rotation)
            error_arr = backend.sum((aligned_source - target_arr) ** 2)
            backend.eval(error_arr)
            error = float(backend.to_numpy(error_arr))

            # Compute angular deviation from previous layer
            angular_deviation = None
            rotation_delta = None
            if prev_rotation is not None:
                R_diff = backend.matmul(rotation, backend.transpose(prev_rotation))
                trace_arr = backend.sum(backend.diag(R_diff))
                backend.eval(trace_arr)
                trace = float(backend.to_numpy(trace_arr).item())
                # Clamp for numerical stability
                cos_angle = (trace - 1) / 2
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angular_deviation = math.acos(cos_angle)

                # Frobenius norm of difference
                diff = rotation - prev_rotation
                fro_norm_arr = backend.sqrt(backend.sum(diff * diff))
                backend.eval(fro_norm_arr)
                rotation_delta = float(backend.to_numpy(fro_norm_arr))

            prev_rotation = rotation

            # Convert rotation to list for result
            backend.eval(rotation)
            rotation_list = backend.to_numpy(rotation).tolist()

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
        U_g, _, Vt_g = backend.svd(M_global)
        global_rotation = backend.matmul(U_g, Vt_g)

        if not config.allow_reflections:
            det_val = backend.det(global_rotation)
            backend.eval(det_val)
            if float(backend.to_numpy(det_val).item()) < 0:
                U_g_fixed = backend.concatenate([U_g[:, :-1], -U_g[:, -1:]], axis=1)
                global_rotation = backend.matmul(U_g_fixed, Vt_g)

        aligned_global = backend.matmul(global_source, global_rotation)
        global_error_arr = backend.sum((aligned_global - global_target) ** 2)
        backend.eval(global_error_arr)
        global_error = float(backend.to_numpy(global_error_arr))

        # Compute metrics
        mean_layer_error = sum(layer_r.error for layer_r in layer_results) / len(layer_results)
        smoothness_ratio = mean_layer_error / max(global_error, 1e-12)

        # Rotation roughness
        rotation_roughness = sum(
            layer_r.rotation_delta**2 for layer_r in layer_results if layer_r.rotation_delta is not None
        )

        # Mean angular velocity
        angular_devs = [
            layer_r.angular_deviation for layer_r in layer_results if layer_r.angular_deviation is not None
        ]
        mean_angular_velocity = sum(angular_devs) / max(len(angular_devs), 1)

        # Requires per-layer alignment if smoothness_ratio < 0.7
        requires_per_layer = smoothness_ratio < 0.7

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
