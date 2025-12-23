
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import mlx.core as mx
import numpy as np
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix

@dataclass(frozen=True)
class Config:
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    allow_reflections: bool = False
    min_models: int = 2
    allow_scaling: bool = False

    @staticmethod
    def default() -> "Config":
        return Config()

@dataclass(frozen=True)
class Result:
    consensus: List[List[float]] # Kept as list for compatibility, could be mx.array in future
    rotations: List[List[List[float]]]
    scales: List[float]
    residuals: List[List[List[float]]]
    converged: bool
    iterations: int
    alignment_error: float
    per_model_errors: List[float]
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
    """
    Generalized Procrustes Analysis using MLX for acceleration.
    """
    
    @staticmethod
    def align(
        activations: List[List[List[float]]],
        config: Config = Config(),
    ) -> Optional[Result]:
        model_count = len(activations)
        if model_count < config.min_models:
            return None
        
        # Convert all to MLX arrays [M, N, K]
        # M models, N samples, K dims
        
        # Verify dims
        n = len(activations[0])
        if n == 0: return None
        k = len(activations[0][0])
        if k == 0: return None
        
        # Check all match
        for act in activations:
            if len(act) != n or len(act[0]) != k:
                return None
        
        # Build tensor stack
        # (Model, Sample, Dim)
        try:
            # mx.array constructor handles nested lists efficiently
            X = mx.array(activations) 
        except Exception:
            return None
            
        # 1. Centering
        # Center each model's configuration: mean over samples (axis 1) should be 0 vector
        means = mx.mean(X, axis=1, keepdims=True)
        X = X - means
        
        # 2. Scaling (Optional)
        scales = mx.ones((model_count,))
        if config.allow_scaling:
            # Frobenius norm of each model config
            norms = mx.sqrt(mx.sum(X**2, axis=(1, 2)))
            # If norm > 0, scale = 1/norm
            scale_factors = mx.where(norms > 1e-12, 1.0 / norms, mx.array(1.0))
            X = X * scale_factors[:, None, None]
            scales = norms
        
        # Initialize Rotations (Identity)
        # Stack k*k identity M times
        base_eye = mx.eye(k)
        Rs = mx.stack([base_eye] * model_count) # [M, K, K]
        
        # Initial Consensus
        consensus = mx.mean(X, axis=0) # [N, K]
        
        aligned_X = X # Initially aligned is just centered X
        
        prev_error = float("inf")
        converged = False
        iterations = 0
        
        for iter_idx in range(config.max_iterations):
            iterations = iter_idx + 1
            
            # For each model, align to consensus
            
            # Target = Consensus [N, K]
            # Source = X[i] [N, K]
            # M = Source^T @ Target [K, K]
        
            # Batch Matmul:
            # X_t = X.transpose(0, 2, 1) # [M, K, N]
            # M_matrices = X_t @ consensus # [M, K, K]
            
            X_t = X.transpose(0, 2, 1) # [M, K, N]
            M_matrices = X_t @ consensus
            
            # SVD (CPU-only usually in MLX)
            with mx.stream(mx.cpu):
                U, _, Vt = mx.linalg.svd(M_matrices) # U: [M, K, K], Vt: [M, K, K]
            
            # R = U @ Vt
            rotation_updates = U @ Vt # [M, K, K]
            
            if not config.allow_reflections:
                # Use numpy for determinant since MLX lacks it and SVD is on CPU anyway
                # R = U @ Vt
                # We can check det of R directly
                # Convert to numpy
                rs_np = np.array(rotation_updates) # [M, K, K]
                dets_np = np.linalg.det(rs_np)
                neg_det_mask_np = dets_np < 0
                
                # Use a loop for reflection handling since model count is small
                for i in range(model_count):
                    if neg_det_mask_np[i]:
                        # Reflection detected and not allowed.
                        # Fix R directly: R_new[i] = U[i] @ F @ Vt[i]
                        # where F has -1 at (k-1, k-1)
                        
                        u_matrix = U[i]
                        vt_matrix = Vt[i]
                        
                        # Construct F as diagonal
                        f_diag = mx.ones((k,))
                        f_diag[-1] = -1
                        f_matrix = mx.diag(f_diag)
                        
                        # Recompute R for this model
                        rotation_updates[i] = u_matrix @ f_matrix @ vt_matrix 

            Rs = rotation_updates
            
            # Update Aligned X
            aligned_X = X @ Rs
            
            # New Consensus
            new_consensus = mx.mean(aligned_X, axis=0)
            
            # Error
            diffs = aligned_X - new_consensus # Broadcasting (M, N, K) - (N, K)
            current_error = mx.sum(diffs**2).item()
            
            rel_change = abs(prev_error - current_error) / max(prev_error, 1e-12)
            if rel_change < config.convergence_threshold:
                converged = True
                consensus = new_consensus
                break
                
            prev_error = current_error
            consensus = new_consensus
        
        # Final outputs
        residuals_mx = aligned_X - consensus
        per_model_errors_mx = mx.sum(residuals_mx**2, axis=(1, 2))
        
        # Variance calc
        # Total variance of Aligned (sum of squares of elements)
        total_var = mx.sum(aligned_X**2).item()
        residual_var = current_error
        ratio = 1.0 - (residual_var / total_var) if total_var > 1e-12 else 0.0
        
        return Result(
            consensus=consensus.tolist(), # Convert to lists for compat
            rotations=Rs.tolist(),
            scales=scales.tolist(),
            residuals=residuals_mx.tolist(),
            converged=converged,
            iterations=iterations,
            alignment_error=current_error,
            per_model_errors=per_model_errors_mx.tolist(),
            consensus_variance_ratio=ratio,
            sample_count=n,
            dimension=k,
            model_count=model_count
        )

    @staticmethod
    def align_crms(
        crms: List[ConceptResponseMatrix],
        layer: int,
        config: Config = Config(),
    ) -> Optional[Result]:
        extracted: list[list[list[float]]] = []
        min_dim = None
        for crm in crms:
            if layer not in crm.activations: return None
            acts = crm.activations[layer]
            anchors = sorted(acts.keys())
            if not anchors: return None
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

        return GeneralizedProcrustes.align(trimmed, config)


# =============================================================================
# Per-Layer Procrustes (H5 Experiment)
# =============================================================================


@dataclass(frozen=True)
class PerLayerProcrustesResult:
    """
    Result of per-layer Procrustes alignment.
    
    Used for H5 experiment: testing whether per-layer rotations differ
    significantly from a single global rotation.
    """
    layer_index: int
    rotation: List[List[float]]  # [k × k]
    error: float
    angular_deviation: Optional[float] = None  # From previous layer
    rotation_delta: Optional[float] = None  # Frobenius norm of difference


@dataclass
class LayerProcrustesExperiment:
    """
    Complete result of H5 per-layer Procrustes experiment.
    
    Tests hypothesis H5: whether per-layer rotations differ significantly
    from a single global rotation.
    """
    source_model: str
    target_model: str
    layers: List[PerLayerProcrustesResult]
    global_rotation_error: float
    smoothness_ratio: float
    rotation_roughness: float
    mean_angular_velocity: float
    h5_null_rejected: bool
    source_dimension: int
    target_dimension: int
    anchor_count: int
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        verdict = (
            "H5-null REJECTED: Per-layer alignment significantly better"
            if self.h5_null_rejected else
            "H5-null NOT rejected: Global rotation sufficient"
        )
        mean_layer_error = (
            sum(l.error for l in self.layers) / len(self.layers)
            if self.layers else 0.0
        )
        return (
            "H5 Experiment: Per-Layer Procrustes Alignment\n"
            "=============================================\n"
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


class PerLayerProcrustes:
    """
    Computes per-layer Procrustes alignment for H5 experiment.
    
    For each layer independently, computes the optimal rotation that aligns
    source activations to target activations. Also computes a global rotation
    for comparison.
    """
    
    @staticmethod
    def compute_per_layer_alignments(
        source_activations: Dict[int, Dict[str, List[float]]],  # layer -> anchor -> activation
        target_activations: Dict[int, Dict[str, List[float]]],
        source_model: str,
        target_model: str,
        config: Config = Config(),
    ) -> Optional[LayerProcrustesExperiment]:
        """
        Compute per-layer Procrustes alignment.
        
        Args:
            source_activations: Source model activations [layer: [anchor: activation]].
            target_activations: Target model activations [layer: [anchor: activation]].
            source_model: Source model identifier.
            target_model: Target model identifier.
            config: GPA configuration.
        
        Returns:
            LayerProcrustesExperiment result, or None if alignment failed.
        """
        # Get common layers
        common_layers = sorted(
            set(source_activations.keys()) & set(target_activations.keys())
        )
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
        layer_results: List[PerLayerProcrustesResult] = []
        prev_rotation: Optional[np.ndarray] = None
        
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
            
            # Compute Procrustes rotation
            source_np = np.array(source_mat)  # [n_anchors, shared_dim]
            target_np = np.array(target_mat)
            
            # Center
            source_np = source_np - source_np.mean(axis=0)
            target_np = target_np - target_np.mean(axis=0)
            
            # M = source^T @ target
            M = source_np.T @ target_np  # [d, d]
            
            # SVD
            U, _, Vt = np.linalg.svd(M)
            
            # R = U @ Vt
            rotation = U @ Vt
            
            # Fix reflection if needed
            if not config.allow_reflections and np.linalg.det(rotation) < 0:
                U[:, -1] *= -1
                rotation = U @ Vt
            
            # Compute error
            aligned_source = source_np @ rotation
            error = float(np.sum((aligned_source - target_np) ** 2))
            
            # Compute angular deviation from previous layer
            angular_deviation = None
            rotation_delta = None
            if prev_rotation is not None:
                # Angular deviation: arccos((trace(R @ R_prev^T) - 1) / 2)
                R_diff = rotation @ prev_rotation.T
                trace = np.trace(R_diff)
                # Clamp for numerical stability
                cos_angle = (trace - 1) / 2
                cos_angle = np.clip(cos_angle, -1, 1)
                angular_deviation = float(np.arccos(cos_angle))
                
                # Frobenius norm of difference
                rotation_delta = float(np.linalg.norm(rotation - prev_rotation, 'fro'))
            
            prev_rotation = rotation
            
            layer_results.append(PerLayerProcrustesResult(
                layer_index=layer_idx,
                rotation=rotation.tolist(),
                error=error,
                angular_deviation=angular_deviation,
                rotation_delta=rotation_delta,
            ))
        
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
        
        global_source = np.array(all_source)
        global_target = np.array(all_target)
        global_source = global_source - global_source.mean(axis=0)
        global_target = global_target - global_target.mean(axis=0)
        
        M_global = global_source.T @ global_target
        U_g, _, Vt_g = np.linalg.svd(M_global)
        global_rotation = U_g @ Vt_g
        
        if not config.allow_reflections and np.linalg.det(global_rotation) < 0:
            U_g[:, -1] *= -1
            global_rotation = U_g @ Vt_g
        
        aligned_global = global_source @ global_rotation
        global_error = float(np.sum((aligned_global - global_target) ** 2))
        
        # Compute metrics
        mean_layer_error = sum(l.error for l in layer_results) / len(layer_results)
        smoothness_ratio = mean_layer_error / max(global_error, 1e-12)
        
        # Rotation roughness
        rotation_roughness = sum(
            l.rotation_delta ** 2 for l in layer_results
            if l.rotation_delta is not None
        )
        
        # Mean angular velocity
        angular_devs = [l.angular_deviation for l in layer_results if l.angular_deviation is not None]
        mean_angular_velocity = sum(angular_devs) / max(len(angular_devs), 1)
        
        # H5 null hypothesis rejected if smoothness_ratio < 0.7
        h5_null_rejected = smoothness_ratio < 0.7
        
        return LayerProcrustesExperiment(
            source_model=source_model,
            target_model=target_model,
            layers=layer_results,
            global_rotation_error=global_error,
            smoothness_ratio=smoothness_ratio,
            rotation_roughness=rotation_roughness,
            mean_angular_velocity=mean_angular_velocity,
            h5_null_rejected=h5_null_rejected,
            source_dimension=source_dim,
            target_dimension=target_dim,
            anchor_count=len(common_anchors),
        )

