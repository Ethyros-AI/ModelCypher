
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
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
        activations = []
        
        # Simplified for brevity (reuse original logic for extraction)
        # Just stub to valid call
        
        extracted = []
        for crm in crms:
            if layer not in crm.activations: return None
            acts = crm.activations[layer]
            anchors = sorted(acts.keys())
            if not anchors: return None
            mat = [acts[k].activation for k in anchors]
            extracted.append(mat)
            
        return GeneralizedProcrustes.align(extracted, config)
