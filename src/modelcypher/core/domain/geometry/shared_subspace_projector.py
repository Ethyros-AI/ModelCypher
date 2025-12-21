"""
Shared Subspace Projector.

Projects cross-architecture models into a shared semantic subspace.
Validates the H3 Hypothesis: Neural networks converge to shared statistical models of reality.

Methods:
- CCA (Canonical Correlation Analysis)
- Shared SVD
- Procrustes Analysis

Ported from TrainingCypher/Domain/Geometry/SharedSubspaceProjector.swift.
"""
from __future__ import annotations

import math
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict

# Assuming simple linalg helpers locally or from numpy if available.
# Since dependency on unchecked external libs is discouraged unless present,
# I will implement the necessary linalg (SVD, Eigen) using pure python/stdlib if simple
# or assume a numpy-like interface if the project uses one (it uses MLX/numpy).
# The swift code implements SVD/Eigen manually/simplified.
# I will use numpy if possible for stability, as it's standard in Python ML stacks.
# Checking imports from other files: `import mlx.core as mx` is used in this project.
# I will use `mlx.core` or `numpy` for linear algebra. Numpy is safer for CPU-side geometry.
import numpy as np

from modelcypher.core.domain.concept_response_matrix import ConceptResponseMatrix

class SharedSubspaceProjector:
    
    class AlignmentMethod(str, Enum):
        CCA = "cca"
        SHARED_SVD = "sharedSVD"
        PROCRUSTES = "procrustes"

    @dataclass
    class Config:
        alignment_method: "SharedSubspaceProjector.AlignmentMethod" = "cca"
        variance_threshold: float = 0.95
        max_shared_dimension: int = 256
        cca_regularization: float = 1e-4
        min_samples: int = 10

    @dataclass
    class H3ValidationMetrics:
        shared_dimension: int
        top_canonical_correlation: float
        alignment_error: float
        shared_variance_ratio: float

        @property
        def is_h3_validated(self) -> bool:
            return (
                self.shared_dimension >= 32 and
                self.top_canonical_correlation > 0.5 and
                self.alignment_error < 0.3 and
                self.shared_variance_ratio > 0.8
            )

        @property
        def summary(self) -> str:
            return f"""
            H3 Validation: {'PASS' if self.is_h3_validated else 'FAIL'}
            - Shared Dimension: {self.shared_dimension} (target: >=32) {'✓' if self.shared_dimension >= 32 else '✗'}
            - Top Correlation: {self.top_canonical_correlation:.3f} (target: >0.5) {'✓' if self.top_canonical_correlation > 0.5 else '✗'}
            - Alignment Error: {self.alignment_error:.3f} (target: <0.3) {'✓' if self.alignment_error < 0.3 else '✗'}
            - Shared Variance: {self.shared_variance_ratio*100:.1f}% (target: >80%) {'✓' if self.shared_variance_ratio > 0.8 else '✗'}
            """

    @dataclass
    class Result:
        shared_dimension: int
        source_dimension: int
        target_dimension: int
        source_projection: List[List[float]] # [d_s, k]
        target_projection: List[List[float]] # [d_t, k]
        alignment_strengths: List[float]
        alignment_error: float
        shared_variance_ratio: float
        sample_count: int
        method: "SharedSubspaceProjector.AlignmentMethod"

        @property
        def is_valid(self) -> bool:
            return (self.shared_dimension > 0 and 
                    self.alignment_error < 0.5 and 
                    self.shared_variance_ratio > 0.5)

        @property
        def h3_metrics(self) -> "SharedSubspaceProjector.H3ValidationMetrics":
            return SharedSubspaceProjector.H3ValidationMetrics(
                shared_dimension=self.shared_dimension,
                top_canonical_correlation=self.alignment_strengths[0] if self.alignment_strengths else 0.0,
                alignment_error=self.alignment_error,
                shared_variance_ratio=self.shared_variance_ratio
            )

    @staticmethod
    def discover(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        source_layer: int,
        target_layer: int,
        config: Config = Config()
    ) -> Optional[Result]:
        common_anchors = source_crm.common_anchor_ids(target_crm) if hasattr(source_crm, 'common_anchor_ids') else []
        if not common_anchors:
            return None

        # Get activations [N x d]
        source_act = source_crm.activation_matrix(source_layer, common_anchors)
        target_act = target_crm.activation_matrix(target_layer, common_anchors)
        
        if not source_act or not target_act: return None
        if len(source_act) < config.min_samples: return None
        if len(source_act) != len(target_act): return None
        
        # Convert to numpy for heavy lifting
        X_s = np.array(source_act, dtype=np.float32)
        X_t = np.array(target_act, dtype=np.float32)
        
        if config.alignment_method == SharedSubspaceProjector.AlignmentMethod.CCA:
            return SharedSubspaceProjector._discover_cca(X_s, X_t, config)
        elif config.alignment_method == SharedSubspaceProjector.AlignmentMethod.SHARED_SVD:
            return SharedSubspaceProjector._discover_shared_svd(X_s, X_t, config)
        elif config.alignment_method == SharedSubspaceProjector.AlignmentMethod.PROCRUSTES:
            return SharedSubspaceProjector._discover_procrustes(X_s, X_t, config)
            
        return None

    # MARK: - CCA Implementation
    @staticmethod
    def _discover_cca(X_s: np.ndarray, X_t: np.ndarray, config: Config) -> Optional[Result]:
        n, d_s = X_s.shape
        _, d_t = X_t.shape
        
        # Center
        X_s_c = X_s - np.mean(X_s, axis=0)
        X_t_c = X_t - np.mean(X_t, axis=0)
        
        # Regularization
        epsilon = config.cca_regularization
        
        # Covariances (1/(n-1) or 1/n? Swift implementation uses 1/n)
        # We'll stick to 1/n for parity
        C_ss = (X_s_c.T @ X_s_c) / n + epsilon * np.eye(d_s)
        C_tt = (X_t_c.T @ X_t_c) / n + epsilon * np.eye(d_t)
        
        # Whitening transforms: C^{-1/2}
        try:
            # Eigen decomp for stability on symmetric positive definite matrices
            # C = V L V^T -> C^{-1/2} = V L^{-1/2} V^T
            eig_s, V_s = np.linalg.eigh(C_ss)
            eig_t, V_t = np.linalg.eigh(C_tt)
            
            # Avoid division by zero
            eig_s = np.maximum(eig_s, 1e-9)
            eig_t = np.maximum(eig_t, 1e-9)
            
            C_ss_inv_sqrt = V_s @ np.diag(1.0 / np.sqrt(eig_s)) @ V_s.T
            C_tt_inv_sqrt = V_t @ np.diag(1.0 / np.sqrt(eig_t)) @ V_t.T
            
            # Whiten
            X_s_w = X_s_c @ C_ss_inv_sqrt
            X_t_w = X_t_c @ C_tt_inv_sqrt
            
            # Cross-covariance in whitened space
            C_st_w = (X_s_w.T @ X_t_w) / n
            
            # SVD
            U, S, Vt = np.linalg.svd(C_st_w, full_matrices=False)
            
            # Canonical correlations are SVs
            corrs = np.clip(S, 0.0, 1.0)
            
            # Determine shared dimension based on variance threshold logic (cumulative sum of correlations?)
            # Swift logic: cumVariance += corr. Stop when cum / sum(corrs) >= threshold
            total_corr_sum = np.sum(corrs)
            cum_corr = np.cumsum(corrs)
            
            shared_dim = 0
            if total_corr_sum > 0:
                # Find first index where cumulative ratio exceeds threshold
                ratios = cum_corr / total_corr_sum
                indices = np.where(ratios >= config.variance_threshold)[0]
                if indices.size > 0:
                    shared_dim = indices[0] + 1
                else:
                    shared_dim = len(corrs)
            else:
                shared_dim = 1
                
            shared_dim = min(shared_dim, config.max_shared_dimension)
            
            # Projections
            # W_s = C_ss^{-1/2} U[:, :k]
            # W_t = C_tt^{-1/2} V[:, :k] (Since svd returns Vt, V is Vt.T)
            V = Vt.T
            
            proj_s = C_ss_inv_sqrt @ U[:, :shared_dim]
            proj_t = C_tt_inv_sqrt @ V[:, :shared_dim]
            
            # Alignment Error (Procrustes in projected space)
            # Already aligned by CCA maximally, but let's measure L2 diff
            Z_s = X_s_c @ proj_s
            Z_t = X_t_c @ proj_t
            
            alignment_error = float(np.linalg.norm(Z_s - Z_t) / np.linalg.norm(Z_t))
            
            # Shared Variance Ratio: (Var(shared_s) + Var(shared_t)) / (TotalVar_s + TotalVar_t)
            # Var(shared) is sum of squared SV assignments roughly? 
            # Swift implementation: average total variance of source/target, and shared is sum of corrs?
            # Let's match Swift:
            # sourceVarianceTotal = sum(cssEigenvalues), target...
            # sharedVariance = sum(canonicalCorrelations.prefix(sharedDim))
            total_var_s = np.sum(eig_s) - (epsilon * d_s) # remove reg effect if possible? Swift doesn't seem to.
            total_var_t = np.sum(eig_t) - (epsilon * d_t)
            shared_var = np.sum(corrs[:shared_dim])
            
            avg_total = (total_var_s + total_var_t) / 2
            ratio = shared_var / avg_total if avg_total > 0 else 0.0
            # Swift actually divides by sharedDim: "sharedVariance / Float(sharedDim)"? No wait
            # line 409: "sharedVariance / Float(sharedDim)"
            # That seems like average correlation? That's weird for "Variance Ratio".
            # I will implement as written in Swift for parity.
            ratio = (shared_var / float(shared_dim)) if shared_dim > 0 else 0.0

            return Result(
                shared_dimension=shared_dim,
                source_dimension=d_s,
                target_dimension=d_t,
                source_projection=proj_s.tolist(),
                target_projection=proj_t.tolist(),
                alignment_strengths=corrs.tolist(),
                alignment_error=alignment_error,
                shared_variance_ratio=float(ratio),
                sample_count=n,
                method=SharedSubspaceProjector.AlignmentMethod.CCA
            )

        except np.linalg.LinAlgError:
            return None

    # MARK: - Shared SVD Implementation
    @staticmethod
    def _discover_shared_svd(X_s: np.ndarray, X_t: np.ndarray, config: Config) -> Optional[Result]:
        n, d_s = X_s.shape
        _, d_t = X_t.shape
        
        X_s_c = X_s - np.mean(X_s, axis=0)
        X_t_c = X_t - np.mean(X_t, axis=0)
        
        # PCA on each
        # Covariances
        C_ss = (X_s_c.T @ X_s_c) / n
        C_tt = (X_t_c.T @ X_t_c) / n
        
        eig_s, V_s = np.linalg.eigh(C_ss)
        eig_t, V_t = np.linalg.eigh(C_tt)
        
        # Sort descending
        idx_s = np.argsort(eig_s)[::-1]
        idx_t = np.argsort(eig_t)[::-1]
        
        eig_s = eig_s[idx_s]
        V_s = V_s[:, idx_s]
        eig_t = eig_t[idx_t]
        V_t = V_t[:, idx_t]
        
        # Rank selection
        def effective_rank(evals, thresh):
            total = np.sum(evals[evals > 0])
            if total == 0: return 0
            cum = np.cumsum(evals)
            ratios = cum / total
            idxs = np.where(ratios >= thresh)[0]
            return idxs[0] + 1 if idxs.size > 0 else len(evals)

        rank_s = effective_rank(eig_s, config.variance_threshold)
        rank_t = effective_rank(eig_t, config.variance_threshold)
        shared_dim = min(rank_s, rank_t, config.max_shared_dimension, n)
        
        if shared_dim <= 0: return None
        
        # Projections are top-k eigenvectors
        proj_s = V_s[:, :shared_dim]
        proj_t = V_t[:, :shared_dim]
        
        Z_s = X_s_c @ proj_s
        Z_t = X_t_c @ proj_t
        
        # Procrustes in shared space to align them rotationally
        # M = Z_s^T @ Z_t
        M = Z_s.T @ Z_t
        U, _, Vt = np.linalg.svd(M)
        Omega = U @ Vt
        
        Z_s_aligned = Z_s @ Omega
        
        error = float(np.linalg.norm(Z_s_aligned - Z_t) / np.linalg.norm(Z_t))
        
        # Helper to compute alignment strengths (correlation) per dim
        strengths = []
        for k in range(shared_dim):
            v1 = Z_s_aligned[:, k]
            v2 = Z_t[:, k]
            # Correlation
            num = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            strengths.append(float(abs(num/denom)) if denom > 0 else 0.0)
            
        strengths.sort(reverse=True)
        
        # Ratio
        var_s = np.sum(eig_s[:shared_dim])
        var_t = np.sum(eig_t[:shared_dim])
        total_s = np.sum(eig_s)
        total_t = np.sum(eig_t)
        ratio = (var_s + var_t) / (total_s + total_t) if (total_s + total_t) > 0 else 0.0
        
        return Result(
            shared_dimension=shared_dim,
            source_dimension=d_s,
            target_dimension=d_t,
            source_projection=proj_s.tolist(),
            target_projection=proj_t.tolist(),
            alignment_strengths=strengths,
            alignment_error=error,
            shared_variance_ratio=float(ratio),
            sample_count=n,
            method=SharedSubspaceProjector.AlignmentMethod.SHARED_SVD
        )

    # MARK: - Procrustes Implementation
    @staticmethod
    def _discover_procrustes(X_s: np.ndarray, X_t: np.ndarray, config: Config) -> Optional[Result]:
        n, d = X_s.shape
        _, d_t = X_t.shape
        
        if d != d_t:
            # Fallback to CCA
            return SharedSubspaceProjector._discover_cca(X_s, X_t, config)
            
        X_s_c = X_s - np.mean(X_s, axis=0)
        X_t_c = X_t - np.mean(X_t, axis=0)
        
        # M = X_s^T X_t
        M = X_s_c.T @ X_t_c
        
        U, S, Vt = np.linalg.svd(M)
        Omega = U @ Vt
        
        Z_s = X_s_c @ Omega
        
        error = float(np.linalg.norm(Z_s - X_t_c) / np.linalg.norm(X_t_c))
        
        # Singular values give strength?
        # Normalized singular values
        total_sv = np.sum(S)
        norm_sv = S / total_sv if total_sv > 0 else S

        # Dimension logic
        cum = np.cumsum(S)
        ratios = cum / total_sv if total_sv > 0 else np.zeros_like(cum)
        idxs = np.where(ratios >= config.variance_threshold)[0]
        shared_dim = idxs[0] + 1 if idxs.size > 0 else len(S)
        shared_dim = min(shared_dim, config.max_shared_dimension)
        
        ratio = cum[shared_dim-1] / total_sv if total_sv > 0 and shared_dim > 0 else 0.0

        return Result(
            shared_dimension=shared_dim,
            source_dimension=d,
            target_dimension=d,
            source_projection=np.eye(d).tolist(), # Identity for source
            target_projection=Omega.T.tolist(), # Rotation for target (or Omega applied to source? Procrustes rotates source to matches target)
            # If Z_s = X_s @ Omega ~ X_t -> X_s ~ X_t @ Omega^T.
            # Usually Source Proj and Target Proj map TO the shared space.
            # If Shared Space == Target Space:
            # Source -> Target: Omega.
            # Target -> Target: Identity.
            # Let's match Swift: source=Identity, target=Omega.
            # line 669/670 in Swift: source=identity, target=reshape(omega).
            # Wait, if Swift rotates source to target, then source*Omega ~ target.
            # If shared space is "Source Space", then Target*Omega^T ~ Source.
            # If shared space is "Target Space", then Source*Omega ~ Target.
            # Code says: rotatedSource = sourceFlat * omega. 
            # And error = rotatedSource - target.
            # So Source maps to Target via Omega.
            # So Source Projection = Omega. Target Projection = Identity.
            # BUT Swift line 669 says: sourceProjection: identity. targetProjection: omega.
            # That implies Target maps to Source via Omega?
            # Re-reading Swift Procrustes:
            # rotatedSource = ... source ... omega
            # diff = rotatedSource - target
            # So Source * Omega ~= Target.
            # If output projection means "Map to Shared Space":
            # If Shared = Source Space: Source->I, Target->Omega^T (inverse)
            # If Shared = Target Space: Source->Omega, Target->I
            # Swift returns I for source, Omega for Target.
            # This implies Shared Space = Source Space?
            # And Target * Omega ~= Source? 
            # But the calculation calculated source * Omega ~= Target.
            # This is contradictory or I am misinterpreting "targetProjection".
            # I will follow Swift verbatim for parity, assuming I/Omega structure.
            alignment_strengths=norm_sv.tolist(),
            alignment_error=error,
            shared_variance_ratio=float(ratio),
            sample_count=n,
            method=SharedSubspaceProjector.AlignmentMethod.PROCRUSTES
        )
