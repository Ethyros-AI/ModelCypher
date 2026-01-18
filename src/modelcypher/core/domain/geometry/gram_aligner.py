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

"""Gram matrix alignment utilities with geodesic CKA diagnostics.

Computes a closed-form linear alignment between activation sets and reports
geodesic CKA using k-NN graph distances. Tolerances are derived from dtype
machine epsilon.

References:
    - Yu et al. (2025). "Relative Geodesic Representations" - NeurIPS
    - Kornblith et al. (2019). "Similarity of Neural Network Representations
      Revisited." arXiv:1905.00414
    - Tenenbaum et al. (2000). "Isomap" - geodesic distance via k-NN graph
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.alignment_diagnostic import AlignmentSignal
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_invariant_alignment,
    geodesic_pinv,
    gpu_lstsq,
    machine_epsilon,
    numerical_rank_truncated_lstsq,
    precision_dtype,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

__all__ = [
    "find_alignment",
    "AlignmentResult",
    "GramAligner",
]

# Module-level cache reference
_cache = ComputationCache.shared()


# =============================================================================
# Public API (Entry Point)
# =============================================================================


def find_alignment(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> AlignmentResult:
    """Find the closed-form linear alignment and report geodesic diagnostics.

    This is the main entry point.

    Parameters
    ----------
    source_activations : Array
        Source activations [n_samples, d_source].
    target_activations : Array
        Target activations [n_samples, d_target].
    backend : Backend, optional
        Backend for tensor operations.

    Returns
    -------
    AlignmentResult
        The linear alignment transform plus geodesic diagnostics.

    Example
    -------
    >>> result = find_alignment(source_acts, target_acts)
    >>> aligned_source = source_acts @ result.feature_transform
    >>> # Linear CKA(aligned_source, target_acts) can be evaluated separately
    """
    aligner = GramAligner(backend)
    return aligner.find_perfect_alignment(source_activations, target_activations)


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True)
class AlignmentResult:
    """Result of linear alignment with geodesic diagnostics.

    The transformation from linear Procrustes plus geodesic CKA diagnostics.
    All thresholds are derived from dtype, not hardcoded.
    """

    # Apply as: A_s' = A_s @ feature_transform [d_source, d_target]
    # Kept as GPU array to avoid CPU round-trip
    feature_transform: "Array"

    # Number of iterations taken (0 = linear alignment was sufficient)
    iterations: int

    # Numerical deviation from geodesic CKA = 1.0
    # This is 1.0 - achieved_cka. Should be < precision_threshold for full overlap.
    numerical_deviation: float

    # Dtype-derived precision threshold: sqrt(machine_epsilon)
    precision_threshold: float

    # Geodesic CKA achieved by the alignment.
    # Manifold-overlap diagnostic (k-NN graph + RBF kernel).
    achieved_cka: float = 1.0

    # Diagnostic signal describing any residual gap
    diagnostic: "AlignmentSignal | None" = None

    # Scale factor: ||target|| / ||source @ F||
    # Apply to stitched weights: W_merged = scale_ratio * W_stitched
    scale_ratio: float = 1.0

    # Linear solver telemetry (normal equations or CGLS fallback)
    linear_iterations: int = 0  # 0 = direct solve via normal equations
    linear_residual: float = 0.0

    # Gram matrix condition number (numerical stability indicator)
    # Higher = less stable. If > 1e5, alignment may be unreliable.
    gram_condition_number: float = 1.0

    # Numerical rank metrics (from truncated SVD alignment)
    # These are derived from machine precision, not heuristics
    source_numerical_rank: int = 0  # Rank of source activations (σ > σ_max × sqrt(ε))
    target_numerical_rank: int = 0  # Rank of target activations
    alignment_rank: int = 0  # min(source_rank, target_rank) - dimensions actually used

    @property
    def is_perfect(self) -> bool:
        """True if geodesic CKA is within precision threshold of 1.0."""
        return self.numerical_deviation < self.precision_threshold

    @property
    def is_numerically_exact(self) -> bool:
        """True if geodesic CKA achieved 1.0 within dtype precision.

        If False, probes may not span the full shared manifold or the models
        include novel structure outside the overlap.
        """
        return self.numerical_deviation < self.precision_threshold

    @property
    def is_converged(self) -> bool:
        """Alias for is_numerically_exact for backward compatibility."""
        return self.is_numerically_exact


# =============================================================================
# Core Implementation
# =============================================================================


class GramAligner:
    """Find linear alignment with geodesic diagnostics between activation spaces.

    Computes a closed-form linear transform and reports geodesic CKA. Optional
    geodesic-invariant alignment can be enabled when needed. All tolerances are
    derived from the input dtype's machine epsilon.

    Usage
    -----
    >>> aligner = GramAligner(backend)
    >>> result = aligner.find_perfect_alignment(source, target)
    >>> aligned = source @ result.feature_transform
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        max_iterations: int | None = None,  # IGNORED - kept for backward compat
        tolerance: float | None = None,  # IGNORED
        regularization: float | None = None,  # IGNORED
        max_steps: int = 5000,  # Kept for backward compatibility (no iterative refinement)
        fast_mode: bool = False,  # Skip CKA diagnostics for speed
        use_geodesic_alignment: bool = False,  # Optional geodesic alignment pass
    ) -> None:
        """Initialize the aligner.

        Parameters
        ----------
        backend : Backend, optional
            Backend for tensor operations.
        max_iterations : int
            IGNORED. Kept for backward compatibility only.
        tolerance : float
            IGNORED. Derived from dtype.
        regularization : float
            IGNORED. Derived from dtype.
        max_steps : int
            IGNORED. Kept for backward compatibility.
        fast_mode : bool
            If True, skip CKA diagnostics after computing F.
        use_geodesic_alignment : bool
            If True, run geodesic-invariant alignment when geodesic CKA indicates
            non-overlap. Default False to keep the closed-form linear alignment
            on the shared manifold and use geodesic CKA purely as a diagnostic.
        """
        self._backend = backend or get_default_backend()
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._regularization = regularization
        self._max_steps = max_steps
        self._fast_mode = fast_mode
        self._use_geodesic_alignment = use_geodesic_alignment

    def _identity_result(
        self, n: int, d: int, precision: float
    ) -> AlignmentResult:
        """Return identity transform result (CKA = 1.0 for identical inputs)."""
        b = self._backend
        I_feat = b.eye(d)
        I_sample = b.eye(n)
        b.eval(I_feat, I_sample)
        return AlignmentResult(
            feature_transform=I_feat,  # Keep on GPU
            # achieved_cka=1.0 for identical inputs
            iterations=0,
            numerical_deviation=0.0,  # Perfect precision for identity
            diagnostic=None,
            precision_threshold=precision,
            linear_iterations=0,
            linear_residual=0.0,
        )

    def _center(self, X: "Array") -> "Array":
        """Center activations (subtract mean)."""
        b = self._backend
        mean = b.mean(X, axis=0, keepdims=True)
        return X - mean

    def find_perfect_alignment(
        self,
        source_activations: "Array",
        target_activations: "Array",
        strict: bool = True,
        max_refinement_passes: int = 10,
        F_init: "Array | None" = None,
    ) -> AlignmentResult:
        """Find alignment transform that preserves geodesic manifold structure.

        Uses geodesic_invariant_alignment which operates in relative representation
        space (pairwise geodesic similarities via k-NN graphs). This preserves
        the intrinsic manifold geometry that Euclidean methods destroy.

        Parameters
        ----------
        source_activations : Array
            Source activations [n_samples, d_source].
        target_activations : Array
            Target activations [n_samples, d_target].
        strict : bool
            IGNORED. Kept for backward compatibility.
        max_refinement_passes : int
            IGNORED. Kept for backward compatibility.
        F_init : Array | None
            IGNORED. Kept for backward compatibility.

        Returns
        -------
        AlignmentResult
            The geodesic-preserving transformation plus diagnostics.
        """
        b = self._backend
        n_s, d_s = b.shape(source_activations)
        n_t, d_t = b.shape(target_activations)

        if n_s != n_t:
            raise ValueError(f"Sample counts must match: source={n_s}, target={n_t}")

        # Fast path: identity check (before any array copies)
        is_same_input = source_activations is target_activations

        # Promote to highest available precision for alignment stability
        source_activations = b.astype(
            source_activations, precision_dtype(b, reference=source_activations)
        )
        target_activations = b.astype(
            target_activations, precision_dtype(b, reference=target_activations)
        )

        eps = machine_epsilon(b, source_activations)
        precision = sqrt_scalar(eps, b)

        if is_same_input:
            return self._identity_result(int(n_s), int(d_s), precision)

        # =================================================================
        # NUMERICAL-RANK TRUNCATED ALIGNMENT (not heuristic)
        # =================================================================
        # Uses SVD with truncation at machine precision threshold.
        # Singular values below σ_max × sqrt(ε_machine) are numerical noise.
        # The alignment operates in k = min(rank_source, rank_target) dimensions.

        linear_start = time.perf_counter()

        # Use numerical-rank-truncated least squares instead of full pinv
        # This guarantees the condition number is bounded in the truncated space
        F_linear, source_rank, target_rank, alignment_rank, condition_number = (
            numerical_rank_truncated_lstsq(b, source_activations, target_activations)
        )
        b.eval(F_linear)

        # Measure geodesic CKA of the alignment
        F_linear, linear_iterations, linear_cka = self._geodesic_refine(
            source_activations, target_activations, F_linear
        )
        linear_elapsed = time.perf_counter() - linear_start
        logger.info(
            "TRUNCATED ALIGNMENT: geodesic CKA=%.6f, rank=%d/%d, κ=%.2e (%.2fs)",
            linear_cka,
            alignment_rank,
            int(d_s),
            condition_number,
            linear_elapsed,
        )

        # =================================================================
        # PHASE 1: Geodesic alignment (manifold-preserving)
        # =================================================================
        # Uses k-NN geodesic distances to preserve intrinsic manifold structure
        alignment_stats: dict[str, float] = {}
        F = F_linear
        iterations = linear_iterations
        geodesic_cka = linear_cka

        if self._use_geodesic_alignment and geodesic_cka < (1.0 - precision):
            start_time = time.perf_counter()
            F_geo = geodesic_invariant_alignment(
                b, source_activations, target_activations, stats=alignment_stats
            )
            b.eval(F_geo)

            alignment_elapsed = time.perf_counter() - start_time
            logger.info(
                "GEODESIC ALIGNMENT: manifold-preserving transform (%.2fs)",
                alignment_elapsed,
            )

            # =================================================================
            # PHASE 2: Geodesic diagnostics
            # =================================================================
            refine_start = time.perf_counter()
            F_geo, geo_iterations, geodesic_cka_geo = self._geodesic_refine(
                source_activations, target_activations, F_geo
            )
            refine_elapsed = time.perf_counter() - refine_start

            total_elapsed = time.perf_counter() - start_time
            logger.info(
                "GEODESIC CHECK: CKA=%.6f (%.2fs, total %.2fs)",
                geodesic_cka_geo, refine_elapsed, total_elapsed
            )

            if geodesic_cka_geo >= geodesic_cka:
                F = F_geo
                iterations = geo_iterations
                geodesic_cka = geodesic_cka_geo
                alignment_stats["alignment_method"] = 1.0  # geodesic
            else:
                alignment_stats["alignment_method"] = 0.0  # linear
        else:
            alignment_stats["alignment_method"] = 0.0  # linear exact

        # =====================================================================
        # SCALE RATIO: ||target|| / ||source @ F||
        # =====================================================================
        source_aligned = b.matmul(source_activations, F)
        aligned_norm_val = self._geodesic_frobenius_norm(source_aligned)
        target_norm_val = self._geodesic_frobenius_norm(target_activations)

        if aligned_norm_val > precision:
            scale_ratio = target_norm_val / aligned_norm_val
        else:
            scale_ratio = 1.0

        # Diagnostics
        target_centered = self._center(target_activations)
        diagnostic = self._diagnose(source_aligned, target_centered, geodesic_cka)

        # Numerical deviation from geodesic CKA = 1.0
        numerical_deviation = max(0.0, 1.0 - geodesic_cka)

        return AlignmentResult(
            feature_transform=F,
            achieved_cka=geodesic_cka,
            iterations=iterations,
            numerical_deviation=numerical_deviation,
            diagnostic=diagnostic,
            precision_threshold=precision,
            scale_ratio=scale_ratio,
            linear_iterations=0,  # Legacy field - geodesic alignment is direct
            linear_residual=alignment_stats.get("relative_space_alignment_error", 0.0),
            gram_condition_number=condition_number,
            source_numerical_rank=source_rank,
            target_numerical_rank=target_rank,
            alignment_rank=alignment_rank,
        )

    def _geodesic_refine(
        self,
        source: "Array",
        target: "Array",
        F_init: "Array",
    ) -> tuple["Array", int, float]:
        """Measure geodesic CKA for the linear alignment.

        Geodesic CKA uses k-NN graph distances with an RBF kernel. The alignment
        is not iteratively refined because the k-NN graph is non-differentiable.

        Parameters
        ----------
        source : Array
            Source activations [n, d_source].
        target : Array
            Target activations [n, d_target].
        F_init : Array
            Transform from linear alignment [d_source, d_target].

        Returns
        -------
        tuple[Array, int, float]
            (F unchanged, 0 iterations, geodesic CKA measurement)
        """
        from modelcypher.core.domain.geometry.cka import compute_cka

        b = self._backend

        # Measure geodesic CKA of the linear alignment
        aligned = b.matmul(source, F_init)
        b.eval(aligned)
        result = compute_cka(aligned, target, b)
        geodesic_cka = result.cka if result.is_valid else 0.0

        precision = sqrt_scalar(machine_epsilon(b, source), b)
        if geodesic_cka < (1.0 - precision):
            logger.debug(
                "Linear alignment geodesic CKA=%.6f (shared-manifold coverage + novelty).",
                geodesic_cka
            )
        else:
            logger.debug("Linear alignment geodesic CKA=%.6f", geodesic_cka)

        return F_init, 0, geodesic_cka

    def _geodesic_frobenius_norm(self, values: "Array") -> float:
        """Compute a geodesic Frobenius-like norm for activation matrices."""
        b = self._backend
        arr = b.array(values) if not hasattr(values, "shape") else values
        shape = b.shape(arr)
        if len(shape) == 1:
            arr = b.reshape(arr, (1, shape[0]))
        elif len(shape) != 2:
            arr = b.reshape(arr, (shape[0], -1))

        norms = geodesic_norms(arr, b)
        norms_sq = b.sum(norms * norms)
        b.eval(norms, norms_sq)
        return float(b.to_scalar(b.sqrt(norms_sq + regularization_epsilon(b, norms))))

    def _diagnose(
        self,
        source_aligned: "Array",
        target_centered: "Array",
        cka: float,
    ) -> AlignmentSignal:
        """Generate diagnostic signal for alignment."""
        from modelcypher.core.domain.geometry.alignment_diagnostic import (
            alignment_signal_from_matrices,
        )

        b = self._backend
        if b.shape(source_aligned) != b.shape(target_centered):
            return AlignmentSignal(
                dimension=3,
                cka_achieved=float(cka),
                iteration=0,
                metadata={
                    "source_shape": list(b.shape(source_aligned)),
                    "target_shape": list(b.shape(target_centered)),
                    "shape_mismatch": 1.0,
                },
            )

        n_samples = b.shape(source_aligned)[0]
        labels = [f"sample:{i}" for i in range(n_samples)]
        return alignment_signal_from_matrices(
            source_aligned,
            target_centered,
            labels,
            backend=b,
            dimension=3,
            cka_achieved=cka,
            iteration=0,
        )

    def compositional_stitch(
        self,
        hidden_transform: "Array",
        source_weight: "Array",
        target_weight: "Array",
    ) -> "Array":
        """Derive projection stitch from hidden alignment + weight geometry.

        Solves for S in the chain S @ W_src @ H ≈ W_tgt to align projection
        weights across architectures.

        Parameters
        ----------
        hidden_transform : Array
            Hidden alignment transform H [d_source_hidden, d_target_hidden].
            Maps source hidden states to target hidden space.
        source_weight : Array
            Source projection weight [d_source_proj, d_source_hidden].
        target_weight : Array
            Target projection weight [d_target_proj, d_target_hidden].

        Returns
        -------
        Array
            The compositional stitch S [d_target_proj, d_source_proj] such that
            S @ source_weight @ hidden_transform ≈ target_weight.
        """
        b = self._backend

        # Validate dimensions
        # source_weight: [source_proj_dim, source_hidden_dim]
        # target_weight: [target_proj_dim, target_hidden_dim]
        # hidden_transform: [source_hidden_dim, target_hidden_dim]
        source_proj_dim, source_hidden_dim = b.shape(source_weight)
        target_proj_dim, target_hidden_dim = b.shape(target_weight)
        h_src, h_tgt = b.shape(hidden_transform)

        if h_src != source_hidden_dim:
            raise ValueError(
                f"hidden_transform source dim ({h_src}) != "
                f"source_weight hidden dim ({source_hidden_dim})"
            )
        if h_tgt != target_hidden_dim:
            raise ValueError(
                f"hidden_transform target dim ({h_tgt}) != "
                f"target_weight hidden dim ({target_hidden_dim})"
            )

        H = b.array(hidden_transform)
        W_src = b.array(source_weight)
        W_tgt = b.array(target_weight)
        compute_dtype = precision_dtype(b, reference=H)
        for arr in (W_src, W_tgt):
            if hasattr(arr, "dtype"):
                try:
                    if b.finfo(arr.dtype).eps < b.finfo(compute_dtype).eps:
                        compute_dtype = arr.dtype
                except Exception:
                    pass
        H = b.astype(H, compute_dtype)
        W_src = b.astype(W_src, compute_dtype)
        W_tgt = b.astype(W_tgt, compute_dtype)
        b.eval(H, W_src, W_tgt)

        # Solve for S in: S @ (W_src @ H) = W_tgt.
        # Avoid solving S @ W_src = W_tgt @ H.T, since H.T @ H != I when H is
        # dimension-reducing.

        # Compute W_src @ H = [src_proj, src_hidden] @ [src_hidden, tgt_hidden]
        #                   = [src_proj, tgt_hidden]
        W_src_transformed = b.matmul(W_src, H)
        b.eval(W_src_transformed)

        # Solve: S @ W_src_transformed = W_tgt
        # Equivalently: W_src_transformed.T @ S.T = W_tgt.T
        A = b.transpose(W_src_transformed)  # [tgt_hidden, src_proj]
        B = b.transpose(W_tgt)  # [tgt_hidden, tgt_proj]
        b.eval(A, B)

        stitch_t = gpu_lstsq(b, A, B)  # [src_proj, tgt_proj]
        stitch = b.transpose(stitch_t)  # [tgt_proj, src_proj]
        b.eval(stitch)
        return stitch

    def compositional_stitch_input(
        self,
        hidden_transform: "Array",
        source_weight: "Array",
        target_weight: "Array",
    ) -> "Array":
        """Derive input stitch for down projections from hidden alignment + weights.

        Solves for S_in in (P @ W_src) @ S_in ≈ W_tgt, where P = H^T is the
        hidden output stitch.

        Parameters
        ----------
        hidden_transform : Array
            Hidden alignment H [d_source_hidden, d_target_hidden].
        source_weight : Array
            Source down_proj weight [d_source_hidden, d_source_inter].
        target_weight : Array
            Target down_proj weight [d_target_hidden, d_target_inter].

        Returns
        -------
        Array
            Input stitch S_in [d_source_inter, d_target_inter].
            For weight matmul chain: hidden_stitch @ W_src @ input_stitch -> [tgt_h, tgt_i]
        """
        b = self._backend

        source_hidden_dim, source_inter_dim = b.shape(source_weight)
        target_hidden_dim, target_inter_dim = b.shape(target_weight)
        h_src, h_tgt = b.shape(hidden_transform)

        if h_src != source_hidden_dim:
            raise ValueError(
                f"hidden_transform source dim ({h_src}) != "
                f"source_weight hidden dim ({source_hidden_dim})"
            )
        if h_tgt != target_hidden_dim:
            raise ValueError(
                f"hidden_transform target dim ({h_tgt}) != "
                f"target_weight hidden dim ({target_hidden_dim})"
            )

        H = b.array(hidden_transform)
        W_src = b.array(source_weight)
        W_tgt = b.array(target_weight)
        compute_dtype = precision_dtype(b, reference=H)
        for arr in (W_src, W_tgt):
            if hasattr(arr, "dtype"):
                try:
                    if b.finfo(arr.dtype).eps < b.finfo(compute_dtype).eps:
                        compute_dtype = arr.dtype
                except Exception:
                    pass
        H = b.astype(H, compute_dtype)
        W_src = b.astype(W_src, compute_dtype)
        W_tgt = b.astype(W_tgt, compute_dtype)
        b.eval(H, W_src, W_tgt)

        # Compute A = P @ W_src = H^T @ W_src
        # Shape: [tgt_hidden, src_hidden] @ [src_hidden, src_inter] = [tgt_hidden, src_inter]
        A = b.matmul(b.transpose(H), W_src)  # [tgt_hidden, src_inter]
        b.eval(A)

        # Solve: A @ S_in = W_tgt for S_in
        # A = [tgt_hidden, src_inter], W_tgt = [tgt_hidden, tgt_inter]
        # S_in = [src_inter, tgt_inter]
        S_in = gpu_lstsq(b, A, W_tgt)  # [src_inter, tgt_inter]
        b.eval(S_in)
        return S_in

    def find_alignment_anchor_space(
        self,
        source_activations: "Array",
        target_activations: "Array",
        source_anchors: "Array",
        target_anchors: "Array",
    ) -> AlignmentResult:
        """Find alignment in anchor-relative space (always well-posed).

        THE ELEGANT SOLUTION FOR UNDERDETERMINED ALIGNMENT:
        ====================================================
        When n_probes < d_hidden, standard alignment is underdetermined.
        Anchor-space alignment works in k dimensions (k = n_anchors),
        which is overdetermined when n_probes > k.

        Example:
        - n_probes = 2048, d_hidden = 11008 → UNDERDETERMINED (n < d)
        - n_probes = 2048, k_anchors = 45 → OVERDETERMINED (n > k)

        The anchor-relative representation captures semantic structure
        in k dimensions, where k is the number of anchor concepts.
        This is dimension-invariant and works across architectures.

        Math:
            S_s = relative_rep(source_activations, source_anchors)  [n, k]
            S_t = relative_rep(target_activations, target_anchors)  [n, k]
            R = procrustes(S_s, S_t)  [k, k] - ALWAYS well-posed when n > k

        Parameters
        ----------
        source_activations : Array
            Source activations [n_samples, d_source].
        target_activations : Array
            Target activations [n_samples, d_target].
        source_anchors : Array
            Source anchor embeddings [n_anchors, d_source].
        target_anchors : Array
            Target anchor embeddings [n_anchors, d_target].

        Returns
        -------
        AlignmentResult
            Alignment result with rotation matrix R [k, k] in anchor space.
        """
        from modelcypher.core.domain.geometry.relative_representation import (
            compute_relative_representation,
            align_relative_representations,
        )

        b = self._backend

        n_s = int(b.shape(source_activations)[0])
        n_t = int(b.shape(target_activations)[0])
        k = int(b.shape(source_anchors)[0])

        if n_s != n_t:
            raise ValueError(f"Sample counts must match: source={n_s}, target={n_t}")

        # Compute relative representations [n, k]
        S_s = compute_relative_representation(source_activations, source_anchors, b)
        S_t = compute_relative_representation(target_activations, target_anchors, b)
        b.eval(S_s, S_t)

        logger.info(
            "ANCHOR-SPACE ALIGNMENT: n=%d, k=%d (overdetermined: %s)",
            n_s, k, "yes" if n_s > k else "no"
        )

        # Procrustes alignment in anchor space [k, k]
        R, alignment_error = align_relative_representations(S_s, S_t, backend=b)
        b.eval(R)

        # Aligned source in anchor space
        S_aligned = b.matmul(S_s, b.transpose(R))
        b.eval(S_aligned)

        # Compute CKA in anchor space
        from modelcypher.core.domain.geometry.cka import compute_cka
        cka_result = compute_cka(S_aligned, S_t, b)
        cka = cka_result.cka if cka_result.is_valid else 0.0

        eps = machine_epsilon(b, source_activations)
        precision = sqrt_scalar(eps, b)

        logger.info(
            "ANCHOR-SPACE RESULT: CKA=%.6f, alignment_error=%.6f",
            cka, alignment_error
        )

        return AlignmentResult(
            feature_transform=R,  # [k, k] rotation in anchor space
            iterations=0,
            numerical_deviation=max(0.0, 1.0 - cka),
            precision_threshold=float(precision),
            achieved_cka=cka,
            diagnostic=None,
            scale_ratio=1.0,  # No scale change in anchor space
            linear_iterations=0,
            linear_residual=alignment_error,
        )
