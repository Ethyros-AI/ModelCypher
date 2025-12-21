from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import mlx.core as mx

from modelcypher.ports.backend import Backend, Array
from modelcypher.core.domain.geometry.permutation_aligner import PermutationAligner as DomainAligner
from modelcypher.core.domain.geometry.permutation_aligner import Config as DomainConfig
from modelcypher.core.domain.geometry.permutation_aligner import AlignmentResult as DomainResult


@dataclass(frozen=True)
class AlignmentResult:
    permutation: Array
    signs: Array
    match_quality: float
    match_confidences: list[float]
    sign_flip_count: int
    is_sparse_permutation: bool = False
    assignment_indices: Optional[list[int]] = None

    @classmethod
    def from_domain(cls, domain_result: DomainResult, backend: Backend) -> AlignmentResult:
        # Convert MLX arrays to backend arrays (likely just wrapping or casting)
        # Assuming backend is MLX or compatible if we are using domain aligner
        
        # We need to bridge the types.
        # If backend is MLX backend, it handles mx.array.
        # If backend is Numpy, we might need conversion (but DomainAligner is MLX-only).
        
        # Ideally, we return backend-wrapped arrays.
        # If the backend is MLX-based, this is trivial.
        
        perm = domain_result.permutation
        signs = domain_result.signs
        
        return cls(
            permutation=perm,
            signs=signs,
            match_quality=domain_result.match_quality,
            match_confidences=domain_result.match_confidences,
            sign_flip_count=domain_result.sign_flip_count,
            is_sparse_permutation=domain_result.is_sparse_permutation,
            assignment_indices=domain_result.assignment_indices
        )


@dataclass(frozen=True)
class Config:
    min_match_threshold: float = 0.1
    use_anchor_grounding: bool = True
    top_k: int = 5
    
    def to_domain(self) -> DomainConfig:
        return DomainConfig(
            min_match_threshold=self.min_match_threshold,
            use_anchor_grounding=self.use_anchor_grounding,
            top_k=self.top_k
        )


@dataclass(frozen=True)
class FusionConfig:
    interference_threshold: float = 0.5
    source_alpha: float = 0.5
    normalize: bool = False


class PermutationAligner:
    """
    Use case wraper for PermutationAligner.
    Delegates to the core domain implementation (MLX-based) for actual logic.
    """
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def align(
        self,
        source_weight: Array,
        target_weight: Array,
        anchors: Array | None = None,
        config: Config = Config(),
    ) -> AlignmentResult:
        # Ensure inputs are MLX arrays
        # The DomainAligner expects mlx.core.array
        
        # If the passed 'Array' is already mx.array (via MLXBackend), good.
        # If it's numpy, we convert.
        
        src = self._ensure_mlx(source_weight)
        tgt = self._ensure_mlx(target_weight)
        anc = self._ensure_mlx(anchors) if anchors is not None else None
        
        domain_res = DomainAligner.align(
            source_weight=src,
            target_weight=tgt,
            anchors=anc,
            config=config.to_domain()
        )
        
        return AlignmentResult.from_domain(domain_res, self.backend)

    def align_via_anchor_projection(
        self,
        source_weight: Array,
        target_weight: Array,
        anchors: Array,
        config: Config = Config(),
    ) -> AlignmentResult:
        src = self._ensure_mlx(source_weight)
        tgt = self._ensure_mlx(target_weight)
        anc = self._ensure_mlx(anchors)
        
        domain_res = DomainAligner.align_via_anchor_projection(
            source_weight=src,
            target_weight=tgt,
            anchors=anc,
            config=config.to_domain()
        )
        
        return AlignmentResult.from_domain(domain_res, self.backend)

    def apply(
        self,
        weight: Array,
        alignment: AlignmentResult,
        align_output: bool = True,
        align_input: bool = False,
    ) -> Array:
        # Construct a DomainResult from our AlignmentResult to pass to DomainAligner.apply
        # Or just use the permutation/signs directly since DomainAligner.apply takes those or result?
        # DomainAligner.apply signature: 
        # static func apply(weight: MLXArray, result: AlignmentResult, ...) -> MLXArray
        
        # We need to reconstruct the domain result object or implement apply manually using MLX
        # Actually DomainAligner.apply takes the alignment result.
        
        # Reconstruct minimal domain result
        d_res = DomainResult(
            permutation=self._ensure_mlx(alignment.permutation),
            signs=self._ensure_mlx(alignment.signs),
            match_quality=0.0,
            match_confidences=[],
            sign_flip_count=0,
            is_sparse_permutation=alignment.is_sparse_permutation,
            assignment_indices=alignment.assignment_indices
        )
        
        w = self._ensure_mlx(weight)
        
        out = DomainAligner.apply(
            weight=w,
            alignment=d_res,
            align_output=align_output,
            align_input=align_input
        )
        
        return out # Return mx.array

    def rebasin_mlp_only(
        self,
        source_weights: dict[str, Array],
        target_weights: dict[str, Array],
        anchors: Array,
        config: Config = Config(),
    ) -> tuple[dict[str, Array], float, int]:
        
        src_map = {k: self._ensure_mlx(v) for k, v in source_weights.items()}
        tgt_map = {k: self._ensure_mlx(v) for k, v in target_weights.items()}
        anc = self._ensure_mlx(anchors)
        
        aligned_weights_mlx, quality, count = DomainAligner.rebasin_mlp_only(
            source_weights=src_map,
            target_weights=tgt_map,
            anchors=anc,
            config=config.to_domain()
        )
        
        # Result is Dict[str, mx.array]
        return aligned_weights_mlx, quality, count

    def fuse(
        self,
        source_weight: Array,
        aligned_target_weight: Array,
        alignment: AlignmentResult,
        config: FusionConfig = FusionConfig(),
    ) -> Array:
        # Simple weighted average with mask
        # Implement using MLX
        
        src = self._ensure_mlx(source_weight)
        tgt = self._ensure_mlx(aligned_target_weight)
        
        # Confidence mask
        conf = mx.array(alignment.match_confidences, dtype=mx.float32)
        mask = conf.reshape((-1, 1))
        
        alpha = config.source_alpha
        
        avg = (src * alpha) + (tgt * (1.0 - alpha))
        
        # "Interference" logic from original use case: fused = (avg * mask) + (source * (1 - mask))
        # Wait, the original logic was: fused = (avg * mask) + (source * (1 - mask))
        # This implies:
        # If high confidence (mask ~ 1), use average.
        # If low confidence (mask ~ 0), keep source original.
        
        fused = (avg * mask) + (src * (1.0 - mask))
        return fused

    def _ensure_mlx(self, arr: Array) -> mx.array:
        if isinstance(arr, mx.array):
            return arr
        if isinstance(arr, np.ndarray):
            return mx.array(arr)
        # Assuming backend might wrap it, try conversion
        if hasattr(self.backend, "to_numpy"):
            return mx.array(self.backend.to_numpy(arr))
        return mx.array(np.array(arr))
