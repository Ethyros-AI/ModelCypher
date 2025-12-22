from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import mlx.core as mx

from modelcypher.core.domain.geometry.permutation_aligner import (
    PermutationAligner as DomainPermutationAligner,
    AlignmentResult as DomainAlignmentResult,
    Config as DomainConfig,
)
from modelcypher.ports.backend import Backend, Array


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
    def from_domain_result(cls, res: DomainAlignmentResult) -> AlignmentResult:
        return cls(
            permutation=res.permutation,
            signs=res.signs,
            match_quality=res.match_quality,
            match_confidences=res.match_confidences,
            sign_flip_count=res.sign_flip_count,
            is_sparse_permutation=res.is_sparse_permutation,
            assignment_indices=res.assignment_indices,
        )


@dataclass(frozen=True)
class Config:
    min_match_threshold: float = 0.1
    use_anchor_grounding: bool = True
    top_k: int = 5

    def to_domain_config(self) -> DomainConfig:
        return DomainConfig(
            min_match_threshold=self.min_match_threshold,
            use_anchor_grounding=self.use_anchor_grounding,
            top_k=self.top_k,
        )


@dataclass(frozen=True)
class FusionConfig:
    interference_threshold: float = 0.5
    source_alpha: float = 0.5
    normalize: bool = False


class PermutationAligner:
    """Permutation alignment wrapper that defaults to domain MLX math."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def align(
        self,
        source_weight: Array,
        target_weight: Array,
        anchors: Array | None = None,
        config: Config = Config(),
    ) -> AlignmentResult:
        src = self._ensure_mx(source_weight)
        tgt = self._ensure_mx(target_weight)
        anc = self._ensure_mx(anchors) if anchors is not None else None
        domain_config = config.to_domain_config() if isinstance(config, Config) else config
        result = DomainPermutationAligner.align(src, tgt, anc, domain_config)
        return AlignmentResult.from_domain_result(result)

    def align_via_anchor_projection(
        self,
        source_weight: Array,
        target_weight: Array,
        anchors: Array,
        config: Config = Config(),
    ) -> AlignmentResult:
        src = self._ensure_mx(source_weight)
        tgt = self._ensure_mx(target_weight)
        anc = self._ensure_mx(anchors)
        domain_config = config.to_domain_config() if isinstance(config, Config) else config
        result = DomainPermutationAligner.align_via_anchor_projection(src, tgt, anc, domain_config)
        return AlignmentResult.from_domain_result(result)

    def rebasin_mlp_only(
        self,
        source_weights: dict[str, Array],
        target_weights: dict[str, Array],
        anchors: Array,
        config: Config = Config(),
    ) -> tuple[dict[str, Array], float, int]:
        src_map = {k: self._ensure_mx(v) for k, v in source_weights.items()}
        tgt_map = {k: self._ensure_mx(v) for k, v in target_weights.items()}
        anc = self._ensure_mx(anchors)
        domain_config = config.to_domain_config() if isinstance(config, Config) else config
        aligned, quality, count = DomainPermutationAligner.rebasin_mlp_only(
            src_map,
            tgt_map,
            anc,
            domain_config,
        )
        return aligned, quality, count

    def apply(
        self,
        weight: Array,
        alignment: AlignmentResult | DomainAlignmentResult,
        align_output: bool = True,
        align_input: bool = False,
    ) -> Array:
        w = self._ensure_mx(weight)
        domain_alignment = self._to_domain_alignment(alignment)
        return DomainPermutationAligner.apply(w, domain_alignment, align_output, align_input)

    def fuse(
        self,
        source_weight: Array,
        aligned_target_weight: Array,
        alignment: AlignmentResult | DomainAlignmentResult,
        config: FusionConfig = FusionConfig(),
    ) -> Array:
        src = self._ensure_mx(source_weight)
        tgt = self._ensure_mx(aligned_target_weight)
        conf = mx.array(alignment.match_confidences, dtype=mx.float32)
        mask = conf.reshape((-1, 1))
        alpha = config.source_alpha
        avg = (src * alpha) + (tgt * (1.0 - alpha))
        fused = (avg * mask) + (src * (1.0 - mask))
        mx.eval(fused)
        return fused

    def _ensure_mx(self, arr: Any) -> mx.array:
        if isinstance(arr, mx.array):
            return arr
        return mx.array(arr)

    def _to_domain_alignment(self, alignment: AlignmentResult | DomainAlignmentResult) -> DomainAlignmentResult:
        if isinstance(alignment, DomainAlignmentResult):
            return alignment
        permutation = self._ensure_mx(alignment.permutation)
        signs = self._ensure_mx(alignment.signs)
        return DomainAlignmentResult(
            permutation=permutation,
            signs=signs,
            match_quality=alignment.match_quality,
            match_confidences=alignment.match_confidences,
            sign_flip_count=alignment.sign_flip_count,
            is_sparse_permutation=alignment.is_sparse_permutation,
            assignment_indices=alignment.assignment_indices,
        )
