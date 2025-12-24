"""
Embedding Bridge for Cross-Vocabulary Model Merging.

Builds transformation bridges between embedding spaces when models have
different vocabularies. Implements multiple strategies with automatic
method selection based on alignment quality.

Methods:
- FVT (Fast Vocabulary Transfer): Copy overlapping tokens, average decompositions
- Procrustes: Orthogonal transformation via SVD on anchor pairs
- Affine: Learned linear transformation via least squares

References:
- Zero-Shot Tokenizer Transfer (ZeTT), NeurIPS 2024
- Model Stitching with Affine Layers, 2025
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modelcypher.core.domain.merging.vocabulary_alignment import (
    AlignmentMethod,
    TokenMapping,
    VocabularyAlignmentResult,
)

logger = logging.getLogger(__name__)


class BridgeMethod(str, Enum):
    """Methods for building embedding bridges."""

    FVT = "fvt"  # Fast Vocabulary Transfer
    PROCRUSTES = "procrustes"  # Orthogonal transformation
    AFFINE = "affine"  # Learned affine transformation
    HYBRID = "hybrid"  # FVT for exact, Procrustes for rest


@dataclass
class EmbeddingBridgeConfig:
    """Configuration for embedding bridge construction."""

    # Method selection
    auto_select: bool = True  # Automatically choose method based on overlap
    fallback_chain: Tuple[str, ...] = ("fvt", "procrustes", "affine")

    # Quality thresholds
    quality_threshold: float = 0.8  # Min anchor preservation to accept
    fvt_threshold: float = 0.9  # Overlap ratio above which FVT is sufficient

    # Procrustes options
    allow_reflections: bool = False
    allow_scaling: bool = False

    # Affine options
    regularization: float = 1e-4  # L2 regularization for affine fit

    # Anchor selection
    use_semantic_primes: bool = True
    min_anchor_pairs: int = 10  # Minimum anchor pairs for Procrustes/Affine


@dataclass
class EmbeddingBridgeResult:
    """Result of embedding bridge construction."""

    method_used: BridgeMethod
    bridged_embeddings: np.ndarray  # [target_vocab_size, hidden_dim]

    # Quality metrics
    alignment_quality: float  # Overall anchor distance preservation
    per_method_quality: Dict[str, float] = field(default_factory=dict)

    # Transformation matrices (for inspection/debugging)
    rotation_matrix: Optional[np.ndarray] = None  # For Procrustes
    affine_weight: Optional[np.ndarray] = None  # For Affine
    affine_bias: Optional[np.ndarray] = None  # For Affine

    # Diagnostics
    anchor_pairs_used: int = 0
    methods_tried: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "methodUsed": self.method_used.value,
            "alignmentQuality": round(self.alignment_quality, 4),
            "perMethodQuality": {
                k: round(v, 4) for k, v in self.per_method_quality.items()
            },
            "anchorPairsUsed": self.anchor_pairs_used,
            "methodsTried": self.methods_tried,
            "warnings": self.warnings,
            "embeddingShape": list(self.bridged_embeddings.shape),
        }


class EmbeddingBridgeBuilder:
    """
    Builds embedding bridges between models with different vocabularies.

    Implements intelligent method selection:
    1. Try FVT, measure anchor preservation
    2. If quality < threshold, try Procrustes
    3. If still insufficient, use Affine layer
    4. Return best result

    Usage:
        builder = EmbeddingBridgeBuilder()
        result = builder.build(
            source_embeddings,
            target_embeddings,
            alignment,
            anchor_pairs=[(100, 200), (101, 201), ...]
        )
    """

    def __init__(self, config: Optional[EmbeddingBridgeConfig] = None) -> None:
        self.config = config or EmbeddingBridgeConfig()

    def build(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        alignment: VocabularyAlignmentResult,
        anchor_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> EmbeddingBridgeResult:
        """
        Build embedding bridge from source to target vocabulary.

        Args:
            source_embeddings: [source_vocab_size, hidden_dim] embeddings
            target_embeddings: [target_vocab_size, hidden_dim] embeddings
            alignment: VocabularyAlignmentResult from VocabularyAligner
            anchor_pairs: List of (source_id, target_id) pairs for alignment

        Returns:
            EmbeddingBridgeResult with bridged embeddings
        """
        source_vocab_size, hidden_dim = source_embeddings.shape
        target_vocab_size = target_embeddings.shape[0]

        # Extract anchor pairs from alignment if not provided
        if anchor_pairs is None:
            anchor_pairs = self._extract_anchor_pairs(alignment)

        methods_tried: List[str] = []
        per_method_quality: Dict[str, float] = {}
        warnings: List[str] = []

        best_result: Optional[EmbeddingBridgeResult] = None
        best_quality = -1.0

        # Determine method order based on config
        if self.config.auto_select:
            methods = self._select_method_order(alignment)
        else:
            methods = [BridgeMethod(m) for m in self.config.fallback_chain]

        for method in methods:
            methods_tried.append(method.value)

            try:
                if method == BridgeMethod.FVT:
                    bridged = self._fvt_bridge(
                        source_embeddings, target_embeddings, alignment
                    )
                elif method == BridgeMethod.PROCRUSTES:
                    if len(anchor_pairs) < self.config.min_anchor_pairs:
                        warnings.append(
                            f"Skipping Procrustes: only {len(anchor_pairs)} anchor pairs "
                            f"(min: {self.config.min_anchor_pairs})"
                        )
                        continue
                    bridged = self._procrustes_bridge(
                        source_embeddings, target_embeddings, anchor_pairs
                    )
                elif method == BridgeMethod.AFFINE:
                    if len(anchor_pairs) < self.config.min_anchor_pairs:
                        warnings.append(
                            f"Skipping Affine: only {len(anchor_pairs)} anchor pairs"
                        )
                        continue
                    bridged = self._affine_bridge(
                        source_embeddings, target_embeddings, anchor_pairs
                    )
                elif method == BridgeMethod.HYBRID:
                    bridged = self._hybrid_bridge(
                        source_embeddings, target_embeddings, alignment, anchor_pairs
                    )
                else:
                    continue

                # Measure quality
                quality = self._measure_quality(
                    bridged, target_embeddings, anchor_pairs
                )
                per_method_quality[method.value] = quality

                if quality > best_quality:
                    best_quality = quality
                    best_result = EmbeddingBridgeResult(
                        method_used=method,
                        bridged_embeddings=bridged,
                        alignment_quality=quality,
                        anchor_pairs_used=len(anchor_pairs),
                    )

                # Early exit if quality threshold met
                if quality >= self.config.quality_threshold:
                    logger.info(f"Method {method.value} achieved quality {quality:.3f}")
                    break

            except Exception as e:
                warnings.append(f"{method.value} failed: {str(e)}")
                logger.warning(f"Bridge method {method.value} failed: {e}")

        if best_result is None:
            # Fallback: use FVT regardless of quality
            warnings.append("All methods failed or insufficient, using FVT fallback")
            bridged = self._fvt_bridge(
                source_embeddings, target_embeddings, alignment
            )
            quality = self._measure_quality(bridged, target_embeddings, anchor_pairs)
            best_result = EmbeddingBridgeResult(
                method_used=BridgeMethod.FVT,
                bridged_embeddings=bridged,
                alignment_quality=quality,
                anchor_pairs_used=len(anchor_pairs),
            )

        # Add diagnostics
        best_result.methods_tried = methods_tried
        best_result.per_method_quality = per_method_quality
        best_result.warnings = warnings

        return best_result

    def _select_method_order(
        self, alignment: VocabularyAlignmentResult
    ) -> List[BridgeMethod]:
        """Select method order based on alignment characteristics."""
        overlap_ratio = alignment.overlap_ratio

        if overlap_ratio >= self.config.fvt_threshold:
            # High overlap: FVT should be sufficient
            return [BridgeMethod.FVT]
        elif overlap_ratio >= 0.5:
            # Medium overlap: try FVT first, then Procrustes
            return [BridgeMethod.FVT, BridgeMethod.PROCRUSTES, BridgeMethod.AFFINE]
        else:
            # Low overlap: Procrustes or Affine needed
            return [BridgeMethod.PROCRUSTES, BridgeMethod.AFFINE, BridgeMethod.FVT]

    def _extract_anchor_pairs(
        self, alignment: VocabularyAlignmentResult
    ) -> List[Tuple[int, int]]:
        """Extract anchor pairs from alignment mappings."""
        pairs = []
        for source_id, mapping in alignment.mappings.items():
            if mapping.method == AlignmentMethod.EXACT and mapping.target_token_id is not None:
                pairs.append((source_id, mapping.target_token_id))
        return pairs

    def _fvt_bridge(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        alignment: VocabularyAlignmentResult,
    ) -> np.ndarray:
        """
        Fast Vocabulary Transfer: copy overlapping, average decompositions.

        For exact matches: copy source embedding
        For decomposed tokens: average the target subtoken embeddings
        For unmapped: use nearest target embedding (cosine similarity)
        """
        source_vocab, hidden_dim = source_embeddings.shape
        target_vocab = target_embeddings.shape[0]

        # Initialize with zeros
        bridged = np.zeros((target_vocab, hidden_dim), dtype=source_embeddings.dtype)
        used_mask = np.zeros(target_vocab, dtype=bool)

        for source_id, mapping in alignment.mappings.items():
            if source_id >= source_vocab:
                continue

            if mapping.method == AlignmentMethod.EXACT:
                if mapping.target_token_id is not None and mapping.target_token_id < target_vocab:
                    bridged[mapping.target_token_id] = source_embeddings[source_id]
                    used_mask[mapping.target_token_id] = True

            elif mapping.method == AlignmentMethod.DECOMPOSED:
                if mapping.decomposition and mapping.target_token_id is not None:
                    # Average the decomposition token embeddings
                    valid_ids = [t for t in mapping.decomposition if t < target_vocab]
                    if valid_ids:
                        avg_embed = np.mean(
                            target_embeddings[valid_ids], axis=0
                        )
                        if mapping.target_token_id < target_vocab:
                            bridged[mapping.target_token_id] = avg_embed
                            used_mask[mapping.target_token_id] = True

            elif mapping.method == AlignmentMethod.SEMANTIC:
                if mapping.target_token_id is not None and mapping.target_token_id < target_vocab:
                    # Use source embedding for semantic matches
                    bridged[mapping.target_token_id] = source_embeddings[source_id]
                    used_mask[mapping.target_token_id] = True

        # For unused target positions, keep original target embeddings
        bridged[~used_mask] = target_embeddings[~used_mask]

        return bridged

    def _procrustes_bridge(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        anchor_pairs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Procrustes alignment: find optimal orthogonal transformation.

        Solves: R* = argmin ||source @ R - target||_F
        Using SVD: R = V @ U^T where M = source^T @ target = U @ S @ V^T

        Returns target-sized output with transformed source embeddings
        placed at anchor target positions.
        """
        target_vocab, hidden_dim = target_embeddings.shape

        if not anchor_pairs:
            # No anchors - return copy of target embeddings
            return target_embeddings.copy()

        # Extract anchor embeddings
        source_anchors = np.array([source_embeddings[s] for s, t in anchor_pairs])
        target_anchors = np.array([target_embeddings[t] for s, t in anchor_pairs])

        # Center the data
        source_mean = np.mean(source_anchors, axis=0, keepdims=True)
        target_mean = np.mean(target_anchors, axis=0, keepdims=True)

        source_centered = source_anchors - source_mean
        target_centered = target_anchors - target_mean

        # Compute optimal rotation via SVD
        # M = source^T @ target
        M = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt

        # Handle reflection if not allowed
        if not self.config.allow_reflections and np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Handle scaling if allowed
        scale = 1.0
        if self.config.allow_scaling:
            source_var = np.sum(source_centered ** 2)
            if source_var > 1e-10:
                scale = np.sum(S) / source_var

        # Start with target embeddings as base
        bridged = target_embeddings.copy()

        # Transform and place source anchors at target positions
        for source_id, target_id in anchor_pairs:
            if source_id < source_embeddings.shape[0] and target_id < target_vocab:
                source_centered_vec = source_embeddings[source_id] - source_mean.flatten()
                transformed = (source_centered_vec @ R) * scale + target_mean.flatten()
                bridged[target_id] = transformed

        return bridged

    def _affine_bridge(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        anchor_pairs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Affine transformation: learn W and b such that source @ W + b ≈ target.

        Uses regularized least squares for stability.
        Returns target-sized output with transformed source embeddings
        placed at anchor target positions.
        """
        target_vocab, hidden_dim = target_embeddings.shape

        if not anchor_pairs:
            return target_embeddings.copy()

        # Extract anchor embeddings
        source_anchors = np.array([source_embeddings[s] for s, t in anchor_pairs])
        target_anchors = np.array([target_embeddings[t] for s, t in anchor_pairs])

        n_anchors = source_anchors.shape[0]

        # Augment source with ones for bias term
        # [n_anchors, hidden_dim + 1]
        source_aug = np.hstack([source_anchors, np.ones((n_anchors, 1))])

        # Solve regularized least squares: (X^T X + λI) W = X^T Y
        XtX = source_aug.T @ source_aug
        XtX += self.config.regularization * np.eye(XtX.shape[0])
        XtY = source_aug.T @ target_anchors

        try:
            W_aug = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            W_aug = np.linalg.pinv(source_aug) @ target_anchors

        # Extract W and b
        W = W_aug[:-1, :]  # [hidden_dim, hidden_dim]
        b = W_aug[-1, :]  # [hidden_dim]

        # Start with target embeddings as base
        bridged = target_embeddings.copy()

        # Transform and place source anchors at target positions
        for source_id, target_id in anchor_pairs:
            if source_id < source_embeddings.shape[0] and target_id < target_vocab:
                transformed = source_embeddings[source_id] @ W + b
                bridged[target_id] = transformed

        return bridged

    def _hybrid_bridge(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        alignment: VocabularyAlignmentResult,
        anchor_pairs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Hybrid approach: FVT for exact matches, Procrustes for the rest.
        """
        target_vocab, hidden_dim = target_embeddings.shape

        # Start with Procrustes transformation for all
        bridged = self._procrustes_bridge(
            source_embeddings, target_embeddings, anchor_pairs
        )

        # Override exact matches with direct copy (higher confidence)
        for source_id, mapping in alignment.mappings.items():
            if mapping.method == AlignmentMethod.EXACT:
                if (mapping.target_token_id is not None and
                    mapping.target_token_id < target_vocab and
                    source_id < source_embeddings.shape[0]):
                    bridged[mapping.target_token_id] = source_embeddings[source_id]

        return bridged

    def _measure_quality(
        self,
        bridged_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        anchor_pairs: List[Tuple[int, int]],
    ) -> float:
        """
        Measure anchor distance preservation quality.

        Computes correlation between pairwise distances in bridged vs target space.
        """
        if len(anchor_pairs) < 2:
            return 0.0

        # Extract anchor embeddings
        bridged_anchors = []
        target_anchors = []

        for source_id, target_id in anchor_pairs:
            if target_id < bridged_embeddings.shape[0] and target_id < target_embeddings.shape[0]:
                bridged_anchors.append(bridged_embeddings[target_id])
                target_anchors.append(target_embeddings[target_id])

        if len(bridged_anchors) < 2:
            return 0.0

        bridged_arr = np.array(bridged_anchors)
        target_arr = np.array(target_anchors)

        # Compute pairwise cosine similarities
        bridged_norm = bridged_arr / (np.linalg.norm(bridged_arr, axis=1, keepdims=True) + 1e-10)
        target_norm = target_arr / (np.linalg.norm(target_arr, axis=1, keepdims=True) + 1e-10)

        bridged_sim = bridged_norm @ bridged_norm.T
        target_sim = target_norm @ target_norm.T

        # Flatten upper triangular (excluding diagonal)
        n = bridged_sim.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        bridged_flat = bridged_sim[triu_indices]
        target_flat = target_sim[triu_indices]

        if len(bridged_flat) == 0:
            return 0.0

        # Compute Pearson correlation
        correlation = np.corrcoef(bridged_flat, target_flat)[0, 1]

        # Handle NaN (constant vectors)
        if np.isnan(correlation):
            return 0.0

        # Convert to [0, 1] range (correlation is in [-1, 1])
        return (correlation + 1.0) / 2.0


def format_bridge_report(result: EmbeddingBridgeResult) -> str:
    """Format a human-readable bridge construction report."""
    lines = [
        "=" * 60,
        "EMBEDDING BRIDGE REPORT",
        "=" * 60,
        "",
        f"Method Used: {result.method_used.value.upper()}",
        f"Alignment Quality: {result.alignment_quality:.1%}",
        f"Anchor Pairs: {result.anchor_pairs_used:,}",
        f"Output Shape: {result.bridged_embeddings.shape}",
        "",
    ]

    if result.per_method_quality:
        lines.append("Per-Method Quality:")
        for method, quality in sorted(
            result.per_method_quality.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {method}: {quality:.1%}")
        lines.append("")

    if result.methods_tried:
        lines.append(f"Methods Tried: {', '.join(result.methods_tried)}")
        lines.append("")

    if result.warnings:
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")
        lines.append("")

    # Recommendation
    if result.alignment_quality >= 0.9:
        recommendation = "HIGH QUALITY - Safe to proceed with merge"
    elif result.alignment_quality >= 0.7:
        recommendation = "ACCEPTABLE - Proceed with caution, validate output"
    elif result.alignment_quality >= 0.5:
        recommendation = "MARGINAL - Consider alternative approaches"
    else:
        recommendation = "LOW QUALITY - Not recommended for production use"

    lines.extend([
        "Recommendation:",
        f"  {recommendation}",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
