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

"""
Stage 0: VOCABULARY ALIGNMENT - Cross-vocabulary merging.

Uses the superior CrossVocabMerger pipeline:
1. Analyze vocabularies (stats, compatibility)
2. Build token alignment map (exact + embedding similarity)
3. Project source embeddings to target space (Procrustes/OT)
4. Blend aligned embeddings with quality-weighted alpha
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.cache import (
    CacheConfig,
    ComputationCache,
    TwoLevelCache,
    content_hash,
)
from modelcypher.core.domain.geometry.alignment_diagnostic import (
    AlignmentSignal,
    alignment_signal_from_matrices,
)
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean

logger = logging.getLogger(__name__)

# Session cache for anchor maps - keyed by (embedding_hash, tokenizer_id, map_type)
_anchor_map_cache: dict[str, dict[str | int, "object"]] = {}
_anchor_disk_cache: "TwoLevelCache[dict[str, list[float]]] | None" = None
_ANCHOR_CACHE_VERSION = 1


@dataclass
class VocabularyConfig:
    """Configuration for Stage 0 vocabulary alignment.

    Note on thresholds:
        All thresholds default to None and are derived from data at runtime:

        - similarity_threshold: If None, derived from distribution of embedding
          cosine similarities. Uses spectral gap to find natural boundary.

        - confidence_threshold: If None, derived from alignment confidence
          distribution using spectral gap detection.

        - blend_alpha: If None, computed from relative alignment quality
          (CKA scores) between source and target. Higher alignment quality
          means higher weight for that model's embeddings.

        - min_compatibility_score: If None, set to machine_epsilon - any
          measurable compatibility is acceptable.

        - min_coverage: If None, set to 0.0 - no arbitrary coverage floor.
    """

    # Projection strategy: procrustes, pca, optimal_transport, cca
    projection_strategy: str = "procrustes"

    # Alignment thresholds - None means derive from data
    similarity_threshold: float | None = None
    confidence_threshold: float | None = None

    # Embedding blending - None means compute from alignment quality
    blend_alpha: float | None = None
    preserve_special_tokens: bool = True

    # Quality thresholds - None means no arbitrary floor
    min_compatibility_score: float | None = None
    min_coverage: float | None = None

    # Advanced
    use_embedding_similarity: bool = True
    anchor_count: int = 1000
    max_similarity_pairs: int = 5_000_000
    max_unmapped_similarity: int = 5000
    max_prefix_length: int = 8
    max_prefix_matches: int = 3
    similarity_batch_size: int = 128

    # Phase-lock alignment tuning
    alignment_iterations: int = 8
    alignment_solver_iterations: int = 5000
    alignment_solver_rounds: int = 1
    alignment_tolerance: float = 1e-12
    phase_lock_max_iterations: int = 0
    use_all_support_texts: bool = True
    use_byte_anchors_for_atlas: bool = True
    balance_anchor_weights: bool = True
    use_coverage_anchor_selection: bool = True
    coverage_k_neighbors: int | None = None
    coverage_candidate_multiplier: int = 3
    strict_token_alignment: bool = True


@dataclass
class VocabularyResult:
    """Result of Stage 0 vocabulary alignment."""

    modified_weights: dict[str, "object"]
    metrics: dict[str, Any]
    was_aligned: bool
    alignment_map: Any | None = None


def stage_vocabulary_align(
    source_weights: dict[str, "object"],
    target_weights: dict[str, "object"],
    source_tokenizer: Any | None,
    target_tokenizer: Any | None,
    config: VocabularyConfig,
) -> VocabularyResult:
    """
    Stage 0: Align source vocabulary to target vocabulary.

    Uses CrossVocabMerger for sophisticated vocabulary alignment with:
    - Multi-strategy projection (Procrustes, PCA, Optimal Transport)
    - Embedding similarity for unmapped tokens
    - Quality-weighted blending

    Args:
        source_weights: Source model weights
        target_weights: Target model weights
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        config: Vocabulary alignment configuration

    Returns:
        VocabularyResult with modified weights, metrics, and alignment status
    """
    backend = get_default_backend()
    cache = ComputationCache.shared()

    metrics: dict[str, Any] = {
        "enabled": True,
        "tokenizers_provided": source_tokenizer is not None and target_tokenizer is not None,
    }
    metrics["alignment_signals"] = {}
    metrics["timing_ms"] = {}

    # Tokenizers are required for deterministic binary/vocab alignment.
    if source_tokenizer is None or target_tokenizer is None:
        raise ValueError("Tokenizers are required for binary/vocabulary alignment.")

    # Find embedding layer keys
    embed_keys = [k for k in source_weights if "embed" in k.lower() and "weight" in k.lower()]
    if not embed_keys:
        logger.info("No embedding layer found, skipping vocabulary alignment")
        metrics["skipped"] = True
        metrics["reason"] = "no_embedding_layer"
        return VocabularyResult(source_weights, metrics, False, None)

    # Import CrossVocabMerger
    try:
        from modelcypher.core.domain.vocabulary.cross_vocab_merger import (
            CrossVocabMergeConfig,
            CrossVocabMerger,
        )
        from modelcypher.core.domain.vocabulary.embedding_projector import (
            ProjectionStrategy,
        )
    except ImportError as e:
        logger.warning("CrossVocabMerger not available: %s", e)
        metrics["skipped"] = True
        metrics["reason"] = f"import_error: {e}"
        return VocabularyResult(source_weights, metrics, False, None)

    # Map config string to ProjectionStrategy enum
    strategy_map = {
        "procrustes": ProjectionStrategy.PROCRUSTES,
        "pca": ProjectionStrategy.PCA,
        "optimal_transport": ProjectionStrategy.OPTIMAL_TRANSPORT,
        "cca": ProjectionStrategy.CCA,
        "truncate": ProjectionStrategy.TRUNCATE,
    }
    projection_strategy = strategy_map.get(
        config.projection_strategy.lower(),
        ProjectionStrategy.PROCRUSTES,
    )

    # Extract vocabulary mappings from tokenizers
    source_vocab = _extract_vocab(source_tokenizer)
    target_vocab = _extract_vocab(target_tokenizer)

    if source_vocab is None or target_vocab is None:
        logger.warning("Could not extract vocabulary from tokenizers")
        metrics["skipped"] = True
        metrics["reason"] = "vocab_extraction_failed"
        return VocabularyResult(source_weights, metrics, False, None)

    source_tokenizer_key = _make_tokenizer_cache_key(source_tokenizer, source_vocab)
    target_tokenizer_key = _make_tokenizer_cache_key(target_tokenizer, target_vocab)

    metrics["source_vocab_size"] = len(source_vocab)
    metrics["target_vocab_size"] = len(target_vocab)

    # Check for vocab compatibility before doing expensive operations
    overlap = set(source_vocab.keys()) & set(target_vocab.keys())
    overlap_ratio = len(overlap) / max(len(source_vocab), 1)
    metrics["overlap_count"] = len(overlap)
    metrics["overlap_ratio"] = overlap_ratio

    if overlap_ratio > 0.95:
        logger.info(
            "Vocabulary overlap %.1f%% - vocabularies compatible, "
            "still seeking exact kernel alignment",
            overlap_ratio * 100,
        )
        metrics["compatible_vocabulary"] = True

    # Apply merger to each embedding layer
    modified_weights = source_weights.copy()
    aligned_layers = 0
    stage_start = time.perf_counter()
    alignment_tol = config.alignment_tolerance
    alignment_iterations = max(1, config.alignment_iterations)
    solver_iterations = config.alignment_solver_iterations
    solver_rounds = max(1, config.alignment_solver_rounds)
    phase_lock_max_iterations = config.phase_lock_max_iterations
    balance_anchor_weights = config.balance_anchor_weights
    use_coverage_anchor_selection = config.use_coverage_anchor_selection
    coverage_k_neighbors = config.coverage_k_neighbors
    coverage_candidate_multiplier = max(1, config.coverage_candidate_multiplier)
    use_byte_anchors_for_atlas = config.use_byte_anchors_for_atlas
    alignment_map_for_probe: Any | None = None

    for embed_key in embed_keys:
        embed_start = time.perf_counter()
        use_all_support_texts = config.use_all_support_texts
        accumulated_transform: "object | None" = None
        source_embed = source_weights.get(embed_key)
        target_embed = target_weights.get(embed_key)

        if source_embed is None or target_embed is None:
            logger.warning("Missing embedding for key %s", embed_key)
            continue

        # Vocabulary is the 2D compression plane; dequantize the 1D binary basis first.
        from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

        source_embed = dequantize_if_needed(source_embed, embed_key, source_weights, backend)
        target_embed = dequantize_if_needed(target_embed, embed_key, target_weights, backend)

        # Ensure backend arrays with stable dtype for linear algebra.
        source_embed = backend.array(source_embed)
        target_embed = backend.array(target_embed)
        source_embed = backend.astype(source_embed, "float32")
        target_embed = backend.astype(target_embed, "float32")
        backend.eval(source_embed, target_embed)
        precision_tol = max(alignment_tol, machine_epsilon(backend, source_embed))

        logger.info(
            "Aligning %s: source=%s, target=%s",
            embed_key,
            source_embed.shape,
            target_embed.shape,
        )

        try:
            source_embed = _ensure_vocab_axis(
                source_embed,
                len(source_vocab),
                backend,
                embed_key,
                "source",
            )
            target_embed = _ensure_vocab_axis(
                target_embed,
                len(target_vocab),
                backend,
                embed_key,
                "target",
            )
            backend.eval(source_embed, target_embed)
            source_embed_original = source_embed
            target_embed_original = target_embed
            source_cache_key = _make_embedding_cache_key(source_embed, backend)
            target_cache_key = _make_embedding_cache_key(target_embed, backend)
            source_cache_key_original = source_cache_key
            target_cache_key_original = target_cache_key

            # Binary (1D) alignment: align byte-level anchors before vocabulary blending.
            # Pre-compute byte maps ONCE to avoid repeated Fréchet mean computation.
            binary_metrics = metrics.setdefault("binary_alignment", {})
            binary_signals: list[dict[str, Any]] = []
            binary_start = time.perf_counter()

            target_bytes = _build_byte_embedding_map(
                target_tokenizer,
                target_embed,
                target_embed.shape[0],
                backend,
                cache_key=target_cache_key,
                tokenizer_key=target_tokenizer_key,
            )
            source_bytes = _build_byte_embedding_map(
                source_tokenizer,
                source_embed,
                source_embed.shape[0],
                backend,
                cache_key=source_cache_key,
                tokenizer_key=source_tokenizer_key,
            )
            shared_bytes = sorted(set(source_bytes) & set(target_bytes))

            if len(shared_bytes) < 2:
                raise RuntimeError(
                    "Binary exact kernel alignment failed: "
                    f"only {len(shared_bytes)} shared byte anchors."
                )
            else:
                # Stack anchor matrices ONCE (full set, then coverage-select if needed)
                max_anchor_count = min(len(shared_bytes), int(source_embed.shape[1]))
                source_byte_matrix_full = backend.stack(
                    [source_bytes[b] for b in shared_bytes], axis=0
                )
                target_byte_matrix_full = backend.stack(
                    [target_bytes[b] for b in shared_bytes], axis=0
                )
                backend.eval(source_byte_matrix_full, target_byte_matrix_full)

                coverage_meta: dict[str, float] | None = None
                if len(shared_bytes) > max_anchor_count:
                    # Keep anchors <= feature dim so the system stays full-row-rank for exact solve.
                    if use_coverage_anchor_selection:
                        selected_indices, coverage_meta = _select_coverage_indices(
                            target_byte_matrix_full,
                            max_anchor_count,
                            backend,
                            k_neighbors=coverage_k_neighbors,
                        )
                        shared_bytes = [shared_bytes[idx] for idx in selected_indices]
                        idx_arr = backend.array(selected_indices)
                        source_byte_matrix = backend.take(
                            source_byte_matrix_full, idx_arr, axis=0
                        )
                        target_byte_matrix = backend.take(
                            target_byte_matrix_full, idx_arr, axis=0
                        )
                        backend.eval(source_byte_matrix, target_byte_matrix)
                    elif balance_anchor_weights:
                        shared_bytes = _uniform_subset(shared_bytes, max_anchor_count)
                        source_byte_matrix = backend.stack(
                            [source_bytes[b] for b in shared_bytes], axis=0
                        )
                        target_byte_matrix = backend.stack(
                            [target_bytes[b] for b in shared_bytes], axis=0
                        )
                        backend.eval(source_byte_matrix, target_byte_matrix)
                    else:
                        shared_bytes = shared_bytes[:max_anchor_count]
                        source_byte_matrix = backend.stack(
                            [source_bytes[b] for b in shared_bytes], axis=0
                        )
                        target_byte_matrix = backend.stack(
                            [target_bytes[b] for b in shared_bytes], axis=0
                        )
                        backend.eval(source_byte_matrix, target_byte_matrix)
                else:
                    source_byte_matrix = source_byte_matrix_full
                    target_byte_matrix = target_byte_matrix_full

                # Ensure full-row-rank anchors for exact phase lock.
                rank_indices, rank_meta = _select_shared_full_rank_indices(
                    source_byte_matrix,
                    target_byte_matrix,
                    int(source_byte_matrix.shape[0]),
                    backend,
                    center=False,
                )
                if len(rank_indices) < int(source_byte_matrix.shape[0]):
                    idx_arr = backend.array(rank_indices)
                    source_byte_matrix = backend.take(source_byte_matrix, idx_arr, axis=0)
                    target_byte_matrix = backend.take(target_byte_matrix, idx_arr, axis=0)
                    shared_bytes = [shared_bytes[idx] for idx in rank_indices]
                    backend.eval(source_byte_matrix, target_byte_matrix)

                if coverage_meta is None:
                    coverage_meta = {}
                coverage_meta.update(rank_meta)

                if len(shared_bytes) < 2:
                    raise RuntimeError(
                        "Binary exact kernel alignment failed: "
                        f"rank-deficient anchors ({len(shared_bytes)})."
                    )

                byte_labels = [f"byte:{b}" for b in shared_bytes]

                best_alignment: dict[str, Any] | None = None
                best_cka = -1.0
                last_signal: AlignmentSignal | None = None
                previous_transform: Any | None = None
                anchor_weights: list[float] | None = None
                iteration = 0
                iteration_budget = alignment_iterations
                stall_count = 0

                while True:
                    prev_best = best_cka
                    byte_alignment = _align_bytes_from_matrices(
                        source_embed,
                        source_byte_matrix,
                        target_byte_matrix,
                        byte_labels,
                        backend,
                        max_iterations=solver_iterations,
                        tolerance=precision_tol,
                        max_rounds=solver_rounds,
                        anchor_weights=anchor_weights,
                        initial_transform=previous_transform,
                        require_phase_lock=True,
                    )

                    if byte_alignment["cka_after"] > best_cka:
                        best_cka = byte_alignment["cka_after"]
                        best_alignment = byte_alignment
                        previous_transform = byte_alignment.get("feature_transform")

                    last_signal = alignment_signal_from_matrices(
                        byte_alignment["aligned_matrix"],
                        target_byte_matrix,
                        byte_labels,
                        backend=backend,
                        dimension=1,
                        cka_achieved=byte_alignment["cka_after"],
                        iteration=iteration,
                    )
                    binary_signals.append(last_signal.to_dict())
                    if balance_anchor_weights:
                        anchor_weights = _compute_anchor_weights(last_signal)
                    else:
                        anchor_weights = None

                    if last_signal.is_phase_locked:
                        break

                    improved = best_cka > prev_best + precision_tol
                    if not improved:
                        stall_count += 1
                        if stall_count >= 2:
                            # CKA=1.0 is ALWAYS achievable with full-rank anchors.
                            # If we stall at <1.0, the anchor selection or solve is wrong.
                            raise RuntimeError(
                                f"Binary exact kernel alignment stalled at CKA={best_cka:.6f}. "
                                f"Expected 1.0 with full-rank anchors. "
                                f"Check anchor selection and numerical stability."
                            )
                    else:
                        stall_count = 0

                    iteration += 1
                    if phase_lock_max_iterations > 0 and iteration >= phase_lock_max_iterations:
                        raise RuntimeError(
                            f"Binary exact kernel alignment failed after {iteration} iterations."
                        )
                    if iteration >= iteration_budget:
                        iteration_budget *= 2
                        solver_iterations = int(solver_iterations * 1.5)
                        solver_rounds = max(solver_rounds + 1, solver_rounds)
                        logger.info(
                            "Binary exact kernel alignment not reached; "
                            "expanding search to %d solver iterations",
                            solver_iterations,
                        )

                source_embed = byte_alignment["aligned_source"]
                backend.eval(source_embed)
                if byte_alignment.get("feature_transform") is not None:
                    transform = byte_alignment["feature_transform"]
                    if accumulated_transform is None:
                        accumulated_transform = transform
                    else:
                        accumulated_transform = backend.matmul(accumulated_transform, transform)
                        backend.eval(accumulated_transform)

                if best_alignment is not None:
                    binary_metrics[embed_key] = {
                        "bytes_shared": len(shared_bytes),
                        "cka_before": best_alignment["cka_before"],
                        "cka_after": best_alignment["cka_after"],
                        "alignment_error": best_alignment["alignment_error"],
                        "iterations": best_alignment["iterations"],
                        "source_dim": source_byte_matrix.shape[1],
                        "target_dim": target_byte_matrix.shape[1],
                        "coverage": coverage_meta,
                        "signals": binary_signals,
                        # CKA=1.0 required for exact kernel alignment - no exceptions
                        "phase_locked": bool(last_signal and last_signal.is_phase_locked),
                        "balance_ratio": (
                            last_signal.metadata.get("balance_ratio")
                            if last_signal is not None
                            else None
                        ),
                    }

            metrics["alignment_signals"].setdefault(embed_key, {})["binary"] = binary_signals
            metrics["timing_ms"].setdefault(embed_key, {})[
                "binary_alignment_ms"
            ] = (time.perf_counter() - binary_start) * 1000

            # Refresh cache key after binary alignment (embedding changed).
            source_cache_key = _make_embedding_cache_key(source_embed, backend)

            # Vocabulary token phase-lock (2D): align shared token embeddings directly.
            token_metrics = metrics.setdefault("token_phase_lock", {})
            if overlap:
                shared_tokens = sorted(overlap)
                source_indices = []
                target_indices = []
                for token in shared_tokens:
                    src_idx = source_vocab.get(token)
                    tgt_idx = target_vocab.get(token)
                    if src_idx is None or tgt_idx is None:
                        continue
                    if src_idx >= source_embed.shape[0] or tgt_idx >= target_embed.shape[0]:
                        continue
                    source_indices.append(src_idx)
                    target_indices.append(tgt_idx)

                if len(source_indices) >= 2:
                    max_anchor_count = min(len(source_indices), int(source_embed.shape[1]))
                    source_token_matrix_full = backend.stack(
                        [source_embed[idx] for idx in source_indices], axis=0
                    )
                    target_token_matrix_full = backend.stack(
                        [target_embed[idx] for idx in target_indices], axis=0
                    )
                    backend.eval(source_token_matrix_full, target_token_matrix_full)

                    token_indices, token_rank_meta = _select_shared_full_rank_indices(
                        source_token_matrix_full,
                        target_token_matrix_full,
                        max_anchor_count,
                        backend,
                        center=False,
                    )
                    if len(token_indices) >= 2:
                        idx_arr = backend.array(token_indices)
                        source_token_matrix = backend.take(
                            source_token_matrix_full, idx_arr, axis=0
                        )
                        target_token_matrix = backend.take(
                            target_token_matrix_full, idx_arr, axis=0
                        )
                        backend.eval(source_token_matrix, target_token_matrix)

                        token_labels = [f"token:{shared_tokens[idx]}" for idx in token_indices]
                        token_alignment = _align_bytes_from_matrices(
                            source_embed,
                            source_token_matrix,
                            target_token_matrix,
                            token_labels,
                            backend,
                            max_iterations=solver_iterations,
                            tolerance=precision_tol,
                            max_rounds=solver_rounds,
                            anchor_weights=None,
                            initial_transform=None,
                            require_phase_lock=True,
                        )

                        token_phase_locked = (
                            token_alignment["cka_after"] >= 1.0 - precision_tol
                        )
                        if token_phase_locked:
                            source_embed = token_alignment["aligned_source"]
                            backend.eval(source_embed)
                            source_cache_key = _make_embedding_cache_key(
                                source_embed, backend
                            )
                            if token_alignment.get("feature_transform") is not None:
                                transform = token_alignment["feature_transform"]
                                if accumulated_transform is None:
                                    accumulated_transform = transform
                                else:
                                    accumulated_transform = backend.matmul(
                                        accumulated_transform, transform
                                    )
                                    backend.eval(accumulated_transform)
                        token_metrics[embed_key] = {
                            "tokens_shared": len(source_indices),
                            "anchors_used": len(token_indices),
                            "cka_before": token_alignment["cka_before"],
                            "cka_after": token_alignment["cka_after"],
                            "alignment_error": token_alignment["alignment_error"],
                            "iterations": token_alignment["iterations"],
                            "rank_source": token_rank_meta.get("rank_source"),
                            "rank_target": token_rank_meta.get("rank_target"),
                            "phase_locked": bool(token_phase_locked),
                        }

            target_byte_map_for_atlas: dict[int, Any] = {}
            source_byte_map_for_atlas: dict[int, Any] = {}
            if use_byte_anchors_for_atlas:
                target_byte_map_for_atlas = target_bytes
                # Rebuild byte anchors from the CURRENT source embed; Frechet means are non-linear.
                source_byte_map_for_atlas = _build_byte_embedding_map(
                    source_tokenizer,
                    source_embed,
                    source_embed.shape[0],
                    backend,
                    cache_key=source_cache_key,
                    tokenizer_key=source_tokenizer_key,
                )

            # Vocabulary (2D) alignment: phase-lock on UnifiedAtlas anchors.
            # Pre-compute atlas anchor maps ONCE (may rebuild if use_all_support_texts changes)
            vocab_metrics = metrics.setdefault("vocab_phase_lock", {})
            vocab_signals: list[dict[str, Any]] = []
            vocab_start = time.perf_counter()

            target_atlas_map = _build_atlas_anchor_map(
                target_tokenizer,
                target_embed_original,
                target_embed_original.shape[0],
                backend,
                use_all_support_texts=use_all_support_texts,
                byte_map=target_byte_map_for_atlas,
                use_byte_anchors=use_byte_anchors_for_atlas,
                cache_key=target_cache_key_original,
                tokenizer_key=target_tokenizer_key,
            )
            # Recompute atlas anchors from the CURRENT source embed to preserve exact geometry.
            source_atlas_map = _build_atlas_anchor_map(
                source_tokenizer,
                source_embed,
                source_embed.shape[0],
                backend,
                use_all_support_texts=use_all_support_texts,
                byte_map=source_byte_map_for_atlas,
                use_byte_anchors=use_byte_anchors_for_atlas,
                cache_key=source_cache_key,
                tokenizer_key=source_tokenizer_key,
            )
            shared_atlas = sorted(set(source_atlas_map) & set(target_atlas_map))

            if len(shared_atlas) < 2 and not use_all_support_texts:
                use_all_support_texts = True
                target_atlas_map = _build_atlas_anchor_map(
                    target_tokenizer,
                    target_embed_original,
                    target_embed_original.shape[0],
                    backend,
                    use_all_support_texts=True,
                    byte_map=target_byte_map_for_atlas,
                    use_byte_anchors=use_byte_anchors_for_atlas,
                    cache_key=target_cache_key_original,
                    tokenizer_key=target_tokenizer_key,
                )
                source_atlas_map = _build_atlas_anchor_map(
                    source_tokenizer,
                    source_embed,
                    source_embed.shape[0],
                    backend,
                    use_all_support_texts=True,
                    byte_map=source_byte_map_for_atlas,
                    use_byte_anchors=use_byte_anchors_for_atlas,
                    cache_key=source_cache_key,
                    tokenizer_key=source_tokenizer_key,
                )
                shared_atlas = sorted(set(source_atlas_map) & set(target_atlas_map))

            coverage_meta: dict[str, float] | None = None
            candidate_atlas: list[str] = []
            selected_indices: list[int] = []
            available_indices: list[int] = []
            target_atlas_matrix = None
            source_atlas_matrix = None
            target_atlas_matrix_full = None
            source_atlas_matrix_full = None

            if len(shared_atlas) >= 2:
                max_anchor_count = min(len(shared_atlas), int(source_embed.shape[1]))
                candidate_atlas = shared_atlas
                if len(shared_atlas) > max_anchor_count and balance_anchor_weights:
                    candidate_count = min(
                        len(shared_atlas), max_anchor_count * coverage_candidate_multiplier
                    )
                    candidate_atlas = _balanced_anchor_subset(shared_atlas, candidate_count)

                target_atlas_matrix_full = backend.stack(
                    [target_atlas_map[k] for k in candidate_atlas], axis=0
                )
                source_atlas_matrix_full = backend.stack(
                    [source_atlas_map[k] for k in candidate_atlas], axis=0
                )
                backend.eval(target_atlas_matrix_full, source_atlas_matrix_full)

                if len(candidate_atlas) > max_anchor_count:
                    if use_coverage_anchor_selection:
                        selected_indices, coverage_meta = _select_coverage_indices(
                            target_atlas_matrix_full,
                            max_anchor_count,
                            backend,
                            k_neighbors=coverage_k_neighbors,
                        )
                    else:
                        if balance_anchor_weights:
                            selected_atlas = _balanced_anchor_subset(
                                candidate_atlas, max_anchor_count
                            )
                        else:
                            selected_atlas = candidate_atlas[:max_anchor_count]
                        atlas_index = {anchor: idx for idx, anchor in enumerate(candidate_atlas)}
                        selected_indices = [
                            atlas_index[a] for a in selected_atlas if a in atlas_index
                        ]
                else:
                    selected_indices = list(range(len(candidate_atlas)))

                if selected_indices:
                    idx_arr = backend.array(selected_indices)
                    target_atlas_matrix = backend.take(
                        target_atlas_matrix_full, idx_arr, axis=0
                    )
                    source_atlas_matrix = backend.take(
                        source_atlas_matrix_full, idx_arr, axis=0
                    )
                    backend.eval(target_atlas_matrix, source_atlas_matrix)

                    rank_indices, rank_meta = _select_shared_full_rank_indices(
                        source_atlas_matrix,
                        target_atlas_matrix,
                        int(source_atlas_matrix.shape[0]),
                        backend,
                        center=False,
                    )
                    if len(rank_indices) < int(source_atlas_matrix.shape[0]):
                        selected_indices = [selected_indices[idx] for idx in rank_indices]
                        idx_arr = backend.array(selected_indices)
                        target_atlas_matrix = backend.take(
                            target_atlas_matrix_full, idx_arr, axis=0
                        )
                        source_atlas_matrix = backend.take(
                            source_atlas_matrix_full, idx_arr, axis=0
                        )
                        backend.eval(target_atlas_matrix, source_atlas_matrix)

                    if coverage_meta is None:
                        coverage_meta = {}
                    coverage_meta.update(rank_meta)

                selected_set = set(selected_indices)
                available_indices = [
                    idx for idx in range(len(candidate_atlas)) if idx not in selected_set
                ]

            if (
                target_atlas_matrix is not None
                and source_atlas_matrix is not None
                and len(selected_indices) >= 2
            ):
                shared_atlas = [candidate_atlas[idx] for idx in selected_indices]
                if len(shared_atlas) < 2:
                    raise RuntimeError(
                        "Vocabulary exact kernel alignment failed: "
                        f"rank-deficient anchors ({len(shared_atlas)})."
                    )
            else:
                target_atlas_matrix = None
                source_atlas_matrix = None

            if len(shared_atlas) < 2 or target_atlas_matrix is None:
                raise RuntimeError(
                    "Vocabulary exact kernel alignment failed: "
                    f"only {len(shared_atlas)} shared anchors."
                )
            else:
                atlas_labels = list(shared_atlas)
                best_source = source_embed
                best_alignment: dict[str, Any] | None = None
                best_cka = -1.0
                last_signal: AlignmentSignal | None = None
                previous_transform: Any | None = None
                anchor_weights: list[float] | None = None
                iteration = 0
                iteration_budget = alignment_iterations
                stall_count = 0

                while True:
                    prev_best = best_cka
                    atlas_alignment = _align_bytes_from_matrices(
                        source_embed,
                        source_atlas_matrix,
                        target_atlas_matrix,
                        atlas_labels,
                        backend,
                        max_iterations=solver_iterations,
                        tolerance=precision_tol,
                        max_rounds=solver_rounds,
                        anchor_weights=anchor_weights,
                        initial_transform=previous_transform,
                        require_phase_lock=True,
                    )

                    if atlas_alignment["cka_after"] > best_cka:
                        best_cka = atlas_alignment["cka_after"]
                        best_source = atlas_alignment["aligned_source"]
                        best_alignment = atlas_alignment
                        previous_transform = atlas_alignment.get("feature_transform")

                    last_signal = alignment_signal_from_matrices(
                        atlas_alignment["aligned_matrix"],
                        target_atlas_matrix,
                        atlas_labels,
                        backend=backend,
                        dimension=2,
                        cka_achieved=atlas_alignment["cka_after"],
                        iteration=iteration,
                    )
                    vocab_signals.append(last_signal.to_dict())
                    if balance_anchor_weights:
                        anchor_weights = _compute_anchor_weights(last_signal)
                    else:
                        anchor_weights = None

                    if last_signal.is_phase_locked:
                        break

                    improved = best_cka > prev_best + precision_tol

                    # CKA=1.0 is ALWAYS achievable with full-rank anchors
                    if not improved:
                        stall_count += 1
                        if stall_count >= 2 and not available_indices:
                            raise RuntimeError(
                                f"Vocabulary exact kernel alignment stalled at CKA={best_cka:.6f}. "
                                f"Expected 1.0 with full-rank anchors. "
                                f"Check anchor selection and numerical stability."
                            )
                    else:
                        stall_count = 0

                    if available_indices and target_atlas_matrix_full is not None:
                        refresh_count = min(
                            len(available_indices),
                            max(1, len(shared_atlas) // 20),
                        )
                        label_to_pos = {
                            label: pos for pos, label in enumerate(shared_atlas)
                        }
                        drop_positions: list[int] = []
                        for label in last_signal.misaligned_anchors:
                            pos = label_to_pos.get(label)
                            if pos is not None and pos not in drop_positions:
                                drop_positions.append(pos)
                            if len(drop_positions) >= refresh_count:
                                break

                        if drop_positions:
                            for pos in sorted(drop_positions, reverse=True):
                                selected_indices.pop(pos)
                            replacements = available_indices[: len(drop_positions)]
                            available_indices = available_indices[len(drop_positions) :]
                            selected_indices.extend(replacements)

                            idx_arr = backend.array(selected_indices)
                            target_atlas_matrix = backend.take(
                                target_atlas_matrix_full, idx_arr, axis=0
                            )
                            source_atlas_matrix = backend.take(
                                source_atlas_matrix_full, idx_arr, axis=0
                            )
                            backend.eval(target_atlas_matrix, source_atlas_matrix)

                            rank_indices, rank_meta = _select_shared_full_rank_indices(
                                source_atlas_matrix,
                                target_atlas_matrix,
                                int(source_atlas_matrix.shape[0]),
                                backend,
                                center=False,
                            )
                            if len(rank_indices) < int(source_atlas_matrix.shape[0]):
                                selected_indices = [
                                    selected_indices[idx] for idx in rank_indices
                                ]
                                idx_arr = backend.array(selected_indices)
                                target_atlas_matrix = backend.take(
                                    target_atlas_matrix_full, idx_arr, axis=0
                                )
                                source_atlas_matrix = backend.take(
                                    source_atlas_matrix_full, idx_arr, axis=0
                                )
                                backend.eval(target_atlas_matrix, source_atlas_matrix)

                            shared_atlas = [
                                candidate_atlas[idx] for idx in selected_indices
                            ]
                            atlas_labels = list(shared_atlas)
                            anchor_weights = None

                            if coverage_meta is None:
                                coverage_meta = {}
                            coverage_meta.update(rank_meta)

                            selected_set = set(selected_indices)
                            available_indices = [
                                idx
                                for idx in range(len(candidate_atlas))
                                if idx not in selected_set
                            ]

                    iteration += 1
                    if (
                        phase_lock_max_iterations > 0
                        and iteration >= phase_lock_max_iterations
                    ):
                        raise RuntimeError(
                            f"Vocabulary exact kernel alignment failed after {iteration} iterations."
                        )
                    if iteration >= iteration_budget:
                        iteration_budget *= 2
                        solver_iterations = int(solver_iterations * 1.5)
                        solver_rounds = max(solver_rounds + 1, solver_rounds)
                        if not use_all_support_texts:
                            use_all_support_texts = True
                            target_atlas_map = _build_atlas_anchor_map(
                                target_tokenizer,
                                target_embed_original,
                                target_embed_original.shape[0],
                                backend,
                                use_all_support_texts=True,
                                byte_map=target_byte_map_for_atlas,
                                use_byte_anchors=use_byte_anchors_for_atlas,
                                cache_key=target_cache_key_original,
                                tokenizer_key=target_tokenizer_key,
                            )
                            source_atlas_map = _build_atlas_anchor_map(
                                source_tokenizer,
                                source_embed,
                                source_embed.shape[0],
                                backend,
                                use_all_support_texts=True,
                                byte_map=source_byte_map_for_atlas,
                                use_byte_anchors=use_byte_anchors_for_atlas,
                                cache_key=source_cache_key,
                                tokenizer_key=source_tokenizer_key,
                            )
                            shared_atlas = sorted(
                                set(source_atlas_map) & set(target_atlas_map)
                            )
                            max_anchor_count = min(
                                len(shared_atlas), int(source_embed.shape[1])
                            )
                            candidate_atlas = shared_atlas
                            if (
                                len(shared_atlas) > max_anchor_count
                                and balance_anchor_weights
                            ):
                                candidate_count = min(
                                    len(shared_atlas),
                                    max_anchor_count * coverage_candidate_multiplier,
                                )
                                candidate_atlas = _balanced_anchor_subset(
                                    shared_atlas, candidate_count
                                )

                            target_atlas_matrix_full = backend.stack(
                                [target_atlas_map[k] for k in candidate_atlas], axis=0
                            )
                            source_atlas_matrix_full = backend.stack(
                                [source_atlas_map[k] for k in candidate_atlas], axis=0
                            )
                            backend.eval(
                                target_atlas_matrix_full, source_atlas_matrix_full
                            )

                            if len(candidate_atlas) > max_anchor_count:
                                if use_coverage_anchor_selection:
                                    (
                                        selected_indices,
                                        coverage_meta,
                                    ) = _select_coverage_indices(
                                        target_atlas_matrix_full,
                                        max_anchor_count,
                                        backend,
                                        k_neighbors=coverage_k_neighbors,
                                    )
                                else:
                                    if balance_anchor_weights:
                                        selected_atlas = _balanced_anchor_subset(
                                            candidate_atlas, max_anchor_count
                                        )
                                    else:
                                        selected_atlas = candidate_atlas[:max_anchor_count]
                                    atlas_index = {
                                        anchor: idx
                                        for idx, anchor in enumerate(candidate_atlas)
                                    }
                                    selected_indices = [
                                        atlas_index[a]
                                        for a in selected_atlas
                                        if a in atlas_index
                                    ]
                            else:
                                selected_indices = list(range(len(candidate_atlas)))

                            if selected_indices:
                                idx_arr = backend.array(selected_indices)
                                target_atlas_matrix = backend.take(
                                    target_atlas_matrix_full, idx_arr, axis=0
                                )
                                source_atlas_matrix = backend.take(
                                    source_atlas_matrix_full, idx_arr, axis=0
                                )
                                backend.eval(target_atlas_matrix, source_atlas_matrix)

                                rank_indices, rank_meta = _select_shared_full_rank_indices(
                                    source_atlas_matrix,
                                    target_atlas_matrix,
                                    int(source_atlas_matrix.shape[0]),
                                    backend,
                                    center=False,
                                )
                                if len(rank_indices) < int(source_atlas_matrix.shape[0]):
                                    selected_indices = [
                                        selected_indices[idx] for idx in rank_indices
                                    ]
                                    idx_arr = backend.array(selected_indices)
                                    target_atlas_matrix = backend.take(
                                        target_atlas_matrix_full, idx_arr, axis=0
                                    )
                                    source_atlas_matrix = backend.take(
                                        source_atlas_matrix_full, idx_arr, axis=0
                                    )
                                    backend.eval(target_atlas_matrix, source_atlas_matrix)

                                if coverage_meta is None:
                                    coverage_meta = {}
                                coverage_meta.update(rank_meta)

                            selected_set = set(selected_indices)
                            available_indices = [
                                idx
                                for idx in range(len(candidate_atlas))
                                if idx not in selected_set
                            ]
                            shared_atlas = [
                                candidate_atlas[idx] for idx in selected_indices
                            ]
                            atlas_labels = list(shared_atlas)
                            anchor_weights = None
                        logger.info(
                            "Vocabulary exact kernel alignment not reached; "
                            "expanding search to %d solver iterations",
                            solver_iterations,
                        )

                source_embed = best_source
                backend.eval(source_embed)

                if best_alignment is not None:
                    vocab_metrics[embed_key] = {
                        "anchors_shared": len(shared_atlas),
                        "cka_before": best_alignment["cka_before"],
                        "cka_after": best_alignment["cka_after"],
                        "alignment_error": best_alignment["alignment_error"],
                        "iterations": best_alignment["iterations"],
                        "signals": vocab_signals,
                        # CKA=1.0 required for exact kernel alignment - no exceptions
                        "phase_locked": bool(last_signal and last_signal.is_phase_locked),
                        "support_texts": "all" if use_all_support_texts else "first",
                        "coverage": coverage_meta,
                        "balance_ratio": (
                            last_signal.metadata.get("balance_ratio")
                            if last_signal is not None
                            else None
                        ),
                    }
                else:
                    vocab_metrics[embed_key] = {
                        "anchors_shared": 0,
                        "cka_before": 0.0,
                        "cka_after": 0.0,
                        "alignment_error": 0.0,
                        "iterations": 0,
                        "signals": vocab_signals,
                        "phase_locked": False,
                        "support_texts": "all" if use_all_support_texts else "first",
                        "balance_ratio": None,
                    }

            metrics["alignment_signals"].setdefault(embed_key, {})["vocab"] = vocab_signals
            metrics["timing_ms"].setdefault(embed_key, {})[
                "vocab_alignment_ms"
            ] = (time.perf_counter() - vocab_start) * 1000

            token_phase_locked = True
            if overlap:
                token_phase_locked = bool(
                    token_metrics.get(embed_key, {}).get("phase_locked")
                )

            phase_locked = (
                bool(binary_metrics.get(embed_key, {}).get("phase_locked"))
                and bool(vocab_metrics.get(embed_key, {}).get("phase_locked"))
                and token_phase_locked
            )
            if not phase_locked:
                raise RuntimeError(
                    "Vocabulary alignment did not reach exact kernel alignment "
                    "across dimensions. "
                    f"binary={bool(binary_metrics.get(embed_key, {}).get('phase_locked'))}, "
                    f"token={token_phase_locked}, "
                    f"atlas={bool(vocab_metrics.get(embed_key, {}).get('phase_locked'))}."
                )

            strict_token_alignment = bool(config.strict_token_alignment and phase_locked)
            effective_strategy = projection_strategy
            if phase_locked:
                if source_embed.shape[1] != target_embed.shape[1]:
                    logger.warning(
                        "Exact kernel alignment produced mismatched embedding dims: "
                        "source=%s, target=%s. Continuing with aligned embeddings.",
                        source_embed.shape[1],
                        target_embed.shape[1],
                    )
                effective_strategy = ProjectionStrategy.TRUNCATE

            merge_blend_alpha = config.blend_alpha
            merge_use_embedding_similarity = config.use_embedding_similarity
            merge_max_prefix_length = config.max_prefix_length
            merge_max_prefix_matches = config.max_prefix_matches
            if strict_token_alignment:
                merge_blend_alpha = 1.0
                merge_use_embedding_similarity = False
                merge_max_prefix_length = 0
                merge_max_prefix_matches = 0

            # Derive thresholds from embedding data if not provided
            effective_similarity = config.similarity_threshold
            effective_confidence = config.confidence_threshold

            if effective_similarity is None or effective_confidence is None:
                # Sample embedding similarities to derive thresholds via spectral gap
                sample_sims = _sample_embedding_similarities(
                    source_embed, target_embed, backend, sample_size=min(500, source_embed.shape[0])
                )
                if sample_sims:
                    derived = _derive_thresholds_from_similarities(sample_sims)
                    if effective_similarity is None:
                        effective_similarity = derived["similarity_threshold"]
                    if effective_confidence is None:
                        effective_confidence = derived["confidence_threshold"]
                else:
                    # No similarities computed - use dtype-derived minimum
                    eps = machine_epsilon(backend, source_embed)
                    if effective_similarity is None:
                        effective_similarity = eps
                    if effective_confidence is None:
                        effective_confidence = eps

            # For blend_alpha, if None, compute from CKA alignment quality
            # Higher source CKA → higher alpha (more weight to source)
            if merge_blend_alpha is None:
                source_cka_score = compute_cka(source_embed, source_embed, backend=backend).cka
                target_cka_score = compute_cka(target_embed, target_embed, backend=backend).cka
                total = source_cka_score + target_cka_score
                if total > 0:
                    merge_blend_alpha = source_cka_score / total
                else:
                    merge_blend_alpha = 0.5  # Equal self-CKA means equal weight

            merge_config = CrossVocabMergeConfig(
                projection_strategy=effective_strategy,
                similarity_threshold=effective_similarity,
                confidence_threshold=effective_confidence,
                blend_alpha=merge_blend_alpha,
                preserve_special_tokens=config.preserve_special_tokens,
                use_embedding_similarity=merge_use_embedding_similarity,
                exact_match_only=strict_token_alignment,
                anchor_count=config.anchor_count,
                max_similarity_pairs=config.max_similarity_pairs,
                max_unmapped_similarity=config.max_unmapped_similarity,
                max_prefix_length=merge_max_prefix_length,
                max_prefix_matches=merge_max_prefix_matches,
                similarity_batch_size=config.similarity_batch_size,
            )
            merger = CrossVocabMerger(merge_config)

            # Run CrossVocabMerger
            merge_start = time.perf_counter()
            result = merger.merge(
                source_embeddings=source_embed,
                target_embeddings=target_embed,
                source_vocab=source_vocab,
                target_vocab=target_vocab,
            )
            if alignment_map_for_probe is None:
                alignment_map_for_probe = result.alignment_map
            metrics["timing_ms"].setdefault(embed_key, {})[
                "cross_vocab_merge_ms"
            ] = (time.perf_counter() - merge_start) * 1000

            # Check quality
            quality_metrics = merger.analyze_merge_quality(result)

            # Warn on low compatibility/coverage only if thresholds were specified
            # If None, no arbitrary floor was set - we accept any measurable value
            if (
                config.min_compatibility_score is not None
                and result.compatibility.compatibility_score < config.min_compatibility_score
            ):
                logger.warning(
                    "Low compatibility score %.2f for %s (continuing alignment)",
                    result.compatibility.compatibility_score,
                    embed_key,
                )
                metrics[f"{embed_key}_warning"] = "low_compatibility"

            if (
                config.min_coverage is not None
                and result.alignment_map.coverage < config.min_coverage
            ):
                logger.warning(
                    "Low coverage %.2f for %s (continuing alignment)",
                    result.alignment_map.coverage,
                    embed_key,
                )
                metrics[f"{embed_key}_warning"] = "low_coverage"

            # Convert result to backend array format, preserving original dtype
            merged_embed = result.merged_embeddings

            # Ensure we have a backend array
            if hasattr(merged_embed, "numpy"):
                # PyTorch or TensorFlow tensor - convert via numpy
                merged_np = merged_embed.numpy()
                merged_embed = backend.array(merged_np)
            elif not hasattr(merged_embed, "shape") or not hasattr(merged_embed, "dtype"):
                # Raw python data - convert to backend array
                merged_embed = backend.array(merged_embed)

            # Keep float32 to preserve the aligned vocabulary plane; requantize at final save.
            merged_embed = backend.astype(merged_embed, "float32")
            backend.eval(merged_embed)
            # Keep on CPU to reduce GPU pressure; will be moved to backend on demand.
            modified_weights[embed_key] = backend.to_numpy(merged_embed)
            aligned_layers += 1

            # Record metrics
            metrics[f"{embed_key}_projection_strategy"] = effective_strategy.value
            metrics[f"{embed_key}_alignment_coverage"] = result.alignment_map.coverage
            metrics[f"{embed_key}_alignment_confidence"] = result.alignment_map.mean_confidence
            metrics[f"{embed_key}_projection_score"] = result.projection_result.alignment_score
            metrics[f"{embed_key}_compatibility_score"] = result.compatibility.compatibility_score
            metrics[f"{embed_key}_overall_quality"] = quality_metrics["overall_quality_score"]
            metrics[f"{embed_key}_warnings"] = result.warnings
            metrics[f"{embed_key}_strict_token_alignment"] = strict_token_alignment

            logger.info(
                "Aligned %s: coverage=%.2f, quality=%.2f",
                embed_key,
                result.alignment_map.coverage,
                quality_metrics["overall_quality_score"],
            )
            metrics["timing_ms"].setdefault(embed_key, {})[
                "total_ms"
            ] = (time.perf_counter() - embed_start) * 1000

        except Exception as e:
            logger.error("Failed to align %s: %s", embed_key, e)
            metrics[f"{embed_key}_error"] = str(e)
            raise

    metrics["aligned_layers"] = aligned_layers
    metrics["alignment_applied"] = aligned_layers > 0
    metrics["timing_ms"]["stage_total_ms"] = (time.perf_counter() - stage_start) * 1000

    cache_stats = cache.get_stats()
    metrics["cache_stats"] = {
        "hits": cache_stats.hits,
        "misses": cache_stats.misses,
        "evictions": cache_stats.evictions,
        "hit_rate": cache_stats.hit_rate,
        "saved_ms": cache_stats.total_compute_time_saved_ms,
        "sizes": cache.get_cache_sizes(),
    }

    if aligned_layers > 0:
        logger.info("Vocabulary alignment applied to %d layers", aligned_layers)
    else:
        logger.info("No vocabulary alignment applied")

    return VocabularyResult(
        modified_weights,
        metrics,
        aligned_layers > 0,
        alignment_map_for_probe,
    )


def _extract_vocab(tokenizer: Any) -> dict[str, int] | None:
    """Extract vocabulary mapping from tokenizer."""
    # Try different tokenizer APIs
    if hasattr(tokenizer, "get_vocab"):
        return tokenizer.get_vocab()
    if hasattr(tokenizer, "vocab"):
        vocab = tokenizer.vocab
        if isinstance(vocab, dict):
            return vocab
    if hasattr(tokenizer, "encoder"):
        return tokenizer.encoder
    if hasattr(tokenizer, "token_to_id"):
        # Tokenizers library - need to iterate
        try:
            vocab = {}
            for token in tokenizer.get_vocab():
                vocab[token] = tokenizer.token_to_id(token)
            return vocab
        except Exception:
            pass

    logger.warning("Could not extract vocabulary from tokenizer type %s", type(tokenizer))
    return None


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)

    if isinstance(encoded, list):
        return encoded
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return []


def _encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _ensure_vocab_axis(
    embedding: "object",
    vocab_size: int,
    backend: "object",
    embed_key: str,
    label: str,
) -> "object":
    if embedding.ndim != 2:
        logger.warning("Embedding %s for %s is not 2D (shape=%s)", embed_key, label, embedding.shape)
        return embedding
    if embedding.shape[0] == vocab_size:
        return embedding
    if embedding.shape[1] == vocab_size:
        logger.info("Transposing %s embedding for %s to match vocab axis", embed_key, label)
        return backend.transpose(embedding)
    logger.warning(
        "Embedding %s for %s does not match vocab size (shape=%s, vocab=%d)",
        embed_key,
        label,
        embedding.shape,
        vocab_size,
    )
    return embedding


def _sample_embedding_similarities(
    source_embed: "object",
    target_embed: "object",
    backend: "object",
    sample_size: int = 500,
) -> list[float]:
    """Sample cosine similarities between source and target embeddings.

    Returns a list of similarity values that can be used to derive
    thresholds via spectral gap detection.
    """
    import math as _math

    n_source = source_embed.shape[0]
    n_target = target_embed.shape[0]

    if n_source == 0 or n_target == 0:
        return []

    # Sample indices uniformly
    sample_size = min(sample_size, n_source, n_target)
    step_source = max(1, n_source // sample_size)
    step_target = max(1, n_target // sample_size)

    source_indices = list(range(0, n_source, step_source))[:sample_size]
    target_indices = list(range(0, n_target, step_target))[:sample_size]

    if not source_indices or not target_indices:
        return []

    # Get sample embeddings
    source_sample = backend.take(source_embed, backend.array(source_indices), axis=0)
    target_sample = backend.take(target_embed, backend.array(target_indices), axis=0)

    # Normalize for cosine similarity
    eps = machine_epsilon(backend, source_sample)
    source_norms = backend.norm(source_sample, axis=1, keepdims=True)
    target_norms = backend.norm(target_sample, axis=1, keepdims=True)

    source_normed = source_sample / (source_norms + eps)
    target_normed = target_sample / (target_norms + eps)

    # Compute pairwise cosine similarities (sample × sample matrix)
    sim_matrix = backend.matmul(source_normed, backend.transpose(target_normed))
    backend.eval(sim_matrix)

    # Flatten and convert to list
    sim_flat = backend.reshape(sim_matrix, (-1,))
    return list(backend.to_numpy(sim_flat).tolist())


def _derive_thresholds_from_similarities(
    similarities: list[float],
) -> dict[str, float]:
    """Derive similarity and confidence thresholds from spectral gap.

    Finds natural boundaries in the similarity distribution using
    the largest gap between consecutive sorted values.

    Returns dict with 'similarity_threshold' and 'confidence_threshold'.
    """
    import math as _math

    if len(similarities) < 2:
        return {"similarity_threshold": 0.0, "confidence_threshold": 0.0}

    sorted_sims = sorted(similarities)

    # Compute gaps between consecutive values
    gaps = [sorted_sims[i + 1] - sorted_sims[i] for i in range(len(sorted_sims) - 1)]

    if not gaps:
        return {"similarity_threshold": 0.0, "confidence_threshold": 0.0}

    # Find the gap that separates "low" from "high" similarity
    # Use mean + 2*stddev as significance threshold
    mean_gap = sum(gaps) / len(gaps)
    if len(gaps) > 1:
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        stddev = _math.sqrt(variance)
        significance_threshold = mean_gap + 2.0 * stddev
    else:
        significance_threshold = 0.0

    # Find largest significant gap
    max_gap = max(gaps)
    max_gap_idx = gaps.index(max_gap)

    if max_gap <= significance_threshold:
        # No significant gap - use median as threshold
        mid_idx = len(sorted_sims) // 2
        threshold = sorted_sims[mid_idx]
    else:
        # Threshold at geometric mean of values around the gap
        lower_val = sorted_sims[max_gap_idx]
        upper_val = sorted_sims[max_gap_idx + 1]
        if lower_val <= 0:
            threshold = upper_val / 2.0
        else:
            threshold = _math.sqrt(lower_val * upper_val)

    # similarity_threshold is for "high quality" alignment
    # confidence_threshold is for "acceptable" alignment
    # Use spectral gap for similarity, and half that for confidence
    return {
        "similarity_threshold": threshold,
        "confidence_threshold": threshold / 2.0 if threshold > 0 else 0.0,
    }


def _frechet_mean_from_ids(
    token_ids: list[int],
    embedding: "object",
    backend: "object",
) -> "object | None":
    if not token_ids:
        return None
    if len(token_ids) == 1:
        return embedding[token_ids[0]]
    idx = backend.array(token_ids)
    vectors = backend.take(embedding, idx, axis=0)
    return _frechet_mean_vectors(vectors, backend)


def _frechet_mean_from_bytes(
    byte_values: list[int],
    byte_map: dict[int, "object"],
    backend: "object",
) -> "object | None":
    if not byte_values:
        return None
    vectors = [byte_map[b] for b in byte_values if b in byte_map]
    if not vectors:
        return None
    if len(vectors) == 1:
        return vectors[0]
    stacked = backend.stack(vectors, axis=0)
    return _frechet_mean_vectors(stacked, backend)


def _frechet_mean_vectors(
    vectors: "object",
    backend: "object",
) -> "object":
    """Compute Fréchet mean of vectors with full geodesic precision.

    For byte/vocabulary anchors, we need EXACT alignment - no approximations.
    """
    n_vectors = vectors.shape[0]
    if n_vectors <= 1:
        return vectors[0]

    # Check if vectors are identical (exact match, no computation needed)
    diff = vectors - vectors[:1]
    diff_norm = backend.norm(diff, axis=1)
    max_norm = backend.max(diff_norm)
    backend.eval(max_norm)
    if float(max_norm) < 1e-12:
        return vectors[0]

    k_neighbors = max(1, n_vectors - 1)
    mean = frechet_mean(
        vectors,
        backend=backend,
        k_neighbors=k_neighbors,
        max_k_neighbors=k_neighbors,
    )
    backend.eval(mean)
    return mean


def _apply_alignment_correction(
    embedding: "object",
    signal: AlignmentSignal | None,
    backend: "object",
) -> "object":
    if signal is None:
        return embedding

    transform = signal.suggested_transformation
    if transform == "scale_normalization":
        scale_ratio = signal.metadata.get("scale_ratio", 1.0)
        if scale_ratio > 0:
            scaled = embedding / float(scale_ratio)
            backend.eval(scaled)
            return scaled
        return embedding

    if transform == "rotation_refine":
        mean = backend.mean(embedding, axis=0, keepdims=True)
        centered = embedding - mean
        norms = backend.norm(centered, axis=1, keepdims=True)
        normalized = centered / (norms + 1e-12)
        backend.eval(normalized)
        return normalized

    return embedding


def _compute_anchor_weights(signal: AlignmentSignal | None) -> list[float] | None:
    if signal is None:
        return None
    divergences = signal.anchor_divergence
    if not divergences:
        return None
    mean_div = signal.metadata.get("mean_divergence", 0.0)
    if mean_div <= 0.0:
        return [1.0 for _ in divergences]

    weights = [float(mean_div) / (float(d) + 1e-12) for d in divergences]
    mean_weight = sum(weights) / len(weights) if weights else 1.0
    if mean_weight > 0:
        weights = [w / mean_weight for w in weights]

    balance_ratio = max(1.0, float(signal.metadata.get("balance_ratio", 1.0)))
    min_weight = 1.0 / balance_ratio
    max_weight = balance_ratio
    weights = [min(max(w, min_weight), max_weight) for w in weights]
    return weights


def _apply_anchor_weights(
    matrix: "object",
    anchor_weights: list[float] | None,
    backend: "object",
) -> "object":
    if anchor_weights is None:
        return matrix
    if matrix.shape[0] != len(anchor_weights):
        return matrix
    weights = backend.array(anchor_weights)
    weights = backend.reshape(weights, (-1, 1))
    scaled = matrix * backend.sqrt(weights)
    backend.eval(scaled)
    return scaled


def _uniform_subset(values: list[int], max_count: int) -> list[int]:
    if max_count <= 0:
        return []
    if len(values) <= max_count:
        return values
    step = len(values) / float(max_count)
    selected = []
    for idx in range(max_count):
        pos = int(idx * step)
        selected.append(values[min(pos, len(values) - 1)])
    return selected


def _select_coverage_indices(
    points: "object",
    max_count: int,
    backend: "object",
    k_neighbors: int | None = None,
) -> tuple[list[int], dict[str, float]]:
    n = int(points.shape[0])
    if max_count <= 0 or n <= max_count:
        return list(range(n)), {"coverage_applied": 0.0}

    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    k_neighbors = k_neighbors if k_neighbors is not None else min(10, n - 1)
    k_neighbors = max(1, min(int(k_neighbors), n - 1))

    # Seed with the largest-norm anchor to maximize initial coverage.
    norms = backend.norm(points, axis=1)
    backend.eval(norms)
    seed_idx = int(backend.to_numpy(backend.argmax(norms)))

    rg = RiemannianGeometry(backend)
    fps_result = rg.farthest_point_sampling(
        points,
        n_samples=max_count,
        seed_idx=seed_idx,
        k_neighbors=k_neighbors,
    )

    return fps_result.selected_indices, {
        "coverage_applied": 1.0,
        "coverage_radius": float(fps_result.coverage_radius),
        "k_neighbors": float(k_neighbors),
    }


def _select_full_rank_indices(
    points: "object",
    max_count: int,
    backend: "object",
) -> tuple[list[int], dict[str, float]]:
    mean = backend.mean(points, axis=0, keepdims=True)
    centered = points - mean
    backend.eval(centered)
    n = int(points.shape[0])
    if max_count <= 0 or n == 0:
        return [], {"rank": 0.0, "selected_count": 0.0}
    if n <= max_count:
        rank = _matrix_rank_for_alignment(centered, backend)
        return list(range(n)), {"rank": float(rank), "selected_count": float(n)}

    norms = backend.norm(centered, axis=1)
    backend.eval(norms)
    norm_list = backend.to_numpy(norms).tolist()
    ranked = sorted(range(n), key=lambda idx: norm_list[idx], reverse=True)

    eps = max(machine_epsilon(backend, points), 1e-12)
    selected: list[int] = []
    basis: list["object"] = []

    for idx in ranked:
        vec = centered[idx]
        if basis:
            basis_matrix = backend.stack(basis, axis=0)
            vec_col = backend.reshape(vec, (-1, 1))
            proj_coeffs = backend.matmul(basis_matrix, vec_col)
            proj = backend.matmul(backend.transpose(basis_matrix), proj_coeffs)
            residual = vec_col - proj
            res_norm = backend.norm(residual)
            backend.eval(res_norm)
            if float(backend.to_numpy(res_norm)) <= eps:
                continue
            vec = backend.reshape(residual / res_norm, (-1,))
        else:
            res_norm = backend.norm(vec)
            backend.eval(res_norm)
            if float(backend.to_numpy(res_norm)) <= eps:
                continue
            vec = vec / res_norm

        basis.append(vec)
        selected.append(idx)
        if len(selected) >= max_count:
            break

    return selected, {"rank": float(len(basis)), "selected_count": float(len(selected))}


def _select_shared_full_rank_indices(
    source_points: "object",
    target_points: "object",
    max_count: int,
    backend: "object",
    *,
    center: bool = True,
) -> tuple[list[int], dict[str, float]]:
    """Select indices where BOTH source and target are linearly independent.

    CRITICAL: Always filter for linear independence. CKA=1.0 requires full-rank
    matrices for the closed-form solve F = A_s^T (A_s A_s^T)^-1 A_t.
    If we pass rank-deficient matrices, the support-space inverse loses information
    and CKA < 1.0.
    """
    source_data = source_points
    target_data = target_points
    if center:
        source_data = source_points - backend.mean(source_points, axis=0, keepdims=True)
        target_data = target_points - backend.mean(target_points, axis=0, keepdims=True)
    backend.eval(source_data, target_data)
    n = int(source_points.shape[0])
    if max_count <= 0 or n == 0:
        return [], {"rank_source": 0.0, "rank_target": 0.0, "selected_count": 0.0}

    # ALWAYS filter for linear independence - never skip this step!
    # The old code had: if n <= max_count: return all indices
    # This was WRONG - it passed rank-deficient matrices to GramAligner

    combined = backend.concatenate([source_data, target_data], axis=1)
    norms = backend.norm(combined, axis=1)
    backend.eval(norms)
    norm_list = backend.to_numpy(norms).tolist()
    ranked = sorted(range(n), key=lambda idx: norm_list[idx], reverse=True)

    eps = max(machine_epsilon(backend, combined) * 1e3, 1e-4)

    def _orthonormalize(
        vec: "object",
        basis: list["object"],
    ) -> tuple[bool, "object"]:
        if not basis:
            res_norm = backend.norm(vec)
            backend.eval(res_norm)
            if float(backend.to_numpy(res_norm)) <= eps:
                return False, vec
            return True, vec / res_norm

        basis_matrix = backend.stack(basis, axis=0)
        vec_col = backend.reshape(vec, (-1, 1))
        proj_coeffs = backend.matmul(basis_matrix, vec_col)
        proj = backend.matmul(backend.transpose(basis_matrix), proj_coeffs)
        residual = vec_col - proj
        res_norm = backend.norm(residual)
        backend.eval(res_norm)
        if float(backend.to_numpy(res_norm)) <= eps:
            return False, vec
        return True, backend.reshape(residual / res_norm, (-1,))

    selected: list[int] = []
    basis_src: list["object"] = []
    basis_tgt: list["object"] = []

    for idx in ranked:
        vec_src = source_data[idx]
        vec_tgt = target_data[idx]
        ok_src, norm_src = _orthonormalize(vec_src, basis_src)
        ok_tgt, norm_tgt = _orthonormalize(vec_tgt, basis_tgt)
        if not (ok_src and ok_tgt):
            continue

        basis_src.append(norm_src)
        basis_tgt.append(norm_tgt)
        selected.append(idx)
        if len(selected) >= max_count:
            break

    def _condition_number(matrix: "object") -> float:
        gram = backend.matmul(matrix, backend.transpose(matrix))
        eigvals, _ = backend.eigh(gram)
        backend.eval(eigvals)
        values = [float(v) for v in backend.to_numpy(eigvals).tolist() if float(v) > eps]
        if not values:
            return float("inf")
        return (max(values) / min(values)) ** 0.5

    max_condition = _dynamic_condition_threshold(combined, backend)
    while selected and len(selected) > 2:
        idx_arr = backend.array(selected)
        src_sel = backend.take(source_points, idx_arr, axis=0)
        tgt_sel = backend.take(target_points, idx_arr, axis=0)
        backend.eval(src_sel, tgt_sel)
        cond_src = _condition_number(src_sel)
        cond_tgt = _condition_number(tgt_sel)
        if cond_src <= max_condition and cond_tgt <= max_condition:
            break
        drop_count = max(1, len(selected) // 10)
        selected = selected[:-drop_count]

    # CRITICAL: Verify numerical rank via SVD and drop anchors until rank = n_selected
    # Use dtype-derived threshold: max_s * max(m,n) * machine_epsilon
    def _numerical_rank_svd(matrix: "object") -> int:
        """Compute numerical rank via SVD with dtype-derived threshold."""
        try:
            _, s, _ = backend.svd(matrix, full_matrices=False)
            backend.eval(s)
            s_vals = backend.to_numpy(s)
            if len(s_vals) == 0:
                return 0
            max_s = float(s_vals[0])
            eps_matrix = machine_epsilon(backend, matrix)
            if max_s < eps_matrix:
                return 0
            # Standard numerical rank threshold: max_s * max(m,n) * eps
            max_dim = max(matrix.shape[0], matrix.shape[1])
            threshold = max_s * max_dim * eps_matrix
            return int(sum(1 for sv in s_vals if float(sv) > threshold))
        except Exception:
            return len(selected)  # Fallback to full count if SVD fails

    # Drop anchors until BOTH matrices have numerical rank >= n_selected
    while selected and len(selected) > 2:
        idx_arr = backend.array(selected)
        src_sel = backend.take(source_points, idx_arr, axis=0)
        tgt_sel = backend.take(target_points, idx_arr, axis=0)
        backend.eval(src_sel, tgt_sel)

        rank_src = _numerical_rank_svd(src_sel)
        rank_tgt = _numerical_rank_svd(tgt_sel)
        min_rank = min(rank_src, rank_tgt)

        if min_rank >= len(selected):
            # Both matrices are full-rank for the closed-form solve
            break

        # Drop to match numerical rank (cannot exceed actual rank)
        if min_rank < len(selected):
            # Drop extra anchors - keep only min_rank indices
            drop_to = max(2, min_rank)
            if drop_to < len(selected):
                selected = selected[:drop_to]
        else:
            break

    cond_src = float("inf")
    cond_tgt = float("inf")
    rank_src_final = 0
    rank_tgt_final = 0
    if selected:
        idx_arr = backend.array(selected)
        src_sel = backend.take(source_points, idx_arr, axis=0)
        tgt_sel = backend.take(target_points, idx_arr, axis=0)
        backend.eval(src_sel, tgt_sel)
        cond_src = float(_condition_number(src_sel))
        cond_tgt = float(_condition_number(tgt_sel))
        rank_src_final = _numerical_rank_svd(src_sel)
        rank_tgt_final = _numerical_rank_svd(tgt_sel)

    return selected, {
        "rank_source": float(len(basis_src)),
        "rank_target": float(len(basis_tgt)),
        "selected_count": float(len(selected)),
        "cond_source": cond_src,
        "cond_target": cond_tgt,
        "svd_rank_source": float(rank_src_final),
        "svd_rank_target": float(rank_tgt_final),
    }


def _balanced_anchor_subset(anchor_ids: list[str], max_count: int) -> list[str]:
    if max_count <= 0:
        return []
    if len(anchor_ids) <= max_count:
        return anchor_ids

    buckets: dict[str, list[str]] = {}
    order: list[str] = []
    for anchor_id in anchor_ids:
        probe_id = anchor_id.split(":", 1)[0]
        if probe_id not in buckets:
            buckets[probe_id] = []
            order.append(probe_id)
        buckets[probe_id].append(anchor_id)

    selected: list[str] = []
    round_idx = 0
    while len(selected) < max_count:
        progressed = False
        for probe_id in order:
            bucket = buckets[probe_id]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                progressed = True
                if len(selected) >= max_count:
                    break
        if not progressed:
            break
        round_idx += 1

    return selected


def _matrix_rank_for_alignment(
    matrix: "object",
    backend: "object",
    eps: float | None = None,
) -> int:
    """Compute effective rank using dtype-derived threshold."""
    n = int(matrix.shape[0])
    if n == 0:
        return 0

    gram = backend.matmul(matrix, backend.transpose(matrix))
    eigvals, _ = backend.eigh(gram)
    backend.eval(eigvals)
    values = list(backend.to_numpy(eigvals).tolist())
    if not values:
        return 0
    max_val = max(values)
    if eps is None:
        eps = machine_epsilon(backend, matrix)
    threshold = max_val * eps
    return sum(1 for val in values if val > threshold)


def _dynamic_condition_threshold(
    matrix: "object",
    backend: "object",
) -> float:
    """Compute condition number threshold from dtype.

    The condition number threshold is 1/sqrt(eps), which represents
    the boundary where numerical operations become unreliable.
    No arbitrary cap - the dtype determines the threshold.
    """
    eps = machine_epsilon(backend, matrix)
    # 1/sqrt(eps) is the natural condition number threshold
    # Beyond this, matrix operations lose significant precision
    return 1.0 / (eps ** 0.5)


def _solve_feature_transform_exact(
    source_matrix: "object",
    target_matrix: "object",
    backend: "object",
    regularization: float = 0.0,
) -> "object | None":
    """Solve source @ F = target for exact alignment.

    Strategy (in order of preference):
    1. QR-based solve for full-rank, well-conditioned cases (fastest)
    2. Rank-truncated spectral inverse (exact on the support space)
    3. Gram alignment in sample space (relational geometry)
    4. Eigendecomposition in sample space (legacy, squares condition number)

    The caller verifies exact kernel alignment (CKA = 1.0). This function only
    proposes candidate transforms; it does not accept approximate alignment.
    """
    from modelcypher.core.domain.geometry.numerical_stability import (
        solve_full_row_rank_via_qr,
        solve_via_gram_alignment,
        solve_via_truncated_svd,
    )

    n = int(source_matrix.shape[0])
    if n == 0:
        return None

    eps = max(machine_epsilon(backend, source_matrix), 1e-12)

    # Try QR-based solve first (fast for full-rank, well-conditioned cases)
    F_qr, diag = solve_full_row_rank_via_qr(backend, source_matrix, target_matrix)

    best_transform: "object | None" = None
    best_residual = float("inf")
    best_label = "none"

    if F_qr is not None:
        logger.debug(
            "QR solve: method=%s, rank=%d/%d, cond=%.2e, residual=%.2e",
            diag.get("method", "unknown"),
            diag.get("rank", 0),
            n,
            diag.get("condition", float("inf")),
            diag.get("residual_norm", float("inf")),
        )
        # Accept if residual is small (system is full-rank and well-conditioned)
        qr_residual = float(diag.get("residual_norm", float("inf")))
        if qr_residual < best_residual:
            best_transform = F_qr
            best_residual = qr_residual
            best_label = "qr"
        if qr_residual < eps * 1000:
            return F_qr

    # QR residual too large - try rank-truncated spectral inverse (handles rank-deficiency)
    logger.debug(
        "QR residual too large (%.2e), trying spectral inverse",
        diag.get("residual_norm", float("inf")) if F_qr is not None else float("inf"),
    )

    F_svd, diag_svd = solve_via_truncated_svd(backend, source_matrix, target_matrix)

    if F_svd is not None:
        logger.debug(
            "Spectral inverse: rank=%d/%d, cond=%.2e, proj_err=%.2e, residual=%.2e",
            diag_svd.get("rank", 0),
            n,
            diag_svd.get("condition", float("inf")),
            diag_svd.get("projection_error", float("inf")),
            diag_svd.get("residual_norm", float("inf")),
        )
        svd_residual = float(diag_svd.get("residual_norm", float("inf")))
        if svd_residual < best_residual:
            best_transform = F_svd
            best_residual = svd_residual
            best_label = "spectral_inverse"
        # Exact alignment only if residual is at precision scale.
        if svd_residual < eps * 100:
            return F_svd

    # Spectral inverse insufficient - try Gram alignment in sample space.
    logger.debug(
        "Spectral inverse residual too large (%.2e), trying Gram alignment",
        diag_svd.get("residual_norm", float("inf")) if F_svd is not None else float("inf"),
    )

    # Gram alignment: align sample structure (relational geometry).
    F_gram, diag_gram = solve_via_gram_alignment(backend, source_matrix, target_matrix)

    if F_gram is not None:
        logger.debug(
            "Gram alignment: rank_s=%d, rank_t=%d, procrustes_err=%.2e",
            diag_gram.get("rank_source", 0),
            diag_gram.get("rank_target", 0),
            diag_gram.get("procrustes_error", float("inf")),
        )
        aligned_gram = backend.matmul(source_matrix, F_gram)
        backend.eval(aligned_gram)
        residual = aligned_gram - target_matrix
        res_norm = backend.norm(residual)
        tgt_norm = backend.norm(target_matrix)
        backend.eval(res_norm, tgt_norm)
        gram_residual = float(backend.to_numpy(res_norm)) / (
            float(backend.to_numpy(tgt_norm)) + eps
        )
        if gram_residual < best_residual:
            best_transform = F_gram
            best_residual = gram_residual
            best_label = "gram_alignment"
        if gram_residual < eps * 100:
            return F_gram

    # Fall back to eigendecomposition (legacy, squares condition number)
    if best_transform is not None:
        logger.debug("Returning best candidate transform (%s) with residual %.2e",
                     best_label, best_residual)
        return best_transform

    logger.debug("All direct methods failed, trying eigen solve")

    gram = backend.matmul(source_matrix, backend.transpose(source_matrix))
    if regularization > 0.0:
        gram = gram + regularization * backend.eye(n)
    backend.eval(gram)

    eigvals, eigvecs = backend.eigh(gram)
    backend.eval(eigvals, eigvecs)
    values = [float(v) for v in backend.to_numpy(eigvals).tolist()]
    if not values:
        return None
    min_eig = min(values)
    max_eig = max(values)
    if max_eig <= eps:
        return None

    eigvals = backend.maximum(eigvals, backend.zeros_like(eigvals))
    inv_vals = backend.where(
        eigvals > 0.0,
        1.0 / eigvals,
        backend.zeros_like(eigvals),
    )
    backend.eval(inv_vals)
    gram_inv = backend.matmul(
        eigvecs * backend.reshape(inv_vals, (1, -1)),
        backend.transpose(eigvecs),
    )
    backend.eval(gram_inv)

    transform = backend.matmul(
        backend.transpose(source_matrix),
        backend.matmul(gram_inv, target_matrix),
    )
    backend.eval(transform)

    logger.debug(
        "Eigen solve: min_eig=%.2e, max_eig=%.2e, cond=%.2e",
        min_eig,
        max_eig,
        max_eig / (min_eig + eps) if min_eig > 0 else float("inf"),
    )

    return transform


def _make_embedding_cache_key(embedding: "object", backend: "object") -> str:
    """Create a cache key from embedding matrix shape and content sample."""
    cache = ComputationCache.shared()
    return cache.make_array_key(embedding, backend)


def _make_tokenizer_cache_key(tokenizer: Any, vocab: dict[str, int] | None) -> str:
    vocab_size = len(vocab) if vocab else 0
    name: str | None = None
    for attr in ("name_or_path", "model_name", "name"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value:
            name = value
            break

    if name:
        return f"{type(tokenizer).__name__}:{name}:{vocab_size}"
    if not vocab:
        return f"{type(tokenizer).__name__}:{vocab_size}"

    sample_ids: set[int] = set()
    sample_ids.update(range(min(8, vocab_size)))
    if vocab_size > 8:
        sample_ids.update(range(max(0, vocab_size - 8), vocab_size))
    if vocab_size > 16:
        sample_ids.add(vocab_size // 2)

    sample_pairs = [
        (idx, token) for token, idx in vocab.items() if idx in sample_ids
    ]
    sample_pairs.sort()
    sample_text = "|".join(f"{idx}:{token}" for idx, token in sample_pairs)
    fingerprint = content_hash({"sample": sample_text, "size": vocab_size})
    return f"{type(tokenizer).__name__}:{vocab_size}:{fingerprint}"


def _get_anchor_disk_cache() -> "TwoLevelCache[dict[str, list[float]]]":
    global _anchor_disk_cache
    if _anchor_disk_cache is None:
        base = Path.home() / "Library" / "Caches" / "ModelCypher" / "anchor_maps"
        config = CacheConfig(
            memory_limit=4,
            disk_ttl_seconds=30 * 24 * 60 * 60,
            cache_version=_ANCHOR_CACHE_VERSION,
        )
        _anchor_disk_cache = TwoLevelCache(
            cache_directory=base,
            serializer=lambda payload: payload,
            deserializer=lambda payload: payload,
            config=config,
        )
    return _anchor_disk_cache


def _build_byte_embedding_map(
    tokenizer: Any,
    embedding: "object",
    vocab_size: int,
    backend: "object",
    cache_key: str | None = None,
    tokenizer_key: str | None = None,
) -> dict[int, "object"]:
    """Build byte anchor map with session caching.

    This is expensive (256 Fréchet mean computations) so we cache the result
    based on the embedding matrix hash and tokenizer signature.
    """
    global _anchor_map_cache

    embed_key = cache_key or _make_embedding_cache_key(embedding, backend)
    tokenizer_key = tokenizer_key or f"{type(tokenizer).__name__}:{vocab_size}"
    key_payload = {
        "type": "byte_map",
        "embed": embed_key,
        "tokenizer": tokenizer_key,
        "version": _ANCHOR_CACHE_VERSION,
    }
    cache_key = f"byte_map_{content_hash(key_payload)}"

    # Check cache
    if cache_key in _anchor_map_cache:
        logger.debug("Cache hit for byte map: %s", cache_key[:16])
        return _anchor_map_cache[cache_key]

    disk_cache = _get_anchor_disk_cache()
    disk_payload = disk_cache.get(cache_key)
    if disk_payload is not None:
        byte_map = {int(k): backend.array(v) for k, v in disk_payload.items()}
        _anchor_map_cache[cache_key] = byte_map
        logger.debug("Cache hit for byte map (disk): %s", cache_key[:16])
        return byte_map

    logger.debug("Cache miss for byte map: %s - computing...", cache_key[:16])

    byte_map: dict[int, "object"] = {}
    for byte_value in range(256):
        text = bytes([byte_value]).decode("latin-1")
        token_ids = _encode_ids(tokenizer, text)
        valid = [tid for tid in token_ids if 0 <= tid < vocab_size]
        if not valid:
            continue
        vec = _frechet_mean_from_ids(valid, embedding, backend)
        if vec is not None:
            byte_map[byte_value] = vec

    # Cache the result
    _anchor_map_cache[cache_key] = byte_map
    try:
        disk_payload = {
            str(key): backend.to_numpy(value).tolist()
            for key, value in byte_map.items()
        }
        disk_cache.set(cache_key, disk_payload)
    except (TypeError, ValueError) as e:
        logger.debug("Skipping byte map disk cache: %s", e)
    logger.debug("Cached byte map with %d entries", len(byte_map))

    return byte_map


def _align_bytes_from_matrices(
    source_embed: "object",
    source_matrix: "object",
    target_matrix: "object",
    anchor_labels: list[str],
    backend: "object",
    max_iterations: int = 1000,
    tolerance: float | None = None,
    max_rounds: int = 1,
    anchor_weights: list[float] | None = None,
    initial_transform: "object | None" = None,
    require_phase_lock: bool = False,
) -> dict[str, Any]:
    """Align using pre-computed anchor matrices (avoids recomputing Fréchet means).

    This is the optimized inner loop function that works with pre-computed
    anchor matrices, avoiding the expensive Fréchet mean recomputation.

    Args:
        tolerance: Alignment tolerance. If None, uses machine_epsilon.
    """
    # Log anchor diagnostics for debugging phase-lock issues
    n_anchors = int(source_matrix.shape[0])
    d_source = int(source_matrix.shape[1])
    d_target = int(target_matrix.shape[1])
    logger.debug(
        "Anchor alignment: n_anchors=%d, d_source=%d, d_target=%d, "
        "require_phase_lock=%s",
        n_anchors,
        d_source,
        d_target,
        require_phase_lock,
    )

    # Derive tolerance from dtype if not specified
    eps = machine_epsilon(backend, source_matrix)
    precision_tol = tolerance if tolerance is not None else eps
    cka_before = compute_cka(source_matrix, target_matrix, backend=backend).cka
    logger.debug("Initial CKA before alignment: %.8f", cka_before)

    weighted_source = source_matrix
    weighted_target = target_matrix
    if anchor_weights and not require_phase_lock:
        weighted_source = _apply_anchor_weights(source_matrix, anchor_weights, backend)
        weighted_target = _apply_anchor_weights(target_matrix, anchor_weights, backend)

    transform = _solve_feature_transform_exact(source_matrix, target_matrix, backend)
    if transform is not None:
        aligned_matrix = backend.matmul(source_matrix, transform)
        backend.eval(aligned_matrix)
        cka_after_direct = compute_cka(aligned_matrix, target_matrix, backend=backend).cka
        residual = aligned_matrix - target_matrix
        res_norm = backend.norm(residual)
        tgt_norm = backend.norm(target_matrix)
        backend.eval(res_norm, tgt_norm)
        rel_error = float(backend.to_numpy(res_norm)) / (
            float(backend.to_numpy(tgt_norm)) + precision_tol
        )

        is_perfect = cka_after_direct >= 1.0 - precision_tol or rel_error <= precision_tol

        logger.debug(
            "Direct solve result: cka_before=%.4f, cka_after=%.4f, rel_error=%.2e, "
            "perfect=%s",
            cka_before, cka_after_direct, rel_error, is_perfect,
        )

        if is_perfect:
            cka_after_direct = 1.0
            aligned_source = backend.matmul(source_embed, transform)
            backend.eval(aligned_source)
            return {
                "aligned_source": aligned_source,
                "aligned_matrix": aligned_matrix,
                "anchor_labels": anchor_labels,
                "feature_transform": transform,
                "cka_before": cka_before,
                "cka_after": cka_after_direct,
                "alignment_error": 0.0,
                "iterations": 0,
            }

    if require_phase_lock:
        try:
            source_f64 = backend.astype(source_matrix, "float64")
            target_f64 = backend.astype(target_matrix, "float64")
            backend.eval(source_f64, target_f64)
            transform_f64 = _solve_feature_transform_exact(
                source_f64,
                target_f64,
                backend,
            )
        except Exception:
            transform_f64 = None

        if transform_f64 is not None:
            transform_f32 = backend.astype(transform_f64, source_matrix.dtype)
            aligned_matrix = backend.matmul(source_matrix, transform_f32)
            backend.eval(aligned_matrix)
            cka_after_direct = compute_cka(
                aligned_matrix, target_matrix, backend=backend
            ).cka
            residual = aligned_matrix - target_matrix
            res_norm = backend.norm(residual)
            tgt_norm = backend.norm(target_matrix)
            backend.eval(res_norm, tgt_norm)
            rel_error = float(backend.to_numpy(res_norm)) / (
                float(backend.to_numpy(tgt_norm)) + precision_tol
            )
            if cka_after_direct >= 1.0 - precision_tol or rel_error <= precision_tol:
                cka_after_direct = 1.0
                aligned_source = backend.matmul(source_embed, transform_f32)
                backend.eval(aligned_source)
                return {
                    "aligned_source": aligned_source,
                    "aligned_matrix": aligned_matrix,
                    "anchor_labels": anchor_labels,
                    "feature_transform": transform_f32,
                    "cka_before": cka_before,
                    "cka_after": cka_after_direct,
                    "alignment_error": 0.0,
                    "iterations": 0,
                }
    # When the direct solve is not exact, continue with GramAligner.
    # The iterative approach handles ill-conditioned matrices and cross-dimensional
    # alignment that the direct solve cannot.

    rank = _matrix_rank_for_alignment(source_matrix, backend, eps=precision_tol)

    aligner = GramAligner(
        backend=backend,
        max_iterations=max_iterations,
        max_rounds=max_rounds,
        tolerance=tolerance,
    )
    init_transform = (
        backend.array(initial_transform) if initial_transform is not None else None
    )
    if transform is not None:
        init_transform = transform
    result = aligner.find_perfect_alignment(
        weighted_source,
        weighted_target,
        initial_transform=init_transform,
    )
    transform = backend.array(result.feature_transform)

    aligned_source = backend.matmul(source_embed, transform)
    aligned_matrix = backend.matmul(source_matrix, transform)
    backend.eval(aligned_source, aligned_matrix)

    cka_after = compute_cka(aligned_matrix, target_matrix, backend=backend).cka

    if cka_after >= 1.0 - precision_tol:
        cka_after = 1.0
    elif require_phase_lock and rank == source_matrix.shape[0]:
        sample_transform = backend.array(result.sample_transform)
        source_mean = backend.mean(source_matrix, axis=0, keepdims=True)
        target_mean = backend.mean(target_matrix, axis=0, keepdims=True)
        source_centered = source_matrix - source_mean
        target_centered = target_matrix - target_mean
        sample_aligned_matrix = backend.matmul(sample_transform, source_centered)
        backend.eval(sample_aligned_matrix, source_centered, target_centered)
        cka_after_sample = compute_cka(
            sample_aligned_matrix, target_centered, backend=backend
        ).cka
        if cka_after_sample >= 1.0 - precision_tol:
            feature_transform = _solve_feature_transform_exact(
                source_centered, sample_aligned_matrix, backend
            )
            if feature_transform is not None:
                aligned_source = backend.matmul(source_embed, feature_transform)
                aligned_matrix = backend.matmul(source_centered, feature_transform)
                backend.eval(aligned_source, aligned_matrix)
                cka_after = 1.0
                transform = feature_transform

    # Log final alignment result
    logger.debug(
        "Alignment complete: CKA %.8f → %.8f (iterations=%d, error=%.2e, phase_locked=%s)",
        cka_before,
        cka_after,
        result.iterations,
        result.alignment_error,
        cka_after >= 1.0 - precision_tol,
    )

    return {
        "aligned_source": aligned_source,
        "aligned_matrix": aligned_matrix,
        "anchor_labels": anchor_labels,
        "feature_transform": transform,
        "cka_before": cka_before,
        "cka_after": cka_after,
        "alignment_error": result.alignment_error,
        "iterations": result.iterations,
    }


def _align_bytes(
    source_embed: "object",
    target_embed: "object",
    source_tokenizer: Any,
    target_tokenizer: Any,
    backend: "object",
    max_iterations: int = 1000,
    tolerance: float | None = None,
    max_rounds: int = 1,
    anchor_weights: list[float] | None = None,
    initial_transform: "object | None" = None,
) -> dict[str, Any] | None:
    """Align embeddings using byte-level anchors.

    Args:
        tolerance: Alignment tolerance. If None, uses machine_epsilon.
    """
    source_bytes = _build_byte_embedding_map(
        source_tokenizer,
        source_embed,
        source_embed.shape[0],
        backend,
    )
    target_bytes = _build_byte_embedding_map(
        target_tokenizer,
        target_embed,
        target_embed.shape[0],
        backend,
    )
    shared = sorted(set(source_bytes) & set(target_bytes))
    if len(shared) < 2:
        return None

    source_matrix = backend.stack([source_bytes[b] for b in shared], axis=0)
    target_matrix = backend.stack([target_bytes[b] for b in shared], axis=0)
    backend.eval(source_matrix, target_matrix)

    cka_before = compute_cka(source_matrix, target_matrix, backend=backend).cka
    weighted_source = _apply_anchor_weights(source_matrix, anchor_weights, backend)
    weighted_target = _apply_anchor_weights(target_matrix, anchor_weights, backend)
    aligner = GramAligner(
        backend=backend,
        max_iterations=max_iterations,
        max_rounds=max_rounds,
        tolerance=tolerance,
    )
    init_transform = (
        backend.array(initial_transform) if initial_transform is not None else None
    )
    result = aligner.find_perfect_alignment(
        weighted_source,
        weighted_target,
        initial_transform=init_transform,
    )
    transform = backend.array(result.feature_transform)
    aligned_source = backend.matmul(source_embed, transform)
    backend.eval(aligned_source)

    aligned_matrix = backend.matmul(source_matrix, transform)
    backend.eval(aligned_matrix)
    cka_after = compute_cka(aligned_matrix, target_matrix, backend=backend).cka

    return {
        "aligned_source": aligned_source,
        "aligned_matrix": aligned_matrix,
        "target_matrix": target_matrix,
        "anchor_labels": [f"byte:{b}" for b in shared],
        "feature_transform": transform,
        "bytes_shared": len(shared),
        "cka_before": cka_before,
        "cka_after": cka_after,
        "alignment_error": result.alignment_error,
        "iterations": result.iterations,
        "source_dim": source_matrix.shape[1],
        "target_dim": target_matrix.shape[1],
    }


def _build_atlas_anchor_map(
    tokenizer: Any,
    embedding: "object",
    vocab_size: int,
    backend: "object",
    use_all_support_texts: bool = False,
    byte_map: dict[int, "object"] | None = None,
    use_byte_anchors: bool = False,
    cache_key: str | None = None,
    tokenizer_key: str | None = None,
) -> dict[str, "object"]:
    """Build UnifiedAtlas anchor map with session caching.

    This is expensive (hundreds of Fréchet mean computations) so we cache the result
    based on the embedding matrix hash, tokenizer signature, and support_texts flag.
    """
    global _anchor_map_cache

    use_byte_anchors = bool(use_byte_anchors and byte_map)
    anchor_mode = "byte" if use_byte_anchors else "token"
    embed_key = cache_key or _make_embedding_cache_key(embedding, backend)
    tokenizer_key = tokenizer_key or f"{type(tokenizer).__name__}:{vocab_size}"
    key_payload = {
        "type": "atlas_map",
        "embed": embed_key,
        "tokenizer": tokenizer_key,
        "support_all": use_all_support_texts,
        "anchor_mode": anchor_mode,
        "version": _ANCHOR_CACHE_VERSION,
    }
    cache_key = f"atlas_map_{content_hash(key_payload)}"

    # Check cache
    if cache_key in _anchor_map_cache:
        logger.debug("Cache hit for atlas map: %s", cache_key[:20])
        return _anchor_map_cache[cache_key]

    disk_cache = _get_anchor_disk_cache()
    disk_payload = disk_cache.get(cache_key)
    if disk_payload is not None:
        anchor_map = {key: backend.array(value) for key, value in disk_payload.items()}
        _anchor_map_cache[cache_key] = anchor_map
        logger.debug("Cache hit for atlas map (disk): %s", cache_key[:20])
        return anchor_map

    logger.debug("Cache miss for atlas map: %s - computing...", cache_key[:20])

    anchor_map: dict[str, "object"] = {}
    probes = UnifiedAtlasInventory.all_probes()

    for probe in probes:
        support_texts = [t for t in probe.support_texts if t and len(t.strip()) >= 2]
        if not support_texts:
            continue

        if not use_all_support_texts:
            text = support_texts[0]
            if use_byte_anchors and byte_map is not None:
                byte_values = _encode_bytes(text)
                vec = _frechet_mean_from_bytes(byte_values, byte_map, backend)
            else:
                token_ids = _encode_ids(tokenizer, text)
                valid = [tid for tid in token_ids if 0 <= tid < vocab_size]
                vec = _frechet_mean_from_ids(valid, embedding, backend)
            if vec is not None:
                anchor_map[probe.probe_id] = vec
            continue

        for idx, text in enumerate(support_texts):
            if use_byte_anchors and byte_map is not None:
                byte_values = _encode_bytes(text)
                vec = _frechet_mean_from_bytes(byte_values, byte_map, backend)
            else:
                token_ids = _encode_ids(tokenizer, text)
                valid = [tid for tid in token_ids if 0 <= tid < vocab_size]
                vec = _frechet_mean_from_ids(valid, embedding, backend)
            if vec is not None:
                anchor_map[f"{probe.probe_id}:{idx}"] = vec

    # Cache the result
    _anchor_map_cache[cache_key] = anchor_map
    try:
        disk_payload = {
            str(key): backend.to_numpy(value).tolist()
            for key, value in anchor_map.items()
        }
        disk_cache.set(cache_key, disk_payload)
    except (TypeError, ValueError) as e:
        logger.debug("Skipping atlas map disk cache: %s", e)
    logger.debug("Cached atlas map with %d entries", len(anchor_map))

    return anchor_map


def _apply_feature_transform_to_anchor_map(
    anchor_map: dict[str | int, "object"],
    transform: "object | None",
    backend: "object",
) -> dict[str | int, "object"]:
    if not anchor_map or transform is None:
        return anchor_map

    labels = list(anchor_map.keys())
    matrix = backend.stack([anchor_map[label] for label in labels], axis=0)
    transformed = backend.matmul(matrix, transform)
    backend.eval(transformed)
    return {label: transformed[idx] for idx, label in enumerate(labels)}


def _align_unified_atlas(
    source_embed: "object",
    target_embed: "object",
    source_tokenizer: Any,
    target_tokenizer: Any,
    backend: "object",
    use_all_support_texts: bool = False,
    max_iterations: int = 1000,
    tolerance: float | None = None,
    max_rounds: int = 1,
    anchor_weights: list[float] | None = None,
    initial_transform: "object | None" = None,
) -> dict[str, Any] | None:
    """Align embeddings using unified atlas anchors.

    Args:
        tolerance: Alignment tolerance. If None, uses machine_epsilon.
    """
    source_anchors = _build_atlas_anchor_map(
        source_tokenizer,
        source_embed,
        source_embed.shape[0],
        backend,
        use_all_support_texts=use_all_support_texts,
    )
    target_anchors = _build_atlas_anchor_map(
        target_tokenizer,
        target_embed,
        target_embed.shape[0],
        backend,
        use_all_support_texts=use_all_support_texts,
    )
    shared = sorted(set(source_anchors) & set(target_anchors))
    if len(shared) < 2:
        return None

    source_matrix = backend.stack([source_anchors[k] for k in shared], axis=0)
    target_matrix = backend.stack([target_anchors[k] for k in shared], axis=0)
    backend.eval(source_matrix, target_matrix)

    cka_before = compute_cka(source_matrix, target_matrix, backend=backend).cka
    weighted_source = _apply_anchor_weights(source_matrix, anchor_weights, backend)
    weighted_target = _apply_anchor_weights(target_matrix, anchor_weights, backend)
    aligner = GramAligner(
        backend=backend,
        max_iterations=max_iterations,
        max_rounds=max_rounds,
        tolerance=tolerance,
    )
    init_transform = (
        backend.array(initial_transform) if initial_transform is not None else None
    )
    result = aligner.find_perfect_alignment(
        weighted_source,
        weighted_target,
        initial_transform=init_transform,
    )
    transform = backend.array(result.feature_transform)
    aligned_source = backend.matmul(source_embed, transform)
    backend.eval(aligned_source)

    aligned_matrix = backend.matmul(source_matrix, transform)
    backend.eval(aligned_matrix)
    cka_after = compute_cka(aligned_matrix, target_matrix, backend=backend).cka

    return {
        "aligned_source": aligned_source,
        "aligned_matrix": aligned_matrix,
        "target_matrix": target_matrix,
        "anchor_labels": shared,
        "feature_transform": transform,
        "anchors_shared": len(shared),
        "cka_before": cka_before,
        "cka_after": cka_after,
        "alignment_error": result.alignment_error,
        "iterations": result.iterations,
    }
