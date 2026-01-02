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
1. Analyze vocabularies (stats, alignment)
2. Build token alignment map (exact + embedding similarity)
3. Project source embeddings to target space (Procrustes/OT)
4. Blend aligned embeddings with geometry-weighted alpha
"""

from __future__ import annotations

import logging
import time
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.alignment_diagnostic import (
    AlignmentSignal,
    alignment_signal_from_matrices,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)

from .anchor_selection import (
    _balanced_anchor_subset,
    _compute_anchor_weights,
    _select_coverage_indices,
    _select_shared_full_rank_indices,
    _uniform_subset,
)
from .atlas_alignment import _build_atlas_anchor_map
from .byte_alignment import _align_bytes_from_matrices, _build_byte_embedding_map
from .caching import _make_embedding_cache_key, _make_tokenizer_cache_key
from .config import VocabularyConfig, VocabularyResult
from .tokenizer_utils import _extract_vocab

logger = logging.getLogger(__name__)


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
    - Null space addition for aligned tokens (no blending)

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

    # Check for vocab alignment before doing expensive operations
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
            target_embed_original = target_embed
            source_cache_key = _make_embedding_cache_key(source_embed, backend)
            target_cache_key = _make_embedding_cache_key(target_embed, backend)
            target_cache_key_original = target_cache_key

            # Binary (1D) alignment: align byte-level anchors before vocabulary merging.
            # Pre-compute byte maps ONCE to avoid repeated Frechet mean computation.
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

            merge_config = CrossVocabMergeConfig(
                projection_strategy=effective_strategy,
                preserve_special_tokens=config.preserve_special_tokens,
                anchor_count=config.anchor_count,
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

            merge_metrics = merger.analyze_merge_quality(result)

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
            metrics[f"{embed_key}_alignment_score"] = result.alignment.alignment_score
            metrics[f"{embed_key}_warnings"] = result.warnings
            metrics[f"{embed_key}_strict_token_alignment"] = strict_token_alignment

            logger.info(
                "Aligned %s: coverage=%.2f, alignment_score=%.2f",
                embed_key,
                result.alignment_map.coverage,
                merge_metrics["alignment_score"],
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
        # Dtype-derived epsilon for normalization
        eps = division_epsilon(backend, centered)
        normalized = centered / (norms + eps)
        backend.eval(normalized)
        return normalized

    return embedding
