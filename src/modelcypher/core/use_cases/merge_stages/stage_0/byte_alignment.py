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

import logging
from typing import Any

from modelcypher.core.domain.cache import content_hash
from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

from .anchor_selection import _apply_anchor_weights
from .caching import (
    _ANCHOR_CACHE_VERSION,
    _anchor_map_cache,
    _get_anchor_disk_cache,
    _make_embedding_cache_key,
)
from .frechet_ops import _frechet_mean_from_ids
from .matrix_ops import _matrix_rank_for_alignment, _solve_feature_transform_exact
from .tokenizer_utils import _encode_ids

logger = logging.getLogger(__name__)


def _build_byte_embedding_map(
    tokenizer: Any,
    embedding: "object",
    vocab_size: int,
    backend: "object",
    cache_key: str | None = None,
    tokenizer_key: str | None = None,
) -> dict[int, "object"]:
    """Build byte anchor map with session caching.

    This is expensive (256 Frechet mean computations) so we cache the result
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
    """Align using pre-computed anchor matrices (avoids recomputing Frechet means).

    This is the optimized inner loop function that works with pre-computed
    anchor matrices, avoiding the expensive Frechet mean recomputation.

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
    # Use corrected CKA to avoid feature-sampling underestimation
    cka_before = compute_cka(
        source_matrix,
        target_matrix,
        backend=backend,
        estimator=HSICEstimator.AUTO,
        feature_bias_correction=True,
    ).best
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
        # Use corrected CKA to avoid feature-sampling underestimation
        cka_after_direct = compute_cka(
            aligned_matrix,
            target_matrix,
            backend=backend,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        ).best
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
            # Use corrected CKA to avoid feature-sampling underestimation
            cka_after_direct = compute_cka(
                aligned_matrix,
                target_matrix,
                backend=backend,
                estimator=HSICEstimator.AUTO,
                feature_bias_correction=True,
            ).best
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

    # Use corrected CKA to avoid feature-sampling underestimation
    cka_after = compute_cka(
        aligned_matrix,
        target_matrix,
        backend=backend,
        estimator=HSICEstimator.AUTO,
        feature_bias_correction=True,
    ).best

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
        # Use corrected CKA to avoid feature-sampling underestimation
        cka_after_sample = compute_cka(
            sample_aligned_matrix,
            target_centered,
            backend=backend,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        ).best
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
        "Alignment complete: CKA %.8f -> %.8f (iterations=%d, error=%.2e, phase_locked=%s)",
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

    cka_before = compute_cka(
        source_matrix,
        target_matrix,
        backend=backend,
        estimator=HSICEstimator.AUTO,
        feature_bias_correction=True,
    ).best
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
    cka_after = compute_cka(
        aligned_matrix,
        target_matrix,
        backend=backend,
        estimator=HSICEstimator.AUTO,
        feature_bias_correction=True,
    ).best

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
