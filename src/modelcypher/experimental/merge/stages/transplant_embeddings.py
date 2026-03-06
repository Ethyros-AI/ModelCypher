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

"""Embedding alignment handling for transplant stage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed
from modelcypher.ports.model_architecture_factory import get_output_projection_key

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _extract_vocab(tokenizer: Any) -> dict[str, int]:
    """Extract vocabulary mapping from tokenizer."""
    if hasattr(tokenizer, "get_vocab"):
        return tokenizer.get_vocab()
    if hasattr(tokenizer, "vocab"):
        vocab = tokenizer.vocab
        if isinstance(vocab, dict):
            return vocab
    if hasattr(tokenizer, "encoder"):
        return tokenizer.encoder
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "get_vocab"):
        return tokenizer.tokenizer.get_vocab()
    logger.warning("Could not extract vocabulary from tokenizer type %s", type(tokenizer))
    return {}


def apply_embedding_alignment(
    *,
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    embedding_transform: "Array | None",
    merged: dict[str, "Array"],
    metrics: dict[str, Any],
    backend: "Backend",
    source_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
    skip_embedding_transplant: bool = False,
) -> None:
    """Handle embedding alignment strategy and preserve target vocab interface.

    For cross-vocabulary merges, uses CrossVocabMerger to project source
    embeddings into target space using Procrustes alignment on shared tokens.
    """
    b = backend

    if skip_embedding_transplant:
        logger.info(
            "EMBEDDING ALIGNMENT: Skipped (skip_embedding_transplant=%s)",
            skip_embedding_transplant,
        )
        metrics["embedding_transplant_skipped"] = True
        return

    source_embed_key = None
    target_embed_key = None

    for key in source_weights:
        if "embed_tokens.weight" in key or "wte.weight" in key:
            source_embed_key = key
            break

    for key in target_weights:
        if "embed_tokens.weight" in key or "wte.weight" in key:
            target_embed_key = key
            break

    cross_vocab_merge = False
    if source_embed_key and target_embed_key:
        src_vocab_shape = b.shape(source_weights[source_embed_key])
        tgt_vocab_shape = b.shape(target_weights[target_embed_key])
        src_vocab_size = int(src_vocab_shape[0])
        tgt_vocab_size = int(tgt_vocab_shape[0])
        cross_vocab_merge = (src_vocab_size != tgt_vocab_size)

    if cross_vocab_merge:
        # SAFETY: Cross-vocab embedding transplant is disabled by default.
        # The naive truncation approach (4096→2048) corrupts embeddings.
        # See Experiment 11 findings: embedding transplant caused garbage output.
        logger.warning(
            "CROSS-VOCAB MERGE: Skipping embedding transplant (src=%d, tgt=%d tokens). "
            "Naive truncation corrupts embeddings.",
            src_vocab_size,
            tgt_vocab_size,
        )
        metrics["cross_vocab_merge"] = True
        metrics["embedding_transplant_skipped"] = True
        metrics["skip_reason"] = "cross_vocab_truncation_unsafe"
        return

        logger.info(
            "CROSS-VOCAB MERGE: Aligning embeddings across vocabularies "
            "(src: %d tokens, tgt: %d tokens) [FORCED - experimental]",
            src_vocab_size,
            tgt_vocab_size,
        )
        metrics["cross_vocab_merge"] = True
        metrics["src_vocab_size"] = src_vocab_size
        metrics["tgt_vocab_size"] = tgt_vocab_size

        # Dequantize embeddings
        src_embed = source_weights[source_embed_key]
        src_embed = dequantize_if_needed(src_embed, source_embed_key, source_weights, b)

        tgt_embed = target_weights[target_embed_key]
        tgt_embed = dequantize_if_needed(tgt_embed, target_embed_key, target_weights, b)

        # Memory-efficient approach: find exact token matches only
        # This avoids building full O(n*m) alignment structures
        src_indices = []
        tgt_indices = []

        if source_tokenizer is not None and target_tokenizer is not None:
            source_vocab = _extract_vocab(source_tokenizer)
            target_vocab = _extract_vocab(target_tokenizer)

            if source_vocab and target_vocab:
                logger.info(
                    "CROSS-VOCAB MERGE: Finding exact token matches "
                    "(source: %d tokens, target: %d tokens)",
                    len(source_vocab),
                    len(target_vocab),
                )

                # Find exact matches (memory efficient - just string lookup)
                for token, src_id in source_vocab.items():
                    if token in target_vocab:
                        tgt_id = target_vocab[token]
                        if src_id < src_vocab_size and tgt_id < tgt_vocab_size:
                            src_indices.append(src_id)
                            tgt_indices.append(tgt_id)

        tokens_matched = len(src_indices)
        logger.info(
            "CROSS-VOCAB MERGE: Found %d exact token matches (%.1f%% of source vocab)",
            tokens_matched,
            100.0 * tokens_matched / src_vocab_size if src_vocab_size > 0 else 0,
        )

        metrics["tokens_matched"] = tokens_matched
        metrics["match_ratio"] = tokens_matched / src_vocab_size if src_vocab_size > 0 else 0

        if tokens_matched > 0:
            # Project source embeddings to target dimension
            src_hidden_dim = int(b.shape(src_embed)[1])
            tgt_hidden_dim = int(b.shape(tgt_embed)[1])

            if src_hidden_dim != tgt_hidden_dim:
                logger.info(
                    "CROSS-VOCAB MERGE: Projecting embeddings (%d -> %d dimensions)",
                    src_hidden_dim,
                    tgt_hidden_dim,
                )
                # Simple dimension adjustment: truncate or pad
                if src_hidden_dim > tgt_hidden_dim:
                    src_embed_proj = src_embed[:, :tgt_hidden_dim]
                else:
                    padding = b.zeros((src_vocab_size, tgt_hidden_dim - src_hidden_dim))
                    src_embed_proj = b.concatenate([src_embed, padding], axis=1)
            else:
                src_embed_proj = src_embed

            # Compute Procrustes alignment on matched tokens
            src_idx_arr = b.array(src_indices)
            tgt_idx_arr = b.array(tgt_indices)

            # Get matched vectors for alignment
            matched_src = b.take(src_embed_proj, src_idx_arr, axis=0)
            matched_tgt = b.take(tgt_embed, tgt_idx_arr, axis=0)

            # Compute Procrustes rotation with SO(n) enforcement (det=+1)
            # This ensures proper rotation, not reflection, for Lie-group correctness
            from modelcypher.core.domain.geometry.numerical_stability import geodesic_svd

            src_mean = b.mean(matched_src, axis=0, keepdims=True)
            tgt_mean = b.mean(matched_tgt, axis=0, keepdims=True)
            src_centered = matched_src - src_mean
            tgt_centered = matched_tgt - tgt_mean

            cross_cov = b.matmul(b.transpose(src_centered), tgt_centered)
            U, _, Vt = geodesic_svd(b, cross_cov)
            # rotation = U @ Vt is already orthogonal from cross-cov SVD
            # Enforce det=+1 to get SO(n) (rotation, not reflection)
            rotation = b.matmul(U, Vt)
            b.eval(rotation)

            # Check determinant for square matrices
            if rotation.shape[0] == rotation.shape[1]:
                det_val = b.det(rotation)
                b.eval(det_val)
                if float(b.to_scalar(det_val)) < 0:
                    # Flip last column of U to make det=+1
                    U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
                    rotation = b.matmul(U_fixed, Vt)
                    b.eval(rotation)

            # Project all source embeddings
            src_centered_all = src_embed_proj - src_mean
            projected = b.matmul(src_centered_all, rotation) + tgt_mean
            b.eval(projected)

            # NULL-SPACE ADDITION (not blending) for matched tokens
            # Blending dilutes information; null-space addition preserves target
            # and adds source knowledge where target has unused capacity.
            matched_projected = b.take(projected, src_idx_arr, axis=0)
            matched_tgt_vecs = b.take(tgt_embed, tgt_idx_arr, axis=0)

            # Compute delta (source - target after projection)
            delta = matched_projected - matched_tgt_vecs

            # Compute variance-based null-space weights for target embeddings
            # High variance = actively used = preserve; Low variance = available = transfer
            tgt_var = b.var(matched_tgt_vecs, axis=0)
            b.eval(tgt_var)
            max_var = b.max(tgt_var)
            b.eval(max_var)
            max_var_val = float(b.to_scalar(max_var))
            eps = float(machine_epsilon(b, tgt_var))
            normalized_var = tgt_var / max(max_var_val, eps)
            # keep_weight = 1 - normalized_var (high variance = low keep = preserve target)
            # But for null-space: we want to ADD where target is sparse (low variance)
            transfer_weight = 1.0 - normalized_var  # Transfer more where variance is low
            b.eval(transfer_weight)

            # Apply weighted delta addition (null-space constrained)
            weighted_delta = delta * transfer_weight  # [n_matched, hidden_dim]
            merged_matched = matched_tgt_vecs + weighted_delta
            b.eval(merged_matched)

            # Create final embedding matrix using a single indexed update.
            final_embed = b.array(tgt_embed)
            tgt_idx_arr = b.array(tgt_indices, dtype="int32")
            idx_mat = b.reshape(tgt_idx_arr, (-1, 1))
            idx_mat = b.broadcast_to(idx_mat, (tokens_matched, tgt_hidden_dim))
            final_embed = b.put_along_axis(final_embed, idx_mat, merged_matched, axis=0)
            b.eval(final_embed)

            merged[target_embed_key] = final_embed
            metrics["tokens_merged_nullspace"] = tokens_matched

            # Compute transfer metrics
            delta_norm = float(b.to_scalar(b.sqrt(b.sum(weighted_delta * weighted_delta))))
            mean_transfer = float(b.to_scalar(b.mean(transfer_weight)))
            metrics["embedding_delta_norm"] = delta_norm
            metrics["embedding_mean_transfer_weight"] = mean_transfer

            logger.info(
                "CROSS-VOCAB MERGE: Null-space merged %d embeddings (%.1f%% of target vocab), "
                "delta_norm=%.4f, mean_transfer=%.3f",
                tokens_matched,
                100.0 * tokens_matched / tgt_vocab_size,
                delta_norm,
                mean_transfer,
            )

            # Clean up
            del projected, matched_projected, merged_matched, weighted_delta
            if hasattr(b, "clear_cache"):
                b.clear_cache()

        else:
            # No tokens matched, keep target embeddings unchanged
            merged[target_embed_key] = tgt_embed
            logger.info("CROSS-VOCAB MERGE: No token matches, keeping target embeddings")

        metrics["preserved_target_vocab"] = True

        # Preserve lm_head (use architecture-aware detection)
        lm_head_key = get_output_projection_key({}, target_weights)

        if lm_head_key:
            tgt_lm_head = target_weights[lm_head_key]
            tgt_lm_head = dequantize_if_needed(tgt_lm_head, lm_head_key, target_weights, b)
            merged[lm_head_key] = tgt_lm_head
            logger.info("LM_HEAD PRESERVED: Keeping target %s", lm_head_key)

        return

    if embedding_transform is None:
        logger.info("EMBEDDING ALIGNMENT: No embedding_transform provided, using target embeddings")
        return

    if not source_embed_key or not target_embed_key:
        logger.info("EMBEDDING ALIGNMENT: Missing embedding keys, using target embeddings")
        return

    src_embed = source_weights[source_embed_key]
    src_embed = dequantize_if_needed(src_embed, source_embed_key, source_weights, b)

    tgt_embed = target_weights[target_embed_key]
    tgt_embed = dequantize_if_needed(tgt_embed, target_embed_key, target_weights, b)

    src_hidden_dim = int(b.shape(src_embed)[1])
    tgt_hidden_dim = int(b.shape(tgt_embed)[1])

    merged[target_embed_key] = tgt_embed
    metrics["embedding_preserved"] = True
    metrics["same_vocab_target_kept"] = True

    logger.info(
        "EMBEDDING PRESERVED: Keeping target embed_tokens [%d,%d] (source was [%d,%d])",
        int(b.shape(tgt_embed)[0]),
        tgt_hidden_dim,
        int(b.shape(src_embed)[0]),
        src_hidden_dim,
    )

    # Use architecture-aware lm_head detection
    lm_head_key = get_output_projection_key({}, target_weights)

    if lm_head_key:
        tgt_lm_head = target_weights[lm_head_key]
        tgt_lm_head = dequantize_if_needed(tgt_lm_head, lm_head_key, target_weights, b)
        merged[lm_head_key] = tgt_lm_head
        logger.info("LM_HEAD PRESERVED: Keeping target %s", lm_head_key)
    else:
        logger.info("LM_HEAD: Weight-tied with embed_tokens (both preserved)")
