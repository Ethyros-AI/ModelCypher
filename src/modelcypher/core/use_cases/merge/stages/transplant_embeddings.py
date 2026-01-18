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

from modelcypher.core.use_cases.quantization_utils import dequantize_if_needed

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
) -> None:
    """Handle embedding alignment strategy and preserve target vocab interface.

    For cross-vocabulary merges, uses CrossVocabMerger to project source
    embeddings into target space using Procrustes alignment on shared tokens.
    """
    b = backend

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
        logger.info(
            "CROSS-VOCAB MERGE: Aligning embeddings across vocabularies "
            "(src: %d tokens, tgt: %d tokens)",
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

            # Compute Procrustes rotation (CPU-friendly version)
            src_mean = b.mean(matched_src, axis=0, keepdims=True)
            tgt_mean = b.mean(matched_tgt, axis=0, keepdims=True)
            src_centered = matched_src - src_mean
            tgt_centered = matched_tgt - tgt_mean

            cross_cov = b.matmul(b.transpose(src_centered), tgt_centered)
            U, _, Vt = b.svd(cross_cov)
            rotation = b.matmul(U, Vt)

            # Project all source embeddings
            src_centered_all = src_embed_proj - src_mean
            projected = b.matmul(src_centered_all, rotation) + tgt_mean
            b.eval(projected)

            # Blend matched tokens: 50% projected source + 50% target
            blend_weight = 0.5
            matched_projected = b.take(projected, src_idx_arr, axis=0)
            blended = blend_weight * matched_projected + (1 - blend_weight) * matched_tgt

            # Create final embedding matrix using scatter-like update
            # Use index_put equivalent: build mask and blend
            tgt_idx_set = set(tgt_indices)
            idx_to_blend = {tgt_idx: i for i, tgt_idx in enumerate(tgt_indices)}

            # Process in batches to avoid memory issues
            batch_size = 10000
            final_rows = []

            for start in range(0, tgt_vocab_size, batch_size):
                end = min(start + batch_size, tgt_vocab_size)
                batch_rows = []

                for i in range(start, end):
                    if i in tgt_idx_set:
                        blend_idx = idx_to_blend[i]
                        batch_rows.append(blended[blend_idx:blend_idx+1])
                    else:
                        batch_rows.append(tgt_embed[i:i+1])

                if batch_rows:
                    batch_arr = b.concatenate(batch_rows, axis=0)
                    final_rows.append(batch_arr)

                # Clear intermediate results
                del batch_rows
                if hasattr(b, "clear_cache"):
                    b.clear_cache()

            final_embed = b.concatenate(final_rows, axis=0)
            b.eval(final_embed)

            merged[target_embed_key] = final_embed
            metrics["tokens_blended"] = tokens_matched
            metrics["blend_weight"] = blend_weight

            logger.info(
                "CROSS-VOCAB MERGE: Blended %d token embeddings (%.1f%% of target vocab)",
                tokens_matched,
                100.0 * tokens_matched / tgt_vocab_size,
            )

            # Clean up
            del projected, matched_projected, blended, final_rows
            if hasattr(b, "clear_cache"):
                b.clear_cache()

        else:
            # No tokens matched, keep target embeddings unchanged
            merged[target_embed_key] = tgt_embed
            logger.info("CROSS-VOCAB MERGE: No token matches, keeping target embeddings")

        metrics["preserved_target_vocab"] = True

        # Preserve lm_head
        lm_head_key = None
        for key in target_weights:
            if "lm_head" in key.lower() and "weight" in key:
                lm_head_key = key
                break

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

    lm_head_key = None
    for key in target_weights:
        if "lm_head" in key.lower() and "weight" in key:
            lm_head_key = key
            break

    if lm_head_key:
        tgt_lm_head = target_weights[lm_head_key]
        tgt_lm_head = dequantize_if_needed(tgt_lm_head, lm_head_key, target_weights, b)
        merged[lm_head_key] = tgt_lm_head
        logger.info("LM_HEAD PRESERVED: Keeping target %s", lm_head_key)
    else:
        logger.info("LM_HEAD: Weight-tied with embed_tokens (both preserved)")
