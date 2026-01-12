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


def apply_embedding_alignment(
    *,
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    embedding_transform: "Array | None",
    merged: dict[str, "Array"],
    metrics: dict[str, Any],
    backend: "Backend",
) -> None:
    """Handle embedding alignment strategy and preserve target vocab interface."""
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
            "CROSS-VOCAB MERGE: Preserving target's native vocabulary interface "
            "(src: %d tokens, tgt: %d tokens)",
            src_vocab_size,
            tgt_vocab_size,
        )
        logger.info(
            "CROSS-VOCAB MERGE: Target keeps its 1D↔2D interface. "
            "Hidden manifold enriched via aligned transplant."
        )
        metrics["cross_vocab_merge"] = True
        metrics["preserved_target_vocab"] = True
        metrics["src_vocab_size"] = src_vocab_size
        metrics["tgt_vocab_size"] = tgt_vocab_size
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
