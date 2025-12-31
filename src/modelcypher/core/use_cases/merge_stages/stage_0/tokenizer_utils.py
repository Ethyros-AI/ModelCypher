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

logger = logging.getLogger(__name__)


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
