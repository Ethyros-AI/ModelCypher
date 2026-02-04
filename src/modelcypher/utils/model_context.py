# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Model context limit utilities.

Shared logic for resolving model context limits from config files and tokenizers.
Used by inference engines and training services.

ONE implementation. No duplicates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def context_from_config(model_path: Path) -> int | None:
    """Read context limit from model config.json.

    Args:
        model_path: Path to model directory containing config.json

    Returns:
        Context limit in tokens, or None if not found
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    for key in (
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_len",
        "max_seq_length",
        "context_length",
        "max_context_length",
        "n_ctx",
        "model_max_length",
        "seq_length",
    ):
        value = config.get(key)
        if isinstance(value, (int, float)):
            int_value = int(value)
            if int_value > 0:
                return int_value
    return None


def resolve_context_limit(
    model_path: Path,
    tokenizer: Any,
    cache: dict[str, int] | None = None,
) -> int | None:
    """Resolve model context length from tokenizer or config.

    Checks tokenizer attributes first, then falls back to config.json.
    Returns the minimum of all valid candidates found.

    Args:
        model_path: Path to model directory
        tokenizer: Tokenizer with potential context limit attributes
        cache: Optional cache dict to store/retrieve resolved limits

    Returns:
        Context limit in tokens, or None if not determinable
    """
    cache_key = str(model_path)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    candidates: list[int] = []
    for attr in (
        "model_max_length",
        "max_length",
        "max_seq_len",
        "max_sequence_length",
        "n_ctx",
        "context_length",
        "max_context_length",
    ):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, (int, float)):
            int_value = int(value)
            if 0 < int_value < 10**7:
                candidates.append(int_value)

    config_value = context_from_config(model_path)
    if config_value is not None:
        candidates.append(config_value)

    if not candidates:
        return None

    resolved = min(candidates)
    if cache is not None:
        cache[cache_key] = resolved
    return resolved


def derive_max_tokens(
    model_path: Path,
    prompt: str,
    tokenizer: Any,
    cache: dict[str, int] | None = None,
) -> int:
    """Derive max tokens available for generation.

    Args:
        model_path: Path to model directory
        prompt: Input prompt to encode
        tokenizer: Tokenizer for encoding prompt
        cache: Optional cache for context limits

    Returns:
        Maximum tokens available for generation (0 if context exhausted)
    """
    context_limit = resolve_context_limit(model_path, tokenizer, cache)
    if context_limit is None:
        return 0
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    available = context_limit - len(token_ids)
    return max(0, available)


__all__ = ["context_from_config", "resolve_context_limit", "derive_max_tokens"]
