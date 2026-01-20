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

"""Inference use cases - orchestration for model inference."""

from __future__ import annotations

import logging
from typing import Any

from .comparison import (
    CheckpointComparisonCoordinator,
    ComparisonError,
    ComparisonEvent,
    ComparisonResult,
    EventType,
)

logger = logging.getLogger(__name__)


def run_inference(
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> dict[str, Any]:
    """Run inference on a loaded model.

    This is a simple greedy wrapper for experiments and use cases.

    Parameters
    ----------
    model : Any
        The loaded MLX model.
    tokenizer : Any
        The tokenizer for the model.
    prompt : str
        The input prompt.

    Returns
    -------
    dict
        Dictionary containing:
        - generated_text: The generated text (excluding prompt)
        - full_text: The full text including prompt
        - prompt: The original prompt
    """
    from modelcypher.core.domain._backend import get_default_backend

    context_candidates = [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(getattr(model, "config", None), "max_seq_len", None),
        getattr(getattr(model, "config", None), "max_seq_length", None),
        getattr(model, "max_seq_len", None),
        getattr(model, "max_seq_length", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    max_context = 0
    for value in context_candidates:
        if isinstance(value, int) and value > 0:
            max_context = value
            break
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    max_tokens = max(0, max_context - len(prompt_tokens))

    if max_tokens <= 0:
        return {
            "generated_text": "",
            "full_text": prompt,
            "prompt": prompt,
        }

    backend = get_default_backend()
    tokens = list(prompt_tokens)
    generated_tokens: list[int] = []

    for _ in range(max_tokens):
        inputs = backend.array([tokens])
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        elif logits.ndim == 2:
            last_logits = logits[-1, :]
        else:
            last_logits = backend.reshape(logits, (-1,))

        next_token_arr = backend.argmax(last_logits, axis=-1)
        backend.eval(next_token_arr)
        next_token_id = int(backend.to_scalar(next_token_arr))
        generated_tokens.append(next_token_id)
        tokens.append(next_token_id)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and next_token_id == eos_id:
            break

    generated_text = tokenizer.decode(generated_tokens) if generated_tokens else ""
    full_text = f"{prompt}{generated_text}"

    return {
        "generated_text": generated_text,
        "full_text": full_text,
        "prompt": prompt,
    }


__all__ = [
    "CheckpointComparisonCoordinator",
    "ComparisonError",
    "ComparisonEvent",
    "ComparisonResult",
    "EventType",
    "run_inference",
]
