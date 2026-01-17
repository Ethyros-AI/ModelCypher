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
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """Run inference on a loaded model.

    This is a simple wrapper around mlx_lm.generate for experiments and use cases.

    Parameters
    ----------
    model : Any
        The loaded MLX model.
    tokenizer : Any
        The tokenizer for the model.
    prompt : str
        The input prompt.
    max_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature (0.0 = greedy).
    top_p : float
        Nucleus sampling parameter.

    Returns
    -------
    dict
        Dictionary containing:
        - generated_text: The generated text (excluding prompt)
        - full_text: The full text including prompt
        - prompt: The original prompt
    """
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise ImportError(
            "mlx_lm is required for inference. Install with: pip install mlx-lm"
        ) from exc

    # Create sampler
    sampler = make_sampler(temp=temperature, top_p=top_p)

    # Generate text
    full_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    # Extract generated portion (remove prompt)
    generated_text = full_text
    if full_text.startswith(prompt):
        generated_text = full_text[len(prompt):]

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
