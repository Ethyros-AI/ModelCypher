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

from modelcypher.ports.inference import InferenceEngine

from .comparison import (
    CheckpointComparisonCoordinator,
    ComparisonError,
    ComparisonEvent,
    ComparisonResult,
    EventType,
)

logger = logging.getLogger(__name__)


def run_inference(
    engine: "InferenceEngine",
    model: str,
    prompt: str,
) -> dict[str, Any]:
    """Run inference via the InferenceEngine port.

    Parameters
    ----------
    engine : InferenceEngine
        Inference engine implementation (adapter).
    model : str
        Path or identifier for the model.
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
    result = engine.infer(model=model, prompt=prompt)
    generated_text = str(result.get("response", ""))
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
