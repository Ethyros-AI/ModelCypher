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

"""Geometric training data augmentation.

Augments training samples with the model's geometric state, enabling
the model to learn to interpret and use its own internal geometry.

The geometric context is prepended to each training sample's prompt,
giving the model access to information like:
- loop_persistence: whether reasoning structures are intact
- expansion_ratio: how the model processes information
- highway_depth: where the compression bottleneck is
- exit_convergence: how consistent the outputs are
- reasoning_loops: whether topological loops are present

This enables "geometric self-awareness" - the model can learn to:
- Recognize when it's in a degraded geometric state
- Adjust its reasoning based on its internal state
- Maintain reasoning quality during continual learning
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def augment_training_data_with_geometry(
    model,
    tokenizer,
    training_samples: list[dict[str, str]],
    highway_layer: int,
    base_delta_entropy: float = 0.0,
) -> list[dict[str, str]]:
    """Augment each training sample with the model's geometric state.

    The highway_layer is detected once from intrinsic dimension profile,
    then used for all samples (it's a property of the model, not the data).

    The geometric context is prepended to each sample's prompt. The model
    learns to interpret these values and use them in reasoning.

    Args:
        model: The loaded model.
        tokenizer: Tokenizer for the model.
        training_samples: List of training samples in prompt/completion format.
            Each sample must have "prompt" and "completion" keys.
        highway_layer: Pre-computed highway layer index
            (from detect_highway_layer).
        base_delta_entropy: Base model's ΔH for comparison
            (from compute_base_entropy_trajectory).

    Returns:
        List of augmented samples with geometric prefix added to prompts.
        Original samples are not modified.
    """
    from modelcypher.core.domain.training.geometric_context import GeometricContext

    augmented: list[dict[str, str]] = []

    for sample in training_samples:
        prompt = sample.get("prompt", "")
        completion = sample.get("completion", "")

        if not prompt:
            # Skip samples without prompts
            logger.warning("Skipping sample without prompt")
            continue

        # Compute geometric context for this prompt
        try:
            context = GeometricContext.from_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                highway_layer=highway_layer,
                base_delta_entropy=base_delta_entropy,
            )

            # Prepend geometric context to prompt
            augmented_prompt = context.format() + prompt

            augmented.append({
                "prompt": augmented_prompt,
                "completion": completion,
            })

        except Exception as e:
            raise RuntimeError(
                f"Geometry computation failed for sample: {e}. "
                f"Mixed augmented/unaugmented datasets are not valid training data."
            ) from e

    logger.info(
        "Augmented %d/%d samples with geometric context",
        len([s for s in augmented if "[GEOMETRY]" in s["prompt"]]),
        len(training_samples),
    )

    return augmented


def augment_batch_with_geometry(
    model,
    tokenizer,
    prompts: list[str],
    highway_layer: int,
    base_delta_entropy: float = 0.0,
) -> list[str]:
    """Augment a batch of prompts with geometric context.

    This is the inference-time version for using geometric self-awareness
    without modifying training data.

    Args:
        model: The loaded model.
        tokenizer: Tokenizer for the model.
        prompts: List of prompts to augment.
        highway_layer: Pre-computed highway layer index.
        base_delta_entropy: Base model's ΔH for comparison.

    Returns:
        List of prompts with geometric prefix prepended.
    """
    from modelcypher.core.domain.training.geometric_context import (
        compute_geometric_context_for_batch,
    )

    contexts = compute_geometric_context_for_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        highway_layer=highway_layer,
        base_delta_entropy=base_delta_entropy,
    )

    augmented: list[str] = []
    for prompt, context in zip(prompts, contexts):
        augmented.append(context.format() + prompt)

    return augmented


def extract_geometry_from_prompt(prompt: str) -> tuple[dict[str, Any] | None, str]:
    """Extract geometric context from an augmented prompt.

    Parses the [GEOMETRY]...[/GEOMETRY] block and returns the values
    along with the original prompt.

    Args:
        prompt: Potentially augmented prompt.

    Returns:
        Tuple of (geometry_dict, original_prompt).
        geometry_dict is None if no geometry block found.
    """
    if "[GEOMETRY]" not in prompt or "[/GEOMETRY]" not in prompt:
        return None, prompt

    try:
        start = prompt.index("[GEOMETRY]")
        end = prompt.index("[/GEOMETRY]") + len("[/GEOMETRY]")

        geometry_block = prompt[start:end]
        original_prompt = prompt[end:].lstrip()

        # Parse the geometry block
        geometry: dict[str, Any] = {}
        for line in geometry_block.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("["):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Parse value
                if value in ("yes", "no"):
                    geometry[key] = value == "yes"
                else:
                    try:
                        geometry[key] = float(value)
                    except ValueError:
                        geometry[key] = value

        return geometry, original_prompt

    except Exception as e:
        logger.warning("Failed to parse geometry block: %s", e)
        return None, prompt


__all__ = [
    "augment_training_data_with_geometry",
    "augment_batch_with_geometry",
    "extract_geometry_from_prompt",
]
