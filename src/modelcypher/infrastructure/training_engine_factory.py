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

"""Factory for creating training engine implementations.

Delegates backend selection to the backends layer so infrastructure
does not depend on specific runtime names or frameworks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def get_training_engine() -> Any:
    """Get the training engine for the current runtime."""
    from modelcypher.backends import get_training_engine as _get_training_engine

    return _get_training_engine()


def get_checkpoint_manager(max_checkpoints: int = 3) -> Any:
    """Get the checkpoint manager for the current runtime."""
    from modelcypher.backends import get_training_checkpoint_manager

    return get_training_checkpoint_manager(max_checkpoints=max_checkpoints)


def get_evaluation_engine(config: Any = None) -> Any:
    """Get the unified evaluation engine with default backend.

    Args:
        config: Optional EvaluationConfig. Uses default if not provided.

    Returns:
        EvaluationEngine instance using the default backend.
    """
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.training.evaluation import EvaluationEngine

    return EvaluationEngine(backend=get_default_backend(), config=config)


# NOTE: No get_lora_config_class() - use LoRALayerConfig from ports/training.py directly.
# All LoRA configs are geometry-derived, not platform-specific types.


def get_loss_landscape_computer() -> Any:
    """Get the loss landscape computer for the current runtime."""
    from modelcypher.backends import get_training_loss_landscape_computer

    return get_training_loss_landscape_computer()


__all__ = [
    "get_training_engine",
    "get_checkpoint_manager",
    "get_evaluation_engine",
    "get_loss_landscape_computer",
]
