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

"""Factory port for model architecture creation.

This port abstracts factory functions that create ModelArchitecturePort instances
and provide config/key introspection. Following hexagonal architecture, domain
code imports from this port rather than directly from adapters.

The module provides a default implementation that can be overridden at runtime:

    from modelcypher.ports.model_architecture_factory import set_factory
    from my_custom_factory import CustomFactory
    set_factory(CustomFactory())

By default, the factory is lazily initialized to the adapter implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modelcypher.ports.model_architecture import ModelArchitecturePort


@runtime_checkable
class ModelArchitectureFactoryPort(Protocol):
    """Protocol for creating ModelArchitecturePort instances and config access.

    Implementations provide:
    - Loading config from model paths
    - Creating architecture wrappers from models
    - Architecture-aware key pattern detection
    """

    def load_config(self, model_path: str | Path) -> dict:
        """Load config.json from model directory.

        Args:
            model_path: Path to model directory

        Returns:
            Config dict from config.json, or empty dict if not found
        """
        ...

    def get_architecture(
        self,
        model: Any,
        config: dict | None = None,
        model_path: str | Path | None = None,
    ) -> "ModelArchitecturePort":
        """Create architecture wrapper from model and config.

        Args:
            model: The loaded model instance
            config: Model config dict. If None, loaded from model_path.
            model_path: Path to model directory (for loading config)

        Returns:
            ModelArchitecturePort implementation for this model family

        Raises:
            ValueError: If architecture cannot be determined
        """
        ...

    def is_causal_model(self, config: dict) -> bool:
        """Check if model is causal (decoder-only) from config.

        Args:
            config: Model config dict from config.json

        Returns:
            True if model is causal, False if bidirectional/encoder
        """
        ...

    def get_output_projection_key(
        self, config: dict, weights: dict[str, Any]
    ) -> str | None:
        """Find the output projection (lm_head) weight key.

        Args:
            config: Model config dict
            weights: Weight dictionary from safetensors

        Returns:
            Weight key for output projection, or None if not found
        """
        ...

    def get_attention_key_pattern(self, config: dict, layer_idx: int) -> list[str]:
        """Get expected attention weight key patterns for a layer.

        Args:
            config: Model config dict
            layer_idx: Layer index

        Returns:
            List of expected attention key patterns
        """
        ...

    def is_attention_key(self, key: str, config: dict, layer_idx: int) -> bool:
        """Check if a weight key is an attention projection for the given layer.

        Args:
            key: Weight key to check
            config: Model config dict
            layer_idx: Expected layer index

        Returns:
            True if key is an attention projection for this layer
        """
        ...


# Module-level default factory
_default_factory: ModelArchitectureFactoryPort | None = None


def _get_default_factory() -> ModelArchitectureFactoryPort:
    """Lazily initialize and return the default factory."""
    global _default_factory
    if _default_factory is None:
        # Import adapter implementation only when needed
        from modelcypher.adapters.model_architecture import AdapterFactory
        _default_factory = AdapterFactory()
    return _default_factory


def set_factory(factory: ModelArchitectureFactoryPort) -> None:
    """Override the default factory.

    Args:
        factory: Factory implementation to use
    """
    global _default_factory
    _default_factory = factory


def get_factory() -> ModelArchitectureFactoryPort:
    """Get the current factory (default or overridden)."""
    return _get_default_factory()


# Convenience functions that delegate to the current factory
def load_config(model_path: str | Path) -> dict:
    """Load config.json from model directory."""
    return get_factory().load_config(model_path)


def get_model_architecture(
    model: Any,
    config: dict | None = None,
    model_path: str | Path | None = None,
) -> "ModelArchitecturePort":
    """Create architecture wrapper from model and config."""
    return get_factory().get_architecture(model, config, model_path)


def is_causal_model(config: dict) -> bool:
    """Check if model is causal (decoder-only) from config."""
    return get_factory().is_causal_model(config)


def get_output_projection_key(config: dict, weights: dict[str, Any]) -> str | None:
    """Find the output projection (lm_head) weight key."""
    return get_factory().get_output_projection_key(config, weights)


def get_attention_key_pattern(config: dict, layer_idx: int) -> list[str]:
    """Get expected attention weight key patterns for a layer."""
    return get_factory().get_attention_key_pattern(config, layer_idx)


def is_attention_key(key: str, config: dict, layer_idx: int) -> bool:
    """Check if a weight key is an attention projection for the given layer."""
    return get_factory().is_attention_key(key, config, layer_idx)


__all__ = [
    "ModelArchitectureFactoryPort",
    "get_factory",
    "set_factory",
    "load_config",
    "get_model_architecture",
    "is_causal_model",
    "get_output_projection_key",
    "get_attention_key_pattern",
    "is_attention_key",
]
