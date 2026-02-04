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

"""Activation provider adapter that delegates to the backend layer.

This module intentionally contains no framework-specific references.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modelcypher.backends import get_activation_provider as _get_activation_provider
from modelcypher.ports.activation_provider import (
    ActivationProvider as ActivationProviderPort,
    ProbeActivationBatch,
    TrajectoryActivations,
)


class ActivationProvider(ActivationProviderPort):
    """Backend-selected activation provider wrapper."""

    def __init__(
        self,
        backend: Any | None = None,
        model_path: str | Path | None = None,
        pooling: str = "auto",
    ) -> None:
        self._impl = _get_activation_provider(
            backend=backend, model_path=model_path, pooling=pooling
        )

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple delegation
        return getattr(self._impl, name)

    def collect_hidden_activations(self, *args: Any, **kwargs: Any) -> dict[int, Any]:
        return self._impl.collect_hidden_activations(*args, **kwargs)

    def collect_embedding_activations(self, *args: Any, **kwargs: Any) -> Any:
        return self._impl.collect_embedding_activations(*args, **kwargs)

    def collect_intermediate_activations(self, *args: Any, **kwargs: Any) -> dict[int, Any]:
        return self._impl.collect_intermediate_activations(*args, **kwargs)

    def collect_attention_activations(self, *args: Any, **kwargs: Any) -> tuple[dict[int, Any], dict[int, Any]]:
        return self._impl.collect_attention_activations(*args, **kwargs)

    def collect_probe_activations_batch(self, *args: Any, **kwargs: Any) -> ProbeActivationBatch:
        return self._impl.collect_probe_activations_batch(*args, **kwargs)

    def collect_hidden_activations_batch(self, *args: Any, **kwargs: Any) -> list[dict[int, Any]]:
        return self._impl.collect_hidden_activations_batch(*args, **kwargs)

    def collect_intermediate_activations_batch(self, *args: Any, **kwargs: Any) -> list[dict[int, Any]]:
        return self._impl.collect_intermediate_activations_batch(*args, **kwargs)

    def collect_attention_activations_batch(self, *args: Any, **kwargs: Any) -> tuple[list[dict[int, Any]], list[dict[int, Any]]]:
        return self._impl.collect_attention_activations_batch(*args, **kwargs)

    def collect_trajectory_batch(self, *args: Any, **kwargs: Any) -> TrajectoryActivations:
        return self._impl.collect_trajectory_batch(*args, **kwargs)


def get_activation_provider(
    backend: Any | None = None,
    model_path: str | Path | None = None,
    pooling: str = "auto",
) -> ActivationProvider:
    """Factory for ActivationProvider."""
    return ActivationProvider(backend=backend, model_path=model_path, pooling=pooling)


__all__ = ["ActivationProvider", "get_activation_provider"]
