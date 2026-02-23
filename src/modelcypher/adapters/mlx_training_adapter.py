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

"""Compatibility adapter for MLX training.

Implementation lives in :mod:`modelcypher.backends.mlx_training_adapter` so all
framework imports remain confined to ``src/modelcypher/backends/``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_IMPL_MODULE = "modelcypher.backends.mlx_training_adapter"


def _impl() -> Any:
    return import_module(_IMPL_MODULE)


def __getattr__(name: str) -> Any:
    return getattr(_impl(), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl())))


# Eagerly bind commonly imported symbols for static tooling and direct imports.
EpochMetrics = _impl().EpochMetrics
MLXTrainingAdapter = _impl().MLXTrainingAdapter
iterate_structured_batches = _impl().iterate_structured_batches
make_geometric_reshaping_loss = _impl().make_geometric_reshaping_loss
make_constrained_loss = _impl().make_constrained_loss
calibrate_geometric_weights = _impl().calibrate_geometric_weights
NBLoRALinear = _impl().NBLoRALinear


__all__ = [
    "EpochMetrics",
    "MLXTrainingAdapter",
    "iterate_structured_batches",
    "make_geometric_reshaping_loss",
    "make_constrained_loss",
    "calibrate_geometric_weights",
    "NBLoRALinear",
]
