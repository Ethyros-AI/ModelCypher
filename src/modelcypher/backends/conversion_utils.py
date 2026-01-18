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

"""Shared array conversion utilities for backend implementations."""

from __future__ import annotations

from typing import Any, Callable


_NUMPY_DISABLED_MESSAGE = (
    "to_numpy() is disabled. ModelCypher does not permit CPU arrays. "
    "Use backend.tolist() for lists, backend.to_scalar() for scalars, "
    "or backend.save_safetensors() for serialization."
)


def raise_numpy_disabled() -> None:
    raise RuntimeError(_NUMPY_DISABLED_MESSAGE)


def to_scalar_with_eval(array: Any, eval_fn: Callable[..., None]) -> float | int:
    if hasattr(array, "shape"):
        eval_fn(array)
    if hasattr(array, "item"):
        return array.item()
    return float(array)


def to_list_with_eval(array: Any, eval_fn: Callable[..., None]) -> list | float | int:
    eval_fn(array)
    return array.tolist()
