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

"""Shared helpers for Riemannian core operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.numerical_stability import precision_dtype

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _dtype_name(dtype: Any) -> str:
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    return name.replace("mlx.core.", "").replace("jax.numpy.", "")


def _default_float_dtype(backend: "Backend") -> Any:
    """Return the model-driven compute dtype for Riemannian operations."""
    return precision_dtype(backend)


def _float_dtype_for(array: "Array | None", backend: "Backend") -> Any:
    if array is not None and hasattr(array, "dtype"):
        name = _dtype_name(array.dtype)
        if "float" in name:
            return array.dtype
    return _default_float_dtype(backend)


def _promote_precision(
    array: "Array",
    backend: "Backend",
    *,
    min_dtype: Any | None = None,
) -> "Array":
    """Promote low-precision or integer arrays to at least float32/default float."""
    if min_dtype is None:
        min_dtype = _default_float_dtype(backend)

    if not hasattr(array, "dtype"):
        return backend.array(array, dtype=min_dtype)

    dtype_name = _dtype_name(array.dtype)
    if (
        "float16" in dtype_name
        or "bfloat16" in dtype_name
        or "int" in dtype_name
        or "uint" in dtype_name
        or "bool" in dtype_name
    ):
        return backend.astype(array, min_dtype)

    try:
        current_eps = backend.finfo(array.dtype).eps
        min_eps = backend.finfo(min_dtype).eps
    except Exception:
        return backend.astype(array, min_dtype)

    if current_eps > min_eps:
        return backend.astype(array, min_dtype)

    return array


def _count_mask(
    mask: "Array",
    backend: "Backend",
    *,
    dtype_source: "Array | None" = None,
) -> "Array":
    dtype = _float_dtype_for(dtype_source, backend)
    return backend.sum(backend.astype(mask, dtype))


def _count_not_mask(
    mask: "Array",
    backend: "Backend",
    *,
    dtype_source: "Array | None" = None,
) -> "Array":
    dtype = _float_dtype_for(dtype_source, backend)
    mask_float = backend.astype(mask, dtype)
    return backend.sum(1 - mask_float)
