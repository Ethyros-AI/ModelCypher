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

from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.precision import (
    _float_dtype_for,
    _mask_sum,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _count_mask(
    mask: "Array",
    backend: "Backend",
    *,
    dtype_source: "Array | None" = None,
) -> "Array":
    return _mask_sum(mask, backend, dtype_source=dtype_source)


def _count_not_mask(
    mask: "Array",
    backend: "Backend",
    *,
    dtype_source: "Array | None" = None,
) -> "Array":
    dtype = _float_dtype_for(dtype_source, backend)
    mask_float = backend.astype(mask, dtype)
    return backend.sum(1 - mask_float)
