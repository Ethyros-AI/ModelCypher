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

"""Shared transplant helpers (precision, pinv, and matrix utilities)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)


if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _geodesic_pinv(backend: "Backend", F: "Array") -> "Array":
    """Compute exact Moore-Penrose pseudo-inverse.

    Raises ValueError if SVD fails (ill-conditioned matrix).
    Regularization is NOT used as it changes the mathematical semantics
    and violates the CKA=1.0 invariant for alignment.
    """
    b = backend
    F = _promote_precision(F, b)
    b.eval(F)

    try:
        F_pinv = b.pinv(F)
        b.eval(F_pinv)
    except Exception as e:
        # Don't regularize - that changes the answer and violates CKA=1.0 invariant
        raise ValueError(
            f"Alignment matrix too ill-conditioned for exact pinv: {e}"
        ) from e

    return F_pinv


def _set_submatrix(
    backend: "Backend",
    target: "Array",
    source: "Array",
    row_offset: int,
    col_offset: int,
) -> "Array":
    """Set a submatrix of target from source at the given offset."""
    src_rows, src_cols = int(source.shape[0]), int(source.shape[1])
    tgt_rows, tgt_cols = int(target.shape[0]), int(target.shape[1])

    if src_rows == 0 or src_cols == 0:
        return target

    row_end = row_offset + src_rows
    col_end = col_offset + src_cols

    mid_parts = []
    if col_offset > 0:
        mid_parts.append(target[row_offset:row_end, :col_offset])
    mid_parts.append(source)
    if col_end < tgt_cols:
        mid_parts.append(target[row_offset:row_end, col_end:])

    mid = mid_parts[0] if len(mid_parts) == 1 else backend.concatenate(mid_parts, axis=1)

    row_parts = []
    if row_offset > 0:
        row_parts.append(target[:row_offset, :])
    row_parts.append(mid)
    if row_end < tgt_rows:
        row_parts.append(target[row_end:, :])

    result = row_parts[0] if len(row_parts) == 1 else backend.concatenate(row_parts, axis=0)
    backend.eval(result)
    return result


def _compute_dimension_projection(
    backend: "Backend",
    src_dim: int,
    tgt_dim: int,
) -> "Array":
    """Compute an orthogonal projection matrix between dimensions."""
    dtype = backend.eye(1).dtype
    if src_dim == tgt_dim:
        return backend.eye(src_dim, dtype=dtype)

    min_dim = min(src_dim, tgt_dim)

    identity_block = backend.eye(min_dim, dtype=dtype)

    if tgt_dim < src_dim:
        zeros_below = backend.zeros((src_dim - min_dim, tgt_dim), dtype=dtype)
        projection = backend.concatenate([identity_block, zeros_below], axis=0)
    else:
        zeros_right = backend.zeros((src_dim, tgt_dim - min_dim), dtype=dtype)
        projection = backend.concatenate([identity_block, zeros_right], axis=1)

    backend.eval(projection)
    return projection
