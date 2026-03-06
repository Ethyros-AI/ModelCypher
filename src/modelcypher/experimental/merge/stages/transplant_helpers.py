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

from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _geodesic_pinv(backend: "Backend", F: "Array") -> "Array":
    """Compute a Moore-Penrose pseudoinverse using the backend SVD.

    Raises ValueError if SVD fails (ill-conditioned matrix).
    Regularization is not used because it changes the computed pseudoinverse.
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
    """Compute a dimension projection matrix.

    Same-dimension identity case only. Cross-dimension projections must come
    from measured alignment transforms.

    Returns identity when src_dim == tgt_dim; otherwise raises RuntimeError to
    signal that alignment-derived stitching is required.
    """
    if src_dim == tgt_dim:
        # Same dimensions: identity is exact.
        dtype = backend.eye(1).dtype
        return backend.eye(src_dim, dtype=dtype)

    # Different dimensions: we CANNOT guess the projection
    # The geometry must come from alignment, not [[I, 0]]
    raise RuntimeError(
        f"Cannot project between dimensions {src_dim} → {tgt_dim} without "
        f"alignment-derived transform. The [[I, 0]] pattern is geometrically "
        f"wrong (10x more error than H-derived). Ensure probe stage computes "
        f"proper stitches for this weight type."
    )
