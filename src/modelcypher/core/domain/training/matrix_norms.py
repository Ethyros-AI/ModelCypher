# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Exact matrix norms used by quantization-training diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def compute_spectral_norm(
    matrix: Any,
    backend: "Backend",
) -> float:
    """Return the exact spectral norm via the top singular value."""
    matrix_f32 = backend.astype(matrix, "float32")
    backend.eval(matrix_f32)
    singular_values = backend.svd(matrix_f32, compute_uv=False)
    backend.eval(singular_values)
    if int(singular_values.shape[0]) <= 0:
        return 0.0
    return float(backend.to_scalar(singular_values[0]))


__all__ = ["compute_spectral_norm"]
