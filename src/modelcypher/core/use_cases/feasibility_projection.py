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

"""Projection helpers for feasibility-map memory extrapolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_BYTES_PER_GIB = 1024**3


def static_weight_memory_gib(param_count: int, precision_bits: int) -> float:
    """Return idealized weight memory in GiB for params/bits."""
    if param_count < 0:
        raise ValueError("param_count must be >= 0")
    if precision_bits <= 0:
        raise ValueError("precision_bits must be > 0")
    return (float(param_count) * float(precision_bits) / 8.0) / float(_BYTES_PER_GIB)


@dataclass(frozen=True)
class RuntimeOverhead:
    """Empirical runtime overhead terms measured from profiled runs."""

    load_overhead_gib: float
    forward_delta_gib: float
    decode_slope_gib_per_token: float


def mean_runtime_overhead(samples: Iterable[RuntimeOverhead]) -> RuntimeOverhead:
    """Compute mean overhead terms across profiled runs."""
    rows = list(samples)
    if not rows:
        raise ValueError("at least one runtime overhead sample is required")
    n = float(len(rows))
    return RuntimeOverhead(
        load_overhead_gib=sum(r.load_overhead_gib for r in rows) / n,
        forward_delta_gib=sum(r.forward_delta_gib for r in rows) / n,
        decode_slope_gib_per_token=sum(r.decode_slope_gib_per_token for r in rows) / n,
    )


def project_memory_gib(
    param_count: int,
    precision_bits: int,
    overhead: RuntimeOverhead,
    decode_tokens: int,
) -> dict[str, float]:
    """Project load/forward/decode memory envelopes for a target parameter count."""
    if decode_tokens < 0:
        raise ValueError("decode_tokens must be >= 0")
    static_gib = static_weight_memory_gib(param_count, precision_bits)
    load_active = static_gib + overhead.load_overhead_gib
    forward_active = load_active + overhead.forward_delta_gib
    decode_active = forward_active + overhead.decode_slope_gib_per_token * float(decode_tokens)
    return {
        "static_weight_gib": static_gib,
        "load_active_gib": load_active,
        "forward_active_gib": forward_active,
        "decode_active_gib": decode_active,
    }


__all__ = [
    "RuntimeOverhead",
    "mean_runtime_overhead",
    "project_memory_gib",
    "static_weight_memory_gib",
]

