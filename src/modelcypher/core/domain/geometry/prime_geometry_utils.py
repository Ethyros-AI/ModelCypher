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

"""Prime geometry helper utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _array_to_list(backend: "Backend", array: "Array") -> list[float]:
    """Convert 1D array to Python list using native tolist() - O(1) vs O(n)."""
    flat = backend.reshape(array, (-1,))
    return backend.tolist(flat)


def _uniform_list(backend: "Backend", count: int) -> list[float]:
    """Draw uniform [0,1) samples via backend and return as Python list."""
    if count <= 0:
        return []
    vals = backend.random_uniform(low=0.0, high=1.0, shape=(count,))
    backend.eval(vals)
    return [float(x) for x in backend.tolist(vals)]


def _randint_list(backend: "Backend", low: int, high: int, count: int) -> list[int]:
    """Draw integer samples via backend and return as Python list."""
    if count <= 0:
        return []
    vals = backend.random_randint(low, high, shape=(count,))
    backend.eval(vals)
    return [int(x) for x in backend.tolist(vals)]


def _uniform_sampler(backend: "Backend", batch_size: int = 1024):
    """Return a callable that yields uniform samples from a buffered pool."""
    pool: list[float] = []
    idx = 0

    def next_uniform() -> float:
        nonlocal pool, idx
        if idx >= len(pool):
            pool = _uniform_list(backend, batch_size)
            idx = 0
        val = float(pool[idx])
        idx += 1
        return val

    return next_uniform
