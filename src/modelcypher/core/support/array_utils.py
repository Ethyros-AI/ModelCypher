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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def array_to_list(backend: "Backend", array: "Array") -> Any:
    """Convert a backend array into nested Python lists.

    Uses native backend tolist() which is MUCH faster than
    element-by-element to_scalar() extraction.
    """
    return backend.tolist(array)


def array_to_flat_list(backend: "Backend", array: "Array") -> list[float]:
    """Convert array to flattened Python list.

    Flattens the array to 1D then converts using native tolist().
    """
    flat = backend.reshape(array, (-1,))
    return backend.tolist(flat)
