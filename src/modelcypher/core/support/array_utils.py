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
    """Convert a backend array into nested Python lists without NumPy."""
    shape = getattr(array, "shape", None)
    if shape is None:
        return array
    if len(shape) == 0:
        return backend.to_scalar(array)
    if len(shape) == 1:
        count = int(shape[0])
        return [backend.to_scalar(array[i]) for i in range(count)]
    count = int(shape[0])
    return [array_to_list(backend, array[i]) for i in range(count)]
