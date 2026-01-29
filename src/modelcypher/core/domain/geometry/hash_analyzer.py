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

"""Hash analyzer placeholder.

The NumPy-based implementation is located in
`modelcypher.adapters.geometry.hash_analyzer` to keep the domain layer
backend-only.
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - explicit import guidance
    raise RuntimeError(
        "Hash analyzer utilities live in "
        "`modelcypher.adapters.geometry.hash_analyzer`. "
        "Import from adapters to use this functionality."
    )


__all__ = []
