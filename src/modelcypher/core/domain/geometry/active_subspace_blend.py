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

"""Active subspace blending (deprecated).

Blending is incompatible with geometric addition. This module is retained as a
stub to prevent accidental use in core workflows.
"""

from __future__ import annotations


def compute_active_subspace_blend(*_args, **_kwargs):
    raise RuntimeError(
        "Active-subspace blending is deprecated. Use null-space addition "
        "with geometry-derived projection instead."
    )


__all__ = ["compute_active_subspace_blend"]
