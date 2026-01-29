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

"""Spectral blending (deprecated).

Geometric addition replaces blending for knowledge transfer. This stub blocks
legacy spectral blend paths from re-entering the core pipeline.
"""

from __future__ import annotations


def compute_spectral_blend(*_args, **_kwargs):
    raise RuntimeError(
        "Spectral blending is deprecated. Use null-space addition "
        "with geometry-derived projection instead."
    )


def compute_adaptive_spectral_blend(*_args, **_kwargs):
    raise RuntimeError(
        "Spectral blending is deprecated. Use null-space addition "
        "with geometry-derived projection instead."
    )


__all__ = [
    "compute_spectral_blend",
    "compute_adaptive_spectral_blend",
]
