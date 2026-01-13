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

"""Bridge module for cross-modal knowledge transfer.

This module provides tools for creating affine bridges between encoder spaces.
Linear alignment is closed-form; geodesic CKA reports manifold overlap across
modalities (vision, audio, text, diffusion).

Key insight: The geometry is discovered, not created. Different encoders learn
the same invariant shape - bridges are just coordinate transforms.
"""

from modelcypher.core.domain.bridge.generator import (
    BridgeGenerator,
    BridgeGeneratorResult,
    CrossModalBridge,
)

__all__ = [
    "BridgeGenerator",
    "BridgeGeneratorResult",
    "CrossModalBridge",
]
