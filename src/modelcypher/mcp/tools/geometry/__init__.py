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

"""Geometry MCP tools package.

This package contains geometry-related MCP tools organized by functional area:
- core: Path detection, metrics, sparse regions, personas, manifold, transport
- invariant: Invariant layer mapping and atlas tools
- safety: Jailbreak testing, DARE sparsity, DoRA decomposition
- primes: Semantic prime probing and comparison
- crm: Concept Response Matrix building and comparison
- spatial: 3D world model analysis (Euclidean, gravity, density)
- interference: Interference prediction and null-space filtering
- baseline: Domain geometry baseline extraction and validation
- visualize: Real-time 3D manifold visualization (curvature, density, trajectories)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .baseline import register_geometry_baseline_tools
from .core import register_geometry_tools
from .crm import register_geometry_crm_tools
from .interference import register_geometry_interference_tools
from .invariant import register_geometry_invariant_tools
from .primes import register_geometry_primes_tools
from .safety import register_geometry_safety_tools
from .spatial import register_geometry_spatial_tools
from .visualize import register_geometry_visualize_tools

if TYPE_CHECKING:
    from ..common import ServiceContext

__all__ = [
    "register_geometry_tools",
    "register_geometry_invariant_tools",
    "register_geometry_safety_tools",
    "register_geometry_primes_tools",
    "register_geometry_crm_tools",
    "register_geometry_spatial_tools",
    "register_geometry_interference_tools",
    "register_geometry_baseline_tools",
    "register_geometry_visualize_tools",
    "register_all_geometry_tools",
]


def register_all_geometry_tools(ctx: "ServiceContext") -> None:
    """Register all geometry MCP tools.

    This is the main entry point for the MCP server to register all geometry tools.
    """
    register_geometry_tools(ctx)
    register_geometry_invariant_tools(ctx)
    register_geometry_safety_tools(ctx)
    register_geometry_primes_tools(ctx)
    register_geometry_crm_tools(ctx)
    register_geometry_spatial_tools(ctx)
    register_geometry_interference_tools(ctx)
    register_geometry_baseline_tools(ctx)
    register_geometry_visualize_tools(ctx)
