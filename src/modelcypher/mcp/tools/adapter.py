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

"""Adapter MCP tools."""

from __future__ import annotations

from modelcypher.mcp.tools.common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_adapter_tools(ctx: ServiceContext) -> None:
    """Register adapter-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_adapter_inspect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_adapter_inspect(adapterPath: str) -> dict:
            """Inspect adapter weights and metadata."""
            adapter_path = require_existing_directory(adapterPath)
            result = ctx.adapter_service.inspect(adapter_path)
            return {
                "_schema": "mc.adapter.inspect.v1",
                "adapterPath": adapter_path,
                "rank": result.rank,
                "alpha": result.alpha,
                "targetModules": result.target_modules,
                "sparsity": result.sparsity,
                "parameterCount": result.parameter_count,
                "layerAnalysis": [
                    {
                        "name": layer.name,
                        "rank": layer.rank,
                        "alpha": layer.alpha,
                        "parameters": layer.parameters,
                    }
                    for layer in result.layer_analysis
                ],
            }
