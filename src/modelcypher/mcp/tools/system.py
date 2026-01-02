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

"""System MCP tools.

Provides inventory, system status, and settings snapshot tools.
"""

from __future__ import annotations

from modelcypher.mcp.tools.common import READ_ONLY_ANNOTATIONS, ServiceContext


def register_system_tools(ctx: ServiceContext) -> None:
    """Register system-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_inventory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_inventory() -> dict:
            """Return inventory snapshot for models, jobs, and checkpoints."""
            payload = ctx.inventory_service.inventory()
            payload["_schema"] = "mc.inventory.v1"
            return payload

    if "mc_system_status" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_system_status() -> dict:
            """Return system readiness and backend availability."""
            status = ctx.system_service.status()
            return {"_schema": "mc.system.status.v1", **status}

    if "mc_settings_snapshot" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_settings_snapshot() -> dict:
            """Return current settings snapshot."""
            snapshot = ctx.settings_service.snapshot()
            return {"_schema": "mc.settings.snapshot.v1", **snapshot.as_dict()}
