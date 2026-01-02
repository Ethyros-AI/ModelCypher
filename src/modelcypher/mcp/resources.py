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

"""MCP resource registrations."""

from __future__ import annotations

from modelcypher.mcp.tools.common import READ_ONLY_ANNOTATIONS, ServiceContext


def register_system_resources(ctx: ServiceContext) -> None:
    """Register system resources for MCP clients."""
    mcp = ctx.mcp

    @mcp.resource("mc://system", mime_type="application/json", annotations=READ_ONLY_ANNOTATIONS)
    def mc_system_resource() -> dict:
        """Return system readiness details."""
        status = ctx.system_service.status()
        return {"_schema": "mc.system.status.v1", **status}
