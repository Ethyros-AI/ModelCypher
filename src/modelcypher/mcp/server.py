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

import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from modelcypher.infrastructure.container import PortRegistry
from modelcypher.infrastructure.service_factory import ServiceFactory
from modelcypher.mcp.security import ConfirmationManager, SecurityConfig
from modelcypher.mcp.tools.agent import register_agent_tools
from modelcypher.mcp.tools.common import ServiceContext
from modelcypher.mcp.tools.evaluation import register_evaluation_tools
from modelcypher.mcp.tools.geometry import register_all_geometry_tools
from modelcypher.mcp.tools.inference import register_inference_tools
from modelcypher.mcp.tools.merge_entropy import register_merge_entropy_tools
from modelcypher.mcp.tools.model import register_model_tools
from modelcypher.mcp.tools.safety_entropy import (
    register_entropy_tools,
    register_safety_tools,
)
from modelcypher.mcp.tools.tasks import register_task_tools
from modelcypher.mcp.tools.thermo import register_thermo_tools
from modelcypher.mcp.tools.training import register_training_tools

_TOOL_NAME_PATTERN = re.compile(r"""['"](mc_[a-zA-Z0-9_]+)['"]""")


def _discover_tool_names() -> set[str]:
    tool_dir = Path(__file__).resolve().parent / "tools"
    tool_names: set[str] = set()
    for path in tool_dir.rglob("*.py"):
        if path.name in {"__init__.py", "common.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        tool_names.update(_TOOL_NAME_PATTERN.findall(text))
    return tool_names


TOOL_PROFILES: dict[str, set[str]] = {
    "full": _discover_tool_names(),
}


def build_server(profile: str | None = None) -> FastMCP:
    profile_name = profile or os.environ.get("MC_MCP_PROFILE", "full")
    tool_set = set(TOOL_PROFILES.get(profile_name, TOOL_PROFILES["full"]))

    mcp = FastMCP("ModelCypher")
    registry = PortRegistry.create_production()
    factory = ServiceFactory(registry)
    security_config = SecurityConfig.from_env()
    confirmation_manager = ConfirmationManager(security_config)
    ctx = ServiceContext(
        mcp=mcp,
        tool_set=tool_set,
        security_config=security_config,
        confirmation_manager=confirmation_manager,
        registry=registry,
        factory=factory,
    )

    register_all_geometry_tools(ctx)
    register_model_tools(ctx)
    register_inference_tools(ctx)
    register_training_tools(ctx)
    register_task_tools(ctx)
    register_merge_entropy_tools(ctx)
    register_safety_tools(ctx)
    register_entropy_tools(ctx)
    register_agent_tools(ctx)
    register_thermo_tools(ctx)
    register_evaluation_tools(ctx)

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
