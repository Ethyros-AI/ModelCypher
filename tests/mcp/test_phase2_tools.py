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

"""Tests for Phase 2 MCP tool modules."""

from unittest.mock import MagicMock

import pytest


class TestServiceContext:
    """Tests for the ServiceContext helper class."""

    def test_service_context_creation(self):
        """ServiceContext can be created with required fields."""
        from modelcypher.mcp.tools.common import ServiceContext

        mock_mcp = MagicMock()
        mock_security = MagicMock()
        mock_confirmation = MagicMock()
        mock_registry = MagicMock()
        mock_factory = MagicMock()

        ctx = ServiceContext(
            mcp=mock_mcp,
            tool_set={"mc_test_tool"},
            security_config=mock_security,
            confirmation_manager=mock_confirmation,
            registry=mock_registry,
            factory=mock_factory,
        )

        assert ctx.mcp == mock_mcp
        assert "mc_test_tool" in ctx.tool_set
        assert ctx.security_config == mock_security

    def test_service_context_lazy_loading(self):
        """Services are lazily loaded on first access."""
        from modelcypher.mcp.tools.common import ServiceContext

        mock_factory = MagicMock()
        mock_inventory = MagicMock()
        mock_factory.inventory_service.return_value = mock_inventory

        ctx = ServiceContext(
            mcp=MagicMock(),
            tool_set=set(),
            security_config=MagicMock(),
            confirmation_manager=MagicMock(),
            registry=MagicMock(),
            factory=mock_factory,
        )

        # Initially None
        assert ctx._inventory_service is None

        # Access triggers lazy load via factory
        service = ctx.inventory_service
        assert service is mock_inventory
        assert ctx._inventory_service is mock_inventory
        mock_factory.inventory_service.assert_called_once()

    def test_idempotency_cache(self):
        """Idempotency cache stores and retrieves values."""
        from modelcypher.mcp.tools.common import ServiceContext

        ctx = ServiceContext(
            mcp=MagicMock(),
            tool_set=set(),
            security_config=MagicMock(),
            confirmation_manager=MagicMock(),
            registry=MagicMock(),
            factory=MagicMock(),
        )

        # Initially empty
        assert ctx.get_idempotency("test", "key1") is None

        # Set value
        ctx.set_idempotency("test", "key1", "value1")

        # Retrieve value
        assert ctx.get_idempotency("test", "key1") == "value1"


class TestCommonHelpers:
    """Tests for common helper functions."""

    def test_require_existing_path_valid(self, tmp_path):
        """require_existing_path returns resolved path for existing files."""
        from modelcypher.mcp.tools.common import require_existing_path

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = require_existing_path(str(test_file))
        assert result == str(test_file.resolve())

    def test_require_existing_path_invalid(self):
        """require_existing_path raises for non-existent paths."""
        from modelcypher.mcp.tools.common import require_existing_path

        with pytest.raises(ValueError, match="does not exist"):
            require_existing_path("/nonexistent/path/file.txt")

    def test_require_existing_directory_valid(self, tmp_path):
        """require_existing_directory returns path for existing directories."""
        from modelcypher.mcp.tools.common import require_existing_directory

        result = require_existing_directory(str(tmp_path))
        assert result == str(tmp_path.resolve())

    def test_require_existing_directory_file(self, tmp_path):
        """require_existing_directory raises for files (not directories)."""
        from modelcypher.mcp.tools.common import require_existing_directory

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Directory does not exist"):
            require_existing_directory(str(test_file))

    def test_map_job_status(self):
        """map_job_status converts internal statuses to external."""
        from modelcypher.mcp.tools.common import map_job_status

        assert map_job_status("pending") == "queued"
        assert map_job_status("cancelled") == "canceled"
        assert map_job_status("running") == "running"


class TestSafetyTools:
    """Tests for Phase 2 safety tool registration."""

    def test_register_safety_tools_no_crash(self):
        """Safety tools can be registered without errors."""
        from modelcypher.mcp.tools.common import ServiceContext
        from modelcypher.mcp.tools.safety_entropy import register_safety_tools

        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock(return_value=lambda f: f)

        ctx = ServiceContext(
            mcp=mock_mcp,
            tool_set={
                "mc_safety_adapter_probe",
            },
            security_config=MagicMock(),
            confirmation_manager=MagicMock(),
            registry=MagicMock(),
            factory=MagicMock(),
        )

        # Should not raise
        register_safety_tools(ctx)


class TestEntropyTools:
    """Tests for Phase 2 entropy tool registration."""

    def test_register_entropy_tools_no_crash(self):
        """Entropy tools can be registered without errors."""
        from modelcypher.mcp.tools.common import ServiceContext
        from modelcypher.mcp.tools.safety_entropy import register_entropy_tools

        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock(return_value=lambda f: f)

        ctx = ServiceContext(
            mcp=mock_mcp,
            tool_set={
                "mc_entropy_window",
                "mc_entropy_conversation_track",
                "mc_entropy_dual_path",
            },
            security_config=MagicMock(),
            confirmation_manager=MagicMock(),
            registry=MagicMock(),
            factory=MagicMock(),
        )

        # Should not raise
        register_entropy_tools(ctx)


class TestAgentTools:
    """Tests for Phase 2 agent tool registration."""

    def test_register_agent_tools_no_crash(self):
        """Agent tools can be registered without errors."""
        from modelcypher.mcp.tools.agent import register_agent_tools
        from modelcypher.mcp.tools.common import ServiceContext

        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock(return_value=lambda f: f)

        ctx = ServiceContext(
            mcp=mock_mcp,
            tool_set={
                "mc_agent_trace_import",
                "mc_agent_trace_analyze",
                "mc_agent_validate_action",
            },
            security_config=MagicMock(),
            confirmation_manager=MagicMock(),
            registry=MagicMock(),
            factory=MagicMock(),
        )

        # Should not raise
        register_agent_tools(ctx)


class TestMCPServerIntegration:
    """Integration tests for the full MCP server."""

    def test_server_builds_with_phase2_tools(self):
        """Server builds successfully with Phase 2 tools in profile."""
        import os

        os.environ["MC_MCP_PROFILE"] = "full"

        from modelcypher.mcp.server import build_server

        server = build_server()
        assert server is not None
        assert server.name == "ModelCypher"

    def test_phase2_tools_in_profiles(self):
        """Phase 2 tool names are in TOOL_PROFILES."""
        from modelcypher.mcp.server import TOOL_PROFILES

        full_profile = TOOL_PROFILES["full"]

        # Safety tools
        assert "mc_safety_adapter_probe" in full_profile

        # Entropy tools
        assert "mc_entropy_window" in full_profile
        assert "mc_entropy_conversation_track" in full_profile
        assert "mc_entropy_dual_path" in full_profile

        # Agent tools
        assert "mc_agent_trace_import" in full_profile
        assert "mc_agent_trace_analyze" in full_profile
        assert "mc_agent_validate_action" in full_profile
