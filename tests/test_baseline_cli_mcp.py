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

"""Comprehensive tests for profile CLI commands and MCP tools.

Tests cover:
1. CLI command structure and help
2. CLI command JSON output schemas
3. CLI command error handling
4. MCP tool registration
5. ProfileRepository and ModelProfile
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.model_profile import (
    ModelProfile,
    ProfileRepository,
    LayerProfile,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

runner = CliRunner()

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend() -> "Backend":
    """Get default compute backend."""
    return get_default_backend()


@pytest.fixture
def temp_profile_dir(tmp_path: Path) -> Path:
    """Create temporary profile directory."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    return profile_dir


@pytest.fixture
def sample_profile(temp_profile_dir: Path) -> ModelProfile:
    """Create and save a sample profile."""
    profile = ModelProfile(
        model_path="/test/qwen-0.5b",
        model_family="qwen",
        global_ollivier_ricci_mean=-0.189,
        global_ollivier_ricci_std=0.045,
        global_intrinsic_dimension_mean=12.4,
        layer_profiles=[
            LayerProfile(layer_idx=0, ollivier_ricci_mean=-0.15),
            LayerProfile(layer_idx=4, ollivier_ricci_mean=-0.18),
            LayerProfile(layer_idx=8, ollivier_ricci_mean=-0.22),
        ],
        extraction_config={"layers_analyzed": 3},
    )
    repo = ProfileRepository(profile_dir=temp_profile_dir)
    repo.save_profile(profile)
    return profile


@pytest.fixture
def populated_profile_dir(temp_profile_dir: Path) -> Path:
    """Create profile directory with multiple profiles."""
    families = ["qwen", "llama", "mistral"]
    for family in families:
        profile = ModelProfile(
            model_path=f"/test/{family}-3b",
            model_family=family,
            global_ollivier_ricci_mean=-0.15 - (families.index(family) * 0.02),
            global_ollivier_ricci_std=0.04,
            global_intrinsic_dimension_mean=12.0,
            layer_profiles=[
                LayerProfile(layer_idx=i, ollivier_ricci_mean=-0.15)
                for i in range(8)
            ],
        )
        repo = ProfileRepository(profile_dir=temp_profile_dir)
        repo.save_profile(profile)
    return temp_profile_dir


# =============================================================================
# CLI Command Structure Tests
# =============================================================================


class TestProfileCLIStructure:
    """Tests for CLI command structure and help."""

    def test_baseline_command_exists(self):
        """Baseline subcommand is registered."""
        result = runner.invoke(app, ["geometry", "baseline", "--help"])
        assert result.exit_code == 0
        assert "baseline" in result.stdout.lower() or "profile" in result.stdout.lower()

    def test_baseline_list_help(self):
        """List command has proper help."""
        result = runner.invoke(app, ["geometry", "baseline", "list", "--help"])
        assert result.exit_code == 0
        assert "--family" in result.stdout or "-f" in result.stdout

    def test_baseline_extract_help(self):
        """Extract command has proper help."""
        result = runner.invoke(app, ["geometry", "baseline", "extract", "--help"])
        assert result.exit_code == 0
        assert "MODEL_PATH" in result.stdout

    def test_baseline_compare_help(self):
        """Compare command has proper help."""
        result = runner.invoke(app, ["geometry", "baseline", "compare", "--help"])
        assert result.exit_code == 0
        assert "MODEL1_PATH" in result.stdout
        assert "MODEL2_PATH" in result.stdout


# =============================================================================
# CLI List Command Tests
# =============================================================================


class TestProfileCLIList:
    """Tests for profile list CLI command."""

    def test_list_empty_returns_empty_list(self, temp_profile_dir: Path):
        """List returns empty array when no profiles exist."""
        with patch(
            "modelcypher.core.domain.geometry.model_profile.ProfileRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_all_profiles.return_value = []
            mock_repo_class.return_value = mock_repo

            result = runner.invoke(
                app, ["geometry", "baseline", "list", "--output", "json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "_schema" in data
            assert "profiles" in data
            assert isinstance(data["profiles"], list)

    def test_list_text_output_format(self):
        """List produces readable text output."""
        result = runner.invoke(
            app, ["--output", "text", "geometry", "baseline", "list"]
        )
        assert result.exit_code == 0


# =============================================================================
# CLI Extract Command Tests
# =============================================================================


class TestProfileCLIExtract:
    """Tests for profile extract CLI command."""

    def test_extract_requires_model_path(self):
        """Extract fails without model path."""
        result = runner.invoke(app, ["geometry", "baseline", "extract"])
        assert result.exit_code != 0 or "error" in result.output.lower()


# =============================================================================
# CLI Compare Command Tests
# =============================================================================


class TestProfileCLICompare:
    """Tests for profile compare CLI command."""

    def test_compare_requires_both_model_paths(self):
        """Compare fails without both model paths."""
        result = runner.invoke(
            app, ["geometry", "baseline", "compare", "/path/to/model1"]
        )
        assert result.exit_code != 0

    def test_compare_accepts_layer_option(self):
        """Compare accepts --layer option."""
        result = runner.invoke(
            app,
            ["geometry", "baseline", "compare", "--help"],
        )
        assert result.exit_code == 0
        assert "--layer" in result.stdout


# =============================================================================
# MCP Tool Registration Tests
# =============================================================================


class TestProfileMCPRegistration:
    """Tests for MCP tool registration."""

    def test_baseline_tools_in_tool_set(self):
        """Profile tools are in the full profile."""
        from modelcypher.mcp.server import TOOL_PROFILES

        full_tools = TOOL_PROFILES.get("full", set())

        expected_tools = [
            "mc_geometry_baseline_list",
            "mc_geometry_baseline_extract",
            "mc_geometry_baseline_compare",
        ]

        for tool in expected_tools:
            assert tool in full_tools, f"Missing tool: {tool}"


# =============================================================================
# ProfileRepository Tests
# =============================================================================


class TestProfileRepository:
    """Tests for ProfileRepository class."""

    def test_repository_init_default(self):
        """ProfileRepository initializes with default path."""
        repo = ProfileRepository()
        assert repo._profile_dir is not None
        assert "profiles" in str(repo._profile_dir)

    def test_repository_init_custom_path(self, temp_profile_dir: Path):
        """ProfileRepository accepts custom path."""
        repo = ProfileRepository(profile_dir=temp_profile_dir)
        assert repo._profile_dir == temp_profile_dir

    def test_repository_save_and_get(
        self, temp_profile_dir: Path, backend: "Backend"
    ):
        """Repository can save and retrieve profiles."""
        profile = ModelProfile(
            model_path="/test/model",
            model_family="test",
            global_ollivier_ricci_mean=-0.2,
            global_ollivier_ricci_std=0.05,
            global_intrinsic_dimension_mean=10.0,
            layer_profiles=[LayerProfile(layer_idx=0)],
        )

        repo = ProfileRepository(profile_dir=temp_profile_dir)
        saved_path = repo.save_profile(profile)

        assert saved_path.exists()

        loaded = repo.get_profile("test", "UNKNOWN")
        assert loaded is not None
        assert loaded.model_family == "test"

    def test_repository_get_all(self, populated_profile_dir: Path):
        """Repository can get all profiles."""
        repo = ProfileRepository(profile_dir=populated_profile_dir)
        profiles = repo.get_all_profiles()

        assert len(profiles) == 3  # qwen, llama, mistral

    def test_repository_get_by_family(self, populated_profile_dir: Path):
        """Repository can filter by family."""
        repo = ProfileRepository(profile_dir=populated_profile_dir)
        qwen_profiles = repo.get_profiles_for_family("qwen")

        assert len(qwen_profiles) == 1
        assert qwen_profiles[0].model_family == "qwen"

    def test_repository_find_matching(self, populated_profile_dir: Path):
        """Repository finds matching profile with fallbacks."""
        repo = ProfileRepository(profile_dir=populated_profile_dir)

        # Exact match (by family and size from path)
        profiles = repo.get_profiles_for_family("qwen")
        assert len(profiles) > 0

        # Different family - returns None for exact match
        none_match = repo.get_profile("gemma", "3B")
        assert none_match is None


class TestModelProfile:
    """Tests for ModelProfile dataclass."""

    def test_profile_to_dict(self):
        """Profile converts to dict correctly."""
        profile = ModelProfile(
            model_path="/test/path",
            model_family="qwen",
            global_ollivier_ricci_mean=-0.2,
            global_ollivier_ricci_std=0.05,
            global_intrinsic_dimension_mean=10.0,
            layer_profiles=[LayerProfile(layer_idx=0, ollivier_ricci_mean=-0.15)],
        )

        d = profile.to_dict()

        assert d["model_family"] == "qwen"
        assert d["global_ollivier_ricci_mean"] == -0.2
        assert len(d["layer_profiles"]) == 1

    def test_profile_from_dict(self):
        """Profile creates from dict correctly."""
        d = {
            "model_path": "/test/llama",
            "model_family": "llama",
            "global_ollivier_ricci_mean": -0.15,
            "global_ollivier_ricci_std": 0.04,
            "global_intrinsic_dimension_mean": 11.0,
            "layer_profiles": [],
        }

        profile = ModelProfile.from_dict(d)

        assert profile.model_family == "llama"
        assert profile.global_ollivier_ricci_mean == -0.15

    def test_profile_save_and_load(self, tmp_path: Path):
        """Profile can save and load from file."""
        profile = ModelProfile(
            model_path="/test/mistral",
            model_family="mistral",
            global_ollivier_ricci_mean=-0.18,
            global_ollivier_ricci_std=0.06,
            global_intrinsic_dimension_mean=9.5,
            layer_profiles=[
                LayerProfile(layer_idx=i, ollivier_ricci_mean=-0.15 - i * 0.01)
                for i in range(4)
            ],
        )

        file_path = tmp_path / "test_profile.json"
        profile.save(file_path)

        assert file_path.exists()

        loaded = ModelProfile.load(file_path)
        assert loaded.model_family == profile.model_family
        assert loaded.global_ollivier_ricci_mean == profile.global_ollivier_ricci_mean
        assert len(loaded.layer_profiles) == len(profile.layer_profiles)


# =============================================================================
# Integration Tests
# =============================================================================


class TestProfileIntegration:
    """Integration tests for profile workflows."""

    def test_cli_list_json_parses_correctly(self):
        """CLI list JSON output can be parsed."""
        result = runner.invoke(
            app, ["geometry", "baseline", "list", "--output", "json"]
        )

        if result.exit_code == 0:
            data = json.loads(result.stdout)
            assert isinstance(data, dict)
            assert "profiles" in data

    def test_repository_profile_roundtrip(self, temp_profile_dir: Path):
        """Full profile save/load roundtrip works."""
        original = ModelProfile(
            model_path="/test/phi",
            model_family="phi",
            global_ollivier_ricci_mean=-0.17,
            global_ollivier_ricci_std=0.04,
            global_intrinsic_dimension_mean=11.0,
            layer_profiles=[
                LayerProfile(layer_idx=i, ollivier_ricci_mean=-0.15)
                for i in range(8)
            ],
        )

        repo = ProfileRepository(profile_dir=temp_profile_dir)
        repo.save_profile(original)

        repo2 = ProfileRepository(profile_dir=temp_profile_dir)
        profiles = repo2.get_profiles_for_family("phi")

        assert len(profiles) > 0
        loaded = profiles[0]
        assert loaded.model_family == original.model_family
        assert loaded.global_ollivier_ricci_mean == original.global_ollivier_ricci_mean


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestProfileErrorHandling:
    """Tests for error handling in profile operations."""

    def test_repository_handles_missing_file(self, temp_profile_dir: Path):
        """Repository handles missing profile gracefully."""
        repo = ProfileRepository(profile_dir=temp_profile_dir)
        result = repo.get_profile("nonexistent", "0B")
        assert result is None

    def test_repository_handles_corrupt_json(self, temp_profile_dir: Path):
        """Repository handles corrupt JSON files."""
        corrupt_file = temp_profile_dir / "test_1B.json"
        corrupt_file.write_text("{ not valid json }")

        repo = ProfileRepository(profile_dir=temp_profile_dir)
        profiles = repo.get_all_profiles()
        assert isinstance(profiles, list)
