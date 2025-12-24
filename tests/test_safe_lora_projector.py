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

"""Tests for Safe LoRA projector."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from modelcypher.core.domain.safety.safe_lora_projector import (
    SafeLoRAConfiguration,
    SafeLoRAProjectionResult,
    SafeLoRAProjectionStatus,
    SafeLoRAProjector,
)


class TestSafeLoRAProjectionStatus:
    """Tests for SafeLoRAProjectionStatus enum."""

    def test_enum_values(self) -> None:
        """All expected statuses exist."""
        assert SafeLoRAProjectionStatus.APPLIED == "applied"
        assert SafeLoRAProjectionStatus.SKIPPED == "skipped"
        assert SafeLoRAProjectionStatus.UNAVAILABLE == "unavailable"


class TestSafeLoRAProjectionResult:
    """Tests for SafeLoRAProjectionResult dataclass."""

    def test_applied_factory(self) -> None:
        """applied() factory creates correct result."""
        result = SafeLoRAProjectionResult.applied(details="test details")
        assert result.status == SafeLoRAProjectionStatus.APPLIED
        assert result.was_applied is True
        assert result.is_available is True
        assert result.details == "test details"
        assert result.warnings == ()

    def test_skipped_factory(self) -> None:
        """skipped() factory creates correct result."""
        result = SafeLoRAProjectionResult.skipped(
            warnings=("warning1", "warning2"), details="skipped reason"
        )
        assert result.status == SafeLoRAProjectionStatus.SKIPPED
        assert result.was_applied is False
        assert result.is_available is True
        assert result.warnings == ("warning1", "warning2")
        assert result.details == "skipped reason"

    def test_unavailable_factory(self) -> None:
        """unavailable() factory creates correct result."""
        result = SafeLoRAProjectionResult.unavailable("no cache found")
        assert result.status == SafeLoRAProjectionStatus.UNAVAILABLE
        assert result.was_applied is False
        assert result.is_available is False
        assert result.warnings == ("no cache found",)

    def test_was_applied_property(self) -> None:
        """was_applied is True only for APPLIED status."""
        assert SafeLoRAProjectionResult.applied().was_applied is True
        assert SafeLoRAProjectionResult.skipped().was_applied is False
        assert SafeLoRAProjectionResult.unavailable("x").was_applied is False

    def test_is_available_property(self) -> None:
        """is_available is False only for UNAVAILABLE status."""
        assert SafeLoRAProjectionResult.applied().is_available is True
        assert SafeLoRAProjectionResult.skipped().is_available is True
        assert SafeLoRAProjectionResult.unavailable("x").is_available is False


class TestSafeLoRAProjector:
    """Tests for SafeLoRAProjector class."""

    def test_sanitize_model_id_with_slashes(self) -> None:
        """_sanitize replaces slashes with underscores."""
        assert SafeLoRAProjector._sanitize("mlx-community/Llama-3.2-3B") == "mlx-community_Llama-3.2-3B"

    def test_sanitize_model_id_with_colons(self) -> None:
        """_sanitize replaces colons with underscores."""
        assert SafeLoRAProjector._sanitize("model:variant") == "model_variant"

    def test_sanitize_model_id_with_spaces(self) -> None:
        """_sanitize replaces spaces with underscores."""
        assert SafeLoRAProjector._sanitize("model name") == "model_name"

    def test_sanitize_model_id_combined(self) -> None:
        """_sanitize handles multiple special characters."""
        assert SafeLoRAProjector._sanitize("org/model:v1 beta") == "org_model_v1_beta"

    def test_find_projection_file_no_resources_path(self) -> None:
        """_find_projection_file returns None when resources_path is None."""
        projector = SafeLoRAProjector(resources_path=None)
        result = projector._find_projection_file("safety/projections/test")
        assert result is None

    def test_find_projection_file_nonexistent_dir(self, tmp_path: Path) -> None:
        """_find_projection_file returns None for nonexistent directory."""
        projector = SafeLoRAProjector(resources_path=tmp_path)
        result = projector._find_projection_file("nonexistent/path")
        assert result is None

    def test_find_projection_file_missing_safetensors(self, tmp_path: Path) -> None:
        """_find_projection_file returns None when safetensors file missing."""
        subdir = tmp_path / "safety" / "projections" / "test_model"
        subdir.mkdir(parents=True)
        projector = SafeLoRAProjector(resources_path=tmp_path)
        result = projector._find_projection_file("safety/projections/test_model")
        assert result is None

    def test_find_projection_file_exists(self, tmp_path: Path) -> None:
        """_find_projection_file returns path when safetensors exists."""
        subdir = tmp_path / "safety" / "projections" / "test_model"
        subdir.mkdir(parents=True)
        projection_file = subdir / "projection.safetensors"
        projection_file.write_bytes(b"dummy")

        projector = SafeLoRAProjector(resources_path=tmp_path)
        result = projector._find_projection_file("safety/projections/test_model")
        assert result == projection_file

    def test_project_unavailable_no_cache(self, tmp_path: Path) -> None:
        """project() returns UNAVAILABLE when no cache exists."""
        projector = SafeLoRAProjector(resources_path=tmp_path)
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()

        result = asyncio.run(projector.project("some/model", adapter_path))

        assert result.status == SafeLoRAProjectionStatus.UNAVAILABLE
        assert "no cached projection matrix" in result.warnings[0]

    def test_project_skipped_with_cache(self, tmp_path: Path) -> None:
        """project() returns SKIPPED when cache exists but math is deferred."""
        # Create cache structure
        model_id = "mlx-community/Llama-3.2-3B"
        sanitized = SafeLoRAProjector._sanitize(model_id)
        cache_dir = tmp_path / "safety" / "projections" / sanitized
        cache_dir.mkdir(parents=True)
        (cache_dir / "projection.safetensors").write_bytes(b"dummy")

        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()

        projector = SafeLoRAProjector(resources_path=tmp_path)
        result = asyncio.run(projector.project(model_id, adapter_path))

        assert result.status == SafeLoRAProjectionStatus.SKIPPED
        assert "deferred" in result.warnings[0]
        assert result.details == "projection.safetensors"


class TestSafeLoRAConfiguration:
    """Tests for SafeLoRAConfiguration dataclass."""

    def test_default_configuration(self) -> None:
        """default() returns enabled configuration."""
        config = SafeLoRAConfiguration.default()
        assert config.enabled is True
        assert config.skip_if_unavailable is True
        assert config.resources_path is None

    def test_disabled_configuration(self) -> None:
        """disabled() returns disabled configuration."""
        config = SafeLoRAConfiguration.disabled()
        assert config.enabled is False

    def test_custom_configuration(self, tmp_path: Path) -> None:
        """Custom configuration with all fields."""
        config = SafeLoRAConfiguration(
            enabled=True,
            resources_path=tmp_path,
            skip_if_unavailable=False,
        )
        assert config.enabled is True
        assert config.resources_path == tmp_path
        assert config.skip_if_unavailable is False
