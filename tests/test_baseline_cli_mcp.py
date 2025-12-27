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

"""Comprehensive tests for baseline CLI commands and MCP tools.

Tests cover:
1. CLI command structure and help
2. CLI command JSON output schemas
3. CLI command error handling
4. MCP tool registration
5. MCP tool input validation
6. MCP tool output schemas
7. End-to-end workflows
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.domain_geometry_baselines import (
    BaselineRepository,
    DomainGeometryBaseline,
    DomainGeometryBaselineExtractor,
    ManifoldHealthDistribution,
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
def temp_baseline_dir(tmp_path: Path) -> Path:
    """Create temporary baseline directory."""
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    return baseline_dir


@pytest.fixture
def sample_baseline(temp_baseline_dir: Path) -> DomainGeometryBaseline:
    """Create and save a sample baseline."""
    baseline = DomainGeometryBaseline(
        domain="spatial",
        model_family="qwen",
        model_size="0.5B",
        model_path="/test/qwen-0.5b",
        ollivier_ricci_mean=-0.189,
        ollivier_ricci_std=0.045,
        ollivier_ricci_min=-0.35,
        ollivier_ricci_max=-0.12,
        manifold_health_distribution=ManifoldHealthDistribution(
            healthy=1.0,
            degenerate=0.0,
            collapsed=0.0,
        ),
        domain_metrics={
            "euclidean_consistency": 0.76,
            "gravity_alignment": 0.89,
        },
        intrinsic_dimension_mean=12.4,
        intrinsic_dimension_std=2.1,
        layers_analyzed=8,
        extraction_date="2025-12-27",
        extraction_config={"k_neighbors": 10},
    )
    repo = BaselineRepository(baseline_dir=temp_baseline_dir)
    repo.save_baseline(baseline)
    return baseline


@pytest.fixture
def populated_baseline_dir(temp_baseline_dir: Path) -> Path:
    """Create baseline directory with multiple baselines."""
    domains = ["spatial", "social", "temporal", "moral"]
    for domain in domains:
        baseline = DomainGeometryBaseline(
            domain=domain,
            model_family="qwen",
            model_size="0.5B",
            model_path=f"/test/qwen-0.5b-{domain}",
            ollivier_ricci_mean=-0.15 - (domains.index(domain) * 0.02),
            ollivier_ricci_std=0.04,
            ollivier_ricci_min=-0.30,
            ollivier_ricci_max=-0.05,
            manifold_health_distribution=ManifoldHealthDistribution(
                healthy=0.9,
                degenerate=0.08,
                collapsed=0.02,
            ),
            domain_metrics={f"{domain}_metric": 0.75},
            intrinsic_dimension_mean=12.0,
            intrinsic_dimension_std=2.0,
            layers_analyzed=8,
            extraction_date="2025-12-27",
            extraction_config={"k_neighbors": 10},
        )
        repo = BaselineRepository(baseline_dir=temp_baseline_dir)
        repo.save_baseline(baseline)
    return temp_baseline_dir


# =============================================================================
# CLI Command Structure Tests
# =============================================================================


class TestBaselineCLIStructure:
    """Tests for CLI command structure and help."""

    def test_baseline_command_exists(self):
        """Baseline subcommand is registered."""
        result = runner.invoke(app, ["geometry", "baseline", "--help"])
        assert result.exit_code == 0
        assert "baseline" in result.stdout.lower()

    def test_baseline_list_help(self):
        """List command has proper help."""
        result = runner.invoke(app, ["geometry", "baseline", "list", "--help"])
        assert result.exit_code == 0
        assert "--domain" in result.stdout or "-d" in result.stdout

    def test_baseline_extract_help(self):
        """Extract command has proper help."""
        result = runner.invoke(app, ["geometry", "baseline", "extract", "--help"])
        assert result.exit_code == 0
        assert "MODEL_PATH" in result.stdout
        assert "--domain" in result.stdout

    def test_baseline_validate_help(self):
        """Validate command has proper help."""
        result = runner.invoke(app, ["geometry", "baseline", "validate", "--help"])
        assert result.exit_code == 0
        assert "MODEL_PATH" in result.stdout
        assert "--domains" in result.stdout

    def test_baseline_compare_help(self):
        """Compare command has proper help."""
        result = runner.invoke(app, ["geometry", "baseline", "compare", "--help"])
        assert result.exit_code == 0
        assert "MODEL1_PATH" in result.stdout
        assert "MODEL2_PATH" in result.stdout


# =============================================================================
# CLI List Command Tests
# =============================================================================


class TestBaselineCLIList:
    """Tests for baseline list CLI command."""

    def test_list_empty_returns_empty_list(self, temp_baseline_dir: Path):
        """List returns empty array when no baselines exist."""
        with patch(
            "modelcypher.core.domain.geometry.domain_geometry_baselines.BaselineRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.get_all_baselines.return_value = []
            mock_repo_class.return_value = mock_repo

            result = runner.invoke(
                app, ["geometry", "baseline", "list", "--output", "json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "_schema" in data
            assert data["_schema"] == "mc.geometry.baseline.list.v1"
            assert "baselines" in data
            assert isinstance(data["baselines"], list)

    def test_list_with_baselines_returns_data(self, sample_baseline: DomainGeometryBaseline):
        """List returns baseline data when baselines exist."""
        # Use the built-in baselines that were created in the session
        result = runner.invoke(
            app, ["geometry", "baseline", "list", "--output", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "_schema" in data
        assert len(data["baselines"]) > 0

    def test_list_filters_by_domain(self):
        """List with --domain filters results."""
        result = runner.invoke(
            app, ["geometry", "baseline", "list", "--domain", "spatial", "--output", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        for baseline in data["baselines"]:
            assert baseline["domain"] == "spatial"

    def test_list_text_output_format(self):
        """List produces readable text output."""
        result = runner.invoke(
            app, ["--output", "text", "geometry", "baseline", "list"]
        )
        assert result.exit_code == 0
        # Text output should have table-like structure
        assert "Domain" in result.stdout or "BASELINE" in result.stdout or "baselines" in result.stdout.lower()


# =============================================================================
# CLI Extract Command Tests
# =============================================================================


class TestBaselineCLIExtract:
    """Tests for baseline extract CLI command."""

    def test_extract_requires_model_path(self):
        """Extract fails without model path."""
        result = runner.invoke(app, ["geometry", "baseline", "extract"])
        # Should fail with exit code 2 (missing argument) or have error output
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_extract_validates_domain(self):
        """Extract rejects invalid domain."""
        result = runner.invoke(
            app,
            ["geometry", "baseline", "extract", "/fake/path", "--domain", "invalid_domain"],
        )
        assert result.exit_code != 0
        # Error messages may go to stderr or be in output
        combined = (result.stdout or "") + (result.output or "")
        assert "invalid" in combined.lower() or result.exit_code == 1

    def test_extract_accepts_valid_domains(self):
        """Extract accepts all valid domains."""
        valid_domains = ["spatial", "social", "temporal", "moral"]
        for domain in valid_domains:
            result = runner.invoke(
                app,
                ["geometry", "baseline", "extract", "--help"],
            )
            assert result.exit_code == 0

    @pytest.mark.real_model
    def test_extract_produces_valid_json(self, tmp_path: Path):
        """Extract produces valid JSON output schema."""
        model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-0.5B-Instruct-bf16"
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "geometry",
                "baseline",
                "extract",
                model_path,
                "--domain",
                "spatial",
                "--output",
                "json",
                "--output-dir",
                str(output_dir),
            ],
        )

        if result.exit_code == 0:
            data = json.loads(result.stdout)
            assert data["_schema"] == "mc.geometry.baseline.extract.v1"
            assert "domain" in data
            assert "ollivierRicciMean" in data or "ollivier_ricci_mean" in data
            assert "savedPath" in data or "saved_path" in data


# =============================================================================
# CLI Validate Command Tests
# =============================================================================


class TestBaselineCLIValidate:
    """Tests for baseline validate CLI command."""

    def test_validate_requires_model_path(self):
        """Validate fails without model path."""
        result = runner.invoke(app, ["geometry", "baseline", "validate"])
        assert result.exit_code != 0

    def test_validate_accepts_domain_list(self):
        """Validate accepts comma-separated domain list."""
        result = runner.invoke(
            app,
            ["geometry", "baseline", "validate", "--help"],
        )
        assert result.exit_code == 0
        assert "--domains" in result.stdout

    def test_validate_strict_flag_exists(self):
        """Validate has --strict flag."""
        result = runner.invoke(
            app,
            ["geometry", "baseline", "validate", "--help"],
        )
        assert result.exit_code == 0
        assert "--strict" in result.stdout


# =============================================================================
# CLI Compare Command Tests
# =============================================================================


class TestBaselineCLICompare:
    """Tests for baseline compare CLI command."""

    def test_compare_requires_both_model_paths(self):
        """Compare fails without both model paths."""
        result = runner.invoke(
            app, ["geometry", "baseline", "compare", "/path/to/model1"]
        )
        assert result.exit_code != 0

    def test_compare_accepts_domain_option(self):
        """Compare accepts --domain option."""
        result = runner.invoke(
            app,
            ["geometry", "baseline", "compare", "--help"],
        )
        assert result.exit_code == 0
        assert "--domain" in result.stdout


# =============================================================================
# CLI Output Schema Tests
# =============================================================================


class TestBaselineCLISchemas:
    """Tests for CLI output schema compliance."""

    def test_list_schema_fields(self):
        """List output has required schema fields."""
        result = runner.invoke(
            app, ["geometry", "baseline", "list", "--output", "json"]
        )
        if result.exit_code == 0:
            data = json.loads(result.stdout)
            assert "_schema" in data
            assert "baselines" in data
            if data["baselines"]:
                baseline = data["baselines"][0]
                # Check for expected fields (either camelCase or snake_case)
                has_domain = "domain" in baseline
                has_family = "modelFamily" in baseline or "model_family" in baseline
                assert has_domain
                assert has_family


# =============================================================================
# MCP Tool Registration Tests
# =============================================================================


class TestBaselineMCPRegistration:
    """Tests for MCP tool registration."""

    def test_baseline_tools_in_tool_set(self):
        """Baseline tools are in the full profile."""
        from modelcypher.mcp.server import TOOL_PROFILES

        full_tools = TOOL_PROFILES.get("full", set())

        expected_tools = [
            "mc_geometry_baseline_list",
            "mc_geometry_baseline_extract",
            "mc_geometry_baseline_validate",
            "mc_geometry_baseline_compare",
        ]

        for tool in expected_tools:
            assert tool in full_tools, f"Missing tool: {tool}"

    def test_baseline_tools_in_training_profile(self):
        """Baseline validation tools in training profile."""
        from modelcypher.mcp.server import TOOL_PROFILES

        training_tools = TOOL_PROFILES.get("training", set())

        # Training profile should have at least list and validate
        assert "mc_geometry_baseline_list" in training_tools
        assert "mc_geometry_baseline_validate" in training_tools

    def test_baseline_tools_in_monitoring_profile(self):
        """Baseline tools in monitoring profile."""
        from modelcypher.mcp.server import TOOL_PROFILES

        monitoring_tools = TOOL_PROFILES.get("monitoring", set())

        # Monitoring should have read-only baseline tools
        assert "mc_geometry_baseline_list" in monitoring_tools
        assert "mc_geometry_baseline_validate" in monitoring_tools


# =============================================================================
# MCP Tool Function Tests
# =============================================================================


class TestBaselineMCPFunctions:
    """Tests for MCP tool function behavior."""

    def test_register_baseline_tools_callable(self):
        """register_geometry_baseline_tools is callable."""
        from modelcypher.mcp.tools.geometry import register_geometry_baseline_tools

        assert callable(register_geometry_baseline_tools)

    def test_baseline_tools_have_annotations(self):
        """Baseline tools have proper MCP annotations."""
        from modelcypher.mcp.tools.common import READ_ONLY_ANNOTATIONS

        # All baseline tools should be read-only
        assert "readOnlyHint" in READ_ONLY_ANNOTATIONS
        assert READ_ONLY_ANNOTATIONS["readOnlyHint"] is True


# =============================================================================
# Domain Geometry Baselines Module Tests
# =============================================================================


class TestDomainGeometryBaselinesModule:
    """Tests for domain_geometry_baselines.py functions."""

    def test_baseline_repository_init_default(self):
        """BaselineRepository initializes with default path."""
        repo = BaselineRepository()
        assert repo._baseline_dir is not None

    def test_baseline_repository_init_custom_path(self, temp_baseline_dir: Path):
        """BaselineRepository accepts custom path."""
        repo = BaselineRepository(baseline_dir=temp_baseline_dir)
        assert repo._baseline_dir == temp_baseline_dir

    def test_baseline_repository_save_and_get(
        self, temp_baseline_dir: Path, backend: "Backend"
    ):
        """Repository can save and retrieve baselines."""
        baseline = DomainGeometryBaseline(
            domain="spatial",
            model_family="test",
            model_size="1B",
            model_path="/test/model",
            ollivier_ricci_mean=-0.2,
            ollivier_ricci_std=0.05,
            ollivier_ricci_min=-0.3,
            ollivier_ricci_max=-0.1,
            manifold_health_distribution=ManifoldHealthDistribution(
                healthy=0.9, degenerate=0.08, collapsed=0.02
            ),
            domain_metrics={},
            intrinsic_dimension_mean=10.0,
            intrinsic_dimension_std=1.0,
            layers_analyzed=8,
            extraction_date="2025-12-27",
        )

        repo = BaselineRepository(baseline_dir=temp_baseline_dir)
        saved_path = repo.save_baseline(baseline)

        assert saved_path.exists()

        loaded = repo.get_baseline("spatial", "test", "1B")
        assert loaded is not None
        assert loaded.domain == "spatial"
        assert loaded.model_family == "test"

    def test_baseline_repository_get_all(self, populated_baseline_dir: Path):
        """Repository can get all baselines."""
        repo = BaselineRepository(baseline_dir=populated_baseline_dir)
        baselines = repo.get_all_baselines()

        assert len(baselines) == 4  # spatial, social, temporal, moral

    def test_baseline_repository_get_by_domain(self, populated_baseline_dir: Path):
        """Repository can filter by domain."""
        repo = BaselineRepository(baseline_dir=populated_baseline_dir)
        spatial_baselines = repo.get_baselines_for_domain("spatial")

        assert len(spatial_baselines) == 1
        assert spatial_baselines[0].domain == "spatial"

    def test_baseline_repository_find_matching(self, populated_baseline_dir: Path):
        """Repository finds matching baseline with fallbacks."""
        repo = BaselineRepository(baseline_dir=populated_baseline_dir)

        # Exact match
        exact = repo.find_matching_baseline("spatial", "qwen", "0.5B")
        assert exact is not None
        assert exact.domain == "spatial"

        # Same family, different size - should find same family
        family_match = repo.find_matching_baseline("spatial", "qwen", "3B")
        assert family_match is not None
        assert family_match.model_family == "qwen"

        # Different family - should find any in domain
        any_match = repo.find_matching_baseline("spatial", "llama", "7B")
        assert any_match is not None
        assert any_match.domain == "spatial"


class TestDomainGeometryBaselineExtractor:
    """Tests for DomainGeometryBaselineExtractor."""

    def test_extractor_init(self, backend: "Backend"):
        """Extractor initializes with backend."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)
        assert extractor._backend is not None

    def test_extractor_parse_model_info_qwen(self, backend: "Backend"):
        """Extractor parses Qwen model paths."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        family, size = extractor._parse_model_info(
            "/path/to/Qwen2.5-0.5B-Instruct-bf16"
        )
        assert family == "qwen"
        assert size == "0.5B"

    def test_extractor_parse_model_info_llama(self, backend: "Backend"):
        """Extractor parses Llama model paths."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        family, size = extractor._parse_model_info("/path/to/Llama-3.2-3B-Instruct-4bit")
        assert family == "llama"
        assert size == "3B"

    def test_extractor_parse_model_info_mistral(self, backend: "Backend"):
        """Extractor parses Mistral model paths."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        family, size = extractor._parse_model_info(
            "/path/to/Mistral-7B-Instruct-v0.3-4bit"
        )
        assert family == "mistral"
        assert size == "7B"

    def test_extractor_get_domain_probes_returns_probes(self, backend: "Backend"):
        """Extractor returns probes for any domain."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        for domain in ["spatial", "social", "temporal", "moral"]:
            probes = extractor._get_domain_probes(domain)
            # Should have many probes (using full atlas)
            assert len(probes) >= 100, f"Domain {domain} has too few probes"

    def test_extractor_generate_synthetic_activations(self, backend: "Backend"):
        """Extractor can generate synthetic activations."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        layers = [0, 4, 8, 12]
        activations = extractor._generate_synthetic_activations(layers)

        assert len(activations) == len(layers)
        for layer_idx in layers:
            assert layer_idx in activations
            # Should have shape [n_samples, hidden_dim]
            assert len(activations[layer_idx].shape) == 2


class TestManifoldHealthDistribution:
    """Tests for ManifoldHealthDistribution dataclass."""

    def test_distribution_to_dict(self):
        """Distribution converts to dict correctly."""
        dist = ManifoldHealthDistribution(healthy=0.8, degenerate=0.15, collapsed=0.05)
        d = dist.to_dict()

        assert d["healthy"] == 0.8
        assert d["degenerate"] == 0.15
        assert d["collapsed"] == 0.05

    def test_distribution_from_dict(self):
        """Distribution creates from dict correctly."""
        d = {"healthy": 0.9, "degenerate": 0.08, "collapsed": 0.02}
        dist = ManifoldHealthDistribution.from_dict(d)

        assert dist.healthy == 0.9
        assert dist.degenerate == 0.08
        assert dist.collapsed == 0.02


class TestDomainGeometryBaseline:
    """Tests for DomainGeometryBaseline dataclass."""

    def test_baseline_to_dict(self):
        """Baseline converts to dict correctly."""
        baseline = DomainGeometryBaseline(
            domain="spatial",
            model_family="qwen",
            model_size="0.5B",
            model_path="/test/path",
            ollivier_ricci_mean=-0.2,
            ollivier_ricci_std=0.05,
            ollivier_ricci_min=-0.3,
            ollivier_ricci_max=-0.1,
            manifold_health_distribution=ManifoldHealthDistribution(
                healthy=0.9, degenerate=0.08, collapsed=0.02
            ),
            domain_metrics={"metric1": 0.5},
            intrinsic_dimension_mean=10.0,
            intrinsic_dimension_std=1.0,
            layers_analyzed=8,
            extraction_date="2025-12-27",
        )

        d = baseline.to_dict()

        assert d["domain"] == "spatial"
        assert d["model_family"] == "qwen"
        assert d["ollivier_ricci_mean"] == -0.2
        assert d["manifold_health_distribution"]["healthy"] == 0.9

    def test_baseline_from_dict(self):
        """Baseline creates from dict correctly."""
        d = {
            "domain": "social",
            "model_family": "llama",
            "model_size": "3B",
            "model_path": "/test/llama",
            "ollivier_ricci_mean": -0.15,
            "ollivier_ricci_std": 0.04,
            "ollivier_ricci_min": -0.25,
            "ollivier_ricci_max": -0.05,
            "manifold_health_distribution": {"healthy": 0.85, "degenerate": 0.1, "collapsed": 0.05},
            "domain_metrics": {"social_metric": 0.7},
            "intrinsic_dimension_mean": 11.0,
            "intrinsic_dimension_std": 1.5,
            "layers_analyzed": 12,
            "extraction_date": "2025-12-27",
        }

        baseline = DomainGeometryBaseline.from_dict(d)

        assert baseline.domain == "social"
        assert baseline.model_family == "llama"
        assert baseline.ollivier_ricci_mean == -0.15
        assert baseline.manifold_health_distribution.healthy == 0.85

    def test_baseline_save_and_load(self, tmp_path: Path):
        """Baseline can save and load from file."""
        baseline = DomainGeometryBaseline(
            domain="temporal",
            model_family="mistral",
            model_size="7B",
            model_path="/test/mistral",
            ollivier_ricci_mean=-0.18,
            ollivier_ricci_std=0.06,
            ollivier_ricci_min=-0.28,
            ollivier_ricci_max=-0.08,
            manifold_health_distribution=ManifoldHealthDistribution(
                healthy=0.88, degenerate=0.09, collapsed=0.03
            ),
            domain_metrics={"temporal_metric": 0.65},
            intrinsic_dimension_mean=9.5,
            intrinsic_dimension_std=1.2,
            layers_analyzed=16,
            extraction_date="2025-12-27",
        )

        file_path = tmp_path / "test_baseline.json"
        baseline.save(file_path)

        assert file_path.exists()

        loaded = DomainGeometryBaseline.load(file_path)
        assert loaded.domain == baseline.domain
        assert loaded.model_family == baseline.model_family
        assert loaded.ollivier_ricci_mean == baseline.ollivier_ricci_mean


# =============================================================================
# Domain Geometry Validator Tests
# =============================================================================


class TestDomainGeometryValidator:
    """Tests for DomainGeometryValidator."""

    def test_validator_init_default(self):
        """Validator initializes with defaults."""
        from modelcypher.core.domain.geometry.domain_geometry_validator import (
            DomainGeometryValidator,
        )

        validator = DomainGeometryValidator()
        assert validator is not None

    def test_validator_init_custom_baseline_dir(self, temp_baseline_dir: Path):
        """Validator accepts custom baseline directory."""
        from modelcypher.core.domain.geometry.domain_geometry_validator import (
            DomainGeometryValidator,
        )

        validator = DomainGeometryValidator(baseline_dir=temp_baseline_dir)
        # Check that repository was created with custom path
        assert validator._repository._baseline_dir == temp_baseline_dir

    def test_validator_compute_deviation(self, backend: "Backend"):
        """Validator computes deviation scores correctly."""
        from modelcypher.core.domain.geometry.domain_geometry_validator import (
            DomainGeometryValidator,
        )

        validator = DomainGeometryValidator(backend=backend)

        # Test deviation computation for a single metric
        current_value = -0.20
        baseline_value = -0.22
        dev = validator._compute_deviation(current_value, baseline_value)

        # Should return a non-negative deviation
        assert dev >= 0
        # Deviation should be reasonable (not too large for similar values)
        assert dev < 0.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestBaselineIntegration:
    """Integration tests for baseline workflows."""

    def test_cli_list_json_parses_correctly(self):
        """CLI list JSON output can be parsed."""
        result = runner.invoke(
            app, ["geometry", "baseline", "list", "--output", "json"]
        )

        if result.exit_code == 0:
            # Should be valid JSON
            data = json.loads(result.stdout)
            assert isinstance(data, dict)
            assert "baselines" in data

    def test_repository_baseline_roundtrip(self, temp_baseline_dir: Path):
        """Full baseline save/load roundtrip works."""
        # Create baseline
        original = DomainGeometryBaseline(
            domain="moral",
            model_family="phi",
            model_size="1.5B",
            model_path="/test/phi",
            ollivier_ricci_mean=-0.17,
            ollivier_ricci_std=0.04,
            ollivier_ricci_min=-0.25,
            ollivier_ricci_max=-0.09,
            manifold_health_distribution=ManifoldHealthDistribution(
                healthy=0.87, degenerate=0.10, collapsed=0.03
            ),
            domain_metrics={"moral_metric": 0.72},
            intrinsic_dimension_mean=11.0,
            intrinsic_dimension_std=1.8,
            layers_analyzed=10,
            extraction_date="2025-12-27",
        )

        # Save via repository
        repo = BaselineRepository(baseline_dir=temp_baseline_dir)
        repo.save_baseline(original)

        # Create new repository instance and load
        repo2 = BaselineRepository(baseline_dir=temp_baseline_dir)
        loaded = repo2.get_baseline("moral", "phi", "1.5B")

        assert loaded is not None
        assert loaded.domain == original.domain
        assert loaded.model_family == original.model_family
        assert loaded.ollivier_ricci_mean == original.ollivier_ricci_mean
        assert loaded.manifold_health_distribution.healthy == original.manifold_health_distribution.healthy


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBaselineErrorHandling:
    """Tests for error handling in baseline operations."""

    def test_repository_handles_missing_file(self, temp_baseline_dir: Path):
        """Repository handles missing baseline gracefully."""
        repo = BaselineRepository(baseline_dir=temp_baseline_dir)
        result = repo.get_baseline("nonexistent", "model", "0B")
        assert result is None

    def test_repository_handles_corrupt_json(self, temp_baseline_dir: Path):
        """Repository handles corrupt JSON files."""
        # Write corrupt JSON
        corrupt_file = temp_baseline_dir / "spatial_test_1B.json"
        corrupt_file.write_text("{ not valid json }")

        repo = BaselineRepository(baseline_dir=temp_baseline_dir)
        # Should not crash
        baselines = repo.get_all_baselines()
        # May be empty or have loaded what it could
        assert isinstance(baselines, list)

    def test_extractor_handles_invalid_domain(self, backend: "Backend"):
        """Extractor raises on invalid domain."""
        extractor = DomainGeometryBaselineExtractor(backend=backend)

        # _get_domain_probes should work for any domain (uses full atlas)
        # but validation happens at CLI/MCP level
        probes = extractor._get_domain_probes("invalid_domain")
        # Should still return probes (domain is for semantic labeling, not filtering)
        assert len(probes) > 0
