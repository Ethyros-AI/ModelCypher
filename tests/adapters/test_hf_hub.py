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

"""Tests for HuggingFace Hub adapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelcypher.adapters.hf_hub import HfHubAdapter
from modelcypher.ports.hub import HubAdapterPort


class TestHfHubAdapterProtocol:
    """Verify HfHubAdapter implements HubAdapterPort protocol."""

    def test_implements_hub_adapter_port(self):
        """HfHubAdapter should implement HubAdapterPort protocol."""
        adapter = HfHubAdapter()
        assert isinstance(adapter, HubAdapterPort)

    def test_has_fetch_method(self):
        """Adapter should have fetch method with correct signature."""
        adapter = HfHubAdapter()
        assert hasattr(adapter, "fetch")
        assert callable(adapter.fetch)

    def test_has_detect_architecture_method(self):
        """Adapter should have detect_architecture method."""
        adapter = HfHubAdapter()
        assert hasattr(adapter, "detect_architecture")
        assert callable(adapter.detect_architecture)

    def test_has_build_model_info_method(self):
        """Adapter should have build_model_info static method."""
        assert hasattr(HfHubAdapter, "build_model_info")
        assert callable(HfHubAdapter.build_model_info)


class TestHfHubAdapterInit:
    """Test HfHubAdapter initialization."""

    def test_default_base_dir(self):
        """Default base_dir should use HF_HOME or ~/.cache/huggingface."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = HfHubAdapter()
            assert adapter.base_dir is not None
            assert Path(adapter.base_dir).exists()

    def test_custom_base_dir(self):
        """Custom base_dir should be used when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = HfHubAdapter(base_dir=tmpdir)
            assert str(adapter.base_dir) == tmpdir

    def test_hf_home_env_var(self):
        """HF_HOME environment variable should be respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"HF_HOME": tmpdir}):
                adapter = HfHubAdapter()
                assert str(adapter.base_dir) == tmpdir


class TestHfHubAdapterFetch:
    """Test HfHubAdapter.fetch() method."""

    def test_fetch_calls_snapshot_download(self):
        """fetch() should call huggingface_hub.snapshot_download."""
        with patch("modelcypher.adapters.hf_hub.snapshot_download") as mock_download:
            mock_download.return_value = "/path/to/downloaded/model"
            adapter = HfHubAdapter()

            result = adapter.fetch("test-org/test-model")

            mock_download.assert_called_once_with(
                repo_id="test-org/test-model",
                revision="main",
                local_dir=None,
            )
            assert result == "/path/to/downloaded/model"

    def test_fetch_with_revision(self):
        """fetch() should pass revision to snapshot_download."""
        with patch("modelcypher.adapters.hf_hub.snapshot_download") as mock_download:
            mock_download.return_value = "/path/to/model"
            adapter = HfHubAdapter()

            adapter.fetch("test-org/test-model", revision="v1.0")

            mock_download.assert_called_once_with(
                repo_id="test-org/test-model",
                revision="v1.0",
                local_dir=None,
            )

    def test_fetch_with_local_dir(self):
        """fetch() should expand and pass local_dir to snapshot_download."""
        with patch("modelcypher.adapters.hf_hub.snapshot_download") as mock_download:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_download.return_value = tmpdir
                adapter = HfHubAdapter()

                result = adapter.fetch("test-org/test-model", local_dir=tmpdir)

                # local_dir should be expanded
                call_args = mock_download.call_args
                assert call_args.kwargs["local_dir"] is not None
                assert result == tmpdir

    def test_fetch_returns_string_path(self):
        """fetch() should always return a string path."""
        with patch("modelcypher.adapters.hf_hub.snapshot_download") as mock_download:
            mock_download.return_value = Path("/some/path")
            adapter = HfHubAdapter()

            result = adapter.fetch("test-org/test-model")

            assert isinstance(result, str)


class TestHfHubAdapterDetectArchitecture:
    """Test HfHubAdapter.detect_architecture() method."""

    def test_detect_architecture_reads_config_json(self):
        """detect_architecture should read model_type from config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"model_type": "llama"}))

            adapter = HfHubAdapter()
            result = adapter.detect_architecture(tmpdir)

            assert result == "llama"

    def test_detect_architecture_returns_none_for_missing_config(self):
        """detect_architecture should return None if config.json doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = HfHubAdapter()
            result = adapter.detect_architecture(tmpdir)

            assert result is None

    def test_detect_architecture_returns_none_for_invalid_json(self):
        """detect_architecture should return None for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("not valid json {{{")

            adapter = HfHubAdapter()
            result = adapter.detect_architecture(tmpdir)

            assert result is None

    def test_detect_architecture_returns_none_for_missing_model_type(self):
        """detect_architecture should return None if model_type key is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"hidden_size": 768}))

            adapter = HfHubAdapter()
            result = adapter.detect_architecture(tmpdir)

            assert result is None

    def test_detect_architecture_handles_various_architectures(self):
        """detect_architecture should handle various architecture types."""
        architectures = ["llama", "qwen2", "mistral", "gemma", "phi3"]

        for arch in architectures:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "config.json"
                config_path.write_text(json.dumps({"model_type": arch}))

                adapter = HfHubAdapter()
                result = adapter.detect_architecture(tmpdir)

                assert result == arch


class TestHfHubAdapterBuildModelInfo:
    """Test HfHubAdapter.build_model_info() static method."""

    def test_build_model_info_creates_model_info(self):
        """build_model_info should create a valid ModelInfo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file to have non-zero size
            test_file = Path(tmpdir) / "model.safetensors"
            test_file.write_bytes(b"x" * 1024)

            result = HfHubAdapter.build_model_info(
                alias="test-model",
                path=tmpdir,
                architecture="llama",
                parameter_count=7_000_000_000,
            )

            assert result.id == "test-model"
            assert result.alias == "test-model"
            assert result.architecture == "llama"
            assert result.format == "safetensors"
            assert result.parameter_count == 7_000_000_000
            assert result.size_bytes >= 1024
            assert result.is_default_chat is False
            assert result.created_at is not None

    def test_build_model_info_without_parameter_count(self):
        """build_model_info should work without parameter_count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = HfHubAdapter.build_model_info(
                alias="test-model",
                path=tmpdir,
                architecture="qwen2",
            )

            assert result.parameter_count is None

    def test_build_model_info_calculates_size(self):
        """build_model_info should calculate total size of all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(3):
                (Path(tmpdir) / f"file{i}.safetensors").write_bytes(b"x" * 100)

            result = HfHubAdapter.build_model_info(
                alias="test-model",
                path=tmpdir,
                architecture="llama",
            )

            assert result.size_bytes >= 300

    def test_build_model_info_handles_nested_files(self):
        """build_model_info should include nested files in size calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.safetensors").write_bytes(b"x" * 500)

            result = HfHubAdapter.build_model_info(
                alias="test-model",
                path=tmpdir,
                architecture="llama",
            )

            assert result.size_bytes >= 500

    def test_build_model_info_expands_tilde(self):
        """build_model_info should expand ~ in path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with actual path (can't reliably test ~ expansion in tests)
            result = HfHubAdapter.build_model_info(
                alias="test-model",
                path=tmpdir,
                architecture="llama",
            )

            assert "~" not in result.path
            assert Path(result.path).is_absolute()
