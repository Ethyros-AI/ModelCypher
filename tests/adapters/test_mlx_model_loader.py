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

"""Tests for MLX model loader adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelcypher.adapters.mlx_model_loader import MLXModelLoader
from modelcypher.ports.model_loader import ModelLoaderPort


class TestMLXModelLoaderProtocol:
    """Verify MLXModelLoader implements ModelLoaderPort protocol."""

    def test_implements_model_loader_port(self):
        """MLXModelLoader should implement ModelLoaderPort protocol."""
        loader = MLXModelLoader()
        assert isinstance(loader, ModelLoaderPort)

    def test_has_load_model_for_training_method(self):
        """Loader should have load_model_for_training method."""
        loader = MLXModelLoader()
        assert hasattr(loader, "load_model_for_training")
        assert callable(loader.load_model_for_training)

    def test_has_load_weights_method(self):
        """Loader should have load_weights method."""
        loader = MLXModelLoader()
        assert hasattr(loader, "load_weights")
        assert callable(loader.load_weights)


class TestMLXModelLoaderLoadModelForTraining:
    """Test load_model_for_training() method."""

    def test_calls_underlying_loader(self):
        """load_model_for_training should call the underlying function."""
        with patch(
            "modelcypher.adapters.mlx_model_loader._load_model_for_training"
        ) as mock_loader:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_loader.return_value = (mock_model, mock_tokenizer)

            loader = MLXModelLoader()
            result = loader.load_model_for_training("/path/to/model")

            mock_loader.assert_called_once_with(
                "/path/to/model",
                None,
                adapter_path=None,
            )
            assert result == (mock_model, mock_tokenizer)

    def test_passes_lora_settings(self):
        """load_model_for_training should pass lora_settings."""
        with patch(
            "modelcypher.adapters.mlx_model_loader._load_model_for_training"
        ) as mock_loader:
            mock_loader.return_value = (MagicMock(), MagicMock())
            mock_lora = MagicMock()

            loader = MLXModelLoader()
            loader.load_model_for_training("/path/to/model", lora_settings=mock_lora)

            mock_loader.assert_called_once_with(
                "/path/to/model",
                mock_lora,
                adapter_path=None,
            )

    def test_passes_adapter_path(self):
        """load_model_for_training should pass adapter_path."""
        with patch(
            "modelcypher.adapters.mlx_model_loader._load_model_for_training"
        ) as mock_loader:
            mock_loader.return_value = (MagicMock(), MagicMock())

            loader = MLXModelLoader()
            loader.load_model_for_training(
                "/path/to/model", adapter_path="/path/to/adapter"
            )

            mock_loader.assert_called_once_with(
                "/path/to/model",
                None,
                adapter_path="/path/to/adapter",
            )


class TestMLXModelLoaderLoadWeights:
    """Test load_weights() method."""

    @pytest.mark.mlx
    def test_raises_error_when_mlx_unavailable(self):
        """load_weights should raise RuntimeError when MLX is unavailable."""
        with patch(
            "modelcypher.adapters.mlx_model_loader.probe_mlx_available",
            return_value=False,
        ):
            with patch(
                "modelcypher.adapters.mlx_model_loader.get_mlx_probe_error",
                return_value="MLX not available",
            ):
                loader = MLXModelLoader()
                with pytest.raises(RuntimeError, match="MLX runtime unavailable"):
                    loader.load_weights("/path/to/model")

    @pytest.mark.mlx
    def test_raises_error_for_missing_safetensors(self):
        """load_weights should raise FileNotFoundError for missing safetensors."""
        with patch(
            "modelcypher.adapters.mlx_model_loader.probe_mlx_available",
            return_value=True,
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                loader = MLXModelLoader()
                with pytest.raises(FileNotFoundError, match="No safetensors files"):
                    loader.load_weights(tmpdir)

    @pytest.mark.mlx
    def test_loads_safetensors_files(self):
        """load_weights should load all safetensors files in directory."""
        # This test requires actual MLX availability
        pytest.importorskip("mlx.core")

        with patch(
            "modelcypher.adapters.mlx_model_loader.probe_mlx_available",
            return_value=True,
        ):
            import mlx.core as mx

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a simple safetensors file
                weights = {"layer.weight": mx.ones((2, 2))}
                mx.save_safetensors(str(Path(tmpdir) / "model.safetensors"), weights)

                loader = MLXModelLoader()
                result = loader.load_weights(tmpdir)

                assert "layer.weight" in result
                assert result["layer.weight"].shape == (2, 2)

    @pytest.mark.mlx
    def test_loads_multiple_safetensors_files(self):
        """load_weights should load weights from multiple safetensors files."""
        pytest.importorskip("mlx.core")

        with patch(
            "modelcypher.adapters.mlx_model_loader.probe_mlx_available",
            return_value=True,
        ):
            import mlx.core as mx

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create multiple safetensors files
                weights1 = {"layer1.weight": mx.ones((2, 2))}
                weights2 = {"layer2.weight": mx.zeros((3, 3))}
                mx.save_safetensors(str(Path(tmpdir) / "model-00001.safetensors"), weights1)
                mx.save_safetensors(str(Path(tmpdir) / "model-00002.safetensors"), weights2)

                loader = MLXModelLoader()
                result = loader.load_weights(tmpdir)

                assert "layer1.weight" in result
                assert "layer2.weight" in result
