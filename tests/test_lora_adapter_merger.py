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

"""Tests for LoRAAdapterMerger module."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.merging.exceptions import MergeError
from modelcypher.core.domain.merging.lora_adapter_merger import (
    AdapterPayload,
    LoRAAdapterMerger,
    MergeReport,
)


def _div_eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


@pytest.fixture
def backend():
    """Get compute backend."""
    return get_default_backend()


class TestAdapterPayloadDataclass:
    """Tests for AdapterPayload dataclass."""

    def test_fields(self, backend, tmp_path):
        """Test all fields are accessible."""
        weights = {"layer.weight": backend.zeros((16, 16))}
        payload = AdapterPayload(
            directory=tmp_path,
            base_model_id="test-model",
            rank=8,
            scale=1.0,
            weights=weights,
            module_keys=["layer.weight"],
        )

        assert payload.directory == tmp_path
        assert payload.base_model_id == "test-model"
        assert payload.rank == 8
        assert payload.scale == 1.0
        assert "layer.weight" in payload.weights
        assert payload.module_keys == ["layer.weight"]


class TestMergeReportDataclass:
    """Tests for MergeReport dataclass."""

    def test_fields(self):
        """Test all fields are accessible."""
        report = MergeReport(
            output_directory="/path/to/output",
            adapter_count=3,
            base_model_id="llama-7b",
            rank=16,
            scale=1.5,
            mean_procrustes_error=0.05,
            mean_permutation_quality=0.92,
            total_merged_parameters=1000000,
            layer_count=32,
            merge_confidence=0.87,
        )

        assert report.output_directory == "/path/to/output"
        assert report.adapter_count == 3
        assert report.base_model_id == "llama-7b"
        assert report.rank == 16
        assert report.scale == 1.5
        assert report.mean_procrustes_error == 0.05
        assert report.mean_permutation_quality == 0.92
        assert report.total_merged_parameters == 1000000
        assert report.layer_count == 32
        assert report.merge_confidence == 0.87


class TestExtractLayerIndex:
    """Tests for LoRAAdapterMerger._extract_layer_index method."""

    def test_standard_key(self):
        """Test extraction from standard layer key."""
        key = "model.layers.5.self_attn.q_proj.lora_A"
        result = LoRAAdapterMerger._extract_layer_index(key)
        assert result == 5

    def test_double_digit_layer(self):
        """Test extraction with double-digit layer number."""
        key = "model.layers.42.mlp.gate_proj.lora_B"
        result = LoRAAdapterMerger._extract_layer_index(key)
        assert result == 42

    def test_zero_layer(self):
        """Test extraction for layer 0."""
        key = "model.layers.0.self_attn.v_proj.lora_A"
        result = LoRAAdapterMerger._extract_layer_index(key)
        assert result == 0

    def test_no_layers_keyword(self):
        """Test key without 'layers' keyword."""
        key = "embedding.weight"
        result = LoRAAdapterMerger._extract_layer_index(key)
        assert result is None

    def test_layers_at_end(self):
        """Test 'layers' at end without number."""
        key = "model.layers"
        result = LoRAAdapterMerger._extract_layer_index(key)
        assert result is None

    def test_non_numeric_after_layers(self):
        """Test non-numeric after 'layers'."""
        key = "model.layers.foo.weight"
        result = LoRAAdapterMerger._extract_layer_index(key)
        assert result is None


class TestGeometricMergeMatrices:
    """Tests for LoRAAdapterMerger._geometric_merge_matrices method."""

    def test_single_matrix_returned_unchanged(self, backend):
        """Test single matrix is returned unchanged."""
        backend.random_seed(42)
        matrix = backend.random_normal((16, 8))

        result, error, quality = LoRAAdapterMerger._geometric_merge_matrices(
            [matrix], backend
        )

        # Should return the matrix with perfect scores
        assert result.shape == matrix.shape
        eps = _div_eps()
        assert abs(error) < eps
        assert abs(quality - 1.0) < eps

    def test_1d_tensors_averaged(self, backend):
        """Test 1D tensors (biases) are simply averaged."""
        bias1 = backend.array([1.0, 2.0, 3.0])
        bias2 = backend.array([3.0, 4.0, 5.0])

        result, error, quality = LoRAAdapterMerger._geometric_merge_matrices(
            [bias1, bias2], backend
        )

        expected = backend.array([2.0, 3.0, 4.0])  # average
        result_np = backend.to_numpy(result)
        expected_np = backend.to_numpy(expected)

        assert result.shape == bias1.shape
        for i in range(3):
            assert abs(result_np[i] - expected_np[i]) < _div_eps()
        eps = _div_eps()
        assert abs(error) < eps
        assert abs(quality - 1.0) < eps

    def test_two_matrices_merged(self, backend):
        """Test two matrices are merged geometrically."""
        backend.random_seed(42)
        m1 = backend.random_normal((16, 8))
        m2 = backend.random_normal((16, 8))

        result, error, quality = LoRAAdapterMerger._geometric_merge_matrices(
            [m1, m2], backend
        )

        assert result.shape == m1.shape
        # Error and quality should be computed
        assert 0.0 <= error
        assert 0.0 <= quality <= 1.0

    def test_three_matrices_merged(self, backend):
        """Test three matrices are merged."""
        backend.random_seed(42)
        m1 = backend.random_normal((8, 8))
        m2 = backend.random_normal((8, 8))
        m3 = backend.random_normal((8, 8))

        result, error, quality = LoRAAdapterMerger._geometric_merge_matrices(
            [m1, m2, m3], backend
        )

        assert result.shape == m1.shape


class TestProcrustesAlign:
    """Tests for LoRAAdapterMerger._procrustes_align method."""

    def test_identical_matrices_zero_error(self, backend):
        """Test identical matrices have zero alignment error."""
        backend.random_seed(42)
        matrix = backend.random_normal((8, 8))

        rotated, error = LoRAAdapterMerger._procrustes_align(matrix, matrix, backend)

        # Aligning to self should have very low error
        assert error < _div_eps()
        assert rotated.shape == matrix.shape

    def test_scaled_matrices(self, backend):
        """Test Procrustes aligns scaled matrices."""
        backend.random_seed(42)
        target = backend.random_normal((8, 8))
        source = target * 2.0  # Scaled version

        rotated, error = LoRAAdapterMerger._procrustes_align(source, target, backend)

        # Should find an alignment
        assert rotated.shape == target.shape

    def test_rotation_applied(self, backend):
        """Test that rotation is actually applied."""
        backend.random_seed(42)
        target = backend.random_normal((8, 8))
        source = backend.random_normal((8, 8))  # Different

        rotated, error = LoRAAdapterMerger._procrustes_align(source, target, backend)

        # Result should be different from source (rotation applied)
        source_np = backend.to_numpy(source)
        rotated_np = backend.to_numpy(rotated)

        # They shouldn't be exactly equal
        diff = abs(source_np - rotated_np).sum()
        eps = _div_eps()
        assert diff > eps or error < eps  # Either different or perfectly aligned


class TestMergeValidation:
    """Tests for merge validation."""

    def test_fewer_than_two_adapters_raises(self):
        """Test that fewer than 2 adapters raises MergeError."""
        with pytest.raises(MergeError, match="At least two adapters"):
            LoRAAdapterMerger.merge(
                adapter_directories=[Path("/fake/path")],
                output_directory=Path("/output"),
            )

    def test_empty_adapter_list_raises(self):
        """Test empty adapter list raises MergeError."""
        with pytest.raises(MergeError, match="At least two adapters"):
            LoRAAdapterMerger.merge(
                adapter_directories=[],
                output_directory=Path("/output"),
            )


class TestLoadAdapterErrors:
    """Tests for adapter loading errors."""

    def test_missing_config_raises(self, tmp_path):
        """Test missing adapter_config.json raises MergeError."""
        # Create empty directory
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        with pytest.raises(MergeError, match="Missing adapter config"):
            LoRAAdapterMerger._load_adapter(adapter_dir)

    def test_missing_weights_raises(self, tmp_path):
        """Test missing adapter weights raises MergeError."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        # Create config but no weights
        config_path = adapter_dir / "adapter_config.json"
        config_path.write_text('{"r": 8}')

        with pytest.raises(MergeError, match="Missing adapter weights"):
            LoRAAdapterMerger._load_adapter(adapter_dir)
