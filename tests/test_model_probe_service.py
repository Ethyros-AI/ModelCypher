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

"""Property tests for ModelProbeService.

**Feature: cli-mcp-parity, Property 1: Model probe returns required fields**
**Validates: Requirements 2.1**
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from safetensors.numpy import save_file

from modelcypher.core.use_cases.model_probe_service import (
    ModelProbeService,
    ModelProbeResult,
)


def _create_mock_model(
    tmp_path: Path,
    architecture: str = "llama",
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_layers: int = 2,
) -> Path:
    """Create a mock model directory with config.json and safetensors weights."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "model_type": architecture,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_layers,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    
    tensors = {}
    for i in range(num_layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(hidden_size * 4, hidden_size).astype(np.float32)
    
    tensors["model.embed_tokens.weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    tensors["lm_head.weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    
    save_file(tensors, model_dir / "model.safetensors")
    
    return model_dir


# **Feature: cli-mcp-parity, Property 1: Model probe returns required fields**
# **Validates: Requirements 2.1**
@given(
    architecture=st.sampled_from(["llama", "mistral", "qwen2", "gemma"]),
    vocab_size=st.integers(min_value=1000, max_value=10000),
    hidden_size=st.sampled_from([64, 128, 256]),
    num_attention_heads=st.sampled_from([4, 8]),
)
@settings(max_examples=20, deadline=None)
def test_model_probe_returns_required_fields(
    tmp_path_factory,
    architecture: str,
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
):
    """Property 1: For any valid model path, probe() returns required fields with non-null values."""
    tmp_path = tmp_path_factory.mktemp("model")
    model_dir = _create_mock_model(
        tmp_path,
        architecture=architecture,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_layers=1,  # Use minimal layers for faster tests
    )
    
    service = ModelProbeService()
    result = service.probe(str(model_dir))
    
    assert isinstance(result, ModelProbeResult)
    assert result.architecture is not None
    assert result.architecture == architecture
    assert result.parameter_count is not None
    assert result.parameter_count > 0
    assert result.layers is not None
    assert len(result.layers) > 0
    assert result.vocab_size is not None
    assert result.vocab_size == vocab_size
    assert result.hidden_size is not None
    assert result.hidden_size == hidden_size
    assert result.num_attention_heads is not None
    assert result.num_attention_heads == num_attention_heads


def test_probe_missing_config_raises_error(tmp_path):
    """Test that probe raises ValueError when config.json is missing."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    service = ModelProbeService()
    with pytest.raises(ValueError, match="config.json not found"):
        service.probe(str(model_dir))


def test_probe_nonexistent_path_raises_error(tmp_path):
    """Test that probe raises ValueError for nonexistent path."""
    service = ModelProbeService()
    with pytest.raises(ValueError, match="does not exist"):
        service.probe(str(tmp_path / "nonexistent"))


def test_probe_file_instead_of_directory_raises_error(tmp_path):
    """Test that probe raises ValueError when path is a file, not directory."""
    file_path = tmp_path / "model.txt"
    file_path.write_text("not a model", encoding="utf-8")
    
    service = ModelProbeService()
    with pytest.raises(ValueError, match="not a directory"):
        service.probe(str(file_path))


def test_validate_merge_compatible_models(tmp_path):
    """Test validate_merge returns compatible=True for matching models."""
    model_a = _create_mock_model(
        tmp_path / "a",
        architecture="llama",
        vocab_size=32000,
        hidden_size=256,
        num_attention_heads=8,
    )
    model_b = _create_mock_model(
        tmp_path / "b",
        architecture="llama",
        vocab_size=32000,
        hidden_size=256,
        num_attention_heads=8,
    )
    
    service = ModelProbeService()
    result = service.validate_merge(str(model_a), str(model_b))
    
    assert result.compatible is True
    assert result.architecture_match is True
    assert result.vocab_match is True
    assert result.dimension_match is True
    assert len(result.warnings) == 0


def test_validate_merge_incompatible_architecture(tmp_path):
    """Test validate_merge detects architecture mismatch."""
    model_a = _create_mock_model(
        tmp_path / "a",
        architecture="llama",
        vocab_size=32000,
        hidden_size=256,
    )
    model_b = _create_mock_model(
        tmp_path / "b",
        architecture="mistral",
        vocab_size=32000,
        hidden_size=256,
    )
    
    service = ModelProbeService()
    result = service.validate_merge(str(model_a), str(model_b))
    
    assert result.compatible is False
    assert result.architecture_match is False
    assert "Architecture mismatch" in result.warnings[0]


# **Feature: cli-mcp-parity, Property 2: Model merge validation is symmetric for compatibility**
# **Validates: Requirements 2.2**
@given(
    arch_a=st.sampled_from(["llama", "mistral", "qwen2"]),
    arch_b=st.sampled_from(["llama", "mistral", "qwen2"]),
    vocab_a=st.sampled_from([1000, 2000, 5000]),
    vocab_b=st.sampled_from([1000, 2000, 5000]),
    hidden_a=st.sampled_from([64, 128]),
    hidden_b=st.sampled_from([64, 128]),
)
@settings(max_examples=20, deadline=None)
def test_merge_validation_symmetry(
    tmp_path_factory,
    arch_a: str,
    arch_b: str,
    vocab_a: int,
    vocab_b: int,
    hidden_a: int,
    hidden_b: int,
):
    """Property 2: validate_merge(A, B).compatible == validate_merge(B, A).compatible."""
    tmp_path = tmp_path_factory.mktemp("models")
    
    model_a = _create_mock_model(
        tmp_path / "a",
        architecture=arch_a,
        vocab_size=vocab_a,
        hidden_size=hidden_a,
        num_layers=1,
    )
    model_b = _create_mock_model(
        tmp_path / "b",
        architecture=arch_b,
        vocab_size=vocab_b,
        hidden_size=hidden_b,
        num_layers=1,
    )
    
    service = ModelProbeService()
    result_ab = service.validate_merge(str(model_a), str(model_b))
    result_ba = service.validate_merge(str(model_b), str(model_a))
    
    # Symmetry property: compatibility should be the same regardless of order
    assert result_ab.compatible == result_ba.compatible
    assert result_ab.architecture_match == result_ba.architecture_match
    assert result_ab.vocab_match == result_ba.vocab_match
    assert result_ab.dimension_match == result_ba.dimension_match


def test_analyze_alignment_identical_models(tmp_path):
    """Test analyze_alignment returns low drift for identical models."""
    model_a = _create_mock_model(
        tmp_path / "a",
        architecture="llama",
        vocab_size=1000,
        hidden_size=64,
        num_layers=1,
    )
    
    service = ModelProbeService()
    result = service.analyze_alignment(str(model_a), str(model_a))
    
    assert result.drift_magnitude == 0.0
    assert result.assessment == "highly_aligned"
    assert 0.0 <= result.drift_magnitude <= 1.0


def test_analyze_alignment_different_models(tmp_path):
    """Test analyze_alignment returns bounded drift for different models."""
    model_a = _create_mock_model(
        tmp_path / "a",
        architecture="llama",
        vocab_size=1000,
        hidden_size=64,
        num_layers=1,
    )
    model_b = _create_mock_model(
        tmp_path / "b",
        architecture="llama",
        vocab_size=1000,
        hidden_size=64,
        num_layers=1,
    )
    
    service = ModelProbeService()
    result = service.analyze_alignment(str(model_a), str(model_b))
    
    # Drift should be bounded in [0.0, 1.0]
    assert 0.0 <= result.drift_magnitude <= 1.0
    assert result.assessment in ["highly_aligned", "moderately_aligned", "divergent", "highly_divergent"]
    assert len(result.layer_drifts) > 0


# **Feature: cli-mcp-parity, Property 3: Alignment analysis returns bounded drift**
# **Validates: Requirements 2.3**
@given(
    arch=st.sampled_from(["llama", "mistral"]),
    vocab=st.sampled_from([1000, 2000]),
    hidden=st.sampled_from([64, 128]),
)
@settings(max_examples=20, deadline=None)
def test_alignment_analysis_bounded_drift(
    tmp_path_factory,
    arch: str,
    vocab: int,
    hidden: int,
):
    """Property 3: analyze_alignment(A, B).drift_magnitude is in range [0.0, 1.0]."""
    tmp_path = tmp_path_factory.mktemp("models")
    
    model_a = _create_mock_model(
        tmp_path / "a",
        architecture=arch,
        vocab_size=vocab,
        hidden_size=hidden,
        num_layers=1,
    )
    model_b = _create_mock_model(
        tmp_path / "b",
        architecture=arch,
        vocab_size=vocab,
        hidden_size=hidden,
        num_layers=1,
    )
    
    service = ModelProbeService()
    result = service.analyze_alignment(str(model_a), str(model_b))
    
    # Property: drift_magnitude must be bounded in [0.0, 1.0]
    assert 0.0 <= result.drift_magnitude <= 1.0
    
    # All layer drifts should also be bounded
    for layer_drift in result.layer_drifts:
        assert 0.0 <= layer_drift.drift_magnitude <= 1.0
    
    # Assessment should be one of the valid values
    assert result.assessment in ["highly_aligned", "moderately_aligned", "divergent", "highly_divergent", "incompatible"]
