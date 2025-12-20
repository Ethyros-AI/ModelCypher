"""Property tests for GeometryPrimesService.

**Feature: cli-mcp-parity, Property 4-6: Geometry primes properties**
**Validates: Requirements 3.1, 3.2, 3.3**
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from safetensors.numpy import save_file

from modelcypher.core.use_cases.geometry_primes_service import (
    GeometryPrimesService,
    SemanticPrime,
    PrimeActivation,
    PrimeComparisonResult,
)


def _create_mock_model(
    tmp_path: Path,
    vocab_size: int = 1000,
    hidden_size: int = 64,
) -> Path:
    """Create a mock model directory with config.json and safetensors weights."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    
    tensors = {
        "model.embed_tokens.weight": np.random.randn(vocab_size, hidden_size).astype(np.float32),
    }
    save_file(tensors, model_dir / "model.safetensors")
    
    return model_dir


# **Feature: cli-mcp-parity, Property 4: Geometry primes list is non-empty**
# **Validates: Requirements 3.1**
def test_primes_list_non_empty():
    """Property 4: list_primes() returns at least one semantic prime with valid id and name."""
    service = GeometryPrimesService()
    primes = service.list_primes()
    
    assert len(primes) > 0, "Primes list should not be empty"
    
    for prime in primes:
        assert isinstance(prime, SemanticPrime)
        assert prime.id is not None and len(prime.id) > 0
        assert prime.name is not None and len(prime.name) > 0
        assert prime.category is not None and len(prime.category) > 0


def test_primes_list_contains_expected_categories():
    """Test that primes list contains expected semantic categories."""
    service = GeometryPrimesService()
    primes = service.list_primes()
    
    categories = {p.category for p in primes}
    
    # Should have at least some core categories
    assert "substantives" in categories
    assert "mentalPredicates" in categories
    assert "logicalConcepts" in categories


def test_primes_list_is_cached():
    """Test that primes list is cached for performance."""
    service = GeometryPrimesService()
    primes1 = service.list_primes()
    primes2 = service.list_primes()
    
    assert primes1 is primes2, "Primes list should be cached"


# **Feature: cli-mcp-parity, Property 5: Prime probe returns activations for all primes**
# **Validates: Requirements 3.2**
@given(
    vocab_size=st.integers(min_value=100, max_value=1000),
    hidden_size=st.sampled_from([32, 64, 128]),
)
@settings(max_examples=10, deadline=None)
def test_prime_probe_returns_activations_for_all_primes(
    tmp_path_factory,
    vocab_size: int,
    hidden_size: int,
):
    """Property 5: probe(model) returns activations for each prime in the inventory."""
    tmp_path = tmp_path_factory.mktemp("model")
    model_dir = _create_mock_model(tmp_path, vocab_size=vocab_size, hidden_size=hidden_size)
    
    service = GeometryPrimesService()
    primes = service.list_primes()
    activations = service.probe(str(model_dir))
    
    # Should have activation for each prime
    assert len(activations) == len(primes)
    
    # Each activation should reference a valid prime
    prime_ids = {p.id for p in primes}
    for activation in activations:
        assert isinstance(activation, PrimeActivation)
        assert activation.prime_id in prime_ids
        assert 0.0 <= activation.activation_strength <= 1.0


def test_probe_nonexistent_path_raises_error(tmp_path):
    """Test that probe raises ValueError for nonexistent path."""
    service = GeometryPrimesService()
    with pytest.raises(ValueError, match="does not exist"):
        service.probe(str(tmp_path / "nonexistent"))


def test_probe_file_instead_of_directory_raises_error(tmp_path):
    """Test that probe raises ValueError when path is a file."""
    file_path = tmp_path / "model.txt"
    file_path.write_text("not a model", encoding="utf-8")
    
    service = GeometryPrimesService()
    with pytest.raises(ValueError, match="not a directory"):
        service.probe(str(file_path))


# **Feature: cli-mcp-parity, Property 6: Prime comparison is symmetric for alignment score**
# **Validates: Requirements 3.3**
@given(
    vocab_a=st.integers(min_value=100, max_value=500),
    vocab_b=st.integers(min_value=100, max_value=500),
    hidden=st.sampled_from([32, 64]),
)
@settings(max_examples=10, deadline=None)
def test_prime_comparison_symmetry(
    tmp_path_factory,
    vocab_a: int,
    vocab_b: int,
    hidden: int,
):
    """Property 6: compare(A, B).alignment_score == compare(B, A).alignment_score."""
    tmp_path = tmp_path_factory.mktemp("models")
    
    model_a = _create_mock_model(tmp_path / "a", vocab_size=vocab_a, hidden_size=hidden)
    model_b = _create_mock_model(tmp_path / "b", vocab_size=vocab_b, hidden_size=hidden)
    
    service = GeometryPrimesService()
    result_ab = service.compare(str(model_a), str(model_b))
    result_ba = service.compare(str(model_b), str(model_a))
    
    # Symmetry property: alignment score should be the same regardless of order
    assert abs(result_ab.alignment_score - result_ba.alignment_score) < 1e-6


def test_compare_returns_valid_result(tmp_path):
    """Test that compare returns a valid PrimeComparisonResult."""
    model_a = _create_mock_model(tmp_path / "a")
    model_b = _create_mock_model(tmp_path / "b")
    
    service = GeometryPrimesService()
    result = service.compare(str(model_a), str(model_b))
    
    assert isinstance(result, PrimeComparisonResult)
    assert 0.0 <= result.alignment_score <= 1.0
    assert isinstance(result.divergent_primes, list)
    assert isinstance(result.convergent_primes, list)
    assert result.interpretation is not None and len(result.interpretation) > 0


def test_compare_identical_models_high_alignment(tmp_path):
    """Test that comparing a model with itself yields high alignment."""
    model = _create_mock_model(tmp_path)
    
    service = GeometryPrimesService()
    result = service.compare(str(model), str(model))
    
    # Same model should have perfect alignment
    assert result.alignment_score == 1.0
    assert len(result.divergent_primes) == 0
