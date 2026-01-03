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

"""
Tests for the Unified Geometric Merge Pipeline.

Validates the 4-stage null-space constrained transplant process:
    VOCAB → PROBE → TRANSPLANT → VALIDATE

Uses REAL model weights from /Volumes/CodeCypher/caches/test_fixtures/
to validate geometric operations on actual latent space structure.
"""

from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.merge import (
    UnifiedGeometricMerger,
    UnifiedMergeResult,
)

# Real weight fixture path
FIXTURE_PATH = Path("/Volumes/CodeCypher/caches/test_fixtures/qwen_0.5b_layers_0_12.safetensors")

# Skip all tests if fixture not available (CI environment)
pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists(), reason=f"Real weight fixture not found at {FIXTURE_PATH}"
)


@pytest.fixture(scope="module")
def real_weights():
    """Load real model weights from external fixture."""
    backend = get_default_backend()
    return backend.load_safetensors(str(FIXTURE_PATH))


@pytest.fixture(scope="module")
def source_target_weights(real_weights):
    """Create source and target weight dicts with slight perturbation."""
    backend = get_default_backend()
    # Source = real weights
    source = real_weights.copy()

    # Target = slightly perturbed version (simulates fine-tuned model)
    target = {}
    backend.random_seed(42)
    for k, v in real_weights.items():
        # Add small noise to simulate fine-tuning delta
        v_tensor = backend.array(v)
        noise = backend.random_normal(v_tensor.shape)
        scale = backend.std(v_tensor)
        noise = noise * scale * 0.01
        target[k] = v_tensor + noise

    return source, target


class MockModelLoader:
    """Mock model loader for testing."""

    def __init__(self, weights: dict | None = None):
        self._weights = weights or {}

    def load_model_for_training(self, model_path, lora_config=None):
        return None, None

    def load_weights_as_numpy(self, model_path):
        return self._weights


@pytest.fixture
def mock_model_loader():
    """Provide a mock model loader for tests."""
    return MockModelLoader()


@pytest.fixture
def mock_model_loader_with_weights(real_weights):
    """Provide a mock model loader with real weights."""
    return MockModelLoader(real_weights)


class TestUnifiedGeometricMerger:
    """Test the merger pipeline stages."""

    def test_merger_initialization(self, mock_model_loader):
        """Test merger initializes successfully."""
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)
        assert merger is not None

    def test_extract_layer_indices(self, real_weights, mock_model_loader):
        """Test layer index extraction from weight keys."""
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)
        indices = merger._extract_layer_indices(real_weights)
        assert 0 in indices
        assert 12 in indices
        assert len(indices) == 2

    def test_extract_layer_index(self, mock_model_loader):
        """Test single layer index extraction."""
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)
        assert merger._extract_layer_index("model.layers.5.self_attn.q_proj.weight") == 5
        assert merger._extract_layer_index("model.layers.12.mlp.up_proj.weight") == 12
        assert merger._extract_layer_index("model.embed_tokens.weight") is None


class TestStageProbe:
    """Test Stage 1: PROBE (Fingerprinting).

    NOTE: ProbeConfig was REMOVED. Probe always uses precise mode which requires
    loaded models to compute activation-level CKA. The "fast" weight-level mode
    was removed because it doesn't properly measure representational similarity.
    """

    def test_probe_requires_models(self, real_weights, mock_model_loader):
        """Probe stage requires loaded models for activation-level CKA."""
        import pytest

        merger = UnifiedGeometricMerger(
            model_loader=mock_model_loader,
        )

        # Probe stage should raise when models are not provided
        with pytest.raises(RuntimeError, match="requires loaded models"):
            merger._stage_probe(
                source_weights=real_weights,
                target_weights=real_weights,
                source_model=None,
                target_model=None,
                source_tokenizer=None,
                target_tokenizer=None,
            )


class TestResultConversion:
    """Test result dataclass."""

    def test_unified_result_fields(self):
        """Test UnifiedMergeResult has all required fields."""
        from datetime import datetime

        result = UnifiedMergeResult(
            merged_weights={},
            probe_metrics={"mean_cka": 0.8},
            permute_metrics={"skipped": True, "reason": "test"},
            transplant_metrics={"layers_transplanted": 5, "weights_transplanted": 10},
            mean_preserved_fraction=0.8,
            mean_procrustes_error=0.05,
            layer_count=32,
            weight_count=200,
            timestamp=datetime.utcnow(),
        )

        assert result.mean_preserved_fraction == 0.8
        assert result.layer_count == 32
        assert not hasattr(result, "safety_verdict")
        assert result.merge_strategy == "transplant"  # default


class TestStageValidate:
    """Test Stage 6: VALIDATE (Safety)."""

    # NOTE: ValidateConfig was REMOVED. Validation always runs all checks.
    # The test_validate_disabled test was removed since validation cannot be disabled.
    # The test_validate_config_defaults test was removed since there's no config.

    def test_validate_always_runs(self):
        """Test validation always runs all checks (ValidateConfig was REMOVED)."""
        from modelcypher.core.use_cases.merge.stages.validate import stage_validate

        result = stage_validate(
            merged_weights={},
            source_weights={},
            target_weights={},
            layer_confidences={},
            layer_indices=[],
            hidden_dim=896,
        )

        # Validation always runs - metrics are returned even if some checks skip
        assert "numerical_stability" in result.metrics
        assert "content_safety" in result.metrics
        assert "behavioral_probes" in result.metrics
        assert "circuit_breaker" in result.metrics
        assert "ridge_resistance" in result.metrics
        assert "refusal_preserved" not in result.metrics
