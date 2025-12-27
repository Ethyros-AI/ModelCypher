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

Validates the 6-stage merge process:
    VOCAB → PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE → VALIDATE

Uses REAL model weights from /Volumes/CodeCypher/caches/test_fixtures/
to validate geometric operations on actual latent space structure.
"""

from pathlib import Path

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.unified_geometric_merge import (
    UnifiedGeometricMerger,
    UnifiedMergeConfig,
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
    from safetensors.numpy import load_file

    return load_file(str(FIXTURE_PATH))


@pytest.fixture(scope="module")
def source_target_weights(real_weights):
    """Create source and target weight dicts with slight perturbation."""
    import numpy as np
    backend = get_default_backend()
    # Source = real weights
    source = real_weights.copy()

    # Target = slightly perturbed version (simulates fine-tuned model)
    target = {}
    backend.random_seed(42)
    for k, v in real_weights.items():
        # Add small noise to simulate fine-tuning delta
        v_tensor = backend.array(v)
        noise = backend.random_normal(v.shape)
        noise = backend.to_numpy(noise).astype(v.dtype) * 0.01 * np.std(v)
        target[k] = v + noise

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


class TestUnifiedMergeConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = UnifiedMergeConfig()
        assert config.probe_mode == "precise"
        assert config.max_probes == 0
        assert config.use_transport_guided is False
        assert config.output_quant is None

    def test_custom_config(self):
        """Test custom config values."""
        config = UnifiedMergeConfig(
            probe_mode="fast",
            max_probes=100,
            use_transport_guided=True,
        )
        assert config.probe_mode == "fast"
        assert config.max_probes == 100
        assert config.use_transport_guided is True


class TestUnifiedGeometricMerger:
    """Test the merger pipeline stages."""

    def test_merger_initialization(self, mock_model_loader):
        """Test merger initializes with default config."""
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)
        assert merger.config is not None
        assert merger.config.probe_mode == "precise"

    def test_merger_with_custom_config(self, mock_model_loader):
        """Test merger accepts custom config."""
        config = UnifiedMergeConfig(probe_mode="fast", max_probes=50)
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader, config=config)
        assert merger.config.probe_mode == "fast"
        assert merger.config.max_probes == 50

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


class TestStageModuleHelpers:
    """Test helper functions in stage modules."""

    def test_infer_hidden_dim(self, real_weights):
        """Test hidden dimension inference from weights."""
        from modelcypher.core.use_cases.merge_stages.stage_2_permute import (
            infer_hidden_dim,
        )

        hidden_dim = infer_hidden_dim(real_weights)
        assert hidden_dim == 896  # Qwen 0.5B hidden dim


class TestStageProbe:
    """Test Stage 1: PROBE (Fingerprinting)."""

    def test_probe_identical_weights(self, real_weights, mock_model_loader):
        """Identical weights should have high confidence."""
        merger = UnifiedGeometricMerger(
            model_loader=mock_model_loader,
            config=UnifiedMergeConfig(probe_mode="fast"),
        )

        probe_result, metrics, src_acts, tgt_acts = merger._stage_probe(
            source_weights=real_weights,
            target_weights=real_weights,
            source_model=None,
            target_model=None,
            source_tokenizer=None,
            target_tokenizer=None,
        )

        assert "confidences" in probe_result
        assert "mean_confidence" in metrics
        # Identical weights should have high confidence (fast mode may not be exactly 1.0)
        assert metrics["mean_confidence"] > 0.5

    def test_probe_perturbed_weights(self, source_target_weights, mock_model_loader):
        """Slightly different weights should have high confidence."""
        source, target = source_target_weights
        merger = UnifiedGeometricMerger(
            model_loader=mock_model_loader,
            config=UnifiedMergeConfig(probe_mode="fast"),
        )

        probe_result, metrics, src_acts, tgt_acts = merger._stage_probe(
            source_weights=source,
            target_weights=target,
            source_model=None,
            target_model=None,
            source_tokenizer=None,
            target_tokenizer=None,
        )

        # Slightly perturbed should still have reasonable confidence
        assert metrics["mean_confidence"] > 0.5


class TestStagePermute:
    """Test Stage 2: PERMUTE (Re-Basin)."""

    def test_permute_basic(self, real_weights, mock_model_loader):
        """Test basic permutation stage."""
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)

        layer_confidences = {0: 0.9, 12: 0.9}
        permuted, metrics = merger._stage_permute(
            real_weights, real_weights, layer_confidences, None
        )
        # Should return some result
        assert isinstance(permuted, dict)
        assert isinstance(metrics, dict)


class TestRealWeightProperties:
    """Test geometric properties of real model weights."""

    def test_real_weights_are_not_gaussian(self, real_weights):
        """Real weights have structure that Gaussian noise doesn't capture."""
        import numpy as np
        backend = get_default_backend()
        q_proj = real_weights["model.layers.0.self_attn.q_proj.weight"]

        # Generate Gaussian noise with same shape and stats
        backend.random_seed(42)
        gaussian = backend.random_normal(q_proj.shape)
        gaussian_np = backend.to_numpy(gaussian).astype(q_proj.dtype)
        gaussian_np = gaussian_np * np.std(q_proj) + np.mean(q_proj)

        # Real weights should have different distribution properties
        # Kurtosis of real weights differs from Gaussian (kurtosis=0)
        from scipy.stats import kurtosis

        real_kurt = kurtosis(q_proj.flatten())
        kurtosis(gaussian_np.flatten())

        # Real model weights typically have higher kurtosis (heavier tails)
        assert abs(real_kurt) > 0.5, "Real weights should have non-Gaussian kurtosis"

    def test_real_weights_have_structure(self, real_weights):
        """Real weights have low-rank structure."""
        import numpy as np
        q_proj = real_weights["model.layers.0.self_attn.q_proj.weight"]

        # Compute SVD and check singular value decay (using numpy for SVD)
        U, S, Vh = np.linalg.svd(q_proj, full_matrices=False)

        # Ratio of top singular value to sum (concentration)
        concentration = S[0] / S.sum()

        # Real weights should have some concentration in top singular values
        assert concentration > 0.01, "Real weights should have structured singular values"

    def test_sparsity_pattern(self, real_weights):
        """Real weights have specific sparsity patterns."""
        import numpy as np
        v_proj = real_weights["model.layers.0.self_attn.v_proj.weight"]

        # V-proj typically has higher near-zero sparsity
        near_zero = (np.abs(v_proj) < 0.01).mean()
        assert near_zero > 0.3, (
            f"V-proj should have significant near-zero values, got {near_zero:.1%}"
        )


class TestGeometricMerge:
    """Test the core geometric merge function."""

    def test_geometric_merge_identical(self, real_weights):
        """Geometric merge of identical weights returns weights unchanged."""
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            geometric_merge_weights,
        )
        backend = get_default_backend()

        W = backend.array(real_weights["model.layers.0.self_attn.q_proj.weight"])
        backend.eval(W)

        merged, metrics = geometric_merge_weights(W, W, backend, key="test")
        backend.eval(merged)

        # Merged should be very close to original (small numerical differences expected)
        diff = backend.abs(merged - W)
        backend.eval(diff)
        max_diff = float(backend.max(diff))
        # Allow small numerical tolerance due to SVD/SLERP operations
        assert max_diff < 0.01, f"Merge of identical weights should be nearly unchanged, got diff {max_diff}"

    def test_geometric_merge_different(self, source_target_weights):
        """Geometric merge of different weights produces valid result."""
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            geometric_merge_weights,
        )
        backend = get_default_backend()

        source, target = source_target_weights
        W_source = backend.array(source["model.layers.0.self_attn.q_proj.weight"])
        W_target = backend.array(target["model.layers.0.self_attn.q_proj.weight"])
        backend.eval(W_source, W_target)

        merged, metrics = geometric_merge_weights(W_source, W_target, backend, key="test")
        backend.eval(merged)

        # Merged should have same shape
        assert merged.shape == W_source.shape

        # Metrics should contain quality info
        assert "mode" in metrics or "merge_quality" in metrics


class TestResultConversion:
    """Test result dataclass."""

    def test_unified_result_fields(self):
        """Test UnifiedMergeResult has all required fields."""
        from datetime import datetime

        result = UnifiedMergeResult(
            merged_weights={},
            vocab_metrics={},
            probe_metrics={"mean_confidence": 0.8},
            permute_metrics={"layers_permuted": 5},
            rotate_metrics={"rotations_applied": 10},
            blend_metrics={"mean_alpha": 0.5},
            mean_confidence=0.8,
            mean_procrustes_error=0.05,
            layer_count=32,
            weight_count=200,
            timestamp=datetime.utcnow(),
        )

        assert result.mean_confidence == 0.8
        assert result.layer_count == 32
        assert result.safety_verdict == "not_validated"  # default


class TestStageValidate:
    """Test Stage 6: VALIDATE (Safety)."""

    def test_validate_disabled(self):
        """Test validation when disabled."""
        from modelcypher.core.use_cases.merge_stages.stage_6_validate import (
            ValidateConfig,
            stage_validate,
        )

        config = ValidateConfig(enable_safety_validation=False)
        result = stage_validate(
            merged_weights={},
            source_weights={},
            target_weights={},
            layer_confidences={},
            config=config,
            layer_indices=[],
            hidden_dim=896,
        )

        assert result.safety_verdict == "not_validated"
        assert result.metrics.get("skipped") is True

    def test_validate_config_defaults(self):
        """Test validation config has reasonable defaults."""
        from modelcypher.core.use_cases.merge_stages.stage_6_validate import (
            ValidateConfig,
        )

        config = ValidateConfig()
        assert config.enable_safety_validation is True
