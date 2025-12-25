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
        noise = backend.random_randn(v.shape)
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
    """Test configuration presets."""

    def test_default_config(self):
        config = UnifiedMergeConfig.default()
        assert config.base_alpha == 0.5
        assert config.enable_permutation is True
        assert config.enable_rotation is True
        assert config.enable_zipper is True

    def test_conservative_config(self):
        config = UnifiedMergeConfig.conservative()
        assert config.base_alpha == 0.7  # Trust target more
        assert config.permutation_confidence_threshold == 0.7

    def test_aggressive_config(self):
        config = UnifiedMergeConfig.aggressive()
        assert config.base_alpha == 0.3  # Trust source more
        assert config.alignment_rank == 48


class TestUnifiedGeometricMerger:
    """Test the merger pipeline stages."""

    def test_merger_initialization(self, mock_model_loader):
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)
        assert merger.config is not None
        assert merger.config.base_alpha == 0.5

    def test_merger_with_custom_config(self, mock_model_loader):
        config = UnifiedMergeConfig(base_alpha=0.3, enable_permutation=False)
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader, config=config)
        assert merger.config.base_alpha == 0.3
        assert merger.config.enable_permutation is False

    def test_extract_layer_indices(self, real_weights, mock_model_loader):
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)
        indices = merger._extract_layer_indices(real_weights)
        assert 0 in indices
        assert 12 in indices
        assert len(indices) == 2

    def test_extract_layer_index(self, mock_model_loader):
        merger = UnifiedGeometricMerger(model_loader=mock_model_loader)
        assert merger._extract_layer_index("model.layers.5.self_attn.q_proj.weight") == 5
        assert merger._extract_layer_index("model.layers.12.mlp.up_proj.weight") == 12
        assert merger._extract_layer_index("model.embed_tokens.weight") is None


class TestStageModuleHelpers:
    """Test helper functions in stage modules."""

    def test_is_residual_output(self):
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _is_residual_output,
        )

        assert _is_residual_output("model.layers.0.self_attn.o_proj.weight") is True
        assert _is_residual_output("model.layers.0.mlp.down_proj.weight") is True
        assert _is_residual_output("model.layers.0.mlp.up_proj.weight") is False

    def test_is_v_proj(self):
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _is_v_proj,
        )

        assert _is_v_proj("model.layers.0.self_attn.v_proj.weight") is True
        assert _is_v_proj("model.layers.0.self_attn.q_proj.weight") is False

    def test_infer_hidden_dim(self, real_weights):
        from modelcypher.core.use_cases.merge_stages.stage_2_permute import (
            infer_hidden_dim,
        )

        hidden_dim = infer_hidden_dim(real_weights)
        assert hidden_dim == 896  # Qwen 0.5B hidden dim


class TestStageProbe:
    """Test Stage 1: PROBE (Fingerprinting)."""

    def test_probe_identical_weights(self, real_weights):
        """Identical weights should have high confidence."""
        merger = UnifiedGeometricMerger(UnifiedMergeConfig(probe_mode="fast"))

        probe_result, metrics = merger._stage_probe(
            source_weights=real_weights,
            target_weights=real_weights,
            source_fingerprints=None,
            target_fingerprints=None,
            source_model=None,
            target_model=None,
            tokenizer=None,
            source_path="",
            target_path="",
        )

        assert "confidences" in probe_result
        assert "mean_confidence" in metrics
        # Identical weights should have high confidence
        assert metrics["mean_confidence"] > 0.8

    def test_probe_perturbed_weights(self, source_target_weights):
        """Slightly different weights should have moderate-high confidence."""
        source, target = source_target_weights
        merger = UnifiedGeometricMerger(UnifiedMergeConfig(probe_mode="fast"))

        probe_result, metrics = merger._stage_probe(
            source_weights=source,
            target_weights=target,
            source_fingerprints=None,
            target_fingerprints=None,
            source_model=None,
            target_model=None,
            tokenizer=None,
            source_path="",
            target_path="",
        )

        # Slightly perturbed should still have reasonable confidence
        assert metrics["mean_confidence"] > 0.5


class TestStagePermute:
    """Test Stage 2: PERMUTE (Re-Basin)."""

    def test_permute_disabled(self):
        """Test permutation when disabled."""
        merger = UnifiedGeometricMerger(UnifiedMergeConfig(enable_permutation=False))
        permuted, metrics = merger._stage_permute({}, {}, None, {})
        assert metrics.get("skipped") is True

    def test_permute_low_confidence(self, real_weights):
        """Test permutation skipped on low confidence."""
        merger = UnifiedGeometricMerger(UnifiedMergeConfig(permutation_confidence_threshold=0.99))

        # Low confidence should skip permutation
        layer_confidences = {0: 0.3, 12: 0.4}
        permuted, metrics = merger._stage_permute(
            real_weights, real_weights, None, layer_confidences
        )
        assert metrics.get("skipped") is True
        assert metrics.get("reason") == "low_confidence"


class TestRealWeightProperties:
    """Test geometric properties of real model weights."""

    def test_real_weights_are_not_gaussian(self, real_weights):
        """Real weights have structure that Gaussian noise doesn't capture."""
        import numpy as np
        backend = get_default_backend()
        q_proj = real_weights["model.layers.0.self_attn.q_proj.weight"]

        # Generate Gaussian noise with same shape and stats
        backend.random_seed(42)
        gaussian = backend.random_randn(q_proj.shape)
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


class TestRotateBlendHelpers:
    """Test rotation and blending helper functions with real weights."""

    def test_compute_procrustes_rotation_identity(self, real_weights):
        """Procrustes on identical matrices gives identity."""
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_procrustes_rotation,
        )

        W = real_weights["model.layers.0.self_attn.q_proj.weight"]
        R, error = _compute_procrustes_rotation(W, W, rank=32)

        # Should be close to identity
        assert R.shape[0] == R.shape[1]
        assert error < 1e-5

    def test_compute_procrustes_rotation_is_orthogonal(self, source_target_weights):
        """Procrustes rotation is orthogonal."""
        import numpy as np
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_procrustes_rotation,
        )

        source, target = source_target_weights
        W_source = source["model.layers.0.self_attn.q_proj.weight"]
        W_target = target["model.layers.0.self_attn.q_proj.weight"]

        R, _ = _compute_procrustes_rotation(W_source, W_target, rank=32)

        # R @ R^T should be identity
        np.testing.assert_allclose(R @ R.T, np.eye(R.shape[0]), atol=1e-5)

    def test_full_rank_rotation_reduces_distance(self, source_target_weights):
        """Full-rank rotation reduces distance to target."""
        import numpy as np
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_full_rank_rotation,
        )

        source, target = source_target_weights
        # Use k_proj which is 128x896 (smaller, faster test)
        W_source = source["model.layers.0.self_attn.k_proj.weight"]
        W_target = target["model.layers.0.self_attn.k_proj.weight"]

        R, error = _compute_full_rank_rotation(W_source, W_target)

        # Rotated source should be closer to target
        original_dist = np.linalg.norm(W_source - W_target)
        aligned = R @ W_source
        aligned_dist = np.linalg.norm(aligned - W_target)

        assert aligned_dist <= original_dist


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


class TestZipperPropagation:
    """Test the geometric zipper (STAGE 5: PROPAGATE)."""

    def test_weight_matching_permutation_identity(self, real_weights):
        """Weight matching on identical matrices gives identity permutation."""
        import numpy as np
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_weight_matching_permutation,
        )

        W = real_weights["model.layers.0.self_attn.q_proj.weight"]
        P = _compute_weight_matching_permutation(W, W)

        # Should be identity
        np.testing.assert_allclose(P, np.eye(P.shape[0]), atol=1e-6)

    def test_weight_matching_permutation_shuffled(self, real_weights):
        """Weight matching recovers shuffled neurons."""
        import numpy as np
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_weight_matching_permutation,
        )

        W_source = real_weights["model.layers.0.self_attn.k_proj.weight"]  # 128x896

        # Shuffle rows
        np.random.seed(42)
        perm = np.random.permutation(W_source.shape[0])
        W_target = W_source[perm]

        P = _compute_weight_matching_permutation(W_source, W_target)

        # P @ W_source should equal W_target
        aligned = P @ W_source
        np.testing.assert_allclose(aligned, W_target, atol=1e-5)

    def test_permutation_is_orthogonal(self, source_target_weights):
        """Permutation matrix is orthogonal: P @ P^T = I."""
        import numpy as np
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_weight_matching_permutation,
        )

        source, target = source_target_weights
        W_source = source["model.layers.0.self_attn.k_proj.weight"]
        W_target = target["model.layers.0.self_attn.k_proj.weight"]

        P = _compute_weight_matching_permutation(W_source, W_target)

        # P @ P^T should be identity
        np.testing.assert_allclose(P @ P.T, np.eye(P.shape[0]), atol=1e-6)


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
        assert config.refusal_preservation_threshold == 0.7
        assert config.max_instability_threshold == 0.8


class TestIntrinsicDimensionGating:
    """Test intrinsic dimension helpers (dimensional hierarchy)."""

    def test_infer_hidden_dim(self, real_weights):
        """Test hidden dimension inference from weights."""
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _infer_hidden_dim,
        )

        hidden_dim = _infer_hidden_dim(real_weights)
        # Qwen2.5-0.5B has hidden_dim of 896
        assert hidden_dim == 896

    def test_compute_layer_intrinsic_dims(self, real_weights):
        """Test intrinsic dimension computation per layer."""
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_layer_intrinsic_dims,
        )

        layer_indices = [0]  # We only have layer 0 in fixtures
        intrinsic_dims = _compute_layer_intrinsic_dims(real_weights, layer_indices, threshold=0.01)

        assert 0 in intrinsic_dims
        # Intrinsic dim should be positive and less than hidden_dim
        assert 0 < intrinsic_dims[0] < 896

    def test_intrinsic_dim_complexity_ratio(self, real_weights):
        """Test complexity ratio is in expected range."""
        from modelcypher.core.use_cases.merge_stages.stage_3_5_rotate_blend import (
            _compute_layer_intrinsic_dims,
            _infer_hidden_dim,
        )

        hidden_dim = _infer_hidden_dim(real_weights)
        intrinsic_dims = _compute_layer_intrinsic_dims(real_weights, [0], threshold=0.01)

        complexity_ratio = intrinsic_dims[0] / hidden_dim
        # Typically 5-50% of dimensions are "significant"
        assert 0.01 < complexity_ratio < 1.0

    def test_intrinsic_dim_gating_config(self):
        """Test intrinsic dim gating config options exist."""
        config = UnifiedMergeConfig()
        assert hasattr(config, "enable_intrinsic_dim_gating")
        assert hasattr(config, "intrinsic_dim_strength")
        assert hasattr(config, "intrinsic_dim_threshold")
        # Default is False for GPU acceleration (uses numpy SVD otherwise)
        assert config.enable_intrinsic_dim_gating is False
        assert config.intrinsic_dim_threshold == 0.01
