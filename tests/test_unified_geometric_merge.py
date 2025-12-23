"""
Tests for the Unified Geometric Merge Pipeline.

Validates the 5-stage merge process: PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE
"""
import numpy as np
import pytest

from modelcypher.core.use_cases.unified_geometric_merge import (
    UnifiedGeometricMerger,
    UnifiedMergeConfig,
    UnifiedMergeResult,
)


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

    @pytest.fixture
    def small_model_weights(self):
        """Create minimal weight dicts for testing."""
        np.random.seed(42)
        hidden_dim = 64
        intermediate = 128

        def make_weights(seed_offset=0):
            np.random.seed(42 + seed_offset)
            return {
                "model.embed_tokens.weight": np.random.randn(100, hidden_dim).astype(np.float32),
                "model.layers.0.self_attn.q_proj.weight": np.random.randn(hidden_dim, hidden_dim).astype(np.float32),
                "model.layers.0.self_attn.k_proj.weight": np.random.randn(hidden_dim, hidden_dim).astype(np.float32),
                "model.layers.0.self_attn.v_proj.weight": np.random.randn(hidden_dim, hidden_dim).astype(np.float32),
                "model.layers.0.self_attn.o_proj.weight": np.random.randn(hidden_dim, hidden_dim).astype(np.float32),
                "model.layers.0.mlp.gate_proj.weight": np.random.randn(intermediate, hidden_dim).astype(np.float32),
                "model.layers.0.mlp.up_proj.weight": np.random.randn(intermediate, hidden_dim).astype(np.float32),
                "model.layers.0.mlp.down_proj.weight": np.random.randn(hidden_dim, intermediate).astype(np.float32),
            }

        return make_weights(0), make_weights(1)

    def test_merger_initialization(self):
        merger = UnifiedGeometricMerger()
        assert merger.config is not None
        assert merger.config.base_alpha == 0.5

    def test_merger_with_custom_config(self):
        config = UnifiedMergeConfig(base_alpha=0.3, enable_permutation=False)
        merger = UnifiedGeometricMerger(config)
        assert merger.config.base_alpha == 0.3
        assert merger.config.enable_permutation is False

    def test_extract_layer_indices(self, small_model_weights):
        source, target = small_model_weights
        merger = UnifiedGeometricMerger()
        indices = merger._extract_layer_indices(target)
        assert 0 in indices  # Layer 0 weights exist
        assert len(indices) == 1

    def test_extract_layer_index(self):
        merger = UnifiedGeometricMerger()
        assert merger._extract_layer_index("model.layers.5.self_attn.q_proj.weight") == 5
        assert merger._extract_layer_index("model.layers.12.mlp.up_proj.weight") == 12
        assert merger._extract_layer_index("model.embed_tokens.weight") is None

    def test_is_residual_output(self):
        merger = UnifiedGeometricMerger()
        assert merger._is_residual_output("model.layers.0.self_attn.o_proj.weight") is True
        assert merger._is_residual_output("model.layers.0.mlp.down_proj.weight") is True
        assert merger._is_residual_output("model.layers.0.mlp.up_proj.weight") is False

    def test_is_attention_input(self):
        merger = UnifiedGeometricMerger()
        assert merger._is_attention_input("model.layers.0.self_attn.q_proj.weight") is True
        assert merger._is_attention_input("model.layers.0.self_attn.k_proj.weight") is True
        assert merger._is_attention_input("model.layers.0.self_attn.v_proj.weight") is True
        assert merger._is_attention_input("model.layers.0.mlp.up_proj.weight") is False

    def test_is_mlp_input(self):
        merger = UnifiedGeometricMerger()
        assert merger._is_mlp_input("model.layers.0.mlp.gate_proj.weight") is True
        assert merger._is_mlp_input("model.layers.0.mlp.up_proj.weight") is True
        assert merger._is_mlp_input("model.layers.0.mlp.down_proj.weight") is False

    def test_infer_hidden_dim(self, small_model_weights):
        source, target = small_model_weights
        merger = UnifiedGeometricMerger()
        hidden_dim = merger._infer_hidden_dim(target)
        assert hidden_dim == 64


class TestStageProbe:
    """Test Stage 1: PROBE (Fingerprinting)."""

    @pytest.fixture
    def merger(self):
        return UnifiedGeometricMerger()

    def test_probe_identical_models(self, merger):
        """Identical models should have high confidence."""
        np.random.seed(42)
        # Use proper layer naming that matches regex: layers\.(\d+)\.
        weights = {"model.layers.0.mlp.weight": np.random.randn(32, 32).astype(np.float32)}

        intersection_map, metrics = merger._stage_probe(weights, weights, None, None)

        assert "correlations" in intersection_map
        assert "mean_confidence" in metrics
        # Identical weights should have reasonably high confidence
        # (ensemble mode combines CKA, cosine, jaccard)
        assert metrics["mean_confidence"] > 0.5

    def test_probe_different_models(self, merger):
        """Different random models should have lower confidence."""
        np.random.seed(42)
        source = {"layer.0.weight": np.random.randn(32, 32).astype(np.float32)}
        np.random.seed(999)
        target = {"layer.0.weight": np.random.randn(32, 32).astype(np.float32)}

        intersection_map, metrics = merger._stage_probe(source, target, None, None)

        # Random weights should have moderate to low confidence
        assert metrics["mean_confidence"] < 0.5


class TestStagePermute:
    """Test Stage 2: PERMUTE (Re-Basin)."""

    @pytest.fixture
    def merger(self):
        return UnifiedGeometricMerger()

    def test_permute_disabled(self, merger):
        """Test permutation when disabled."""
        merger.config = UnifiedMergeConfig(enable_permutation=False)
        permuted, metrics = merger._stage_permute({}, {}, {}, [])
        assert metrics.get("skipped") is True

    def test_permute_low_confidence(self, merger):
        """Test permutation skipped on low confidence."""
        merger.config = UnifiedMergeConfig(permutation_confidence_threshold=0.9)
        intersection = {"confidences": {0: 0.3, 1: 0.4}}  # Below threshold

        np.random.seed(42)
        weights = {
            "model.layers.0.mlp.up_proj.weight": np.random.randn(64, 32).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(64, 32).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(32, 64).astype(np.float32),
        }

        permuted, metrics = merger._stage_permute(weights, weights, intersection, [0, 1])
        assert metrics.get("skipped") is True
        assert metrics.get("reason") == "low_confidence"


class TestResultConversion:
    """Test report generation."""

    def test_unified_result_fields(self):
        """Test UnifiedMergeResult has all required fields."""
        from datetime import datetime

        result = UnifiedMergeResult(
            merged_weights={},
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
