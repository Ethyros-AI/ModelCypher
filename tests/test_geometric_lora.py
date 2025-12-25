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

"""Tests for geometric LoRA generation (geometric_lora.py).

Validates the implementation of low-rank adaptation derived from
geometric specifications, as described in Hu et al. (2021) LoRA paper.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.geometric_lora import (
    AdaptationQuality,
    GeometricLoRA,
    GeometricLoRAConfig,
    GeometricLoRAGenerator,
    LayerLoRAWeights,
    generate_geometric_lora,
)
from modelcypher.core.domain.geometry.manifold_transfer import (
    AnchorDistanceProfile,
    ProjectionQuality,
    TransferPoint,
)


class TestGeometricLoRAConfig:
    """Tests for GeometricLoRAConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GeometricLoRAConfig()

        assert config.target_rank == 4
        assert config.auto_rank is True
        assert config.singular_value_threshold == 0.01
        assert config.max_rank == 64
        assert config.min_rank == 1
        assert config.regularization == 1e-6
        assert config.scale_factor == 1.0
        assert "q_proj" in config.target_projections
        assert "v_proj" in config.target_projections

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = GeometricLoRAConfig(
            target_rank=8,
            auto_rank=False,
            max_rank=32,
            target_projections=["q_proj", "k_proj", "v_proj"],
        )

        assert config.target_rank == 8
        assert config.auto_rank is False
        assert config.max_rank == 32
        assert len(config.target_projections) == 3


class TestLayerLoRAWeights:
    """Tests for LayerLoRAWeights dataclass."""

    @pytest.fixture
    def sample_weights(self) -> LayerLoRAWeights:
        """Create sample LoRA weights for testing."""
        backend = get_default_backend()
        rank = 4
        in_features = 512
        out_features = 512

        backend.random_seed(42)
        A = backend.random_randn((rank, in_features)) * 0.01
        B = backend.random_randn((out_features, rank)) * 0.01
        singular_values = backend.array([1.0, 0.5, 0.2, 0.1])

        return LayerLoRAWeights(
            layer_idx=0,
            projection_name="q_proj",
            A=A,
            B=B,
            rank=rank,
            singular_values=singular_values,
            geometric_loss=0.05,
        )

    def test_in_features(self, sample_weights: LayerLoRAWeights) -> None:
        """Test in_features property."""
        assert sample_weights.in_features == 512

    def test_out_features(self, sample_weights: LayerLoRAWeights) -> None:
        """Test out_features property."""
        assert sample_weights.out_features == 512

    def test_delta_W(self, sample_weights: LayerLoRAWeights) -> None:
        """Test delta_W reconstruction."""
        backend = get_default_backend()
        delta_W = sample_weights.delta_W

        assert delta_W.shape == (512, 512)
        # Verify it's the product B @ A
        expected = sample_weights.B @ sample_weights.A
        assert backend.allclose(delta_W, expected)

    def test_effective_rank(self, sample_weights: LayerLoRAWeights) -> None:
        """Test effective rank computation."""
        # With singular values [1.0, 0.5, 0.2, 0.1], all > 0.01
        eff_rank = sample_weights.effective_rank
        assert eff_rank == 4.0

    def test_to_dict(self, sample_weights: LayerLoRAWeights) -> None:
        """Test serialization to dictionary."""
        d = sample_weights.to_dict()

        assert d["layer"] == 0
        assert d["projection"] == "q_proj"
        assert d["rank"] == 4
        assert d["inFeatures"] == 512
        assert d["outFeatures"] == 512
        assert d["geometricLoss"] == pytest.approx(0.05)


class TestGeometricLoRA:
    """Tests for GeometricLoRA dataclass."""

    @pytest.fixture
    def sample_transfer_point(self) -> TransferPoint:
        """Create a sample transfer point."""
        backend = get_default_backend()
        profile = AnchorDistanceProfile(
            concept_id="test_concept",
            anchor_ids=["a1", "a2"],
            distances=backend.array([1.0, 2.0]),
            weights=backend.array([0.5, 0.5]),
            source_curvature=None,
            source_volume=None,
        )
        backend.random_seed(42)
        return TransferPoint(
            concept_id="test_concept",
            source_profile=profile,
            coordinates=backend.random_randn((512,)),
            projected_volume=None,
            stress=0.05,
            quality=ProjectionQuality.GOOD,
            curvature_mismatch=0.02,
            confidence=0.9,
        )

    @pytest.fixture
    def sample_lora(
        self,
        sample_transfer_point: TransferPoint,
    ) -> GeometricLoRA:
        """Create a sample GeometricLoRA."""
        weights = [
            LayerLoRAWeights(
                layer_idx=i,
                projection_name="q_proj",
                A=np.random.randn(4, 512) * 0.01,
                B=np.random.randn(512, 4) * 0.01,
                rank=4,
                singular_values=np.array([1.0, 0.5, 0.2, 0.1]),
                geometric_loss=0.05,
            )
            for i in range(4)
        ]

        return GeometricLoRA(
            transfer_point=sample_transfer_point,
            weights=weights,
            config=GeometricLoRAConfig(),
            mean_geometric_loss=0.05,
            total_rank=16,
            quality=AdaptationQuality.OPTIMAL,
        )

    def test_num_layers(self, sample_lora: GeometricLoRA) -> None:
        """Test num_layers property."""
        assert sample_lora.num_layers == 4

    def test_num_parameters(self, sample_lora: GeometricLoRA) -> None:
        """Test num_parameters property."""
        # Each layer: A is 4x512, B is 512x4
        # Total per layer: 4*512 + 512*4 = 4096
        # Total: 4 layers * 4096 = 16384
        assert sample_lora.num_parameters == 16384

    def test_get_weights_for_layer(self, sample_lora: GeometricLoRA) -> None:
        """Test getting weights for a specific layer."""
        layer_0_weights = sample_lora.get_weights_for_layer(0)
        assert len(layer_0_weights) == 1
        assert layer_0_weights[0].layer_idx == 0

        layer_2_weights = sample_lora.get_weights_for_layer(2)
        assert len(layer_2_weights) == 1
        assert layer_2_weights[0].layer_idx == 2

    def test_to_safetensors_dict(self, sample_lora: GeometricLoRA) -> None:
        """Test conversion to safetensors format."""
        tensors = sample_lora.to_safetensors_dict()

        # Should have A and B for each layer
        assert len(tensors) == 8  # 4 layers * 2 (A + B)

        # Check naming convention
        assert "base_model.model.layers.0.self_attn.q_proj.lora_A.weight" in tensors
        assert "base_model.model.layers.0.self_attn.q_proj.lora_B.weight" in tensors

    def test_to_dict(self, sample_lora: GeometricLoRA) -> None:
        """Test serialization to dictionary."""
        d = sample_lora.to_dict()

        assert d["conceptId"] == "test_concept"
        assert d["numLayers"] == 4
        assert d["totalRank"] == 16
        assert d["numParameters"] == 16384
        assert d["quality"] == "optimal"


class TestGeometricLoRAGenerator:
    """Tests for GeometricLoRAGenerator."""

    @pytest.fixture
    def generator(self) -> GeometricLoRAGenerator:
        """Create a generator with default config."""
        return GeometricLoRAGenerator()

    @pytest.fixture
    def sample_inputs(self) -> tuple[TransferPoint, dict, dict]:
        """Generate sample inputs for testing."""
        np.random.seed(42)
        d = 256

        # Create transfer point
        profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=[f"anchor_{i}" for i in range(10)],
            distances=np.random.rand(10),
            weights=np.ones(10) / 10,
            source_curvature=None,
            source_volume=None,
        )
        transfer_point = TransferPoint(
            concept_id="test",
            source_profile=profile,
            coordinates=np.random.randn(d),
            projected_volume=None,
            stress=0.05,
            quality=ProjectionQuality.GOOD,
            curvature_mismatch=0.02,
            confidence=0.9,
        )

        # Model weights
        model_weights = {
            layer: {
                "q_proj": np.random.randn(d, d) * 0.01,
                "v_proj": np.random.randn(d, d) * 0.01,
            }
            for layer in range(4)
        }

        # Anchor activations
        anchor_activations = {f"anchor_{i}": np.random.randn(3, d) for i in range(10)}

        return transfer_point, model_weights, anchor_activations

    def test_generate_produces_lora(
        self,
        generator: GeometricLoRAGenerator,
        sample_inputs: tuple[TransferPoint, dict, dict],
    ) -> None:
        """Test that generate produces a valid LoRA."""
        transfer_point, model_weights, anchor_activations = sample_inputs

        lora = generator.generate(
            transfer_point=transfer_point,
            model_weights=model_weights,
            anchor_activations=anchor_activations,
        )

        assert lora is not None
        assert lora.num_layers > 0
        assert lora.total_rank > 0
        assert lora.num_parameters > 0

    def test_generate_respects_target_layers(
        self,
        sample_inputs: tuple[TransferPoint, dict, dict],
    ) -> None:
        """Test that target_layers config is respected."""
        transfer_point, model_weights, anchor_activations = sample_inputs

        config = GeometricLoRAConfig(target_layers=[0, 2])
        generator = GeometricLoRAGenerator(config)

        lora = generator.generate(
            transfer_point=transfer_point,
            model_weights=model_weights,
            anchor_activations=anchor_activations,
        )

        # Should only have weights for layers 0 and 2
        layer_indices = set(w.layer_idx for w in lora.weights)
        assert layer_indices == {0, 2}

    def test_generate_respects_target_projections(
        self,
        sample_inputs: tuple[TransferPoint, dict, dict],
    ) -> None:
        """Test that target_projections config is respected."""
        transfer_point, model_weights, anchor_activations = sample_inputs

        config = GeometricLoRAConfig(target_projections=["q_proj"])
        generator = GeometricLoRAGenerator(config)

        lora = generator.generate(
            transfer_point=transfer_point,
            model_weights=model_weights,
            anchor_activations=anchor_activations,
        )

        # Should only have q_proj weights
        projections = set(w.projection_name for w in lora.weights)
        assert projections == {"q_proj"}

    def test_determine_rank_auto(
        self,
        generator: GeometricLoRAGenerator,
    ) -> None:
        """Test automatic rank determination from singular values."""
        # Singular values with clear cutoff
        sv = np.array([1.0, 0.5, 0.1, 0.001, 0.0001])

        rank = generator._determine_rank(sv)

        # Should select rank based on threshold (0.01 * 1.0 = 0.01)
        # Values > 0.01: 1.0, 0.5, 0.1 → rank 3
        assert rank == 3

    def test_determine_rank_fixed(self) -> None:
        """Test fixed rank when auto_rank is False."""
        config = GeometricLoRAConfig(target_rank=2, auto_rank=False)
        generator = GeometricLoRAGenerator(config)

        sv = np.array([1.0, 0.5, 0.1, 0.001])

        rank = generator._determine_rank(sv)

        assert rank == 2

    def test_assess_quality_optimal(
        self,
        generator: GeometricLoRAGenerator,
    ) -> None:
        """Test optimal quality assessment."""
        weights = [
            LayerLoRAWeights(
                layer_idx=0,
                projection_name="q_proj",
                A=np.zeros((2, 64)),
                B=np.zeros((64, 2)),
                rank=2,
                singular_values=np.array([1.0, 0.5]),
                geometric_loss=0.05,  # < 0.1
            )
        ]

        quality = generator._assess_quality(weights)
        assert quality == AdaptationQuality.OPTIMAL

    def test_assess_quality_compressed(
        self,
        generator: GeometricLoRAGenerator,
    ) -> None:
        """Test compressed quality assessment."""
        weights = [
            LayerLoRAWeights(
                layer_idx=0,
                projection_name="q_proj",
                A=np.zeros((2, 64)),
                B=np.zeros((64, 2)),
                rank=2,
                singular_values=np.array([1.0, 0.5]),
                geometric_loss=0.2,  # 0.1 <= loss < 0.3
            )
        ]

        quality = generator._assess_quality(weights)
        assert quality == AdaptationQuality.COMPRESSED

    def test_assess_quality_degraded(
        self,
        generator: GeometricLoRAGenerator,
    ) -> None:
        """Test degraded quality assessment."""
        weights = [
            LayerLoRAWeights(
                layer_idx=0,
                projection_name="q_proj",
                A=np.zeros((4, 64)),
                B=np.zeros((64, 4)),
                rank=4,
                singular_values=np.array([1.0, 0.5, 0.2, 0.1]),
                geometric_loss=0.5,  # >= 0.3
            )
        ]

        quality = generator._assess_quality(weights)
        assert quality == AdaptationQuality.DEGRADED


class TestGenerateGeometricLoraFunction:
    """Tests for the convenience function."""

    def test_generate_geometric_lora(self) -> None:
        """Test the convenience function."""
        np.random.seed(42)
        d = 128

        profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a1", "a2", "a3"],
            distances=np.array([1.0, 2.0, 3.0]),
            weights=np.array([0.4, 0.35, 0.25]),
            source_curvature=None,
            source_volume=None,
        )

        transfer_point = TransferPoint(
            concept_id="test",
            source_profile=profile,
            coordinates=np.random.randn(d),
            projected_volume=None,
            stress=0.05,
            quality=ProjectionQuality.GOOD,
            curvature_mismatch=0.02,
            confidence=0.9,
        )

        model_weights = {
            0: {
                "q_proj": np.random.randn(d, d) * 0.01,
                "v_proj": np.random.randn(d, d) * 0.01,
            }
        }

        anchor_activations = {
            "a1": np.random.randn(2, d),
            "a2": np.random.randn(2, d),
            "a3": np.random.randn(2, d),
        }

        lora = generate_geometric_lora(
            transfer_point=transfer_point,
            model_weights=model_weights,
            anchor_activations=anchor_activations,
        )

        assert lora is not None
        assert lora.transfer_point.concept_id == "test"


class TestAdaptationQuality:
    """Tests for AdaptationQuality enum."""

    def test_quality_values(self) -> None:
        """Test quality enum values."""
        assert AdaptationQuality.OPTIMAL.value == "optimal"
        assert AdaptationQuality.COMPRESSED.value == "compressed"
        assert AdaptationQuality.MINIMAL.value == "minimal"
        assert AdaptationQuality.DEGRADED.value == "degraded"
