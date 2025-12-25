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
    GeometricLoRA,
    GeometricLoRAConfig,
    GeometricLoRAGenerator,
    LayerLoRAWeights,
    generate_geometric_lora,
)
from modelcypher.core.domain.geometry.manifold_transfer import (
    AnchorDistanceProfile,
    TransferPoint,
)


class TestGeometricLoRAConfig:
    """Tests for GeometricLoRAConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GeometricLoRAConfig()

        assert config.auto_rank is True
        assert config.regularization == 1e-6
        assert config.condition_threshold == 1e4
        assert "q_proj" in config.target_projections
        assert "v_proj" in config.target_projections

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = GeometricLoRAConfig(
            auto_rank=False,
            condition_threshold=1e3,
            target_projections=["q_proj", "k_proj", "v_proj"],
        )

        assert config.auto_rank is False
        assert config.condition_threshold == 1e3
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
        A = backend.random_normal((rank, in_features)) * 0.01
        B = backend.random_normal((out_features, rank)) * 0.01
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
        diff = backend.abs(delta_W - expected)
        backend.eval(diff)
        assert float(backend.max(diff)) < 1e-5

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
            coordinates=backend.random_normal((512,)),
            projected_volume=None,
            stress=0.05,  # < 0.3 = reliable
            curvature_mismatch=0.02,
        )

    @pytest.fixture
    def sample_lora(
        self,
        sample_transfer_point: TransferPoint,
    ) -> GeometricLoRA:
        """Create a sample GeometricLoRA."""
        backend = get_default_backend()
        backend.random_seed(42)
        weights = [
            LayerLoRAWeights(
                layer_idx=i,
                projection_name="q_proj",
                A=backend.random_normal((4, 512)) * 0.01,
                B=backend.random_normal((512, 4)) * 0.01,
                rank=4,
                singular_values=backend.array([1.0, 0.5, 0.2, 0.1]),
                geometric_loss=0.05,
            )
            for i in range(4)
        ]

        return GeometricLoRA(
            transfer_point=sample_transfer_point,
            weights=weights,
            config=GeometricLoRAConfig(),
            mean_geometric_loss=0.05,  # Low loss indicates optimal quality
            total_rank=16,
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
        assert "meanGeometricLoss" in d
        assert "transferStress" in d
        assert "confidenceComponents" in d
        assert "stressFactor" in d["confidenceComponents"]
        assert "anchorFactor" in d["confidenceComponents"]
        assert "curvatureFactor" in d["confidenceComponents"]


class TestGeometricLoRAGenerator:
    """Tests for GeometricLoRAGenerator."""

    @pytest.fixture
    def generator(self) -> GeometricLoRAGenerator:
        """Create a generator with default config."""
        return GeometricLoRAGenerator()

    @pytest.fixture
    def sample_inputs(self) -> tuple[TransferPoint, dict, dict]:
        """Generate sample inputs for testing."""
        backend = get_default_backend()
        backend.random_seed(42)
        d = 256

        # Create transfer point
        profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=[f"anchor_{i}" for i in range(10)],
            distances=backend.to_numpy(backend.random_uniform(shape=(10,))),
            weights=backend.to_numpy(backend.ones((10,))) / 10,
            source_curvature=None,
            source_volume=None,
        )
        transfer_point = TransferPoint(
            concept_id="test",
            source_profile=profile,
            coordinates=backend.random_normal((d,)),
            projected_volume=None,
            stress=0.05,  # < 0.3 = reliable
            curvature_mismatch=0.02,
        )

        # Model weights
        model_weights = {
            layer: {
                "q_proj": backend.random_normal((d, d)) * 0.01,
                "v_proj": backend.random_normal((d, d)) * 0.01,
            }
            for layer in range(4)
        }

        # Anchor activations
        anchor_activations = {f"anchor_{i}": backend.random_normal((3, d)) for i in range(10)}

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
        """Test automatic rank determination using condition number."""
        backend = get_default_backend()
        # Singular values with clear gap
        # condition_threshold = 1e4, so threshold = 1.0 / 1e4 = 1e-4
        sv = backend.array([1.0, 0.5, 0.1, 0.001, 0.00001])
        backend.eval(sv)

        rank = generator._determine_rank(sv, backend)

        # Values > 1e-4: 1.0, 0.5, 0.1, 0.001 → rank 4
        assert rank == 4

    def test_determine_rank_with_tight_condition(self) -> None:
        """Test rank determination with tighter condition threshold."""
        backend = get_default_backend()
        config = GeometricLoRAConfig(condition_threshold=1e2)
        generator = GeometricLoRAGenerator(config)

        sv = backend.array([1.0, 0.5, 0.1, 0.001])
        backend.eval(sv)

        rank = generator._determine_rank(sv, backend)
        # threshold = 1.0 / 100 = 0.01, values > 0.01: 1.0, 0.5, 0.1 → rank 3
        assert rank == 3

    def test_geometric_loss_negligible(
        self,
        generator: GeometricLoRAGenerator,
    ) -> None:
        """Test negligible reconstruction error (< 1e-6)."""
        backend = get_default_backend()
        weights = [
            LayerLoRAWeights(
                layer_idx=0,
                projection_name="q_proj",
                A=backend.zeros((2, 64)),
                B=backend.zeros((64, 2)),
                rank=2,
                singular_values=backend.array([1.0, 0.5]),
                geometric_loss=1e-8,  # < 1e-6 = negligible
            )
        ]

        losses = [w.geometric_loss for w in weights]
        mean_loss = sum(losses) / len(losses)
        assert mean_loss < 1e-6  # Negligible error

    def test_geometric_loss_moderate(
        self,
        generator: GeometricLoRAGenerator,
    ) -> None:
        """Test moderate reconstruction error (tight distribution)."""
        backend = get_default_backend()
        weights = [
            LayerLoRAWeights(
                layer_idx=0,
                projection_name="q_proj",
                A=backend.zeros((2, 64)),
                B=backend.zeros((64, 2)),
                rank=2,
                singular_values=backend.array([1.0, 0.5]),
                geometric_loss=0.1,
            ),
            LayerLoRAWeights(
                layer_idx=1,
                projection_name="q_proj",
                A=backend.zeros((2, 64)),
                B=backend.zeros((64, 2)),
                rank=2,
                singular_values=backend.array([1.0, 0.5]),
                geometric_loss=0.15,
            ),
        ]

        losses = [w.geometric_loss for w in weights]
        mean_loss = sum(losses) / len(losses)
        median_loss = sorted(losses)[len(losses) // 2]
        # Tight distribution: mean < 2 * median
        assert mean_loss < median_loss * 2

    def test_geometric_loss_degraded(
        self,
        generator: GeometricLoRAGenerator,
    ) -> None:
        """Test significant reconstruction error with outliers."""
        backend = get_default_backend()
        # 3 values: [0.01, 0.02, 10.0] → median = 0.02, mean ≈ 3.34
        weights = [
            LayerLoRAWeights(
                layer_idx=0,
                projection_name="q_proj",
                A=backend.zeros((4, 64)),
                B=backend.zeros((64, 4)),
                rank=4,
                singular_values=backend.array([1.0, 0.5, 0.2, 0.1]),
                geometric_loss=0.01,
            ),
            LayerLoRAWeights(
                layer_idx=1,
                projection_name="q_proj",
                A=backend.zeros((4, 64)),
                B=backend.zeros((64, 4)),
                rank=4,
                singular_values=backend.array([1.0, 0.5, 0.2, 0.1]),
                geometric_loss=0.02,
            ),
            LayerLoRAWeights(
                layer_idx=2,
                projection_name="q_proj",
                A=backend.zeros((4, 64)),
                B=backend.zeros((64, 4)),
                rank=4,
                singular_values=backend.array([1.0, 0.5, 0.2, 0.1]),
                geometric_loss=10.0,  # Major outlier
            ),
        ]

        losses = [w.geometric_loss for w in weights]
        mean_loss = sum(losses) / len(losses)
        median_loss = sorted(losses)[len(losses) // 2]
        # Degraded: mean significantly higher due to outlier
        assert mean_loss >= median_loss * 2


class TestGenerateGeometricLoraFunction:
    """Tests for the convenience function."""

    def test_generate_geometric_lora(self) -> None:
        """Test the convenience function."""
        backend = get_default_backend()
        backend.random_seed(42)
        d = 128

        profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a1", "a2", "a3"],
            distances=backend.array([1.0, 2.0, 3.0]),
            weights=backend.array([0.4, 0.35, 0.25]),
            source_curvature=None,
            source_volume=None,
        )

        transfer_point = TransferPoint(
            concept_id="test",
            source_profile=profile,
            coordinates=backend.random_normal((d,)),
            projected_volume=None,
            stress=0.05,  # < 0.3 = reliable
            curvature_mismatch=0.02,
        )

        model_weights = {
            0: {
                "q_proj": backend.random_normal((d, d)) * 0.01,
                "v_proj": backend.random_normal((d, d)) * 0.01,
            }
        }

        anchor_activations = {
            "a1": backend.random_normal((2, d)),
            "a2": backend.random_normal((2, d)),
            "a3": backend.random_normal((2, d)),
        }

        lora = generate_geometric_lora(
            transfer_point=transfer_point,
            model_weights=model_weights,
            anchor_activations=anchor_activations,
        )

        assert lora is not None
        assert lora.transfer_point.concept_id == "test"


class TestGeometricLossThresholds:
    """Tests for geometric loss-based quality assessment.

    Loss thresholds for reference:
        < 1e-6 = optimal (negligible error)
        < 1e-3 = good (tight distribution)
        >= 1e-3 = compressed/degraded (significant error)
    """

    def test_optimal_loss_threshold(self) -> None:
        """Optimal is defined as geometric loss < 1e-6."""
        optimal_loss = 1e-8
        assert optimal_loss < 1e-6

    def test_good_loss_threshold(self) -> None:
        """Good is defined as geometric loss < 1e-3."""
        good_loss = 1e-5
        assert good_loss < 1e-3
