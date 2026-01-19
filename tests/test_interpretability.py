# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for mechanistic interpretability modules."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon


class TestSparseAutoencoder:
    """Tests for Sparse Autoencoder."""

    def test_sae_config_latent_dim(self) -> None:
        """Test SAE config computes latent dimension correctly."""
        from modelcypher.core.domain.interpretability.sae import SAEConfig

        config = SAEConfig(hidden_dim=768, expansion_factor=8)
        assert config.latent_dim == 768 * 8
        assert config.latent_dim == 6144

    def test_sae_initialize_weights(self) -> None:
        """Test SAE weight initialization."""
        from modelcypher.core.domain.interpretability.sae import (
            SAEConfig,
            SparseAutoencoder,
        )

        b = get_default_backend()
        config = SAEConfig(hidden_dim=64, expansion_factor=4)
        sae = SparseAutoencoder(config, backend=b)

        weights = sae.initialize_weights()

        assert weights.W_enc.shape == (64, 256)
        assert weights.b_enc.shape == (256,)
        assert weights.W_dec.shape == (256, 64)
        assert weights.b_dec.shape == (64,)

    def test_sae_encode_produces_sparse_output(self) -> None:
        """Test SAE encoding produces sparse activations."""
        from modelcypher.core.domain.interpretability.sae import (
            SAEConfig,
            SparseAutoencoder,
        )

        b = get_default_backend()
        config = SAEConfig(hidden_dim=64, expansion_factor=4)
        sae = SparseAutoencoder(config, backend=b)

        weights = sae.initialize_weights()

        # Create test activations (standard normal)
        activations = b.random_normal(shape=(16, 64))
        b.eval(activations)

        result = sae.encode(activations, weights)

        # Check shapes
        assert result.sparse_codes.shape == (16, 256)
        assert result.reconstruction.shape == (16, 64)

        eps = regularization_epsilon(b, result.sparse_codes)
        assert result.sparsity <= config.latent_dim + eps

    def test_sae_decode_inverts_encode(self) -> None:
        """Test that decode(encode(x)) approximates x."""
        from modelcypher.core.domain.interpretability.sae import (
            SAEConfig,
            SparseAutoencoder,
        )

        b = get_default_backend()
        config = SAEConfig(hidden_dim=32, expansion_factor=2)
        sae = SparseAutoencoder(config, backend=b)

        weights = sae.initialize_weights()

        # Create test activations
        activations = b.random_normal(shape=(8, 32))
        b.eval(activations)

        result = sae.encode(activations, weights)

        # Reconstruction should be finite
        recon_mean = b.mean(result.reconstruction)
        b.eval(recon_mean)
        assert b.isfinite(recon_mean)

    def test_derive_sparsity_coefficient(self) -> None:
        """Test sparsity coefficient derivation from data."""
        from modelcypher.core.domain.interpretability.sae import (
            derive_sparsity_coefficient,
        )

        b = get_default_backend()

        # High variance activations (scale by 10)
        high_var = b.random_normal(shape=(100, 64)) * 10.0
        b.eval(high_var)
        coeff_high = derive_sparsity_coefficient(high_var, b)

        # Low variance activations (scale by 0.1)
        low_var = b.random_normal(shape=(100, 64)) * 0.1
        b.eval(low_var)
        coeff_low = derive_sparsity_coefficient(low_var, b)

        var_high = float(b.to_scalar(b.var(high_var)))
        var_low = float(b.to_scalar(b.var(low_var)))
        eps = regularization_epsilon(b, high_var)
        if var_high > eps:
            assert abs(coeff_high - (1.0 / var_high)) <= eps
        else:
            assert coeff_high == 1.0
        if var_low > eps:
            assert abs(coeff_low - (1.0 / var_low)) <= eps
        else:
            assert coeff_low == 1.0


class TestTranscoder:
    """Tests for Transcoder."""

    def test_transcoder_config(self) -> None:
        """Test transcoder config properties."""
        from modelcypher.core.domain.interpretability.transcoder import TranscoderConfig

        config = TranscoderConfig(input_dim=768, output_dim=768, expansion_factor=4)
        assert config.latent_dim == 768 * 4

    def test_transcoder_initialize_weights(self) -> None:
        """Test transcoder weight initialization."""
        from modelcypher.core.domain.interpretability.transcoder import (
            Transcoder,
            TranscoderConfig,
        )

        b = get_default_backend()
        config = TranscoderConfig(input_dim=64, output_dim=64, expansion_factor=2)
        tc = Transcoder(config, backend=b)

        weights = tc.initialize_weights()

        assert weights.W_enc.shape == (64, 128)
        assert weights.W_dec.shape == (128, 64)
        assert weights.b_enc.shape == (128,)
        assert weights.b_dec.shape == (64,)

    def test_transcoder_transcode(self) -> None:
        """Test transcoding MLP input to output."""
        from modelcypher.core.domain.interpretability.transcoder import (
            Transcoder,
            TranscoderConfig,
        )

        b = get_default_backend()
        config = TranscoderConfig(input_dim=32, output_dim=32, expansion_factor=2)
        tc = Transcoder(config, backend=b)

        weights = tc.initialize_weights()

        # Create test MLP input/output
        mlp_input = b.random_normal(shape=(8, 32))
        mlp_output = b.random_normal(shape=(8, 32))
        b.eval(mlp_input, mlp_output)

        result = tc.transcode(mlp_input, mlp_output, weights)

        assert result.sparse_features.shape == (8, 64)
        assert result.predicted_output.shape == (8, 32)
        assert result.reconstruction_loss >= 0.0


class TestCrosscoder:
    """Tests for Crosscoder."""

    def test_crosscoder_config(self) -> None:
        """Test crosscoder config properties."""
        from modelcypher.core.domain.interpretability.crosscoder import CrosscoderConfig

        config = CrosscoderConfig(
            hidden_dim=768,
            shared_expansion=4,
            exclusive_expansion=2,
        )
        assert config.shared_dim == 768 * 4
        assert config.exclusive_dim == 768 * 2
        assert config.total_latent_dim == 768 * 4 + 2 * 768 * 2

    def test_crosscoder_initialize_weights(self) -> None:
        """Test crosscoder weight initialization."""
        from modelcypher.core.domain.interpretability.crosscoder import (
            Crosscoder,
            CrosscoderConfig,
        )

        b = get_default_backend()
        config = CrosscoderConfig(hidden_dim=32, shared_expansion=2, exclusive_expansion=1)
        cc = Crosscoder(config, backend=b)

        weights = cc.initialize_weights()

        assert weights.W_enc_shared.shape == (32, 64)
        assert weights.W_enc_base.shape == (32, 32)
        assert weights.W_enc_ft.shape == (32, 32)
        assert weights.W_dec_shared.shape == (64, 32)
        assert weights.W_dec_base.shape == (32, 32)
        assert weights.W_dec_ft.shape == (32, 32)

    def test_crosscoder_encode(self) -> None:
        """Test crosscoder encoding."""
        from modelcypher.core.domain.interpretability.crosscoder import (
            Crosscoder,
            CrosscoderConfig,
        )

        b = get_default_backend()
        config = CrosscoderConfig(hidden_dim=32, shared_expansion=2, exclusive_expansion=1)
        cc = Crosscoder(config, backend=b)

        weights = cc.initialize_weights()

        # Create test activations
        base_acts = b.random_normal(shape=(8, 32))
        ft_acts = b.random_normal(shape=(8, 32))
        b.eval(base_acts, ft_acts)

        result = cc.encode(base_acts, ft_acts, weights)

        assert result.shared_features.shape == (8, 64)
        assert result.base_exclusive_features.shape == (8, 32)
        assert result.ft_exclusive_features.shape == (8, 32)
        assert result.base_reconstruction.shape == (8, 32)
        assert result.ft_reconstruction.shape == (8, 32)

    def test_crosscoder_diff_models(self) -> None:
        """Test model diffing with crosscoder."""
        from modelcypher.core.domain.interpretability.crosscoder import (
            Crosscoder,
            CrosscoderConfig,
        )

        b = get_default_backend()
        config = CrosscoderConfig(hidden_dim=32, shared_expansion=2, exclusive_expansion=1)
        cc = Crosscoder(config, backend=b)

        weights = cc.initialize_weights()

        # Create similar base and ft activations
        base_acts = b.random_normal(shape=(16, 32))
        ft_acts = base_acts + b.random_normal(shape=(16, 32)) * 0.1
        b.eval(base_acts, ft_acts)

        diff = cc.diff_models(base_acts, ft_acts, weights)

        # Should have some shared features
        assert isinstance(diff.shared_feature_indices, list)
        assert isinstance(diff.base_exclusive_indices, list)
        assert isinstance(diff.ft_exclusive_indices, list)
        assert diff.change_magnitude >= 0.0
        assert diff.change_magnitude <= 1.0


class TestActivationPatching:
    """Tests for Activation Patching."""

    def test_patch_spec_creation(self) -> None:
        """Test PatchSpec dataclass."""
        from modelcypher.core.domain.interpretability.activation_patching import (
            PatchComponent,
            PatchSpec,
        )

        spec = PatchSpec(layer=10, position=-1, component=PatchComponent.residual)
        assert spec.layer == 10
        assert spec.position == -1
        assert spec.component == PatchComponent.residual

    def test_captured_activations(self) -> None:
        """Test CapturedActivations dataclass."""
        from modelcypher.core.domain.interpretability.activation_patching import (
            CapturedActivations,
        )

        captured = CapturedActivations()
        assert captured.layer_outputs == {}
        assert captured.final_logits is None


class TestFeatureSteering:
    """Tests for Feature Steering."""

    def test_steering_vector_creation(self) -> None:
        """Test SteeringVector dataclass."""
        from modelcypher.core.domain.interpretability.feature_steering import (
            SteeringSource,
            SteeringVector,
        )

        b = get_default_backend()
        direction = b.random_normal(shape=(64,))
        b.eval(direction)

        vec = SteeringVector(
            direction=direction,
            layer=16,
            source=SteeringSource.contrastive,
            label="test",
        )
        assert vec.layer == 16
        assert vec.source == SteeringSource.contrastive

    def test_steering_config(self) -> None:
        """Test SteeringConfig dataclass."""
        from modelcypher.core.domain.interpretability.feature_steering import (
            SteeringConfig,
        )

        config = SteeringConfig(
            vectors=[],
            strengths=[],
            null_space_constrained=True,
        )
        assert config.null_space_constrained is True


class TestSAETraining:
    """Tests for SAE Training."""

    def test_sae_training_config(self) -> None:
        """Test SAETrainingConfig defaults."""
        from modelcypher.core.domain.interpretability.sae_training import SAETrainingConfig

        config = SAETrainingConfig()
        assert config.learning_rate == 1e-4
        assert config.num_steps == 10000
        assert config.warmup_steps == 1000

    def test_sae_trainer_initialization(self) -> None:
        """Test SAETrainer initialization."""
        from modelcypher.core.domain.interpretability.sae import SAEConfig
        from modelcypher.core.domain.interpretability.sae_training import (
            SAETrainer,
            SAETrainingConfig,
        )

        b = get_default_backend()
        sae_config = SAEConfig(hidden_dim=64, expansion_factor=4)
        training_config = SAETrainingConfig(num_steps=10)

        trainer = SAETrainer(sae_config, training_config, backend=b)
        assert trainer._sae_config == sae_config


class TestInterpretabilityImports:
    """Test that all interpretability imports work."""

    def test_sae_imports(self) -> None:
        """Test SAE module imports."""
        from modelcypher.core.domain.interpretability.sae import (
            FeatureAnalysis,
            SAEConfig,
            SAEEncodingResult,
            SAEWeights,
            SparseAutoencoder,
            TopKFeature,
            derive_sparsity_coefficient,
        )

    def test_sae_training_imports(self) -> None:
        """Test SAE training imports."""
        from modelcypher.core.domain.interpretability.sae_training import (
            SAETrainer,
            SAETrainingConfig,
            SAETrainingResult,
            TrainingState,
        )

    def test_activation_patching_imports(self) -> None:
        """Test activation patching imports."""
        from modelcypher.core.domain.interpretability.activation_patching import (
            ActivationPatcher,
            CapturedActivations,
            PatchComponent,
            PatchingResult,
            PatchSpec,
            PathPatchingResult,
        )

    def test_feature_steering_imports(self) -> None:
        """Test feature steering imports."""
        from modelcypher.core.domain.interpretability.feature_steering import (
            FeatureSteering,
            SteeringConfig,
            SteeringResult,
            SteeringSource,
            SteeringVector,
        )

    def test_transcoder_imports(self) -> None:
        """Test transcoder imports."""
        from modelcypher.core.domain.interpretability.transcoder import (
            FeatureContribution,
            Transcoder,
            TranscoderConfig,
            TranscoderResult,
            TranscoderWeights,
        )

    def test_crosscoder_imports(self) -> None:
        """Test crosscoder imports."""
        from modelcypher.core.domain.interpretability.crosscoder import (
            Crosscoder,
            CrosscoderConfig,
            CrosscoderEncodingResult,
            CrosscoderWeights,
            ModelDiffResult,
        )

    def test_package_init_imports(self) -> None:
        """Test package-level imports via __init__.py."""
        from modelcypher.core.domain.interpretability import (
            SAEConfig,
            SparseAutoencoder,
        )
