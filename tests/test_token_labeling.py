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

"""Tests for token_labeling.py - SAE-based token labeling for data filtering.

Tests cover:
- TokenLabelingConfig dataclass
- LatentActivationStats dataclass
- TokenLabelResult dataclass
- SAETokenLabeler.compute_activation_stats()
- SAETokenLabeler.label_tokens()
- SAETokenLabeler._expand_labels()
- SAETokenLabeler.calibrate_threshold()
- Edge cases: empty input, no domain latents, text boundaries
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.interpretability.token_labeling import (
    LatentActivationStats,
    SAETokenLabeler,
    TokenLabelingConfig,
    TokenLabelResult,
)


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestTokenLabelingConfig:
    """Tests for TokenLabelingConfig dataclass."""

    def test_default_values(self):
        """TokenLabelingConfig has correct defaults."""
        config = TokenLabelingConfig()
        assert config.min_active_latents == 2
        assert config.activation_threshold_sigma == 4.0
        assert config.expand_adjacent is True
        assert config.expansion_radius == 1

    def test_custom_values(self):
        """TokenLabelingConfig accepts custom values."""
        config = TokenLabelingConfig(
            min_active_latents=3,
            activation_threshold_sigma=3.0,
            expand_adjacent=False,
            expansion_radius=2,
        )
        assert config.min_active_latents == 3
        assert config.activation_threshold_sigma == 3.0
        assert config.expand_adjacent is False
        assert config.expansion_radius == 2

    def test_frozen(self):
        """TokenLabelingConfig is immutable."""
        config = TokenLabelingConfig()
        with pytest.raises(AttributeError):
            config.min_active_latents = 5


class TestLatentActivationStats:
    """Tests for LatentActivationStats dataclass."""

    def test_fields_stored_correctly(self):
        """LatentActivationStats stores all fields."""
        backend = get_default_backend()
        mean = backend.array([0.1, 0.2, 0.3])
        std = backend.array([0.01, 0.02, 0.03])

        stats = LatentActivationStats(mean=mean, std=std, sample_count=100)
        assert stats.sample_count == 100
        backend.eval(stats.mean, stats.std)
        assert int(stats.mean.shape[0]) == 3
        assert int(stats.std.shape[0]) == 3


class TestTokenLabelResult:
    """Tests for TokenLabelResult dataclass."""

    def test_fields_stored_correctly(self):
        """TokenLabelResult stores all fields."""
        backend = get_default_backend()
        labels = backend.array([1, 0, 1, 0], dtype="int32")
        confidence = backend.array([0.9, 0.1, 0.8, 0.2])
        counts = backend.array([3, 1, 2, 0], dtype="int32")

        result = TokenLabelResult(
            labels=labels,
            confidence_scores=confidence,
            active_latent_counts=counts,
            text_lengths=[2, 2],
        )
        assert result.text_lengths == [2, 2]
        backend.eval(result.labels)
        assert int(result.labels.shape[0]) == 4


# =============================================================================
# SAETokenLabeler Tests
# =============================================================================


class TestSAETokenLabelerInit:
    """Tests for SAETokenLabeler initialization."""

    def test_default_config(self):
        """SAETokenLabeler uses default config if none provided."""
        labeler = SAETokenLabeler()
        assert labeler.config.min_active_latents == 2
        assert labeler.config.activation_threshold_sigma == 4.0

    def test_custom_config(self):
        """SAETokenLabeler accepts custom config."""
        config = TokenLabelingConfig(min_active_latents=5)
        labeler = SAETokenLabeler(config=config)
        assert labeler.config.min_active_latents == 5


class TestComputeActivationStats:
    """Tests for SAETokenLabeler.compute_activation_stats()."""

    def test_basic_stats(self):
        """compute_activation_stats computes correct mean and std."""
        backend = get_default_backend()
        labeler = SAETokenLabeler(backend=backend)

        # Create activations with known stats
        # 10 tokens, 4 latents
        activations = backend.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ])

        stats = labeler.compute_activation_stats(activations)

        backend.eval(stats.mean, stats.std)
        mean_list = [float(x) for x in backend.tolist(stats.mean)]
        std_list = [float(x) for x in backend.tolist(stats.std)]

        assert stats.sample_count == 10
        assert abs(mean_list[0] - 1.0) < 0.001
        assert abs(mean_list[1] - 2.0) < 0.001
        assert abs(std_list[0]) < 0.001  # Zero variance

    def test_empty_activations(self):
        """compute_activation_stats handles empty input."""
        backend = get_default_backend()
        labeler = SAETokenLabeler(backend=backend)

        activations = backend.zeros((0, 4))
        stats = labeler.compute_activation_stats(activations)

        assert stats.sample_count == 0


class TestLabelTokens:
    """Tests for SAETokenLabeler.label_tokens()."""

    def test_labels_high_activation_tokens(self):
        """label_tokens correctly identifies tokens with high latent activation."""
        backend = get_default_backend()
        config = TokenLabelingConfig(
            min_active_latents=2,
            activation_threshold_sigma=1.0,  # Lower threshold for testing
            expand_adjacent=False,
        )
        labeler = SAETokenLabeler(config=config, backend=backend)

        # Create activations where token 1 and 3 have high values on domain latents
        # 5 tokens, 6 latents (indices 0, 2, 4 are domain latents)
        activations = backend.array([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Low activation
            [5.0, 0.1, 5.0, 0.1, 5.0, 0.1],  # High on domain latents
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Low activation
            [5.0, 0.1, 5.0, 0.1, 0.1, 0.1],  # High on 2 domain latents
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Low activation
        ])

        stats = labeler.compute_activation_stats(activations)
        result = labeler.label_tokens(
            activations=activations,
            domain_latent_indices=[0, 2, 4],
            stats=stats,
            text_lengths=[5],
        )

        backend.eval(result.labels)
        labels_list = [int(x) for x in backend.tolist(result.labels)]

        # Token 1 and 3 should be labeled positive (high on 2+ domain latents)
        assert labels_list[0] == 0
        assert labels_list[1] == 1
        assert labels_list[2] == 0
        assert labels_list[3] == 1
        assert labels_list[4] == 0

    def test_no_domain_latents(self):
        """label_tokens handles empty domain latent list."""
        backend = get_default_backend()
        labeler = SAETokenLabeler(backend=backend)

        activations = backend.ones((10, 4))
        stats = labeler.compute_activation_stats(activations)
        result = labeler.label_tokens(
            activations=activations,
            domain_latent_indices=[],
            stats=stats,
            text_lengths=[10],
        )

        backend.eval(result.labels)
        labels_list = [int(x) for x in backend.tolist(result.labels)]
        assert all(l == 0 for l in labels_list)


class TestExpandLabels:
    """Tests for SAETokenLabeler._expand_labels()."""

    def test_expansion_within_text(self):
        """_expand_labels expands labels within text boundaries."""
        backend = get_default_backend()
        config = TokenLabelingConfig(
            expand_adjacent=True,
            expansion_radius=1,
        )
        labeler = SAETokenLabeler(config=config, backend=backend)

        # Label: 0 0 1 0 0
        labels = backend.array([0, 0, 1, 0, 0], dtype="int32")
        text_lengths = [5]

        expanded = labeler._expand_labels(labels, text_lengths)
        backend.eval(expanded)
        expanded_list = [int(x) for x in backend.tolist(expanded)]

        # Should expand to: 0 1 1 1 0
        assert expanded_list == [0, 1, 1, 1, 0]

    def test_expansion_respects_text_boundaries(self):
        """_expand_labels does not expand across text boundaries."""
        backend = get_default_backend()
        config = TokenLabelingConfig(
            expand_adjacent=True,
            expansion_radius=1,
        )
        labeler = SAETokenLabeler(config=config, backend=backend)

        # Two texts: [0 1] [0 0]
        # Text 1: token at index 1 is positive
        # Text 2: all negative
        labels = backend.array([0, 1, 0, 0], dtype="int32")
        text_lengths = [2, 2]

        expanded = labeler._expand_labels(labels, text_lengths)
        backend.eval(expanded)
        expanded_list = [int(x) for x in backend.tolist(expanded)]

        # Text 1: expand to [1 1]
        # Text 2: stays [0 0] (no expansion across boundary)
        assert expanded_list == [1, 1, 0, 0]


class TestCalibrateThreshold:
    """Tests for SAETokenLabeler.calibrate_threshold()."""

    def test_calibration_finds_threshold(self):
        """calibrate_threshold finds threshold for target positive rate."""
        backend = get_default_backend()
        config = TokenLabelingConfig(min_active_latents=1)
        labeler = SAETokenLabeler(config=config, backend=backend)

        # Create activations with gradient of values
        # Domain latent at index 0 has increasing values
        n_tokens = 100
        activations_list = [[i / 50.0, 0.0, 0.0, 0.0] for i in range(n_tokens)]
        activations = backend.array(activations_list)

        sigma = labeler.calibrate_threshold(
            activations=activations,
            domain_latent_indices=[0],
            target_positive_rate=0.2,  # 20% positive
        )

        # Verify sigma is reasonable (should be positive)
        assert sigma > 0.0

    def test_calibration_empty_input(self):
        """calibrate_threshold handles empty input."""
        backend = get_default_backend()
        labeler = SAETokenLabeler(backend=backend)

        activations = backend.zeros((0, 4))
        sigma = labeler.calibrate_threshold(
            activations=activations,
            domain_latent_indices=[0, 1],
            target_positive_rate=0.1,
        )

        # Should return default threshold
        assert sigma == labeler.config.activation_threshold_sigma


# =============================================================================
# TokenLabelingService Tests
# =============================================================================


class TestTokenLabelingService:
    """Tests for TokenLabelingService."""

    def test_run_labeling_basic(self):
        """run_labeling produces valid results."""
        from modelcypher.core.use_cases.token_labeling_service import TokenLabelingService

        backend = get_default_backend()
        service = TokenLabelingService(backend)

        # Create test activations
        activations = backend.array([
            [0.1, 0.2, 0.3],
            [5.0, 5.0, 0.3],  # High on domain latents
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ])

        config = TokenLabelingConfig(
            min_active_latents=2,
            activation_threshold_sigma=1.0,
            expand_adjacent=False,
        )

        summary, result = service.run_labeling(
            sae_activations=activations,
            domain_latent_indices=[0, 1],
            text_lengths=[2, 2],
            config=config,
        )

        assert summary.total_tokens == 4
        assert summary.texts_processed == 2
        assert 0.0 <= summary.positive_rate <= 1.0

    def test_calibrate_basic(self):
        """calibrate produces valid results."""
        from modelcypher.core.use_cases.token_labeling_service import TokenLabelingService

        backend = get_default_backend()
        service = TokenLabelingService(backend)

        # Create test activations
        n_tokens = 50
        activations = backend.random_normal(shape=(n_tokens, 10))

        result = service.calibrate(
            sae_activations=activations,
            domain_latent_indices=[0, 1, 2],
            target_positive_rate=0.1,
        )

        assert result.calibrated_sigma > 0.0
        assert result.target_positive_rate == 0.1
        assert result.sample_count == n_tokens
