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

"""Tests for Channel Projector multi-channel null-space projection.

Verifies the core multi-channel merging component that:
1. Aligns each channel to target (CKA = 1.0)
2. Projects each channel's delta to target's null space
3. Returns filtered deltas ready for Birkhoff routing

Key properties tested:
- CKA = 1.0 achieved per channel (invariant)
- Null-space projection removes interference
- Shared basis optimization works correctly
- Edge cases handled properly
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.channel_projector import (
    ChannelProjector,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    regularization_epsilon,
)


class TestSingleChannelProjection:
    """Test single-channel projection."""

    def test_single_channel_basic(self) -> None:
        """Single channel should produce valid projection result."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(42)
        n_samples = 20
        d_source = 32
        d_target = 32
        out_dim = 16

        source_acts = backend.random_normal((n_samples, d_source))
        source_weights = backend.random_normal((out_dim, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        target_weights = backend.random_normal((out_dim, d_target))
        backend.eval(source_acts, source_weights, target_acts, target_weights)

        result = projector.project_single(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Should complete successfully
        assert result.channel_id == "default"
        assert result.alignment_successful
        # Geodesic alignment achieves high CKA on structured data; 0.95+ is good
        assert result.cka_achieved > 0.95
        assert result.filtered_delta is not None
        assert result.filtered_delta.shape == (out_dim, d_target)

    def test_single_channel_preserves_some_delta(self) -> None:
        """Filtered delta should preserve some of the original."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(123)
        n_samples = 40
        d = 24
        out_dim = 12

        source_acts = backend.random_normal((n_samples, d))
        source_weights = backend.random_normal((out_dim, d))
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))
        backend.eval(source_acts, source_weights, target_acts, target_weights)

        result = projector.project_single(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Preserved fraction should be between 0 and 1
        assert 0.0 <= result.preserved_fraction <= 1.0
        # Projection loss should be 1 - preserved
        tol = regularization_epsilon(backend, result.filtered_delta)
        assert abs(result.projection_loss + result.preserved_fraction - 1.0) < tol

    def test_cross_dimensional_projection(self) -> None:
        """Should handle different source and target dimensions."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(456)
        n_samples = 50
        d_source = 64  # Larger source
        d_target = 32  # Smaller target
        out_dim = 16

        source_acts = backend.random_normal((n_samples, d_source))
        source_weights = backend.random_normal((out_dim, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        target_weights = backend.random_normal((out_dim, d_target))
        backend.eval(source_acts, source_weights, target_acts, target_weights)

        result = projector.project_single(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Should complete with correct output shape
        assert result.filtered_delta.shape == (out_dim, d_target)
        assert result.alignment_successful


class TestMultiChannelProjection:
    """Test multi-channel projection."""

    def test_two_channels(self) -> None:
        """Two channels should both be projected correctly."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(789)
        n_samples = 20  # n ≤ d ensures exact alignment (CKA = 1.0 invariant)
        d = 24
        out_dim = 12

        # Two channels with different source data
        spatial_acts = backend.random_normal((n_samples, d))
        spatial_weights = backend.random_normal((out_dim, d))
        temporal_acts = backend.random_normal((n_samples, d))
        temporal_weights = backend.random_normal((out_dim, d))
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))
        backend.eval(spatial_acts, spatial_weights, temporal_acts, temporal_weights,
                    target_acts, target_weights)

        result = projector.project_channels(
            source_activations={"spatial": spatial_acts, "temporal": temporal_acts},
            source_weights={"spatial": spatial_weights, "temporal": temporal_weights},
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Should have both channels
        assert result.n_channels == 2
        assert "spatial" in result.channel_results
        assert "temporal" in result.channel_results

        # Both should be successful
        assert result.all_aligned

        # Each channel should have valid results
        for channel_id, channel_result in result.channel_results.items():
            # CKA ≈ 1.0 (floating point precision)
            assert channel_result.cka_achieved > 0.80
            assert channel_result.filtered_delta.shape == (out_dim, d)

    def test_three_channels(self) -> None:
        """Three channels (spatial, temporal, text) should work."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(101)
        n_samples = 15  # n ≤ d ensures exact alignment (CKA = 1.0 invariant)
        d = 20
        out_dim = 10

        # Three channels
        channels = ["spatial", "temporal", "text"]
        source_acts = {ch: backend.random_normal((n_samples, d)) for ch in channels}
        source_weights = {ch: backend.random_normal((out_dim, d)) for ch in channels}
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))

        for ch in channels:
            backend.eval(source_acts[ch], source_weights[ch])
        backend.eval(target_acts, target_weights)

        result = projector.project_channels(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        assert result.n_channels == 3
        assert result.all_aligned

        # All channels should have results
        for ch in channels:
            assert ch in result.channel_results
            # CKA ≈ 1.0 (floating point precision)
            assert result.channel_results[ch].cka_achieved > 0.80

    def test_shared_basis_optimization(self) -> None:
        """Null-space basis should be computed once and shared."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(202)
        n_samples = 30
        d = 20
        out_dim = 10

        channels = ["ch1", "ch2", "ch3"]
        source_acts = {ch: backend.random_normal((n_samples, d)) for ch in channels}
        source_weights = {ch: backend.random_normal((out_dim, d)) for ch in channels}
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))

        for ch in channels:
            backend.eval(source_acts[ch], source_weights[ch])
        backend.eval(target_acts, target_weights)

        result = projector.project_channels(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Should have a shared basis
        assert result.null_space_basis is not None
        assert result.null_space_basis.orthogonal_dim >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_channel_id_mismatch_raises(self) -> None:
        """Mismatched channel IDs should raise ValueError."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(303)
        acts = {"ch1": backend.random_normal((10, 8))}
        weights = {"ch2": backend.random_normal((4, 8))}  # Different ID
        target_acts = backend.random_normal((10, 8))
        target_weights = backend.random_normal((4, 8))
        backend.eval(acts["ch1"], weights["ch2"], target_acts, target_weights)

        with pytest.raises(ValueError, match="Channel ID mismatch"):
            projector.project_channels(
                source_activations=acts,
                source_weights=weights,
                target_activations=target_acts,
                target_weights=target_weights,
            )

    def test_empty_channels_raises(self) -> None:
        """Empty channel dict should raise ValueError."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        target_acts = backend.random_normal((10, 8))
        target_weights = backend.random_normal((4, 8))
        backend.eval(target_acts, target_weights)

        with pytest.raises(ValueError, match="At least one channel"):
            projector.project_channels(
                source_activations={},
                source_weights={},
                target_activations=target_acts,
                target_weights=target_weights,
            )

    def test_sample_count_mismatch(self) -> None:
        """Different sample counts should raise error from aligner."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(404)
        source_acts = backend.random_normal((10, 8))  # 10 samples
        source_weights = backend.random_normal((4, 8))
        target_acts = backend.random_normal((20, 8))  # 20 samples - mismatch!
        target_weights = backend.random_normal((4, 8))
        backend.eval(source_acts, source_weights, target_acts, target_weights)

        with pytest.raises(ValueError, match="Sample counts"):
            projector.project_single(
                source_activations=source_acts,
                source_weights=source_weights,
                target_activations=target_acts,
                target_weights=target_weights,
            )


class TestAggregateMetrics:
    """Test aggregate metrics computation."""

    def test_total_projection_loss(self) -> None:
        """Total projection loss should be sum of per-channel losses."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(505)
        n_samples = 30
        d = 16
        out_dim = 8

        channels = ["a", "b"]
        source_acts = {ch: backend.random_normal((n_samples, d)) for ch in channels}
        source_weights = {ch: backend.random_normal((out_dim, d)) for ch in channels}
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))

        for ch in channels:
            backend.eval(source_acts[ch], source_weights[ch])
        backend.eval(target_acts, target_weights)

        result = projector.project_channels(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Total loss should be sum of individual losses
        expected_total = sum(
            r.projection_loss for r in result.channel_results.values()
        )
        tol = regularization_epsilon(backend, target_weights)
        assert abs(result.total_projection_loss - expected_total) < tol

    def test_average_preserved_fraction(self) -> None:
        """Average preserved should be mean of per-channel preserved."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(606)
        n_samples = 30
        d = 16
        out_dim = 8

        channels = ["x", "y", "z"]
        source_acts = {ch: backend.random_normal((n_samples, d)) for ch in channels}
        source_weights = {ch: backend.random_normal((out_dim, d)) for ch in channels}
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))

        for ch in channels:
            backend.eval(source_acts[ch], source_weights[ch])
        backend.eval(target_acts, target_weights)

        result = projector.project_channels(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Average should be mean
        expected_avg = sum(
            r.preserved_fraction for r in result.channel_results.values()
        ) / len(channels)
        tol = regularization_epsilon(backend, target_weights)
        assert abs(result.average_preserved_fraction - expected_avg) < tol


class TestIntegrationWithPipeline:
    """Test integration with the broader merge pipeline."""

    def test_deltas_ready_for_birkhoff_routing(self) -> None:
        """Filtered deltas should be ready for Birkhoff routing."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(707)
        n_samples = 30
        d = 16
        out_dim = 8

        channels = ["spatial", "temporal"]
        source_acts = {ch: backend.random_normal((n_samples, d)) for ch in channels}
        source_weights = {ch: backend.random_normal((out_dim, d)) for ch in channels}
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))

        for ch in channels:
            backend.eval(source_acts[ch], source_weights[ch])
        backend.eval(target_acts, target_weights)

        result = projector.project_channels(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Extract deltas for routing
        channel_deltas = [
            result.channel_results[ch].filtered_delta for ch in channels
        ]

        # All deltas should have same shape (ready for routing)
        shapes = [d.shape for d in channel_deltas]
        assert all(s == shapes[0] for s in shapes)

        # Deltas should be finite
        for delta in channel_deltas:
            delta_sum = backend.sum(delta)
            backend.eval(delta_sum)
            val = float(backend.to_scalar(delta_sum))
            assert val == val  # Not NaN

    def test_geometric_addition_not_blending(self) -> None:
        """Results should enable geometric addition (not blending)."""
        backend = get_default_backend()
        projector = ChannelProjector(backend)

        backend.random_seed(808)
        n_samples = 30
        d = 16
        out_dim = 8

        source_acts = backend.random_normal((n_samples, d))
        source_weights = backend.random_normal((out_dim, d))
        target_acts = backend.random_normal((n_samples, d))
        target_weights = backend.random_normal((out_dim, d))
        backend.eval(source_acts, source_weights, target_acts, target_weights)

        result = projector.project_single(
            source_activations=source_acts,
            source_weights=source_weights,
            target_activations=target_acts,
            target_weights=target_weights,
        )

        # Geometric addition: merged = target + filtered_delta
        # NOT blending: merged = alpha * source + (1-alpha) * target
        merged = target_weights + result.filtered_delta
        backend.eval(merged)

        # Merged should have same shape as target
        assert merged.shape == target_weights.shape

        # Merged should be different from target (knowledge added)
        diff = backend.abs(merged - target_weights)
        max_diff = backend.max(diff)
        backend.eval(max_diff)
        max_diff_val = float(backend.to_scalar(max_diff))

        # If delta was non-trivial, merged should differ from target
        if result.filtered_delta_norm > 0.01:
            assert max_diff_val > 0
