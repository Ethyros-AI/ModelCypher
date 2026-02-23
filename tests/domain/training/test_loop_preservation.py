# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.loop_preservation import (
    LoopPreservationConfig,
    compute_entropy_delta,
    compute_spectral_entropy,
    derive_loop_config_from_geometry,
    find_highway_layer_from_intrinsic_dims,
    loop_preservation_loss,
    select_layers_to_sample,
)


class TestFindHighwayLayer:
    def test_24_layers_known_minimum(self):
        """24-layer model with known minimum → correct index returned."""
        # Layer 10 has the minimum intrinsic dim in search range
        dims = [10.0] * 24
        dims[10] = 2.0  # Highway at layer 10
        result = find_highway_layer_from_intrinsic_dims(dims, n_layers=24)
        # skip_early=4, skip_late=4 → search [4,20), layer 10 is min
        assert result == 10

    def test_empty_dims(self):
        """Empty list → returns 0."""
        assert find_highway_layer_from_intrinsic_dims([], n_layers=0) == 0

    def test_small_model_uses_middle(self):
        """Very small model (3 layers) → search range collapses, use n//2."""
        # skip_early=1, skip_late=1 → search [1,2), only layer 1
        dims = [5.0, 3.0, 7.0]
        result = find_highway_layer_from_intrinsic_dims(dims, n_layers=3)
        assert result == 1

    def test_skips_early_and_late(self):
        """Minimum in early/late layers is ignored."""
        dims = [0.1] + [10.0] * 22 + [0.1]  # 24 layers, min at 0 and 23
        dims[12] = 5.0  # Minimum in valid search range
        result = find_highway_layer_from_intrinsic_dims(dims, n_layers=24)
        assert result == 12


class TestComputeEntropyDelta:
    def test_normal_delta(self):
        """highway=1.0, exit=2.0 → delta=1.0."""
        delta = compute_entropy_delta([1.0], [2.0])
        assert delta == pytest.approx(1.0)

    def test_empty_lists(self):
        """Empty inputs → 0.0."""
        assert compute_entropy_delta([], []) == 0.0
        assert compute_entropy_delta([1.0], []) == 0.0
        assert compute_entropy_delta([], [1.0]) == 0.0

    def test_multiple_samples(self):
        """Mean of multiple samples."""
        delta = compute_entropy_delta([1.0, 2.0], [3.0, 4.0])
        # mean_exit=3.5, mean_highway=1.5, delta=2.0
        assert delta == pytest.approx(2.0)


class TestLoopPreservationLoss:
    def test_improvement_zero_loss(self):
        """Current delta > base delta → loss = 0."""
        config = LoopPreservationConfig(
            highway_layer=4, base_delta_entropy=0.5, lambda_scale=0.1,
        )
        trajectory = {4: 1.0, 11: 2.0}  # current_delta = 2.0 - 1.0 = 1.0 > 0.5
        loss, current_delta = loop_preservation_loss(trajectory, config)
        assert loss == pytest.approx(0.0)
        assert current_delta == pytest.approx(1.0)

    def test_degradation_positive_loss(self):
        """Current delta < base delta → loss = λ * (base - current)."""
        config = LoopPreservationConfig(
            highway_layer=4, base_delta_entropy=0.5, lambda_scale=0.1,
        )
        trajectory = {4: 1.0, 11: 1.1}  # current_delta = 0.1 < 0.5
        loss, current_delta = loop_preservation_loss(trajectory, config)
        # gap = 0.5 - 0.1 = 0.4, loss = 0.1 * 0.4 = 0.04
        assert loss == pytest.approx(0.04)
        assert current_delta == pytest.approx(0.1)

    def test_fewer_than_two_layers(self):
        """Single-layer trajectory → (0.0, 0.0)."""
        config = LoopPreservationConfig(
            highway_layer=4, base_delta_entropy=0.5, lambda_scale=0.1,
        )
        loss, delta = loop_preservation_loss({4: 1.0}, config)
        assert loss == 0.0
        assert delta == 0.0

    def test_empty_trajectory(self):
        """Empty trajectory → (0.0, 0.0)."""
        config = LoopPreservationConfig(
            highway_layer=4, base_delta_entropy=0.5, lambda_scale=0.1,
        )
        loss, delta = loop_preservation_loss({}, config)
        assert loss == 0.0
        assert delta == 0.0


class TestDeriveLoopConfig:
    def test_sigma_max_10(self):
        config = derive_loop_config_from_geometry(
            highway_layer=8, base_delta_entropy=0.5, sigma_max=10.0,
        )
        assert config.highway_layer == 8
        assert config.base_delta_entropy == 0.5
        assert config.lambda_scale == pytest.approx(0.1)

    def test_sigma_max_near_zero(self):
        import math
        # With weight_shape=(1024, 1024), SVD floor = sqrt(1024) * eps_f32
        config = derive_loop_config_from_geometry(
            highway_layer=4, base_delta_entropy=0.3, sigma_max=0.0,
            weight_shape=(1024, 1024),
        )
        eps_f32 = math.ldexp(1.0, -23)
        svd_floor = math.sqrt(1024) * eps_f32
        assert config.lambda_scale == pytest.approx(1.0 / svd_floor)


class TestSelectLayersToSample:
    def test_12_layers(self):
        """12 layers → highway=4, middle=6, exit=11."""
        layers = select_layers_to_sample(12)
        assert layers == [4, 6, 11]

    def test_zero_layers(self):
        assert select_layers_to_sample(0) == []

    def test_one_layer(self):
        # n=1 → highway=0, middle=0, exit=0 → [0]
        assert select_layers_to_sample(1) == [0]


class TestComputeSpectralEntropy:
    def test_rank_one_matrix(self, any_backend):
        """Rank-1 matrix → single dominant SV → low entropy."""
        b = any_backend
        # Outer product: rank-1
        v = b.array([[1.0, 0.0, 0.0, 0.0]] * 10)
        b.eval(v)
        H = compute_spectral_entropy(v, b)
        assert H == pytest.approx(0.0, abs=0.01)

    def test_full_rank_higher_entropy(self, any_backend):
        """Identity-like matrix → uniform SVs → higher entropy."""
        b = any_backend
        mat = b.eye(8)
        b.eval(mat)
        H = compute_spectral_entropy(mat, b)
        assert H > 1.0  # Entropy of uniform distribution over 8 values

    def test_too_small_returns_zero(self, any_backend):
        """Single row → returns 0.0."""
        b = any_backend
        v = b.array([[1.0, 2.0, 3.0]])
        b.eval(v)
        H = compute_spectral_entropy(v, b)
        assert H == 0.0
