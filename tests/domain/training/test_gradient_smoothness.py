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

from modelcypher.core.domain.training.gradient_smoothness_estimator import (
    GradientSmoothnessEstimator,
    LayerGradientQuality,
)


class TestGradientQuality:
    """Tests for GradientSmoothnessEstimator.gradient_quality()."""

    def test_single_sample_returns_none(self, any_backend):
        """1 sample → None (need ≥ 2 for variance)."""
        b = any_backend
        grads = [{"w": b.array([1.0, 2.0, 3.0])}]
        b.eval(grads[0]["w"])
        result = GradientSmoothnessEstimator.gradient_quality(grads, b)
        assert result is None

    def test_constant_grads_low_variance(self, any_backend):
        """3 identical gradient samples → variance ≈ 0, high SNR."""
        b = any_backend
        g = b.array([1.0, 2.0, 3.0])
        b.eval(g)
        grads = [{"w": g}, {"w": g}, {"w": g}]

        result = GradientSmoothnessEstimator.gradient_quality(grads, b)
        assert result is not None
        assert result.variance == pytest.approx(0.0, abs=1e-5)
        assert result.snr > 1e6  # Very high SNR

    def test_zero_grads(self, any_backend):
        """All-zero gradients → mean_norm = 0."""
        b = any_backend
        z = b.array([0.0, 0.0, 0.0])
        b.eval(z)
        grads = [{"w": z}, {"w": z}]

        result = GradientSmoothnessEstimator.gradient_quality(grads, b)
        assert result is not None
        assert result.mean_norm == pytest.approx(0.0, abs=1e-5)

    def test_known_variance(self, any_backend):
        """2 samples with known difference → predictable variance."""
        b = any_backend
        g1 = b.array([1.0, 0.0])
        g2 = b.array([0.0, 1.0])
        b.eval(g1, g2)
        grads = [{"w": g1}, {"w": g2}]

        result = GradientSmoothnessEstimator.gradient_quality(grads, b)
        assert result is not None
        assert result.sample_count == 2
        assert result.variance > 0.0


class TestPerLayerQuality:
    """Tests for GradientSmoothnessEstimator.per_layer_quality()."""

    def test_layers_extracted(self, any_backend):
        """Keys like .layers.0.weight → dict keyed by layer index."""
        b = any_backend
        g = b.array([1.0, 2.0])
        b.eval(g)
        grads = [
            {"model.layers.0.weight": g, "model.layers.1.weight": g},
            {"model.layers.0.weight": g, "model.layers.1.weight": g},
        ]

        result = GradientSmoothnessEstimator.per_layer_quality(grads, b)
        assert 0 in result
        assert 1 in result

    def test_no_layer_pattern_empty(self, any_backend):
        """Keys without layer patterns → empty dict."""
        b = any_backend
        g = b.array([1.0])
        b.eval(g)
        grads = [{"embed.weight": g}, {"embed.weight": g}]

        result = GradientSmoothnessEstimator.per_layer_quality(grads, b)
        assert result == {}
