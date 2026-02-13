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

import modelcypher.core.domain.training.hessian_estimator as hessian_mod
from modelcypher.core.domain.training.hessian_estimator import (
    Config,
    GradientQualityMetrics,
    PerLayerStats,
    TrajectoryMetrics,
    condition_proxy,
    effective_step_ratio,
    gradient_quality,
    per_layer_analysis,
    trajectory,
)


class TestConfig:
    """Tests for Config presets — verify they control estimation behavior."""

    def test_moderate_fewer_iterations_than_full(self):
        """moderate() uses less computation than full()."""
        moderate = Config.moderate()
        full = Config.full()
        assert moderate.hutchinson_vectors < full.hutchinson_vectors
        assert moderate.power_iterations < full.power_iterations

    def test_zero_vectors_disables_hutchinson(self, monkeypatch, any_backend):
        """hutchinson_vectors=0 → hutchinson_trace_estimate returns None."""
        from modelcypher.core.domain.training.hessian_estimator import hutchinson_trace_estimate
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        params = {"w": b.array([1.0, 2.0])}
        b.eval(params["w"])

        cfg = Config(hutchinson_vectors=0, power_iterations=0)
        result = hutchinson_trace_estimate(lambda p: (b.array(0.0), p), params, cfg)
        assert result is None

    def test_frozen_prevents_mutation(self):
        """Config is frozen — cannot be accidentally mutated during training."""
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.hutchinson_vectors = 99


class TestGradientQuality:
    """Tests for gradient_quality() top-level function."""

    def test_single_sample_returns_none(self, monkeypatch, any_backend):
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        grads = [{"w": b.array([1.0, 2.0])}]
        b.eval(grads[0]["w"])

        result = gradient_quality(grads)
        assert result is None

    def test_two_samples_returns_metrics(self, monkeypatch, any_backend):
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        g1 = b.array([1.0, 0.0])
        g2 = b.array([0.0, 1.0])
        b.eval(g1, g2)

        result = gradient_quality([{"w": g1}, {"w": g2}])
        assert result is not None
        assert isinstance(result, GradientQualityMetrics)
        assert result.variance > 0
        assert result.mean_norm > 0


class TestPerLayerAnalysis:
    """Tests for per_layer_analysis()."""

    def test_norm_fractions_computed(self, monkeypatch, any_backend):
        """Fractions are norm_i / total_L2_norm."""
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        grads = {
            "layer_a": b.array([3.0, 0.0]),
            "layer_b": b.array([0.0, 4.0]),
        }
        b.eval(grads["layer_a"], grads["layer_b"])

        stats = per_layer_analysis(grads)
        # total_norm = sqrt(3^2 + 4^2) = 5
        # frac_a = 3/5 = 0.6, frac_b = 4/5 = 0.8
        assert stats.fractions["layer_a"] == pytest.approx(0.6, abs=0.05)
        assert stats.fractions["layer_b"] == pytest.approx(0.8, abs=0.05)

    def test_active_layers(self, monkeypatch, any_backend):
        """Param with larger norm should be in active_layers."""
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        grads = {
            "big": b.array([10.0, 10.0]),
            "small": b.array([0.001, 0.001]),
        }
        b.eval(grads["big"], grads["small"])

        stats = per_layer_analysis(grads)
        assert "big" in stats.active_layers


class TestTrajectory:
    """Tests for trajectory()."""

    def test_identical_params_cosine_one(self, monkeypatch, any_backend):
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        params = {"w": b.array([1.0, 2.0, 3.0])}
        b.eval(params["w"])

        result = trajectory(params, params)
        assert result is not None
        assert result.cosine_similarity == pytest.approx(1.0, abs=1e-4)
        assert result.divergence == pytest.approx(0.0, abs=1e-5)

    def test_divergence_nonzero(self, monkeypatch, any_backend):
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        initial = {"w": b.array([0.0, 0.0])}
        current = {"w": b.array([3.0, 4.0])}
        b.eval(initial["w"], current["w"])

        result = trajectory(current, initial)
        assert result is not None
        assert result.divergence == pytest.approx(5.0, abs=0.1)

    def test_empty_returns_none(self, monkeypatch, any_backend):
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        assert trajectory({}, {"w": any_backend.array([1.0])}) is None
        assert trajectory({"w": any_backend.array([1.0])}, {}) is None


class TestEffectiveStepRatio:
    """Tests for effective_step_ratio()."""

    def test_exact_match(self, monkeypatch, any_backend):
        """actual = lr * grad → ratio ≈ 1.0."""
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        grad = {"w": b.array([1.0, 2.0])}
        lr = 0.1
        actual = {"w": b.array([0.1, 0.2])}
        b.eval(grad["w"], actual["w"])

        ratio = effective_step_ratio(actual, grad, lr)
        assert ratio is not None
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_half_step(self, monkeypatch, any_backend):
        """actual = 0.5 * lr * grad → ratio ≈ 0.5."""
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        b = any_backend
        grad = {"w": b.array([1.0, 0.0])}
        lr = 0.1
        actual = {"w": b.array([0.05, 0.0])}
        b.eval(grad["w"], actual["w"])

        ratio = effective_step_ratio(actual, grad, lr)
        assert ratio is not None
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_invalid_returns_none(self, monkeypatch, any_backend):
        monkeypatch.setattr(hessian_mod, "get_default_backend", lambda: any_backend)
        assert effective_step_ratio({}, {"w": any_backend.array([1.0])}, 0.1) is None
        assert effective_step_ratio({"w": any_backend.array([1.0])}, {}, 0.1) is None
        b = any_backend
        g = b.array([1.0])
        b.eval(g)
        assert effective_step_ratio({"w": g}, {"w": g}, 0.0) is None


class TestConditionProxy:
    """Tests for condition_proxy()."""

    def test_normal(self):
        # top_eigenvalue=10, trace=100, params=10 → avg=10, proxy=1.0
        result = condition_proxy(10.0, 100.0, 10)
        assert result == pytest.approx(1.0)

    def test_zero_trace_returns_none(self):
        assert condition_proxy(10.0, 0.0, 10) is None

    def test_zero_params_returns_none(self):
        assert condition_proxy(10.0, 100.0, 0) is None

    def test_negative_avg_returns_none(self):
        # trace=-100, params=10 → avg=-10 → None
        assert condition_proxy(10.0, -100.0, 10) is None
