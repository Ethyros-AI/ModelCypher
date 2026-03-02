# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for logit perturbation bound derivation.

Verifies the mathematical chain:
    ||ΔW||₂ → ||Δh_L||₂ → ||Δlogits||_∞ ≤ bound

Uses synthetic weight matrices with known singular values to verify
tightness and correctness of the bound.
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.perturbation_bound import (
    LogitPerturbationBound,
    MarginSafetyResult,
    check_margin_safety,
    compute_logit_perturbation_bound,
    compute_readout_effective_rank,
)


def _make_mock_model(n_layers: int):
    """Create a minimal mock model with n_layers."""

    class MockLayer:
        pass

    class MockBase:
        def __init__(self, n):
            self.layers = [MockLayer() for _ in range(n)]

    class MockModel:
        def __init__(self, n):
            self.model = MockBase(n)

    return MockModel(n_layers)


class TestCheckMarginSafety:
    """Tests for the margin safety check — the decision criterion."""

    def test_safe_when_bound_below_margin(self):
        result = check_margin_safety(logit_bound=0.5, min_margin=2.0)
        assert result.safe is True
        assert result.safety_ratio == pytest.approx(4.0)

    def test_unsafe_when_bound_above_margin(self):
        result = check_margin_safety(logit_bound=3.0, min_margin=2.0)
        assert result.safe is False
        assert result.safety_ratio == pytest.approx(2.0 / 3.0)

    def test_unsafe_when_bound_equals_margin(self):
        """Equal is NOT safe — need strict inequality."""
        result = check_margin_safety(logit_bound=2.0, min_margin=2.0)
        assert result.safe is False

    def test_unsafe_when_margin_zero(self):
        """Zero margin = tie at baseline → cannot guarantee safety."""
        result = check_margin_safety(logit_bound=0.001, min_margin=0.0)
        assert result.safe is False
        assert result.safety_ratio == 0.0

    def test_unsafe_when_margin_negative(self):
        """Negative margin = baseline already wrong."""
        result = check_margin_safety(logit_bound=0.001, min_margin=-1.0)
        assert result.safe is False

    def test_safe_with_zero_bound(self):
        """Zero perturbation → safe if margin positive."""
        result = check_margin_safety(logit_bound=0.0, min_margin=1.0)
        assert result.safe is True
        assert result.safety_ratio == math.inf

    def test_result_fields(self):
        result = check_margin_safety(logit_bound=1.5, min_margin=3.0)
        assert isinstance(result, MarginSafetyResult)
        assert result.logit_bound == 1.5
        assert result.min_margin == 3.0
        assert result.safety_ratio == pytest.approx(2.0)


class TestComputeLogitPerturbationBound:
    """Tests for the bound computation with synthetic inputs."""

    def test_no_perturbation_returns_zero(self, any_backend):
        """No perturbed layers → bound = 0."""
        model = _make_mock_model(4)
        lipschitz = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        result = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={},
            layer_lipschitz=lipschitz,
            sigma_max_readout=10.0,
        )

        assert result.bound == 0.0
        assert result.n_perturbed_layers == 0

    def test_single_layer_perturbation(self, any_backend):
        """Single perturbed layer at the end → minimal propagation."""
        model = _make_mock_model(4)
        # All layers have Lipschitz = 2.0
        lipschitz = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}

        # Perturb layer 3 (last) with ||scale*BA||₂ = 0.1
        result = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={3: 0.1},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )

        # Propagation from layer 3: (1+2.0) = 3.0
        # Injection: 0.1 * 1.0 (default activation norm)
        # Bound: 5.0 * 3.0 * 0.1 = 1.5
        assert result.bound == pytest.approx(1.5)
        assert result.n_perturbed_layers == 1

    def test_early_layer_perturbation_amplifies_more(self, any_backend):
        """Perturbation at early layer propagates through more layers."""
        model = _make_mock_model(4)
        lipschitz = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}

        # Perturb layer 0 (first) with ||scale*BA||₂ = 0.1
        result = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={0: 0.1},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )

        # Propagation from layer 0: (1+2)^4 = 81.0
        # Injection: 0.1 * 1.0
        # Bound: 5.0 * 81.0 * 0.1 = 40.5
        assert result.bound == pytest.approx(40.5)

    def test_multiple_perturbed_layers_sum(self, any_backend):
        """Multiple perturbed layers contribute additively (triangle inequality)."""
        model = _make_mock_model(4)
        lipschitz = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}  # L_i=0 → prop=1.0

        # Perturb layers 1 and 3
        result = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={1: 0.2, 3: 0.3},
            layer_lipschitz=lipschitz,
            sigma_max_readout=1.0,
        )

        # L_i = 0 means propagation = 1.0 for all layers
        # Total: 1.0 * (0.2 + 0.3) = 0.5
        assert result.bound == pytest.approx(0.5)

    def test_activation_norms_scale_injection(self, any_backend):
        """Activation norms scale the injection at each perturbed layer."""
        model = _make_mock_model(4)
        lipschitz = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        result = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={2: 0.1},
            layer_lipschitz=lipschitz,
            sigma_max_readout=1.0,
            activation_norms={2: 3.0},
        )

        # Injection: 0.1 * 3.0 = 0.3
        # Propagation: 1.0 (all L_i = 0)
        # Bound: 1.0 * 0.3 = 0.3
        assert result.bound == pytest.approx(0.3)

    def test_bound_is_always_non_negative(self, any_backend):
        """Bound must be >= 0 regardless of inputs."""
        model = _make_mock_model(2)
        result = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={0: 0.0},
            layer_lipschitz={0: 0.0, 1: 0.0},
            sigma_max_readout=1.0,
        )
        assert result.bound >= 0.0

    def test_result_diagnostics(self, any_backend):
        """Result contains per-layer diagnostic breakdown."""
        model = _make_mock_model(3)
        lipschitz = {0: 1.0, 1: 1.0, 2: 1.0}

        result = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={0: 0.5, 2: 0.3},
            layer_lipschitz=lipschitz,
            sigma_max_readout=2.0,
        )

        assert isinstance(result, LogitPerturbationBound)
        assert result.n_perturbed_layers == 2
        assert 0 in result.per_layer_injection_norm
        assert 2 in result.per_layer_injection_norm
        assert 0 in result.per_layer_propagation_factor
        assert 2 in result.per_layer_propagation_factor
        assert result.sigma_max_readout == 2.0


class TestComputeReadoutEffectiveRank:
    """Tests for Shannon effective rank of the readout weight matrix."""

    def test_rank_one_readout(self, any_backend):
        """Rank-1 readout matrix has effective rank ~ 1."""

        class MockWeight:
            def __init__(self, w):
                self.weight = w

        class MockModel:
            def __init__(self, w):
                self.lm_head = MockWeight(w)

        # Rank-1 matrix: outer product of two vectors
        u = any_backend.reshape(any_backend.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
        v = any_backend.reshape(any_backend.array([1.0, 0.0, 0.0, 0.0]), (1, 4))
        w = any_backend.matmul(u, v)
        any_backend.eval(w)

        model = MockModel(w)
        erank = compute_readout_effective_rank(model, any_backend)
        assert erank == pytest.approx(1.0, abs=0.1)

    def test_full_rank_readout(self, any_backend):
        """Identity matrix has effective rank = dimension."""

        class MockWeight:
            def __init__(self, w):
                self.weight = w

        class MockModel:
            def __init__(self, w):
                self.lm_head = MockWeight(w)

        # Identity: all singular values equal → max entropy → erank = d
        d = 8
        w = any_backend.array(
            [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        )
        any_backend.eval(w)

        model = MockModel(w)
        erank = compute_readout_effective_rank(model, any_backend)
        assert erank == pytest.approx(float(d), abs=0.5)

    def test_embed_tokens_fallback(self, any_backend):
        """Falls back to embed_tokens when lm_head is absent."""

        class MockWeight:
            def __init__(self, w):
                self.weight = w

        class MockBase:
            def __init__(self, w):
                self.embed_tokens = MockWeight(w)

        class MockModel:
            def __init__(self, w):
                self.model = MockBase(w)

        d = 4
        w = any_backend.array(
            [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        )
        any_backend.eval(w)

        model = MockModel(w)
        erank = compute_readout_effective_rank(model, any_backend)
        assert erank == pytest.approx(float(d), abs=0.5)

    def test_erank_between_one_and_dim(self, any_backend):
        """Effective rank is always in [1, min(rows, cols)]."""

        class MockWeight:
            def __init__(self, w):
                self.weight = w

        class MockModel:
            def __init__(self, w):
                self.lm_head = MockWeight(w)

        # Random-ish matrix with decaying spectrum
        rows, cols = 6, 4
        data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(float((i * 7 + j * 3 + 1) % 11) / 10.0)
            data.append(row)
        w = any_backend.array(data)
        any_backend.eval(w)

        model = MockModel(w)
        erank = compute_readout_effective_rank(model, any_backend)
        assert 1.0 <= erank <= float(min(rows, cols))


class TestBoundMathProperties:
    """Mathematical properties the bound must satisfy."""

    def test_monotone_in_perturbation_norm(self, any_backend):
        """Larger perturbation → larger bound."""
        model = _make_mock_model(4)
        lipschitz = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        small = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={2: 0.01},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )
        large = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={2: 1.0},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )

        assert large.bound > small.bound

    def test_monotone_in_readout_norm(self, any_backend):
        """Larger readout σ_max → larger bound."""
        model = _make_mock_model(4)
        lipschitz = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        small_readout = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={2: 0.1},
            layer_lipschitz=lipschitz,
            sigma_max_readout=1.0,
        )
        large_readout = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={2: 0.1},
            layer_lipschitz=lipschitz,
            sigma_max_readout=100.0,
        )

        assert large_readout.bound > small_readout.bound

    def test_monotone_in_layer_depth(self, any_backend):
        """Earlier perturbation → larger bound (more propagation)."""
        model = _make_mock_model(8)
        lipschitz = {i: 1.0 for i in range(8)}

        early = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={0: 0.1},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )
        late = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={7: 0.1},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )

        assert early.bound > late.bound

    def test_subadditivity(self, any_backend):
        """Bound of combined perturbation ≤ sum of individual bounds.

        Triangle inequality: ||Δ1 + Δ2|| ≤ ||Δ1|| + ||Δ2||.
        Our bound computes the sum directly, so equality should hold.
        """
        model = _make_mock_model(4)
        lipschitz = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        b1 = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={1: 0.2},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )
        b2 = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={3: 0.3},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )
        combined = compute_logit_perturbation_bound(
            model=model,
            backend=any_backend,
            perturbed_layers={1: 0.2, 3: 0.3},
            layer_lipschitz=lipschitz,
            sigma_max_readout=5.0,
        )

        # Our bound uses sum, so combined = b1 + b2 exactly
        assert combined.bound == pytest.approx(b1.bound + b2.bound)
