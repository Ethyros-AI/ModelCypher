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

"""Tests for QK spectral bound domain module and analysis service."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelcypher.core.domain.geometry.qk_spectral_bound import (
    HeadCompositionChange,
    HeadSpectralBound,
    composition_change_bound,
    composition_relative_change,
    composition_significant,
    max_logit_magnitude,
    qk_projection_scale,
    qk_spectral_product,
    softcap_equivalent_bound,
    softcap_utilization,
)


# === Domain math tests ===


class TestSoftcapEquivalentBound:
    def test_known_values(self):
        """d_k=64, d_model=512, soft_cap=50 → 50 * 8 / 512 = 0.78125."""
        assert softcap_equivalent_bound(50.0, 64, 512) == pytest.approx(0.78125)

    def test_unit_dimensions(self):
        """d_k=1, d_model=1, soft_cap=1 → 1.0."""
        assert softcap_equivalent_bound(1.0, 1, 1) == pytest.approx(1.0)

    def test_gemma2_typical(self):
        """Gemma-2 2B: soft_cap=50, d_k=256, d_model=2304."""
        # 50 * sqrt(256) / 2304 = 50 * 16 / 2304 = 800 / 2304
        expected = 800.0 / 2304.0
        assert softcap_equivalent_bound(50.0, 256, 2304) == pytest.approx(expected)

    def test_scales_linearly_with_softcap(self):
        b1 = softcap_equivalent_bound(50.0, 64, 512)
        b2 = softcap_equivalent_bound(100.0, 64, 512)
        assert b2 == pytest.approx(2.0 * b1)


class TestQKSpectralProduct:
    def test_basic(self):
        assert qk_spectral_product(2.0, 3.0) == pytest.approx(6.0)

    def test_zero(self):
        assert qk_spectral_product(0.0, 5.0) == pytest.approx(0.0)


class TestQKProjectionScale:
    def test_below_bound_returns_one(self):
        """No projection needed when product is below bound."""
        assert qk_projection_scale(0.5, 0.5, 1.0) == pytest.approx(1.0)

    def test_exactly_at_bound_returns_one(self):
        """At boundary: no projection needed."""
        assert qk_projection_scale(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_above_bound_correct_alpha(self):
        """sigma_q=2, sigma_k=2, bound=1 → alpha=sqrt(1/4)=0.5."""
        alpha = qk_projection_scale(2.0, 2.0, 1.0)
        assert alpha == pytest.approx(0.5)

    def test_projection_restores_bound(self):
        """After scaling by alpha, product equals bound."""
        sigma_q, sigma_k, bound = 3.0, 4.0, 2.0
        alpha = qk_projection_scale(sigma_q, sigma_k, bound)
        projected_product = (alpha * sigma_q) * (alpha * sigma_k)
        assert projected_product == pytest.approx(bound)

    def test_asymmetric_sigmas(self):
        """Projection works correctly with unequal sigmas."""
        sigma_q, sigma_k, bound = 10.0, 0.5, 1.0
        alpha = qk_projection_scale(sigma_q, sigma_k, bound)
        projected_product = (alpha * sigma_q) * (alpha * sigma_k)
        assert projected_product == pytest.approx(bound)


class TestSoftcapUtilization:
    def test_zero_product(self):
        assert softcap_utilization(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_exactly_one(self):
        assert softcap_utilization(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_above_one(self):
        assert softcap_utilization(2.0, 2.0, 1.0) == pytest.approx(4.0)

    def test_zero_bound(self):
        assert softcap_utilization(1.0, 1.0, 0.0) == math.inf


class TestMaxLogitMagnitude:
    def test_known_values(self):
        """d_model=512, sigma_q=1, sigma_k=1, d_k=64 → 512*1*1/8 = 64."""
        assert max_logit_magnitude(1.0, 1.0, 64, 512) == pytest.approx(64.0)

    def test_consistency_with_bound(self):
        """When spectral product == bound, max logit == soft_cap."""
        soft_cap, d_k, d_model = 50.0, 64, 512
        bound = softcap_equivalent_bound(soft_cap, d_k, d_model)
        # sigma_q * sigma_k = bound → pick sigma_q = sigma_k = sqrt(bound)
        sigma = math.sqrt(bound)
        ml = max_logit_magnitude(sigma, sigma, d_k, d_model)
        assert ml == pytest.approx(soft_cap)


class TestCompositionChangeBound:
    def test_known_values(self):
        """sigma_q=1, sigma_k=1, delta_q=0.1, delta_k=0.1 → 0.21."""
        # 1*0.1 + 1*0.1 + 0.1*0.1 = 0.21
        assert composition_change_bound(1.0, 1.0, 0.1, 0.1) == pytest.approx(0.21)

    def test_zero_perturbation(self):
        assert composition_change_bound(1.0, 1.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_one_sided_perturbation(self):
        """Only K is perturbed: sigma_q * delta_k + 0 + 0 = 2.0 * 0.5 = 1.0."""
        assert composition_change_bound(2.0, 3.0, 0.0, 0.5) == pytest.approx(1.0)

    def test_typical_lora_regime(self):
        """sigma_max ≈ 1.0, sigma_k ≈ 0.01 (condition number 100).

        Cross-terms ≈ 2 * 1.0 * 0.01 + 0.01^2 = 0.0201.
        Relative change ≈ 2% — above sqrt(eps) but small.
        """
        change = composition_change_bound(1.0, 1.0, 0.01, 0.01)
        assert change == pytest.approx(0.0201)
        rel = composition_relative_change(1.0, 1.0, 0.01, 0.01)
        assert rel == pytest.approx(0.0201)


class TestCompositionRelativeChange:
    def test_zero_base(self):
        """Zero base product → zero relative change (no attention)."""
        assert composition_relative_change(0.0, 0.0, 0.1, 0.1) == pytest.approx(0.0)

    def test_known_value(self):
        # base = 2*3 = 6. change = 2*0.1 + 3*0.2 + 0.2*0.1 = 0.82. rel = 0.82/6
        rel = composition_relative_change(2.0, 3.0, 0.2, 0.1)
        assert rel == pytest.approx(0.82 / 6.0)


class TestCompositionSignificant:
    def test_below_sqrt_eps(self):
        eps = 1e-6
        # sqrt(eps) = 1e-3. Relative change 1e-4 < 1e-3 → not significant.
        assert composition_significant(1e-4, eps) is False

    def test_above_sqrt_eps(self):
        eps = 1e-6
        # sqrt(eps) = 1e-3. Relative change 0.01 > 1e-3 → significant.
        assert composition_significant(0.01, eps) is True

    def test_at_f32_boundary(self):
        """Typical LoRA regime: 2% change vs sqrt(eps_f32) ≈ 0.03%."""
        eps_f32 = 2.0**-23
        assert composition_significant(0.02, eps_f32) is True
        assert composition_significant(1e-5, eps_f32) is False


class TestHeadSpectralBound:
    def test_frozen(self):
        h = HeadSpectralBound(
            layer_idx=0, head_idx=0, sigma_q=1.0, sigma_k=1.0,
            spectral_product=1.0, bound=1.0, utilization=1.0,
            projection_scale=1.0, max_logit=64.0, softcap_active=False,
        )
        with pytest.raises(AttributeError):
            h.sigma_q = 2.0  # type: ignore[misc]


class TestHeadCompositionChange:
    def test_frozen(self):
        h = HeadCompositionChange(
            layer_idx=0, head_idx=0, base_product=1.0,
            modified_product=1.02, absolute_change=0.02,
            relative_change=0.02, significant=True,
        )
        with pytest.raises(AttributeError):
            h.base_product = 2.0  # type: ignore[misc]


# === Service tests with mock model ===


def _make_mock_model(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    num_layers: int,
) -> MagicMock:
    """Build a minimal mock model with config and attention layers."""
    model = MagicMock()
    model.config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )

    d_k = hidden_size // num_heads
    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        attn = MagicMock()

        # Q weight: [num_heads * d_k, hidden_size]
        q_weight = MagicMock()
        q_weight.shape = (num_heads * d_k, hidden_size)
        attn.q_proj.weight = q_weight

        # K weight: [num_kv_heads * d_k, hidden_size]
        k_weight = MagicMock()
        k_weight.shape = (num_kv_heads * d_k, hidden_size)
        attn.k_proj.weight = k_weight

        layer.self_attn = attn
        layers.append(layer)

    model.model.layers = layers
    return model


class TestQKSpectralServiceNoBackend:
    """Test service logic without requiring a real backend.

    These tests verify config extraction, GQA pairing, and aggregation.
    Backend-dependent sigma_max computation is tested via the validation script
    on real models.
    """

    def test_missing_config_raises(self):
        from modelcypher.core.use_cases.qk_spectral_service import QKSpectralService

        backend = MagicMock()
        service = QKSpectralService(backend)
        model = MagicMock(spec=[])  # no config attribute
        del model.config
        with pytest.raises(ValueError, match="no config"):
            service.analyze_model(model)

    def test_zero_dimensions_raises(self):
        from modelcypher.core.use_cases.qk_spectral_service import QKSpectralService

        backend = MagicMock()
        service = QKSpectralService(backend)
        model = MagicMock()
        model.config = SimpleNamespace(
            hidden_size=0, num_attention_heads=0, num_key_value_heads=0,
        )
        with pytest.raises(ValueError, match="Cannot extract dimensions"):
            service.analyze_model(model)
