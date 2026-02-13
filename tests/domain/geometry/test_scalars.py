# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import pytest

import modelcypher.core.domain.geometry.scalars as mod


# ---------------------------------------------------------------------------
# sqrt_scalar
# ---------------------------------------------------------------------------


class TestSqrtScalar:
    def test_sqrt_of_four(self, any_backend) -> None:
        assert mod.sqrt_scalar(4.0, any_backend) == pytest.approx(2.0)

    def test_sqrt_of_zero(self, any_backend) -> None:
        assert mod.sqrt_scalar(0.0, any_backend) == pytest.approx(0.0)

    def test_sqrt_of_one(self, any_backend) -> None:
        assert mod.sqrt_scalar(1.0, any_backend) == pytest.approx(1.0)

    def test_sqrt_negative_clamped_to_zero(self, any_backend) -> None:
        """Negative values are clamped to 0 via max(0.0, value)."""
        assert mod.sqrt_scalar(-1.0, any_backend) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# is_finite / is_inf / is_nan
# ---------------------------------------------------------------------------


class TestFiniteInfNan:
    def test_is_finite_normal(self, any_backend) -> None:
        assert mod.is_finite(1.0, any_backend) is True

    def test_is_finite_inf(self, any_backend) -> None:
        assert mod.is_finite(float("inf"), any_backend) is False

    def test_is_finite_nan(self, any_backend) -> None:
        assert mod.is_finite(float("nan"), any_backend) is False

    def test_is_inf_positive(self, any_backend) -> None:
        assert mod.is_inf(float("inf"), any_backend) is True

    def test_is_inf_negative(self, any_backend) -> None:
        assert mod.is_inf(float("-inf"), any_backend) is True

    def test_is_inf_normal(self, any_backend) -> None:
        assert mod.is_inf(1.0, any_backend) is False

    def test_is_nan_true(self, any_backend) -> None:
        assert mod.is_nan(float("nan"), any_backend) is True

    def test_is_nan_false(self, any_backend) -> None:
        assert mod.is_nan(1.0, any_backend) is False

    def test_is_nan_zero(self, any_backend) -> None:
        assert mod.is_nan(0.0, any_backend) is False


# ---------------------------------------------------------------------------
# all_finite
# ---------------------------------------------------------------------------


class TestAllFinite:
    def test_all_finite_true(self, any_backend) -> None:
        arr = any_backend.array([1.0, 2.0, 3.0])
        assert mod.all_finite(arr, any_backend) is True

    def test_all_finite_with_inf(self, any_backend) -> None:
        arr = any_backend.array([1.0, float("inf"), 3.0])
        assert mod.all_finite(arr, any_backend) is False

    def test_all_finite_with_nan(self, any_backend) -> None:
        arr = any_backend.array([1.0, float("nan"), 3.0])
        assert mod.all_finite(arr, any_backend) is False

    def test_all_finite_single_element(self, any_backend) -> None:
        arr = any_backend.array([42.0])
        assert mod.all_finite(arr, any_backend) is True


# ---------------------------------------------------------------------------
# log_scalar / exp_scalar
# ---------------------------------------------------------------------------


class TestLogExp:
    def test_log_of_one(self, any_backend) -> None:
        assert mod.log_scalar(1.0, any_backend) == pytest.approx(0.0)

    def test_log_of_e(self, any_backend) -> None:
        assert mod.log_scalar(math.e, any_backend) == pytest.approx(1.0, abs=1e-6)

    def test_exp_of_zero(self, any_backend) -> None:
        assert mod.exp_scalar(0.0, any_backend) == pytest.approx(1.0)

    def test_exp_of_one(self, any_backend) -> None:
        assert mod.exp_scalar(1.0, any_backend) == pytest.approx(math.e, abs=1e-6)

    def test_log_exp_roundtrip(self, any_backend) -> None:
        val = 2.5
        result = mod.exp_scalar(mod.log_scalar(val, any_backend), any_backend)
        assert result == pytest.approx(val, abs=1e-5)


# ---------------------------------------------------------------------------
# power_scalar
# ---------------------------------------------------------------------------


class TestPowerScalar:
    def test_two_cubed(self, any_backend) -> None:
        assert mod.power_scalar(2.0, 3.0, any_backend) == pytest.approx(8.0)

    def test_anything_to_zero(self, any_backend) -> None:
        assert mod.power_scalar(5.0, 0.0, any_backend) == pytest.approx(1.0)

    def test_square_root_via_power(self, any_backend) -> None:
        assert mod.power_scalar(9.0, 0.5, any_backend) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ceil_scalar / floor_scalar
# ---------------------------------------------------------------------------


class TestCeilFloor:
    def test_ceil_of_1_3(self, any_backend) -> None:
        assert mod.ceil_scalar(1.3, any_backend) == 2

    def test_ceil_of_integer(self, any_backend) -> None:
        assert mod.ceil_scalar(3.0, any_backend) == 3

    def test_ceil_negative(self, any_backend) -> None:
        assert mod.ceil_scalar(-1.3, any_backend) == -1

    def test_floor_of_1_7(self, any_backend) -> None:
        assert mod.floor_scalar(1.7, any_backend) == 1

    def test_floor_of_integer(self, any_backend) -> None:
        assert mod.floor_scalar(3.0, any_backend) == 3

    def test_floor_negative(self, any_backend) -> None:
        assert mod.floor_scalar(-1.3, any_backend) == -2


# ---------------------------------------------------------------------------
# ulp_scalar
# ---------------------------------------------------------------------------


class TestUlpScalar:
    def test_ulp_of_one(self, any_backend) -> None:
        result = mod.ulp_scalar(1.0, any_backend)
        assert result > 0.0
        # For float32 eps ~ 1.19e-7, for float64 eps ~ 2.22e-16
        assert result < 1e-5

    def test_ulp_of_zero(self, any_backend) -> None:
        """When value is 0, ulp returns eps directly."""
        result = mod.ulp_scalar(0.0, any_backend)
        assert result > 0.0
        assert result < 1e-5


# ---------------------------------------------------------------------------
# Trig functions
# ---------------------------------------------------------------------------


class TestTrig:
    def test_acos_of_one(self, any_backend) -> None:
        assert mod.acos_scalar(1.0, any_backend) == pytest.approx(0.0)

    def test_acos_of_zero(self, any_backend) -> None:
        assert mod.acos_scalar(0.0, any_backend) == pytest.approx(math.pi / 2, abs=1e-5)

    def test_cos_of_zero(self, any_backend) -> None:
        assert mod.cos_scalar(0.0, any_backend) == pytest.approx(1.0)

    def test_sin_of_zero(self, any_backend) -> None:
        assert mod.sin_scalar(0.0, any_backend) == pytest.approx(0.0, abs=1e-7)

    def test_sin_of_pi_over_2(self, any_backend) -> None:
        pi = mod.pi_value(any_backend)
        assert mod.sin_scalar(pi / 2, any_backend) == pytest.approx(1.0, abs=1e-5)

    def test_cos_sin_pythagorean(self, any_backend) -> None:
        """sin^2 + cos^2 = 1 for any angle."""
        angle = 1.23
        s = mod.sin_scalar(angle, any_backend)
        c = mod.cos_scalar(angle, any_backend)
        assert s**2 + c**2 == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# atan2_scalar
# ---------------------------------------------------------------------------


class TestAtan2:
    def test_atan2_zero_one(self, any_backend) -> None:
        assert mod.atan2_scalar(0.0, 1.0, any_backend) == pytest.approx(0.0, abs=1e-5)

    def test_atan2_one_zero(self, any_backend) -> None:
        pi = mod.pi_value(any_backend)
        assert mod.atan2_scalar(1.0, 0.0, any_backend) == pytest.approx(pi / 2, abs=1e-5)

    def test_atan2_negative_one_zero(self, any_backend) -> None:
        pi = mod.pi_value(any_backend)
        assert mod.atan2_scalar(-1.0, 0.0, any_backend) == pytest.approx(-pi / 2, abs=1e-5)

    def test_atan2_origin(self, any_backend) -> None:
        assert mod.atan2_scalar(0.0, 0.0, any_backend) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# log2_scalar
# ---------------------------------------------------------------------------


class TestLog2:
    def test_log2_of_eight(self, any_backend) -> None:
        assert mod.log2_scalar(8.0, any_backend) == pytest.approx(3.0)

    def test_log2_of_one(self, any_backend) -> None:
        assert mod.log2_scalar(1.0, any_backend) == pytest.approx(0.0)

    def test_log2_of_two(self, any_backend) -> None:
        assert mod.log2_scalar(2.0, any_backend) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Constants: pi_value, e_value, inf_value
# ---------------------------------------------------------------------------


class TestConstants:
    def test_pi_value(self, any_backend) -> None:
        assert mod.pi_value(any_backend) == pytest.approx(math.pi, abs=1e-5)

    def test_e_value(self, any_backend) -> None:
        assert mod.e_value(any_backend) == pytest.approx(math.e, abs=1e-5)

    def test_inf_value(self, any_backend) -> None:
        result = mod.inf_value(any_backend)
        assert result == float("inf")
        assert math.isinf(result)


# ---------------------------------------------------------------------------
# lgamma_scalar
# ---------------------------------------------------------------------------


class TestLgamma:
    def test_lgamma_of_one(self, any_backend) -> None:
        """lgamma(1) = log(Gamma(1)) = log(1) = 0."""
        assert mod.lgamma_scalar(1.0, any_backend) == pytest.approx(0.0, abs=1e-5)

    def test_lgamma_of_two(self, any_backend) -> None:
        """lgamma(2) = log(Gamma(2)) = log(1!) = 0."""
        assert mod.lgamma_scalar(2.0, any_backend) == pytest.approx(0.0, abs=1e-5)

    def test_lgamma_of_three(self, any_backend) -> None:
        """lgamma(3) = log(Gamma(3)) = log(2!) = log(2)."""
        assert mod.lgamma_scalar(3.0, any_backend) == pytest.approx(math.log(2.0), abs=1e-5)

    def test_lgamma_of_five(self, any_backend) -> None:
        """lgamma(5) = log(Gamma(5)) = log(4!) = log(24)."""
        assert mod.lgamma_scalar(5.0, any_backend) == pytest.approx(math.log(24.0), abs=1e-4)
