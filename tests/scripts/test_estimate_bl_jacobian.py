# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Deterministic unit tests for B_l Jacobian estimator numerics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import estimate_bl_jacobian as bl_jac


def test_ridge_caps_condition_number_full_rank():
    ptp = np.diag(np.array([9.0, 1.0, 0.25], dtype=np.float64))
    ridge = bl_jac._ridge_for_gram(ptp)

    assert ridge["lam_max"] == 9.0
    assert ridge["lam_min"] == 0.25
    assert ridge["ridge_lambda"] > 0.0
    assert np.isfinite(ridge["kappa_raw"])
    assert np.isfinite(ridge["kappa_reg"])
    assert ridge["kappa_reg"] <= ridge["kappa_target"] * (1.0 + 1e-9)


def test_ridge_caps_condition_number_near_singular():
    ptp = np.diag(np.array([10.0, 1.0e-12], dtype=np.float64))
    ridge = bl_jac._ridge_for_gram(ptp)

    assert ridge["ridge_lambda"] > 0.0
    assert np.isfinite(ridge["kappa_reg"])
    assert ridge["kappa_reg"] <= ridge["kappa_target"] * (1.0 + 1e-6)


def test_ridge_handles_nonpositive_lambda_min():
    # Symmetric matrix with eigenvalues (3, -1).
    ptp = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    ridge = bl_jac._ridge_for_gram(ptp)

    assert np.isnan(ridge["kappa_raw"])
    assert ridge["ridge_lambda"] > 0.0
    assert np.isfinite(ridge["kappa_reg"])
    assert ridge["kappa_reg"] <= ridge["kappa_target"] * (1.0 + 1e-6)


def test_scaled_prediction_roundtrip_consistency():
    p_fit = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    f_fit = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.0, 1.5, -3.0],
        ],
        dtype=np.float64,
    )
    coeff_true = np.array([0.2, -1.1, 0.5], dtype=np.float64)
    p_val = p_fit @ coeff_true
    expected_f = f_fit @ coeff_true

    solve = bl_jac._solve_scaled_linear_map(
        P_fit=p_fit,
        F_fit=f_fit,
        p_val=p_val,
        ridge_lambda=0.0,
    )

    assert solve["solve_ok"] is True
    assert solve["finite_ok"] is True
    assert np.allclose(solve["coeff"], coeff_true, rtol=1e-10, atol=1e-10)
    assert np.allclose(solve["f_pred"], expected_f, rtol=1e-10, atol=1e-10)


def test_nan_contract_when_insufficient_dirs():
    baseline = {
        "h_q": np.array([1.0], dtype=np.float32),
        "f_val": np.array([0.0], dtype=np.float64),
        "p_val": np.array([1.0], dtype=np.float64),
        "h_norm": 1.0,
        "h_out_norm_sq": 1.0,
    }

    measured = bl_jac._estimate_local_coupling(
        layer=None,
        h_in=None,
        q_idx=0,
        layer_mask=None,
        baseline=baseline,
        final_norm=None,
        output_head=None,
        embed_tokens=None,
        mx=None,
        epsilon=bl_jac._SQRT_EPS_BF16,
        n_dirs=1,
        seed=0,
    )

    assert measured["n_dirs"] == 0
    assert measured["holdout_attempted"] == 0
    assert measured["holdout_used"] == 0
    assert measured["solve_fail_count"] == 0
    assert measured["nonfinite_fail_count"] == 0
    assert np.isnan(measured["n_fit_dirs"])
    assert np.isnan(measured["n_holdout_dirs"])
    assert np.isnan(measured["p_scale"])
    assert np.isnan(measured["f_scale"])
    assert np.isnan(measured["ptp_lam_max"])
    assert np.isnan(measured["ptp_lam_min"])
    assert np.isnan(measured["ptp_cond_raw"])
    assert np.isnan(measured["ridge_lambda"])
    assert np.isnan(measured["ptp_cond_reg"])
    assert np.isnan(measured["kappa_target"])
    assert np.isnan(measured["c_l_candidates_max"])
    assert np.isnan(measured["c_l_candidates_mean"])
    assert np.isnan(measured["c_l_candidates_std"])
