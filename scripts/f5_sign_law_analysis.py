#!/usr/bin/env python3
"""F5 sign-law analysis for entropy->curvature coupling.

This script is pure post-processing on CR-EC-001 operator-split artifacts.
It decomposes the depth-controlled H_logit->theta_total relation into:

1) core component (attention or conv)
2) MLP component
3) cross-term component (sublayer perturbation alignment proxy)

Geometry identity used at layer level (angle-space proxy):
    theta_total^2 = theta_core^2 + theta_mlp^2 + cross_energy
where:
    cross_energy := theta_total^2 - theta_core^2 - theta_mlp^2
and an alignment proxy can be computed as:
    cos_proxy := cross_energy / (2 * theta_core * theta_mlp)

Usage:
    poetry run python scripts/f5_sign_law_analysis.py
    poetry run python scripts/f5_sign_law_analysis.py --models LFM2-700M Qwen2.5-3B
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# TODO(derivation): Replace this heuristic with a derived minimum decomposition
# coverage criterion based on operator-count stability / estimator uncertainty.
DECOMP_COVERAGE_MIN_HEURISTIC = 0.5


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y after removing linear effect of x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def _residualize_quadratic(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y after removing quadratic depth trend."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x, x * x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def _depth_controlled_slope(y: np.ndarray, h: np.ndarray, depth: np.ndarray) -> dict:
    """Slope of y on h controlling depth via residualization."""
    y_res = _residualize(y, depth)
    h_res = _residualize(h, depth)
    slope, intercept, r_value, p_value, std_err = stats.linregress(h_res, y_res)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman correlation with finite-mask handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(mask.sum())}
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {"rho": float(rho), "p_value": float(p), "n": int(mask.sum())}


def _effective_df(
    residuals: np.ndarray, n_controls: int = 1,
) -> tuple[float, float]:
    """Effective degrees of freedom for partial correlation under autocorrelation.

    Adjacent transformer layers share residual stream state, so n layers are
    not n independent observations. This computes the effective sample size
    using the Bretherton et al. (1999) AR(1) correction.

    Args:
        residuals: Depth-controlled residuals, ordered by layer index.
        n_controls: Number of controlled variables (1 for depth).

    Returns:
        (n_eff, rho_1): Effective sample size and lag-1 autocorrelation.

    Reference:
        Bretherton, C.S. et al. (1999). "The effective number of spatial
        degrees of freedom of a time-varying field." J. Climate 12:1990-2009.
        Eq. (31) for AR(1) case.
    """
    n = len(residuals)
    if n < 4:
        return float(n), 0.0

    r = np.asarray(residuals, dtype=float)
    mean_r = np.mean(r)
    r_centered = r - mean_r

    denom = np.sum(r_centered ** 2)
    if abs(denom) < 1e-15:
        return float(n), 0.0

    # Unbiased lag-1 autocorrelation.
    numer = np.sum(r_centered[:-1] * r_centered[1:])
    rho_1 = float(numer / denom)

    # Bretherton et al. (1999) Eq. (31): n_eff = n * (1 - rho_1) / (1 + rho_1)
    rho_1_clamped = max(-0.99, min(0.99, rho_1))
    n_eff = n * (1 - rho_1_clamped) / (1 + rho_1_clamped)
    # Floor at 4 (minimum for partial correlation), ceiling at n (cannot
    # have more independent observations than physical layers).
    n_eff = max(4.0, min(float(n), n_eff))

    return n_eff, rho_1


def _minimum_detectable_effect(n_eff: float, n_controls: int = 1) -> float:
    """Minimum detectable partial correlation magnitude (Fisher-SE MDE).

    The Fisher transform z = atanh(r) has asymptotic standard error
    1/sqrt(df) where df = n_eff - n_controls - 2. The MDE is the smallest
    |r| where the signal-to-noise ratio of the estimator equals 1:
        |z| / SE >= 1  =>  |r| >= tanh(SE)

    This is not an imposed threshold — it is the measurement resolution
    of the partial correlation estimator at this effective sample size.

    Args:
        n_eff: Effective sample size (after autocorrelation correction).
        n_controls: Number of controlled variables.

    Returns:
        MDE as |partial_r|. Returns 1.0 if df <= 0 (unresolvable).
    """
    df = n_eff - n_controls - 2
    if df <= 0:
        return 1.0
    se_z = 1.0 / np.sqrt(df)
    return float(np.tanh(se_z))


def _depth_model_diagnostics(
    h_total: np.ndarray, theta_total: np.ndarray, depth_total: np.ndarray,
) -> dict:
    """Compare linear vs quadratic depth control for residual autocorrelation.

    Reports whether the H-theta coupling remains above the measurement floor
    under each depth model. This is diagnostic only; it does not override the
    primary detection-floor result.
    """

    def _run(mode: str) -> dict:
        if mode == "linear":
            h_res = _residualize(h_total, depth_total)
            th_res = _residualize(theta_total, depth_total)
            n_controls = 1
        else:
            h_res = _residualize_quadratic(h_total, depth_total)
            th_res = _residualize_quadratic(theta_total, depth_total)
            n_controls = 2

        pearson_r = float(np.corrcoef(h_res, th_res)[0, 1])
        spearman = _safe_spearman(h_res, th_res)

        n_eff_h, rho1_h = _effective_df(h_res, n_controls=n_controls)
        n_eff_th, rho1_th = _effective_df(th_res, n_controls=n_controls)
        if abs(rho1_h) >= abs(rho1_th):
            n_eff, rho1_used = n_eff_h, rho1_h
            rho1_source = "H_logit_residuals"
        else:
            n_eff, rho1_used = n_eff_th, rho1_th
            rho1_source = "theta_total_residuals"

        mde = _minimum_detectable_effect(n_eff, n_controls=n_controls)
        return {
            "pearson_r": pearson_r,
            "spearman_r": spearman["rho"],
            "spearman_p": spearman["p_value"],
            "rho_1_h": rho1_h,
            "rho_1_theta": rho1_th,
            "rho_1_used": rho1_used,
            "rho_1_source": rho1_source,
            "n_eff": n_eff,
            "n_controls": n_controls,
            "mde": mde,
            "resolvable": abs(pearson_r) > mde,
        }

    return {
        "linear": _run("linear"),
        "quadratic": _run("quadratic"),
    }


def _permutation_null_diagnostic(
    h_resid: np.ndarray,
    theta_resid: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42,
) -> dict:
    """Permutation null diagnostic for observed |Spearman r|.

    Permutes h_resid (breaking H-theta association while preserving marginal
    distributions) and computes the distribution of |Spearman r| under the
    null. Reports the fraction of null draws that exceed the observed |r|
    (empirical p-value) and summary statistics of the null distribution.

    This is a diagnostic — the Fisher-SE MDE is the primary derived threshold.
    The permutation validates that the Fisher MDE is consistent with the
    empirical null structure.

    Args:
        h_resid: Depth-residualized H_logit values (layer-ordered).
        theta_resid: Depth-residualized theta values (layer-ordered).
        n_permutations: Number of permutations.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with null_mean, null_max, observed_abs_r, exceedance_fraction.
    """
    n = len(h_resid)
    if n < 4:
        return {"null_mean": float("nan"), "null_max": float("nan"),
                "exceedance_fraction": float("nan")}

    # Use Pearson r to match the Fisher-SE MDE statistic (OLS r_value).
    obs_r = float(np.corrcoef(h_resid, theta_resid)[0, 1])
    obs_abs_r = abs(obs_r)

    rng = np.random.default_rng(seed=seed)
    null_abs_r = np.empty(n_permutations)

    for i in range(n_permutations):
        h_perm = rng.permutation(h_resid)
        r_perm = float(np.corrcoef(h_perm, theta_resid)[0, 1])
        null_abs_r[i] = abs(r_perm)

    exceedance = float(np.mean(null_abs_r >= obs_abs_r))

    return {
        "null_mean": float(np.mean(null_abs_r)),
        "null_max": float(np.max(null_abs_r)),
        "observed_abs_r": obs_abs_r,
        "exceedance_fraction": exceedance,
        "n_permutations": n_permutations,
    }


def _sign_of(x: float) -> str:
    if math.isnan(x):
        return "unknown"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"


def _load_operator_split(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_model(model_result: dict) -> dict:
    """Compute F5 decomposition terms for a single model."""
    model_name = model_result["model_name"]
    architecture = model_result["architecture"]
    measurements = model_result["measurements"]

    rows_total = []
    rows_decomp = []
    for m in measurements:
        h = m.get("H_logit")
        th_t = m.get("theta_total")
        th_core = m.get("theta_core")
        if th_core is None:
            # Backward compatibility with pre-core artifacts.
            th_core = m.get("theta_attn")
        th_m = m.get("theta_mlp")
        core_op = m.get("core_operator")
        d = m.get("depth_fraction")
        if h is None or th_t is None or d is None:
            continue
        if not all(np.isfinite([h, th_t, d])):
            continue
        rows_total.append((float(h), float(th_t), float(d), int(m["layer_idx"])))
        if th_core is None or th_m is None:
            continue
        if not all(np.isfinite([th_core, th_m])):
            continue
        rows_decomp.append(
            (
                float(h),
                float(th_t),
                float(th_core),
                float(th_m),
                float(d),
                int(m["layer_idx"]),
                str(core_op) if core_op is not None else "unknown",
            )
        )

    if len(rows_decomp) < 6:
        return {
            "model_name": model_name,
            "architecture": architecture,
            "status": "INSUFFICIENT_DATA",
            "n_rows_total": len(rows_total),
            "n_rows_decomp": len(rows_decomp),
        }

    h_total = np.array([r[0] for r in rows_total], dtype=float)
    theta_t_total = np.array([r[1] for r in rows_total], dtype=float)
    depth_total = np.array([r[2] for r in rows_total], dtype=float)
    layers_total = np.array([r[3] for r in rows_total], dtype=int)

    h = np.array([r[0] for r in rows_decomp], dtype=float)
    theta_t = np.array([r[1] for r in rows_decomp], dtype=float)
    theta_core = np.array([r[2] for r in rows_decomp], dtype=float)
    theta_m = np.array([r[3] for r in rows_decomp], dtype=float)
    depth = np.array([r[4] for r in rows_decomp], dtype=float)
    layers = np.array([r[5] for r in rows_decomp], dtype=int)
    core_ops = [r[6] for r in rows_decomp]

    theta_t2 = theta_t * theta_t
    theta_core2 = theta_core * theta_core
    theta_m2 = theta_m * theta_m

    # Exact layer-wise residual in squared-angle proxy space.
    cross_energy = theta_t2 - theta_core2 - theta_m2

    denom = 2.0 * theta_core * theta_m
    cos_proxy = np.full_like(cross_energy, np.nan)
    valid = np.abs(denom) > 1e-12
    cos_proxy[valid] = cross_energy[valid] / denom[valid]

    # Optional clamped version for interpretability (physical cosine range).
    cos_proxy_clamped = np.clip(cos_proxy, -1.0, 1.0)
    outside_unit = np.isfinite(cos_proxy) & ((cos_proxy < -1.0) | (cos_proxy > 1.0))

    # Raw (F5-compatible) Spearman terms in theta-space.
    raw_t_all = _safe_spearman(h_total, theta_t_total)
    raw_t = _safe_spearman(h, theta_t)
    raw_core = _safe_spearman(h, theta_core)
    raw_m = _safe_spearman(h, theta_m)
    beta_cross_raw = raw_t["rho"] - raw_core["rho"] - raw_m["rho"]

    # Depth-controlled slopes in theta-space (secondary diagnostic).
    beta_t_all = _depth_controlled_slope(theta_t_total, h_total, depth_total)
    beta_t = _depth_controlled_slope(theta_t, h, depth)
    beta_core = _depth_controlled_slope(theta_core, h, depth)
    beta_m = _depth_controlled_slope(theta_m, h, depth)

    # --- Detection floor (Fisher-SE MDE + Bretherton autocorrelation) ---
    # Compute depth-residualized series for autocorrelation analysis.
    h_resid_all = _residualize(h_total, depth_total)
    theta_resid_all = _residualize(theta_t_total, depth_total)

    # Use the more conservative (higher |rho_1|) of the two residual series.
    n_eff_h, rho1_h = _effective_df(h_resid_all)
    n_eff_th, rho1_th = _effective_df(theta_resid_all)
    if abs(rho1_h) >= abs(rho1_th):
        n_eff, rho1_used = n_eff_h, rho1_h
        rho1_source = "H_logit_residuals"
    else:
        n_eff, rho1_used = n_eff_th, rho1_th
        rho1_source = "theta_total_residuals"

    mde = _minimum_detectable_effect(n_eff)
    perm_diag = _permutation_null_diagnostic(h_resid_all, theta_resid_all)
    depth_model_diag = _depth_model_diagnostics(h_total, theta_t_total, depth_total)

    # The OLS r_value from beta_t_all is the depth-controlled correlation.
    obs_r = abs(beta_t_all["r_value"])
    resolvable = obs_r > mde

    beta_cross_angle = beta_t["slope"] - beta_core["slope"] - beta_m["slope"]

    # Depth-controlled slopes in squared-angle proxy space with exact closure.
    beta_t2 = _depth_controlled_slope(theta_t2, h, depth)
    beta_core2 = _depth_controlled_slope(theta_core2, h, depth)
    beta_m2 = _depth_controlled_slope(theta_m2, h, depth)
    beta_cross_energy = _depth_controlled_slope(cross_energy, h, depth)
    closure_error_t2 = (
        beta_t2["slope"] - beta_core2["slope"] - beta_m2["slope"] - beta_cross_energy["slope"]
    )

    # Cross-term diagnostics against H_logit.
    cross_corr = _safe_spearman(h, cos_proxy)
    cross_corr_clamped = _safe_spearman(h, cos_proxy_clamped)
    cross_energy_corr = _safe_spearman(h, cross_energy)
    cross_slope = _depth_controlled_slope(cos_proxy_clamped, h, depth)

    # Measurable sign-law candidate:
    # sign(theta_total slope) predicted by sign(beta_a + beta_m + beta_cross_angle).
    beta_sum_angle = beta_core["slope"] + beta_m["slope"] + beta_cross_angle
    observed_sign = _sign_of(beta_t["slope"])
    predicted_sign = _sign_of(beta_sum_angle)

    observed_sign_raw = _sign_of(raw_t_all["rho"])
    predicted_sign_raw = _sign_of(raw_core["rho"] + raw_m["rho"] + beta_cross_raw)

    # Architecture-conditioned mechanism classification.
    if raw_core["rho"] > 0 and raw_m["rho"] < 0:
        mechanism = "competing_sublayers"
    elif raw_core["rho"] > 0 and raw_m["rho"] >= 0:
        mechanism = "core_pass_through"
    elif raw_core["rho"] <= 0 and raw_m["rho"] > 0:
        mechanism = "mlp_dominant"
    else:
        mechanism = "mixed_or_flat"

    return {
        "model_name": model_name,
        "architecture": architecture,
        "status": "OK",
        "n_rows_total": len(rows_total),
        "n_rows_decomp": len(rows_decomp),
        "decomp_coverage_fraction": float(len(rows_decomp) / max(len(rows_total), 1)),
        "layers_total": layers_total.tolist(),
        "layers_used": layers.tolist(),
        "theta_space": {
            "raw_spearman": {
                "beta_total_all_layers": raw_t_all,
                "beta_total_decomp_layers": raw_t,
                "beta_core": raw_core,
                "beta_mlp": raw_m,
                "beta_cross_residual": {
                    "rho": float(beta_cross_raw),
                    "definition": "rho_decomp(H,theta_total) - rho(H,theta_core) - rho(H,theta_mlp)",
                },
                "observed_sign": observed_sign_raw,
                "predicted_sign": predicted_sign_raw,
            },
            "depth_controlled": {
            "beta_total_all_layers": beta_t_all,
            "beta_total_decomp_layers": beta_t,
            "beta_core": beta_core,
            "beta_mlp": beta_m,
            "beta_cross_residual": {
                "slope": float(beta_cross_angle),
                "definition": "beta_total_decomp - beta_core - beta_mlp",
            },
            "beta_sum_check_decomp": float(beta_sum_angle),
            "observed_sign": observed_sign,
            "predicted_sign": predicted_sign,
            },
        },
        "theta_squared_space": {
            "beta_total_sq": beta_t2,
            "beta_core_sq": beta_core2,
            "beta_mlp_sq": beta_m2,
            "beta_cross_energy": beta_cross_energy,
            "closure_error": float(closure_error_t2),
        },
        "cross_term_diagnostics": {
            "cross_energy_spearman_with_H_logit": cross_energy_corr,
            "cos_proxy_spearman_with_H_logit": cross_corr,
            "cos_proxy_clamped_spearman_with_H_logit": cross_corr_clamped,
            "cos_proxy_clamped_depth_controlled_slope": cross_slope,
            "cos_proxy_outside_unit_fraction": float(np.mean(outside_unit)) if len(outside_unit) else 0.0,
        },
        "mechanism_classification": mechanism,
        "core_operator_counts": {
            op: core_ops.count(op) for op in sorted(set(core_ops))
        },
        "detection_floor": {
            "n_raw": len(rows_total),
            "rho_1_h": rho1_h,
            "rho_1_theta": rho1_th,
            "rho_1_used": rho1_used,
            "rho_1_source": rho1_source,
            "n_eff": n_eff,
            "mde": mde,
            "observed_abs_r": obs_r,
            "resolvable": resolvable,
            "permutation_diagnostic": perm_diag,
            "derivation": (
                "Fisher-SE MDE with Bretherton (1999) autocorrelation correction. "
                "MDE = tanh(1/sqrt(n_eff - 3)), n_eff = n*(1-rho1)/(1+rho1). "
                "Permutation null as diagnostic validation (not threshold)."
            ),
        },
        "depth_model_diagnostics": depth_model_diag,
    }


def _predict_mechanism(
    architecture: str, core_operator_counts: dict, decomp_coverage: float = 1.0,
) -> str:
    """Pre-registered mechanism prediction from architecture type.

    Hybrid architectures (conv+attn) → competing_sublayers:
        MLP opposes core signal because core handles specialized function.
    Pure attention → core_pass_through:
        MLP extends/amplifies attention signal without opposition.
    Identity-core dominant (no explicit attention/conv core) → mlp_dominant:
        Layer update is carried primarily by the MLP path.

    Uses architecture identity for known-hybrid families (coverage-independent).
    For unknown architectures, requires decomp_coverage >=
    DECOMP_COVERAGE_MIN_HEURISTIC to trust
    core_operator_counts — low coverage makes operator counts unreliable.
    """
    # Known hybrid architectures predict competing_sublayers directly.
    KNOWN_HYBRID = {"lfm2"}
    if architecture in KNOWN_HYBRID:
        return "competing_sublayers"
    # Low coverage → unreliable core_operator_counts.
    if decomp_coverage < DECOMP_COVERAGE_MIN_HEURISTIC:
        return "coverage_insufficient"
    # Standard prediction from dominant core operator in decomposition.
    # Use argmax over measured operator counts (no heuristic thresholds).
    if core_operator_counts:
        dominant_core = max(core_operator_counts.items(), key=lambda kv: kv[1])[0]
    else:
        dominant_core = "unknown"

    if dominant_core == "conv":
        return "competing_sublayers"
    if dominant_core == "identity":
        return "mlp_dominant"
    return "core_pass_through"


def build_cross_model_summary(per_model: list[dict]) -> dict:
    """Aggregate per-model F5 sign-law diagnostics."""
    rows = []
    ok_models = [m for m in per_model if m.get("status") == "OK"]
    for m in ok_models:
        t_raw = m["theta_space"]["raw_spearman"]
        t_dc = m["theta_space"]["depth_controlled"]
        rows.append({
            "model": m["model_name"],
            "architecture": m["architecture"],
            "rho_total_all_layers": t_raw["beta_total_all_layers"]["rho"],
            "rho_total_decomp_layers": t_raw["beta_total_decomp_layers"]["rho"],
            "rho_core": t_raw["beta_core"]["rho"],
            "rho_mlp": t_raw["beta_mlp"]["rho"],
            "rho_cross_residual": t_raw["beta_cross_residual"]["rho"],
            "observed_sign_raw": t_raw["observed_sign"],
            "beta_total_all_layers_depth_controlled": t_dc["beta_total_all_layers"]["slope"],
            "beta_total_decomp_layers_depth_controlled": t_dc["beta_total_decomp_layers"]["slope"],
            "beta_core_depth_controlled": t_dc["beta_core"]["slope"],
            "beta_mlp_depth_controlled": t_dc["beta_mlp"]["slope"],
            "decomp_coverage_fraction": m.get("decomp_coverage_fraction"),
            "core_operator_counts": m.get("core_operator_counts", {}),
            "mechanism_classification": m["mechanism_classification"],
        })

    # F5 law candidate: sign(beta_mlp) as architecture term gate when beta_core > 0.
    gate_checks = []
    for m in ok_models:
        ba = m["theta_space"]["raw_spearman"]["beta_core"]["rho"]
        bm = m["theta_space"]["raw_spearman"]["beta_mlp"]["rho"]
        bt = m["theta_space"]["raw_spearman"]["beta_total_all_layers"]["rho"]
        if ba <= 0:
            continue
        if bm < 0:
            gate_checks.append(abs(bt) <= abs(ba))
        else:
            gate_checks.append(bt >= 0)

    # --- Formal sign law test (depth-controlled) ---
    # Prediction: depth-controlled β_total is negative for all architectures.
    # At fixed depth, higher logit entropy (more diffuse posterior) → less
    # angular change (model makes smaller geometric moves when uncertain).
    sign_law = _build_sign_law_test(ok_models)

    return {
        "models": rows,
        "f5_gate_check_count": len(gate_checks),
        "f5_gate_check_passes": int(sum(1 for x in gate_checks if x)),
        "f5_gate_check_status": (
            "PASS" if gate_checks and all(gate_checks) else
            "FAIL" if gate_checks else "INCONCLUSIVE"
        ),
        "candidate_law": (
            "When beta_core > 0, sign/magnitude of beta_mlp acts as suppression gate; "
            "negative beta_mlp yields competition, non-negative beta_mlp allows core pass-through."
        ),
        "sign_law_test": sign_law,
    }


def _build_sign_law_test(ok_models: list[dict]) -> dict:
    """Depth-controlled sign law test with derived detection floor.

    Uses Fisher-SE minimum detectable effect (MDE) corrected for layer
    autocorrelation (Bretherton 1999) to classify each model's depth-controlled
    correlation as resolvable or below measurement resolution. No heuristic
    thresholds — the MDE is the measurement resolution of the partial
    correlation estimator at each model's effective sample size.

    Mechanism predictions (coverage-gated) are independent of detection floor.
    """
    per_model = {}
    mechanism_predictions = {}

    for m in ok_models:
        name = m["model_name"]
        t_dc = m["theta_space"]["depth_controlled"]
        beta_all = t_dc["beta_total_all_layers"]
        beta_core = t_dc["beta_core"]
        beta_mlp = t_dc["beta_mlp"]
        core_ops = m.get("core_operator_counts", {})
        decomp_cov = m.get("decomp_coverage_fraction", 1.0)
        det_floor = m.get("detection_floor", {})

        slope = beta_all["slope"]
        p_val = beta_all["p_value"]
        r_val = beta_all.get("r_value", float("nan"))

        per_model[name] = {
            "beta_total": slope,
            "p_value": p_val,
            "r_value": r_val,
            "sign": _sign_of(slope),
            "beta_core": beta_core["slope"],
            "beta_core_p": beta_core["p_value"],
            "beta_mlp": beta_mlp["slope"],
            "beta_mlp_p": beta_mlp["p_value"],
            "detection_floor": det_floor,
            "depth_model_diagnostics": m.get("depth_model_diagnostics", {}),
        }

        # Mechanism prediction (coverage-gated)
        predicted = _predict_mechanism(m["architecture"], core_ops, decomp_cov)
        observed = m["mechanism_classification"]
        mechanism_predictions[name] = {
            "predicted": predicted,
            "observed": observed,
            "match": predicted == observed,
        }

    # --- Detection-floor-based sign consistency ---
    resolvable_models = {
        k: v for k, v in per_model.items()
        if v.get("detection_floor", {}).get("resolvable", False)
    }
    below_floor_models = {
        k: v for k, v in per_model.items()
        if not v.get("detection_floor", {}).get("resolvable", False)
    }

    resolvable_signs = [
        v["sign"] for v in resolvable_models.values() if v["sign"] != "unknown"
    ]
    all_signs = [v["sign"] for v in per_model.values() if v["sign"] != "unknown"]

    if len(resolvable_signs) == 0:
        status = "BELOW_MEASUREMENT_RESOLUTION"
    elif len(set(resolvable_signs)) == 1:
        status = "CONSISTENT_SIGN"
    else:
        status = "SIGN_DISAGREEMENT"

    # Mechanism prediction accuracy (exclude coverage_insufficient from match count)
    testable = {k: v for k, v in mechanism_predictions.items()
                if v["predicted"] != "coverage_insufficient"}
    mech_matches = sum(1 for v in testable.values() if v["match"])

    return {
        "prediction": (
            "Depth-controlled OLS r_value sign consistency among models whose "
            "|r| exceeds the Fisher-SE MDE (Bretherton autocorrelation-corrected)."
        ),
        "per_model": per_model,
        "n_total": len(per_model),
        "n_resolvable": len(resolvable_models),
        "n_below_floor": len(below_floor_models),
        "resolvable_models": list(resolvable_models.keys()),
        "below_floor_models": list(below_floor_models.keys()),
        "resolvable_signs": resolvable_signs,
        "all_signs": all_signs,
        "threshold_status": "DERIVED",
        "threshold_derivation": (
            "Fisher-SE MDE with Bretherton (1999) autocorrelation correction. "
            "MDE = tanh(1/sqrt(n_eff - 3)), n_eff = n*(1-rho1)/(1+rho1). "
            "Permutation null (2000 perms, Pearson r) as diagnostic validation (not threshold)."
        ),
        "status": status,
        "mechanism_predictions": mechanism_predictions,
        "mechanism_accuracy": f"{mech_matches}/{len(testable)}" if testable else "0/0",
    }


def render_report(summary: dict, out_path: Path) -> None:
    lines = [
        "# F5 Sign Law Analysis",
        "",
        "## Cross-Model Summary",
        "",
        "| Model | Arch | rho_total(all) | rho_total(decomp) | rho_core | rho_mlp | rho_cross | coverage | Mechanism |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["models"]:
        lines.append(
            f"| {row['model']} | {row['architecture']} | {row['rho_total_all_layers']:.4f} | "
            f"{row['rho_total_decomp_layers']:.4f} | "
            f"{row['rho_core']:.4f} | {row['rho_mlp']:.4f} | "
            f"{row['rho_cross_residual']:.4f} | {row['decomp_coverage_fraction']:.3f} | "
            f"{row['mechanism_classification']} |"
        )

    lines.extend([
        "",
        f"Gate check: **{summary['f5_gate_check_status']}** "
        f"({summary['f5_gate_check_passes']}/{summary['f5_gate_check_count']})",
        "",
        "Candidate law:",
        summary["candidate_law"],
        "",
    ])

    # Sign law test results
    slt = summary.get("sign_law_test", {})
    if slt:
        lines.extend([
            "## Formal Sign Law Test (Depth-Controlled)",
            "",
            f"**Prediction:** {slt.get('prediction', 'N/A')}",
            "",
            f"**Status:** {slt.get('status', '?')} "
            f"(threshold: {slt.get('threshold_status', 'unknown')})",
            "",
        ])

        # Detection floor table
        lines.extend([
            "### Detection Floor (Fisher-SE MDE + Bretherton autocorrelation)",
            "",
            "| Model | n | rho_1 | n_eff | MDE | |r| | Resolvable | Perm exceedance |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ])
        for name, v in slt.get("per_model", {}).items():
            df = v.get("detection_floor", {})
            if df:
                perm = df.get("permutation_diagnostic", {})
                exc = perm.get("exceedance_fraction", float("nan"))
                exc_str = f"{exc:.3f}" if not math.isnan(exc) else "N/A"
                lines.append(
                    f"| {name} | {df.get('n_raw', '?')} | "
                    f"{df.get('rho_1_used', 0):.3f} | "
                    f"{df.get('n_eff', 0):.1f} | "
                    f"{df.get('mde', 0):.3f} | "
                    f"{df.get('observed_abs_r', 0):.3f} | "
                    f"{'Yes' if df.get('resolvable') else 'No'} | "
                    f"{exc_str} |"
                )

        lines.extend([
            "",
            f"Resolvable: {slt.get('n_resolvable', 0)}/{slt.get('n_total', 0)} models",
            f"Below floor: {', '.join(slt.get('below_floor_models', [])) or 'none'}",
            "",
        ])

        # Depth model diagnostics (linear vs quadratic residualization)
        lines.extend([
            "### Depth Model Diagnostics",
            "",
            "| Model | Mode | rho_1 | n_eff | MDE | Pearson r | Spearman r | Spearman p | Resolvable |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for name, v in slt.get("per_model", {}).items():
            dm = v.get("depth_model_diagnostics", {})
            for mode in ("linear", "quadratic"):
                row = dm.get(mode, {})
                if not row:
                    continue
                lines.append(
                    f"| {name} | {mode} | "
                    f"{row.get('rho_1_used', 0):.3f} | "
                    f"{row.get('n_eff', 0):.1f} | "
                    f"{row.get('mde', 0):.3f} | "
                    f"{row.get('pearson_r', 0):+.3f} | "
                    f"{row.get('spearman_r', 0):+.3f} | "
                    f"{row.get('spearman_p', 1):.3f} | "
                    f"{'Yes' if row.get('resolvable') else 'No'} |"
                )
        lines.extend([""])

        # Per-model slopes
        lines.extend([
            "### Depth-Controlled Slopes",
            "",
            "| Model | beta_total | p-value | Sign | Resolvable |",
            "|---|---:|---:|---|---|",
        ])
        for name, v in slt.get("per_model", {}).items():
            resolvable = v.get("detection_floor", {}).get("resolvable", False)
            lines.append(
                f"| {name} | {v['beta_total']:.4f} | {v['p_value']:.4f} | "
                f"{v.get('sign', 'unknown')} | "
                f"{'Yes' if resolvable else 'No'} |"
            )
        lines.extend([
            "",
            "### Mechanism Predictions",
            "",
            "| Model | Predicted | Observed | Match |",
            "|---|---|---|---|",
        ])
        for name, v in slt.get("mechanism_predictions", {}).items():
            lines.append(
                f"| {name} | {v['predicted']} | {v['observed']} | "
                f"{'Yes' if v['match'] else '**NO**'} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="F5 sign-law analysis (post-processing).")
    parser.add_argument(
        "--input",
        default="results/entropy_curvature_operator_split",
        help="Input directory containing <model>/operator_split.json files.",
    )
    parser.add_argument(
        "--output",
        default="results/f5_sign_law",
        help="Output directory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional explicit model names to include.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_dirs = [input_dir / m for m in args.models]
    else:
        model_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())

    per_model = []
    for model_dir in model_dirs:
        source = model_dir / "operator_split.json"
        if not source.exists():
            logger.warning("Missing operator_split.json: %s", source)
            continue
        data = _load_operator_split(source)
        result = analyze_model(data)
        per_model.append(result)

        out_model_dir = output_dir / data["model_name"]
        out_model_dir.mkdir(parents=True, exist_ok=True)
        with open(out_model_dir / "f5_sign_law.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(
            "%s: status=%s, rho_total(all)=%.4f, rho_total(decomp)=%.4f, rho_core=%.4f, rho_mlp=%.4f, mech=%s",
            data["model_name"],
            result.get("status", "?"),
            result.get("theta_space", {}).get("raw_spearman", {}).get("beta_total_all_layers", {}).get("rho", float("nan")),
            result.get("theta_space", {}).get("raw_spearman", {}).get("beta_total_decomp_layers", {}).get("rho", float("nan")),
            result.get("theta_space", {}).get("raw_spearman", {}).get("beta_core", {}).get("rho", float("nan")),
            result.get("theta_space", {}).get("raw_spearman", {}).get("beta_mlp", {}).get("rho", float("nan")),
            result.get("mechanism_classification", "unknown"),
        )

    summary = build_cross_model_summary(per_model)
    with open(output_dir / "cross_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    render_report(summary, output_dir / "report.md")

    logger.info("Saved cross-model summary to %s", output_dir / "cross_model_summary.json")
    logger.info("Saved report to %s", output_dir / "report.md")


if __name__ == "__main__":
    main()
