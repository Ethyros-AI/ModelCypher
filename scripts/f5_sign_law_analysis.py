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


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y after removing linear effect of x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
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
    }


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
        # Candidate expectation: negative bm suppresses/attenuates total effect.
        # This check is directional, not a full theorem.
        if bm < 0:
            gate_checks.append(abs(bt) <= abs(ba))
        else:
            gate_checks.append(bt >= 0)

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
