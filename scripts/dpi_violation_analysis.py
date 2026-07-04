#!/usr/bin/env python3
"""DPI Violation Analysis: Why does the Data Processing Inequality fail?

Investigates the mechanism behind DPI violations in normalized MI trajectories.

Theory:
    For a transformer, h_{l+1} = h_l + F_l(h_l) is deterministic, so the
    unnormalized chain h_0 -> h_1 -> ... -> h_L is Markov. DPI holds for true MI.

    But Regime 5 L2-normalizes: X_tilde_l = h_l / ||h_l||. Given only X_tilde_l,
    you cannot reconstruct h_l (scale is lost). So X_tilde_{l+1} is NOT a function
    of X_tilde_l alone -- the normalized chain is not Markov. DPI need not hold.

    This script measures:
    1. DPI violations: Delta_l = I_2(X_tilde_0; X_tilde_{l+1}) - I_2(X_tilde_0; X_tilde_l)
    2. Bypass metrics: ||h_l||, ||F_l(h_l)||, residual ratio, cosine change
    3. Spearman correlations between violations and bypass metrics

Usage:
    poetry run python scripts/dpi_violation_analysis.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --output results/dpi_analysis/LFM2-350M/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)

# Reuse probes and activation collection from the main experiment
from information_bridge_experiment import (
    ALL_PROBES,
    collect_all_layer_activations,
    l2_normalize_rows,
)


def compute_bypass_metrics(layer_acts_list, backend):
    """Compute per-layer residual bypass metrics from raw activations.

    For each consecutive pair (h_l, h_{l+1}), computes:
    - ||h_l||: mean norm of hidden state
    - ||F_l(h_l)||: mean norm of residual contribution (h_{l+1} - h_l)
    - ||h_{l+1}||: mean norm of next hidden state
    - residual_ratio: ||F_l|| / ||h_l|| (relative layer contribution)
    - cosine_change: mean cosine similarity between L2-normalized h_l and h_{l+1}

    Args:
        layer_acts_list: List of [N, D] activation matrices, one per layer.
        backend: Backend for tensor operations.

    Returns:
        Dict with per-transition metric lists (L-1 entries each).
    """
    n_transitions = len(layer_acts_list) - 1
    norm_h = []
    norm_delta = []
    norm_h_next = []
    residual_ratio = []
    cosine_change = []

    for l in range(n_transitions):
        h_l = layer_acts_list[l]
        h_next = layer_acts_list[l + 1]

        # Residual contribution
        delta = h_next - h_l
        backend.eval(delta)

        # Per-probe norms, then mean over probes
        norms_l = backend.norm(h_l, axis=1)
        norms_delta = backend.norm(delta, axis=1)
        norms_next = backend.norm(h_next, axis=1)
        backend.eval(norms_l, norms_delta, norms_next)

        mean_norm_l = float(backend.to_scalar(backend.mean(norms_l, axis=0)))
        mean_norm_delta = float(backend.to_scalar(backend.mean(norms_delta, axis=0)))
        mean_norm_next = float(backend.to_scalar(backend.mean(norms_next, axis=0)))

        norm_h.append(mean_norm_l)
        norm_delta.append(mean_norm_delta)
        norm_h_next.append(mean_norm_next)
        residual_ratio.append(mean_norm_delta / mean_norm_l if mean_norm_l > 0 else 0.0)

        # Cosine similarity between L2-normalized representations
        h_l_norm = l2_normalize_rows(h_l, backend)
        h_next_norm = l2_normalize_rows(h_next, backend)
        # Per-probe dot product, then mean
        dots = backend.sum(h_l_norm * h_next_norm, axis=1)
        backend.eval(dots)
        mean_cos = float(backend.to_scalar(backend.mean(dots, axis=0)))
        cosine_change.append(mean_cos)

        if (l + 1) % 5 == 0:
            logger.info(
                "  Transition %d->%d: ||h||=%.3f, ||F||=%.3f, ratio=%.4f, cos=%.4f",
                l, l + 1, mean_norm_l, mean_norm_delta,
                residual_ratio[-1], cosine_change[-1],
            )

    return {
        "norm_h": norm_h,
        "norm_delta": norm_delta,
        "norm_h_next": norm_h_next,
        "residual_ratio": residual_ratio,
        "cosine_change": cosine_change,
    }


def compute_norm_identity_error(layer_acts_list, backend):
    """Verify ||h_{l+1}||^2 = ||h_l||^2 + ||delta||^2 + 2<h_l, delta>.

    Returns max relative error across all transitions and probes.
    """
    max_error = 0.0
    for l in range(len(layer_acts_list) - 1):
        h_l = layer_acts_list[l]
        h_next = layer_acts_list[l + 1]
        delta = h_next - h_l

        # ||h_{l+1}||^2 per probe
        norm_next_sq = backend.sum(h_next * h_next, axis=1)
        # ||h_l||^2 + ||delta||^2 + 2<h_l, delta>
        norm_l_sq = backend.sum(h_l * h_l, axis=1)
        norm_delta_sq = backend.sum(delta * delta, axis=1)
        cross_term = 2.0 * backend.sum(h_l * delta, axis=1)
        reconstructed = norm_l_sq + norm_delta_sq + cross_term

        backend.eval(norm_next_sq, reconstructed)

        # Relative error
        diff = backend.tolist(norm_next_sq - reconstructed)
        norms = backend.tolist(norm_next_sq)
        for d, n in zip(diff, norms):
            if n > 0:
                max_error = max(max_error, abs(d) / n)

    return max_error


def main():
    parser = argparse.ArgumentParser(
        description="DPI Violation Analysis",
    )
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-probes", type=int, default=200, help="Number of probes")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    logger.info("Loading model: %s", args.model)
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()
    model, tokenizer = backend.load_model(args.model)

    # --- Select probes (same as main experiment) ---
    probes = ALL_PROBES[: args.n_probes]
    logger.info("Using %d probes", len(probes))

    # --- Step 1: Collect activations ---
    logger.info("Step 1: Collecting per-layer activations...")
    t0 = time.time()
    layer_activations = collect_all_layer_activations(
        model, tokenizer, backend, probes
    )
    sorted_layers = sorted(layer_activations.keys())
    num_layers = len(sorted_layers)
    logger.info("  Collected %d layers in %.1fs", num_layers, time.time() - t0)

    # Build ordered list of raw activation matrices
    layer_acts_list = [layer_activations[idx] for idx in sorted_layers]

    # --- Step 2: Compute MI trajectory at calibrated sigma ---
    logger.info("Step 2: Computing MI trajectory (Regime 5: calibrated sigma)...")
    t0 = time.time()
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_normalized_mi_trajectory,
    )

    mi_traj, sigma_star, cal_result = compute_normalized_mi_trajectory(
        layer_acts_list, backend
    )
    logger.info(
        "  MI trajectory computed in %.1fs (sigma*=%.6f, %d values)",
        time.time() - t0, sigma_star, len(mi_traj),
    )
    if cal_result and not cal_result.is_multi_scale:
        logger.info(
            "  Calibration: sigma*=%.6f, feasible=[%.6f, %.6f]",
            cal_result.sigma_star, cal_result.feasible_lower, cal_result.feasible_upper,
        )

    # --- Step 3: Compute DPI violations ---
    logger.info("Step 3: Computing DPI violations...")
    dpi_violations = [
        mi_traj[i + 1] - mi_traj[i] for i in range(len(mi_traj) - 1)
    ]
    n_positive = sum(1 for v in dpi_violations if v > 0)
    logger.info(
        "  %d/%d transitions have positive DPI violations",
        n_positive, len(dpi_violations),
    )

    # --- Step 4: Compute bypass metrics ---
    logger.info("Step 4: Computing bypass metrics from raw activations...")
    t0 = time.time()
    bypass = compute_bypass_metrics(layer_acts_list, backend)
    logger.info("  Bypass metrics computed in %.1fs", time.time() - t0)

    # --- Step 5: Verification ---
    logger.info("Step 5: Verification checks...")

    # Telescoping: sum(violations) should equal mi_traj[-1] - mi_traj[0]
    telescoping_sum = math.fsum(dpi_violations)
    telescoping_expected = mi_traj[-1] - mi_traj[0]
    telescoping_match = abs(telescoping_sum - telescoping_expected) < 1e-10
    logger.info(
        "  Telescoping: sum=%.10f, expected=%.10f, match=%s",
        telescoping_sum, telescoping_expected, telescoping_match,
    )

    # Norm identity
    norm_error = compute_norm_identity_error(layer_acts_list, backend)
    logger.info("  Norm identity max relative error: %.2e", norm_error)

    # --- Step 6: Spearman correlations ---
    logger.info("Step 6: Computing Spearman correlations...")
    from scipy import stats

    # Alignment: compute_normalized_mi_trajectory returns L entries INCLUDING
    # self-MI at index 0: [I_2(X_0,X_0), I_2(X_0,X_1), ..., I_2(X_0,X_{L-1})].
    #
    # dpi_violations[k] = mi_traj[k+1] - mi_traj[k]
    #                    = I_2(X_0; X_{k+1}) - I_2(X_0; X_k)
    # This is the MI change across the transition from X_k to X_{k+1}.
    #
    # bypass[k] measures the transition from layer_acts[k] to layer_acts[k+1],
    # which is exactly X_k to X_{k+1}.
    #
    # Both have L-1 entries. Direct alignment, no slicing needed.
    bypass_residual = bypass["residual_ratio"]
    bypass_cosine = bypass["cosine_change"]

    correlations = {}
    if len(dpi_violations) >= 3:
        # Signed violation vs residual ratio
        r, p = stats.spearmanr(dpi_violations, bypass_residual)
        correlations["violation_vs_residual_ratio"] = {"rho": r, "p": p}
        logger.info("  rho(Delta, residual_ratio) = %.4f, p = %.2e", r, p)

        # Absolute violation vs residual ratio
        abs_violations = [abs(v) for v in dpi_violations]
        r, p = stats.spearmanr(abs_violations, bypass_residual)
        correlations["abs_violation_vs_residual_ratio"] = {"rho": r, "p": p}
        logger.info("  rho(|Delta|, residual_ratio) = %.4f, p = %.2e", r, p)

        # Violation vs cosine change
        r, p = stats.spearmanr(dpi_violations, bypass_cosine)
        correlations["violation_vs_cosine_change"] = {"rho": r, "p": p}
        logger.info("  rho(Delta, cosine_change) = %.4f, p = %.2e", r, p)
    else:
        logger.warning("  Too few transitions for correlation (%d)", len(dpi_violations))

    # --- Step 7: Save results ---
    logger.info("Step 7: Saving results...")
    results = {
        "model": args.model,
        "model_name": Path(args.model).name,
        "n_probes": len(probes),
        "n_layers": num_layers,
        "sigma_star": sigma_star,
        "calibration": {
            "sigma_star": cal_result.sigma_star if cal_result else None,
            "feasible_lower": cal_result.feasible_lower if cal_result else None,
            "feasible_upper": cal_result.feasible_upper if cal_result else None,
            "is_multi_scale": cal_result.is_multi_scale if cal_result else None,
        },
        "mi_trajectory": mi_traj,
        "dpi_violations": dpi_violations,
        "n_positive_violations": n_positive,
        "n_transitions": len(dpi_violations),
        "bypass_metrics": bypass,
        "correlations": correlations,
        "verification": {
            "telescoping_sum": telescoping_sum,
            "telescoping_expected": telescoping_expected,
            "telescoping_match": telescoping_match,
            "norm_identity_max_error": norm_error,
        },
        "theory": {
            "markov_chain_holds_unnormalized": True,
            "markov_chain_holds_normalized": False,
            "dpi_violation_mechanism": (
                "L2 normalization discards scale (||h_l||), breaking the Markov "
                "property. X_tilde_{l+1} = (h_l + F_l(h_l)) / ||h_l + F_l(h_l)|| "
                "is NOT a function of X_tilde_l = h_l / ||h_l|| alone."
            ),
        },
    }

    with open(output_dir / "dpi_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Generate report ---
    report_lines = [
        f"# DPI Violation Analysis: {Path(args.model).name}",
        "",
        f"**Model:** {args.model}",
        f"**Probes:** {len(probes)}",
        f"**Layers:** {num_layers}",
        f"**Sigma*:** {sigma_star:.6f}",
        "",
        "## Theory",
        "",
        "The unnormalized chain h_0 -> h_1 -> ... is Markov (deterministic).",
        "DPI holds for true MI: I(h_0; h_{l+1}) <= I(h_0; h_l).",
        "",
        "L2 normalization maps h_l -> h_l/||h_l||, discarding scale information.",
        "The normalized chain is NOT Markov. DPI violations are genuine.",
        "",
        "## DPI Violations",
        "",
        f"**{n_positive}/{len(dpi_violations)} transitions** have positive violations",
        "(MI increased through the layer).",
        "",
        "| Transition | Delta_l | ||h_l|| | ||F_l|| | Residual Ratio | Cosine |",
        "|------------|---------|---------|---------|---------------|--------|",
    ]
    for k in range(len(dpi_violations)):
        # dpi_violations[k] = MI change across transition from X_k to X_{k+1}
        # bypass[k] measures the same transition
        report_lines.append(
            f"| L{k}->L{k+1} | {dpi_violations[k]:+.6f} | "
            f"{bypass['norm_h'][k]:.3f} | {bypass['norm_delta'][k]:.3f} | "
            f"{bypass['residual_ratio'][k]:.4f} | {bypass['cosine_change'][k]:.4f} |"
        )

    report_lines.extend([
        "",
        "## Correlations",
        "",
        "| Metric Pair | Spearman rho | p-value | Significant (p<0.01) |",
        "|-------------|-------------|---------|---------------------|",
    ])
    for name, vals in correlations.items():
        sig = "YES" if vals["p"] < 0.01 else "no"
        report_lines.append(
            f"| {name} | {vals['rho']:.4f} | {vals['p']:.2e} | {sig} |"
        )

    report_lines.extend([
        "",
        "## Verification",
        "",
        f"- Telescoping check: sum(Delta_l) = {telescoping_sum:.10f}, "
        f"expected = {telescoping_expected:.10f}, "
        f"{'PASS' if telescoping_match else 'FAIL'}",
        f"- Norm identity max relative error: {norm_error:.2e}",
    ])

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Done. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
