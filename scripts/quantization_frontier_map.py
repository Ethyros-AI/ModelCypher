#!/usr/bin/env python3
"""Quantization Frontier Map: Join Weyl + CKA + Tikhonov data.

Joins existing experiment results to build a per-model frontier table:
  arch, params, bits, Weyl safety%, baseline CKA, post-Tikhonov CKA,
  recovery ratio, PPL delta, degeneration delta.

Also computes:
  - Spearman correlations where per-layer overlap exists
    (error_over_gap_ratio vs correction_fraction)
  - Recovery ratio: actual_gain / (1 - baseline_CKA) — fraction of
    theoretical gap closed by Tikhonov correction.

Data sources (existing JSON — no model loading required):
  - results/weyl_quantization_validation/20260226T015425Z/
  - results/closedform_sequential_correction/ (3 model subdirectories)

Usage:
    poetry run python scripts/quantization_frontier_map.py
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("frontier_map")

# ── Data paths ───────────────────────────────────────────────────────────

WEYL_PATH = Path(
    "results/weyl_quantization_validation/20260226T015425Z/"
    "weyl_quantization_validation.json"
)

CORRECTION_PATHS = {
    "Qwen3-1.7B": Path(
        "results/closedform_sequential_correction/20260227T173057Z/"
        "closedform_correction.json"
    ),
    "Qwen3-8B": Path(
        "results/closedform_sequential_correction/qwen3-8b/20260227T182433Z/"
        "closedform_correction.json"
    ),
    "Llama-3.2-3B": Path(
        "results/closedform_sequential_correction/llama-3.2-3b/20260227T203307Z/"
        "closedform_correction.json"
    ),
}

OUTPUT_DIR = Path("results/quantization_frontier")


# ── Spearman rank correlation (no scipy dependency) ──────────────────────


def _rank(values: list[float]) -> list[float]:
    """Assign fractional ranks to values (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-15:
        return float("nan")
    return cov / denom


# ── Main ─────────────────────────────────────────────────────────────────


def _load_weyl_data() -> dict[str, dict]:
    """Load Weyl validation, keyed by model name."""
    if not WEYL_PATH.exists():
        logger.warning("Weyl data not found: %s", WEYL_PATH)
        return {}

    with open(WEYL_PATH) as f:
        data = json.load(f)

    result = {}
    for pair in data.get("pairs", []):
        fp_id = pair["fp_model_id"]
        # Normalize model name
        if "1.7B" in fp_id:
            name = "Qwen3-1.7B"
        elif "8B" in fp_id:
            name = "Qwen3-8B"
        else:
            name = fp_id

        n_layers = pair["n_layers"]
        n_safe = pair.get("n_weyl_safe", 0)
        per_layer = pair.get("per_layer", [])

        # Aggregate Weyl metrics
        gap_ratios = [l["error_over_gap_ratio"] for l in per_layer]
        safe_pct = (n_safe / n_layers * 100) if n_layers > 0 else 0.0

        result[name] = {
            "fp_model": fp_id,
            "q_model": pair["q_model_id"],
            "bits": 8,  # All Weyl runs are 8-bit
            "n_projections": n_layers,
            "weyl_safe_pct": round(safe_pct, 1),
            "mean_error_over_gap": sum(gap_ratios) / len(gap_ratios) if gap_ratios else 0.0,
            "max_error_over_gap": max(gap_ratios) if gap_ratios else 0.0,
            "per_layer": per_layer,
        }

    return result


def _load_correction_data() -> dict[str, dict]:
    """Load Tikhonov correction results, keyed by model name."""
    result = {}
    for name, path in CORRECTION_PATHS.items():
        if not path.exists():
            logger.warning("Correction data not found for %s: %s", name, path)
            continue

        with open(path) as f:
            data = json.load(f)

        baseline = data.get("baseline", {})
        post = data.get("post_correction", {})
        deltas = data.get("deltas", {})
        correction = data.get("correction", {})

        # Determine architecture from model name
        if "Llama" in name:
            arch = "Llama"
        elif "Qwen" in name:
            arch = "Qwen"
        else:
            arch = "unknown"

        # Determine params from model name
        params = name  # e.g. "Qwen3-1.7B"

        baseline_cka_mean = baseline.get("cka", {}).get("mean_cka", 0.0)
        post_cka_mean = post.get("cka", {}).get("mean_cka", 0.0)
        cka_gain = post_cka_mean - baseline_cka_mean

        # Recovery ratio: actual_gain / (1 - baseline_CKA)
        # How much of the theoretical gap (from perfect CKA=1.0) was closed
        gap = 1.0 - baseline_cka_mean
        recovery_ratio = cka_gain / gap if gap > 1e-10 else float("nan")

        result[name] = {
            "arch": arch,
            "params": params,
            "bits": 4,  # All correction runs are 4-bit
            "baseline_cka_mean": baseline_cka_mean,
            "baseline_cka_min": baseline.get("cka", {}).get("min_cka", 0.0),
            "post_cka_mean": post_cka_mean,
            "post_cka_min": post.get("cka", {}).get("min_cka", 0.0),
            "cka_gain_mean": deltas.get("cka_mean", 0.0),
            "cka_gain_min": deltas.get("cka_min", 0.0),
            "ppl_delta": deltas.get("ppl", 0.0),
            "degen_delta": deltas.get("max_4gram_repeat", 0.0),
            "recovery_ratio": recovery_ratio,
            "n_layers": correction.get("n_layers", 0),
            "n_projections_corrected": correction.get("n_projections_corrected", 0),
            "per_layer": correction.get("per_layer", []),
            "baseline_ppl": baseline.get("ppl", {}).get("perplexity", 0.0)
            if isinstance(baseline.get("ppl"), dict)
            else baseline.get("ppl", 0.0),
            "post_ppl": post.get("ppl", {}).get("perplexity", 0.0)
            if isinstance(post.get("ppl"), dict)
            else post.get("ppl", 0.0),
        }

    return result


def _compute_per_layer_correlations(
    weyl_data: dict[str, dict],
    correction_data: dict[str, dict],
) -> dict[str, dict]:
    """Compute Spearman correlations where per-layer data overlaps.

    Weyl data is per-projection (7 projections/layer for Qwen).
    Correction data is per-layer (one entry per transformer layer).
    Join by layer index: for each layer, take max error_over_gap_ratio
    across its projections and correlate with correction_fraction.
    """
    correlations = {}

    for model_name in correction_data:
        if model_name not in weyl_data:
            logger.info(
                "No Weyl data for %s (different bit width) — skipping correlation",
                model_name,
            )
            continue

        weyl_layers = weyl_data[model_name]["per_layer"]
        corr_layers = correction_data[model_name]["per_layer"]

        # Build per-layer-index map from Weyl data
        # layer_key format: "model.layers.{idx}.{block}.{proj}.weight"
        layer_gap_ratios: dict[int, list[float]] = {}
        for wl in weyl_layers:
            key = wl.get("layer_key", "")
            parts = key.split(".")
            # Find the layer index
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            layer_gap_ratios.setdefault(layer_idx, []).append(
                wl["error_over_gap_ratio"]
            )

        # For each correction layer, find matching Weyl data
        gap_ratios = []
        corr_fractions = []
        for cl in corr_layers:
            idx = cl["layer_idx"]
            if idx not in layer_gap_ratios:
                continue
            # Take max error_over_gap across projections for this layer
            max_gap = max(layer_gap_ratios[idx])
            cf = cl.get("correction_fraction", 0.0)
            gap_ratios.append(max_gap)
            corr_fractions.append(cf)

        if len(gap_ratios) >= 3:
            rho = spearman_rho(gap_ratios, corr_fractions)
            correlations[model_name] = {
                "spearman_rho": rho,
                "n_layers_matched": len(gap_ratios),
                "note": "error_over_gap_ratio (Weyl 8-bit) vs correction_fraction (Tikhonov 4-bit)",
            }
            logger.info(
                "%s: Spearman(gap_ratio, correction_fraction) = %.4f (n=%d)",
                model_name,
                rho,
                len(gap_ratios),
            )
        else:
            logger.warning(
                "%s: insufficient overlap for correlation (n=%d)",
                model_name,
                len(gap_ratios),
            )

    return correlations


def main():
    logger.info("Quantization Frontier Map")
    logger.info("=" * 60)

    weyl_data = _load_weyl_data()
    correction_data = _load_correction_data()

    if not correction_data:
        logger.error("No correction data found. Nothing to analyze.")
        return

    # ── Frontier table ───────────────────────────────────────────────────
    frontier_table = []
    for name, cd in correction_data.items():
        wd = weyl_data.get(name, {})
        entry = {
            "model": name,
            "arch": cd["arch"],
            "params": cd["params"],
            "correction_bits": cd["bits"],
            "weyl_bits": wd.get("bits", None),
            "weyl_safe_pct": wd.get("weyl_safe_pct", None),
            "baseline_cka_mean": round(cd["baseline_cka_mean"], 6),
            "post_cka_mean": round(cd["post_cka_mean"], 6),
            "cka_gain_mean": round(cd["cka_gain_mean"], 6),
            "recovery_ratio": round(cd["recovery_ratio"], 4)
            if not math.isnan(cd["recovery_ratio"])
            else None,
            "ppl_delta": round(cd["ppl_delta"], 4),
            "degen_delta": round(cd["degen_delta"], 4),
            "n_layers": cd["n_layers"],
            "n_projections_corrected": cd["n_projections_corrected"],
        }
        frontier_table.append(entry)

    # Sort by CKA gain descending
    frontier_table.sort(key=lambda x: x.get("cka_gain_mean", 0), reverse=True)

    # ── Correlations ─────────────────────────────────────────────────────
    correlations = _compute_per_layer_correlations(weyl_data, correction_data)

    # ── Output ───────────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": "quantization_frontier_map",
        "timestamp": timestamp,
        "data_sources": {
            "weyl": str(WEYL_PATH),
            "correction": {k: str(v) for k, v in CORRECTION_PATHS.items()},
        },
        "frontier_table": frontier_table,
        "correlations": correlations,
        "summary": {
            "n_models": len(frontier_table),
            "architectures": list(set(e["arch"] for e in frontier_table)),
            "mean_recovery_ratio": round(
                sum(
                    e["recovery_ratio"]
                    for e in frontier_table
                    if e["recovery_ratio"] is not None
                )
                / max(
                    sum(1 for e in frontier_table if e["recovery_ratio"] is not None), 1
                ),
                4,
            ),
            "all_ppl_improved": all(e["ppl_delta"] < 0 for e in frontier_table),
            "all_degen_improved": all(e["degen_delta"] <= 0 for e in frontier_table),
        },
    }

    output_path = run_dir / "quantization_frontier.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────
    logger.info("")
    logger.info("FRONTIER TABLE")
    logger.info("=" * 90)
    logger.info(
        "%-14s %-6s %-5s %-10s %-10s %-10s %-10s %-8s %-8s",
        "Model",
        "Arch",
        "Bits",
        "Base CKA",
        "Post CKA",
        "CKA Gain",
        "Recovery",
        "PPL Δ",
        "Degen Δ",
    )
    logger.info("-" * 90)
    for e in frontier_table:
        recovery_str = f"{e['recovery_ratio']:.4f}" if e["recovery_ratio"] else "N/A"
        logger.info(
            "%-14s %-6s %-5d %-10.6f %-10.6f %-10.6f %-10s %-8.4f %-8.4f",
            e["model"],
            e["arch"],
            e["correction_bits"],
            e["baseline_cka_mean"],
            e["post_cka_mean"],
            e["cka_gain_mean"],
            recovery_str,
            e["ppl_delta"],
            e["degen_delta"],
        )

    if correlations:
        logger.info("")
        logger.info("SPEARMAN CORRELATIONS (Weyl gap_ratio vs Tikhonov correction)")
        logger.info("-" * 60)
        for name, corr in correlations.items():
            logger.info(
                "  %-14s rho=%.4f  (n=%d layers)",
                name,
                corr["spearman_rho"],
                corr["n_layers_matched"],
            )

    logger.info("")
    logger.info("Results saved: %s", output_path)


if __name__ == "__main__":
    main()
