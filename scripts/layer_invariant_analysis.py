#!/usr/bin/env python3
"""Layer-wise Invariant Analysis: What is preserved vs transformed across layers?

Tests 6 pre-registered invariant hypotheses using existing per-layer measurements
from the information bridge and DPI analysis pipelines. Pure post-processing on
JSON files — no model loading, no activation collection.

Pre-registered hypotheses:
    I1: Norm monotonicity — ||h_l|| non-decreasing with depth
    I2: ID phase-invariance — ID has lower variance within phases than across
    I3: Cosine preservation is phase-conditional — highway cos > processing cos
    I4: Residual ratio is phase-conditional — highway ratio < processing ratio
    I5: Spectral entropy separates phases — ANOVA F-test across phases
    I6: Effective rank tracks ID — exp(S_spec) correlates with TwoNN ID

Scope boundary:
    I3/I4 are falsifier tests only. This script must not be used to operationalize
    "highway implies bypass dominance" as a control or routing prior.

Usage:
    poetry run python scripts/layer_invariant_analysis.py \
        --models LFM2-350M LFM2-700M Qwen3.5-0.8B \
        --results-dir results/ \
        --output results/layer_invariants/
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_data(results_dir: Path, model_name: str) -> dict:
    """Load trajectories and DPI analysis data for a model."""
    traj_path = results_dir / "information_bridge" / model_name / "trajectories.json"
    dpi_path = results_dir / "dpi_analysis" / model_name / "dpi_analysis.json"

    with open(traj_path) as f:
        traj = json.load(f)
    with open(dpi_path) as f:
        dpi = json.load(f)

    return {"trajectories": traj, "dpi": dpi}


def permutation_test_mean_diff(
    values: list[float],
    group_a_indices: list[int],
    group_b_indices: list[int],
    n_layers: int,
) -> tuple[float, float, float]:
    """Exact or Monte Carlo permutation test for difference in means.

    Tests whether mean(values[group_a]) < mean(values[group_b]).

    Returns (observed_diff, p_value, n_permutations).
    """
    mean_a = np.mean([values[i] for i in group_a_indices])
    mean_b = np.mean([values[i] for i in group_b_indices])
    observed_diff = mean_a - mean_b  # negative means A < B

    n_a = len(group_a_indices)
    total_perms = math.comb(n_layers, n_a)

    all_indices = list(range(n_layers))

    if total_perms <= 10000:
        # Exact permutation test
        extreme_count = 0
        for perm_a in itertools.combinations(all_indices, n_a):
            perm_b = [i for i in all_indices if i not in perm_a]
            mean_pa = np.mean([values[i] for i in perm_a])
            mean_pb = np.mean([values[i] for i in perm_b])
            if mean_pa - mean_pb <= observed_diff:
                extreme_count += 1
        p_value = extreme_count / total_perms
        return float(observed_diff), p_value, total_perms
    else:
        # Monte Carlo permutation test
        rng = np.random.RandomState(42)
        n_mc = 10000
        extreme_count = 0
        for _ in range(n_mc):
            perm = rng.permutation(all_indices)
            perm_a = perm[:n_a]
            perm_b = perm[n_a:]
            mean_pa = np.mean([values[i] for i in perm_a])
            mean_pb = np.mean([values[i] for i in perm_b])
            if mean_pa - mean_pb <= observed_diff:
                extreme_count += 1
        p_value = extreme_count / n_mc
        return float(observed_diff), p_value, n_mc


def test_i1_norm_monotonicity(dpi_data: dict) -> dict:
    """I1: Norm monotonicity — ||h_l|| non-decreasing with depth."""
    bypass = dpi_data["bypass_metrics"]
    # Build full norm trajectory: norm_h covers transitions 0..L-2,
    # norm_h_next[-1] gives the final layer norm
    norm_h = bypass["norm_h"]
    norm_h_next = bypass["norm_h_next"]
    full_norms = norm_h + [norm_h_next[-1]]

    layer_indices = list(range(len(full_norms)))
    r, p = stats.spearmanr(layer_indices, full_norms)

    return {
        "hypothesis": "I1: Norm monotonicity",
        "test": "Spearman(layer_index, ||h_l||) > 0, p < 0.01",
        "spearman_r": float(r),
        "p_value": float(p),
        "n_layers": len(full_norms),
        "norm_trajectory": full_norms,
        "pass": r > 0 and p < 0.01,
        "status": "CONFIRMED" if (r > 0 and p < 0.01) else
                  "REFUTED" if (r <= 0 or p >= 0.05) else "INCONCLUSIVE",
    }


def test_i2_id_phase_invariance(traj_data: dict) -> dict:
    """I2: ID phase-invariance — ID lower variance within phases than across."""
    ids = traj_data["intrinsic_dimension"]
    phases = traj_data["phases"]

    # Group IDs by phase
    phase_groups = {}
    for i, phase in enumerate(phases):
        phase_groups.setdefault(phase, []).append(ids[i])

    # Overall CV
    all_cv = float(np.std(ids) / np.mean(ids)) if np.mean(ids) > 0 else float("inf")

    # Per-phase CV
    phase_stats = {}
    for phase, values in phase_groups.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        cv = std_val / mean_val if mean_val > 0 else float("inf")
        phase_stats[phase] = {
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
            "n": len(values),
            "cv_less_than_overall": cv < all_cv,
        }

    # ANOVA F-test (need ≥2 groups with ≥2 members)
    groups_for_anova = [v for v in phase_groups.values() if len(v) >= 2]
    anova_possible = len(groups_for_anova) >= 2
    if anova_possible:
        f_stat, p_anova = stats.f_oneway(*groups_for_anova)
    else:
        f_stat, p_anova = float("nan"), float("nan")

    # All phase CVs less than overall?
    all_cv_pass = all(s["cv_less_than_overall"] for s in phase_stats.values() if s["n"] >= 2)

    if not anova_possible:
        status = "INCONCLUSIVE"
        passed = False
        note = f"ANOVA requires ≥2 groups with ≥2 members; only {len(groups_for_anova)} qualify"
    elif p_anova < 0.01 and all_cv_pass:
        status = "CONFIRMED"
        passed = True
        note = None
    elif p_anova >= 0.05:
        status = "REFUTED"
        passed = False
        note = None
    else:
        status = "INCONCLUSIVE"
        passed = False
        note = None

    result = {
        "hypothesis": "I2: ID phase-invariance",
        "test": "ANOVA F-test of ID by phase, p < 0.01; CV(phase) < CV(all)",
        "overall_cv": all_cv,
        "phase_stats": phase_stats,
        "anova_f": float(f_stat) if not math.isnan(f_stat) else None,
        "anova_p": float(p_anova) if not math.isnan(p_anova) else None,
        "anova_possible": anova_possible,
        "all_phase_cv_less_than_overall": all_cv_pass,
        "pass": passed,
        "status": status,
    }
    if note:
        result["note"] = note
    return result


def test_i3_cosine_phase_conditional(traj_data: dict, dpi_data: dict) -> dict:
    """I3 falsifier: test whether highway cos > processing cos.

    This is hypothesis testing only, not an operational predictor.
    """
    phases = traj_data["phases"]
    cosine = dpi_data["bypass_metrics"]["cosine_change"]

    # cosine_change[k] is for transition k->k+1, assign to phase of layer k
    n_transitions = len(cosine)
    highway_idx = [i for i in range(n_transitions) if phases[i] == "highway"]
    processing_idx = [i for i in range(n_transitions) if phases[i] == "processing"]

    if not highway_idx or not processing_idx:
        return {
            "hypothesis": "I3: Cosine preservation phase-conditional",
            "pass": False,
            "status": "INCONCLUSIVE",
            "note": "Need both highway and processing phases with transitions",
        }

    mean_highway = float(np.mean([cosine[i] for i in highway_idx]))
    mean_processing = float(np.mean([cosine[i] for i in processing_idx]))

    diff, p_perm, n_perms = permutation_test_mean_diff(
        cosine, highway_idx, processing_idx, n_transitions
    )

    return {
        "hypothesis": "I3: Cosine preservation phase-conditional",
        "test": "mean(cosine|highway) > mean(cosine|processing), permutation p < 0.01",
        "mean_highway": mean_highway,
        "mean_processing": mean_processing,
        "observed_diff": diff,
        "p_value": p_perm,
        "n_permutations": n_perms,
        "n_highway": len(highway_idx),
        "n_processing": len(processing_idx),
        "pass": mean_highway > mean_processing and p_perm < 0.01,
        "status": ("CONFIRMED" if (mean_highway > mean_processing and p_perm < 0.01) else
                   "REFUTED" if mean_highway <= mean_processing else "INCONCLUSIVE"),
    }


def test_i4_residual_ratio_phase_conditional(traj_data: dict, dpi_data: dict) -> dict:
    """I4 falsifier: test whether highway ratio < processing ratio.

    This is hypothesis testing only, not an operational predictor.
    """
    phases = traj_data["phases"]
    ratios = dpi_data["bypass_metrics"]["residual_ratio"]

    n_transitions = len(ratios)
    highway_idx = [i for i in range(n_transitions) if phases[i] == "highway"]
    processing_idx = [i for i in range(n_transitions) if phases[i] == "processing"]

    if not highway_idx or not processing_idx:
        return {
            "hypothesis": "I4: Residual ratio phase-conditional",
            "pass": False,
            "status": "INCONCLUSIVE",
            "note": "Need both highway and processing phases with transitions",
        }

    mean_highway = float(np.mean([ratios[i] for i in highway_idx]))
    mean_processing = float(np.mean([ratios[i] for i in processing_idx]))

    diff, p_perm, n_perms = permutation_test_mean_diff(
        ratios, highway_idx, processing_idx, n_transitions
    )

    return {
        "hypothesis": "I4: Residual ratio phase-conditional",
        "test": "mean(ratio|highway) < mean(ratio|processing), permutation p < 0.01",
        "mean_highway": mean_highway,
        "mean_processing": mean_processing,
        "observed_diff": diff,
        "p_value": p_perm,
        "n_permutations": n_perms,
        "n_highway": len(highway_idx),
        "n_processing": len(processing_idx),
        "pass": mean_highway < mean_processing and p_perm < 0.01,
        "status": ("CONFIRMED" if (mean_highway < mean_processing and p_perm < 0.01) else
                   "REFUTED" if mean_highway >= mean_processing else "INCONCLUSIVE"),
    }


def test_i5_spectral_entropy_separates_phases(traj_data: dict) -> dict:
    """I5: Spectral entropy separates phases — ANOVA across phases."""
    s_spec = traj_data["spectral_entropy_nats"]
    phases = traj_data["phases"]

    phase_groups = {}
    for i, phase in enumerate(phases):
        phase_groups.setdefault(phase, []).append(s_spec[i])

    phase_means = {p: float(np.mean(v)) for p, v in phase_groups.items()}

    groups_for_anova = [v for v in phase_groups.values() if len(v) >= 2]
    anova_possible = len(groups_for_anova) >= 2
    if anova_possible:
        f_stat, p_anova = stats.f_oneway(*groups_for_anova)
    else:
        f_stat, p_anova = float("nan"), float("nan")

    if not anova_possible:
        status = "INCONCLUSIVE"
        passed = False
        note = f"ANOVA requires ≥2 groups with ≥2 members; only {len(groups_for_anova)} qualify"
    elif p_anova < 0.01:
        status = "CONFIRMED"
        passed = True
        note = None
    elif p_anova >= 0.05:
        status = "REFUTED"
        passed = False
        note = None
    else:
        status = "INCONCLUSIVE"
        passed = False
        note = None

    result = {
        "hypothesis": "I5: Spectral entropy separates phases",
        "test": "ANOVA F-test of S_spec by phase, p < 0.01",
        "phase_means": phase_means,
        "anova_f": float(f_stat) if not math.isnan(f_stat) else None,
        "anova_p": float(p_anova) if not math.isnan(p_anova) else None,
        "anova_possible": anova_possible,
        "pass": passed,
        "status": status,
    }
    if note:
        result["note"] = note
    return result


def test_i6_effective_rank_tracks_id(traj_data: dict) -> dict:
    """I6: Effective rank tracks ID — exp(S_spec) correlates with TwoNN ID."""
    s_spec = traj_data["spectral_entropy_nats"]
    ids = traj_data["intrinsic_dimension"]

    eff_rank = [math.exp(s) for s in s_spec]

    r, p = stats.spearmanr(eff_rank, ids)

    return {
        "hypothesis": "I6: Effective rank tracks ID",
        "test": "Spearman(exp(S_spec), ID) > 0, p < 0.01",
        "spearman_r": float(r),
        "p_value": float(p),
        "effective_rank": eff_rank,
        "pass": r > 0 and p < 0.01,
        "status": ("CONFIRMED" if (r > 0 and p < 0.01) else
                   "REFUTED" if (r <= 0 or p >= 0.05) else "INCONCLUSIVE"),
    }


def compute_phase_summary(traj_data: dict, dpi_data: dict) -> dict:
    """Compute phase-conditional summary statistics for all quantities."""
    phases = traj_data["phases"]
    s_spec = traj_data["spectral_entropy_nats"]
    ids = traj_data["intrinsic_dimension"]
    curvature = traj_data["curvature_radians"]
    c_ex = traj_data["curvature_excess_nats"]
    bypass = dpi_data["bypass_metrics"]

    summary = {}
    unique_phases = sorted(set(phases))

    for phase in unique_phases:
        layer_indices = [i for i, p in enumerate(phases) if p == phase]
        transition_indices = [i for i in layer_indices if i < len(bypass["cosine_change"])]

        phase_data = {
            "layers": layer_indices,
            "n_layers": len(layer_indices),
        }

        # Per-layer quantities
        for name, values in [
            ("spectral_entropy_nats", s_spec),
            ("intrinsic_dimension", ids),
            ("curvature_radians", curvature),
            ("curvature_excess_nats", c_ex),
        ]:
            phase_vals = [values[i] for i in layer_indices]
            mean_val = float(np.mean(phase_vals))
            phase_data[name] = {
                "mean": mean_val,
                "std": float(np.std(phase_vals)),
                "cv": float(np.std(phase_vals) / mean_val) if mean_val > 0 else 0.0,
                "min": float(np.min(phase_vals)),
                "max": float(np.max(phase_vals)),
            }

        # Per-transition quantities
        for name, values in [
            ("residual_ratio", bypass["residual_ratio"]),
            ("cosine_change", bypass["cosine_change"]),
            ("norm_h", bypass["norm_h"]),
            ("norm_delta", bypass["norm_delta"]),
        ]:
            phase_vals = [values[i] for i in transition_indices]
            if phase_vals:
                mean_val = float(np.mean(phase_vals))
                phase_data[name] = {
                    "mean": mean_val,
                    "std": float(np.std(phase_vals)),
                    "cv": float(np.std(phase_vals) / mean_val) if mean_val > 0 else 0.0,
                    "min": float(np.min(phase_vals)),
                    "max": float(np.max(phase_vals)),
                }

        summary[phase] = phase_data

    return summary


def analyze_model(results_dir: Path, model_name: str) -> dict:
    """Run all invariant tests for a single model."""
    logger.info("Analyzing model: %s", model_name)
    data = load_model_data(results_dir, model_name)
    traj = data["trajectories"]
    dpi = data["dpi"]

    results = {
        "model": model_name,
        "n_layers": len(traj["layers"]),
        "phases": traj["phases"],
    }

    # Run all hypothesis tests
    results["I1"] = test_i1_norm_monotonicity(dpi)
    logger.info("  I1 (Norm monotonicity): %s (r=%.4f, p=%.2e)",
                results["I1"]["status"], results["I1"]["spearman_r"], results["I1"]["p_value"])

    results["I2"] = test_i2_id_phase_invariance(traj)
    i2 = results["I2"]
    if i2.get("anova_f") is not None:
        logger.info("  I2 (ID phase-invariance): %s (F=%.2f, p=%.2e)",
                    i2["status"], i2["anova_f"], i2["anova_p"])
    else:
        logger.info("  I2 (ID phase-invariance): %s (%s)",
                    i2["status"], i2.get("note", ""))

    results["I3"] = test_i3_cosine_phase_conditional(traj, dpi)
    logger.info("  I3 (Cosine phase-conditional): %s (p=%.2e)",
                results["I3"]["status"], results["I3"].get("p_value", 1.0))

    results["I4"] = test_i4_residual_ratio_phase_conditional(traj, dpi)
    logger.info("  I4 (Residual ratio phase-conditional): %s (p=%.2e)",
                results["I4"]["status"], results["I4"].get("p_value", 1.0))

    results["I5"] = test_i5_spectral_entropy_separates_phases(traj)
    i5 = results["I5"]
    if i5.get("anova_f") is not None:
        logger.info("  I5 (S_spec separates phases): %s (F=%.2f, p=%.2e)",
                    i5["status"], i5["anova_f"], i5["anova_p"])
    else:
        logger.info("  I5 (S_spec separates phases): %s (%s)",
                    i5["status"], i5.get("note", ""))

    results["I6"] = test_i6_effective_rank_tracks_id(traj)
    logger.info("  I6 (Effective rank tracks ID): %s (r=%.4f, p=%.2e)",
                results["I6"]["status"], results["I6"]["spearman_r"], results["I6"]["p_value"])

    # Phase-conditional summary
    results["phase_summary"] = compute_phase_summary(traj, dpi)

    return results


def generate_report(results: dict, output_path: Path) -> None:
    """Generate markdown report for a single model."""
    lines = [
        f"# Layer Invariant Analysis: {results['model']}",
        "",
        f"**Layers:** {results['n_layers']}",
        f"**Phases:** {results['phases']}",
        "",
        "## Hypothesis Tests",
        "",
        "| # | Hypothesis | Status | Evidence |",
        "|---|-----------|--------|----------|",
    ]

    for key in ["I1", "I2", "I3", "I4", "I5", "I6"]:
        h = results[key]
        if "spearman_r" in h:
            evidence = f"r={h['spearman_r']:.4f}, p={h['p_value']:.2e}"
        elif "anova_f" in h and h.get("anova_f") is not None:
            evidence = f"F={h['anova_f']:.2f}, p={h['anova_p']:.2e}"
        elif "observed_diff" in h:
            evidence = f"diff={h['observed_diff']:.4f}, p={h['p_value']:.2e}"
        else:
            evidence = h.get("note", "insufficient data")
        lines.append(f"| {key} | {h['hypothesis']} | **{h['status']}** | {evidence} |")

    passed = sum(1 for k in ["I1", "I2", "I3", "I4", "I5", "I6"] if results[k]["pass"])
    lines.extend(["", f"**{passed}/6 hypotheses confirmed.**"])

    # Phase summary table
    lines.extend(["", "## Phase-Conditional Summary", ""])
    ps = results["phase_summary"]
    for phase, data in sorted(ps.items()):
        lines.append(f"### Phase: {phase} ({data['n_layers']} layers: {data['layers']})")
        lines.append("")
        lines.append("| Quantity | Mean | Std | CV |")
        lines.append("|----------|------|-----|-----|")
        for qty in ["spectral_entropy_nats", "intrinsic_dimension", "curvature_radians",
                     "curvature_excess_nats", "residual_ratio", "cosine_change",
                     "norm_h", "norm_delta"]:
            if qty in data:
                d = data[qty]
                lines.append(f"| {qty} | {d['mean']:.4f} | {d['std']:.4f} | {d['cv']:.4f} |")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Layer-wise Invariant Analysis")
    parser.add_argument("--models", nargs="+", required=True, help="Model names")
    parser.add_argument("--results-dir", required=True, help="Base results directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    all_results = {}

    for model_name in args.models:
        model_output = output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        results = analyze_model(results_dir, model_name)
        all_results[model_name] = results

        # Save per-model JSON
        with open(model_output / "invariant_analysis.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        # Generate per-model report
        generate_report(results, model_output / "report.md")
        logger.info("  Saved to %s", model_output)

    # Cross-model summary
    logger.info("=" * 60)
    logger.info("CROSS-MODEL SUMMARY")
    logger.info("=" * 60)

    cross_model = {}
    for key in ["I1", "I2", "I3", "I4", "I5", "I6"]:
        per_model = {}
        for model_name in args.models:
            r = all_results[model_name][key]
            per_model[model_name] = {
                "status": r["status"],
                "pass": r["pass"],
            }

        n_confirmed = sum(1 for m in per_model.values() if m["pass"])
        n_refuted = sum(1 for m in per_model.values() if m["status"] == "REFUTED")
        n_models = len(per_model)

        if n_confirmed == n_models:
            verdict = "VALIDATED"
        elif n_confirmed >= 2:
            verdict = "EMPIRICAL"
        elif n_confirmed == 1:
            verdict = "EMPIRICAL"
        elif n_refuted > 0:
            verdict = "REFUTED"
        else:
            # All INCONCLUSIVE — couldn't test on any model
            verdict = "INCONCLUSIVE"

        cross_model[key] = {
            "hypothesis": all_results[args.models[0]][key]["hypothesis"],
            "per_model": per_model,
            "n_confirmed": n_confirmed,
            "n_models": n_models,
            "verdict": verdict,
        }
        logger.info("  %s: %s (%d/%d)", key, verdict, n_confirmed, n_models)

    summary_path = output_dir / "cross_model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(cross_model, f, indent=2, cls=NumpyEncoder)
    logger.info("Cross-model summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
