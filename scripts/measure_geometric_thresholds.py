#!/usr/bin/env python3
"""Measure geometric quantities on real models to derive thresholds.

No thresholds are assumed. This is measurement only. The geometry tells us
where the natural breaks are — we don't import them from statistics papers.

Measures:
A. Eigenvalue gap structure at variance boundary (null_space.py * 0.99)
B. Geodesic deviation profile curvature (geodesic_trajectory_service.py * 0.25)
C. Variance concentration distribution (variance_concentration.py = 0.70)
D. GW inner Sinkhorn convergence iterations (gromov_wasserstein.py = 500)
E. Divergence onset relative spread (geodesic_trajectory_service.py = 0.1)

Usage:
    poetry run python scripts/measure_geometric_thresholds.py \
        --models 350M 700M \
        --output /Volumes/CodeCypher/models/experiments/geometric-thresholds/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = {
    "350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "1.2B": "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
}

# Diverse prompts for activation collection — covers different geometric regimes
PROBE_PROMPTS = {
    "arithmetic": [
        "What is 7 times 8?",
        "Calculate 145 plus 278.",
        "What is 1000 divided by 25?",
        "If I have 3 groups of 12, how many total?",
    ],
    "factual": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What year did World War II end?",
        "What is the chemical symbol for gold?",
    ],
    "reasoning": [
        "If all cats are animals and some animals are pets, what can we conclude about cats?",
        "A train leaves at 9am going 60mph. Another leaves at 10am going 80mph. When do they meet?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
        "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
    ],
    "linguistic": [
        "Explain the difference between 'affect' and 'effect'.",
        "What does the idiom 'break the ice' mean?",
        "Rewrite this sentence in passive voice: The cat chased the mouse.",
        "What is the past participle of 'swim'?",
    ],
}


def _forward_to_layer(embed_tokens, layers, input_ids, target_layer, backend):
    """Forward pass collecting hidden state at target layer."""
    hidden = embed_tokens(input_ids)
    backend.eval(hidden)
    for i, layer in enumerate(layers):
        is_attn = getattr(layer, "is_attention_layer", True)
        mask = "causal" if is_attn else None
        hidden = layer(hidden, mask=mask)
        backend.eval(hidden)
        if i == target_layer:
            break
    return hidden


def _collect_layer_activations(model, tokenizer, backbone, backend, texts, target_layer):
    """Collect mean-pooled activations at target_layer for a list of texts.

    Returns backend array [n_texts, hidden_dim].
    """
    embed_tokens, layers, norm = backbone
    vecs = []
    for text in texts:
        tokens = tokenizer.encode(text)
        input_ids = backend.array([tokens])
        hidden = _forward_to_layer(embed_tokens, layers, input_ids, target_layer, backend)
        h = backend.mean(hidden, axis=1)  # [1, hidden_dim]
        backend.eval(h)
        vecs.append(h)
    stacked = backend.concatenate(vecs, axis=0)
    backend.eval(stacked)
    return stacked


def _all_probe_texts():
    """Flatten all probe prompts into a single list."""
    texts = []
    for category_texts in PROBE_PROMPTS.values():
        texts.extend(category_texts)
    return texts


# ---------------------------------------------------------------------------
# Measurement A: Eigenvalue gap structure at cumulative variance boundary
# ---------------------------------------------------------------------------

def measure_eigenvalue_gaps(model, tokenizer, backbone, backend, n_layers):
    """For each layer, compute eigenvalue spectrum of activation covariance.

    Records:
    - Full eigenvalue spectrum (descending)
    - Consecutive gap ratios λ_i / λ_{i+1}
    - Cumulative variance fraction at each eigenvalue
    - Index where cumulative variance crosses 0.9997 (the 1 - sqrt(eps) threshold)
    - Gap ratio at that boundary index
    """
    logger.info("=== Measurement A: Eigenvalue gap structure ===")
    texts = _all_probe_texts()
    results = {}

    eps_f32 = math.ldexp(1.0, -23)
    target_fraction = 1.0 - math.sqrt(eps_f32)  # ~0.9997

    for layer_idx in range(n_layers):
        logger.info("  Layer %d/%d", layer_idx, n_layers - 1)
        acts = _collect_layer_activations(
            model, tokenizer, backbone, backend, texts, layer_idx,
        )
        n_samples = int(acts.shape[0])
        hidden_dim = int(acts.shape[1])

        # Center activations
        centered = acts - backend.mean(acts, axis=0, keepdims=True)
        backend.eval(centered)

        # Covariance: C = A^T @ A / n
        C = backend.matmul(backend.transpose(centered), centered)
        C = C / float(n_samples)
        backend.eval(C)

        # Eigendecomposition (ascending)
        eigvals = backend.eigvalsh(C)
        backend.eval(eigvals)

        # Reverse to descending
        n_eigs = int(eigvals.shape[0])
        reverse_idx = backend.arange(n_eigs - 1, -1, -1)
        eigvals = backend.take(eigvals, reverse_idx, axis=0)
        backend.eval(eigvals)

        # Clamp negatives
        eigvals = backend.maximum(eigvals, backend.zeros_like(eigvals))
        backend.eval(eigvals)

        # Extract to list
        eig_list = backend.tolist(eigvals)

        # Total variance
        total_var = sum(eig_list)
        if total_var <= 0:
            results[layer_idx] = {
                "n_samples": n_samples,
                "hidden_dim": hidden_dim,
                "total_variance": 0.0,
                "note": "zero_variance",
            }
            continue

        # Cumulative variance fractions
        cumsum = 0.0
        cum_fractions = []
        for e in eig_list:
            cumsum += e
            cum_fractions.append(cumsum / total_var)

        # Find boundary index where cumulative crosses target
        boundary_idx = None
        for i, cf in enumerate(cum_fractions):
            if cf >= target_fraction:
                boundary_idx = i
                break
        if boundary_idx is None:
            boundary_idx = len(eig_list) - 1

        # Consecutive gap ratios: λ_i / λ_{i+1}
        gap_ratios = []
        for i in range(len(eig_list) - 1):
            if eig_list[i + 1] > 0:
                gap_ratios.append(eig_list[i] / eig_list[i + 1])
            else:
                gap_ratios.append(float("inf"))

        # Gap at boundary
        boundary_gap = gap_ratios[boundary_idx] if boundary_idx < len(gap_ratios) else float("inf")

        # Find largest gap in spectrum
        if gap_ratios:
            finite_gaps = [(i, g) for i, g in enumerate(gap_ratios) if math.isfinite(g)]
            if finite_gaps:
                max_gap_idx, max_gap_val = max(finite_gaps, key=lambda x: x[1])
            else:
                max_gap_idx, max_gap_val = 0, float("inf")
        else:
            max_gap_idx, max_gap_val = 0, 0.0

        # Nearest spectral gap to boundary (within ±5 indices)
        nearby_range = range(
            max(0, boundary_idx - 5),
            min(len(gap_ratios), boundary_idx + 6),
        )
        nearby_gaps = [(i, gap_ratios[i]) for i in nearby_range if math.isfinite(gap_ratios[i])]
        if nearby_gaps:
            nearest_gap_idx, nearest_gap_val = max(nearby_gaps, key=lambda x: x[1])
        else:
            nearest_gap_idx, nearest_gap_val = boundary_idx, 0.0

        results[layer_idx] = {
            "n_samples": n_samples,
            "hidden_dim": hidden_dim,
            "total_variance": total_var,
            "n_eigenvalues": len(eig_list),
            "boundary_idx": boundary_idx,
            "boundary_cum_fraction": cum_fractions[boundary_idx],
            "boundary_eigenvalue": eig_list[boundary_idx],
            "boundary_gap_ratio": boundary_gap if math.isfinite(boundary_gap) else "inf",
            "max_gap_idx": max_gap_idx,
            "max_gap_ratio": max_gap_val if math.isfinite(max_gap_val) else "inf",
            "nearest_gap_to_boundary_idx": nearest_gap_idx,
            "nearest_gap_to_boundary_ratio": nearest_gap_val if math.isfinite(nearest_gap_val) else "inf",
            "top_10_eigenvalues": eig_list[:10],
            "boundary_region_eigenvalues": eig_list[max(0, boundary_idx - 3):boundary_idx + 4],
            "boundary_region_gaps": [
                gap_ratios[i] if math.isfinite(gap_ratios[i]) else "inf"
                for i in range(max(0, boundary_idx - 3), min(len(gap_ratios), boundary_idx + 3))
            ],
        }

        logger.info(
            "    boundary_idx=%d, boundary_gap=%.2f, max_gap=%.2f@%d, nearest_gap=%.2f@%d",
            boundary_idx,
            boundary_gap if math.isfinite(boundary_gap) else -1,
            max_gap_val if math.isfinite(max_gap_val) else -1,
            max_gap_idx,
            nearest_gap_val if math.isfinite(nearest_gap_val) else -1,
            nearest_gap_idx,
        )

    return results


# ---------------------------------------------------------------------------
# Measurement B: Geodesic deviation profile — second derivative analysis
# ---------------------------------------------------------------------------

def measure_geodesic_profiles(model, tokenizer, backend, activation_provider):
    """Compute per-layer geodesic deviation and its second derivative.

    The inflection point is where d²(deviation)/d(layer)² changes sign —
    a geometric property of the curvature profile, not a fraction of the peak.
    """
    logger.info("=== Measurement B: Geodesic deviation profiles ===")

    from modelcypher.core.use_cases.geodesic_trajectory_service import GeodesicTrajectoryService

    service = GeodesicTrajectoryService(
        backend=backend,
        activation_provider=activation_provider,
    )

    results = {}
    for category, prompts in PROBE_PROMPTS.items():
        category_results = []
        for prompt in prompts:
            try:
                profile = service.measure_layer_profile(model, tokenizer, prompt)
            except Exception as e:
                logger.warning("  Skipping prompt '%s': %s", prompt[:40], e)
                continue

            # Extract deviation profile
            deviations = [lp.mean_deviation for lp in profile.layer_profiles]
            layers = [lp.layer for lp in profile.layer_profiles]

            if len(deviations) < 3:
                continue

            # Discrete second derivative: d²dev/dlayer²
            second_deriv = []
            for i in range(1, len(deviations) - 1):
                d2 = deviations[i + 1] - 2 * deviations[i] + deviations[i - 1]
                second_deriv.append(d2)

            # Find inflection: where second derivative changes sign (+ to -)
            inflection_layer = None
            for i in range(len(second_deriv) - 1):
                if second_deriv[i] > 0 and second_deriv[i + 1] <= 0:
                    inflection_layer = layers[i + 1]  # +1 because second_deriv starts at layer 1
                    break

            # Peak
            peak_dev = max(deviations)
            peak_layer = layers[deviations.index(peak_dev)]

            # Current heuristic result for comparison
            threshold_025 = peak_dev * 0.25
            heuristic_inflection = None
            for i, d in enumerate(deviations):
                if d > threshold_025:
                    heuristic_inflection = layers[i]
                    break

            category_results.append({
                "prompt": prompt[:60],
                "layers": layers,
                "deviations": deviations,
                "second_derivative": second_deriv,
                "second_derivative_layers": layers[1:-1],
                "peak_layer": peak_layer,
                "peak_deviation": peak_dev,
                "inflection_layer_geometric": inflection_layer,
                "inflection_layer_heuristic_025": heuristic_inflection,
            })

            logger.info(
                "  [%s] '%s': peak=%.4f@L%d, inflection_geom=L%s, inflection_025=L%s",
                category, prompt[:30], peak_dev, peak_layer,
                inflection_layer, heuristic_inflection,
            )

        results[category] = category_results

    return results


# ---------------------------------------------------------------------------
# Measurement C: Variance concentration distribution across layers
# ---------------------------------------------------------------------------

def measure_variance_concentration(model, tokenizer, backbone, backend, n_layers):
    """Compute var_top1 for every layer. Look for natural gaps in the distribution."""
    logger.info("=== Measurement C: Variance concentration distribution ===")

    from modelcypher.core.domain.geometry.variance_concentration import (
        compute_variance_concentration,
    )

    texts = _all_probe_texts()
    results = {}

    for layer_idx in range(n_layers):
        logger.info("  Layer %d/%d", layer_idx, n_layers - 1)
        acts = _collect_layer_activations(
            model, tokenizer, backbone, backend, texts, layer_idx,
        )

        vc = compute_variance_concentration(acts, backend=backend)

        results[layer_idx] = {
            "var_top1": vc.var_top1,
            "var_top_k": vc.var_top_k,
            "effective_rank": vc.effective_rank,
            "total_variance": vc.total_variance,
            "n_singular_values": vc.n_singular_values,
        }

        logger.info(
            "    var_top1=%.4f, effective_rank=%.1f",
            vc.var_top1, vc.effective_rank,
        )

    # Sort var_top1 values and find gaps
    var_top1_sorted = sorted(
        [(layer_idx, r["var_top1"]) for layer_idx, r in results.items()],
        key=lambda x: x[1],
    )
    var_top1_values = [v for _, v in var_top1_sorted]

    # Consecutive gaps in sorted var_top1
    var_gaps = []
    for i in range(len(var_top1_values) - 1):
        gap = var_top1_values[i + 1] - var_top1_values[i]
        var_gaps.append({
            "between_layers": (var_top1_sorted[i][0], var_top1_sorted[i + 1][0]),
            "between_values": (var_top1_values[i], var_top1_values[i + 1]),
            "gap": gap,
        })

    # Largest gap = natural break in the distribution
    if var_gaps:
        max_gap = max(var_gaps, key=lambda x: x["gap"])
    else:
        max_gap = None

    return {
        "per_layer": results,
        "sorted_var_top1": var_top1_sorted,
        "var_top1_gaps": var_gaps,
        "largest_gap": max_gap,
    }


# ---------------------------------------------------------------------------
# Measurement D: GW inner Sinkhorn convergence behavior
# ---------------------------------------------------------------------------

def measure_gw_sinkhorn_convergence(backend):
    """Measure actual Sinkhorn iteration counts on GW inner loop.

    Creates synthetic distance matrices at various sizes and runs GW,
    instrumenting the inner Sinkhorn to log actual convergence iterations.
    """
    logger.info("=== Measurement D: GW inner Sinkhorn convergence ===")

    from modelcypher.core.domain.geometry.gromov_wasserstein import GromovWassersteinDistance
    from modelcypher.core.domain.geometry.optimal_transport import SinkhornSolver

    # Monkey-patch solve_linear_ot to log iteration counts
    original_solve = SinkhornSolver.solve_linear_ot
    iteration_log = []

    def instrumented_solve(self, cost, p, q, epsilon, max_iterations=None, threshold=None):
        """Instrumented version that logs actual iteration count."""
        from modelcypher.core.domain.geometry.numerical_stability import (
            division_epsilon,
            is_finite,
            regularization_epsilon,
            tiny_value,
            log_scalar,
        )
        b = self._backend
        n = int(cost.shape[0])
        m = int(cost.shape[1])

        if n == 0 or m == 0:
            return b.zeros((n, m))

        convergence_threshold = (
            float(threshold)
            if threshold is not None and threshold > 0.0
            else regularization_epsilon(b, cost)
        )

        eps = division_epsilon(b, cost)
        floor = tiny_value(b, cost)
        floor_vec_n = b.full((n,), floor)
        floor_vec_m = b.full((m,), floor)
        floor_mat = b.full((n, m), floor)

        cost_min = b.min(cost, axis=1, keepdims=True)
        cost_centered = cost - cost_min
        epsilon_floor = self._entropy_precision_floor(cost_centered, floor)
        epsilon = max(epsilon, epsilon_floor)
        log_K = -cost_centered / max(epsilon, eps)

        log_floor = log_scalar(floor, b)
        log_K = b.maximum(log_K, b.full((n, m), log_floor))
        K = b.exp(log_K)
        K = b.maximum(K, floor_mat)

        u = b.ones((n,))
        v = b.ones((m,))
        K_T = b.transpose(K)

        iterations = 0
        converged = False
        final_error = float("inf")

        # NO CAP — let it run until convergence or failure
        while iterations < 10000:  # absolute guard against infinite loop only
            iterations += 1
            Kv = b.matmul(K, v)
            Kv = b.maximum(Kv, floor_vec_n)
            u_new = p / Kv

            Ktu = b.matmul(K_T, u_new)
            Ktu = b.maximum(Ktu, floor_vec_m)
            v_new = q / Ktu

            plan = K * b.reshape(u_new, (n, 1)) * b.reshape(v_new, (1, m))
            row_sums = b.sum(plan, axis=1)
            col_sums = b.sum(plan, axis=0)
            row_error = b.max(b.abs(row_sums - p))
            col_error = b.max(b.abs(col_sums - q))
            b.eval(row_error, col_error)
            max_error = max(
                float(self._to_scalar(row_error)),
                float(self._to_scalar(col_error)),
            )

            u = u_new
            v = v_new
            if not is_finite(max_error, b):
                final_error = max_error
                break
            if max_error <= convergence_threshold:
                converged = True
                final_error = max_error
                break
            final_error = max_error

        iteration_log.append({
            "n": n,
            "m": m,
            "epsilon": epsilon,
            "iterations": iterations,
            "converged": converged,
            "final_error": final_error if math.isfinite(final_error) else "inf",
            "convergence_threshold": convergence_threshold,
        })

        G = K * b.reshape(u, (n, 1)) * b.reshape(v, (1, m))
        return G

    # Install instrumented solver
    SinkhornSolver.solve_linear_ot = instrumented_solve

    gw = GromovWassersteinDistance(backend=backend)
    test_results = []

    def pairwise_dist(pts):
        d = len(pts)
        mat = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                sq = sum((pts[i][k] - pts[j][k]) ** 2 for k in range(len(pts[i])))
                mat[i][j] = math.sqrt(sq)
        return mat

    # Test at various problem sizes with genuinely different geometries
    for n in [4, 8, 16, 32]:
        logger.info("  GW problem size n=%d", n)

        # Source: points on a unit circle (constant curvature)
        angles = [2 * math.pi * i / n for i in range(n)]
        points_source = [[math.cos(a), math.sin(a)] for a in angles]

        # Target: points on an ellipse (different curvature) with different spacing
        points_target = [
            [2.0 * math.cos(a * 1.3 + 0.5), 0.5 * math.sin(a * 1.3 + 0.5)]
            for a in angles
        ]

        C1 = backend.array(pairwise_dist(points_source))
        C2 = backend.array(pairwise_dist(points_target))
        backend.eval(C1, C2)

        iteration_log.clear()
        try:
            result = gw.compute(C1, C2)
            test_results.append({
                "n": n,
                "gw_distance": result.distance,
                "gw_converged": result.converged,
                "gw_outer_iterations": result.iterations,
                "inner_sinkhorn_calls": len(iteration_log),
                "inner_sinkhorn_iterations": [e["iterations"] for e in iteration_log],
                "inner_sinkhorn_converged": [e["converged"] for e in iteration_log],
                "inner_sinkhorn_errors": [e["final_error"] for e in iteration_log],
                "inner_max_iterations": max(e["iterations"] for e in iteration_log) if iteration_log else 0,
                "inner_all_converged": all(e["converged"] for e in iteration_log),
            })
            logger.info(
                "    GW: dist=%.4f, outer_iters=%d, inner_calls=%d, inner_max_iter=%d, all_converged=%s",
                result.distance, result.iterations,
                len(iteration_log),
                max(e["iterations"] for e in iteration_log) if iteration_log else 0,
                all(e["converged"] for e in iteration_log),
            )
        except Exception as e:
            logger.error("    GW failed for n=%d: %s", n, e)
            test_results.append({"n": n, "error": str(e)})

    # Restore original solver
    SinkhornSolver.solve_linear_ot = original_solve

    return test_results


# ---------------------------------------------------------------------------
# Measurement E: Divergence onset — relative spread
# ---------------------------------------------------------------------------

def measure_divergence_onset(model, tokenizer, backend, activation_provider):
    """Compute per-category geodesic deviations and measure relative spread."""
    logger.info("=== Measurement E: Divergence onset relative spread ===")

    from modelcypher.core.use_cases.geodesic_trajectory_service import GeodesicTrajectoryService

    service = GeodesicTrajectoryService(
        backend=backend,
        activation_provider=activation_provider,
    )

    try:
        batch_result = service.measure_layer_profile_batch(
            model, tokenizer, PROBE_PROMPTS, model_path="measurement",
        )
    except Exception as e:
        logger.error("  Batch profile failed: %s", e)
        return {"error": str(e)}

    # Extract per-layer per-category deviations
    all_layers = set()
    category_layer_devs = {}  # category -> {layer: mean_deviation}
    for cp in batch_result.category_profiles:
        layer_devs = {}
        for lp in cp.layer_profiles:
            all_layers.add(lp.layer)
            layer_devs[lp.layer] = lp.mean_deviation
        category_layer_devs[cp.category] = layer_devs

    # Per-layer: spread and relative spread
    layer_analysis = {}
    for layer_idx in sorted(all_layers):
        devs = []
        for cat, ld in category_layer_devs.items():
            if layer_idx in ld:
                devs.append(ld[layer_idx])

        if len(devs) < 2:
            continue

        spread = max(devs) - min(devs)
        mean_dev = sum(devs) / len(devs)
        relative_spread = spread / mean_dev if mean_dev > 0 else 0.0

        layer_analysis[layer_idx] = {
            "deviations_by_category": {
                cat: ld.get(layer_idx) for cat, ld in category_layer_devs.items()
            },
            "spread": spread,
            "mean_deviation": mean_dev,
            "relative_spread": relative_spread,
        }

        logger.info(
            "  Layer %d: spread=%.4f, mean_dev=%.4f, relative=%.4f",
            layer_idx, spread, mean_dev, relative_spread,
        )

    # Also compute second derivative of spread profile
    layers_sorted = sorted(layer_analysis.keys())
    spreads = [layer_analysis[l]["spread"] for l in layers_sorted]
    if len(spreads) >= 3:
        spread_second_deriv = []
        for i in range(1, len(spreads) - 1):
            d2 = spreads[i + 1] - 2 * spreads[i] + spreads[i - 1]
            spread_second_deriv.append(d2)
    else:
        spread_second_deriv = []

    return {
        "per_layer": layer_analysis,
        "current_threshold_absolute": 0.1,
        "divergence_onset_current": batch_result.divergence_onset_layer,
        "spread_second_derivative": spread_second_deriv,
        "spread_second_derivative_layers": layers_sorted[1:-1] if len(layers_sorted) >= 3 else [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Measure geometric thresholds")
    parser.add_argument(
        "--models", nargs="+", default=["350M"],
        choices=list(MODELS.keys()),
        help="Models to measure",
    )
    parser.add_argument(
        "--output", type=str,
        default="/Volumes/CodeCypher/models/experiments/geometric-thresholds",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-geodesic", action="store_true",
        help="Skip geodesic measurements B and E (slow)",
    )
    parser.add_argument(
        "--skip-gw", action="store_true",
        help="Skip GW Sinkhorn measurement D",
    )
    args = parser.parse_args()

    # Initialize backend
    from modelcypher.backends import initialize_default_backend
    backend = initialize_default_backend()

    # GW measurement is model-independent
    if not args.skip_gw:
        gw_results = measure_gw_sinkhorn_convergence(backend)
    else:
        gw_results = {"skipped": True}

    os.makedirs(args.output, exist_ok=True)

    for model_name in args.models:
        model_path = MODELS[model_name]
        if not os.path.exists(model_path):
            logger.error("Model not found: %s", model_path)
            continue

        logger.info("========== %s: %s ==========", model_name, model_path)

        # Load model
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.adapters.model_backbone import resolve_model_backbone
        model, tokenizer = ModelLoader().load_model(model_path)
        backbone = resolve_model_backbone(model)
        _, layers, _ = backbone
        n_layers = len(layers)
        logger.info("  %d layers", n_layers)

        # A: Eigenvalue gaps
        eig_results = measure_eigenvalue_gaps(
            model, tokenizer, backbone, backend, n_layers,
        )

        # C: Variance concentration
        vc_results = measure_variance_concentration(
            model, tokenizer, backbone, backend, n_layers,
        )

        # B & E: Geodesic (optional, slow)
        if not args.skip_geodesic:
            from modelcypher.adapters.activation_provider import ActivationProviderAdapter
            activation_provider = ActivationProviderAdapter(
                backend=backend, model_path=model_path,
            )
            geodesic_results = measure_geodesic_profiles(
                model, tokenizer, backend, activation_provider,
            )
            divergence_results = measure_divergence_onset(
                model, tokenizer, backend, activation_provider,
            )
        else:
            geodesic_results = {"skipped": True}
            divergence_results = {"skipped": True}

        # Compile results
        all_results = {
            "model": model_name,
            "model_path": model_path,
            "n_layers": n_layers,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "A_eigenvalue_gaps": {str(k): v for k, v in eig_results.items()},
            "B_geodesic_profiles": geodesic_results,
            "C_variance_concentration": vc_results,
            "D_gw_sinkhorn_convergence": gw_results,
            "E_divergence_onset": divergence_results,
        }

        # Write results
        out_path = os.path.join(args.output, f"{model_name}_geometric_thresholds.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("  Results written to %s", out_path)

        # Clean up model to free memory before next
        del model, tokenizer, backbone
        import gc
        gc.collect()

    logger.info("Done.")


if __name__ == "__main__":
    main()
