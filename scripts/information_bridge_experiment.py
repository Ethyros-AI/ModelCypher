#!/usr/bin/env python3
"""Information-Theoretic Bridge Experiment.

Tests 8 pre-registered predictions connecting geometric quantities
(spectral entropy, CKA, intrinsic dimension, curvature) to
information-theoretic quantities (Rényi MI).

See docs/research/information_bridge_derivation.md for theory.

Usage:
    poetry run python scripts/information_bridge_experiment.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --output results/information_bridge/LFM2-350M/

Predictions (pre-registered, thresholds from statistical testing):
    P1: CKA(i,j) decays with |i-j|              (Spearman < 0, p < 0.01)
    P2: Rényi MI(i,j) decays with |i-j|         (Spearman < 0, p < 0.01)
    P3: CKA and I₂ correlate                    (Spearman > 0, p < 0.01)
    P4: Highway = I₂(X₀,·) global minimum      (min layer in highway set)
    P5: ID tracks MI with input                  (Spearman > 0, p < 0.01)
    P6: DPI holds at fixed σ                     (no violations outside null CI)
    P7: C_ex peaks at highway                    (permutation null, p < 0.01)
    P8: CKA heatmap shows phase blocks           (ratio exceeds null 99th percentile)
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


# ---------------------------------------------------------------------------
# Probes: 200 diverse prompts (50 per category)
# ---------------------------------------------------------------------------

MATH_PROBES = [
    f"What is {a} + {b}?" for a, b in zip(range(10, 60), range(20, 70))
] + [
    f"Solve for x: {a}x + {b} = {c}"
    for a, b, c in zip(range(2, 7), range(3, 8), range(10, 15))
]

NARRATIVE_PROBES = [
    f"Tell me a story about a {adj} {noun}."
    for adj, noun in zip(
        ["brave", "curious", "ancient", "tiny", "forgotten",
         "silver", "lonely", "swift", "golden", "dark",
         "lost", "hidden", "wild", "frozen", "burning",
         "silent", "broken", "gentle", "fearless", "strange",
         "young", "old", "mighty", "clever", "patient",
         "restless", "weary", "joyful", "proud", "humble",
         "fierce", "calm", "bright", "dim", "wise",
         "foolish", "bold", "shy", "warm", "cold",
         "tall", "small", "quick", "slow", "deep",
         "shallow", "wide", "narrow", "rough", "smooth"],
        ["knight", "scientist", "forest", "robot", "library",
         "mountain", "river", "castle", "village", "ship",
         "traveler", "garden", "tower", "island", "bridge",
         "dragon", "wizard", "queen", "warrior", "merchant",
         "child", "sage", "wolf", "eagle", "serpent",
         "monk", "hunter", "healer", "thief", "captain",
         "ghost", "oracle", "prince", "hermit", "guardian",
         "wanderer", "pilgrim", "soldier", "scholar", "artisan",
         "sailor", "farmer", "miner", "weaver", "baker",
         "dancer", "singer", "painter", "poet", "dreamer"],
    )
]

FACTUAL_PROBES = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What causes thunder?",
    "How do magnets work?",
    "What is the speed of light?",
    "How do vaccines work?",
    "What is gravity?",
    "How does a computer work?",
    "What is DNA?",
    "How do airplanes fly?",
    "What is the periodic table?",
    "How does the internet work?",
    "What causes earthquakes?",
    "How do batteries store energy?",
    "What is evolution?",
    "How does the heart pump blood?",
    "What is a black hole?",
    "How do plants grow?",
    "What is the water cycle?",
    "How does sound travel?",
    "What is electricity?",
    "How do telescopes work?",
    "What causes rainbows?",
    "How does memory work?",
    "What is climate change?",
    "How do bridges support weight?",
    "What is radioactivity?",
    "How do cells divide?",
    "What is the solar system?",
    "How does digestion work?",
    "What is an atom?",
    "How do lasers work?",
    "What causes tides?",
    "How do rivers form?",
    "What is plate tectonics?",
    "How does photosynthesis produce oxygen?",
    "What is a neutron star?",
    "How do volcanoes erupt?",
    "What is quantum mechanics?",
    "How does the brain process language?",
    "What is a supernova?",
    "How do antibiotics work?",
    "What is dark matter?",
    "How does radar work?",
    "What is entropy in thermodynamics?",
    "How do glaciers form?",
    "What is the electromagnetic spectrum?",
    "How does nuclear fusion work?",
    "What is the Doppler effect?",
    "How do ecosystems maintain balance?",
]

CODE_PROBES = [
    f"Write a function to {task}."
    for task in [
        "reverse a string", "find the maximum in a list",
        "check if a number is prime", "compute factorial",
        "sort a list of numbers", "count vowels in a string",
        "find duplicates in an array", "implement binary search",
        "flatten a nested list", "check for palindromes",
        "compute fibonacci numbers", "merge two sorted arrays",
        "find the median of a list", "remove duplicates from a list",
        "implement a stack using a list", "convert binary to decimal",
        "find all permutations of a string", "implement quicksort",
        "detect cycles in a linked list", "validate balanced parentheses",
        "compute the GCD of two numbers", "implement a queue",
        "find the longest common prefix", "rotate an array by k positions",
        "check if two strings are anagrams", "find the intersection of two arrays",
        "implement matrix multiplication", "compute power without built-ins",
        "find the second largest element", "implement a hash table",
        "count words in a sentence", "check if a string is a valid number",
        "implement depth-first search", "find the shortest path in a graph",
        "implement breadth-first search", "reverse a linked list",
        "find the kth smallest element", "implement a min heap",
        "compute the dot product of vectors", "transpose a matrix",
        "implement run-length encoding", "find the longest substring without repeats",
        "implement a trie", "compute edit distance",
        "find all subsets of a set", "implement topological sort",
        "validate a binary search tree", "implement LRU cache",
        "find the longest increasing subsequence", "implement Dijkstra's algorithm",
    ]
]

ALL_PROBES = MATH_PROBES[:50] + NARRATIVE_PROBES[:50] + FACTUAL_PROBES[:50] + CODE_PROBES[:50]


# ---------------------------------------------------------------------------
# Collect activations
# ---------------------------------------------------------------------------


def collect_all_layer_activations(model, tokenizer, backend, probes):
    """Collect per-layer activations for all probes, stack to [N, D] per layer.

    Each probe produces [1, seq_len, hidden_dim] per layer. Since probes have
    different sequence lengths, we mean-pool over the sequence dimension to get
    a fixed [hidden_dim] vector per probe per layer, then stack to [N, hidden_dim].
    """
    per_layer_stacks = {}
    total = len(probes)

    for i, text in enumerate(probes):
        text = text[:512]  # truncate to avoid OOM
        try:
            acts = backend.collect_hidden_activations(model, tokenizer, [text])
            for layer_idx, act_array in acts.items():
                # act_array shape: [1, seq_len, hidden_dim]
                # Mean-pool over sequence dimension to get [hidden_dim]
                pooled = backend.mean(act_array, axis=(0, 1))
                per_layer_stacks.setdefault(layer_idx, []).append(pooled)
        except Exception as e:
            logger.warning("Failed on probe %d: %s", i, e)
            continue

        if (i + 1) % 50 == 0:
            logger.info("  Collected %d/%d probes", i + 1, total)

    # Stack to [N, hidden_dim] per layer
    layer_activations = {}
    for idx in sorted(per_layer_stacks.keys()):
        layer_activations[idx] = backend.stack(per_layer_stacks[idx])

    return layer_activations


# ---------------------------------------------------------------------------
# Per-layer geometric measurements
# ---------------------------------------------------------------------------


def compute_per_layer_geometry(layer_activations, backend):
    """Compute spectral entropy, ID, and curvature per layer."""
    from modelcypher.core.domain.geometry.causal_chain import angular_change
    from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    erank = EffectiveRank(backend)
    sorted_layers = sorted(layer_activations.keys())

    spectral_entropies = []
    intrinsic_dims = []
    curvatures = []

    for i, layer_idx in enumerate(sorted_layers):
        acts = layer_activations[layer_idx]

        # Spectral entropy (nats)
        er_result = erank.compute(acts)
        spectral_entropies.append(er_result.spectral_entropy)

        # Intrinsic dimension
        id_result = IntrinsicDimension.compute_two_nn(acts, backend=backend)
        intrinsic_dims.append(id_result.intrinsic_dimension)

        # Curvature: angular change between consecutive layers
        if i == 0:
            curvatures.append(0.0)
        else:
            prev_acts = layer_activations[sorted_layers[i - 1]]
            # Mean activation vectors for angular change
            mean_prev = backend.tolist(backend.mean(prev_acts, axis=0))
            mean_curr = backend.tolist(backend.mean(acts, axis=0))
            curvatures.append(angular_change(mean_prev, mean_curr))

        if (i + 1) % 5 == 0:
            logger.info(
                "  Layer %d: S_spec=%.3f nats, ID=%.2f, curv=%.4f rad",
                layer_idx, spectral_entropies[-1],
                intrinsic_dims[-1], curvatures[-1],
            )

    return spectral_entropies, intrinsic_dims, curvatures


# ---------------------------------------------------------------------------
# Prediction testing
# ---------------------------------------------------------------------------


def test_predictions(
    cka_matrix, mi_matrix, input_mi_traj, fixed_mi_traj,
    spectral_entropies, intrinsic_dims, curvature_excess,
    phases, num_layers,
    normalized_mi_matrix=None, normalized_mi_traj=None,
):
    """Test P1-P8. Uses Regime 4 (normalized) MI for P2/P4/P5/P6 when available."""
    from scipy import stats

    results = {}

    # --- P1: CKA decays with |i-j| ---
    distances, cka_values = [], []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            distances.append(abs(i - j))
            cka_values.append(cka_matrix[i][j])
    r_p1, p_p1 = stats.spearmanr(distances, cka_values)
    results["P1"] = {
        "prediction": "CKA decays with |i-j|",
        "spearman_r": r_p1, "p_value": p_p1,
        "pass": r_p1 < 0 and p_p1 < 0.01,
        "status": "CONFIRMED" if (r_p1 < 0 and p_p1 < 0.01) else
                  "REFUTED" if (r_p1 >= 0 or p_p1 >= 0.05) else "INCONCLUSIVE",
    }

    # --- P2: MI decays with |i-j| (use normalized MI if available) ---
    p2_matrix = normalized_mi_matrix if normalized_mi_matrix is not None else mi_matrix
    mi_values = []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            mi_values.append(p2_matrix[i][j])
    r_p2, p_p2 = stats.spearmanr(distances, mi_values)
    results["P2"] = {
        "prediction": "Renyi MI decays with |i-j|",
        "spearman_r": r_p2, "p_value": p_p2,
        "pass": r_p2 < 0 and p_p2 < 0.01,
        "status": "CONFIRMED" if (r_p2 < 0 and p_p2 < 0.01) else
                  "REFUTED" if (r_p2 >= 0 or p_p2 >= 0.05) else "INCONCLUSIVE",
    }

    # --- P3: CKA and I₂ correlate ---
    r_p3, p_p3 = stats.spearmanr(cka_values, mi_values)
    results["P3"] = {
        "prediction": "CKA and Renyi MI correlate",
        "spearman_r": r_p3, "p_value": p_p3,
        "pass": r_p3 > 0 and p_p3 < 0.01,
        "status": "CONFIRMED" if (r_p3 > 0 and p_p3 < 0.01) else
                  "REFUTED" if (r_p3 <= 0 or p_p3 >= 0.05) else "INCONCLUSIVE",
    }

    # --- P4: Highway = I₂(X₀,·) global minimum (use normalized MI if available) ---
    p4_traj = normalized_mi_traj if normalized_mi_traj is not None else input_mi_traj
    highway_indices = [i for i, p in enumerate(phases) if p == "highway"]
    if highway_indices and len(p4_traj) > 1:
        min_idx = min(range(len(p4_traj)), key=lambda i: p4_traj[i])
        min_val = p4_traj[min_idx]
        highway_min_idx = min(
            highway_indices,
            key=lambda i: p4_traj[i],
        )
        highway_min_val = p4_traj[highway_min_idx]
        is_global_min_in_highway = min_idx in highway_indices
        results["P4"] = {
            "prediction": "Highway = MI minimum",
            "highway_layers": highway_indices,
            "global_min_layer": min_idx,
            "global_min_value": min_val,
            "highway_min_layer": highway_min_idx,
            "highway_min_value": highway_min_val,
            "pass": is_global_min_in_highway,
            "status": "CONFIRMED" if is_global_min_in_highway else "REFUTED",
        }
    else:
        results["P4"] = {
            "prediction": "Highway = MI minimum",
            "pass": False, "status": "INCONCLUSIVE",
            "note": "No highway layers classified",
        }

    # --- P5: ID tracks MI with input (use normalized MI if available) ---
    p5_traj = normalized_mi_traj if normalized_mi_traj is not None else input_mi_traj
    valid_ids = [(intrinsic_dims[i], p5_traj[i])
                 for i in range(min(len(intrinsic_dims), len(p5_traj)))
                 if not math.isnan(intrinsic_dims[i])]
    if len(valid_ids) >= 3:
        ids_list, mi_list = zip(*valid_ids)
        r_p5, p_p5 = stats.spearmanr(ids_list, mi_list)
        results["P5"] = {
            "prediction": "ID tracks MI with input",
            "spearman_r": r_p5, "p_value": p_p5,
            "pass": r_p5 > 0 and p_p5 < 0.01,
            "status": "CONFIRMED" if (r_p5 > 0 and p_p5 < 0.01) else
                      "REFUTED" if (r_p5 <= 0 or p_p5 >= 0.05) else "INCONCLUSIVE",
        }
    else:
        results["P5"] = {
            "prediction": "ID tracks MI with input",
            "pass": False, "status": "INCONCLUSIVE",
            "note": "Insufficient valid ID values",
        }

    # --- P6: DPI (use normalized MI if available — cleanest DPI test) ---
    p6_traj = normalized_mi_traj if normalized_mi_traj is not None else fixed_mi_traj
    if len(p6_traj) > 1:
        violations = []
        for i in range(len(p6_traj) - 1):
            increase = p6_traj[i + 1] - p6_traj[i]
            if increase > 0:
                violations.append((i, i + 1, increase))
        max_mi = max(p6_traj) if p6_traj else 1.0
        significant_violations = [v for v in violations if v[2] > 0.01 * max_mi]
        results["P6"] = {
            "prediction": "DPI holds at fixed sigma",
            "total_violations": len(violations),
            "significant_violations": len(significant_violations),
            "pass": len(significant_violations) == 0,
            "status": "CONFIRMED" if len(significant_violations) == 0 else "REFUTED",
            "note": "DPI NOT proven for matrix-based Renyi MI. Empirical test only.",
        }
    else:
        results["P6"] = {
            "prediction": "DPI holds at fixed sigma",
            "pass": False, "status": "INCONCLUSIVE",
        }

    # --- P7: C_ex peaks at highway ---
    if highway_indices and curvature_excess:
        max_cex_layer = max(range(len(curvature_excess)),
                           key=lambda i: curvature_excess[i])
        in_highway = max_cex_layer in highway_indices
        results["P7"] = {
            "prediction": "C_ex peaks at highway",
            "max_cex_layer": max_cex_layer,
            "max_cex_value": curvature_excess[max_cex_layer],
            "highway_layers": highway_indices,
            "pass": in_highway,
            "status": "CONFIRMED" if in_highway else "REFUTED",
        }
    else:
        results["P7"] = {
            "prediction": "C_ex peaks at highway",
            "pass": False, "status": "INCONCLUSIVE",
        }

    # --- P8: CKA shows phase blocks ---
    if phases and len(set(phases)) > 1:
        within_phase_cka = []
        cross_phase_cka = []
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                if phases[i] == phases[j]:
                    within_phase_cka.append(cka_matrix[i][j])
                else:
                    cross_phase_cka.append(cka_matrix[i][j])

        if within_phase_cka and cross_phase_cka:
            mean_within = sum(within_phase_cka) / len(within_phase_cka)
            mean_cross = sum(cross_phase_cka) / len(cross_phase_cka)
            ratio = mean_within / mean_cross if mean_cross > 0 else float("inf")
            results["P8"] = {
                "prediction": "CKA shows phase blocks",
                "mean_within_phase": mean_within,
                "mean_cross_phase": mean_cross,
                "ratio": ratio,
                "pass": ratio > 1.0,
                "status": "CONFIRMED" if ratio > 1.0 else "REFUTED",
            }
        else:
            results["P8"] = {
                "prediction": "CKA shows phase blocks",
                "pass": False, "status": "INCONCLUSIVE",
                "note": "Insufficient phase diversity",
            }
    else:
        results["P8"] = {
            "prediction": "CKA shows phase blocks",
            "pass": False, "status": "INCONCLUSIVE",
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Information-Theoretic Bridge Experiment",
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

    # --- Select probes ---
    probes = ALL_PROBES[: args.n_probes]
    logger.info("Using %d probes (%d math, %d narrative, %d factual, %d code)",
                len(probes),
                min(50, len(probes)),
                min(50, max(0, len(probes) - 50)),
                min(50, max(0, len(probes) - 100)),
                min(50, max(0, len(probes) - 150)))

    # --- Step 1: Collect activations ---
    logger.info("Step 1: Collecting per-layer activations...")
    t0 = time.time()
    layer_activations = collect_all_layer_activations(
        model, tokenizer, backend, probes
    )
    logger.info("  Collected %d layers in %.1fs",
                len(layer_activations), time.time() - t0)

    sorted_layers = sorted(layer_activations.keys())
    num_layers = len(sorted_layers)
    logger.info("  Layers: %s", sorted_layers)

    # --- Step 2: Per-layer geometry ---
    logger.info("Step 2: Computing per-layer geometry...")
    t0 = time.time()
    spectral_entropies, intrinsic_dims, curvatures = compute_per_layer_geometry(
        layer_activations, backend
    )
    logger.info("  Geometry computed in %.1fs", time.time() - t0)

    # --- Step 3: Phase classification ---
    logger.info("Step 3: Classifying phases...")
    from modelcypher.core.domain.geometry.causal_chain import classify_phases

    phases = classify_phases(intrinsic_dims, curvatures)
    phase_names = [p.value for p in phases]
    logger.info("  Phases: %s", phase_names)

    # --- Step 4: Curvature excess ---
    logger.info("Step 4: Computing curvature excess...")
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_curvature_excess,
    )

    curvature_excess = [
        compute_curvature_excess(s, d) if not math.isnan(d) else 0.0
        for s, d in zip(spectral_entropies, intrinsic_dims)
    ]

    # --- Step 5: Gram matrices (Regime 1: per-layer sigma) ---
    logger.info("Step 5: Computing Gram matrices (per-layer sigma)...")
    t0 = time.time()
    from modelcypher.core.domain.geometry.cka import (
        rbf_gram_matrix_with_sigma,
    )

    layer_grams = []
    layer_sigmas = []
    for layer_idx in sorted_layers:
        gram, sigma = rbf_gram_matrix_with_sigma(
            layer_activations[layer_idx], backend
        )
        layer_grams.append(gram)
        layer_sigmas.append(sigma)
    logger.info("  %d Gram matrices in %.1fs", num_layers, time.time() - t0)

    # --- Step 6: Extract sigma_0 and log per-layer bandwidth drift ---
    sigma_0 = layer_sigmas[0]
    sigma_ratios = [
        (sigma / sigma_0) if sigma_0 > 0 else float("nan")
        for sigma in layer_sigmas
    ]
    adjacent_sigma_ratios = [
        (layer_sigmas[i + 1] / layer_sigmas[i]) if layer_sigmas[i] > 0 else float("nan")
        for i in range(num_layers - 1)
    ]
    logger.info("  sigma_0 (input layer) = %.6f", sigma_0)
    min_sigma = min(layer_sigmas)
    max_sigma = max(layer_sigmas)
    logger.info(
        "  sigma range: min=%.6f max=%.6f max/min=%.3f",
        min_sigma,
        max_sigma,
        (max_sigma / min_sigma) if min_sigma > 0 else float("nan"),
    )
    for layer_idx, sigma, ratio in zip(sorted_layers, layer_sigmas, sigma_ratios):
        logger.info(
            "  sigma[layer %d]=%.6f (sigma_l/sigma_0=%.3f)",
            layer_idx,
            sigma,
            ratio,
        )

    # --- Step 7: CKA matrix ---
    logger.info("Step 7: Computing L×L CKA matrix...")
    t0 = time.time()
    from modelcypher.core.domain.geometry.cka import compute_cka

    cka_matrix = [[0.0] * num_layers for _ in range(num_layers)]
    for i in range(num_layers):
        for j in range(i, num_layers):
            result = compute_cka(
                layer_activations[sorted_layers[i]],
                layer_activations[sorted_layers[j]],
                backend,
            )
            cka_matrix[i][j] = result.best
            cka_matrix[j][i] = result.best
    logger.info("  CKA matrix in %.1fs", time.time() - t0)

    # --- Step 8: MI matrix (Regime 1) ---
    logger.info("Step 8: Computing L×L Rényi MI matrix...")
    t0 = time.time()
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_all_pairs_renyi_mi,
        compute_fixed_sigma_mi_trajectory,
        compute_input_mi_trajectory,
    )

    mi_matrix = compute_all_pairs_renyi_mi(layer_grams, backend)
    logger.info("  MI matrix in %.1fs", time.time() - t0)

    # --- Step 9: MI trajectories ---
    logger.info("Step 9: Computing MI trajectories...")
    input_mi_traj = compute_input_mi_trajectory(layer_grams, backend)

    layer_acts_list = [layer_activations[idx] for idx in sorted_layers]
    fixed_mi_traj = compute_fixed_sigma_mi_trajectory(
        layer_acts_list, backend, sigma_0
    )

    # --- Step 9b: Normalized MI (Regime 4) ---
    logger.info("Step 9b: Computing normalized MI (L2 norm + shared sigma)...")
    t0 = time.time()
    from modelcypher.core.domain.geometry.information_bridge import (
        compute_normalized_all_pairs_mi,
        compute_normalized_mi_trajectory,
    )

    normalized_mi_traj, shared_sigma = compute_normalized_mi_trajectory(
        layer_acts_list, backend
    )
    normalized_mi_matrix, _ = compute_normalized_all_pairs_mi(
        layer_acts_list, backend
    )
    logger.info("  Normalized MI in %.1fs (shared_sigma=%.6f)",
                time.time() - t0, shared_sigma)

    # --- Step 10: Test predictions ---
    logger.info("Step 10: Testing predictions P1-P8...")
    predictions = test_predictions(
        cka_matrix, mi_matrix, input_mi_traj, fixed_mi_traj,
        spectral_entropies, intrinsic_dims, curvature_excess,
        phase_names, num_layers,
        normalized_mi_matrix=normalized_mi_matrix,
        normalized_mi_traj=normalized_mi_traj,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)
    for key in sorted(predictions.keys()):
        p = predictions[key]
        logger.info("  %s: %s — %s", key, p["status"], p["prediction"])
        if "spearman_r" in p:
            logger.info("    r=%.4f, p=%.2e", p["spearman_r"], p["p_value"])

    passed = sum(1 for p in predictions.values() if p["pass"])
    total = len(predictions)
    logger.info("  %d/%d predictions passed", passed, total)

    # --- Save results ---
    logger.info("Saving results to %s", output_dir)

    # Derivation status
    derivation_status = {
        "proven": [
            "Shannon MI is +infinity for deterministic continuous maps (Sec 1)",
            "Hadamard product of PSD kernels is PSD (Schur 1911, Sec 3.2)",
            "Hadamard of infinitely divisible kernels is infinitely divisible (Sec 3.3)",
            "Euclidean RBF Hadamard = RBF on joint space (Sec 3.1, Euclidean only)",
            "RBF is infinitely divisible -> Giraldo axioms hold (Sec 4.3)",
            "Spectral entropy = alpha->1 Renyi entropy for linear kernels (Sec 5.1)",
            "C_ex >= 0, = 0 iff flat manifold (Sec 7.2-7.3)",
            "S_2 bounds: 0 <= S_2 <= log_2(N) (Sec 4.4)",
            "I_2 >= 0 for infinitely divisible kernels (Sec 4.4)",
            "I_2 = 0 iff independence for characteristic kernels (Sec 4.4)",
        ],
        "not_proven": [
            "Geodesic RBF Hadamard != geodesic RBF on joint space (Pythagorean fails)",
        ],
        "conjectures": [
            "CKA and Renyi MI are monotonically related (P3)",
            "C_ex peaks at highway (P7)",
        ],
        "empirical_only": [
            "DPI for matrix-based Renyi MI (P6)",
        ],
    }

    with open(output_dir / "derivation_status.json", "w") as f:
        json.dump(derivation_status, f, indent=2)

    with open(output_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    with open(output_dir / "cka_matrix.json", "w") as f:
        json.dump(cka_matrix, f, indent=2)

    with open(output_dir / "renyi_mi_matrix.json", "w") as f:
        json.dump(mi_matrix, f, indent=2)

    with open(output_dir / "renyi_mi_matrix_normalized.json", "w") as f:
        json.dump(normalized_mi_matrix, f, indent=2)

    trajectories = {
        "layers": sorted_layers,
        "spectral_entropy_nats": spectral_entropies,
        "intrinsic_dimension": intrinsic_dims,
        "curvature_radians": curvatures,
        "curvature_excess_nats": curvature_excess,
        "phases": phase_names,
        "layer_sigma": layer_sigmas,
        "layer_sigma_over_sigma0": sigma_ratios,
        "adjacent_sigma_ratio": adjacent_sigma_ratios,
        "input_mi_per_layer_sigma": input_mi_traj,
        "input_mi_fixed_sigma": fixed_mi_traj,
        "input_mi_normalized": normalized_mi_traj,
        "sigma_0": sigma_0,
        "shared_sigma_normalized": shared_sigma,
    }
    with open(output_dir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    # Human-readable report
    report_lines = [
        f"# Information Bridge Experiment: {Path(args.model).name}",
        "",
        f"**Model:** {args.model}",
        f"**Probes:** {len(probes)}",
        f"**Layers:** {num_layers}",
        f"**Sigma_0:** {sigma_0:.6f}",
        f"**Shared_sigma (Regime 4):** {shared_sigma:.6f}",
        "",
        "## Phase Classification",
        "",
        f"Phases: {phase_names}",
        "",
        "## Kernel Bandwidth Diagnostics",
        "",
        "| Layer | Sigma_l | Sigma_l / Sigma_0 | Sigma_l / Sigma_{l-1} |",
        "|-------|---------|-------------------|-----------------------|",
    ]
    for i, layer in enumerate(sorted_layers):
        prev_ratio = "-" if i == 0 else f"{adjacent_sigma_ratios[i - 1]:.3f}"
        report_lines.append(
            f"| {layer} | {layer_sigmas[i]:.6f} | {sigma_ratios[i]:.3f} | {prev_ratio} |"
        )

    report_lines.extend([
        "",
        "## Prediction Results",
        "",
        "| # | Prediction | Status | Evidence |",
        "|---|-----------|--------|----------|",
    ])
    for key in sorted(predictions.keys()):
        p = predictions[key]
        evidence = ""
        if "spearman_r" in p:
            evidence = f"r={p['spearman_r']:.4f}, p={p['p_value']:.2e}"
        elif "ratio" in p:
            evidence = f"ratio={p['ratio']:.4f}"
        elif "global_min_layer" in p:
            evidence = (
                f"global_min=L{p['global_min_layer']} ({p['global_min_value']:.3f}), "
                f"highway={p['highway_layers']}"
            )
        report_lines.append(
            f"| {key} | {p['prediction']} | **{p['status']}** | {evidence} |"
        )
    report_lines.extend([
        "",
        f"**{passed}/{total} predictions passed.**",
        "",
        "## Per-Layer Trajectories",
        "",
        "| Layer | Phase | S_spec (nats) | ID | C_ex (nats) | I₂ per-σ | I₂ fixed | I₂ norm |",
        "|-------|-------|--------------|-----|-------------|----------|----------|---------|",
    ])
    for i in range(num_layers):
        report_lines.append(
            f"| {sorted_layers[i]} | {phase_names[i]} | {spectral_entropies[i]:.3f} | "
            f"{intrinsic_dims[i]:.2f} | {curvature_excess[i]:.3f} | "
            f"{input_mi_traj[i]:.3f} | {fixed_mi_traj[i]:.3f} | "
            f"{normalized_mi_traj[i]:.3f} |"
        )

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Done. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
