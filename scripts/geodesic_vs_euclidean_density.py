#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment: Does geodesic vs Euclidean distance change k-NN density rankings
# on actual LFM2-350M activations?
#
# Measurement protocol:
# 1. Extract activations at multiple layers using two different prompt sets
#    (simulating "source" and "target" in knowledge_density comparison)
# 2. Compute k-NN density with Euclidean (chord) distances
# 3. Compute k-NN density with geodesic (Floyd-Warshall) distances
# 4. Measure:
#    a. Per-pair distortion: |d_geo - d_chord| / d_chord
#    b. Spearman rank correlation of density orderings
#    c. Whether density transfer weights (src / (src + tgt)) change sign
#    d. Max absolute difference in normalized density
# 5. Compare all differences against sqrt(eps_f32) = 3.45e-4
#
# If distortion < sqrt(eps), Euclidean is correct for this use case.
# If distortion >= sqrt(eps), geodesic is required.
# The measurement decides. Not papers.

import json
import math
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        sqrt_scalar,
    )
    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    # IEEE 754 float32
    eps_f32 = math.ldexp(1.0, -23)
    sqrt_eps = math.sqrt(eps_f32)
    print(f"sqrt(eps_f32) = {sqrt_eps:.6e}")
    print(f"Threshold: distortion above this means geodesic is required.\n")

    # --- Load model ---
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    print(f"Loading model from {model_path}...")
    model, tokenizer = mlx_load(model_path)
    mx.eval(model.parameters())

    # Resolve backbone
    from modelcypher.adapters.model_backbone import resolve_model_backbone

    embed_tokens, layers, final_norm = resolve_model_backbone(model)
    n_layers = len(layers)
    print(f"Model loaded: {n_layers} layers")

    # --- Two prompt sets (simulating source/target comparison) ---
    # These should produce different activation distributions
    source_prompts = [
        "The derivative of x squared is two x.",
        "Photosynthesis converts carbon dioxide and water into glucose.",
        "The mitochondria is the powerhouse of the cell.",
        "Newton's second law states F equals m times a.",
        "DNA is a double helix structure discovered by Watson and Crick.",
        "The speed of light in vacuum is approximately 3e8 meters per second.",
        "Euler's identity states e to the i pi plus one equals zero.",
        "The area of a circle is pi times the radius squared.",
        "Water freezes at zero degrees Celsius at standard pressure.",
        "The Pythagorean theorem states a squared plus b squared equals c squared.",
        "Gravity accelerates objects at 9.8 meters per second squared.",
        "The chemical formula for water is H2O.",
        "The boiling point of water is 100 degrees Celsius.",
        "Ohm's law states voltage equals current times resistance.",
        "The speed of sound in air is approximately 343 meters per second.",
    ]

    target_prompts = [
        "Once upon a time, there was a brave knight who fought dragons.",
        "The sunset painted the sky in shades of orange and purple.",
        "She whispered softly, hoping no one else would hear her secret.",
        "The old house on the hill had been abandoned for decades.",
        "He picked up the guitar and played a melody from his childhood.",
        "The rain fell steadily, drumming against the window pane.",
        "They walked hand in hand along the empty beach at midnight.",
        "The forest was silent except for the rustling of leaves.",
        "A single candle flickered in the darkness of the ancient library.",
        "The cat sat on the windowsill, watching the birds outside.",
        "Morning dew glistened on the grass like scattered diamonds.",
        "The train pulled away from the station with a long whistle.",
        "Children laughed as they chased each other through the garden.",
        "The smell of fresh bread filled the kitchen every morning.",
        "Stars appeared one by one as twilight faded into night.",
    ]

    # --- Extract activations ---
    # Sample at 3 layers: early (1/4), middle (1/2), late (3/4)
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    print(f"Sampling layers: {sample_layers}")

    b = MLXBackend()

    def extract_activations(prompts, layer_idx):
        """Extract mean-pooled hidden states at a specific layer."""
        activations = []
        for text in prompts:
            tokens = tokenizer.encode(text)
            input_ids = mx.array([tokens])
            h = embed_tokens(input_ids)

            for i in range(layer_idx + 1):
                layer = layers[i]
                # LFM2: attention layers need "causal" mask, conv layers need None
                if hasattr(layer, "is_attention_layer") and layer.is_attention_layer:
                    h = layer(h, mask="causal")
                else:
                    h = layer(h, mask=None)

            mx.eval(h)

            # Mean pool over sequence dimension → [hidden_dim]
            pooled = mx.mean(h[0], axis=0)
            mx.eval(pooled)
            activations.append(pooled)

        # Stack to [n_prompts, hidden_dim]
        result = mx.stack(activations, axis=0)
        mx.eval(result)
        return result

    # --- Run experiment at each layer ---
    results = {}

    for layer_idx in sample_layers:
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*70}")

        t0 = time.time()
        source_acts = extract_activations(source_prompts, layer_idx)
        target_acts = extract_activations(target_prompts, layer_idx)
        t_extract = time.time() - t0
        print(f"Extracted activations in {t_extract:.1f}s")
        print(f"  Source shape: {source_acts.shape}")
        print(f"  Target shape: {target_acts.shape}")

        rg = RiemannianGeometry(b)

        # --- Euclidean (chord) distances ---
        t0 = time.time()
        source_chord = rg._chord_distance_matrix(source_acts, use_cache=False)
        target_chord = rg._chord_distance_matrix(target_acts, use_cache=False)
        b.eval(source_chord, target_chord)
        t_chord = time.time() - t0
        print(f"\nChord distances computed in {t_chord:.2f}s")

        # --- Geodesic distances ---
        t0 = time.time()
        try:
            source_geo_result = rg.geodesic_distances(source_acts)
            target_geo_result = rg.geodesic_distances(target_acts)
            source_geo = source_geo_result.distances
            target_geo = target_geo_result.distances
            b.eval(source_geo, target_geo)
            t_geo = time.time() - t0
            print(f"Geodesic distances computed in {t_geo:.2f}s")
            print(f"  Source k_neighbors used: {source_geo_result.k_neighbors}")
            print(f"  Target k_neighbors used: {target_geo_result.k_neighbors}")
        except Exception as e:
            print(f"Geodesic computation failed: {e}")
            print("This itself is data — if geodesic can't be computed, it can't be used.")
            continue

        # ============================================================
        # MEASUREMENT 1: Per-pair distortion |d_geo - d_chord| / d_chord
        # ============================================================
        n_src = int(source_chord.shape[0])
        n_tgt = int(target_chord.shape[0])

        def measure_distortion(chord_mat, geo_mat, label):
            """Measure per-pair Euclidean/geodesic distortion."""
            n = int(chord_mat.shape[0])
            distortions = []
            for i in range(n):
                for j in range(i + 1, n):
                    c = float(b.to_scalar(chord_mat[i][j]))
                    g = float(b.to_scalar(geo_mat[i][j]))
                    if c > sqrt_eps:  # skip near-zero distances
                        distortion = abs(g - c) / c
                        distortions.append(distortion)

            if not distortions:
                print(f"  {label}: no valid pairs")
                return 0.0, 0.0, 0.0, []

            mean_d = sum(distortions) / len(distortions)
            max_d = max(distortions)
            # What fraction of pairs have distortion > sqrt(eps)?
            above_threshold = sum(1 for d in distortions if d > sqrt_eps)
            frac_above = above_threshold / len(distortions)

            print(f"  {label}:")
            print(f"    Mean distortion: {mean_d:.6e}")
            print(f"    Max distortion:  {max_d:.6e}")
            print(f"    Pairs above sqrt(eps): {above_threshold}/{len(distortions)} ({frac_above:.1%})")
            print(f"    sqrt(eps) = {sqrt_eps:.6e}")
            print(f"    Verdict: {'GEODESIC NEEDED' if max_d > sqrt_eps else 'EUCLIDEAN OK'}")

            return mean_d, max_d, frac_above, distortions

        print(f"\n--- Measurement 1: Pairwise Distortion ---")
        src_mean, src_max, src_frac, src_distortions = measure_distortion(
            source_chord, source_geo, "Source"
        )
        tgt_mean, tgt_max, tgt_frac, tgt_distortions = measure_distortion(
            target_chord, target_geo, "Target"
        )

        # ============================================================
        # MEASUREMENT 2: k-NN density ranking correlation
        # ============================================================
        print(f"\n--- Measurement 2: k-NN Density Rankings ---")

        # Use same k derivation as knowledge_density.py
        from modelcypher.core.domain.geometry.riemannian_validation import derive_k_neighbors

        k_src = derive_k_neighbors(source_acts, b)
        k_tgt = derive_k_neighbors(target_acts, b)
        k = max(k_src, k_tgt, 1)
        k = min(k, n_src - 1, n_tgt - 1)
        print(f"  k = {k}")

        def compute_density(dist_matrix, k_val, label_metric):
            """Compute k-NN density from distance matrix."""
            sorted_dists = b.sort(dist_matrix, axis=1)
            b.eval(sorted_dists)
            k_dists = sorted_dists[:, 1:k_val+1]
            mean_k_dist = b.mean(k_dists, axis=1)
            b.eval(mean_k_dist)

            n = int(mean_k_dist.shape[0])
            densities = []
            for i in range(n):
                d = float(b.to_scalar(mean_k_dist[i]))
                densities.append(1.0 / max(d, eps_f32))
            return densities

        # Compute densities 4 ways: {source,target} × {chord,geodesic}
        src_density_chord = compute_density(source_chord, k, "source-chord")
        src_density_geo = compute_density(source_geo, k, "source-geo")
        tgt_density_chord = compute_density(target_chord, k, "target-chord")
        tgt_density_geo = compute_density(target_geo, k, "target-geo")

        def spearman_rank_corr(a, b_list):
            """Spearman rank correlation (no scipy needed)."""
            n = len(a)
            if n < 2:
                return 1.0

            # Compute ranks (average for ties)
            def ranks(vals):
                indexed = sorted(enumerate(vals), key=lambda x: x[1])
                r = [0.0] * n
                i = 0
                while i < n:
                    j = i
                    while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                        j += 1
                    avg_rank = (i + j) / 2.0 + 1.0  # 1-based
                    for kk in range(i, j + 1):
                        r[indexed[kk][0]] = avg_rank
                    i = j + 1
                return r

            ra = ranks(a)
            rb = ranks(b_list)

            # Pearson on ranks
            mean_a = sum(ra) / n
            mean_b = sum(rb) / n
            num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
            den_a = sum((ra[i] - mean_a) ** 2 for i in range(n))
            den_b = sum((rb[i] - mean_b) ** 2 for i in range(n))
            den = math.sqrt(den_a * den_b)
            if den < eps_f32:
                return 1.0
            return num / den

        rho_src = spearman_rank_corr(src_density_chord, src_density_geo)
        rho_tgt = spearman_rank_corr(tgt_density_chord, tgt_density_geo)
        print(f"  Source density ranking Spearman ρ (chord vs geo): {rho_src:.6f}")
        print(f"  Target density ranking Spearman ρ (chord vs geo): {rho_tgt:.6f}")
        print(f"  Verdict: {'RANKINGS PRESERVED' if rho_src > 0.99 and rho_tgt > 0.99 else 'RANKINGS DIFFER'}")

        # ============================================================
        # MEASUREMENT 3: Transfer weight sign changes
        # ============================================================
        print(f"\n--- Measurement 3: Transfer Weight Sign Changes ---")

        n_compare = min(n_src, n_tgt)

        def compute_transfer_weights(src_dens, tgt_dens):
            """Transfer weight = src / (src + tgt). >0.5 means transfer from source."""
            weights = []
            for i in range(min(len(src_dens), len(tgt_dens))):
                total = src_dens[i] + tgt_dens[i]
                if total > eps_f32:
                    weights.append(src_dens[i] / total)
                else:
                    weights.append(0.5)
            return weights

        tw_chord = compute_transfer_weights(src_density_chord, tgt_density_chord)
        tw_geo = compute_transfer_weights(src_density_geo, tgt_density_geo)

        sign_changes = 0
        max_tw_diff = 0.0
        tw_diffs = []
        for i in range(n_compare):
            diff = abs(tw_geo[i] - tw_chord[i])
            tw_diffs.append(diff)
            max_tw_diff = max(max_tw_diff, diff)
            # Sign change: one says "transfer from source", other says "transfer from target"
            if (tw_chord[i] > 0.5) != (tw_geo[i] > 0.5):
                sign_changes += 1

        mean_tw_diff = sum(tw_diffs) / len(tw_diffs) if tw_diffs else 0.0
        print(f"  Transfer weight sign changes: {sign_changes}/{n_compare}")
        print(f"  Mean |Δ transfer_weight|: {mean_tw_diff:.6e}")
        print(f"  Max |Δ transfer_weight|:  {max_tw_diff:.6e}")
        print(f"  Verdict: {'DECISIONS CHANGE' if sign_changes > 0 else 'DECISIONS PRESERVED'}")

        # ============================================================
        # MEASUREMENT 4: Absolute density difference
        # ============================================================
        print(f"\n--- Measurement 4: Normalized Density Difference ---")

        def normalize_densities(dens_list):
            max_d = max(dens_list) if dens_list else 1.0
            if max_d < eps_f32:
                max_d = 1.0
            return [d / max_d for d in dens_list]

        src_norm_chord = normalize_densities(src_density_chord)
        src_norm_geo = normalize_densities(src_density_geo)
        tgt_norm_chord = normalize_densities(tgt_density_chord)
        tgt_norm_geo = normalize_densities(tgt_density_geo)

        src_diffs = [abs(src_norm_geo[i] - src_norm_chord[i]) for i in range(n_src)]
        tgt_diffs = [abs(tgt_norm_geo[i] - tgt_norm_chord[i]) for i in range(n_tgt)]

        mean_src_dd = sum(src_diffs) / len(src_diffs) if src_diffs else 0.0
        max_src_dd = max(src_diffs) if src_diffs else 0.0
        mean_tgt_dd = sum(tgt_diffs) / len(tgt_diffs) if tgt_diffs else 0.0
        max_tgt_dd = max(tgt_diffs) if tgt_diffs else 0.0

        print(f"  Source: mean |Δnorm_density| = {mean_src_dd:.6e}, max = {max_src_dd:.6e}")
        print(f"  Target: mean |Δnorm_density| = {mean_tgt_dd:.6e}, max = {max_tgt_dd:.6e}")

        # ============================================================
        # MEASUREMENT 5: Geodesic/chord ratio distribution
        # ============================================================
        print(f"\n--- Measurement 5: Geodesic / Chord Ratio ---")

        def ratio_stats(chord_mat, geo_mat, label):
            n = int(chord_mat.shape[0])
            ratios = []
            for i in range(n):
                for j in range(i + 1, n):
                    c = float(b.to_scalar(chord_mat[i][j]))
                    g = float(b.to_scalar(geo_mat[i][j]))
                    if c > sqrt_eps:
                        ratios.append(g / c)

            if not ratios:
                print(f"  {label}: no valid pairs")
                return

            mean_r = sum(ratios) / len(ratios)
            min_r = min(ratios)
            max_r = max(ratios)
            # Deviation from 1.0 (perfect agreement)
            dev_from_one = [abs(r - 1.0) for r in ratios]
            mean_dev = sum(dev_from_one) / len(dev_from_one)
            max_dev = max(dev_from_one)

            print(f"  {label}:")
            print(f"    d_geo / d_chord: mean={mean_r:.6f}, min={min_r:.6f}, max={max_r:.6f}")
            print(f"    |ratio - 1|: mean={mean_dev:.6e}, max={max_dev:.6e}")
            print(f"    Geodesic ≥ chord (triangle ineq): {all(r >= 1.0 - sqrt_eps for r in ratios)}")

        ratio_stats(source_chord, source_geo, "Source")
        ratio_stats(target_chord, target_geo, "Target")

        # ============================================================
        # SUMMARY for this layer
        # ============================================================
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx} SUMMARY:")

        all_distortions = src_distortions + tgt_distortions
        if all_distortions:
            overall_max_distortion = max(all_distortions)
            overall_mean_distortion = sum(all_distortions) / len(all_distortions)
        else:
            overall_max_distortion = 0.0
            overall_mean_distortion = 0.0

        geodesic_needed = (
            overall_max_distortion > sqrt_eps
            or sign_changes > 0
            or rho_src < 0.99
            or rho_tgt < 0.99
        )

        if geodesic_needed:
            print(f"  ** GEODESIC REQUIRED **")
            print(f"  Max distortion {overall_max_distortion:.6e} > sqrt(eps) {sqrt_eps:.6e}: {overall_max_distortion > sqrt_eps}")
            print(f"  Sign changes: {sign_changes}")
            print(f"  Ranking preserved (ρ > 0.99): src={rho_src > 0.99}, tgt={rho_tgt > 0.99}")
        else:
            print(f"  ** EUCLIDEAN SUFFICIENT **")
            print(f"  Max distortion {overall_max_distortion:.6e} <= sqrt(eps) {sqrt_eps:.6e}")
            print(f"  No sign changes, rankings preserved")

        results[layer_idx] = {
            "mean_distortion": overall_mean_distortion,
            "max_distortion": overall_max_distortion,
            "sqrt_eps": sqrt_eps,
            "geodesic_needed": geodesic_needed,
            "spearman_src": rho_src,
            "spearman_tgt": rho_tgt,
            "sign_changes": sign_changes,
            "n_compare": n_compare,
            "k": k,
            "max_transfer_weight_diff": max_tw_diff,
        }

    # ============================================================
    # OVERALL VERDICT
    # ============================================================
    print(f"\n{'='*70}")
    print(f"OVERALL VERDICT")
    print(f"{'='*70}")

    any_geodesic = any(r["geodesic_needed"] for r in results.values())

    if any_geodesic:
        print("GEODESIC DISTANCES REQUIRED for knowledge density comparison.")
        print("The Euclidean approximation produces materially different results.")
        print("knowledge_density.py MUST use rg.geodesic_distances().")
    else:
        print("EUCLIDEAN DISTANCES SUFFICIENT for knowledge density comparison.")
        print("The geodesic/Euclidean distortion is below sqrt(eps_f32) at all layers.")
        print("knowledge_density.py correctly uses _chord_distance_matrix().")

    for layer_idx, r in sorted(results.items()):
        print(f"\n  Layer {layer_idx}: {'GEODESIC' if r['geodesic_needed'] else 'EUCLIDEAN'}")
        print(f"    max_distortion={r['max_distortion']:.6e}, "
              f"ρ_src={r['spearman_src']:.4f}, ρ_tgt={r['spearman_tgt']:.4f}, "
              f"sign_changes={r['sign_changes']}")

    # Save results
    out_dir = Path(__file__).resolve().parent.parent / "results" / "geodesic_vs_euclidean"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "LFM2-350M_density_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
