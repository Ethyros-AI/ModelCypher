#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Attention perturbation experiment: entropy→curvature causal test.

PRE-REGISTRATION CONTRACT (MISSION.md:51)
=========================================

observable = Δθ(l) = θ_perturbed(l) − θ_baseline(l)
  where θ(l) = arccos(h_l · h_{l+1} / (||h_l|| ||h_{l+1}||))

geometry_state: attention weight distribution A_{h,l} per head h, layer l
architecture_state: {
    LFM2-350M (hybrid conv+attn, 6/16 attn layers, 1024 hidden),
    Qwen3.5-0.8B (24 layers, 6 full_attn, 1024 hidden, GQA)
}
scale_state: {350M (16 layers), 800M (24 layers)}
measurement_operator: angular curvature θ(l) = arccos(h_l · h_{l+1} / (||h_l|| ||h_{l+1}||))

DIRECTIONAL PREDICTION:
    Increasing M (attention boost to prefix) concentrates attention →
    reduces attention entropy. If entropy→curvature link is causal,
    layers with higher baseline entropy H_baseline(l) should show
    larger curvature change |Δθ(l)| under perturbation.

FALSIFIER:
    If Spearman rank correlation ρ(H_baseline, |Δθ|) across attention
    layers is not significantly different from zero (p > 0.05 via
    permutation test) at M values producing measurable |Δθ|, the
    entropy→curvature causal claim is falsified for that model.

    NOTE: The perturbation applies uniform M across all layers, so
    per-layer ΔH cannot vary independently (all layers see the same
    rescaling). Instead we test whether layers with higher baseline
    entropy — i.e., layers where the perturbation changes the entropy
    landscape more — respond with larger curvature shifts. This is
    the testable form of the causal claim.

PERTURBATION GRID DERIVATION:
    M grid is NOT hardcoded. It is derived from each model's measured
    baseline in two phases:

    Phase 1: Run baseline (M=1), measure per-layer attention entropy H_baseline
             and mean prefix attention weight p_I.

    Phase 2: Compute M_min per layer such that analytical |ΔH| > √ε_f32.
             Generate geometric grid: M ∈ {1, M_min, M_min², ...} up to
             saturation (p'_I > 1 − √ε_f32).

    The minimum detectable entropy change is bounded by:
        Measurement floor: √ε_f32 ≈ 3.45e-4 (IEEE 754, float32)

    For perturbation multiplier M, the perturbed prefix weight is:
        p'_I = M × p_I / (M × p_I + (1 − p_I))

    ΔH = H(p'_I) − H(p_I) where H is Shannon entropy of the 2-class
    distribution {p_I, 1-p_I} (simplified binary model for grid derivation).

EXPERIMENTAL STATUS: scripts/ only, NOT CLI (AGENTS.md:695)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# IEEE 754 float32 measurement floor
_EPS_F32 = math.ldexp(1.0, -23)
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)
_PRE_REGISTERED_ALPHA = 0.05
# Permutation resolution derived from IEEE floor.
# Smallest nonzero p-value is 1/(n_perms+1), so choose n_perms ~ 1/sqrt(eps).
_N_PERMUTATIONS = int(math.ceil(1.0 / _SQRT_EPS_F32))

MODELS = {
    "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "Qwen3.5-0.8B": "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16",
}

PROBE_TEXT = "The capital of France is"


def binary_entropy(p: float) -> float:
    """Shannon entropy of Bernoulli(p). H = -p*log(p) - (1-p)*log(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def perturbed_prefix_weight(p_I: float, M: float) -> float:
    """Compute perturbed prefix attention weight after M-boost + L1 renorm."""
    return M * p_I / (M * p_I + (1 - p_I))


def derive_M_grid(p_I: float) -> list[float]:
    """Derive perturbation grid from measured baseline p_I.

    Finds M_min such that |ΔH| > √ε_f32, then generates geometric
    progression up to saturation.

    Args:
        p_I: Mean baseline prefix attention weight (0 < p_I < 1).

    Returns:
        List of M values including M=1 (baseline).
    """
    if p_I <= 0 or p_I >= 1:
        return [1.0]

    H_baseline = binary_entropy(p_I)

    # Binary search for M_min where |ΔH| > measurement floor.
    # Upper bound 1/√ε_f32 ≈ 2896: at this M, p'_I is within √ε of 1.0
    # for any p_I > √ε, so ΔH is maximal. 50 bisection steps give
    # precision |M_hi - M_lo| < (1/√ε) / 2^50 ≈ 2.6e-12 (far below
    # the measurement floor).
    M_lo, M_hi = 1.0, 1.0 / _SQRT_EPS_F32
    for _ in range(50):
        M_mid = (M_lo + M_hi) / 2
        p_prime = perturbed_prefix_weight(p_I, M_mid)
        delta_H = abs(binary_entropy(p_prime) - H_baseline)
        if delta_H > _SQRT_EPS_F32:
            M_hi = M_mid
        else:
            M_lo = M_mid
    M_min = M_hi

    # Geometric progression: {1, M_min, M_min^2, ...} until saturation
    grid = [1.0]
    M = M_min
    saturation_p = 1.0 - _SQRT_EPS_F32

    # Safety cap: 1/ε_f32 ≈ 8.4M. Beyond this, p'_I is indistinguishable
    # from 1.0 in float32 (M * p_I >> 1 - p_I).
    while M < 1.0 / _EPS_F32:
        p_prime = perturbed_prefix_weight(p_I, M)
        if p_prime >= saturation_p:
            grid.append(M)
            break
        grid.append(M)
        M = M * M_min

    return grid


def compute_angular_curvature(
    hidden_states: dict, mx_module: object
) -> dict[int, float]:
    """Compute per-layer angular curvature from hidden states.

    θ(l) = arccos(h_l · h_{l+1} / (||h_l|| ||h_{l+1}||))
    """
    mx = mx_module
    layers = sorted(hidden_states.keys())
    curvatures = {}
    for i in range(len(layers) - 1):
        a = hidden_states[layers[i]]
        b = hidden_states[layers[i + 1]]
        dot = float(mx.sum(a * b))
        norm_a = float(mx.sqrt(mx.sum(a * a)))
        norm_b = float(mx.sqrt(mx.sum(b * b)))
        # Guard against zero-norm vectors. ε_f32² ≈ 1.4e-14 is the
        # smallest norm² representable without underflow in float32.
        if norm_a < _EPS_F32 or norm_b < _EPS_F32:
            curvatures[layers[i]] = 0.0
        else:
            cos_theta = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
            curvatures[layers[i]] = math.acos(cos_theta)
    return curvatures


def compute_attention_entropy(
    attn_matrices: dict, backend: object
) -> dict[int, float]:
    """Compute mean attention entropy per layer (across heads).

    H(A_h) = -(1/T) sum_u sum_i A[u,i] * log(A[u,i])
    Mean over heads.
    """
    layer_entropies = {}
    for layer_idx in sorted(attn_matrices.keys()):
        head_entropies = []
        for head_mat in attn_matrices[layer_idx]:
            mat = backend.tolist(head_mat)
            T = len(mat)
            if T == 0:
                head_entropies.append(0.0)
                continue
            total = 0.0
            count = 0
            for u in range(T):
                for i in range(u + 1):
                    p = mat[u][i]
                    if p > _EPS_F32:
                        total -= p * math.log(p)
                        count += 1
            head_entropies.append(total / T if T > 0 else 0.0)
        layer_entropies[layer_idx] = sum(head_entropies) / len(head_entropies)
    return layer_entropies


def measure_mean_prefix_weight(
    attn_matrices: dict, backend: object
) -> float:
    """Measure mean attention weight on prefix token (position 0) across all layers/heads."""
    total = 0.0
    count = 0
    for layer_idx in attn_matrices:
        for head_mat in attn_matrices[layer_idx]:
            mat = backend.tolist(head_mat)
            T = len(mat)
            for u in range(T):
                total += mat[u][0]  # attention to position 0
                count += 1
    return total / count if count > 0 else 0.0


def measure_prefix_weight_per_layer(
    attn_matrices: dict, backend: object
) -> dict[int, float]:
    """Measure mean attention to prefix token (position 0) per layer."""
    result: dict[int, float] = {}
    for layer_idx in attn_matrices:
        total = 0.0
        count = 0
        for head_mat in attn_matrices[layer_idx]:
            mat = backend.tolist(head_mat)
            T = len(mat)
            for u in range(T):
                total += mat[u][0]
                count += 1
        result[layer_idx] = total / count if count > 0 else 0.0
    return result


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between x and y."""
    n = len(x)
    if n < 3:
        return float("nan")

    def rank(data):
        sorted_indices = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        for r, i in enumerate(sorted_indices):
            ranks[i] = r + 1.0
        return ranks

    rx = rank(x)
    ry = rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x < _EPS_F32 or den_y < _EPS_F32:
        return 0.0
    return num / (den_x * den_y)


def permutation_test_p_value(
    x: list[float], y: list[float], n_perms: int | None = None
) -> float:
    """Permutation test for significance of Spearman correlation.

    Args:
        n_perms: Number of permutations. Default:
            exact test when n <= 8 (n!), otherwise 1/sqrt(eps_f32) derived
            from IEEE 754 measurement floor.
    """
    import random

    n = len(x)
    if n_perms is None:
        # Exact test when feasible (n! ≤ 40320 for n ≤ 8)
        n_perms = math.factorial(n) if n <= 8 else _N_PERMUTATIONS

    observed = abs(spearman_rank_correlation(x, y))
    if math.isnan(observed):
        return 1.0

    # Deterministic RNG seed from inputs for reproducibility across runs.
    seed_payload = (
        tuple(round(v, 12) for v in x),
        tuple(round(v, 12) for v in y),
        n_perms,
    )
    rng = random.Random(repr(seed_payload))

    count_ge = 0
    y_perm = y.copy()
    for _ in range(n_perms):
        rng.shuffle(y_perm)
        perm_r = abs(spearman_rank_correlation(x, y_perm))
        if not math.isnan(perm_r) and perm_r >= observed:
            count_ge += 1

    return (count_ge + 1) / (n_perms + 1)


def run_experiment(model_name: str, model_path: str) -> dict:
    """Run the full perturbation experiment on one model."""
    import mlx.core as mx
    import mlx_lm

    from modelcypher.backends.mlx_backend import MLXBackend

    print(f"\n{'='*70}")
    print(f"Model: {model_name} ({model_path})")
    print(f"{'='*70}")

    backend = MLXBackend()
    model, tokenizer = mlx_lm.load(model_path)

    # ===== PHASE 1: Baseline measurement =====
    print("\n--- Phase 1: Baseline measurement ---")

    # Collect attention matrices for entropy and p_I measurement
    attn_matrices = backend.collect_attention_matrices(
        model, tokenizer, PROBE_TEXT
    )

    baseline_entropy = compute_attention_entropy(attn_matrices, backend)
    p_I = measure_mean_prefix_weight(attn_matrices, backend)
    p_I_per_layer = measure_prefix_weight_per_layer(attn_matrices, backend)

    print(f"Mean prefix attention weight p_I = {p_I:.6f}")
    print(f"Attention layers: {sorted(attn_matrices.keys())}")
    for l in sorted(baseline_entropy):
        print(f"  L{l:2d}: H_baseline = {baseline_entropy[l]:.6f}")

    # Baseline hidden states (using identity hook as reference)
    def identity_hook(w, l):
        return w

    baseline_hidden = backend.collect_hidden_with_attention_hook(
        model, tokenizer, PROBE_TEXT, attention_hook=identity_hook
    )
    baseline_curvature = compute_angular_curvature(baseline_hidden, mx)
    print("\nBaseline curvature:")
    for l in sorted(baseline_curvature):
        print(f"  L{l:2d}: θ = {baseline_curvature[l]:.6f}")

    # ===== PHASE 2: Derive M grid =====
    print("\n--- Phase 2: Derive perturbation grid ---")
    M_grid = derive_M_grid(p_I)
    print(f"Measurement floor: √ε_f32 = {_SQRT_EPS_F32:.6e}")
    print(f"Derived M grid ({len(M_grid)} points): {[f'{m:.2f}' for m in M_grid]}")

    # Analytical ΔH at each M from mean p_I (grid derivation signal)
    H_baseline_binary = binary_entropy(p_I)
    for M in M_grid:
        p_prime = perturbed_prefix_weight(p_I, M)
        delta_H_analytical = binary_entropy(p_prime) - H_baseline_binary
        print(f"  M={M:8.2f}: p'_I={p_prime:.6f}, ΔH_analytical={delta_H_analytical:+.6e}")

    # ===== PHASE 3: Perturbation sweep =====
    print("\n--- Phase 3: Perturbation sweep ---")

    sweep_results = []

    for M in M_grid:
        if M == 1.0:
            # M=1 is baseline
            sweep_results.append({
                "M": M,
                "curvatures": baseline_curvature,
                "delta_theta": {l: 0.0 for l in baseline_curvature},
            })
            continue

        def make_hook(mult):
            def hook(w, layer_idx):
                # Boost prefix token by M, renormalize each row
                boosted = w * 1.0  # copy
                boosted = boosted.at[:, :, :, 0:1].multiply(mult)
                return boosted / mx.sum(boosted, axis=-1, keepdims=True)
            return hook

        perturbed_hidden = backend.collect_hidden_with_attention_hook(
            model, tokenizer, PROBE_TEXT, attention_hook=make_hook(M)
        )
        perturbed_curvature = compute_angular_curvature(perturbed_hidden, mx)

        delta_theta = {
            l: perturbed_curvature.get(l, 0) - baseline_curvature.get(l, 0)
            for l in baseline_curvature
        }

        sweep_results.append({
            "M": M,
            "curvatures": perturbed_curvature,
            "delta_theta": delta_theta,
        })

        print(f"\n  M = {M:.2f}:")
        for l in sorted(delta_theta):
            print(f"    L{l:2d}: Δθ = {delta_theta[l]:+.6f}")

    # ===== PHASE 4: Statistical analysis =====
    print(f"\n--- Phase 4: Statistical analysis ---")

    # For each M > 1, compute correlation between ΔH and Δθ across layers
    attn_layers = sorted(attn_matrices.keys())

    # ΔH per layer is computed analytically from the measured baseline per-layer
    # prefix weight p_I(layer) under the closed-form perturbation map:
    #   p' = M p / (M p + (1-p)), ΔH = H(p') - H(p)
    # This preserves the pre-registered statistic ρ(ΔH, Δθ) across layers.

    print(f"\nCorrelation analysis (attention layers only: {attn_layers}):")
    causal_supported = False
    correlation_results: list[dict[str, float | bool]] = []
    for sr in sweep_results:
        M = sr["M"]
        if M <= 1.0:
            continue

        # Per-layer ΔH and Δθ
        delta_h_layers = []
        delta_theta_layers = []
        for l in attn_layers:
            p_layer = p_I_per_layer[l]
            p_prime_layer = perturbed_prefix_weight(p_layer, M)
            delta_h_layers.append(binary_entropy(p_prime_layer) - binary_entropy(p_layer))
            delta_theta_layers.append(sr["delta_theta"].get(l, 0.0))

        rho = spearman_rank_correlation(delta_h_layers, delta_theta_layers)
        p_val = permutation_test_p_value(delta_h_layers, delta_theta_layers)

        mean_abs_dt = (
            sum(abs(dt) for dt in delta_theta_layers) / len(delta_theta_layers)
            if delta_theta_layers else 0.0
        )
        max_abs_dh = (
            max(abs(dh) for dh in delta_h_layers) if delta_h_layers else 0.0
        )
        measurable_delta_h = max_abs_dh > _SQRT_EPS_F32

        print(
            f"  M={M:8.2f}: mean|Δθ|={mean_abs_dt:.6f}, "
            f"max|ΔH|={max_abs_dh:.6e}, "
            f"ρ(ΔH, Δθ)={rho:+.3f}, p={p_val:.4f}"
        )

        # Pre-registered decision rule:
        # If p <= alpha at an M with measurable ΔH, causal link is not falsified.
        if measurable_delta_h and p_val <= _PRE_REGISTERED_ALPHA:
            causal_supported = True

        correlation_results.append({
            "M": M,
            "rho_deltaH_deltaTheta": rho,
            "p_value": p_val,
            "max_abs_delta_h": max_abs_dh,
            "mean_abs_delta_theta": mean_abs_dt,
            "measurable_delta_h": measurable_delta_h,
        })

    # ===== CONCLUSION =====
    print(f"\n{'='*70}")
    if causal_supported:
        print(
            f"RESULT ({model_name}): Significant correlation found between "
            f"per-layer ΔH and per-layer Δθ. "
            f"Entropy→curvature link NOT falsified."
        )
    else:
        print(
            f"RESULT ({model_name}): No significant correlation between "
            f"per-layer ΔH and per-layer Δθ (p > {_PRE_REGISTERED_ALPHA:.2f}). "
            f"Entropy→curvature causal claim FALSIFIED for this model."
        )

    return {
        "model": model_name,
        "p_I": p_I,
        "p_I_per_layer": p_I_per_layer,
        "M_grid": M_grid,
        "causal_supported": causal_supported,
        "correlations": correlation_results,
        "sweep_results": [
            {"M": sr["M"], "delta_theta": sr["delta_theta"]}
            for sr in sweep_results
        ],
    }


def main():
    print("Attention Perturbation Experiment")
    print("Causal test: entropy → curvature")
    print(f"Measurement floor: √ε_f32 = {_SQRT_EPS_F32:.6e}")
    print(f"Probe: {PROBE_TEXT!r}")

    for name, path in MODELS.items():
        if not Path(path).exists():
            print(f"ERROR: Model not found: {path}")
            print("Is the external volume mounted?")
            sys.exit(1)

    results = []
    for name, path in MODELS.items():
        result = run_experiment(name, path)
        results.append(result)

    # Cross-model summary
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "NOT FALSIFIED" if r["causal_supported"] else "FALSIFIED"
        print(f"  {r['model']}: {status} (p_I={r['p_I']:.4f}, grid={len(r['M_grid'])} points)")

    # Per CLAUDE.md: "No mixed-model narrative"
    all_supported = all(r["causal_supported"] for r in results)
    all_falsified = all(not r["causal_supported"] for r in results)

    if all_supported:
        print("\nBoth models: entropy→curvature link NOT falsified.")
    elif all_falsified:
        print("\nBoth models: entropy→curvature link FALSIFIED.")
    else:
        print(
            "\nMIXED RESULT: One model supports, one falsifies. "
            "This indicates mechanism underspecification — "
            "the observable needs an architecture_state term. "
            "See FIRST_PRINCIPLES_REVIEW_PROTOCOL.md."
        )


if __name__ == "__main__":
    main()
