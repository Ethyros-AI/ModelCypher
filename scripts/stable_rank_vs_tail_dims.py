#!/usr/bin/env python3
"""Compare SR-LoRA stable rank vs ModelCypher tail_dims.

SR-LoRA (Bian et al. 2024): uses stable_rank = ||W||²_F / ||W||²_2
  as rank selection criterion (collapse condition: when α/√r > 1).
ModelCypher: uses tail_dims = full_rank - floor(Shannon effective rank)
  as null-space capacity (structural rank from spectral entropy).

This script computes both for every weight matrix in a model and outputs
a comparison table showing the relationship.
"""

import sys
import math
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

from modelcypher.core.domain.training.geometric_lora import compute_layer_geometry
from modelcypher.backends.mlx_backend import MLXBackend


def compute_stable_rank_from_svd(S):
    """||W||²_F / ||W||²_2 = sum(σ²) / σ_max²."""
    S_sq = S * S
    frobenius_sq = float(mx.sum(S_sq).item())
    spectral_sq = float(S[0].item()) ** 2
    if spectral_sq == 0:
        return 0.0
    return frobenius_sq / spectral_sq


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    )
    print(f"Model: {model_path.name}")
    print()

    model, tokenizer = load(str(model_path))
    backend = MLXBackend()

    # Collect all weight matrices
    results = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight
        if W.ndim != 2:
            continue

        key = name
        shape = (int(W.shape[0]), int(W.shape[1]))
        full_rank = min(shape)

        # Compute geometry (includes SVD)
        W_f32 = W.astype(mx.float32)
        mx.eval(W_f32)
        S = mx.linalg.svd(W_f32, compute_uv=False, stream=mx.cpu)
        mx.eval(S)

        # Stable rank: ||W||²_F / ||W||²_2
        stable_rank = compute_stable_rank_from_svd(S)

        # Shannon effective rank: exp(H(σ²))
        eigvals = S * S
        sum_eig = float(mx.sum(eigvals).item())
        if sum_eig > 0:
            p = eigvals / sum_eig
            p_safe = mx.where(p > 1e-30, p, mx.full(p.shape, 1e-30))
            entropy = float(mx.sum(-p * mx.log(p_safe)).item())
            shannon_eff_rank = math.exp(entropy)
        else:
            shannon_eff_rank = 0.0

        structural_rank = max(1, min(math.floor(shannon_eff_rank), int(S.shape[0]) - 1))
        tail_dims = max(0, full_rank - structural_rank)

        results.append({
            "name": key,
            "shape": shape,
            "full_rank": full_rank,
            "stable_rank": stable_rank,
            "shannon_eff_rank": shannon_eff_rank,
            "structural_rank": structural_rank,
            "tail_dims": tail_dims,
            "targetable": tail_dims > 0,
            "sigma_max": float(S[0].item()),
            "sigma_min": float(S[-1].item()),
        })

    # Print comparison table
    print(f"{'Layer':<45} {'Shape':>12} {'StableR':>8} {'ShannonR':>9} {'tail_dims':>10} {'Target':>7}")
    print("-" * 95)

    targetable_stable_ranks = []
    non_targetable_stable_ranks = []

    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        target_str = "YES" if r["targetable"] else "no"
        print(
            f"{r['name']:<45} {shape_str:>12} {r['stable_rank']:>8.1f} "
            f"{r['shannon_eff_rank']:>9.1f} {r['tail_dims']:>10} {target_str:>7}"
        )
        if r["targetable"]:
            targetable_stable_ranks.append(r["stable_rank"])
        else:
            non_targetable_stable_ranks.append(r["stable_rank"])

    # Summary statistics
    print()
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(f"Total layers: {len(results)}")
    print(f"Targetable (tail_dims > 0): {sum(1 for r in results if r['targetable'])}")
    print(f"Non-targetable: {sum(1 for r in results if not r['targetable'])}")
    print()

    if targetable_stable_ranks:
        print(f"Targetable layers stable_rank: min={min(targetable_stable_ranks):.1f}, "
              f"max={max(targetable_stable_ranks):.1f}, "
              f"mean={sum(targetable_stable_ranks)/len(targetable_stable_ranks):.1f}")
    if non_targetable_stable_ranks:
        print(f"Non-targetable layers stable_rank: min={min(non_targetable_stable_ranks):.1f}, "
              f"max={max(non_targetable_stable_ranks):.1f}, "
              f"mean={sum(non_targetable_stable_ranks)/len(non_targetable_stable_ranks):.1f}")

    # Correlation analysis
    print()
    print("CORRELATION: stable_rank vs tail_dims")
    print("-" * 50)

    # Only for layers with tail_dims > 0
    if targetable_stable_ranks:
        td_vals = [r["tail_dims"] for r in results if r["targetable"]]
        sr_vals = [r["stable_rank"] for r in results if r["targetable"]]

        # Pearson correlation
        n = len(td_vals)
        mean_td = sum(td_vals) / n
        mean_sr = sum(sr_vals) / n
        cov = sum((t - mean_td) * (s - mean_sr) for t, s in zip(td_vals, sr_vals)) / n
        std_td = (sum((t - mean_td) ** 2 for t in td_vals) / n) ** 0.5
        std_sr = (sum((s - mean_sr) ** 2 for s in sr_vals) / n) ** 0.5
        if std_td > 0 and std_sr > 0:
            pearson_r = cov / (std_td * std_sr)
            print(f"Pearson r (targetable only): {pearson_r:.4f}")
        else:
            print("Cannot compute Pearson r (zero variance)")

    # SR-LoRA vs ModelCypher rank selection comparison
    print()
    print("SR-LoRA RANK SELECTION COMPARISON")
    print("-" * 60)
    print(f"{'Layer':<45} {'SR-LoRA_r':>10} {'MC_r':>6} {'Match':>6}")
    print("-" * 60)

    matches = 0
    for r in results:
        # SR-LoRA rank: they typically use stable_rank / some_factor or min(stable_rank, max_rank)
        # For comparison, use floor(stable_rank) as their implied rank
        sr_lora_rank = max(1, min(math.floor(r["stable_rank"]), r["full_rank"]))
        mc_rank = r["tail_dims"] if r["targetable"] else 0

        match = "~" if abs(sr_lora_rank - mc_rank) <= max(1, mc_rank * 0.2) else ""
        if mc_rank > 0:
            print(
                f"{r['name']:<45} {sr_lora_rank:>10} {mc_rank:>6} {match:>6}"
            )
            if match:
                matches += 1

    targetable = sum(1 for r in results if r["targetable"])
    if targetable > 0:
        print(f"\nAgreement (within 20%): {matches}/{targetable} = {100*matches/targetable:.0f}%")

    print()
    print("KEY INSIGHT:")
    print("  stable_rank = ||W||²_F / ||W||²_2 (energy concentration)")
    print("  tail_dims = full_rank - floor(exp(H(σ²))) (structural null-space)")
    print("  These measure DIFFERENT things:")
    print("    stable_rank → how concentrated the energy is (all SVs contribute)")
    print("    tail_dims → how many dimensions are structurally unused (entropy-based cutoff)")


if __name__ == "__main__":
    main()
