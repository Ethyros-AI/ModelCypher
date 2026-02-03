#!/usr/bin/env python3
"""Analyze singular value spectrum for natural rank structure.

Question: What does the geometry tell us about correct LoRA rank?

Looking for:
1. Spectral gaps (sudden drops in singular values)
2. Elbow points (change in decay rate)
3. Null space dimension (SVs below noise floor)
4. Natural rank structure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import initialize_default_backend
initialize_default_backend()


def analyze_spectrum(model_path: str, layer_idx: int = 0):
    """Analyze singular value spectrum of attention projections."""
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load

    print(f"Loading model: {model_path}")
    model, _ = load(model_path)
    base_model = getattr(model, "model", model)
    layer = base_model.layers[layer_idx]
    attn = layer.self_attn

    sqrt_eps = np.sqrt(np.finfo(np.float32).eps)

    print(f"\nAnalyzing layer {layer_idx} singular value spectra...")
    print("=" * 70)

    for proj_name in ["v_proj", "k_proj"]:  # Focus on the good targets
        proj = getattr(attn, proj_name)
        W = proj.weight
        W_f32 = W.astype(mx.float32)
        mx.eval(W_f32)
        W_np = np.array(W_f32.tolist(), dtype=np.float32)

        # Full SVD
        _, S, _ = np.linalg.svd(W_np, full_matrices=False)

        sigma_max = S[0]
        threshold = sqrt_eps * sigma_max

        print(f"\n{proj_name} ({W_np.shape[0]}×{W_np.shape[1]}):")
        print("-" * 50)

        # 1. Basic stats
        print(f"σ_max: {sigma_max:.4f}")
        print(f"σ_min: {S[-1]:.6f}")
        print(f"Noise threshold (√ε × σ_max): {threshold:.6f}")

        # 2. Look for spectral gaps (ratio between consecutive SVs)
        ratios = S[:-1] / S[1:]
        max_gap_idx = np.argmax(ratios)
        max_gap_ratio = ratios[max_gap_idx]

        print(f"\nSpectral gap analysis:")
        print(f"  Largest gap: σ_{max_gap_idx}/σ_{max_gap_idx+1} = {max_gap_ratio:.2f}× at index {max_gap_idx}")

        # Find gaps > 1.5×
        significant_gaps = np.where(ratios > 1.5)[0]
        if len(significant_gaps) > 0:
            print(f"  Gaps > 1.5×: indices {significant_gaps[:10]}...")  # First 10

        # 3. Look for elbow (change in decay rate)
        # Use second derivative of log(S)
        log_S = np.log(S + 1e-10)
        d1 = np.diff(log_S)  # First derivative
        d2 = np.diff(d1)     # Second derivative
        elbow_idx = np.argmax(np.abs(d2)) + 1

        print(f"\nElbow analysis (max curvature in log-spectrum):")
        print(f"  Elbow at index: {elbow_idx}")
        print(f"  σ at elbow: {S[elbow_idx]:.6f}")

        # 4. Effective rank at various thresholds
        print(f"\nEffective rank at different thresholds:")
        for thresh_mult in [1.0, 0.1, 0.01, 0.001]:
            thresh = thresh_mult * sigma_max
            eff_rank = np.sum(S > thresh)
            print(f"  σ > {thresh_mult}×σ_max: rank = {eff_rank}")

        # 5. Energy distribution (what fraction of Frobenius norm is in top-k SVs)
        total_energy = np.sum(S**2)
        print(f"\nEnergy distribution (cumulative % of ||W||²_F):")
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if k <= len(S):
                energy_k = np.sum(S[:k]**2) / total_energy * 100
                print(f"  Top {k:3d} SVs: {energy_k:5.1f}%")

        # 6. Natural rank candidates
        print(f"\nNatural rank candidates:")

        # a) 90% energy
        cumsum = np.cumsum(S**2) / total_energy
        rank_90 = np.searchsorted(cumsum, 0.90) + 1
        print(f"  90% energy: rank = {rank_90}")

        # b) 99% energy
        rank_99 = np.searchsorted(cumsum, 0.99) + 1
        print(f"  99% energy: rank = {rank_99}")

        # c) Above noise floor
        rank_noise = np.sum(S > threshold)
        print(f"  Above √ε×σ_max: rank = {rank_noise}")

        # d) Elbow point
        print(f"  Elbow point: rank = {elbow_idx}")

        # 7. The geometry-derived rank recommendation
        print(f"\n>>> GEOMETRIC RANK RECOMMENDATION:")

        # The insight: LoRA should add capacity in the "tail" of the spectrum
        # where the base model has less energy. This is the complement of the
        # dominant subspace.

        # If 90% of energy is in top-k, then the remaining 10% is spread across
        # n-k dimensions. LoRA rank should be related to this.

        tail_dims = len(S) - rank_90
        print(f"  Dominant subspace (90% energy): {rank_90} dims")
        print(f"  Tail subspace (10% energy): {tail_dims} dims")
        print(f"  Suggested LoRA rank: {min(tail_dims, 64)} (capped at 64)")


def main():
    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    analyze_spectrum(model_path, layer_idx=0)
    print("\n" + "=" * 70)
    analyze_spectrum(model_path, layer_idx=15)  # Middle layer


if __name__ == "__main__":
    main()
