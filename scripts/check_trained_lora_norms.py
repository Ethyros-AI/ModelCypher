#!/usr/bin/env python3
"""Check the spectral norms of a trained LoRA adapter.

The key insight: ||B@A||_spectral grows during training.
We need to know typical trained values to understand the rank constraint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# We deleted all adapters, but let me check what the check-scale output showed
# The δ_spectral values were in the output

# From the check-scale output for geometric-awareness-8B-v1:
# Looking at v_proj and k_proj layers...

print("TRAINED LORA DELTA SPECTRAL NORMS")
print("=" * 70)
print("(From check-scale output on geometric-awareness-8B-v1)")
print()

# Sample values from the JSON output (v_proj and k_proj only)
# These are the delta_spectral values = ||B@A||_spectral
trained_norms = {
    "v_proj": [
        1.0346,  # layer 3
        0.8096,  # layer 4
        0.9084,  # layer 5
        0.7186,  # layer 6
        1.1208,  # layer 7
        0.7566,  # layer 8
        0.8420,  # layer 9
    ],
    "k_proj": [
        0.7703,  # layer 3
        1.0596,  # layer 4
        0.6849,  # layer 5
        0.7414,  # layer 6
        0.8148,  # layer 7
        1.2342,  # layer 8
        1.8970,  # layer 9
    ],
}

import numpy as np

for proj, norms in trained_norms.items():
    mean = np.mean(norms)
    std = np.std(norms)
    print(f"{proj}:")
    print(f"  ||B@A||_spectral: mean={mean:.3f}, std={std:.3f}, range=[{min(norms):.3f}, {max(norms):.3f}]")

print()
print("=" * 70)
print("COMPARISON: Random Init vs Trained")
print("-" * 70)
print()
print("Random init (std=0.01, rank=16): ||B@A|| ≈ 0.23")
print("Trained adapter:                 ||B@A|| ≈ 0.7-1.9")
print()
print("Training increased ||B@A|| by 3-8×")
print()
print("=" * 70)
print("REVISED GEOMETRIC CONSTRAINT")
print("-" * 70)
print()

# σ_k values
sigma_k_v = 0.46
sigma_k_k = 0.30

# With trained norms
mean_norm_v = np.mean(trained_norms["v_proj"])
mean_norm_k = np.mean(trained_norms["k_proj"])

scale_bound_v = sigma_k_v / mean_norm_v
scale_bound_k = sigma_k_k / mean_norm_k

print(f"v_proj: σ_k={sigma_k_v:.2f}, ||B@A||={mean_norm_v:.2f} → scale ≤ {scale_bound_v:.3f}")
print(f"k_proj: σ_k={sigma_k_k:.2f}, ||B@A||={mean_norm_k:.2f} → scale ≤ {scale_bound_k:.3f}")
print()
print("Standard scale = alpha/rank = 32/16 = 2.0")
print()
print(f"v_proj: 2.0 / {scale_bound_v:.3f} = {2.0/scale_bound_v:.1f}× over bound")
print(f"k_proj: 2.0 / {scale_bound_k:.3f} = {2.0/scale_bound_k:.1f}× over bound")
print()
print("=" * 70)
print("THE GEOMETRIC ANSWER FOR RANK")
print("-" * 70)
print()
print("The constraint is NOT directly on rank.")
print("The constraint is on ||scale × B @ A||_spectral ≤ σ_k")
print()
print("Rank affects ||B@A|| indirectly - higher rank = more capacity = ")
print("potentially larger trained norm. But the DIRECT constraint is on")
print("the final delta magnitude, not rank.")
print()
print("WHAT GEOMETRY ACTUALLY TELLS US:")
print()
print("1. TARGET MODULES: v_proj, k_proj (not q_proj, o_proj)")
print("   Reason: σ_k is 100× larger for v/k than q/o")
print()
print("2. SCALE: ≤ σ_k / ||B@A||_trained")
print("   For v_proj: ≤ 0.5")
print("   For k_proj: ≤ 0.3")
print()
print("3. RANK: Constrained by tail subspace energy, not scale")
print("   For v/k_proj: ≤ 300 (tail dimensions)")
print("   Practical: 32-128 (allows capacity without dominating tail)")
print()
print("4. THE MISSING PIECE: Constrain ||B@A|| during training")
print("   Add regularization: loss += λ × ||B@A||_spectral")
print("   Or use spectral normalization on LoRA weights")
