#!/usr/bin/env python3
"""Derive the correct MLP scale correction from geometry, not heuristics.

The question: What scale factor makes merged MLP output match target MLP output?

This is measurable:
1. Run merged MLP on target hidden states
2. Run target MLP on same inputs
3. correction = ||target_output|| / ||merged_output||
"""

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import json


def load_model_weights(model_path: str) -> dict[str, mx.array]:
    """Load model weights from safetensors using mlx."""
    model_path = Path(model_path)
    weights = {}

    for sf_file in model_path.glob("*.safetensors"):
        w = mx.load(str(sf_file))
        weights.update(w)

    return weights


def silu(x: mx.array) -> mx.array:
    """SiLU activation: x * sigmoid(x)"""
    return x * mx.sigmoid(x)


def run_mlp(hidden: mx.array, w1: mx.array, w2: mx.array, w3: mx.array) -> mx.array:
    """Run SwiGLU MLP: down(silu(gate(x)) * up(x))"""
    gate_out = hidden @ w1.T  # [batch, inter]
    up_out = hidden @ w3.T    # [batch, inter]
    intermediate = silu(gate_out) * up_out
    output = intermediate @ w2.T  # [batch, hidden]
    return output


def main():
    # Paths
    merged_path = "/Volumes/CodeCypher/models/merged/qwen-lfm2-joint-scale"
    target_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"

    print("Loading models...")
    merged_weights = load_model_weights(merged_path)
    target_weights = load_model_weights(target_path)

    # Layer 2 MLP weights
    layer = 2
    w1_key = f"model.layers.{layer}.feed_forward.w1.weight"
    w2_key = f"model.layers.{layer}.feed_forward.w2.weight"
    w3_key = f"model.layers.{layer}.feed_forward.w3.weight"

    merged_w1 = merged_weights[w1_key]
    merged_w2 = merged_weights[w2_key]
    merged_w3 = merged_weights[w3_key]

    target_w1 = target_weights[w1_key]
    target_w2 = target_weights[w2_key]
    target_w3 = target_weights[w3_key]

    print(f"\nLayer {layer} MLP shapes:")
    print(f"  w1 (gate): merged={merged_w1.shape}, target={target_w1.shape}")
    print(f"  w2 (down): merged={merged_w2.shape}, target={target_w2.shape}")
    print(f"  w3 (up):   merged={merged_w3.shape}, target={target_w3.shape}")

    # Load REAL activations from profile (not random!)
    from safetensors import safe_open

    target_profile_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16/.modelcypher/activations.safetensors"

    print("\nLoading real activations from profile...")
    with safe_open(target_profile_path, framework="numpy") as f:
        hidden_key = "hidden_2"  # Layer 2 hidden states
        test_hidden_np = f.get_tensor(hidden_key)

    test_hidden = mx.array(test_hidden_np.astype("float32"))

    print(f"Real hidden states: {test_hidden.shape}, norm={float(mx.linalg.norm(test_hidden)):.4f}")

    # Run both MLPs
    target_output = run_mlp(test_hidden, target_w1, target_w2, target_w3)
    merged_output = run_mlp(test_hidden, merged_w1, merged_w2, merged_w3)

    # Compute norms
    target_norm = mx.linalg.norm(target_output)
    merged_norm = mx.linalg.norm(merged_output)

    # The empirical correction factor
    empirical_correction = float(target_norm / merged_norm)

    print(f"\n=== EMPIRICAL SCALE CORRECTION ===")
    print(f"Target MLP output norm:  {float(target_norm):.6f}")
    print(f"Merged MLP output norm:  {float(merged_norm):.6f}")
    print(f"Empirical correction:    {empirical_correction:.6f}")
    print(f"  (multiply merged output by this to match target)")

    # Also compute per-sample statistics
    target_sample_norms = mx.linalg.norm(target_output, axis=1)
    merged_sample_norms = mx.linalg.norm(merged_output, axis=1)
    sample_ratios = target_sample_norms / (merged_sample_norms + 1e-8)

    print(f"\nPer-sample correction statistics:")
    print(f"  Mean:   {float(mx.mean(sample_ratios)):.6f}")
    print(f"  Std:    {float(mx.std(sample_ratios)):.6f}")
    print(f"  Min:    {float(mx.min(sample_ratios)):.6f}")
    print(f"  Max:    {float(mx.max(sample_ratios)):.6f}")

    # Compare to the joint scale we computed (1.4422)
    joint_scale = 1.4422
    print(f"\n=== COMPARISON ===")
    print(f"Joint scale we applied:     {joint_scale:.6f}")
    print(f"Empirical correction needed: {empirical_correction:.6f}")
    print(f"Ratio (empirical/joint):     {empirical_correction/joint_scale:.6f}")

    # What should we have done?
    # If merged output is X times target, we need to divide by X
    # The joint scale MULTIPLIED by 1.4422, but we needed to multiply by empirical_correction
    # So the error factor is empirical_correction / joint_scale

    if empirical_correction > 1.0:
        print(f"\n→ Merged MLP outputs are TOO SMALL")
        print(f"  Need to MULTIPLY down_proj output by {empirical_correction:.4f}")
    else:
        print(f"\n→ Merged MLP outputs are TOO LARGE")
        print(f"  Need to DIVIDE down_proj output by {1/empirical_correction:.4f}")

    # Also check individual components
    print(f"\n=== COMPONENT ANALYSIS ===")

    # Gate output (pre-silu)
    target_gate = test_hidden @ target_w1.T
    merged_gate = test_hidden @ merged_w1.T
    print(f"Gate (pre-silu): target_norm={float(mx.linalg.norm(target_gate)):.4f}, merged_norm={float(mx.linalg.norm(merged_gate)):.4f}, ratio={float(mx.linalg.norm(target_gate)/mx.linalg.norm(merged_gate)):.4f}")

    # Gate output (post-silu)
    target_gate_silu = silu(target_gate)
    merged_gate_silu = silu(merged_gate)
    print(f"Gate (post-silu): target_norm={float(mx.linalg.norm(target_gate_silu)):.4f}, merged_norm={float(mx.linalg.norm(merged_gate_silu)):.4f}, ratio={float(mx.linalg.norm(target_gate_silu)/mx.linalg.norm(merged_gate_silu)):.4f}")

    # Up output
    target_up = test_hidden @ target_w3.T
    merged_up = test_hidden @ merged_w3.T
    print(f"Up: target_norm={float(mx.linalg.norm(target_up)):.4f}, merged_norm={float(mx.linalg.norm(merged_up)):.4f}, ratio={float(mx.linalg.norm(target_up)/mx.linalg.norm(merged_up)):.4f}")

    # Intermediate (gate * up)
    target_inter = target_gate_silu * target_up
    merged_inter = merged_gate_silu * merged_up
    print(f"Intermediate: target_norm={float(mx.linalg.norm(target_inter)):.4f}, merged_norm={float(mx.linalg.norm(merged_inter)):.4f}, ratio={float(mx.linalg.norm(target_inter)/mx.linalg.norm(merged_inter)):.4f}")

    # Final output
    print(f"Output: target_norm={float(target_norm):.4f}, merged_norm={float(merged_norm):.4f}, ratio={empirical_correction:.4f}")


if __name__ == "__main__":
    main()
