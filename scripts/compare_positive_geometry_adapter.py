#!/usr/bin/env python3
"""Compare positive geometry signatures before/after adapter."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.adapters.model_loader import load_model
from modelcypher.backends import initialize_default_backend

# Initialize backend
initialize_default_backend()
from modelcypher.core.domain.geometry.positive_geometry import (
    compute_positive_grassmann_signature,
    PositiveGrassmannSignature,
)
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
import mlx.core as mx


def get_layer_activations(model, tokenizer, probes: list[str], layer_idx: int) -> mx.array:
    """Get activations at a specific layer for all probes."""
    activations = []

    for probe in probes:
        tokens = tokenizer.encode(probe)
        input_ids = mx.array([tokens])

        # Get hidden states at layer
        hidden = model.model.embed_tokens(input_ids)

        for i, layer in enumerate(model.model.layers):
            hidden = layer(hidden, mask=None, cache=None)
            if i == layer_idx:
                # Take last token's hidden state
                act = hidden[0, -1, :]
                activations.append(act)
                break

    return mx.stack(activations)


def main():
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    adapter_path = "/Volumes/CodeCypher/models/adapters/self-reflection-lora-v5"
    layers = [0, 4, 7, 8, 9, 12, 15]
    probe_count = 256
    max_minors = 256

    # Load probes
    all_probes = UnifiedAtlasInventory.all_probes()
    # Sample evenly across probes
    step = max(1, len(all_probes) // probe_count)
    sampled = all_probes[::step][:probe_count]
    probes = [p.support_texts[0] if p.support_texts else p.probe_id for p in sampled]
    print(f"Loaded {len(probes)} probes")

    results = {"base": {}, "adapter": {}}

    # Run WITHOUT adapter
    print("Loading base model...")
    model, tokenizer = load_model(model_path, adapter_path=None)

    for layer_idx in layers:
        print(f"  Layer {layer_idx} (base)...")
        acts = get_layer_activations(model, tokenizer, probes, layer_idx)
        acts = acts.astype(mx.float32)  # Convert bfloat16 -> float32 for SVD
        mx.eval(acts)

        sig = compute_positive_grassmann_signature(
            acts,
            rank_source="spectral-gap",
            max_minors=max_minors,
        )

        results["base"][layer_idx] = {
            "rank": sig.subspace_rank,
            "positive_fraction": sig.positive_fraction,
            "negative_fraction": sig.negative_fraction,
            "sign_entropy": sig.sign_entropy,
            "max_abs_minor": sig.max_abs_minor,
        }

    # Clear memory
    del model, tokenizer
    mx.metal.clear_cache()

    # Run WITH adapter
    print("\nLoading model with adapter...")
    model, tokenizer = load_model(model_path, adapter_path=adapter_path)

    for layer_idx in layers:
        print(f"  Layer {layer_idx} (adapter)...")
        acts = get_layer_activations(model, tokenizer, probes, layer_idx)
        acts = acts.astype(mx.float32)  # Convert bfloat16 -> float32 for SVD
        mx.eval(acts)

        sig = compute_positive_grassmann_signature(
            acts,
            rank_source="spectral-gap",
            max_minors=max_minors,
        )

        results["adapter"][layer_idx] = {
            "rank": sig.subspace_rank,
            "positive_fraction": sig.positive_fraction,
            "negative_fraction": sig.negative_fraction,
            "sign_entropy": sig.sign_entropy,
            "max_abs_minor": sig.max_abs_minor,
        }

    # Print comparison
    print("\n" + "="*80)
    print("POSITIVE GEOMETRY COMPARISON: Base vs v5 Adapter")
    print("="*80)
    print(f"{'Layer':<8} {'Rank':<12} {'Pos% (base)':<14} {'Pos% (adapt)':<14} {'Delta':<10} {'SignEnt Δ':<10}")
    print("-"*80)

    for layer_idx in layers:
        base = results["base"][layer_idx]
        adapt = results["adapter"][layer_idx]

        rank_str = f"{base['rank']}/{adapt['rank']}"
        pos_base = base["positive_fraction"] * 100
        pos_adapt = adapt["positive_fraction"] * 100
        delta = pos_adapt - pos_base
        ent_delta = adapt["sign_entropy"] - base["sign_entropy"]

        print(f"{layer_idx:<8} {rank_str:<12} {pos_base:<14.1f} {pos_adapt:<14.1f} {delta:+.1f}%     {ent_delta:+.3f}")

    # Save results
    output_path = Path("data/experiments/positive_geometry_comparison_v5.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert keys to strings for JSON
    json_results = {
        "base": {str(k): v for k, v in results["base"].items()},
        "adapter": {str(k): v for k, v in results["adapter"].items()},
    }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
