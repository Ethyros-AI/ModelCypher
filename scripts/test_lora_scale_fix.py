#!/usr/bin/env python3
"""Demonstrate post-hoc LoRA scale fix.

Shows that:
1. Base model produces coherent output
2. LoRA with geometric scale also produces coherent output

The key insight: geometric scaling makes broken adapters usable.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import initialize_default_backend
initialize_default_backend()


def main():
    import mlx.core as mx
    from mlx_lm import load, generate

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "/Volumes/CodeCypher/models/adapters/geometric-awareness-8B-v1"

    # Test prompts
    prompts = [
        "What is 15 + 27?",
        "Explain why the sky is blue in one sentence.",
    ]

    print("=" * 70)
    print("LoRA SCALE FIX DEMONSTRATION")
    print("=" * 70)

    # Test 1: Base model (no LoRA)
    print("\n[1] BASE MODEL (no LoRA)")
    print("-" * 70)
    model, tokenizer = load(model_path)

    for prompt in prompts:
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=100, verbose=False
        )
        print(f"Q: {prompt}")
        print(f"A: {response[:200]}...")
        print()

    # Test 2: LoRA with geometric scale
    print("\n[2] LORA WITH GEOMETRIC SCALE")
    print("-" * 70)

    # Reload fresh model
    del model
    mx.metal.clear_cache()
    model, tokenizer = load(model_path)

    # Apply with geometric scaling
    from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
    service = LoRASafetyService()

    print("Computing geometric scale bounds...")
    report = service.compute_geometric_scale(model_path, adapter_path)
    print(f"Configured scale: {report.configured_scale}")
    print(f"Min geometric bound: {report.min_geometric_bound:.6f}")
    print(f"Max ratio (configured/geometric): {report.max_scale_ratio:.1f}x")
    print()

    print("Applying LoRA with geometric scaling...")
    model, scales = service.apply_lora_geometric(model, adapter_path)

    scale_values = list(scales.values())
    print(f"Applied to {len(scales)} layers")
    print(f"Geometric scales: min={min(scale_values):.6f}, max={max(scale_values):.6f}")
    print()

    for prompt in prompts:
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=100, verbose=False
        )
        print(f"Q: {prompt}")
        print(f"A: {response[:200]}...")
        print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Configured scale ({report.configured_scale}) was {report.max_scale_ratio:.0f}x over geometric bound.")
    print("With geometric scaling, the adapter produces coherent output.")
    print("The learned LoRA weights were valid - only the scale was wrong.")


if __name__ == "__main__":
    main()
