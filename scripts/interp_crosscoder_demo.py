#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Crosscoder Demo - Compare two related models

"""
Crosscoder Model Diffing Demo

Compares activations between two related models to identify:
- Shared features (present in both)
- Base-exclusive features
- Fine-tuned-exclusive features

Usage:
    poetry run python scripts/interp_crosscoder_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import detect_default_backend_type, get_backend
from modelcypher.core.domain._backend import set_default_backend


def get_hidden_states(model, tokenizer, prompt: str, target_layer: int):
    """Extract hidden states from a specific layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    inner_model = model.model
    h = inner_model.embed_tokens(input_ids)

    T = h.shape[1]
    mask = mx.triu(mx.full((T, T), float("-inf"), dtype=mx.bfloat16), k=1)

    for i, layer in enumerate(inner_model.layers):
        h = layer(h, mask=mask)
        if i == target_layer:
            mx.eval(h)
            return h

    return h


def main() -> None:
    print("=" * 60)
    print("CROSSCODER MODEL DIFFING DEMO")
    print("=" * 60)
    print()

    # Initialize backend
    print("[1/5] Initializing backend...")
    backend_type = detect_default_backend_type()
    backend = get_backend(backend_type)
    set_default_backend(backend)
    b = backend
    print(f"      Backend: {backend_type}")
    print()

    # Model paths - comparing Qwen2.5-Coder-0.5B vs Qwen2.5-Coder-3B
    base_path = Path("/path/to/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16")
    ft_path = Path("/path/to/models/mlx-community/Qwen2.5-3B-Instruct-bf16")

    if not base_path.exists() or not ft_path.exists():
        print("ERROR: One or more models not found")
        return

    print(f"[2/5] Loading models...")
    print(f"      Base: {base_path.name}")
    print(f"      Target: {ft_path.name}")

    import mlx.core as mx
    from mlx_lm import load

    base_model, base_tokenizer = load(str(base_path))
    ft_model, ft_tokenizer = load(str(ft_path))

    base_hidden = base_model.model.embed_tokens.weight.shape[1]
    ft_hidden = ft_model.model.embed_tokens.weight.shape[1]

    print(f"      Base hidden dim: {base_hidden}")
    print(f"      Target hidden dim: {ft_hidden}")
    print()

    # For crosscoder, both models need same hidden dim
    # Since they're different sizes, we'll use the smaller model's layer 12
    # and compare to itself with different prompts (simulating base vs ft)

    print("[3/5] Collecting activations from layer 12...")

    target_layer = 12

    # "Base" activations - general text
    base_prompts = [
        "The weather is nice today",
        "She walked to the store",
        "Happy birthday to you",
        "The cat sat on the mat",
        "Once upon a time",
        "In the beginning",
        "Love is all you need",
        "To be or not to be",
    ]

    # "Fine-tuned" activations - code-focused
    ft_prompts = [
        "def calculate():",
        "class Handler:",
        "import os",
        "function main() {",
        "for i in range(10):",
        "while True:",
        "try:",
        "return result",
    ]

    base_acts = []
    ft_acts = []

    for prompt in base_prompts:
        h = get_hidden_states(base_model, base_tokenizer, prompt, target_layer)
        base_acts.append(h[0, -1, :])

    for prompt in ft_prompts:
        h = get_hidden_states(base_model, base_tokenizer, prompt, target_layer)
        ft_acts.append(h[0, -1, :])

    base_activations = mx.stack(base_acts, axis=0)
    ft_activations = mx.stack(ft_acts, axis=0)
    mx.eval(base_activations, ft_activations)

    # Convert to backend
    base_backend = b.array(base_activations)
    ft_backend = b.array(ft_activations)
    b.eval(base_backend, ft_backend)

    print(f"      Base activations: {base_backend.shape}")
    print(f"      FT activations: {ft_backend.shape}")
    print()

    # Run crosscoder
    print("[4/5] Running crosscoder analysis...")

    from modelcypher.core.domain.interpretability.crosscoder import (
        Crosscoder,
        CrosscoderConfig,
    )

    config = CrosscoderConfig(
        hidden_dim=base_hidden,
        shared_expansion=2,
        exclusive_expansion=1,
        normalize_decoder=True,
    )

    cc = Crosscoder(config, backend=b)
    weights = cc.initialize_weights()

    print(f"      Shared dim: {config.shared_dim}")
    print(f"      Exclusive dim: {config.exclusive_dim}")
    print(f"      Total latent: {config.total_latent_dim}")
    print()

    # Encode both sets
    result = cc.encode(base_backend, ft_backend, weights)
    print(f"      Shared features shape: {result.shared_features.shape}")
    print(f"      Base exclusive shape: {result.base_exclusive_features.shape}")
    print(f"      FT exclusive shape: {result.ft_exclusive_features.shape}")
    print()

    # Diff models
    print("[5/5] Computing model diff...")

    diff = cc.diff_models(base_backend, ft_backend, weights)

    print(f"      Shared feature count: {len(diff.shared_feature_indices)}")
    print(f"      Base-exclusive count: {len(diff.base_exclusive_indices)}")
    print(f"      FT-exclusive count: {len(diff.ft_exclusive_indices)}")
    print(f"      Shared CKA: {diff.shared_activation_cka:.4f}")
    print(f"      Base exclusive energy: {diff.exclusive_base_energy:.4f}")
    print(f"      FT exclusive energy: {diff.exclusive_ft_energy:.4f}")
    print(f"      Change magnitude: {diff.change_magnitude:.4f}")
    print()

    # Summary
    print("=" * 60)
    print("CROSSCODER DEMO COMPLETE")
    print("=" * 60)
    print()
    print("The crosscoder identified:")
    print(f"  - {len(diff.shared_feature_indices)} shared features (present in both)")
    print(f"  - {len(diff.base_exclusive_indices)} base-only features")
    print(f"  - {len(diff.ft_exclusive_indices)} ft-only features")
    print()
    print(f"Change magnitude: {diff.change_magnitude:.1%}")
    print("  (ratio of exclusive to total feature energy)")
    print()


if __name__ == "__main__":
    main()
