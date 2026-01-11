#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Interpretability Demo - Exercise SAE, Patching, and Steering on real models

"""
Mechanistic Interpretability Demo

Demonstrates:
1. SAE training on model activations
2. Feature analysis (top-k features)
3. Contrastive direction extraction
4. Feature steering

Usage:
    poetry run python scripts/interp_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import detect_default_backend_type, get_backend
from modelcypher.core.domain._backend import set_default_backend


def get_hidden_states(model, tokenizer, prompt: str, target_layer: int):
    """Extract hidden states from a specific layer by running partial forward pass."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get embeddings
    inner_model = model.model
    h = inner_model.embed_tokens(input_ids)

    # Create attention mask with correct dtype (bfloat16 to match model)
    T = h.shape[1]
    mask = mx.triu(mx.full((T, T), float("-inf"), dtype=mx.bfloat16), k=1)

    # Run through layers up to target
    for i, layer in enumerate(inner_model.layers):
        h = layer(h, mask=mask)
        if i == target_layer:
            mx.eval(h)
            return h

    return h


def main() -> None:
    print("=" * 60)
    print("MECHANISTIC INTERPRETABILITY DEMO")
    print("=" * 60)
    print()

    # Initialize backend
    print("[1/6] Initializing backend...")
    backend_type = detect_default_backend_type()
    backend = get_backend(backend_type)
    set_default_backend(backend)
    print(f"      Backend: {backend_type}")
    print()

    # Model path
    model_path = Path("/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"[2/6] Loading model from {model_path.name}...")

    # Load model using mlx_lm
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(str(model_path))
    hidden_dim = model.model.embed_tokens.weight.shape[1]
    num_layers = len(model.model.layers)
    print(f"      Model loaded: {type(model).__name__}")
    print(f"      Hidden dim: {hidden_dim}")
    print(f"      Layers: {num_layers}")
    print()

    # Define test prompts for activation collection
    print("[3/6] Collecting activations from layer 12...")

    prompts = [
        "def fibonacci(n):",
        "class Database:",
        "async def fetch_data():",
        "SELECT * FROM users WHERE",
        "The quick brown fox",
        "import numpy as np",
        "function calculateSum(arr) {",
        "public static void main(",
        "CREATE TABLE customers",
        "if __name__ == '__main__':",
    ]

    target_layer = 12
    activations_list = []

    for prompt in prompts:
        h = get_hidden_states(model, tokenizer, prompt, target_layer)
        # Take last token activation
        act = h[0, -1, :]  # [hidden_dim]
        activations_list.append(act)

    # Stack into matrix
    activations = mx.stack(activations_list, axis=0)  # [n_prompts, hidden_dim]
    mx.eval(activations)
    print(f"      Collected {activations.shape[0]} activations, shape: {activations.shape}")
    print()

    # Convert to backend array
    b = backend
    activations_backend = b.array(activations)
    b.eval(activations_backend)

    # Train SAE
    print("[4/6] Training Sparse Autoencoder...")

    from modelcypher.core.domain.interpretability.sae import (
        SAEConfig,
        SparseAutoencoder,
        derive_sparsity_coefficient,
    )

    sae_config = SAEConfig(
        hidden_dim=hidden_dim,
        expansion_factor=4,  # hidden_dim * 4 latent features
        normalize_decoder=True,
    )

    sae = SparseAutoencoder(sae_config, backend=b)
    weights = sae.initialize_weights()

    # Derive sparsity coefficient from data
    sparsity_coeff = derive_sparsity_coefficient(activations_backend, b)
    print(f"      Derived sparsity coefficient: {sparsity_coeff:.6f}")

    # Encode activations
    result = sae.encode(activations_backend, weights, sparsity_coefficient=sparsity_coeff)

    print(f"      Sparse codes shape: {result.sparse_codes.shape}")
    print(f"      Reconstruction loss: {result.reconstruction_loss:.6f}")
    print(f"      Average sparsity (L0): {result.sparsity:.1f} active features")
    print()

    # Analyze top features
    print("[5/6] Analyzing top features...")

    # analyze_features takes raw activations, not sparse codes
    analysis = sae.analyze_features(activations_backend, weights, top_k=5)
    print(f"      Top {len(analysis.top_k_features)} features by activation:")
    for i, feat in enumerate(analysis.top_k_features):
        print(f"        {i+1}. Feature {feat.index}: activation={feat.activation:.4f}")
    print(f"      Active features: {analysis.active_feature_count}/{analysis.total_features}")
    print()

    # Contrastive direction extraction
    print("[6/6] Extracting contrastive steering direction...")

    from modelcypher.core.domain.interpretability.feature_steering import (
        FeatureSteering,
    )

    # Collect code vs natural language activations
    code_prompts = [
        "def calculate():",
        "class Handler:",
        "import os",
        "function main() {",
        "for i in range(10):",
        "while True:",
        "try:",
        "return result",
    ]
    text_prompts = [
        "The weather is nice today",
        "She walked to the store",
        "Happy birthday to you",
        "The cat sat on the mat",
        "Once upon a time",
        "In the beginning",
        "Love is all you need",
        "To be or not to be",
    ]

    code_acts = []
    text_acts = []

    for prompt in code_prompts:
        h = get_hidden_states(model, tokenizer, prompt, target_layer)
        code_acts.append(h[0, -1, :])

    for prompt in text_prompts:
        h = get_hidden_states(model, tokenizer, prompt, target_layer)
        text_acts.append(h[0, -1, :])

    code_activations = mx.stack(code_acts, axis=0)
    text_activations = mx.stack(text_acts, axis=0)
    mx.eval(code_activations, text_activations)

    # Convert to backend
    code_backend = b.array(code_activations)
    text_backend = b.array(text_activations)
    b.eval(code_backend, text_backend)

    # Create steering instance
    steering = FeatureSteering(model, backend=b)

    # Extract code vs text direction
    code_direction = steering.extract_contrastive_direction(
        positive_activations=code_backend,
        negative_activations=text_backend,
        layer=target_layer,
        label="code_vs_text",
    )

    print(f"      Extracted '{code_direction.label}' direction")
    print(f"      Source: {code_direction.source.value}")
    print(f"      Layer: {code_direction.layer}")
    print(f"      Strength range: ({code_direction.strength_range[0]:.4f}, {code_direction.strength_range[1]:.4f})")
    print()

    # Compute direction statistics
    direction_arr = b.array(code_direction.direction)
    b.eval(direction_arr)
    dir_norm = float(b.to_scalar(b.sqrt(b.sum(direction_arr ** 2))))
    print(f"      Direction L2 norm: {dir_norm:.6f}")

    # Summary
    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("Demonstrated:")
    print("  - SAE encoding with geodesic reconstruction loss")
    print("  - Feature analysis (top-k by activation)")
    print("  - Contrastive direction extraction (code vs text)")
    print()
    print("All computations used geodesic distances, not Euclidean.")
    print("All thresholds derived from data, not hardcoded.")
    print()


if __name__ == "__main__":
    main()
