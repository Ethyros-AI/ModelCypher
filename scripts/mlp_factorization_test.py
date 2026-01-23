#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# MLP Factorization Test
"""
MLP Factorization Test

THE THEORY:
The hourglass analysis shows each MLP layer has effective rank 1-24.
This means we can factor: W = U @ S @ V^T with very few singular values.

THE TEST:
1. Get MLP weights (gate_proj, up_proj, down_proj)
2. Factor them using SVD
3. Keep only top-K singular values (K = helix dimension)
4. Replace weights with factored version
5. Test if generation still works

If this works, we have STATIC 100x+ compression with elegant math.

Usage:
    python mlp_factorization_test.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_layer_helix_dim(layer_idx: int, n_layers: int) -> int:
    """Return optimal helix dimension for each layer based on analysis."""
    # From hourglass analysis for 16-layer model
    # Scale for different layer counts
    if n_layers == 16:
        dims = {
            0: 24, 1: 19, 2: 15, 3: 22, 4: 21, 5: 13, 6: 20,
            7: 1,   # BOTTLENECK 1
            8: 10, 9: 11, 10: 12, 11: 12, 12: 12, 13: 12,
            14: 1,  # BOTTLENECK 2
            15: 12,
        }
        return dims.get(layer_idx, 24)
    elif n_layers == 28:
        # Scale bottlenecks to ~45% and ~90% of depth
        bottleneck1 = int(0.45 * n_layers)  # ~12
        bottleneck2 = int(0.9 * n_layers)   # ~25
        if layer_idx == bottleneck1 or layer_idx == bottleneck2:
            return 1
        elif bottleneck1 < layer_idx < bottleneck2:
            return 12  # Highway
        else:
            return 24  # Wide
    else:
        # Default: use 24 for all
        return 24


def analyze_mlp_rank(model: Any, layer_idx: int) -> dict:
    """Analyze the effective rank of an MLP layer."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    layer = inner_model.layers[layer_idx]
    mlp = layer.mlp

    # Get weights
    gate_proj = np.array(mlp.gate_proj.weight.astype(mx.float32))  # (intermediate, hidden)
    up_proj = np.array(mlp.up_proj.weight.astype(mx.float32))      # (intermediate, hidden)
    down_proj = np.array(mlp.down_proj.weight.astype(mx.float32))  # (hidden, intermediate)

    # SVD on down_proj (the main compression target)
    U, S, Vh = np.linalg.svd(down_proj, full_matrices=False)

    # Compute effective rank (99% of variance)
    total_var = np.sum(S ** 2)
    cumsum_var = np.cumsum(S ** 2)
    rank_99 = np.searchsorted(cumsum_var / total_var, 0.99) + 1
    rank_95 = np.searchsorted(cumsum_var / total_var, 0.95) + 1
    rank_90 = np.searchsorted(cumsum_var / total_var, 0.90) + 1

    return {
        'gate_shape': gate_proj.shape,
        'up_shape': up_proj.shape,
        'down_shape': down_proj.shape,
        'singular_values': S,
        'rank_90': rank_90,
        'rank_95': rank_95,
        'rank_99': rank_99,
        'top_10_sv': S[:10],
    }


def test_factored_mlp_layer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    rank: int,
    max_tokens: int = 5,
) -> tuple[str, str, float]:
    """Test generation with a single layer's MLP factored."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    normal_generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        normal_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    normal_output = tokenizer.decode(normal_generated)

    # Get original down_proj and factor it
    layer = inner_model.layers[layer_idx]
    mlp = layer.mlp
    down_proj = np.array(mlp.down_proj.weight.astype(mx.float32))

    U, S, Vh = np.linalg.svd(down_proj, full_matrices=False)

    # Reconstruct with reduced rank
    S_truncated = S.copy()
    S_truncated[rank:] = 0
    down_proj_factored = U @ np.diag(S_truncated) @ Vh

    # Compute reconstruction error
    error = np.linalg.norm(down_proj - down_proj_factored) / np.linalg.norm(down_proj)

    # Replace weight temporarily
    original_weight = mlp.down_proj.weight
    mlp.down_proj.weight = mx.array(down_proj_factored.astype(np.float32)).astype(original_weight.dtype)
    mx.eval(mlp.down_proj.weight)

    # Generate with factored weight
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    factored_generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        factored_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    factored_output = tokenizer.decode(factored_generated)

    # Restore original weight
    mlp.down_proj.weight = original_weight
    mx.eval(mlp.down_proj.weight)

    return normal_output, factored_output, error


def test_all_layers_factored(
    model: Any,
    tokenizer: Any,
    prompt: str,
    rank_schedule: dict[int, int],  # layer_idx -> rank
    max_tokens: int = 5,
) -> tuple[str, str]:
    """Test generation with all layers factored according to schedule."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    normal_generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        normal_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    normal_output = tokenizer.decode(normal_generated)

    # Factor all MLP down_proj weights
    original_weights = {}
    for layer_idx, rank in rank_schedule.items():
        layer = inner_model.layers[layer_idx]
        mlp = layer.mlp
        down_proj = np.array(mlp.down_proj.weight.astype(mx.float32))

        U, S, Vh = np.linalg.svd(down_proj, full_matrices=False)
        S_truncated = S.copy()
        S_truncated[rank:] = 0
        down_proj_factored = U @ np.diag(S_truncated) @ Vh

        original_weights[layer_idx] = mlp.down_proj.weight
        mlp.down_proj.weight = mx.array(down_proj_factored.astype(np.float32)).astype(original_weights[layer_idx].dtype)
        mx.eval(mlp.down_proj.weight)

    # Generate with factored weights
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    factored_generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        factored_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    factored_output = tokenizer.decode(factored_generated)

    # Restore original weights
    for layer_idx, orig_weight in original_weights.items():
        inner_model.layers[layer_idx].mlp.down_proj.weight = orig_weight
        mx.eval(inner_model.layers[layer_idx].mlp.down_proj.weight)

    return normal_output, factored_output


def main():
    parser = argparse.ArgumentParser(description="MLP factorization test")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("MLP FACTORIZATION TEST")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Phase 1: Analyze MLP ranks
    print(f"\n{'='*80}")
    print("PHASE 1: MLP EFFECTIVE RANK ANALYSIS")
    print("="*80)

    print(f"\n{'Layer':>6} | {'Rank 90%':>10} | {'Rank 95%':>10} | {'Rank 99%':>10} | Top 3 SVs")
    print("-" * 70)

    for layer_idx in range(min(n_layers, 10)):  # First 10 layers
        stats = analyze_mlp_rank(model, layer_idx)
        top3 = ', '.join(f'{s:.1f}' for s in stats['top_10_sv'][:3])
        print(f"{layer_idx:>6} | {stats['rank_90']:>10} | {stats['rank_95']:>10} | {stats['rank_99']:>10} | {top3}")

    # Phase 2: Test single layer factorization
    print(f"\n{'='*80}")
    print("PHASE 2: SINGLE LAYER FACTORIZATION TEST")
    print("="*80)

    test_prompt = "The capital of France is"

    # Test middle layer with different ranks
    test_layer = n_layers // 2

    print(f"\nTesting layer {test_layer} with different ranks:")
    print(f"{'Rank':>6} | {'Error':>10} | {'Result'}")
    print("-" * 50)

    for rank in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        normal, factored, error = test_factored_mlp_layer(
            model, tokenizer, test_prompt, test_layer, rank, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        factored_first = factored.split()[0] if factored.split() else "(empty)"
        match = "✓" if normal_first == factored_first else f"✗ ({normal_first}→{factored_first})"
        print(f"{rank:>6} | {error:>10.6f} | {match}")

    # Phase 3: Test all layers with hourglass schedule
    print(f"\n{'='*80}")
    print("PHASE 3: ALL LAYERS WITH HOURGLASS SCHEDULE")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    # Test with different global ranks
    for global_rank in [512, 256, 128, 64]:
        rank_schedule = {i: global_rank for i in range(n_layers)}

        matches = 0
        for prompt in test_prompts:
            normal, factored = test_all_layers_factored(
                model, tokenizer, prompt, rank_schedule, max_tokens=5
            )
            normal_first = normal.split()[0] if normal.split() else "(empty)"
            factored_first = factored.split()[0] if factored.split() else "(empty)"
            if normal_first == factored_first:
                matches += 1

        compression = hidden_dim / global_rank if global_rank > 0 else 0
        print(f"Rank={global_rank:>4}: {matches}/{len(test_prompts)} matches, ~{compression:.1f}x compression on down_proj")

    # Insight
    print(f"\n{'='*80}")
    print("MLP FACTORIZATION INSIGHT")
    print("="*80)
    print("""
THE KEY FINDING:

The down_proj weight matrix can be approximated with reduced rank SVD.

If rank R works for all layers:
    Original: hidden_dim × intermediate_dim = 2048 × 4608 = 9.4M params/layer
    Factored: hidden_dim × R + R × intermediate_dim = much smaller

For R=64: 2048×64 + 64×4608 = 131K + 295K = 426K params/layer
    Compression: 9.4M / 426K = 22x

For R=32: 2048×32 + 32×4608 = 65K + 147K = 213K params/layer
    Compression: 9.4M / 213K = 44x

The hourglass dimensions (1-24) would give even more compression,
but need to verify they actually preserve generation quality.
""")


if __name__ == "__main__":
    main()
