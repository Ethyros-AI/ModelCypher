#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Invariant Subspace Discovery
"""
Invariant Subspace Discovery

THE INSIGHT:
Top-K compression works (3.8x at K=543), but K is input-dependent.
Different inputs might need different dimensions.

THE QUESTION:
Are there dimensions that are CONSISTENTLY important across all inputs?
If yes, we can use a FIXED basis instead of input-dependent top-K.

THE APPROACH:
1. Collect delta vectors from many diverse inputs
2. For each layer, count how often each dimension is in the top-K
3. Find dimensions that are ALWAYS important (intersection)
4. Find dimensions that are NEVER important (can discard)
5. Test compression using only the invariant dimensions

If successful, this gives us a LEARNED basis that's input-independent.

Usage:
    python invariant_subspace_discovery.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Diverse test prompts to find invariant dimensions
DIVERSE_PROMPTS = [
    # Factual
    "The capital of France is",
    "The capital of Japan is",
    "The largest planet is",
    "Water boils at",
    "The speed of light is",

    # Math
    "2 + 2 =",
    "10 - 3 =",
    "5 * 5 =",
    "100 / 4 =",

    # Opposites
    "The opposite of hot is",
    "The opposite of big is",
    "The opposite of fast is",
    "The opposite of happy is",

    # Completion
    "Once upon a time",
    "In the beginning",
    "The quick brown fox",
    "To be or not to",

    # Technical
    "Python is a",
    "Machine learning is",
    "The internet was",
    "Artificial intelligence",

    # Abstract
    "Love is",
    "Time is",
    "Life is",
    "Truth is",
]


def collect_topk_indices(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    k: int,
    layers: list[int],
) -> dict[int, list[set[int]]]:
    """Collect top-K indices for each layer across all prompts."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # layer_idx -> list of sets of top-K indices (one set per prompt)
    layer_topk: dict[int, list[set[int]]] = defaultdict(list)

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            h_in_np = np.array(h[0, -1, :].astype(mx.float32))

            h_true = layer(h, mask, None)
            mx.eval(h_true)

            h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))
            delta = h_out_np - h_in_np

            if idx in layers:
                # Find top-K indices by magnitude
                abs_delta = np.abs(delta)
                topk_idx = set(np.argpartition(abs_delta, -k)[-k:].tolist())
                layer_topk[idx].append(topk_idx)

            h = h_true

    return layer_topk


def analyze_dimension_frequency(
    layer_topk: dict[int, list[set[int]]],
    hidden_dim: int,
) -> dict[int, dict]:
    """Analyze how often each dimension appears in top-K."""

    results = {}

    for layer_idx, topk_sets in layer_topk.items():
        n_prompts = len(topk_sets)

        # Count frequency of each dimension
        dim_counts = np.zeros(hidden_dim, dtype=int)
        for topk_set in topk_sets:
            for dim in topk_set:
                dim_counts[dim] += 1

        # Find invariant dimensions (appear in ALL prompts' top-K)
        always_important = set(np.where(dim_counts == n_prompts)[0].tolist())

        # Find dimensions that appear in MOST prompts (>90%)
        mostly_important = set(np.where(dim_counts >= 0.9 * n_prompts)[0].tolist())

        # Find dimensions that appear in >50% of prompts
        often_important = set(np.where(dim_counts >= 0.5 * n_prompts)[0].tolist())

        # Find dimensions that NEVER appear
        never_important = set(np.where(dim_counts == 0)[0].tolist())

        results[layer_idx] = {
            'dim_counts': dim_counts,
            'always_important': always_important,
            'mostly_important': mostly_important,
            'often_important': often_important,
            'never_important': never_important,
            'n_prompts': n_prompts,
        }

    return results


def test_fixed_basis_compression(
    model: Any,
    tokenizer: Any,
    prompt: str,
    fixed_dims: dict[int, set[int]],  # layer -> set of dims to keep
    max_tokens: int = 5,
) -> tuple[str, str]:
    """Test generation using only fixed dimensions."""
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

    # Fixed-basis compressed generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        h_in_np = np.array(h[0, -1, :].astype(mx.float32))

        h_true = layer(h, mask, None)
        mx.eval(h_true)

        if idx in fixed_dims:
            h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))
            delta_true = h_out_np - h_in_np

            # Keep only fixed dimensions
            delta_compressed = np.zeros_like(delta_true)
            keep_dims = list(fixed_dims[idx])
            delta_compressed[keep_dims] = delta_true[keep_dims]

            # Reconstruct
            h_new = h_in_np + delta_compressed

            h_true_np = np.array(h_true.astype(mx.float32))
            h_true_np[0, -1, :] = h_new

            h = mx.array(h_true_np).astype(h_true.dtype)
            mx.eval(h)
        else:
            h = h_true

    # Final norm
    h = inner_model.norm(h)
    mx.eval(h)

    # Get logits
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)

    logits_np = np.array(logits[0, -1, :].astype(mx.float32))
    next_token = int(np.argmax(logits_np))

    compressed_generated = []
    if next_token != tokenizer.eos_token_id:
        compressed_generated.append(next_token)

    # Continue normally
    input_ids = mx.array([[next_token]])
    for _ in range(max_tokens - 1):
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        next_token = int(np.argmax(logits_np))

        if next_token == tokenizer.eos_token_id:
            break

        compressed_generated.append(next_token)
        input_ids = mx.array([[next_token]])

    compressed_output = tokenizer.decode(compressed_generated)

    return normal_output, compressed_output


def main():
    parser = argparse.ArgumentParser(description="Invariant subspace discovery")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--k", type=int, default=543, help="K for top-K analysis")
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
    print("INVARIANT SUBSPACE DISCOVERY")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Analyzing top-{args.k} dimensions across {len(DIVERSE_PROMPTS)} prompts")

    # Transmission layers
    transmission_layers = list(range(3, 27)) if n_layers == 28 else list(range(3, n_layers - 1))
    print(f"Analyzing layers: {transmission_layers[0]}-{transmission_layers[-1]}")

    # Phase 1: Collect top-K indices
    print(f"\n{'='*80}")
    print("PHASE 1: COLLECTING TOP-K INDICES")
    print("="*80)

    layer_topk = collect_topk_indices(
        model, tokenizer, DIVERSE_PROMPTS, args.k, transmission_layers
    )

    # Phase 2: Analyze frequency
    print(f"\n{'='*80}")
    print("PHASE 2: DIMENSION FREQUENCY ANALYSIS")
    print("="*80)

    analysis = analyze_dimension_frequency(layer_topk, hidden_dim)

    print(f"\n{'Layer':>6} | {'Always':>8} | {'Mostly':>8} | {'Often':>8} | {'Never':>8}")
    print("-" * 55)

    # Aggregate across layers
    global_always = None
    global_never = set(range(hidden_dim))

    for layer_idx in sorted(analysis.keys()):
        stats = analysis[layer_idx]
        print(f"{layer_idx:>6} | {len(stats['always_important']):>8} | "
              f"{len(stats['mostly_important']):>8} | "
              f"{len(stats['often_important']):>8} | "
              f"{len(stats['never_important']):>8}")

        if global_always is None:
            global_always = stats['always_important'].copy()
        else:
            global_always &= stats['always_important']

        global_never &= stats['never_important']

    print(f"\n{'='*80}")
    print("GLOBAL ANALYSIS (across ALL transmission layers)")
    print("="*80)
    print(f"Dimensions ALWAYS in top-{args.k} for ALL layers: {len(global_always)}")
    print(f"Dimensions NEVER in top-{args.k} for ANY layer: {len(global_never)}")

    # Phase 3: Test different fixed-basis strategies
    print(f"\n{'='*80}")
    print("PHASE 3: TESTING FIXED-BASIS COMPRESSION")
    print("="*80)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The opposite of hot is",
    ]

    # Strategy 1: Use "often important" dims (>50% frequency)
    print("\n--- Strategy 1: Use dimensions important >50% of the time ---")

    fixed_dims_often = {
        layer_idx: stats['often_important']
        for layer_idx, stats in analysis.items()
    }

    avg_dims = np.mean([len(d) for d in fixed_dims_often.values()])
    print(f"Average dimensions per layer: {avg_dims:.0f}")

    matches = 0
    for prompt in test_prompts:
        normal, compressed = test_fixed_basis_compression(
            model, tokenizer, prompt, fixed_dims_often, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        compressed_first = compressed.split()[0] if compressed.split() else "(empty)"
        match = "✓" if normal_first == compressed_first else "✗"
        if normal_first == compressed_first:
            matches += 1
        print(f"  {prompt[:30]}: {normal_first} → {compressed_first} {match}")

    print(f"  Matches: {matches}/{len(test_prompts)}")
    if avg_dims > 0:
        print(f"  Compression: {hidden_dim / avg_dims:.1f}x")

    # Strategy 2: Use "mostly important" dims (>90% frequency)
    print("\n--- Strategy 2: Use dimensions important >90% of the time ---")

    fixed_dims_mostly = {
        layer_idx: stats['mostly_important']
        for layer_idx, stats in analysis.items()
    }

    avg_dims = np.mean([len(d) for d in fixed_dims_mostly.values()])
    print(f"Average dimensions per layer: {avg_dims:.0f}")

    matches = 0
    for prompt in test_prompts:
        normal, compressed = test_fixed_basis_compression(
            model, tokenizer, prompt, fixed_dims_mostly, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        compressed_first = compressed.split()[0] if compressed.split() else "(empty)"
        match = "✓" if normal_first == compressed_first else "✗"
        if normal_first == compressed_first:
            matches += 1
        print(f"  {prompt[:30]}: {normal_first} → {compressed_first} {match}")

    print(f"  Matches: {matches}/{len(test_prompts)}")
    if avg_dims > 0:
        print(f"  Compression: {hidden_dim / avg_dims:.1f}x")

    # Strategy 3: Union of all top-K across all prompts (upper bound)
    print("\n--- Strategy 3: Union of all top-K (upper bound) ---")

    fixed_dims_union = {}
    for layer_idx, topk_sets in layer_topk.items():
        union_set = set()
        for s in topk_sets:
            union_set |= s
        fixed_dims_union[layer_idx] = union_set

    avg_dims = np.mean([len(d) for d in fixed_dims_union.values()])
    print(f"Average dimensions per layer: {avg_dims:.0f}")

    matches = 0
    for prompt in test_prompts:
        normal, compressed = test_fixed_basis_compression(
            model, tokenizer, prompt, fixed_dims_union, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        compressed_first = compressed.split()[0] if compressed.split() else "(empty)"
        match = "✓" if normal_first == compressed_first else "✗"
        if normal_first == compressed_first:
            matches += 1
        print(f"  {prompt[:30]}: {normal_first} → {compressed_first} {match}")

    print(f"  Matches: {matches}/{len(test_prompts)}")
    if avg_dims > 0:
        print(f"  Compression: {hidden_dim / avg_dims:.1f}x")

    # Analysis
    print(f"\n{'='*80}")
    print("INVARIANT SUBSPACE INSIGHT")
    print("="*80)

    # Check overlap between layers
    first_layer = transmission_layers[0]
    last_layer = transmission_layers[-1]
    overlap = analysis[first_layer]['often_important'] & analysis[last_layer]['often_important']

    print(f"""
DIMENSION FREQUENCY DISTRIBUTION:

Layer {first_layer} (first transmission):
  - Always important: {len(analysis[first_layer]['always_important'])} dims
  - Often important: {len(analysis[first_layer]['often_important'])} dims

Layer {last_layer} (last transmission):
  - Always important: {len(analysis[last_layer]['always_important'])} dims
  - Often important: {len(analysis[last_layer]['often_important'])} dims

Cross-layer overlap (often important):
  - Shared dimensions: {len(overlap)} / {hidden_dim}

INTERPRETATION:
If "always important" is small → dimensions are input-dependent
If "often important" is large → there IS a stable subspace

NEXT STEP:
If fixed-basis works, we have a LEARNED compression scheme.
If it fails, we need to understand WHY some dimensions matter
only for specific inputs (semantic routing?).
""")


if __name__ == "__main__":
    main()
