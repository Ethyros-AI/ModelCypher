#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Embedding-Guided Compression
"""
Embedding-Guided Compression

THE DISCOVERY:
- Correlation 0.51 between embedding similarity and routing similarity
- Within-category routing 2x higher than between-category
- Categories have signature dimensions

THE HYPOTHESIS:
If the embedding predicts routing, then:
1. Large embedding dimensions → important delta dimensions?
2. Embedding's structure → delta's active dimensions?

THE EXPERIMENT:
1. Compare embedding's top-K to delta's top-K
2. If overlap is high, use embedding to guide compression
3. This would be elegant: no learned parameters needed!

Alternative: Use embedding PCA components as the compression basis.

Usage:
    python embedding_guided_compression.py --model /path/to/model
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


TEST_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "The opposite of hot is",
    "Once upon a time",
    "Python is a",
]


def analyze_embedding_delta_correlation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    layers: list[int],
) -> dict:
    """Analyze correlation between embedding dimensions and delta dimensions."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    # Get embedding
    emb = np.array(h[0, -1, :].astype(mx.float32))
    emb_abs = np.abs(emb)
    emb_topk = set(np.argpartition(emb_abs, -k)[-k:].tolist())

    mask = create_attention_mask(h, None)

    layer_results = {}

    for idx, layer in enumerate(inner_model.layers):
        h_in_np = np.array(h[0, -1, :].astype(mx.float32))

        h_true = layer(h, mask, None)
        mx.eval(h_true)

        if idx in layers:
            h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))
            delta = h_out_np - h_in_np
            delta_abs = np.abs(delta)
            delta_topk = set(np.argpartition(delta_abs, -k)[-k:].tolist())

            # Also get h_in's top-K
            h_in_abs = np.abs(h_in_np)
            h_in_topk = set(np.argpartition(h_in_abs, -k)[-k:].tolist())

            # Jaccard overlap
            emb_delta_overlap = len(emb_topk & delta_topk) / len(emb_topk | delta_topk)
            hin_delta_overlap = len(h_in_topk & delta_topk) / len(h_in_topk | delta_topk)

            layer_results[idx] = {
                'emb_delta_jaccard': emb_delta_overlap,
                'hin_delta_jaccard': hin_delta_overlap,
                'delta_topk': delta_topk,
            }

        h = h_true

    return {
        'emb_topk': emb_topk,
        'layers': layer_results,
    }


def test_hin_guided_compression(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    compress_layers: list[int],
    max_tokens: int = 5,
) -> tuple[str, str, dict]:
    """
    Use h_in's top-K dimensions to guide compression.

    Instead of computing delta, then finding its top-K,
    predict that delta will be large where h_in is large.
    """
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

    # h_in guided compression
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    stats = []

    for idx, layer in enumerate(inner_model.layers):
        h_in_np = np.array(h[0, -1, :].astype(mx.float32))

        h_true = layer(h, mask, None)
        mx.eval(h_true)

        if idx in compress_layers:
            h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))
            delta_true = h_out_np - h_in_np

            # GUIDED SELECTION: Use h_in's top-K to select dims
            h_in_abs = np.abs(h_in_np)
            guided_dims = np.argpartition(h_in_abs, -k)[-k:]

            # Keep only those dims from delta
            delta_compressed = np.zeros_like(delta_true)
            delta_compressed[guided_dims] = delta_true[guided_dims]

            # Compute energy kept for stats
            energy_kept = np.sum(delta_compressed**2) / (np.sum(delta_true**2) + 1e-10)
            stats.append({'layer': idx, 'energy_kept': energy_kept})

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

    return normal_output, compressed_output, stats


def test_adaptive_k_compression(
    model: Any,
    tokenizer: Any,
    prompt: str,
    compress_layers: list[int],
    energy_threshold: float = 0.95,  # Keep 95% of energy
    max_tokens: int = 5,
) -> tuple[str, str, dict]:
    """
    Adaptive compression: use however many dims needed to capture energy_threshold.

    This discovers the ACTUAL sparsity per layer.
    """
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

    # Adaptive compression
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    stats = []

    for idx, layer in enumerate(inner_model.layers):
        h_in_np = np.array(h[0, -1, :].astype(mx.float32))

        h_true = layer(h, mask, None)
        mx.eval(h_true)

        if idx in compress_layers:
            h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))
            delta_true = h_out_np - h_in_np

            # Sort by magnitude
            delta_abs = np.abs(delta_true)
            sorted_indices = np.argsort(delta_abs)[::-1]  # Descending

            # Find K needed for energy threshold
            total_energy = np.sum(delta_true ** 2)
            cumulative_energy = 0
            k_needed = 0

            for i, dim in enumerate(sorted_indices):
                cumulative_energy += delta_true[dim] ** 2
                k_needed = i + 1
                if cumulative_energy / total_energy >= energy_threshold:
                    break

            # Keep top k_needed dims
            keep_dims = sorted_indices[:k_needed]
            delta_compressed = np.zeros_like(delta_true)
            delta_compressed[keep_dims] = delta_true[keep_dims]

            stats.append({
                'layer': idx,
                'k_needed': k_needed,
                'energy_kept': cumulative_energy / total_energy,
            })

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

    return normal_output, compressed_output, stats


def main():
    parser = argparse.ArgumentParser(description="Embedding-guided compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--k", type=int, default=543, help="K for compression")
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
    print("EMBEDDING-GUIDED COMPRESSION")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    transmission_layers = list(range(3, 27)) if n_layers == 28 else list(range(3, n_layers - 1))

    # Phase 1: Analyze embedding-delta correlation
    print(f"\n{'='*80}")
    print("PHASE 1: EMBEDDING-DELTA CORRELATION")
    print("="*80)

    prompt = "The capital of France is"
    results = analyze_embedding_delta_correlation(
        model, tokenizer, prompt, args.k, transmission_layers
    )

    print(f"\nPrompt: \"{prompt}\"")
    print(f"\n{'Layer':>6} | {'Emb-Delta Jaccard':>17} | {'h_in-Delta Jaccard':>18}")
    print("-" * 50)

    for layer_idx in sorted(results['layers'].keys()):
        lr = results['layers'][layer_idx]
        print(f"{layer_idx:>6} | {lr['emb_delta_jaccard']:>17.3f} | {lr['hin_delta_jaccard']:>18.3f}")

    avg_emb = np.mean([r['emb_delta_jaccard'] for r in results['layers'].values()])
    avg_hin = np.mean([r['hin_delta_jaccard'] for r in results['layers'].values()])
    print(f"\nAverage Emb-Delta: {avg_emb:.3f}")
    print(f"Average h_in-Delta: {avg_hin:.3f}")

    # Phase 2: Test h_in-guided compression
    print(f"\n{'='*80}")
    print("PHASE 2: h_in-GUIDED COMPRESSION")
    print("="*80)
    print("Use h_in's top-K dimensions instead of computing delta's top-K")

    matches = 0
    for prompt in TEST_PROMPTS:
        normal, compressed, stats = test_hin_guided_compression(
            model, tokenizer, prompt, args.k, transmission_layers, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        compressed_first = compressed.split()[0] if compressed.split() else "(empty)"
        match = "✓" if normal_first == compressed_first else "✗"
        if normal_first == compressed_first:
            matches += 1

        avg_energy = np.mean([s['energy_kept'] for s in stats])
        print(f"  {prompt[:30]}: {normal_first} → {compressed_first} {match} (energy: {avg_energy:.3f})")

    print(f"\nMatches: {matches}/{len(TEST_PROMPTS)}")
    print(f"Compression: {hidden_dim / args.k:.1f}x")

    # Phase 3: Adaptive K compression
    print(f"\n{'='*80}")
    print("PHASE 3: ADAPTIVE-K COMPRESSION")
    print("="*80)
    print("Use however many dims needed for 95% energy")

    for prompt in TEST_PROMPTS:
        normal, compressed, stats = test_adaptive_k_compression(
            model, tokenizer, prompt, transmission_layers, energy_threshold=0.95, max_tokens=5
        )
        normal_first = normal.split()[0] if normal.split() else "(empty)"
        compressed_first = compressed.split()[0] if compressed.split() else "(empty)"
        match = "✓" if normal_first == compressed_first else "✗"

        avg_k = np.mean([s['k_needed'] for s in stats])
        print(f"  {prompt[:30]}: {normal_first} → {compressed_first} {match} (avg K: {avg_k:.0f})")

    # Phase 4: Find energy threshold that works
    print(f"\n{'='*80}")
    print("PHASE 4: FIND WORKING ENERGY THRESHOLD")
    print("="*80)

    for threshold in [0.80, 0.85, 0.90, 0.95, 0.99, 1.0]:
        matches = 0
        k_values = []

        for prompt in TEST_PROMPTS:
            normal, compressed, stats = test_adaptive_k_compression(
                model, tokenizer, prompt, transmission_layers,
                energy_threshold=threshold, max_tokens=5
            )
            normal_first = normal.split()[0] if normal.split() else "(empty)"
            compressed_first = compressed.split()[0] if compressed.split() else "(empty)"
            if normal_first == compressed_first:
                matches += 1

            k_values.extend([s['k_needed'] for s in stats])

        avg_k = np.mean(k_values)
        compression = hidden_dim / avg_k
        print(f"  {threshold*100:.0f}% energy: {matches}/{len(TEST_PROMPTS)} matches, avg K={avg_k:.0f}, {compression:.1f}x compression")

    # Insight
    print(f"\n{'='*80}")
    print("EMBEDDING-GUIDED INSIGHT")
    print("="*80)

    if avg_hin > 0.4:
        print(f"""
h_in PREDICTS delta's important dimensions!

h_in-Delta Jaccard: {avg_hin:.3f}

This means:
- The dimensions that are large in h_in tend to have large delta
- We can use h_in's structure to guide compression
- No learning needed - pure math!

COMPRESSION STRATEGY:
1. At sender: compute h_in's top-K indices
2. Run layer, keep only those K values from delta
3. At receiver: reconstruct using same indices
4. Both sides compute same indices from h_in!
""")
    else:
        print(f"""
h_in does NOT predict delta's structure.

h_in-Delta Jaccard: {avg_hin:.3f}

Delta's important dimensions emerge from computation,
not from input structure. Need different approach.
""")


if __name__ == "__main__":
    main()
