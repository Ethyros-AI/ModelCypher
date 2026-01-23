#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Dimension Routing Analysis
"""
Dimension Routing Analysis

THE DISCOVERY:
- Zero invariant dimensions across inputs
- Different inputs use different dimensions
- This is SEMANTIC ROUTING

THE QUESTION:
Can we predict WHICH dimensions will be important
from the input embedding alone?

If h_emb → important_dims is learnable, we can:
1. Send a tiny "selector" vector instead of full hidden state
2. Receiver reconstructs using the selector to pick dimensions
3. Massive compression with input-dependent basis

THE EXPERIMENT:
1. Collect (input_embedding, topk_indices) pairs for many inputs
2. See if similar embeddings → similar topk_indices
3. If yes, the routing IS predictable
4. If no, the routing emerges from internal computation

Usage:
    python dimension_routing_analysis.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Organized by semantic category
CATEGORIZED_PROMPTS = {
    "geography": [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Germany is",
        "The largest country is",
    ],
    "math": [
        "2 + 2 =",
        "5 + 5 =",
        "10 - 3 =",
        "100 / 4 =",
    ],
    "opposites": [
        "The opposite of hot is",
        "The opposite of big is",
        "The opposite of fast is",
        "The opposite of happy is",
    ],
    "completion": [
        "Once upon a time",
        "In the beginning",
        "The quick brown fox",
        "To be or not to",
    ],
    "definitions": [
        "Python is a",
        "Machine learning is",
        "The internet is",
        "Artificial intelligence is",
    ],
}


def collect_routing_data(
    model: Any,
    tokenizer: Any,
    prompts_by_category: dict[str, list[str]],
    k: int,
    target_layer: int,
) -> tuple[dict, list[np.ndarray], list[set[int]], list[str]]:
    """Collect embeddings and top-K indices."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    categories = []
    embeddings = []
    topk_indices = []
    prompts_flat = []

    for category, prompts in prompts_by_category.items():
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            # Get embedding
            h = inner_model.embed_tokens(input_ids)
            mx.eval(h)

            # Store embedding of last token
            emb = np.array(h[0, -1, :].astype(mx.float32))
            embeddings.append(emb)
            categories.append(category)
            prompts_flat.append(prompt)

            # Run through layers to get top-K at target layer
            mask = create_attention_mask(h, None)

            for idx, layer in enumerate(inner_model.layers):
                h_in_np = np.array(h[0, -1, :].astype(mx.float32))

                h_true = layer(h, mask, None)
                mx.eval(h_true)

                if idx == target_layer:
                    h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))
                    delta = h_out_np - h_in_np

                    abs_delta = np.abs(delta)
                    topk = set(np.argpartition(abs_delta, -k)[-k:].tolist())
                    topk_indices.append(topk)

                h = h_true

    return {
        'categories': categories,
        'prompts': prompts_flat,
    }, embeddings, topk_indices, prompts_flat


def compute_jaccard(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def analyze_routing_predictability(
    embeddings: list[np.ndarray],
    topk_indices: list[set[int]],
    categories: list[str],
):
    """Analyze if embedding similarity predicts dimension overlap."""

    n = len(embeddings)

    # Compute pairwise similarities
    embedding_sims = []
    routing_sims = []
    same_category = []

    for i in range(n):
        for j in range(i + 1, n):
            # Embedding similarity (cosine)
            emb_sim = 1 - cosine(embeddings[i], embeddings[j])
            embedding_sims.append(emb_sim)

            # Routing similarity (Jaccard overlap of top-K)
            route_sim = compute_jaccard(topk_indices[i], topk_indices[j])
            routing_sims.append(route_sim)

            # Same category?
            same_category.append(categories[i] == categories[j])

    embedding_sims = np.array(embedding_sims)
    routing_sims = np.array(routing_sims)
    same_category = np.array(same_category)

    # Correlation
    corr, p_value = spearmanr(embedding_sims, routing_sims)

    # Within vs between category
    within_route_sim = np.mean(routing_sims[same_category])
    between_route_sim = np.mean(routing_sims[~same_category])

    within_emb_sim = np.mean(embedding_sims[same_category])
    between_emb_sim = np.mean(embedding_sims[~same_category])

    return {
        'correlation': corr,
        'p_value': p_value,
        'within_category_routing_sim': within_route_sim,
        'between_category_routing_sim': between_route_sim,
        'within_category_emb_sim': within_emb_sim,
        'between_category_emb_sim': between_emb_sim,
    }


def analyze_category_routing(
    topk_indices: list[set[int]],
    categories: list[str],
    hidden_dim: int,
):
    """Analyze if categories have distinct routing patterns."""

    # Group by category
    category_dims: dict[str, list[set[int]]] = defaultdict(list)
    for cat, topk in zip(categories, topk_indices):
        category_dims[cat].append(topk)

    category_stats = {}
    for cat, dim_sets in category_dims.items():
        # Union (all dims used by this category)
        union = set()
        for s in dim_sets:
            union |= s

        # Intersection (dims used by ALL in this category)
        intersection = dim_sets[0].copy()
        for s in dim_sets[1:]:
            intersection &= s

        category_stats[cat] = {
            'union_size': len(union),
            'intersection_size': len(intersection),
            'union': union,
            'intersection': intersection,
        }

    # Cross-category analysis
    cat_names = list(category_stats.keys())
    cross_category = np.zeros((len(cat_names), len(cat_names)))

    for i, cat1 in enumerate(cat_names):
        for j, cat2 in enumerate(cat_names):
            union1 = category_stats[cat1]['union']
            union2 = category_stats[cat2]['union']
            cross_category[i, j] = compute_jaccard(union1, union2)

    return category_stats, cross_category, cat_names


def main():
    parser = argparse.ArgumentParser(description="Dimension routing analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--k", type=int, default=543, help="K for top-K")
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
    print("DIMENSION ROUTING ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Top-K: {args.k}")

    # Test at middle transmission layer
    target_layer = 15

    print(f"\n{'='*80}")
    print(f"ANALYZING LAYER {target_layer}")
    print("="*80)

    # Collect data
    meta, embeddings, topk_indices, prompts = collect_routing_data(
        model, tokenizer, CATEGORIZED_PROMPTS, args.k, target_layer
    )

    categories = meta['categories']

    # Analyze predictability
    print(f"\n--- Embedding → Routing Correlation ---")

    stats = analyze_routing_predictability(embeddings, topk_indices, categories)

    print(f"Spearman correlation: {stats['correlation']:.4f} (p={stats['p_value']:.6f})")
    print(f"\nWithin-category:")
    print(f"  Embedding similarity: {stats['within_category_emb_sim']:.4f}")
    print(f"  Routing similarity:   {stats['within_category_routing_sim']:.4f}")
    print(f"\nBetween-category:")
    print(f"  Embedding similarity: {stats['between_category_emb_sim']:.4f}")
    print(f"  Routing similarity:   {stats['between_category_routing_sim']:.4f}")

    # Category analysis
    print(f"\n--- Category Routing Patterns ---")

    cat_stats, cross_cat, cat_names = analyze_category_routing(
        topk_indices, categories, hidden_dim
    )

    print(f"\n{'Category':<12} | {'Union':>8} | {'Intersection':>12}")
    print("-" * 40)
    for cat in cat_names:
        print(f"{cat:<12} | {cat_stats[cat]['union_size']:>8} | {cat_stats[cat]['intersection_size']:>12}")

    print(f"\n--- Cross-Category Jaccard (union overlap) ---")
    print(f"{'':12}", end="")
    for cat in cat_names:
        print(f" | {cat[:8]:>8}", end="")
    print()
    print("-" * (12 + 11 * len(cat_names)))

    for i, cat1 in enumerate(cat_names):
        print(f"{cat1:<12}", end="")
        for j, cat2 in enumerate(cat_names):
            print(f" | {cross_cat[i,j]:>8.3f}", end="")
        print()

    # Check if there are "signature" dimensions for each category
    print(f"\n--- Category Signature Dimensions ---")
    print("(dimensions used by ALL prompts in category but <50% of other categories)")

    for cat in cat_names:
        cat_intersection = cat_stats[cat]['intersection']
        signature_dims = set()

        for dim in cat_intersection:
            # Count how many other categories use this dim
            other_usage = 0
            other_total = 0
            for other_cat in cat_names:
                if other_cat != cat:
                    other_total += 1
                    if dim in cat_stats[other_cat]['union']:
                        other_usage += 1

            if other_total > 0 and other_usage / other_total < 0.5:
                signature_dims.add(dim)

        print(f"  {cat}: {len(signature_dims)} signature dims")

    # Insight
    print(f"\n{'='*80}")
    print("ROUTING INSIGHT")
    print("="*80)

    if stats['correlation'] > 0.5:
        print(f"""
ROUTING IS PREDICTABLE FROM EMBEDDING!

Correlation {stats['correlation']:.3f} means:
- Similar inputs → similar dimension routing
- We can potentially learn a predictor: embedding → active dims

IMPLICATION FOR COMPRESSION:
- Send embedding summary (e.g., 64 dims) instead of full state
- Receiver predicts which dims will be active
- Only transmit those dims

This could achieve {hidden_dim / args.k:.1f}x compression with
a learned routing prediction.
""")
    elif stats['correlation'] > 0.2:
        print(f"""
PARTIAL ROUTING PREDICTABILITY

Correlation {stats['correlation']:.3f} is moderate:
- Some structure, but not fully predictable from embedding
- Routing depends on computation within the layer

KEY FINDING:
Within-category routing sim: {stats['within_category_routing_sim']:.3f}
Between-category routing sim: {stats['between_category_routing_sim']:.3f}

Ratio: {stats['within_category_routing_sim'] / stats['between_category_routing_sim']:.2f}x

This means similar semantic content uses similar dimensions.
""")
    else:
        print(f"""
ROUTING IS NOT PREDICTABLE FROM EMBEDDING

Correlation {stats['correlation']:.3f} is low:
- Dimension selection emerges from layer computation
- Cannot be predicted from input alone

This suggests the model computes WHAT matters dynamically.
Compression must preserve this adaptive selection.
""")


if __name__ == "__main__":
    main()
