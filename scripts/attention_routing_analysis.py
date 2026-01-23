#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Attention Routing Analysis
"""
Attention Routing Analysis

THE HYPOTHESIS:
- Delta's important dimensions emerge from computation, not h_in
- Attention mechanism routes information through different heads
- Each head writes to a subset of dimensions
- Which heads activate depends on input content

THE EXPERIMENT:
1. Track which attention heads are "active" per input
2. See if head activation correlates with dimension importance
3. If yes: attention routing → dimension routing

This would explain:
- Input-dependent dimension routing
- Semantic correlation between similar inputs
- ~50% sparsity (only some heads active)

If attention heads are the routing mechanism:
- Compress by predicting WHICH HEADS will fire
- Much smaller than predicting 543/2048 dimensions

Usage:
    python attention_routing_analysis.py --model /path/to/model
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
    # Geography
    "The capital of France is",
    "The capital of Japan is",
    # Math
    "2 + 2 =",
    "10 - 3 =",
    # Opposites
    "The opposite of hot is",
    "The opposite of big is",
]


def analyze_attention_patterns(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_layer: int,
) -> dict:
    """Analyze attention patterns at a specific layer."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    # Run through layers, capturing attention at target layer
    for idx, layer in enumerate(inner_model.layers):
        if idx == target_layer:
            # Access the attention module
            attn = layer.self_attn

            # Get Q, K, V projections
            B, L, _ = h.shape

            if hasattr(attn, 'q_proj'):
                queries = attn.q_proj(h)
                keys = attn.k_proj(h)
                values = attn.v_proj(h)

                n_heads = attn.n_heads
                n_kv_heads = attn.n_kv_heads
                head_dim = queries.shape[-1] // n_heads

                # Reshape for multi-head
                queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

                mx.eval(queries)
                mx.eval(keys)
                mx.eval(values)

                # Compute attention scores manually for last position only
                scale = head_dim ** -0.5

                # Expand keys/values for grouped query attention
                if n_kv_heads < n_heads:
                    repeats = n_heads // n_kv_heads
                    keys = mx.repeat(keys, repeats, axis=1)
                    values = mx.repeat(values, repeats, axis=1)

                # For last position query, compute scores against all keys
                # queries[:, :, -1:, :] @ keys.transpose -> (B, n_heads, 1, L)
                last_query = queries[:, :, -1:, :]  # (B, n_heads, 1, head_dim)
                scores = (last_query @ keys.transpose(0, 1, 3, 2)) * scale  # (B, n_heads, 1, L)

                # Apply causal mask manually for last position
                # Only attend to positions 0..L-1 (all valid for last position)
                attn_weights = mx.softmax(scores, axis=-1)  # (B, n_heads, 1, L)
                mx.eval(attn_weights)

                # Compute output per head
                # attn_weights @ values -> (B, n_heads, 1, head_dim)
                attn_output = attn_weights @ values
                mx.eval(attn_output)

                # Get numpy arrays
                attn_weights_np = np.array(attn_weights[0, :, 0, :].astype(mx.float32))  # (n_heads, L)
                attn_output_np = np.array(attn_output[0, :, 0, :].astype(mx.float32))  # (n_heads, head_dim)

                # Per-head statistics
                head_stats = []
                for head_idx in range(n_heads):
                    # Attention entropy (how focused is this head?)
                    weights = attn_weights_np[head_idx]
                    entropy = -np.sum(weights * np.log(weights + 1e-10))

                    # Attention norm (how "active" is this head?)
                    output_norm = np.linalg.norm(attn_output_np[head_idx])

                    head_stats.append({
                        'head': head_idx,
                        'entropy': entropy,
                        'output_norm': output_norm,
                        'max_attention': np.max(weights),
                    })

                return {
                    'n_heads': n_heads,
                    'head_dim': head_dim,
                    'head_stats': head_stats,
                    'attn_output_per_head': attn_output_np,
                }

        h_true = layer(h, mask, None)
        mx.eval(h_true)
        h = h_true

    return {}


def analyze_head_dimension_mapping(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_layer: int,
    k: int = 543,
) -> dict:
    """Analyze which dimensions each head contributes to."""
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == target_layer:
            h_in_np = np.array(h[0, -1, :].astype(mx.float32))

            # Run layer to get true output
            h_true = layer(h, mask, None)
            mx.eval(h_true)

            h_out_np = np.array(h_true[0, -1, :].astype(mx.float32))
            delta = h_out_np - h_in_np

            # Get top-K of delta
            delta_abs = np.abs(delta)
            topk_dims = set(np.argpartition(delta_abs, -k)[-k:].tolist())

            # Get the o_proj weight to understand head → output mapping
            attn = layer.self_attn
            o_proj_weight = np.array(attn.o_proj.weight.astype(mx.float32))  # (hidden, hidden)

            n_heads = attn.n_heads
            head_dim = o_proj_weight.shape[0] // n_heads

            # Each head contributes: output_dim = sum over head_dims of (o_proj @ head_output)
            # o_proj has shape (hidden_dim, hidden_dim)
            # The first head_dim columns correspond to head 0, etc.

            # For each head, find which output dimensions it primarily affects
            head_output_dims = []
            for head_idx in range(n_heads):
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                # This head's contribution weight: columns start:end of o_proj
                head_weight = o_proj_weight[:, start:end]  # (hidden_dim, head_dim)

                # Which output dimensions does this head write to most?
                # Use norm across head_dim
                dim_importance = np.linalg.norm(head_weight, axis=1)  # (hidden_dim,)

                # Top dims for this head
                head_top = set(np.argpartition(dim_importance, -50)[-50:].tolist())
                head_output_dims.append(head_top)

            # Check overlap between head's output dims and delta's top-K
            head_delta_overlap = []
            for head_idx, head_dims in enumerate(head_output_dims):
                overlap = len(head_dims & topk_dims) / len(head_dims)
                head_delta_overlap.append({
                    'head': head_idx,
                    'overlap': overlap,
                    'n_head_dims': len(head_dims),
                })

            return {
                'topk_dims': topk_dims,
                'head_output_dims': head_output_dims,
                'head_delta_overlap': head_delta_overlap,
            }

        h_true = layer(h, mask, None)
        mx.eval(h_true)
        h = h_true

    return {}


def main():
    parser = argparse.ArgumentParser(description="Attention routing analysis")
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
    print("ATTENTION ROUTING ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    target_layer = 15  # Middle transmission layer

    # Phase 1: Attention patterns per input
    print(f"\n{'='*80}")
    print(f"PHASE 1: ATTENTION PATTERNS AT LAYER {target_layer}")
    print("="*80)

    all_head_stats = {}
    for prompt in TEST_PROMPTS:
        results = analyze_attention_patterns(model, tokenizer, prompt, target_layer)

        if results:
            print(f"\n--- {prompt[:40]} ---")
            print(f"n_heads: {results['n_heads']}, head_dim: {results['head_dim']}")

            # Find most active heads (by output norm)
            head_stats = results['head_stats']
            sorted_by_norm = sorted(head_stats, key=lambda x: x['output_norm'], reverse=True)

            print("Top 5 most active heads:")
            for hs in sorted_by_norm[:5]:
                print(f"  Head {hs['head']:2d}: norm={hs['output_norm']:.3f}, entropy={hs['entropy']:.3f}")

            all_head_stats[prompt] = head_stats

    # Phase 2: Head consistency across similar prompts
    print(f"\n{'='*80}")
    print("PHASE 2: HEAD ACTIVATION CONSISTENCY")
    print("="*80)

    # Group by category
    categories = {
        'geography': [p for p in TEST_PROMPTS if 'capital' in p],
        'math': [p for p in TEST_PROMPTS if any(op in p for op in ['+', '-'])],
        'opposites': [p for p in TEST_PROMPTS if 'opposite' in p],
    }

    for cat, prompts in categories.items():
        if len(prompts) < 2:
            continue

        # Find top-5 heads for each prompt in category
        top_heads_per_prompt = []
        for prompt in prompts:
            if prompt in all_head_stats:
                sorted_heads = sorted(all_head_stats[prompt], key=lambda x: x['output_norm'], reverse=True)
                top5 = set(h['head'] for h in sorted_heads[:5])
                top_heads_per_prompt.append(top5)

        if len(top_heads_per_prompt) >= 2:
            # Find intersection
            common_heads = top_heads_per_prompt[0]
            for s in top_heads_per_prompt[1:]:
                common_heads = common_heads & s

            print(f"{cat}: common top-5 heads across prompts: {sorted(common_heads)}")

    # Phase 3: Head → dimension mapping
    print(f"\n{'='*80}")
    print(f"PHASE 3: HEAD → DIMENSION MAPPING AT LAYER {target_layer}")
    print("="*80)

    prompt = "The capital of France is"
    mapping = analyze_head_dimension_mapping(model, tokenizer, prompt, target_layer, args.k)

    if mapping:
        print(f"\nPrompt: \"{prompt}\"")
        print(f"Delta top-K dimensions: {len(mapping['topk_dims'])}")

        print(f"\nHead → Delta overlap (how much does each head's output dims overlap with delta's top-K):")

        sorted_overlap = sorted(mapping['head_delta_overlap'], key=lambda x: x['overlap'], reverse=True)
        for ho in sorted_overlap[:10]:
            print(f"  Head {ho['head']:2d}: {ho['overlap']*100:.1f}% overlap")

        # Check if a few heads cover most of delta
        all_head_dims = set()
        n_heads_to_cover = 0
        for ho in sorted_overlap:
            head_dims = mapping['head_output_dims'][ho['head']]
            all_head_dims |= (head_dims & mapping['topk_dims'])
            n_heads_to_cover += 1
            coverage = len(all_head_dims) / len(mapping['topk_dims'])
            if coverage >= 0.9:
                break

        print(f"\n{n_heads_to_cover} heads cover 90% of delta's top-K dimensions")

    # Insight
    print(f"\n{'='*80}")
    print("ATTENTION ROUTING INSIGHT")
    print("="*80)

    if all_head_stats:
        # Check variance in head activation
        first_prompt = TEST_PROMPTS[0]
        first_stats = all_head_stats[first_prompt]
        norms = [h['output_norm'] for h in first_stats]
        norm_std = np.std(norms)
        norm_mean = np.mean(norms)
        cv = norm_std / norm_mean  # Coefficient of variation

        print(f"""
HEAD ACTIVATION ANALYSIS:

Coefficient of variation in head norms: {cv:.3f}
(Higher = more selective head activation)

If CV > 0.5: Attention is SELECTIVE (good for routing)
If CV < 0.2: Attention is UNIFORM (all heads similar)

INTERPRETATION:
- If selective: Different heads handle different content
- If uniform: All heads contribute similarly

For compression:
- Selective → compress by predicting active heads
- Uniform → need to preserve all heads equally
""")


if __name__ == "__main__":
    main()
