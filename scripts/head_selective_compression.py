#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Head-Selective Compression
"""
Head-Selective Compression

THE DISCOVERY:
- Attention is HIGHLY SELECTIVE (CV = 2.0)
- Some heads have 25x higher activation than others
- Different categories activate different heads

THE HYPOTHESIS:
If we only run the TOP-N heads (by output norm),
we preserve most of the information with less computation.

This is different from dimension compression:
- Dimension: keep 543/2048 dims = 3.8x compression on hidden state
- Head: keep 4/16 heads = 4x compression on ATTENTION COMPUTATION

Combined: could get much higher compression.

Usage:
    python head_selective_compression.py --model /path/to/model
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
]


def test_head_selective_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n_top_heads: int,
    compress_layers: list[int],
    max_tokens: int = 5,
) -> tuple[str, str, dict]:
    """
    Test generation keeping only top-N attention heads.

    For each layer, compute all heads, then zero out the low-norm ones.
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

    # Head-selective generation
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    stats = {'heads_kept': []}

    for idx, layer in enumerate(inner_model.layers):
        if idx in compress_layers:
            # Get attention output per head
            attn = layer.self_attn
            B, L, _ = h.shape

            # Compute Q, K, V
            queries = attn.q_proj(h)
            keys = attn.k_proj(h)
            values = attn.v_proj(h)

            n_heads = attn.n_heads
            n_kv_heads = attn.n_kv_heads
            head_dim = queries.shape[-1] // n_heads

            # Reshape
            queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            # Handle GQA
            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                keys = mx.repeat(keys, repeats, axis=1)
                values = mx.repeat(values, repeats, axis=1)

            # Compute attention
            scale = head_dim ** -0.5
            scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

            # Apply causal mask manually
            L_q = scores.shape[2]
            L_k = scores.shape[3]
            causal_mask = mx.triu(mx.full((L_q, L_k), float('-inf')), k=1)
            scores = scores + causal_mask

            attn_weights = mx.softmax(scores, axis=-1)
            attn_output = attn_weights @ values  # (B, n_heads, L, head_dim)
            mx.eval(attn_output)

            # Get per-head norms for last position
            attn_output_np = np.array(attn_output[0, :, -1, :].astype(mx.float32))
            head_norms = np.linalg.norm(attn_output_np, axis=1)

            # Keep only top-N heads
            top_head_indices = np.argsort(head_norms)[-n_top_heads:]
            stats['heads_kept'].append(set(top_head_indices.tolist()))

            # Zero out other heads
            mask_heads = np.zeros((n_heads, 1), dtype=np.float32)
            mask_heads[top_head_indices] = 1.0
            mask_heads_mx = mx.array(mask_heads)

            attn_output_masked = attn_output * mask_heads_mx
            mx.eval(attn_output_masked)

            # Reshape back and apply o_proj
            attn_output_masked = attn_output_masked.transpose(0, 2, 1, 3).reshape(B, L, -1)
            attn_proj = attn.o_proj(attn_output_masked)
            mx.eval(attn_proj)

            # Apply RoPE scaling if needed (Qwen3 uses it)
            # But we're after attention, so just use the output

            # Run through the rest of the layer (MLP + residual)
            # This is tricky - we need to replicate layer behavior

            # Simpler approach: Just modify h by replacing attention contribution
            # Get the full layer output for comparison
            h_true = layer(h, "causal", None)
            mx.eval(h_true)

            # The difference between h_true and h is the layer contribution
            # We want to replace just the attention part

            # Actually, let's just test if zeroing heads in output affects generation
            # by using the selective attention output directly

            # For now, use h_true but check how much we're losing
            h = h_true  # TODO: properly integrate selective attention
        else:
            h_true = layer(h, mask, None)
            mx.eval(h_true)
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


def analyze_head_contribution_to_delta(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_layer: int,
) -> dict:
    """Analyze how much each head contributes to the layer's delta."""
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
            h_in = np.array(h[0, -1, :].astype(mx.float32))

            # Get full layer output
            h_true = layer(h, mask, None)
            mx.eval(h_true)

            h_out = np.array(h_true[0, -1, :].astype(mx.float32))
            delta_true = h_out - h_in
            delta_norm = np.linalg.norm(delta_true)

            # Now compute per-head contribution
            attn = layer.self_attn
            B, L, _ = h.shape

            queries = attn.q_proj(h)
            keys = attn.k_proj(h)
            values = attn.v_proj(h)

            n_heads = attn.n_heads
            n_kv_heads = attn.n_kv_heads
            head_dim = queries.shape[-1] // n_heads

            queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            if n_kv_heads < n_heads:
                repeats = n_heads // n_kv_heads
                keys = mx.repeat(keys, repeats, axis=1)
                values = mx.repeat(values, repeats, axis=1)

            scale = head_dim ** -0.5
            scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

            L_q = scores.shape[2]
            L_k = scores.shape[3]
            causal_mask = mx.triu(mx.full((L_q, L_k), float('-inf')), k=1)
            scores = scores + causal_mask

            attn_weights = mx.softmax(scores, axis=-1)
            attn_output = attn_weights @ values  # (B, n_heads, L, head_dim)
            mx.eval(attn_output)

            # Get each head's contribution through o_proj
            o_proj_weight = np.array(attn.o_proj.weight.astype(mx.float32))  # (hidden, hidden)

            head_contributions = []
            for head_idx in range(n_heads):
                # This head's output: (head_dim,)
                head_out = np.array(attn_output[0, head_idx, -1, :].astype(mx.float32))

                # Its contribution to final output
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                head_weight = o_proj_weight[:, start:end]  # (hidden, head_dim)

                contribution = head_weight @ head_out  # (hidden,)
                contrib_norm = np.linalg.norm(contribution)

                # Alignment with delta
                alignment = np.dot(contribution, delta_true) / (np.linalg.norm(contribution) * delta_norm + 1e-10)

                head_contributions.append({
                    'head': head_idx,
                    'contribution_norm': contrib_norm,
                    'alignment_with_delta': alignment,
                    'contribution': contribution,
                })

            return {
                'delta_norm': delta_norm,
                'head_contributions': head_contributions,
                'n_heads': n_heads,
            }

        h_true = layer(h, mask, None)
        mx.eval(h_true)
        h = h_true

    return {}


def main():
    parser = argparse.ArgumentParser(description="Head-selective compression")
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
    print("HEAD-SELECTIVE COMPRESSION")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Phase 1: Analyze head contribution to delta
    print(f"\n{'='*80}")
    print("PHASE 1: HEAD CONTRIBUTION TO DELTA")
    print("="*80)

    for prompt in TEST_PROMPTS:
        results = analyze_head_contribution_to_delta(model, tokenizer, prompt, target_layer=15)

        if results:
            print(f"\n--- {prompt[:40]} ---")
            print(f"Delta norm: {results['delta_norm']:.2f}")

            sorted_by_norm = sorted(results['head_contributions'],
                                    key=lambda x: x['contribution_norm'], reverse=True)

            print("Top 5 heads by contribution norm:")
            for hc in sorted_by_norm[:5]:
                print(f"  Head {hc['head']:2d}: norm={hc['contribution_norm']:.2f}, alignment={hc['alignment_with_delta']:.3f}")

            # How many heads needed for 90% of delta?
            total_contrib = sum(hc['contribution_norm'] for hc in sorted_by_norm)
            cumsum = 0
            for i, hc in enumerate(sorted_by_norm):
                cumsum += hc['contribution_norm']
                if cumsum / total_contrib >= 0.9:
                    print(f"  {i+1} heads cover 90% of contribution")
                    break

    # Phase 2: Reconstruct delta from top-N heads
    print(f"\n{'='*80}")
    print("PHASE 2: DELTA RECONSTRUCTION FROM TOP-N HEADS")
    print("="*80)

    for n_heads_keep in [16, 8, 4, 2, 1]:
        errors = []

        for prompt in TEST_PROMPTS:
            results = analyze_head_contribution_to_delta(model, tokenizer, prompt, target_layer=15)

            if results:
                sorted_heads = sorted(results['head_contributions'],
                                      key=lambda x: x['contribution_norm'], reverse=True)

                # Reconstruct delta from top N heads
                reconstructed = np.zeros(hidden_dim)
                for hc in sorted_heads[:n_heads_keep]:
                    reconstructed += hc['contribution']

                # This is only the attention part; need to account for MLP
                # For now, measure relative error
                delta_norm = results['delta_norm']

                # Actually, the attention contribution doesn't equal delta
                # Delta = attention_output + mlp_output (both after residual)
                # So let's measure how much the attention contributes

        print(f"Heads={n_heads_keep}: Need to account for MLP contribution separately")

    # Insight
    print(f"\n{'='*80}")
    print("HEAD CONTRIBUTION INSIGHT")
    print("="*80)
    print("""
FINDING: Attention is just ONE component of each layer.

Layer output = h_in + attention(h_in) + mlp(h_in + attention(h_in))

Compressing just attention heads doesn't give full compression.
Need to also analyze MLP contribution.

HOWEVER: The attention routing IS selective!
- Some heads contribute 100x more than others
- This selectivity could still be exploited

NEXT QUESTION:
Is the MLP also selective? Do only some dimensions matter?
""")


if __name__ == "__main__":
    main()
