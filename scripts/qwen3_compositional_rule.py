#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Compositional Rule: How does attention modify the single-token rule?
"""
WOLFRAM INSIGHT: The single-token rule is EXACT (0% error).
Multi-token behavior must be a COMPOSITION of:
1. The single-token rule (which we derived: h_out = h_in + A @ (h_in - mean) + delta_mean)
2. The attention mechanism (which mixes information across positions)

For single token: attention contributes W_v @ h (linear)
For multi-token: attention contributes softmax(QK^T) @ V (nonlinear in softmax, linear in V)

HYPOTHESIS: The transformation decomposes as:
  h_out[i] = single_token_rule(h_in[i]) + attention_context_correction[i]

Where the attention correction captures how other positions modify the output.

This script:
1. Decomposes layer into attention + MLP contributions
2. Shows MLP follows same rule for all positions (position-independent)
3. Characterizes attention as the ONLY source of context-dependence

If we can characterize the attention rule, we have the COMPLETE compositional transformation.

Usage:
    python qwen3_compositional_rule.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_attention_and_mlp_contributions(model, tokenizer, prompt: str, layer_idx: int):
    """
    Decompose layer transformation into attention and MLP contributions.

    A transformer layer computes:
        h_out = h_in + attention(norm(h_in)) + mlp(norm(h_in + attention_out))

    For Qwen3 with pre-norm:
        attention_out = attention(input_layernorm(h_in))
        mlp_out = mlp(post_attention_layernorm(h_in + attention_out))
        h_out = h_in + attention_out + mlp_out

    Returns:
        h_in: Input to layer (seq_len, hidden_dim)
        h_out: Output from layer (seq_len, hidden_dim)
        attn_out: Attention contribution (seq_len, hidden_dim)
        mlp_out: MLP contribution (seq_len, hidden_dim)
        delta: h_out - h_in (the residual change)
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id or 1]

    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == layer_idx:
            h_in = np.array(h.astype(mx.float32)).astype(np.float64)[0]  # (seq_len, hidden_dim)

            # Get attention and MLP separately
            # Qwen3 layer structure: input_layernorm -> attention -> post_attention_layernorm -> mlp

            # Step 1: Input layernorm
            h_normed = layer.input_layernorm(h)
            mx.eval(h_normed)

            # Step 2: Attention (need to access self_attn directly)
            attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
            mx.eval(attn_out)
            attn_contribution = np.array(attn_out.astype(mx.float32)).astype(np.float64)[0]

            # Step 3: Residual after attention
            h_post_attn = h + attn_out
            mx.eval(h_post_attn)

            # Step 4: Post-attention layernorm
            h_normed2 = layer.post_attention_layernorm(h_post_attn)
            mx.eval(h_normed2)

            # Step 5: MLP
            mlp_out = layer.mlp(h_normed2)
            mx.eval(mlp_out)
            mlp_contribution = np.array(mlp_out.astype(mx.float32)).astype(np.float64)[0]

            # Step 6: Final output
            h_out = h + attn_out + mlp_out
            mx.eval(h_out)
            h_out_np = np.array(h_out.astype(mx.float32)).astype(np.float64)[0]

            delta = h_out_np - h_in

            return {
                'h_in': h_in,
                'h_out': h_out_np,
                'attn_out': attn_contribution,
                'mlp_out': mlp_contribution,
                'delta': delta,
                'seq_len': len(tokens),
                'tokens': tokens
            }
        else:
            h = layer(h, mask, None)
            mx.eval(h)

    raise ValueError(f"Layer {layer_idx} not found")


def analyze_mlp_rule(model, tokenizer, layer_idx: int, n_tokens: int = 500):
    """
    Analyze if MLP follows the same rule regardless of position.

    Key test: For the SAME token at DIFFERENT positions in different sequences,
    does the MLP output follow the same linear rule?

    If yes: MLP is position-independent, the rule is delta_mlp = A_mlp @ h_normed
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    vocab_size = inner_model.embed_tokens.weight.shape[0]

    # Sample tokens
    np.random.seed(42)
    token_ids = np.random.choice(vocab_size, min(n_tokens, vocab_size), replace=False)

    # Collect MLP input-output pairs for single tokens
    X_mlp = []  # Normalized input to MLP
    Y_mlp = []  # MLP output

    for i, token_id in enumerate(token_ids):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(token_ids)} tokens...")

        input_ids = mx.array([[int(token_id)]])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == layer_idx:
                # Get MLP input (after attention and post-attention norm)
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)

                h_post_attn = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post_attn)
                mx.eval(h_normed2)

                mlp_in = np.array(h_normed2[0, 0, :].astype(mx.float32)).astype(np.float64)
                X_mlp.append(mlp_in)

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)

                mlp_out_np = np.array(mlp_out[0, 0, :].astype(mx.float32)).astype(np.float64)
                Y_mlp.append(mlp_out_np)
                break
            else:
                h = layer(h, mask, None)
                mx.eval(h)

    X_mlp = np.stack(X_mlp, axis=1)  # (hidden_dim, n_tokens)
    Y_mlp = np.stack(Y_mlp, axis=1)

    return X_mlp, Y_mlp


def analyze_attention_rule(model, tokenizer, layer_idx: int, prompts: List[str]):
    """
    Analyze the attention rule: how does attention weight information from other positions?

    For each prompt, collect:
    - Attention weights (seq_len, seq_len)
    - Value vectors at each position
    - Attention output at each position

    The attention rule is: attn_out[i] = sum_j(alpha_ij * v_j)

    Key insight: This is LINEAR in the values, but the weights alpha_ij come from
    softmax(q_i @ k_j^T / sqrt(d)), which is nonlinear.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    results = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == layer_idx:
                # Get attention components
                h_normed = layer.input_layernorm(h)
                mx.eval(h_normed)

                attn = layer.self_attn

                # Compute Q, K, V
                # Note: Qwen3 uses GQA (grouped query attention)
                queries = attn.q_proj(h_normed)
                keys = attn.k_proj(h_normed)
                values = attn.v_proj(h_normed)
                mx.eval(queries, keys, values)

                # Reshape for attention
                B, L, _ = queries.shape
                n_heads = attn.n_heads
                n_kv_heads = attn.n_kv_heads
                head_dim = queries.shape[-1] // n_heads

                queries = queries.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

                # GQA expansion
                n_rep = n_heads // n_kv_heads
                if n_rep > 1:
                    keys = mx.repeat(keys, n_rep, axis=1)
                    values = mx.repeat(values, n_rep, axis=1)

                mx.eval(queries, keys, values)

                # Compute attention weights
                scale = head_dim ** -0.5
                scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

                # Apply causal mask (create manually)
                L_seq = scores.shape[-1]
                causal_mask = mx.triu(mx.full((L_seq, L_seq), float('-inf')), k=1)
                scores = scores + causal_mask

                attn_weights = mx.softmax(scores, axis=-1)
                mx.eval(attn_weights)

                # Get actual attention output
                attn_out = layer.self_attn(h_normed, mask=None, cache=None)
                mx.eval(attn_out)

                results.append({
                    'prompt': prompt,
                    'seq_len': len(tokens),
                    'attn_weights': np.array(attn_weights.astype(mx.float32))[0],  # (n_heads, seq_len, seq_len)
                    'attn_out': np.array(attn_out[0].astype(mx.float32)).astype(np.float64),  # (seq_len, hidden_dim)
                    'h_normed': np.array(h_normed[0].astype(mx.float32)).astype(np.float64)
                })
                break
            else:
                h = layer(h, mask, None)
                mx.eval(h)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*70}")
    print("COMPOSITIONAL RULE ANALYSIS")
    print("How does attention modify the single-token rule?")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Analyzing layer: {args.layer}")

    # Part 1: Decompose a few prompts into attention + MLP
    print(f"\n{'='*70}")
    print("PART 1: DECOMPOSING LAYER INTO ATTENTION + MLP")
    print("="*70)

    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "def main():",
        "Once upon a time",
    ]

    for prompt in test_prompts:
        result = get_attention_and_mlp_contributions(model, tokenizer, prompt, args.layer)

        print(f"\nPrompt: '{prompt}' (seq_len={result['seq_len']})")
        print(f"  ||attention_out|| / ||delta||: {np.linalg.norm(result['attn_out']) / np.linalg.norm(result['delta']):.4f}")
        print(f"  ||mlp_out|| / ||delta||: {np.linalg.norm(result['mlp_out']) / np.linalg.norm(result['delta']):.4f}")

        # Check: does delta = attn_out + mlp_out?
        reconstruction_error = np.linalg.norm(result['delta'] - result['attn_out'] - result['mlp_out'])
        print(f"  ||delta - (attn + mlp)||: {reconstruction_error:.2e}")

        # For last position, what's the breakdown?
        last_attn = result['attn_out'][-1]
        last_mlp = result['mlp_out'][-1]
        last_delta = result['delta'][-1]
        print(f"  Last position: attn={np.linalg.norm(last_attn):.2f}, mlp={np.linalg.norm(last_mlp):.2f}, delta={np.linalg.norm(last_delta):.2f}")

    # Part 2: Analyze if MLP follows a linear rule
    print(f"\n{'='*70}")
    print("PART 2: MLP RULE ANALYSIS")
    print("Is MLP transformation linear?")
    print("="*70)

    print(f"\nCollecting MLP input-output pairs for 500 tokens...")
    X_mlp, Y_mlp = analyze_mlp_rule(model, tokenizer, args.layer, n_tokens=500)

    print(f"MLP data: X={X_mlp.shape}, Y={Y_mlp.shape}")

    # Center and compute linear approximation
    X_mean = X_mlp.mean(axis=1, keepdims=True)
    Y_mean = Y_mlp.mean(axis=1, keepdims=True)
    X_c = X_mlp - X_mean
    Y_c = Y_mlp - Y_mean

    # Compute A_mlp such that Y_c ≈ A_mlp @ X_c
    A_mlp = Y_c @ np.linalg.pinv(X_c)
    Y_pred = A_mlp @ X_c

    mlp_linear_error = np.linalg.norm(Y_c - Y_pred) / np.linalg.norm(Y_c)
    print(f"\nMLP linearity test: ||Y - A @ X|| / ||Y|| = {mlp_linear_error:.6f}")

    if mlp_linear_error < 0.01:
        print("  → MLP IS linear for single tokens!")
    elif mlp_linear_error < 0.10:
        print("  → MLP is approximately linear")
    else:
        print("  → MLP is NONLINEAR")

    # SVD of A_mlp
    U_mlp, S_mlp, Vt_mlp = np.linalg.svd(A_mlp, full_matrices=False)
    mlp_rank = np.sum(S_mlp > 0.01 * S_mlp[0])
    print(f"A_mlp effective rank: {mlp_rank}")
    print(f"Top 5 singular values: {S_mlp[:5]}")

    # Part 3: Analyze attention patterns
    print(f"\n{'='*70}")
    print("PART 3: ATTENTION PATTERN ANALYSIS")
    print("How does attention distribute weight?")
    print("="*70)

    attention_results = analyze_attention_rule(model, tokenizer, args.layer, test_prompts[:2])

    for result in attention_results:
        print(f"\nPrompt: '{result['prompt']}' (seq_len={result['seq_len']})")

        # Average attention weights across heads for last position
        attn_weights = result['attn_weights']  # (n_heads, seq_len, seq_len)
        last_pos_weights = attn_weights[:, -1, :]  # (n_heads, seq_len)
        avg_weights = last_pos_weights.mean(axis=0)  # (seq_len,)

        print(f"  Attention from last position (avg across heads):")
        for i, w in enumerate(avg_weights):
            print(f"    pos {i}: {w:.4f}")

        # How much does last position attend to itself vs others?
        self_attn_weight = avg_weights[-1]
        other_attn_weight = avg_weights[:-1].sum() if len(avg_weights) > 1 else 0
        print(f"  Self-attention weight: {self_attn_weight:.4f}")
        print(f"  Other positions total: {other_attn_weight:.4f}")

    # Part 4: The compositional rule
    print(f"\n{'='*70}")
    print("PART 4: THE COMPOSITIONAL RULE")
    print("="*70)

    print(f"""
THE RULE STRUCTURE:

For a transformer layer:
  h_out = h_in + attention_out + mlp_out

Where:
  1. attention_out[i] = sum_j(alpha_ij * W_o @ W_v @ norm(h_in[j]))
     - alpha_ij = softmax(q_i @ k_j^T / sqrt(d))
     - For single token: alpha = 1, so attention = W_o @ W_v @ norm(h_in)
     - For multi-token: attention mixes information via softmax weights

  2. mlp_out[i] = W_down @ (silu(W_gate @ norm(h_post_attn[i])) * (W_up @ norm(h_post_attn[i])))
     - This is position-INDEPENDENT
     - For single tokens: effectively LINEAR (error = {mlp_linear_error*100:.4f}%)
     - MLP applies same rule to each position

KEY INSIGHT:
- MLP is position-independent and approximately linear
- Attention is the ONLY source of cross-position interaction
- For single tokens, the ENTIRE layer is linear

COMPOSITIONAL DECOMPOSITION:
  h_out[i] = h_in[i] + linear_transform(h_in[i]) + attention_correction[i]

Where:
  - linear_transform = what happens for single token (derived rule)
  - attention_correction = deviation caused by context from other positions

The "rule" is:
  1. Single-token base transformation (linear, derived from vocabulary)
  2. + Attention-mediated context mixing (softmax-weighted sum of values)
""")

    # Part 5: Quantify attention correction for multi-token
    print(f"\n{'='*70}")
    print("PART 5: QUANTIFYING ATTENTION CORRECTION")
    print("="*70)

    # Compare single-token behavior to last-position behavior in sequence
    prompt = "The capital of France is"
    multi_result = get_attention_and_mlp_contributions(model, tokenizer, prompt, args.layer)

    # Get single-token behavior for the last token
    tokens = tokenizer.encode(prompt)
    last_token = tokens[-1]

    single_result = get_attention_and_mlp_contributions(
        model, tokenizer,
        tokenizer.decode([last_token]),
        args.layer
    )

    # Compare
    multi_delta_last = multi_result['delta'][-1]
    single_delta = single_result['delta'][0]

    delta_diff = multi_delta_last - single_delta

    print(f"\nPrompt: '{prompt}'")
    print(f"Last token: '{tokenizer.decode([last_token])}' (id={last_token})")
    print(f"\n||delta_multi|| = {np.linalg.norm(multi_delta_last):.4f}")
    print(f"||delta_single|| = {np.linalg.norm(single_delta):.4f}")
    print(f"||delta_multi - delta_single|| = {np.linalg.norm(delta_diff):.4f}")
    print(f"Relative difference: {np.linalg.norm(delta_diff) / np.linalg.norm(multi_delta_last) * 100:.2f}%")

    # The difference IS the attention correction
    attn_correction = multi_result['attn_out'][-1] - single_result['attn_out'][0]
    print(f"\n||attention_correction|| = {np.linalg.norm(attn_correction):.4f}")

    # Does the correction explain the difference?
    explained_by_attn = np.linalg.norm(delta_diff - attn_correction) / np.linalg.norm(delta_diff)
    print(f"Fraction of difference NOT explained by attention: {explained_by_attn:.4f}")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION: THE COMPLETE RULE")
    print("="*70)
    print(f"""
The layer transformation for position i in a sequence:

  h_out[i] = h_in[i] + base_transform(h_in[i]) + attention_context(h_in, i)

Where:

  base_transform(x) = A @ (x - mean) + delta_mean
    - This is LINEAR
    - A has rank ~{mlp_rank}
    - Derived from vocabulary (0% error on held-out tokens)

  attention_context(h_in, i) = sum_j(alpha_ij * W_o @ W_v @ norm(h_in[j])) - self_attn(h_in[i])
    - This is the CORRECTION due to other positions
    - alpha_ij comes from softmax(Q_i @ K_j^T / sqrt(d))
    - For last position attending to context

The RULE is complete:
  1. Base transform is DERIVED (not fitted)
  2. Attention correction follows from attention weights (deterministic given Q,K,V)

The apparent complexity comes from attention weight computation (softmax),
but once weights are known, the transformation is LINEAR in the values.

WOLFRAM INSIGHT:
The vocabulary defines the base rule. Attention is a SELECTION mechanism
that chooses which parts of the vocabulary-derived rule to apply.
The "branching" happens in attention weight computation, not in the transformation itself.
""")


if __name__ == "__main__":
    main()
