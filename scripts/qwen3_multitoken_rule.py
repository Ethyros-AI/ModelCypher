#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Multi-Token Rule: Derive MLP rule from actual multi-token sequences
"""
KEY INSIGHT:

The single-token rule works perfectly for single-token inputs.
But for multi-token sequences, the MLP input (h_normed2) is DIFFERENT
because attention mixes information from all positions.

SOLUTION:
Derive the MLP rule from MULTI-TOKEN sequences, using the actual
distribution of h_normed2 values at the LAST position.

This should capture the true activation distribution and enable
chained compression.

Usage:
    python qwen3_multitoken_rule.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import Dict, List, Tuple
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_diverse_prompts() -> List[str]:
    """Generate diverse multi-token prompts for calibration."""
    prompts = []

    # Geography
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Thailand", "Vietnam", "South Korea", "Poland", "Sweden", "Norway",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"{c} is known for its")

    # Math
    for a in range(1, 20):
        for b in range(1, 20):
            prompts.append(f"{a} + {b} =")

    # Code
    code_prompts = [
        "def main():", "class User:", "import numpy as", "from typing import",
        "def __init__(self,", "return self.", "if __name__ ==",
        "for i in range(", "while True:", "try:", "except Exception as",
    ]
    prompts.extend(code_prompts)

    # Natural language
    nl_prompts = [
        "Once upon a time", "The quick brown fox", "What is the meaning of",
        "How does photosynthesis", "The speed of light is", "Hello, my name is",
        "In the year 2024", "The best way to learn", "Scientists have discovered",
        "According to recent studies", "The weather today is", "I believe that",
    ]
    prompts.extend(nl_prompts)

    return prompts


def derive_multitoken_mlp_rule(model, tokenizer, layer_idx: int, prompts: List[str]) -> Dict:
    """
    Derive the MLP rule from MULTI-TOKEN sequences.

    This uses the actual h_normed2 values at the LAST position of each prompt.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    layer = inner_model.layers[layer_idx]

    X_mlp_list = []  # Input to MLP (h_normed2 at last position)
    Y_mlp_list = []  # Output from MLP

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        # Forward to target layer
        for idx, l in enumerate(inner_model.layers):
            if idx == layer_idx:
                # Compute attention
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)

                # Post-attention
                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                # Get LAST position MLP input/output
                mlp_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_mlp_list.append(mlp_in)

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)

                mlp_out_np = np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64)
                Y_mlp_list.append(mlp_out_np)
                break
            else:
                h = l(h, mask, None)
                mx.eval(h)

    X_mlp = np.stack(X_mlp_list, axis=1)
    Y_mlp = np.stack(Y_mlp_list, axis=1)

    # Compute linear approximation
    X_mean = X_mlp.mean(axis=1, keepdims=True)
    Y_mean = Y_mlp.mean(axis=1, keepdims=True)
    X_c = X_mlp - X_mean
    Y_c = Y_mlp - Y_mean

    A = Y_c @ np.linalg.pinv(X_c)

    # SVD for compression
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Error
    Y_pred = A @ X_c
    error = np.linalg.norm(Y_c - Y_pred) / (np.linalg.norm(Y_c) + 1e-10)

    eff_rank = np.sum(S > 0.01 * S[0]) if len(S) > 0 else 0

    return {
        'A': A,
        'U': U,
        'S': S,
        'Vt': Vt,
        'X_mean': X_mean.flatten(),
        'Y_mean': Y_mean.flatten(),
        'error': error,
        'eff_rank': eff_rank,
        'hidden_dim': hidden_dim,
        'n_samples': len(prompts)
    }


def compress_rule(rule: Dict, target_rank: int) -> Dict:
    """Compress rule to target rank."""
    k = min(target_rank, len(rule['S']))

    U_k = rule['U'][:, :k]
    S_k = rule['S'][:k]
    Vt_k = rule['Vt'][:k, :]

    A_compressed = U_k @ np.diag(S_k) @ Vt_k
    A_error = np.linalg.norm(rule['A'] - A_compressed) / (np.linalg.norm(rule['A']) + 1e-10)

    return {
        'U': U_k,
        'S': S_k,
        'Vt': Vt_k,
        'X_mean': rule['X_mean'],
        'Y_mean': rule['Y_mean'],
        'rank': k,
        'compression_error': A_error,
        'hidden_dim': rule['hidden_dim']
    }


def apply_compressed_mlp(h_normed2: np.ndarray, compressed: Dict) -> np.ndarray:
    """Apply compressed MLP rule."""
    h_centered = h_normed2 - compressed['X_mean']
    x = compressed['Vt'] @ h_centered
    x = compressed['S'] * x
    y_centered = compressed['U'] @ x
    return y_centered + compressed['Y_mean']


def test_compression(model, tokenizer, compressed_rules: Dict[int, Dict],
                     test_prompts: List[str], layer_range: Tuple[int, int]) -> Dict:
    """Test hybrid compression."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    start_layer, end_layer = layer_range

    results = []
    matches = 0

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        # Original
        input_ids = mx.array([tokens])
        logits_orig = model(input_ids)
        mx.eval(logits_orig)
        orig_token = int(np.argmax(np.array(logits_orig[0, -1, :].astype(mx.float32))))

        # Compressed
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if start_layer <= idx <= end_layer and idx in compressed_rules:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)

                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                # Compressed MLP for last position
                h_normed2_np = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                mlp_out_last = apply_compressed_mlp(h_normed2_np, compressed_rules[idx])

                # Original MLP for other positions
                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)

                # Replace last position
                mlp_out_np = np.array(mlp_out.astype(mx.float32))
                mlp_out_np[0, -1, :] = mlp_out_last.astype(np.float32)
                mlp_out = mx.array(mlp_out_np).astype(h.dtype)

                h = h_post + mlp_out
                mx.eval(h)
            else:
                h = layer(h, mask, None)
                mx.eval(h)

        h = inner_model.norm(h)
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(h)
        else:
            logits = inner_model.embed_tokens.as_linear(h)
        mx.eval(logits)

        comp_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

        match = (orig_token == comp_token)
        if match:
            matches += 1

        results.append({
            'prompt': prompt,
            'original': tokenizer.decode([orig_token]),
            'compressed': tokenizer.decode([comp_token]),
            'match': match
        })

    return {
        'accuracy': matches / len(test_prompts) if test_prompts else 0,
        'matches': matches,
        'total': len(test_prompts),
        'results': results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target-rank", type=int, default=500)
    parser.add_argument("--start-layer", type=int, default=15)
    parser.add_argument("--end-layer", type=int, default=20)
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
    print("MULTI-TOKEN RULE DERIVATION")
    print("Using actual multi-token sequences for calibration")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Compressing layers {args.start_layer}-{args.end_layer}")

    # Generate calibration prompts
    calibration = generate_diverse_prompts()
    print(f"Calibration: {len(calibration)} multi-token prompts")

    # Derive rules
    print(f"\n{'='*70}")
    print("DERIVING MULTI-TOKEN MLP RULES")
    print("="*70)

    compressed_rules = {}

    for layer_idx in range(args.start_layer, args.end_layer + 1):
        print(f"\nLayer {layer_idx}:")

        t0 = time.time()
        rule = derive_multitoken_mlp_rule(model, tokenizer, layer_idx, calibration)
        t1 = time.time()

        print(f"  Derivation: {t1-t0:.1f}s ({rule['n_samples']} samples)")
        print(f"  MLP linear error: {rule['error']*100:.6f}%")
        print(f"  Effective rank: {rule['eff_rank']}")

        compressed = compress_rule(rule, args.target_rank)
        compressed_rules[layer_idx] = compressed

        print(f"  Compressed to rank {compressed['rank']}")
        print(f"  Compression error: {compressed['compression_error']*100:.4f}%")

    # Test
    print(f"\n{'='*70}")
    print("TESTING COMPRESSION")
    print("="*70)

    # Held-out test prompts
    test_prompts = [
        "The capital of Mongolia is",  # Held-out country
        "The capital of Nepal is",
        "99 + 88 =",  # Larger numbers
        "def factorial(",  # Different function
        "Scientists believe that",  # Different phrasing
        "The history of programming",
        "Why do birds fly",
        "Explain quantum computing",
        "The tallest mountain in",
        "Write a function to",
    ]

    results = test_compression(
        model, tokenizer, compressed_rules,
        test_prompts, (args.start_layer, args.end_layer)
    )

    print(f"\nHeld-out accuracy: {results['matches']}/{results['total']} ({results['accuracy']*100:.1f}%)")

    print(f"\nDetailed results:")
    for r in results['results']:
        status = "Y" if r['match'] else "X"
        print(f"  {status} '{r['prompt']}' -> orig='{r['original']}', comp='{r['compressed']}'")


if __name__ == "__main__":
    main()
