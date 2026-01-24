#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Scaling Test: Verify 100 prompts/layer hypothesis
"""
Previous findings:
- 6 layers with 426 prompts (71/layer) → 100%
- 16 layers with 215 prompts (13/layer) → 33%

Hypothesis: Need ~100 prompts/layer for 100% accuracy.

Test: 16 layers with 1600 prompts (100/layer)
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


def generate_scaled_prompts(target_count: int = 1600) -> List[str]:
    """Generate target number of prompts using proven patterns."""
    prompts = []

    # Geography - extended list
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Thailand", "Vietnam", "South Korea", "Poland", "Sweden", "Norway",
        "Argentina", "Chile", "Peru", "Colombia", "Indonesia", "Philippines",
        "Malaysia", "Singapore", "New Zealand", "South Africa", "Nigeria",
        "Kenya", "Morocco", "Iran", "Iraq", "Saudi Arabia", "Israel", "Pakistan",
        "Bangladesh", "Nepal", "Ukraine", "Romania", "Hungary", "Austria",
        "Switzerland", "Belgium", "Netherlands", "Portugal", "Greece", "Finland",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"{c} is known for its")

    # Math - expanded grid 25x25 = 625
    for a in range(1, 26):
        for b in range(1, 26):
            prompts.append(f"{a} + {b} =")

    # Multiplication 10x10 = 100
    for a in range(2, 12):
        for b in range(2, 12):
            prompts.append(f"{a} * {b} =")

    # Code patterns
    code_prompts = [
        "def main():", "class User:", "import numpy as", "from typing import",
        "def __init__(self,", "return self.", "if __name__ ==",
        "for i in range(", "while True:", "try:", "except Exception as",
        "def process(", "class Config:", "import torch", "from collections import",
        "def run(", "async def main(", "yield", "raise ValueError",
    ]
    prompts.extend(code_prompts)

    # Natural language
    nl_prompts = [
        "Once upon a time", "The quick brown fox", "What is the meaning of",
        "How does photosynthesis", "The speed of light is", "Hello, my name is",
        "In the year 2024", "The best way to learn", "Scientists have discovered",
        "According to recent studies", "The weather today is", "I believe that",
        "On the other hand", "Furthermore,", "However,", "Therefore,",
        "First,", "Second,", "Finally,", "In conclusion,",
    ]
    prompts.extend(nl_prompts)

    # Subtraction (new category) 15x15 = 225
    for a in range(5, 20):
        for b in range(1, min(a, 16)):
            prompts.append(f"{a} - {b} =")

    # Questions (new category)
    q_words = ["What", "Why", "How", "When", "Where", "Who"]
    q_verbs = ["is", "are", "does", "do", "can", "will"]
    for qw in q_words:
        for qv in q_verbs:
            prompts.append(f"{qw} {qv} the")

    # Truncate or pad to target
    if len(prompts) > target_count:
        prompts = prompts[:target_count]

    return prompts


def derive_multitoken_mlp_rule(model, tokenizer, layer_idx: int, prompts: List[str]) -> Dict:
    """Derive MLP rule from multi-token sequences (proven approach)."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    layer = inner_model.layers[layer_idx]

    X_mlp_list = []
    Y_mlp_list = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, l in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)

                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

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

    X_mean = X_mlp.mean(axis=1, keepdims=True)
    Y_mean = Y_mlp.mean(axis=1, keepdims=True)
    X_c = X_mlp - X_mean
    Y_c = Y_mlp - Y_mean

    # Use pinv (proven to work with 426 prompts)
    A = Y_c @ np.linalg.pinv(X_c)

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    Y_pred = A @ X_c
    error = np.linalg.norm(Y_c - Y_pred) / (np.linalg.norm(Y_c) + 1e-10)

    eff_rank = np.sum(S > 0.01 * S[0]) if len(S) > 0 else 0

    return {
        'A': A, 'U': U, 'S': S, 'Vt': Vt,
        'X_mean': X_mean.flatten(), 'Y_mean': Y_mean.flatten(),
        'error': error, 'eff_rank': eff_rank,
        'hidden_dim': hidden_dim, 'n_samples': len(prompts)
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
        'U': U_k, 'S': S_k, 'Vt': Vt_k,
        'X_mean': rule['X_mean'], 'Y_mean': rule['Y_mean'],
        'rank': k, 'compression_error': A_error,
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

        input_ids = mx.array([tokens])
        logits_orig = model(input_ids)
        mx.eval(logits_orig)
        orig_token = int(np.argmax(np.array(logits_orig[0, -1, :].astype(mx.float32))))

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

                h_normed2_np = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                mlp_out_last = apply_compressed_mlp(h_normed2_np, compressed_rules[idx])

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)

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
    parser.add_argument("--start-layer", type=int, default=10)
    parser.add_argument("--end-layer", type=int, default=25)
    parser.add_argument("--calibration-size", type=int, default=1600)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    n_compressed = args.end_layer - args.start_layer + 1
    prompts_per_layer = args.calibration_size // n_compressed

    print(f"\n{'='*70}")
    print("SCALING LAW TEST")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Compressing layers {args.start_layer}-{args.end_layer} ({n_compressed} layers)")
    print(f"Calibration: {args.calibration_size} prompts ({prompts_per_layer}/layer)")

    # Generate calibration prompts
    calibration = generate_scaled_prompts(args.calibration_size)
    print(f"Actual calibration prompts: {len(calibration)}")

    # Derive rules
    print(f"\n{'='*70}")
    print("DERIVING MLP RULES")
    print("="*70)

    compressed_rules = {}

    for layer_idx in range(args.start_layer, args.end_layer + 1):
        t0 = time.time()
        rule = derive_multitoken_mlp_rule(model, tokenizer, layer_idx, calibration)
        t1 = time.time()

        compressed = compress_rule(rule, args.target_rank)
        compressed_rules[layer_idx] = compressed

        print(f"Layer {layer_idx}: error={rule['error']*100:.4f}%, rank={rule['eff_rank']}, "
              f"comp_err={compressed['compression_error']*100:.4f}%, time={t1-t0:.1f}s")

    # Held-out test prompts
    test_prompts = [
        "The capital of Mongolia is",
        "The capital of Nepal is",
        "99 + 88 =",
        "def factorial(",
        "Scientists believe that",
        "The history of programming",
        "Why do birds fly",
        "Explain quantum computing",
        "The tallest mountain in",
        "Write a function to",
        "23 * 17 =",
        "In the year 2050",
        "The chemical formula for",
        "async def process(",
        "What is machine learning",
    ]

    print(f"\n{'='*70}")
    print("TESTING COMPRESSION")
    print("="*70)

    results = test_compression(
        model, tokenizer, compressed_rules,
        test_prompts, (args.start_layer, args.end_layer)
    )

    print(f"\nRESULT: {results['matches']}/{results['total']} ({results['accuracy']*100:.1f}%) exact match")

    print(f"\nDetailed results:")
    for r in results['results']:
        status = "Y" if r['match'] else "X"
        print(f"  {status} '{r['prompt']}' -> orig='{r['original']}', comp='{r['compressed']}'")

    if results['accuracy'] >= 1.0:
        print(f"\n{'='*70}")
        print("SUCCESS! 100% EXACT MATCH")
        print("="*70)
        print(f"Scaling law verified:")
        print(f"  - {n_compressed} layers compressed")
        print(f"  - {len(calibration)} calibration prompts ({prompts_per_layer}/layer)")
        print(f"  - {len(test_prompts)} held-out prompts: 100% exact match")


if __name__ == "__main__":
    main()
