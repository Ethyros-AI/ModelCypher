#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Non-contiguous Layer Compression Test
"""
Test compressing MULTIPLE non-contiguous regions simultaneously:
- Layers 7-8 (100% individual and group)
- Layers 14-21 (100% individual and group)

Can we get 100% when compressing both regions?
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import Dict, List, Set
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_calibration_prompts(n: int = 500) -> List[str]:
    """Generate calibration prompts."""
    prompts = []

    for a in range(1, 21):
        for b in range(1, 21):
            prompts.append(f"{a} + {b} =")

    countries = ["France", "Japan", "Germany", "Italy", "Spain", "China", "India",
                 "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt",
                 "UK", "Thailand", "Vietnam", "South Korea", "Poland", "Sweden"]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    for kw in ["def", "class", "import", "return", "if", "for", "while", "try", "except"]:
        prompts.append(f"{kw} ")

    return prompts[:n]


def generate_heldout_prompts() -> List[str]:
    return [
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


def derive_mlp_rule(model, tokenizer, layer_idx: int, prompts: List[str]) -> Dict:
    """Derive MLP linear rule."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]
    layer = inner_model.layers[layer_idx]

    X_list, Y_list = [], []

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
                X_list.append(mlp_in)

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)
                Y_list.append(np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64))
                break
            else:
                h = l(h, mask, None)
                mx.eval(h)

    X = np.stack(X_list, axis=1)
    Y = np.stack(Y_list, axis=1)

    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    A = Y_c @ np.linalg.pinv(X_c)

    return {
        'A': A,
        'X_mean': X_mean.flatten(),
        'Y_mean': Y_mean.flatten(),
        'hidden_dim': hidden_dim
    }


def apply_compressed_mlp(h_normed2: np.ndarray, rule: Dict) -> np.ndarray:
    h_centered = h_normed2 - rule['X_mean']
    y_centered = rule['A'] @ h_centered
    return y_centered + rule['Y_mean']


def test_noncontiguous(model, tokenizer, rules: Dict[int, Dict],
                       compress_layers: Set[int], test_prompts: List[str]) -> Dict:
    """Test with non-contiguous layer compression."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    matches = 0
    results = []

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
            if idx in compress_layers and idx in rules:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)
                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                h_normed2_np = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                mlp_out_last = apply_compressed_mlp(h_normed2_np, rules[idx])

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
            'match': match,
            'original': tokenizer.decode([orig_token]),
            'compressed': tokenizer.decode([comp_token]),
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
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    calibration = generate_calibration_prompts(500)
    heldout = generate_heldout_prompts()

    print(f"\n{'='*70}")
    print("NON-CONTIGUOUS LAYER COMPRESSION TEST")
    print("="*70)
    print(f"Calibration: {len(calibration)} prompts")

    # Test configurations
    configs = [
        ("7-8 only", {7, 8}),
        ("14-21 only", set(range(14, 22))),
        ("7-8 AND 14-21", {7, 8} | set(range(14, 22))),
        ("1-5 AND 14-21", set(range(1, 6)) | set(range(14, 22))),
    ]

    for name, layers in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {name}")
        print(f"Layers: {sorted(layers)}")
        print("="*70)

        # Derive rules
        rules = {}
        for layer_idx in sorted(layers):
            t0 = time.time()
            rules[layer_idx] = derive_mlp_rule(model, tokenizer, layer_idx, calibration)
            print(f"  Layer {layer_idx}: derived ({time.time()-t0:.1f}s)")

        # Test
        results = test_noncontiguous(model, tokenizer, rules, layers, heldout)

        print(f"\n  RESULT: {results['matches']}/{results['total']} ({results['accuracy']*100:.1f}%)")

        if results['accuracy'] < 1.0:
            print(f"\n  Failures:")
            for r in results['results']:
                if not r['match']:
                    print(f"    '{r['prompt'][:40]}' -> got '{r['compressed']}' (expected '{r['original']}')")


if __name__ == "__main__":
    main()
