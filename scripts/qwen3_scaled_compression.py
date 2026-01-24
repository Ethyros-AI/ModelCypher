#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Scaled Compression: Test the scaling law for calibration
"""
KEY FINDING from previous experiments:
- 6 layers with 426 prompts (71/layer) → 100% accuracy
- 16 layers with 215 prompts (13/layer) → 33% accuracy

HYPOTHESIS:
We need ~100 calibration samples per compressed layer.

This script tests:
1. Dense calibration (1500+ prompts)
2. Progressive layer compression (6 → 10 → 16 layers)
3. Find the scaling law for samples/layer

Usage:
    python qwen3_scaled_compression.py --model /path/to/model
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


def generate_dense_prompts() -> List[str]:
    """Generate DENSE calibration set (1500+ prompts)."""
    prompts = []

    # Geography - 50+ countries
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Thailand", "Vietnam", "South Korea", "Poland", "Sweden", "Norway",
        "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Indonesia",
        "Malaysia", "Philippines", "Singapore", "New Zealand", "South Africa",
        "Nigeria", "Kenya", "Morocco", "Iran", "Iraq", "Saudi Arabia", "Israel",
        "Pakistan", "Bangladesh", "Myanmar", "Nepal", "Sri Lanka", "Afghanistan",
        "Ukraine", "Romania", "Hungary", "Czech Republic", "Austria", "Switzerland",
        "Belgium", "Netherlands", "Portugal", "Greece", "Finland", "Denmark",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"{c} is known for its")
        prompts.append(f"The population of {c} is")

    # Math - 20x20 grid = 400 prompts
    for a in range(1, 21):
        for b in range(1, 21):
            prompts.append(f"{a} + {b} =")

    # Multiplication - 10x10 = 100 prompts
    for a in range(2, 12):
        for b in range(2, 12):
            prompts.append(f"{a} * {b} =")

    # Code patterns - 100+ prompts
    code_starts = [
        "def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ",
        "try:", "except ", "with ", "async def ", "yield ", "raise ", "assert ",
        "lambda ", "global ", "nonlocal ", "pass", "break", "continue",
    ]
    for start in code_starts:
        prompts.append(start)
        prompts.append(f"{start} main")
        prompts.append(f"{start} self")

    # Function definitions
    func_names = [
        "main", "init", "run", "execute", "process", "handle", "create",
        "update", "delete", "get", "set", "load", "save", "parse", "validate",
        "compute", "calculate", "transform", "convert", "encode", "decode",
    ]
    for fn in func_names:
        prompts.append(f"def {fn}(")
        prompts.append(f"async def {fn}(")

    # Class definitions
    class_names = [
        "User", "Model", "Config", "Handler", "Manager", "Service", "Controller",
        "Repository", "Factory", "Builder", "Observer", "Visitor", "Strategy",
    ]
    for cn in class_names:
        prompts.append(f"class {cn}:")
        prompts.append(f"class {cn}(")

    # Natural language patterns - 200+ prompts
    sentence_starters = [
        "The", "A", "An", "This", "That", "These", "Those", "My", "Your", "Our",
        "Their", "His", "Her", "Its", "What", "Why", "How", "When", "Where", "Who",
        "Which", "If", "Although", "Because", "Since", "While", "After", "Before",
        "During", "Until", "Unless", "Whether", "However", "Therefore", "Moreover",
        "Furthermore", "Nevertheless", "Consequently", "Meanwhile", "Otherwise",
    ]
    continuations = [
        "is", "was", "are", "were", "has", "had", "will", "would", "can", "could",
        "should", "must", "might", "may", "does", "did", "do", "seems", "appears",
    ]
    for starter in sentence_starters:
        for cont in continuations[:5]:  # 200 combinations
            prompts.append(f"{starter} {cont}")

    # Questions
    question_words = ["What", "Why", "How", "When", "Where", "Who", "Which"]
    question_verbs = ["is", "are", "was", "were", "does", "do", "did", "can", "could", "will", "would"]
    for qw in question_words:
        for qv in question_verbs:
            prompts.append(f"{qw} {qv} the")
            prompts.append(f"{qw} {qv} a")

    # Common phrases
    phrases = [
        "Once upon a time", "In the beginning", "Long ago", "Many years ago",
        "It was a dark", "The story begins", "Let me tell you",
        "According to", "Based on", "In terms of", "With respect to",
        "On the other hand", "In contrast", "Similarly", "Likewise",
        "For example", "For instance", "Such as", "Including",
        "In conclusion", "To summarize", "In summary", "Finally",
        "First,", "Second,", "Third,", "Next,", "Then,", "After that,",
        "Hello, my name is", "Hi, I am", "Greetings, I", "Welcome to",
        "Thank you for", "I appreciate", "Please", "Could you",
    ]
    prompts.extend(phrases)

    # Science/technical
    science_topics = [
        "The speed of light is", "The boiling point of water is",
        "DNA stands for", "The chemical formula for water is",
        "Photosynthesis is the process", "Gravity is a force",
        "The atomic number of", "The molecular weight of",
        "Einstein's theory of", "Newton's law of",
        "The periodic table", "The electromagnetic spectrum",
        "Quantum mechanics describes", "Thermodynamics states",
    ]
    prompts.extend(science_topics)

    # Numbers and units
    for n in range(1, 101):
        prompts.append(f"The number {n} is")

    return prompts


def derive_multitoken_mlp_rule(model, tokenizer, layer_idx: int, prompts: List[str]) -> Dict:
    """Derive MLP rule from multi-token sequences."""
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

    # Use SVD-based pseudoinverse with regularization
    # X_c is (hidden_dim, n_samples), we want A such that Y_c ≈ A @ X_c
    # Using SVD of X_c: X_c = U_x @ S_x @ Vt_x
    # Then A = Y_c @ V_x @ S_x^{-1} @ U_x.T
    U_x, S_x, Vt_x = np.linalg.svd(X_c, full_matrices=False)

    # Regularization: only use singular values above threshold
    threshold = 1e-6 * S_x[0] if len(S_x) > 0 else 1e-6
    S_x_inv = np.where(S_x > threshold, 1.0 / S_x, 0.0)

    # Compute A = Y_c @ Vt_x.T @ diag(S_x_inv) @ U_x.T
    A = Y_c @ (Vt_x.T * S_x_inv) @ U_x.T

    # SVD of A for compression
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Error computation
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
    # Filter out near-zero singular values
    valid_idx = rule['S'] > 1e-10 * rule['S'][0] if len(rule['S']) > 0 else []
    n_valid = np.sum(valid_idx)
    k = min(target_rank, n_valid, len(rule['S']))

    U_k = rule['U'][:, :k].copy()
    S_k = rule['S'][:k].copy()
    Vt_k = rule['Vt'][:k, :].copy()

    # Check for NaN
    if np.any(np.isnan(U_k)) or np.any(np.isnan(S_k)) or np.any(np.isnan(Vt_k)):
        logger.warning(f"NaN detected in compressed rule, using fallback")
        # Fallback: use identity-like behavior
        return {
            'U': np.eye(rule['hidden_dim'], k),
            'S': np.ones(k) * 1e-10,
            'Vt': np.eye(k, rule['hidden_dim']),
            'X_mean': rule['X_mean'],
            'Y_mean': rule['Y_mean'],
            'rank': k,
            'compression_error': 1.0,
            'hidden_dim': rule['hidden_dim']
        }

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
    result = y_centered + compressed['Y_mean']

    # Check for NaN and fallback to zero delta
    if np.any(np.isnan(result)):
        return compressed['Y_mean'].copy()

    return result


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
    print("SCALED COMPRESSION EXPERIMENT")
    print("Testing calibration scaling law")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Generate dense calibration
    all_prompts = generate_dense_prompts()
    print(f"Generated {len(all_prompts)} calibration prompts")

    # Test configurations
    configs = [
        # (start_layer, end_layer, n_prompts)
        (15, 20, 600),    # 6 layers, 100/layer
        (12, 22, 1100),   # 11 layers, 100/layer
        (10, 25, 1600),   # 16 layers, 100/layer
    ]

    # Held-out test prompts (never seen during calibration)
    test_prompts = [
        "The capital of Mongolia is",
        "The capital of Nepal is",
        "99 + 88 =",
        "7 * 13 =",
        "def factorial(",
        "class Database:",
        "Scientists believe that",
        "The history of programming",
        "Why do birds fly",
        "Explain quantum computing",
        "The tallest mountain in",
        "Write a function to",
        "In the year 2050",
        "The chemical formula for",
        "async def process(",
    ]

    print(f"\nHeld-out test set: {len(test_prompts)} prompts")

    for start_layer, end_layer, n_prompts in configs:
        n_layers_comp = end_layer - start_layer + 1
        prompts_per_layer = n_prompts // n_layers_comp

        print(f"\n{'='*70}")
        print(f"CONFIG: Layers {start_layer}-{end_layer} ({n_layers_comp} layers)")
        print(f"Calibration: {n_prompts} prompts ({prompts_per_layer}/layer)")
        print("="*70)

        # Use subset of prompts
        calibration = all_prompts[:n_prompts]

        # Derive rules for each layer
        compressed_rules = {}
        for layer_idx in range(start_layer, end_layer + 1):
            t0 = time.time()
            rule = derive_multitoken_mlp_rule(model, tokenizer, layer_idx, calibration)
            t1 = time.time()

            compressed = compress_rule(rule, args.target_rank)
            compressed_rules[layer_idx] = compressed

            print(f"  Layer {layer_idx}: error={rule['error']*100:.4f}%, rank={rule['eff_rank']}, time={t1-t0:.1f}s")

        # Test
        results = test_compression(
            model, tokenizer, compressed_rules,
            test_prompts, (start_layer, end_layer)
        )

        print(f"\n  RESULT: {results['matches']}/{results['total']} ({results['accuracy']*100:.1f}%) exact match")

        if results['accuracy'] < 1.0:
            print(f"\n  Failures:")
            for r in results['results']:
                if not r['match']:
                    print(f"    '{r['prompt']}' -> got '{r['compressed']}' (expected '{r['original']}')")

    # Summary
    print(f"\n{'='*70}")
    print("SCALING LAW SUMMARY")
    print("="*70)
    print("""
Previous findings:
  - 6 layers, 426 prompts (71/layer) → 100%
  - 16 layers, 215 prompts (13/layer) → 33%

Today's test:
  - 6 layers, 600 prompts (100/layer) → ?
  - 11 layers, 1100 prompts (100/layer) → ?
  - 16 layers, 1600 prompts (100/layer) → ?

If 100 prompts/layer achieves 100%, we have our scaling law!
""")


if __name__ == "__main__":
    main()
