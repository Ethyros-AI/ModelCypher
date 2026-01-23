#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Self-Supervised Calibration: Let the Model Define Its Own Manifold
"""
KEY INSIGHT: The manifold of coherent activations IS what the model produces.

Instead of hand-crafting prompts by semantic category (which misses directions),
we sample from the model's own distribution. Every activation the model produces
during normal generation is, by definition, ON the manifold.

APPROACH:
1. Start with diverse seed prompts
2. Let the model generate continuations (with temperature for diversity)
3. Collect activations at EVERY position (not just last token)
4. Build calibration from the model's own geometry

This guarantees manifold coverage because:
- The model only produces coherent activations
- Sampling with temperature explores the full distribution
- Every position contributes a valid manifold point

Usage:
    python qwen3_self_supervised_calibration.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import List, Tuple, Dict
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_sequence_activations(model, tokenizer, prompt: str,
                                  start_layer: int, end_layer: int,
                                  max_positions: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Collect activations at ALL positions in a sequence."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.bos_token_id or 1]

    # Limit positions for memory
    tokens = tokens[:max_positions]

    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    mask = create_attention_mask(h, None)

    X_positions = None
    Y_positions = None

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            # Collect ALL positions
            X_positions = np.array(h[0].astype(mx.float32)).astype(np.float64)

        h = layer(h, mask, None)
        mx.eval(h)

        if idx == end_layer:
            Y_positions = np.array(h[0].astype(mx.float32)).astype(np.float64)
            break

    return X_positions, Y_positions  # (seq_len, hidden_dim)


def generate_with_model(model, tokenizer, prompt: str, max_tokens: int = 50,
                        temperature: float = 0.8) -> str:
    """Generate text from prompt using temperature sampling."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    generated = list(tokens)

    for _ in range(max_tokens):
        input_ids = mx.array([generated])
        logits = model(input_ids)
        mx.eval(logits)

        # Get logits for last position
        last_logits = np.array(logits[0, -1, :].astype(mx.float32))

        # Apply temperature
        if temperature > 0:
            scaled_logits = last_logits / temperature
            # Numerical stability
            scaled_logits = scaled_logits - np.max(scaled_logits)
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

            # Sample
            next_token = np.random.choice(len(probs), p=probs)
        else:
            next_token = np.argmax(last_logits)

        generated.append(int(next_token))

        # Stop on EOS
        if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)


def test_transform(model, tokenizer, transform: dict, prompts: List[str],
                   start_layer: int, end_layer: int) -> Tuple[int, int, List[Tuple]]:
    """Test transform on prompts, return (matches, total, details)."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    matches = 0
    total = 0
    details = []

    for prompt in prompts:
        # Normal forward
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

        # Transform-based forward
        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)

                # Apply whitened transform
                x_c = h_in - transform['X_mean']
                x_proj = transform['U_X'].T @ x_c
                x_w = x_proj / (transform['S_X'] + 1e-10)
                y_w = transform['T_w'] @ x_w
                y_proj = transform['S_Y'] * y_w
                y_c = transform['U_Y'] @ y_proj
                h_out = y_c + transform['Y_mean']

                h_np = np.array(h.astype(mx.float32))
                h_np[0, -1, :] = h_out.astype(np.float32)
                h = mx.array(h_np).astype(h.dtype)
                mx.eval(h)
            elif start_layer < idx <= end_layer:
                pass
            else:
                h = layer(h, mask, None)
                mx.eval(h)

        h = inner_model.norm(h)
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(h)
        else:
            logits = inner_model.embed_tokens.as_linear(h)
        mx.eval(logits)
        t_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

        match = (normal_token == t_token)
        if match:
            matches += 1
        total += 1

        details.append((prompt, match, tokenizer.decode([normal_token]), tokenizer.decode([t_token])))

    return matches, total, details


def compute_residual(x: np.ndarray, U: np.ndarray, S: np.ndarray, X_mean: np.ndarray) -> float:
    """Compute orthogonal residual."""
    tol = 1e-10 * S[0] if len(S) > 0 else 1e-10
    rank = np.sum(S > tol)

    x_c = x - X_mean
    U_r = U[:, :rank]
    x_proj = U_r @ (U_r.T @ x_c)
    x_orth = x_c - x_proj

    return np.linalg.norm(x_orth) / (np.linalg.norm(x_c) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-generations", type=int, default=50,
                        help="Number of sequences to generate")
    parser.add_argument("--gen-length", type=int, default=30,
                        help="Tokens to generate per sequence")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more diverse)")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    start_layer = 7
    end_layer = 33

    print(f"\n{'='*70}")
    print("SELF-SUPERVISED CALIBRATION")
    print("Let the Model Define Its Own Manifold")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Generations: {args.num_generations} sequences, {args.gen_length} tokens each")
    print(f"Temperature: {args.temperature}")

    # Diverse seed prompts to start generation from different regions
    seed_prompts = [
        "The capital of France is Paris. The capital of Germany is",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "What is the meaning of consciousness? Many philosophers argue that",
        "In a galaxy far away, a young hero discovered that",
        "To make the perfect omelette, you should first",
        "The speed of light is approximately 299,792,458 meters per second. This",
        "Machine learning models can be trained to recognize patterns in data by",
        "Once upon a time, in a kingdom by the sea, there lived",
        "SELECT * FROM users WHERE age > 18 ORDER BY",
        "The chemical formula for water is H2O, which means each molecule",
        "Why do birds migrate south for winter? Scientists believe",
        "function calculateSum(arr) {\n    let total = 0;\n    for",
        "The year is 2050, and humanity has finally",
        "Einstein's theory of relativity states that",
        "To write clean code, developers should follow",
        "The president announced that the new policy would",
        "In organic chemistry, carbon atoms can form",
        "The stock market today showed unexpected",
        "My favorite memory from childhood is when",
        "The ancient Egyptians built the pyramids using",
        "According to quantum mechanics, particles can",
        "The recipe for chocolate cake requires",
        "During the Renaissance, artists like Leonardo da Vinci",
        "The human brain contains approximately",
        "Climate change is caused by",
    ]

    # Phase 1: Generate sequences
    print(f"\n{'='*70}")
    print("PHASE 1: Generating sequences from model")
    print("="*70)

    generated_sequences = []
    for i, seed in enumerate(seed_prompts[:args.num_generations]):
        print(f"  [{i+1}/{min(args.num_generations, len(seed_prompts))}] Generating from: '{seed[:40]}...'")
        full_text = generate_with_model(model, tokenizer, seed,
                                        max_tokens=args.gen_length,
                                        temperature=args.temperature)
        generated_sequences.append(full_text)

    # If we need more generations, resample from seeds with higher temperature
    while len(generated_sequences) < args.num_generations:
        seed = random.choice(seed_prompts)
        temp = args.temperature * 1.2  # Higher temp for more diversity
        full_text = generate_with_model(model, tokenizer, seed,
                                        max_tokens=args.gen_length,
                                        temperature=temp)
        generated_sequences.append(full_text)
        print(f"  [{len(generated_sequences)}/{args.num_generations}] Additional generation...")

    print(f"\nGenerated {len(generated_sequences)} sequences")

    # Phase 2: Collect activations at ALL positions
    print(f"\n{'='*70}")
    print("PHASE 2: Collecting activations at every position")
    print("="*70)

    X_all = []
    Y_all = []

    for i, sequence in enumerate(generated_sequences):
        X_seq, Y_seq = collect_sequence_activations(
            model, tokenizer, sequence, start_layer, end_layer
        )
        if X_seq is not None and Y_seq is not None:
            X_all.append(X_seq)
            Y_all.append(Y_seq)

        if (i + 1) % 10 == 0:
            total_points = sum(x.shape[0] for x in X_all)
            print(f"  Processed {i+1} sequences, {total_points} total activations")

    # Stack all activations
    X_calib = np.vstack(X_all).T  # (hidden_dim, n_total)
    Y_calib = np.vstack(Y_all).T

    print(f"\nCalibration data: {X_calib.shape[1]} activation vectors in {hidden_dim}D")

    # Phase 3: Build whitened transform
    print(f"\n{'='*70}")
    print("PHASE 3: Building whitened transform")
    print("="*70)

    X_mean = X_calib.mean(axis=1, keepdims=True)
    Y_mean = Y_calib.mean(axis=1, keepdims=True)

    X_c = X_calib - X_mean
    Y_c = Y_calib - Y_mean

    U_X, S_X, Vt_X = np.linalg.svd(X_c, full_matrices=False)
    U_Y, S_Y, Vt_Y = np.linalg.svd(Y_c, full_matrices=False)

    # Compute numerical rank
    tol = 1e-10 * S_X[0]
    rank = np.sum(S_X > tol)
    rank = min(rank, len(S_X), len(S_Y))

    print(f"Numerical rank: {rank}")
    print(f"Top singular values: {S_X[:10]}")

    # Cumulative variance
    cumvar = np.cumsum(S_X**2) / np.sum(S_X**2)
    dim_999 = np.searchsorted(cumvar, 0.999) + 1
    dim_9999 = np.searchsorted(cumvar, 0.9999) + 1
    print(f"99.9% variance in {dim_999} dimensions")
    print(f"99.99% variance in {dim_9999} dimensions")

    # Whitened transform
    T_w = Vt_Y[:rank, :] @ Vt_X[:rank, :].T

    transform = {
        'U_X': U_X[:, :rank],
        'S_X': S_X[:rank],
        'U_Y': U_Y[:, :rank],
        'S_Y': S_Y[:rank],
        'T_w': T_w,
        'X_mean': X_mean.flatten(),
        'Y_mean': Y_mean.flatten(),
        'rank': rank
    }

    print(f"Transform T_w shape: {T_w.shape}")
    print(f"T_w condition number: {np.linalg.cond(T_w):.2f}")

    # Phase 4: Verify on calibration
    print(f"\n{'='*70}")
    print("PHASE 4: Verifying on calibration data")
    print("="*70)

    # Sample some calibration points
    n_verify = min(100, X_calib.shape[1])
    indices = np.random.choice(X_calib.shape[1], n_verify, replace=False)

    max_error = 0
    for idx in indices:
        x = X_calib[:, idx]
        y_true = Y_calib[:, idx]

        # Apply transform
        x_c = x - transform['X_mean']
        x_proj = transform['U_X'].T @ x_c
        x_w = x_proj / (transform['S_X'] + 1e-10)
        y_w = transform['T_w'] @ x_w
        y_proj = transform['S_Y'] * y_w
        y_c = transform['U_Y'] @ y_proj
        y_pred = y_c + transform['Y_mean']

        error = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        max_error = max(max_error, error)

    print(f"Max reconstruction error on calibration: {max_error:.6e}")

    # Compute CKA on calibration subset
    Y_pred = np.zeros_like(Y_calib[:, :n_verify])
    for i, idx in enumerate(indices):
        x = X_calib[:, idx]
        x_c = x - transform['X_mean']
        x_proj = transform['U_X'].T @ x_c
        x_w = x_proj / (transform['S_X'] + 1e-10)
        y_w = transform['T_w'] @ x_w
        y_proj = transform['S_Y'] * y_w
        y_c = transform['U_Y'] @ y_proj
        Y_pred[:, i] = y_c + transform['Y_mean']

    Y_subset = Y_calib[:, indices]

    # CKA computation
    Y_true_c = Y_subset - Y_subset.mean(axis=1, keepdims=True)
    Y_pred_c = Y_pred - Y_pred.mean(axis=1, keepdims=True)
    G_true = Y_true_c.T @ Y_true_c
    G_pred = Y_pred_c.T @ Y_pred_c
    hsic = np.sum(G_true * G_pred)
    cka = hsic / (np.sqrt(np.sum(G_true**2) * np.sum(G_pred**2)) + 1e-10)
    print(f"CKA on calibration subset: {cka:.6f}")

    # Phase 5: Test on held-out prompts
    print(f"\n{'='*70}")
    print("PHASE 5: Testing on held-out prompts")
    print("="*70)

    held_out = [
        # Very different from calibration seeds
        "The capital of Liechtenstein is",
        "The capital of Andorra is",
        "99 + 87 =",
        "256 - 128 =",
        "def quicksort(arr):",
        "class AbstractFactory:",
        "What is the tallest mountain",
        "How does photosynthesis work",
        "Why do leaves change color",
        "The Higgs boson is",
        "Dark matter is",
        "String theory proposes",
        "Descartes said",
        "Kant argued that",
        "The dragon breathed",
        "In a galaxy far",
        "The recipe calls for",
        "My favorite color is",
        "The economy showed signs of",
        "During the medieval period",
        # Completely random
        "Banana phone",
        "42 is the answer to",
        "Lorem ipsum dolor",
        "asdf jkl;",
    ]

    matches, total, details = test_transform(model, tokenizer, transform, held_out,
                                              start_layer, end_layer)

    print(f"\nHeld-out accuracy: {matches}/{total} ({100*matches/total:.0f}%)")

    print(f"\n{'Prompt':<40} | {'Match':>6} | {'Expected':>12} | {'Got':>12} | {'Residual':>8}")
    print("-" * 90)

    for prompt, match, expected, got in details[:15]:
        x_test, _ = collect_sequence_activations(model, tokenizer, prompt, start_layer, end_layer)
        if x_test is not None and len(x_test) > 0:
            residual = compute_residual(x_test[-1], U_X, S_X, X_mean.flatten())
        else:
            residual = 1.0
        status = "OK" if match else "FAIL"
        print(f"{prompt[:40]:<40} | {status:>6} | {expected[:12]:>12} | {got[:12]:>12} | {residual*100:>7.2f}%")

    # Phase 6: Analysis
    print(f"\n{'='*70}")
    print("PHASE 6: Analysis")
    print("="*70)

    # Compute residuals for all held-out
    successes_residual = []
    failures_residual = []

    for prompt, match, _, _ in details:
        x_test, _ = collect_sequence_activations(model, tokenizer, prompt, start_layer, end_layer)
        if x_test is not None and len(x_test) > 0:
            residual = compute_residual(x_test[-1], U_X, S_X, X_mean.flatten())
            if match:
                successes_residual.append(residual)
            else:
                failures_residual.append(residual)

    if successes_residual:
        print(f"\nSuccesses ({len(successes_residual)}): avg residual = {np.mean(successes_residual)*100:.2f}%")
    if failures_residual:
        print(f"Failures ({len(failures_residual)}): avg residual = {np.mean(failures_residual)*100:.2f}%")

    if successes_residual and failures_residual:
        if np.mean(failures_residual) > np.mean(successes_residual):
            print(f"\n>>> CORRELATION CONFIRMED: Failures have higher residual")
        else:
            print(f"\n>>> No clear correlation with residual")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    print(f"""
Self-supervised calibration uses the model's OWN generations to define the manifold.

Results:
- Calibration size: {X_calib.shape[1]} activation vectors
- Numerical rank: {rank}
- 99.99% variance in {dim_9999} dimensions
- Calibration reconstruction error: {max_error:.2e}
- Calibration CKA: {cka:.6f}
- Held-out accuracy: {matches}/{total} ({100*matches/total:.0f}%)

The key insight: every activation the model produces during generation
is BY DEFINITION on the coherent manifold. We don't need to guess
which semantic categories to include - we sample from the model itself.

If held-out accuracy is still < 100%, it means:
1. The held-out prompts produce activations the model wouldn't naturally generate
2. OR we need more diverse seed prompts for generation
3. OR the manifold has more structure than our linear approximation captures
""")


if __name__ == "__main__":
    main()
