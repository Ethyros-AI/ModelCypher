#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Gram Matrix Preserving Transformation
"""
INSIGHT: The relational structure IS the Gram matrix.
Preserving relations = preserving the Gram matrix.

WRONG OBJECTIVE: min ||Y - T @ X||  (vector error)
RIGHT OBJECTIVE: min ||G_Y - T @ G_X @ T.T||  (Gram error)

CLOSED-FORM SOLUTION:
Given SVD: X = U_X @ S_X @ V.T
           Y = U_Y @ S_Y @ V.T

The Gram-preserving transformation is:
T = U_Y @ diag(S_Y / S_X) @ U_X.T

This maps principal directions of X to principal directions of Y,
preserving the relational structure.

Usage:
    python qwen3_gram_preserving.py --model /path/to/model
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


def generate_calibration() -> List[str]:
    """Generate diverse calibration set."""
    prompts = []

    # Geography
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Pakistan", "Thailand", "Vietnam", "Indonesia", "South Korea", "Poland",
        "Sweden", "Norway", "Netherlands", "Switzerland", "Argentina", "Chile",
        "Nigeria", "Kenya", "Morocco", "Peru", "Colombia", "Mongolia", "Nepal",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # Math
    for a in range(1, 20):
        for b in range(1, 20):
            prompts.append(f"{a} + {b} =")

    # Code
    prompts.extend([
        "def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ",
        "def main():", "def __init__(", "class Config:", "import numpy",
        "def fibonacci(", "def factorial(", "SELECT * FROM", "console.log(",
    ])

    # Questions
    prompts.extend([
        "What is", "How do", "Why is", "When was", "Where is", "Who is",
        "Can you", "Could you", "Should I", "Is it", "Are there",
        "What causes", "Why do birds", "How many", "What is the speed",
    ])

    # Conversational
    prompts.extend([
        "Actually,", "However,", "Therefore,", "In fact,", "To be honest,",
        "Basically,", "Essentially,", "Well,", "So,", "Look,",
        "Furthermore,", "Moreover,", "Consequently,", "Honestly,",
    ])

    # Science
    prompts.extend([
        "The speed of light is", "The speed of sound is",
        "Photosynthesis produces", "DNA stands for", "Gravity is",
        "Quantum mechanics", "Black holes are", "The Big Bang",
        "Neural networks", "The atom consists",
    ])

    # Abstract
    prompts.extend([
        "The meaning of life is", "Consciousness is", "Artificial intelligence can",
        "The nature of reality", "The future of technology",
    ])

    return prompts


def collect_activations(model, tokenizer, prompts: List[str],
                        start_layer: int, end_layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collect activations at start and end layers."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    X_list = []
    Y_list = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                x = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_list.append(x)

            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                y = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                Y_list.append(y)
                break

    return np.stack(X_list, axis=1), np.stack(Y_list, axis=1)  # (d, n)


def compute_gram_preserving_T(X: np.ndarray, Y: np.ndarray, rank: int = None) -> np.ndarray:
    """
    Compute T that preserves the Gram matrix.

    X: (d, n) input activations
    Y: (d, n) output activations

    The Gram-preserving T maps principal directions of X to Y.
    """
    d, n = X.shape

    # Center the data (important for Gram matrix interpretation)
    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # SVD of centered data
    U_X, S_X, Vt_X = np.linalg.svd(X_centered, full_matrices=False)
    U_Y, S_Y, Vt_Y = np.linalg.svd(Y_centered, full_matrices=False)

    # Determine effective rank
    tol = 1e-10 * S_X[0]
    if rank is None:
        rank = np.sum(S_X > tol)
    rank = min(rank, len(S_X), len(S_Y))

    print(f"  X singular values (top 10): {S_X[:10]}")
    print(f"  Y singular values (top 10): {S_Y[:10]}")
    print(f"  Effective rank: {rank}")

    # Compute scaling factors (with regularization for small singular values)
    eps = 1e-10
    scaling = np.zeros(rank)
    for i in range(rank):
        if S_X[i] > eps:
            scaling[i] = S_Y[i] / S_X[i]
        else:
            scaling[i] = 0  # Don't scale directions with no variance

    print(f"  Scaling factors (top 10): {scaling[:10]}")

    # Build T = U_Y @ diag(scaling) @ U_X.T
    # Only use top 'rank' components
    T = U_Y[:, :rank] @ np.diag(scaling) @ U_X[:, :rank].T

    # Add mean shift: T @ X_mean = Y_mean approximately
    # Actually, for centered data: T @ (x - X_mean) + Y_mean
    # So: T_full @ x = T @ x + (Y_mean - T @ X_mean)
    bias = Y_mean - T @ X_mean

    return T, bias.flatten()


def test_generation(model, tokenizer, prompt: str, T: np.ndarray, bias: np.ndarray,
                    start_layer: int, end_layer: int) -> Tuple[bool, str, str, float]:
    """Test generation with Gram-preserving T."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_logits = np.array(logits[0, -1, :].astype(mx.float32))
    normal_token = int(np.argmax(normal_logits))

    # T-based
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    h_out_pred = None

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out_pred = T @ h_in + bias

            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out_pred.astype(np.float32)
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
    t_logits = np.array(logits[0, -1, :].astype(mx.float32))
    t_token = int(np.argmax(t_logits))

    # Compute logit cosine similarity
    cos_sim = np.dot(normal_logits, t_logits) / (np.linalg.norm(normal_logits) * np.linalg.norm(t_logits) + 1e-10)

    return normal_token == t_token, tokenizer.decode([normal_token]), tokenizer.decode([t_token]), cos_sim


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between X and Y."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XX = X @ X.T
    YY = Y @ Y.T

    hsic_xy = np.sum(XX * YY)
    hsic_xx = np.sum(XX * XX)
    hsic_yy = np.sum(YY * YY)

    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--rank", type=int, default=None,
                       help="Truncation rank (None = auto)")
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
    print("GRAM MATRIX PRESERVING TRANSFORMATION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Calibration
    prompts = generate_calibration()
    print(f"Calibration: {len(prompts)} prompts")

    # Collect activations
    print(f"\nCollecting activations...")
    X, Y = collect_activations(model, tokenizer, prompts, start_layer, end_layer)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Compute Gram matrices
    G_X = X.T @ X  # (n, n)
    G_Y = Y.T @ Y  # (n, n)
    print(f"\nGram matrix shapes: G_X={G_X.shape}, G_Y={G_Y.shape}")

    # CKA between input and output
    cka_io = compute_cka(X.T, Y.T)
    print(f"CKA(X, Y) = {cka_io:.6f}")

    # Compute Gram-preserving T
    print(f"\nComputing Gram-preserving T...")
    T, bias = compute_gram_preserving_T(X, Y, rank=args.rank)
    print(f"T shape: {T.shape}")

    # Verify Gram preservation on calibration
    print(f"\n{'='*70}")
    print("GRAM PRESERVATION VERIFICATION")
    print("="*70)

    Y_pred = T @ X + bias[:, np.newaxis]
    G_Y_pred = Y_pred.T @ Y_pred

    # Gram matrix similarity
    G_Y_flat = G_Y.flatten()
    G_Y_pred_flat = G_Y_pred.flatten()
    gram_cos = np.dot(G_Y_flat, G_Y_pred_flat) / (np.linalg.norm(G_Y_flat) * np.linalg.norm(G_Y_pred_flat))
    print(f"Gram matrix cosine similarity: {gram_cos:.6f}")

    # CKA between true Y and predicted Y
    cka_pred = compute_cka(Y.T, Y_pred.T)
    print(f"CKA(Y_true, Y_pred) = {cka_pred:.6f}")

    # Vector reconstruction error (for comparison)
    vec_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Vector relative error: {vec_error:.4f}")

    # Test generation
    print(f"\n{'='*70}")
    print("GENERATION TEST")
    print("="*70)

    test_prompts = [
        # In calibration
        "The capital of France is",
        "5 + 7 =",
        "def main():",
        "Actually,",
        # Outside calibration
        "The capital of Zimbabwe is",
        "25 + 37 =",
        "def fibonacci(",
        "What causes earthquakes",
        "Why do birds migrate",
        "The meaning of life is",
        "Quantum mechanics",
        "Black holes are",
    ]

    matches = 0
    print(f"\n{'Prompt':<35} | {'Match':>6} | {'LogitCos':>8} | {'Exp':>10} | {'Got':>10}")
    print("-" * 85)

    for prompt in test_prompts:
        match, expected, got, cos_sim = test_generation(model, tokenizer, prompt, T, bias, start_layer, end_layer)
        status = "OK" if match else "FAIL"
        print(f"{prompt[:35]:<35} | {status:>6} | {cos_sim:>8.4f} | {expected[:10]:>10} | {got[:10]:>10}")
        if match:
            matches += 1

    print(f"\n{'='*70}")
    print(f"RESULT: {matches}/{len(test_prompts)} ({100*matches/len(test_prompts):.0f}%)")
    print("="*70)

    # Compare methods
    print(f"\n{'='*70}")
    print("METHOD COMPARISON")
    print("="*70)

    # Standard lstsq T
    T_lstsq = Y @ np.linalg.pinv(X)
    Y_lstsq = T_lstsq @ X
    cka_lstsq = compute_cka(Y.T, Y_lstsq.T)
    vec_err_lstsq = np.linalg.norm(Y - Y_lstsq) / np.linalg.norm(Y)

    print(f"""
Method          | CKA(Y, Y_pred) | Vector Error
----------------|----------------|-------------
Gram-preserving | {cka_pred:.6f}       | {vec_error:.4f}
Least-squares   | {cka_lstsq:.6f}       | {vec_err_lstsq:.4f}

Insight:
- Least-squares minimizes vector error (point-wise)
- Gram-preserving preserves relational structure (pairwise)

For lossless compression, we care about RELATIONS, not individual values!
""")


if __name__ == "__main__":
    main()
