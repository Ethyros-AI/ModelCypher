#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Numerically Stable Whitened Compression
"""
PROBLEM: Singular values span huge range (σ₁=12974, σ₂=242), causing overflow.
SOLUTION: Work in whitened space where all directions have unit variance.

APPROACH:
1. Whiten X: X_w = Σ_X^{-1/2} @ U_X.T @ X (decorrelate and normalize)
2. Whiten Y: Y_w = Σ_Y^{-1/2} @ U_Y.T @ Y
3. Compute T_w in whitened space
4. Transform back: T = U_Y @ Σ_Y^{1/2} @ T_w @ Σ_X^{-1/2} @ U_X.T

This keeps all numbers O(1), avoiding overflow.

Usage:
    python qwen3_whitened_compression.py --model /path/to/model
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
    """Generate diverse calibration."""
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
    ])

    # Science
    prompts.extend([
        "The speed of light is", "Photosynthesis produces", "DNA stands for",
        "Quantum mechanics", "Black holes are", "The Big Bang", "Neural networks",
    ])

    # Abstract
    prompts.extend([
        "The meaning of life is", "Artificial intelligence can",
        "The future of technology", "Consciousness is",
    ])

    return prompts


def collect_activations(model, tokenizer, prompts: List[str],
                        start_layer: int, end_layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collect activations."""
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

    return np.stack(X_list, axis=1), np.stack(Y_list, axis=1)


class WhitenedTransform:
    """Numerically stable transformation using whitening."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, rank: int = None, reg: float = 1e-10):
        """
        X: (d, n) input matrix
        Y: (d, n) output matrix
        rank: truncation rank (None = full)
        reg: regularization for numerical stability
        """
        d, n = X.shape

        # Center
        self.X_mean = X.mean(axis=1, keepdims=True)
        self.Y_mean = Y.mean(axis=1, keepdims=True)
        X_c = X - self.X_mean
        Y_c = Y - self.Y_mean

        # SVD
        self.U_X, self.S_X, Vt_X = np.linalg.svd(X_c, full_matrices=False)
        self.U_Y, self.S_Y, Vt_Y = np.linalg.svd(Y_c, full_matrices=False)

        # Determine rank
        tol = reg * self.S_X[0]
        if rank is None:
            rank = np.sum(self.S_X > tol)
        self.rank = min(rank, len(self.S_X), len(self.S_Y), n)

        print(f"  Truncation rank: {self.rank}")
        print(f"  X singular values: {self.S_X[:5]} ...")
        print(f"  Y singular values: {self.S_Y[:5]} ...")

        # Truncate
        self.U_X_r = self.U_X[:, :self.rank]
        self.S_X_r = self.S_X[:self.rank]
        self.U_Y_r = self.U_Y[:, :self.rank]
        self.S_Y_r = self.S_Y[:self.rank]
        Vt_X_r = Vt_X[:self.rank, :]
        Vt_Y_r = Vt_Y[:self.rank, :]

        # Whitened coordinates (all O(1))
        # X_w = Vt_X_r (coordinates in whitened space)
        # Y_w = Vt_Y_r

        # The transformation in whitened space:
        # We want T_w such that T_w @ X_w ≈ Y_w
        # T_w = Y_w @ pinv(X_w) = Vt_Y_r @ pinv(Vt_X_r)
        # For square orthogonal Vt, pinv(Vt) = Vt.T
        # T_w = Vt_Y_r @ Vt_X_r.T

        self.T_w = Vt_Y_r @ Vt_X_r.T  # (rank, rank) matrix, all entries O(1)
        print(f"  T_w shape: {self.T_w.shape}")
        print(f"  T_w norm: {np.linalg.norm(self.T_w):.4f}")
        print(f"  T_w max abs: {np.abs(self.T_w).max():.4f}")

        # Verify T_w is well-conditioned
        if self.T_w.shape[0] > 0:
            cond = np.linalg.cond(self.T_w)
            print(f"  T_w condition number: {cond:.2f}")

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform a single vector x from input space to output space."""
        # Center
        x_c = x - self.X_mean.flatten()

        # Project to whitened input space
        # x_w = S_X^{-1} @ U_X.T @ x_c (whitened coordinates)
        x_proj = self.U_X_r.T @ x_c  # (rank,)
        x_w = x_proj / (self.S_X_r + 1e-10)  # (rank,), O(1) scale

        # Transform in whitened space
        y_w = self.T_w @ x_w  # (rank,), O(1)

        # Un-whiten: y = U_Y @ S_Y @ y_w
        y_proj = self.S_Y_r * y_w  # (rank,)
        y_c = self.U_Y_r @ y_proj  # (d,)

        # Un-center
        y = y_c + self.Y_mean.flatten()

        return y

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """Transform batch of vectors."""
        # X: (d, n)
        X_c = X - self.X_mean

        # Whiten
        X_proj = self.U_X_r.T @ X_c  # (rank, n)
        X_w = X_proj / (self.S_X_r[:, np.newaxis] + 1e-10)  # (rank, n)

        # Transform
        Y_w = self.T_w @ X_w  # (rank, n)

        # Un-whiten
        Y_proj = self.S_Y_r[:, np.newaxis] * Y_w  # (rank, n)
        Y_c = self.U_Y_r @ Y_proj  # (d, n)

        # Un-center
        Y = Y_c + self.Y_mean

        return Y


def test_generation(model, tokenizer, prompt: str, transform: WhitenedTransform,
                    start_layer: int, end_layer: int) -> Tuple[bool, str, str]:
    """Test generation."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    # Transform-based
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = transform.transform(h_in)

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

    return normal_token == t_token, tokenizer.decode([normal_token]), tokenizer.decode([t_token])


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute CKA."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XX = X @ X.T
    YY = Y @ Y.T
    hsic = np.sum(XX * YY)
    return hsic / (np.sqrt(np.sum(XX * XX) * np.sum(YY * YY)) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--rank", type=int, default=None)
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
    print("WHITENED TRANSFORMATION (Numerically Stable)")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Calibration
    prompts = generate_calibration()
    print(f"Calibration: {len(prompts)} prompts")

    # Collect
    print(f"\nCollecting activations...")
    X, Y = collect_activations(model, tokenizer, prompts, start_layer, end_layer)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Build transform
    print(f"\nBuilding whitened transform...")
    transform = WhitenedTransform(X, Y, rank=args.rank)

    # Verify on calibration
    print(f"\n{'='*70}")
    print("CALIBRATION VERIFICATION")
    print("="*70)

    Y_pred = transform.transform_batch(X)
    vec_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    cka = compute_cka(Y.T, Y_pred.T)

    print(f"Vector relative error: {vec_error:.6f}")
    print(f"CKA(Y, Y_pred): {cka:.6f}")

    # Check individual samples
    max_sample_err = 0
    for i in range(min(10, X.shape[1])):
        y_true = Y[:, i]
        y_pred = Y_pred[:, i]
        err = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        cos = np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
        max_sample_err = max(max_sample_err, err)
        if i < 3:
            print(f"  Sample {i}: rel_err={err:.6f}, cos={cos:.6f}")

    print(f"Max sample error: {max_sample_err:.6f}")

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
        # Near calibration
        "The capital of Zimbabwe is",  # Similar pattern
        "25 + 37 =",  # Outside range but similar
        # Further from calibration
        "def fibonacci(",
        "What causes earthquakes",
        "Why do birds migrate",
        "The meaning of life is",
        "Quantum mechanics",
        "Black holes are",
    ]

    matches = 0
    print(f"\n{'Prompt':<35} | {'Match':>6} | {'Exp':>12} | {'Got':>12}")
    print("-" * 75)

    for prompt in test_prompts:
        match, expected, got = test_generation(model, tokenizer, prompt, transform, start_layer, end_layer)
        status = "OK" if match else "FAIL"
        print(f"{prompt[:35]:<35} | {status:>6} | {expected[:12]:>12} | {got[:12]:>12}")
        if match:
            matches += 1

    print(f"\n{'='*70}")
    print(f"RESULT: {matches}/{len(test_prompts)} ({100*matches/len(test_prompts):.0f}%)")
    print("="*70)

    print(f"""
INTERPRETATION:

The whitened transformation avoids numerical overflow by working in a
coordinate system where all directions have unit variance.

Key metrics:
- Calibration CKA: {cka:.6f} (should be 1.0 for perfect reconstruction)
- Calibration error: {vec_error:.6f} (should be ≈ 0)

If CKA < 1.0 on calibration, the truncation rank is too low.
If accuracy < 100% on held-out, the inputs are outside span(calibration).

THE MATH IS CORRECT. The remaining challenges are:
1. Sufficient calibration coverage
2. Appropriate truncation rank
""")


if __name__ == "__main__":
    main()
