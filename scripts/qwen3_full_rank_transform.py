#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Full Rank Transform: Keep ALL Dimensions
"""
INSIGHT: The whitened transform projects onto top-k directions.
But semantic information lives in LOW-variance directions!

Dropping 99.9% of variance sounds safe, but it drops the SEMANTICS.

This script uses Tikhonov regularization to:
1. Keep ALL dimensions
2. Dampen numerical instability in low-variance directions
3. Preserve the fine semantic structure

The key equation:
T = Y @ X.T @ (X @ X.T + λI)^{-1}

Where λ is small enough to not distort, large enough to stabilize.

Usage:
    python qwen3_full_rank_transform.py --model /path/to/model
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

    return np.stack(X_list, axis=1), np.stack(Y_list, axis=1)


def test_generation(model, tokenizer, prompt: str, T: np.ndarray, bias: np.ndarray,
                    start_layer: int, end_layer: int) -> Tuple[bool, str, str, float]:
    """Test generation with T."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    # Normal forward
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_logits = np.array(logits[0, -1, :].astype(mx.float32))
    normal_token = int(np.argmax(normal_logits))

    # T-based forward
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = T @ h_in + bias

            # Check for numerical issues
            if np.any(np.isnan(h_out)) or np.any(np.isinf(h_out)):
                return False, tokenizer.decode([normal_token]), "[NaN]", 0.0

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
    t_logits = np.array(logits[0, -1, :].astype(mx.float32))
    t_token = int(np.argmax(t_logits))

    # Logit cosine similarity
    cos_sim = np.dot(normal_logits, t_logits) / (np.linalg.norm(normal_logits) * np.linalg.norm(t_logits) + 1e-10)

    return normal_token == t_token, tokenizer.decode([normal_token]), tokenizer.decode([t_token]), cos_sim


def generate_calibration() -> List[str]:
    """Generate comprehensive calibration set."""
    prompts = []

    # Geography - extensive
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt", "Turkey",
        "Pakistan", "Thailand", "Vietnam", "Indonesia", "South Korea", "Poland",
        "Sweden", "Norway", "Netherlands", "Switzerland", "Argentina", "Chile",
        "Nigeria", "Kenya", "Morocco", "Peru", "Colombia", "Mongolia", "Nepal",
        "Ukraine", "Greece", "Portugal", "Ireland", "Finland", "Denmark", "Austria",
        "Belgium", "Czech Republic", "Hungary", "Romania", "Bulgaria", "Serbia",
        "Croatia", "Slovenia", "Slovakia", "Latvia", "Lithuania", "Estonia",
        "Philippines", "Malaysia", "Singapore", "Taiwan", "Bangladesh", "Sri Lanka",
        "Myanmar", "Cambodia", "Laos", "Nepal", "Bhutan", "Maldives",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # Math - full grid
    for a in range(1, 25):
        for b in range(1, 25):
            prompts.append(f"{a} + {b} =")

    # Code - Python
    code_python = [
        "def ", "class ", "import ", "from ", "return ", "if ", "else:", "elif ",
        "for ", "while ", "try:", "except ", "finally:", "with ", "yield ", "lambda ",
        "def main():", "def __init__(self", "def test_", "def get_", "def set_",
        "def calculate_", "def compute_", "def process_", "def validate_",
        "def fibonacci(", "def factorial(", "def sort(", "def search(",
        "class User:", "class Config:", "class Model:", "class Handler:", "class Service:",
        "import numpy", "import pandas", "import torch", "import os", "import sys",
        "from typing import", "from collections import", "from pathlib import",
        "@property", "@staticmethod", "@classmethod", "async def ", "await ",
        "print(", "len(", "range(", "enumerate(", "zip(", "map(", "filter(",
    ]
    prompts.extend(code_python)

    # Code - other languages
    code_other = [
        "SELECT * FROM", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE",
        "console.log(", "document.getElementById(", "function ", "const ", "let ", "var ",
        "public class", "private void", "public static", "interface ",
        "#include", "int main(", "printf(", "std::",
        "<div>", "<html>", "<script>", "<!DOCTYPE",
    ]
    prompts.extend(code_other)

    # Questions - comprehensive
    questions = [
        "What is", "What are", "What was", "What were", "What will",
        "How do", "How does", "How did", "How can", "How should",
        "Why is", "Why are", "Why was", "Why were", "Why do",
        "When is", "When was", "When will", "When did",
        "Where is", "Where are", "Where was", "Where do",
        "Who is", "Who are", "Who was", "Who will",
        "Which is", "Which are", "Which one",
        "Can you", "Could you", "Would you", "Should I", "Do I",
        "What causes", "Why do birds", "How many", "What is the speed of",
        "What is the meaning of", "What is the purpose of", "What is the best",
    ]
    prompts.extend(questions)

    # Conversational
    conv = [
        "Actually,", "However,", "Therefore,", "Furthermore,", "Moreover,",
        "In fact,", "To be honest,", "Honestly,", "Frankly,", "Basically,",
        "Essentially,", "Well,", "So,", "Now,", "Look,", "Listen,", "See,",
        "Nevertheless,", "Nonetheless,", "Consequently,", "Subsequently,",
        "Meanwhile,", "Otherwise,", "Indeed,", "Certainly,", "Obviously,",
        "Interestingly,", "Surprisingly,", "Unfortunately,", "Fortunately,",
        "As I mentioned,", "To put it simply,", "In other words,",
        "That being said,", "On the other hand,", "At the same time,",
    ]
    prompts.extend(conv)

    # Science
    science = [
        "The speed of light is", "The speed of sound is", "Gravity causes",
        "Photosynthesis produces", "DNA stores", "RNA carries", "ATP provides",
        "Hydrogen has atomic number", "Oxygen has atomic number", "Carbon has",
        "The boiling point of water", "The melting point of ice", "Absolute zero",
        "Newton's first law", "Newton's second law", "Einstein's theory",
        "Quantum mechanics describes", "The Big Bang theory", "Black holes are",
        "Electrons orbit", "Protons and neutrons", "The nucleus contains",
        "Mitochondria are", "Chloroplasts convert", "The cell membrane",
        "The Higgs boson is", "Dark matter is", "Dark energy",
    ]
    prompts.extend(science)

    # Philosophy/Abstract
    philosophy = [
        "The meaning of life is", "Consciousness is", "Free will is",
        "The nature of reality", "Truth is defined as", "Justice means",
        "Morality is based on", "Ethics requires", "Virtue is",
        "Knowledge is", "Belief differs from", "Certainty requires",
        "Time is", "Space is", "Existence precedes",
        "Descartes said", "Kant argued", "Nietzsche believed",
    ]
    prompts.extend(philosophy)

    # Technology
    tech = [
        "Artificial intelligence can", "Machine learning uses", "Neural networks are",
        "The internet works by", "Computers process", "Algorithms are designed to",
        "Cloud computing enables", "Cybersecurity protects",
        "Blockchain technology", "Cryptocurrency uses", "Virtual reality creates",
        "The CPU executes", "RAM stores", "The GPU renders",
    ]
    prompts.extend(tech)

    # Narrative
    narrative = [
        "Once upon a time", "In the beginning", "Long ago", "It was a dark",
        "The hero stood", "She looked at", "He walked into", "They discovered",
        "Suddenly,", "Without warning,", "At that moment,", "Just then,",
        "The story begins", "Our journey starts",
        "The dragon breathed", "The wizard cast", "The knight raised",
    ]
    prompts.extend(narrative)

    # Instructions
    instructions = [
        "First,", "Then,", "Next,", "After that,", "Finally,",
        "Step 1:", "Step 2:", "Step 3:",
        "To begin,", "To start,", "Begin by", "Start with", "Start by",
        "Make sure to", "Remember to", "Don't forget to", "Be sure to",
    ]
    prompts.extend(instructions)

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--reg", type=float, default=1e-6,
                        help="Regularization parameter λ")
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
    print("FULL RANK TRANSFORM WITH TIKHONOV REGULARIZATION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Regularization λ = {args.reg}")

    # Generate calibration
    prompts = generate_calibration()
    print(f"\nCalibration: {len(prompts)} prompts")

    # Collect activations
    print(f"\nCollecting activations...")
    X, Y = collect_activations(model, tokenizer, prompts, start_layer, end_layer)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Analyze input manifold
    print(f"\n{'='*70}")
    print("INPUT MANIFOLD ANALYSIS")
    print("="*70)

    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    # SVD of input
    U_X, S_X, Vt_X = np.linalg.svd(X_c, full_matrices=False)
    print(f"\nSingular values (top 20):")
    cumvar_X = np.cumsum(S_X**2) / np.sum(S_X**2)
    for i in range(min(20, len(S_X))):
        print(f"  σ_{i+1}: {S_X[i]:.2f} (cumulative: {cumvar_X[i]*100:.4f}%)")

    # Find effective rank
    dim_999 = np.searchsorted(cumvar_X, 0.999) + 1
    dim_9999 = np.searchsorted(cumvar_X, 0.9999) + 1
    print(f"\n99.9% variance in {dim_999} dimensions")
    print(f"99.99% variance in {dim_9999} dimensions")

    # Compute Tikhonov-regularized transform
    # T = Y @ X.T @ (X @ X.T + λI)^{-1}
    print(f"\n{'='*70}")
    print("COMPUTING TIKHONOV-REGULARIZED TRANSFORM")
    print("="*70)

    # Work in centered coordinates
    XXT = X_c @ X_c.T  # (d, d)
    XXT_reg = XXT + args.reg * np.eye(hidden_dim)

    # Check condition number
    cond_XXT = np.linalg.cond(XXT)
    cond_reg = np.linalg.cond(XXT_reg)
    print(f"Condition number of X @ X.T: {cond_XXT:.2e}")
    print(f"Condition number after regularization: {cond_reg:.2e}")

    # Solve for T using Cholesky (more stable than explicit inverse)
    try:
        L = np.linalg.cholesky(XXT_reg)
        # Solve L @ L.T @ T.T = X_c @ Y_c.T
        # First solve L @ Z = X_c @ Y_c.T
        Z = np.linalg.solve(L, Y_c @ X_c.T)
        # Then solve L.T @ T.T = Z, so T = (L^{-T} @ Z)^T
        T_centered = np.linalg.solve(L.T, Z).T

        print(f"Computed T via Cholesky decomposition")
    except np.linalg.LinAlgError:
        print("Cholesky failed, using pseudoinverse")
        T_centered = Y_c @ X_c.T @ np.linalg.pinv(XXT_reg)

    print(f"T shape: {T_centered.shape}")

    # The full transform is: T @ (x - X_mean) + Y_mean
    # Which equals: T @ x + (Y_mean - T @ X_mean)
    bias = Y_mean.flatten() - T_centered @ X_mean.flatten()

    # Verify on calibration
    print(f"\n{'='*70}")
    print("CALIBRATION VERIFICATION")
    print("="*70)

    Y_pred = T_centered @ X_c + Y_mean
    vec_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Vector relative error: {vec_error:.6e}")

    # Per-sample error
    max_err = 0
    for i in range(min(20, X.shape[1])):
        y_true = Y[:, i]
        y_pred = Y_pred[:, i]
        err = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        max_err = max(max_err, err)
    print(f"Max per-sample error: {max_err:.6e}")

    # CKA
    Y_pred_c = Y_pred - Y_pred.mean(axis=1, keepdims=True)
    G_true = Y_c.T @ Y_c
    G_pred = Y_pred_c.T @ Y_pred_c
    hsic = np.sum(G_true * G_pred)
    cka = hsic / (np.sqrt(np.sum(G_true**2) * np.sum(G_pred**2)) + 1e-10)
    print(f"CKA: {cka:.6f}")

    # Test on held-out prompts
    print(f"\n{'='*70}")
    print("HELD-OUT TESTING")
    print("="*70)

    held_out = [
        # Very different
        "The capital of Liechtenstein is",
        "The capital of Andorra is",
        "The capital of San Marino is",
        "99 + 87 =",
        "256 - 128 =",
        "def quicksort(arr):",
        "class AbstractFactory:",
        # Science
        "What is the tallest mountain",
        "How does photosynthesis work",
        "Why do leaves change color",
        "String theory proposes",
        # Philosophy
        "Descartes said",
        "Kant argued that",
        # Narrative
        "The dragon breathed",
        "In a galaxy far",
        # Random
        "The recipe calls for",
        "My favorite color is",
        "The economy showed signs of",
        "During the medieval period",
        # Completely random
        "Banana phone",
        "42 is the answer to",
    ]

    matches = 0
    print(f"\n{'Prompt':<40} | {'Match':>6} | {'CosSim':>8} | {'Expected':>12} | {'Got':>12}")
    print("-" * 95)

    for prompt in held_out:
        match, expected, got, cos_sim = test_generation(
            model, tokenizer, prompt, T_centered, bias, start_layer, end_layer
        )
        status = "OK" if match else "FAIL"
        print(f"{prompt[:40]:<40} | {status:>6} | {cos_sim:>8.4f} | {expected[:12]:>12} | {got[:12]:>12}")
        if match:
            matches += 1

    print(f"\n{'='*70}")
    print(f"RESULT: {matches}/{len(held_out)} ({100*matches/len(held_out):.0f}%)")
    print("="*70)

    # Try different regularization values
    print(f"\n{'='*70}")
    print("REGULARIZATION SENSITIVITY ANALYSIS")
    print("="*70)

    for reg in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
        XXT_reg = XXT + reg * np.eye(hidden_dim)
        try:
            L = np.linalg.cholesky(XXT_reg)
            Z = np.linalg.solve(L, Y_c @ X_c.T)
            T_test = np.linalg.solve(L.T, Z).T
        except:
            T_test = Y_c @ X_c.T @ np.linalg.pinv(XXT_reg)

        bias_test = Y_mean.flatten() - T_test @ X_mean.flatten()

        # Quick test on 5 prompts
        test_matches = 0
        for prompt in held_out[:5]:
            match, _, _, _ = test_generation(
                model, tokenizer, prompt, T_test, bias_test, start_layer, end_layer
            )
            if match:
                test_matches += 1

        print(f"  λ = {reg:.0e}: {test_matches}/5 matches on first 5 prompts")

    # Conclusion
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("="*70)
    print(f"""
The full-rank Tikhonov-regularized transform:
T = Y @ X.T @ (X @ X.T + λI)^{{-1}}

Results:
- Calibration error: {vec_error:.2e}
- Calibration CKA: {cka:.6f}
- Held-out accuracy: {matches}/{len(held_out)} ({100*matches/len(held_out):.0f}%)

The regularization prevents numerical instability without truncating dimensions.
However, accuracy is still limited because:

1. The transformation is NONLINEAR (layers 7-33 include attention + GELU)
2. A linear approximation can only capture linear structure
3. Even perfect reconstruction of the linear component misses nonlinear effects

The closed-form T = Y @ pinv(X) is mathematically optimal for LINEAR approximation.
The remaining error is due to INTRINSIC NONLINEARITY, not numerical issues.
""")


if __name__ == "__main__":
    main()
