#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Massive Calibration: Test if More Coverage Helps
"""
FINDING: Single-layer compression fails with 249 calibration prompts.
Rank = 248, held-out accuracy = 20-60%.

HYPOTHESIS: We need MORE calibration to span the full manifold.
If the manifold is ~1000D, we need ~1000+ calibration points.

This script tests single-layer compression with 2000+ prompts.

Usage:
    python qwen3_massive_calibration.py --model /path/to/model
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


def collect_layer_pair(model, tokenizer, prompts: List[str],
                       layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collect activations before and after a single layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    X_list = []
    Y_list = []

    for i, prompt in enumerate(prompts):
        if (i + 1) % 200 == 0:
            print(f"    Collected {i+1}/{len(prompts)} prompts...")

        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        mask = create_attention_mask(h, None)

        for idx, layer in enumerate(inner_model.layers):
            if idx == layer_idx:
                x = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_list.append(x)

                h = layer(h, mask, None)
                mx.eval(h)

                y = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                Y_list.append(y)
                break
            else:
                h = layer(h, mask, None)
                mx.eval(h)

    return np.stack(X_list, axis=1), np.stack(Y_list, axis=1)


def compute_whitened_transform(X: np.ndarray, Y: np.ndarray) -> dict:
    """Compute numerically stable whitened transform."""
    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)

    X_c = X - X_mean
    Y_c = Y - Y_mean

    U_X, S_X, Vt_X = np.linalg.svd(X_c, full_matrices=False)
    U_Y, S_Y, Vt_Y = np.linalg.svd(Y_c, full_matrices=False)

    tol = 1e-10 * S_X[0]
    rank = np.sum(S_X > tol)
    rank = min(rank, len(S_X), len(S_Y))

    T_w = Vt_Y[:rank, :] @ Vt_X[:rank, :].T

    return {
        'U_X': U_X[:, :rank],
        'S_X': S_X[:rank],
        'U_Y': U_Y[:, :rank],
        'S_Y': S_Y[:rank],
        'T_w': T_w,
        'X_mean': X_mean.flatten(),
        'Y_mean': Y_mean.flatten(),
        'rank': rank
    }


def apply_whitened_transform(x: np.ndarray, transform: dict) -> np.ndarray:
    """Apply whitened transform to a vector."""
    x_c = x - transform['X_mean']
    x_proj = transform['U_X'].T @ x_c
    x_w = x_proj / (transform['S_X'] + 1e-10)
    y_w = transform['T_w'] @ x_w
    y_proj = transform['S_Y'] * y_w
    y_c = transform['U_Y'] @ y_proj
    return y_c + transform['Y_mean']


def test_single_layer(model, tokenizer, prompt: str, transform: dict,
                      layer_idx: int) -> Tuple[bool, str, str]:
    """Test single layer compression."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

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
        if idx == layer_idx:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = apply_whitened_transform(h_in, transform)

            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
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
    t_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    return normal_token == t_token, tokenizer.decode([normal_token]), tokenizer.decode([t_token])


def generate_massive_calibration() -> List[str]:
    """Generate massive calibration set (2000+ prompts)."""
    prompts = []

    # Geography - all countries
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
        "Myanmar", "Cambodia", "Laos", "Bhutan", "Maldives", "Cuba", "Jamaica",
        "Haiti", "Dominican Republic", "Guatemala", "Honduras", "El Salvador",
        "Nicaragua", "Costa Rica", "Panama", "Venezuela", "Ecuador", "Bolivia",
        "Paraguay", "Uruguay", "Zimbabwe", "Zambia", "Tanzania", "Uganda", "Rwanda",
        "Ethiopia", "Ghana", "Senegal", "Mali", "Niger", "Chad", "Sudan", "Libya",
        "Tunisia", "Algeria", "Iraq", "Iran", "Syria", "Lebanon", "Jordan", "Israel",
        "Saudi Arabia", "UAE", "Kuwait", "Qatar", "Bahrain", "Oman", "Yemen",
        "Afghanistan", "Uzbekistan", "Kazakhstan", "Turkmenistan", "Kyrgyzstan",
        "Tajikistan", "Azerbaijan", "Georgia", "Armenia", "Belarus", "Moldova",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")
        prompts.append(f"The population of {c} is")
        prompts.append(f"{c} is known for")

    # Math - extended range
    for a in range(1, 35):
        for b in range(1, 35):
            prompts.append(f"{a} + {b} =")

    # Code - extensive
    code = [
        "def ", "class ", "import ", "from ", "return ", "if ", "else:", "elif ",
        "for ", "while ", "try:", "except ", "finally:", "with ", "yield ", "lambda ",
        "def main():", "def __init__(self", "def test_", "def get_", "def set_",
        "def calculate_", "def compute_", "def process_", "def validate_", "def create_",
        "def fibonacci(", "def factorial(", "def sort(", "def search(", "def parse(",
        "def quicksort(", "def mergesort(", "def binary_search(",
        "class User:", "class Config:", "class Model:", "class Handler:", "class Service:",
        "class Factory:", "class Builder:", "class Singleton:", "class Observer:",
        "import numpy", "import pandas", "import torch", "import tensorflow",
        "import os", "import sys", "import json", "import yaml", "import logging",
        "from typing import", "from collections import", "from pathlib import",
        "from dataclasses import", "from abc import", "from functools import",
        "@property", "@staticmethod", "@classmethod", "@decorator", "@cached_property",
        "async def ", "await ", "async for", "async with",
        "print(", "len(", "range(", "enumerate(", "zip(", "map(", "filter(", "reduce(",
        "list(", "dict(", "set(", "tuple(", "str(", "int(", "float(", "bool(",
        "SELECT * FROM", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE",
        "ALTER TABLE", "DROP TABLE", "JOIN ", "LEFT JOIN", "INNER JOIN",
        "WHERE ", "GROUP BY", "ORDER BY", "HAVING ", "LIMIT ",
        "console.log(", "document.getElementById(", "function ", "const ", "let ", "var ",
        "=>", "async function", "await ", "Promise.", "fetch(",
        "public class", "private void", "public static", "interface ", "extends ",
        "#include", "int main(", "printf(", "std::", "vector<", "map<",
    ]
    prompts.extend(code)

    # Questions - comprehensive
    questions = [
        "What is", "What are", "What was", "What were", "What will", "What would",
        "How do", "How does", "How did", "How can", "How should", "How would", "How might",
        "Why is", "Why are", "Why was", "Why were", "Why do", "Why does", "Why did",
        "When is", "When was", "When will", "When did", "When does", "When would",
        "Where is", "Where are", "Where was", "Where were", "Where do", "Where does",
        "Who is", "Who are", "Who was", "Who were", "Who will", "Who would",
        "Which is", "Which are", "Which was", "Which one", "Which of", "Which type",
        "Can you", "Could you", "Would you", "Should I", "Do I", "Will you", "Did you",
        "What causes", "Why do birds", "How many", "What is the speed of",
        "What is the meaning of", "What is the purpose of", "What is the best",
        "How do I", "How can I", "Where can I", "When should I",
    ]
    prompts.extend(questions)

    # Conversational - extensive
    conv = [
        "Actually,", "However,", "Therefore,", "Furthermore,", "Moreover,",
        "In fact,", "To be honest,", "Honestly,", "Frankly,", "Basically,",
        "Essentially,", "Well,", "So,", "Now,", "Look,", "Listen,", "See,",
        "Nevertheless,", "Nonetheless,", "Consequently,", "Subsequently,",
        "Meanwhile,", "Otherwise,", "Indeed,", "Certainly,", "Obviously,",
        "Interestingly,", "Surprisingly,", "Unfortunately,", "Fortunately,",
        "As I mentioned,", "To put it simply,", "In other words,", "That is to say,",
        "On the other hand,", "At the same time,", "In contrast,", "Similarly,",
        "For example,", "For instance,", "Specifically,", "Generally,",
        "Typically,", "Usually,", "Often,", "Sometimes,", "Rarely,", "Never,",
        "First,", "Second,", "Third,", "Finally,", "Lastly,", "Additionally,",
    ]
    prompts.extend(conv)

    # Science - extensive
    science = [
        "The speed of light is", "The speed of sound is", "Gravity causes",
        "Photosynthesis produces", "DNA stores", "RNA carries", "ATP provides",
        "Hydrogen has atomic number", "Oxygen has atomic number", "Carbon has",
        "Nitrogen has", "Helium has", "Iron has", "Gold has", "Silver has",
        "The boiling point of water", "The melting point of ice", "Absolute zero is",
        "Newton's first law", "Newton's second law", "Newton's third law",
        "Einstein's theory", "Maxwell's equations", "Schrödinger's equation",
        "Quantum mechanics describes", "The Big Bang theory", "Black holes are",
        "Dark matter is", "Dark energy is", "The Higgs boson is", "String theory",
        "Electrons orbit", "Protons and neutrons", "The nucleus contains",
        "Mitochondria are", "Chloroplasts convert", "The cell membrane",
        "Ribosomes synthesize", "The endoplasmic", "The Golgi apparatus",
        "Evolution occurs", "Natural selection", "Genetic mutation",
        "The periodic table", "Chemical bonds", "Ionic compounds",
    ]
    prompts.extend(science)

    # Philosophy/Abstract
    philosophy = [
        "The meaning of life is", "Consciousness is", "Free will is",
        "The nature of reality", "Truth is defined as", "Justice means",
        "Morality is based on", "Ethics requires", "Virtue is",
        "Knowledge is", "Belief differs from", "Certainty requires",
        "Time is", "Space is", "Existence precedes", "Being and nothingness",
        "Descartes said", "Kant argued", "Nietzsche believed", "Plato thought",
        "Aristotle proposed", "Hume argued", "Locke believed", "Hegel's dialectic",
        "Phenomenology studies", "Existentialism holds", "Pragmatism suggests",
    ]
    prompts.extend(philosophy)

    # Technology
    tech = [
        "Artificial intelligence can", "Machine learning uses", "Neural networks are",
        "Deep learning requires", "Natural language processing", "Computer vision",
        "The internet works by", "Computers process", "Algorithms are designed to",
        "Data structures include", "Cloud computing enables", "Cybersecurity protects",
        "Blockchain technology", "Cryptocurrency uses", "Virtual reality creates",
        "Augmented reality overlays", "The CPU executes", "RAM stores", "The GPU renders",
        "APIs allow", "Microservices are", "Containers provide", "Kubernetes manages",
        "DevOps practices", "Agile methodology", "CI/CD pipelines",
    ]
    prompts.extend(tech)

    # Narrative
    narrative = [
        "Once upon a time", "In the beginning", "Long ago", "It was a dark",
        "The hero stood", "She looked at", "He walked into", "They discovered",
        "Suddenly,", "Without warning,", "At that moment,", "Just then,",
        "The story begins", "Our journey starts", "The adventure awaited",
        "The dragon breathed", "The wizard cast", "The knight raised", "The princess",
        "In a galaxy far", "On a distant planet", "The spaceship landed",
        "The detective examined", "The mystery deepened", "The clue revealed",
    ]
    prompts.extend(narrative)

    # Random phrases
    random_phrases = [
        "The recipe calls for", "My favorite color is", "The weather today is",
        "Music can be", "Art represents", "Love is when", "Happiness comes from",
        "Success requires", "Failure teaches", "Learning is a",
        "Books provide", "Movies entertain", "Games offer", "Sports teach",
        "Food is essential", "Water is necessary", "Sleep is important",
    ]
    prompts.extend(random_phrases)

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=15,
                        help="Which layer to test (default: 15)")
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
    print("MASSIVE CALIBRATION TEST")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Testing layer: {args.layer}")

    # Generate massive calibration
    prompts = generate_massive_calibration()
    print(f"Calibration: {len(prompts)} prompts")

    # Held-out test prompts
    held_out = [
        # Completely different patterns
        "The capital of Liechtenstein is",
        "The capital of Andorra is",
        "The capital of San Marino is",
        "99 + 87 =",
        "256 - 128 =",
        "def binary_tree(",
        "class AbstractFactory:",
        "What is the tallest mountain",
        "How does photosynthesis work",
        "Why do leaves change color",
        "String theory proposes",
        "Descartes said",
        "Kant argued that",
        "The dragon breathed",
        "In a galaxy far",
        "The recipe calls for",
        "My favorite color is",
        "The economy showed signs of",
        "During the medieval period",
        "Banana phone",
        "42 is the answer to",
        "Lorem ipsum dolor",
    ]

    # Collect activations
    print(f"\nCollecting activations for layer {args.layer}...")
    X, Y = collect_layer_pair(model, tokenizer, prompts, args.layer)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Compute transform
    print(f"\nComputing whitened transform...")
    transform = compute_whitened_transform(X, Y)
    print(f"Rank: {transform['rank']}")

    # Calibration error
    Y_pred = np.zeros_like(Y)
    for i in range(X.shape[1]):
        Y_pred[:, i] = apply_whitened_transform(X[:, i], transform)

    calib_err = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Calibration error: {calib_err:.6e}")

    # Analyze SVD
    print(f"\nSingular value spectrum:")
    cumvar = np.cumsum(transform['S_X']**2) / np.sum(transform['S_X']**2)
    for pct in [0.90, 0.95, 0.99, 0.999, 0.9999]:
        dim = np.searchsorted(cumvar, pct) + 1
        print(f"  {pct*100:.2f}% variance in {dim} dimensions")

    # Held-out accuracy
    print(f"\n{'='*70}")
    print("HELD-OUT TESTING")
    print("="*70)

    matches = 0
    print(f"\n{'Prompt':<40} | {'Match':>6} | {'Expected':>12} | {'Got':>12}")
    print("-" * 80)

    for prompt in held_out:
        match, expected, got = test_single_layer(model, tokenizer, prompt, transform, args.layer)
        status = "OK" if match else "FAIL"
        print(f"{prompt[:40]:<40} | {status:>6} | {expected[:12]:>12} | {got[:12]:>12}")
        if match:
            matches += 1

    print(f"\n{'='*70}")
    print(f"RESULT: {matches}/{len(held_out)} ({100*matches/len(held_out):.0f}%)")
    print("="*70)

    # Compute residuals for held-out
    print(f"\nHeld-out residuals:")
    for prompt in held_out[:10]:
        x_test, _ = collect_layer_pair(model, tokenizer, [prompt], args.layer)
        x_test = x_test.flatten()

        # Project onto span
        x_c = x_test - transform['X_mean']
        U_r = transform['U_X']
        x_proj = U_r @ (U_r.T @ x_c)
        x_orth = x_c - x_proj
        residual = np.linalg.norm(x_orth) / (np.linalg.norm(x_c) + 1e-10)

        print(f"  '{prompt[:35]}...' residual: {residual*100:.2f}%")

    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print("="*70)
    print(f"""
With {len(prompts)} calibration prompts:
- Rank: {transform['rank']}
- Calibration error: {calib_err:.2e}
- Held-out accuracy: {matches}/{len(held_out)} ({100*matches/len(held_out):.0f}%)

If held-out accuracy increases with more calibration:
  → Coverage is the bottleneck, keep adding prompts

If held-out accuracy plateaus despite more calibration:
  → The transformation is intrinsically nonlinear
  → Linear approximation has a ceiling
""")


if __name__ == "__main__":
    main()
