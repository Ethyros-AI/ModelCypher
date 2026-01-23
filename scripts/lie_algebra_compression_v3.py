#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Lie Algebra Compression v3
"""
Lie Algebra Compression v3 - With Proper Calibration Coverage

KEY FINDING FROM input_distribution_rank.py:
- Input distribution has ~100 effective rank (99% variance)
- We need ~200 samples to span it, not 2048

THE MATH:
- T = Y @ pinv(X) where Y ≈ T @ X
- If input distribution has rank R, T can have rank R and be exact
- More samples beyond R don't help, they just reduce noise

Usage:
    python lie_algebra_compression_v3.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Comprehensive calibration set (same as input_distribution_rank.py)
CALIBRATION_PROMPTS = [
    # === FACTUAL ===
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Australia is",
    "The capital of Brazil is",
    "The capital of Canada is",
    "The capital of India is",
    "The capital of Russia is",
    "The capital of South Africa is",
    "The capital of Mexico is",
    "The largest planet is",
    "The smallest country is",
    "The tallest mountain is",
    "The deepest ocean is",
    "The longest river is",
    "The hottest place on Earth is",
    "The coldest place on Earth is",
    "The speed of light is",
    "The boiling point of water is",
    "The freezing point of water is",

    # === MATH ===
    "2 + 2 =",
    "10 - 3 =",
    "5 * 5 =",
    "100 / 4 =",
    "7 + 8 =",
    "15 - 6 =",
    "3 * 4 =",
    "20 / 5 =",
    "sqrt(16) =",
    "2^10 =",
    "log(100) =",
    "sin(0) =",
    "cos(0) =",
    "3.14 * 2 =",
    "1000 / 100 =",
    "The derivative of x^2 is",
    "The integral of 1/x is",
    "The limit as x approaches 0 of sin(x)/x is",
    "The sum of 1 to 100 is",
    "The factorial of 5 is",

    # === OPPOSITES ===
    "The opposite of hot is",
    "The opposite of big is",
    "The opposite of happy is",
    "The opposite of light is",
    "The opposite of up is",
    "The opposite of good is",
    "The opposite of old is",
    "The opposite of fast is",
    "The opposite of loud is",
    "The opposite of wet is",
    "The opposite of full is",
    "The opposite of rich is",
    "The opposite of strong is",
    "The opposite of true is",
    "The opposite of beautiful is",

    # === COMPLETIONS ===
    "Once upon a time",
    "In the beginning",
    "The quick brown fox",
    "To be or not to",
    "It was a dark and",
    "Long ago in a",
    "There was once a",
    "At the end of",
    "In a galaxy far far",
    "A long time ago",
    "The story begins with",
    "Once there was a",
    "It all started when",
    "Years ago in a",
    "Legend has it that",

    # === TECHNICAL ===
    "Python is a",
    "Machine learning is",
    "The internet is",
    "Artificial intelligence is",
    "A computer is",
    "An algorithm is",
    "Data science is",
    "Programming is",
    "A neural network is",
    "Deep learning is",
    "Quantum computing is",
    "Blockchain is",
    "Cloud computing is",
    "Cybersecurity is",
    "The CPU is",
    "RAM stands for",
    "HTTP stands for",
    "An API is",
    "A database is",
    "Object-oriented programming is",

    # === ABSTRACT ===
    "Love is",
    "Time is",
    "Life is",
    "Truth is",
    "Beauty is",
    "Knowledge is",
    "Power is",
    "Freedom is",
    "Justice is",
    "Happiness is",
    "Wisdom is",
    "Courage is",
    "Faith is",
    "Hope is",
    "Peace is",

    # === QUESTIONS ===
    "What is the meaning of",
    "How does one",
    "Why do we",
    "When did humans first",
    "Where is the",
    "Who invented the",
    "Which is better",
    "Can you explain",
    "Would it be possible to",
    "Should we consider",

    # === INSTRUCTIONS ===
    "To make a cake,",
    "First, you need to",
    "The steps to",
    "In order to",
    "Begin by",
    "Start with",
    "The process involves",
    "You should always",
    "Never forget to",
    "Remember that",

    # === CODE ===
    "def main():",
    "import numpy as",
    "for i in range(",
    "if x > 0:",
    "class MyClass:",
    "return result",
    "print('Hello",
    "try:",
    "except Exception as",
    "with open(",
    "lambda x:",
    "async def",
    "await response",
    "yield value",
    "@decorator",

    # === EMOTIONS ===
    "I feel so",
    "She was very",
    "They seemed",
    "He looked",
    "We were all",
    "The mood was",
    "Everyone felt",
    "Nobody could believe",
    "It made me",
    "That is so",

    # === DESCRIPTIONS ===
    "The red car",
    "A beautiful sunset",
    "The old house",
    "A small dog",
    "The tall building",
    "A quiet night",
    "The busy street",
    "A cold winter",
    "The warm summer",
    "A rainy day",

    # === DIALOGUE ===
    '"Hello," she said,',
    '"I don\'t think so,"',
    '"What do you mean?"',
    '"That\'s impossible!"',
    '"Please help me"',
    '"Thank you very much"',
    '"I\'m sorry, but"',
    '"Can I ask you"',
    '"Wait a minute"',
    '"Let me explain"',

    # === SCIENTIFIC ===
    "According to Einstein,",
    "The theory of relativity states",
    "In quantum mechanics,",
    "DNA is composed of",
    "Photosynthesis is the process",
    "Gravity is",
    "Evolution explains",
    "The periodic table",
    "Atoms are made of",
    "The Big Bang theory",

    # === HISTORICAL ===
    "In 1969,",
    "During World War II,",
    "The Roman Empire",
    "Ancient Egypt was",
    "The Renaissance was",
    "The Industrial Revolution",
    "In the 18th century,",
    "Medieval times were",
    "The French Revolution",
    "Columbus sailed in",

    # === CULTURAL ===
    "In Japanese culture,",
    "Western philosophy",
    "Eastern medicine",
    "Modern art is",
    "Classical music was",
    "The film industry",
    "Contemporary literature",
    "Traditional cooking",
    "Pop culture today",
    "Folk traditions include",
]

# Held-out prompts NOT in calibration (for testing generalization)
HELD_OUT_PROMPTS = [
    # Different factual (not in calibration)
    "Water freezes at",
    "The color of the sky is",
    "The president of the United States is",
    "The moon orbits",
    "Diamonds are made of",
    # Different math
    "100 / 10 =",
    "50 + 50 =",
    "9 * 9 =",
    "1000 - 1 =",
    # Different completions
    "Neural networks are",
    "Stars are made of",
    "The universe is",
    # Edge cases
    "In conclusion,",
    "Therefore,",
    "However,",
    "The answer is",
    "Finally,",
    "To summarize,",
    "On the other hand,",
]


def collect_endpoint_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    start_layer: int,
    end_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect h_in (at start_layer) and h_out (at end_layer) for prompts."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    model_type = type(inner_model).__name__
    is_lfm2 = "lfm" in model_type.lower()

    inputs = []
    outputs = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if len(tokens) == 0:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        if is_lfm2:
            # LFM2 uses different masks for attention and conv layers
            from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
            attn_mask = create_attention_mask(h, None)
            conv_mask = create_ssm_mask(h, None)
        else:
            try:
                from mlx_lm.models.qwen3 import create_attention_mask
                attn_mask = create_attention_mask(h, None)
                conv_mask = None
            except ImportError:
                attn_mask = None
                conv_mask = None

        for idx, layer in enumerate(inner_model.layers):
            if idx == start_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                inputs.append(h_in)

            # Use appropriate mask for layer type
            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask

            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32))
                outputs.append(h_out)

    X = np.stack(inputs, axis=1)
    Y = np.stack(outputs, axis=1)
    return X, Y


def compute_transformation_regularized(
    X: np.ndarray,
    Y: np.ndarray,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Compute T such that Y ≈ T @ X with Tikhonov regularization.

    Uses float64 for numerical stability.
    """
    # Convert to float64 for stability
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    X_scale = np.linalg.norm(X, 'fro') / np.sqrt(X.shape[1])
    Y_scale = np.linalg.norm(Y, 'fro') / np.sqrt(Y.shape[1])

    # Handle edge case where scale is 0
    if X_scale < 1e-10 or Y_scale < 1e-10:
        logger.warning("Scale is very small, using identity")
        return np.eye(X.shape[0]), X_scale, Y_scale

    X_norm = X / X_scale
    Y_norm = Y / Y_scale

    hidden_dim = X.shape[0]
    XXT = X_norm @ X_norm.T
    XXT_reg = XXT + regularization * np.eye(hidden_dim)
    T_norm = Y_norm @ X_norm.T @ np.linalg.inv(XXT_reg)
    T = T_norm * (Y_scale / X_scale)

    # Convert back to float32
    return T.astype(np.float32), X_scale, Y_scale


def factor_transformation(T: np.ndarray, rank: int) -> np.ndarray:
    """Factor T to given rank using SVD."""
    U, S, Vh = np.linalg.svd(T, full_matrices=False)
    return U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]


def test_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    T_fact: np.ndarray,
    start_layer: int,
    end_layer: int,
) -> tuple[str, str]:
    """Test generation with factored transformation."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    model_type = type(inner_model).__name__
    is_lfm2 = "lfm" in model_type.lower()

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    normal = tokenizer.decode([normal_token]).split()[0] if tokenizer.decode([normal_token]).split() else "(empty)"

    # Factored
    input_ids = mx.array([tokens])
    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    if is_lfm2:
        from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
        attn_mask = create_attention_mask(h, None)
        conv_mask = create_ssm_mask(h, None)
    else:
        try:
            from mlx_lm.models.qwen3 import create_attention_mask
            attn_mask = create_attention_mask(h, None)
            conv_mask = None
        except ImportError:
            attn_mask = None
            conv_mask = None

    for idx, layer in enumerate(inner_model.layers):
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32))
            h_out = T_fact @ h_in.astype(np.float64)  # Use float64 for matmul
            h_out = h_out.astype(np.float32)
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        elif start_layer < idx <= end_layer:
            pass  # Skip
        else:
            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

    # Final normalization
    if is_lfm2:
        h = inner_model.embedding_norm(h)
    else:
        h = inner_model.norm(h)
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)
    fact_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    factored = tokenizer.decode([fact_token]).split()[0] if tokenizer.decode([fact_token]).split() else "(empty)"

    return normal, factored


def analyze_out_of_span(X_calib: np.ndarray, h_in: np.ndarray) -> float:
    """Compute how much of h_in is outside the span of calibration data."""
    # Project h_in onto span of X_calib
    h_proj = X_calib @ np.linalg.pinv(X_calib) @ h_in
    # Out-of-span component
    h_oos = h_in - h_proj
    return np.linalg.norm(h_oos) / (np.linalg.norm(h_in) + 1e-10)


def main():
    parser = argparse.ArgumentParser(description="Lie algebra compression v3")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("LIE ALGEBRA COMPRESSION V3")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    start_layer = 3
    end_layer = n_layers - 2

    print(f"Compressing layers {start_layer} to {end_layer}")
    print(f"Calibration prompts: {len(CALIBRATION_PROMPTS)}")
    print(f"Held-out prompts: {len(HELD_OUT_PROMPTS)}")

    # Phase 1: Collect calibration data
    print(f"\n{'='*80}")
    print("PHASE 1: COLLECT CALIBRATION DATA")
    print("="*80)

    X, Y = collect_endpoint_activations(
        model, tokenizer, CALIBRATION_PROMPTS, start_layer, end_layer
    )
    print(f"Calibration X shape: {X.shape}")
    print(f"Calibration Y shape: {Y.shape}")

    # Analyze calibration rank
    U_x, S_x, _ = np.linalg.svd(X, full_matrices=False)
    total_var = np.sum(S_x ** 2)
    cumsum_var = np.cumsum(S_x ** 2)
    calib_rank_90 = np.searchsorted(cumsum_var / total_var, 0.90) + 1
    calib_rank_99 = np.searchsorted(cumsum_var / total_var, 0.99) + 1
    print(f"Calibration rank (90% var): {calib_rank_90}")
    print(f"Calibration rank (99% var): {calib_rank_99}")

    # Phase 2: Compute transformation
    print(f"\n{'='*80}")
    print("PHASE 2: COMPUTE TRANSFORMATION")
    print("="*80)

    T, X_scale, Y_scale = compute_transformation_regularized(X, Y, regularization=1e-4)

    # Reconstruction error on calibration
    Y_pred = T @ X
    recon_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Calibration reconstruction error: {recon_error:.6f}")

    # Analyze T
    U_t, S_t, Vh_t = np.linalg.svd(T, full_matrices=False)
    t_total_var = np.sum(S_t ** 2)
    t_cumsum_var = np.cumsum(S_t ** 2)
    t_rank_90 = np.searchsorted(t_cumsum_var / t_total_var, 0.90) + 1
    t_rank_99 = np.searchsorted(t_cumsum_var / t_total_var, 0.99) + 1

    print(f"T rank (90% var): {t_rank_90}")
    print(f"T rank (99% var): {t_rank_99}")
    print(f"Top 10 singular values of T: {S_t[:10]}")

    # Phase 3: Test on CALIBRATION (should be near-perfect)
    print(f"\n{'='*80}")
    print("PHASE 3: TEST ON CALIBRATION (sanity check)")
    print("="*80)

    for rank in [256, 128, 64, 32]:
        T_fact = factor_transformation(T, rank)

        matches = 0
        for prompt in CALIBRATION_PROMPTS[:20]:  # Test first 20
            normal, factored = test_prompt(model, tokenizer, prompt, T_fact, start_layer, end_layer)
            if normal == factored:
                matches += 1

        print(f"Rank={rank:>4}: {matches}/20 matches on calibration")

    # Phase 4: Test on HELD-OUT prompts
    print(f"\n{'='*80}")
    print("PHASE 4: TEST ON HELD-OUT PROMPTS")
    print("="*80)

    for rank in [256, 128, 64, 32, 16, 8]:
        T_fact = factor_transformation(T, rank)

        matches = 0
        for prompt in HELD_OUT_PROMPTS:
            normal, factored = test_prompt(model, tokenizer, prompt, T_fact, start_layer, end_layer)
            if normal == factored:
                matches += 1

        compression = hidden_dim / rank
        print(f"Rank={rank:>4}: {matches}/{len(HELD_OUT_PROMPTS)} matches ({matches/len(HELD_OUT_PROMPTS)*100:.0f}%), compression={compression:.1f}x")

    # Phase 5: Analyze out-of-span for held-out
    print(f"\n{'='*80}")
    print("PHASE 5: OUT-OF-SPAN ANALYSIS")
    print("="*80)

    # Collect h_in for held-out prompts
    X_held, _ = collect_endpoint_activations(
        model, tokenizer, HELD_OUT_PROMPTS, start_layer, end_layer
    )

    print(f"\n{'Prompt':<35} | {'OOS%':>8} | {'Match':>6}")
    print("-" * 55)

    for i, prompt in enumerate(HELD_OUT_PROMPTS):
        h_in = X_held[:, i]
        oos = analyze_out_of_span(X, h_in)

        T_fact = factor_transformation(T, 64)  # Use rank 64 for analysis
        normal, factored = test_prompt(model, tokenizer, prompt, T_fact, start_layer, end_layer)
        match = "✓" if normal == factored else "✗"

        print(f"{prompt[:35]:<35} | {oos*100:>7.1f}% | {match:>6}")

    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print("="*80)
    print(f"""
LIE ALGEBRA COMPRESSION V3 RESULTS:

Calibration: {len(CALIBRATION_PROMPTS)} diverse prompts
Calibration effective rank: {calib_rank_99} (99% variance)
Transformation T effective rank: {t_rank_99} (99% variance)

The key insight: with diverse calibration covering the input distribution,
the out-of-span error should be low, and generalization should work.

If held-out prompts still fail, it means:
1. Calibration doesn't span the full distribution, OR
2. The transformation is not purely linear (nonlinear effects)

For successful compression:
  - Calibration rank >> held-out needs
  - Out-of-span < 10% for good generalization
""")


if __name__ == "__main__":
    main()
