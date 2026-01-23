#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Input Distribution Rank Analysis
"""
Input Distribution Rank Analysis

THE QUESTION:
We have 2048-dim hidden state, but does h_in ever span all 2048 dims?

If the input distribution lies in a rank-R subspace (R << 2048), then:
1. We only need R samples to span it
2. T can be rank R and still be exact
3. Out-of-span = inputs outside the true distribution (adversarial/rare)

THE EXPERIMENT:
1. Collect h_in from MANY diverse prompts (1000+)
2. Compute the effective rank of the collected samples
3. If rank << 2048, the compression story changes dramatically

Usage:
    python input_distribution_rank.py --model /path/to/model
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


# Diverse prompt categories to cover distribution
DIVERSE_PROMPTS = [
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

    # === RANDOM/EDGE ===
    "asdfghjkl",
    "123456789",
    "!@#$%^&*()",
    "lorem ipsum dolor",
    "foo bar baz",
    "xyzzy",
    "the the the",
    "a a a a",
    ".",
    "",
    " ",
    "\n",
    "🔥🔥🔥",
    "---",
    "***",
]


def collect_inputs_at_layer(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    target_layer: int,
) -> np.ndarray:
    """
    Collect h_in at the target layer for all prompts.

    Returns:
        X: (hidden_dim, n_samples) - inputs at target layer
    """
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model

    # Detect model type and get appropriate mask function
    model_type = type(inner_model).__name__
    is_lfm2 = "lfm" in model_type.lower()

    if not is_lfm2:
        try:
            from mlx_lm.models.qwen3 import create_attention_mask as qwen_mask
            create_mask = qwen_mask
        except ImportError:
            create_mask = None
    else:
        create_mask = None

    inputs = []

    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")

        tokens = tokenizer.encode(prompt)
        if len(tokens) == 0:
            tokens = [tokenizer.bos_token_id or 1]  # Fallback for empty prompts

        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        # Create appropriate mask
        if is_lfm2:
            # LFM2 expects a boolean causal mask tensor
            seq_len = h.shape[1]
            # Create causal mask: True for valid positions, False for masked
            mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        elif create_mask is not None:
            mask = create_mask(h, None)
        else:
            mask = None

        for idx, layer in enumerate(inner_model.layers):
            if idx == target_layer:
                # Capture input (last position)
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                inputs.append(h_in)
                break

            h = layer(h, mask, None)
            mx.eval(h)

    X = np.stack(inputs, axis=1)  # (hidden_dim, n_samples)
    return X


def analyze_rank(X: np.ndarray) -> dict:
    """Analyze the effective rank of sample matrix X."""
    hidden_dim, n_samples = X.shape

    # Normalize for numerical stability
    X_scale = np.linalg.norm(X, 'fro') / np.sqrt(n_samples)
    X_norm = X / (X_scale + 1e-10)

    # SVD
    U, S, Vh = np.linalg.svd(X_norm, full_matrices=False)

    # Variance explained
    total_var = np.sum(S ** 2)
    cumsum_var = np.cumsum(S ** 2)

    # Effective ranks at different thresholds
    rank_80 = np.searchsorted(cumsum_var / total_var, 0.80) + 1
    rank_90 = np.searchsorted(cumsum_var / total_var, 0.90) + 1
    rank_95 = np.searchsorted(cumsum_var / total_var, 0.95) + 1
    rank_99 = np.searchsorted(cumsum_var / total_var, 0.99) + 1
    rank_999 = np.searchsorted(cumsum_var / total_var, 0.999) + 1

    # Shannon entropy of normalized singular values (effective dimension)
    S_norm = S / (np.sum(S) + 1e-10)
    S_norm = S_norm[S_norm > 1e-10]  # Remove zeros
    shannon_entropy = -np.sum(S_norm * np.log(S_norm))
    effective_dim = np.exp(shannon_entropy)

    # Participation ratio
    participation_ratio = (np.sum(S) ** 2) / (np.sum(S ** 2) + 1e-10)

    return {
        'n_samples': n_samples,
        'hidden_dim': hidden_dim,
        'singular_values': S,
        'rank_80': int(rank_80),
        'rank_90': int(rank_90),
        'rank_95': int(rank_95),
        'rank_99': int(rank_99),
        'rank_999': int(rank_999),
        'shannon_effective_dim': float(effective_dim),
        'participation_ratio': float(participation_ratio),
        'max_possible_rank': min(n_samples, hidden_dim),
    }


def main():
    parser = argparse.ArgumentParser(description="Input distribution rank analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--layer", type=int, default=3, help="Target layer (default: 3)")
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
    print("INPUT DISTRIBUTION RANK ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Target layer: {args.layer}")
    print(f"Number of diverse prompts: {len(DIVERSE_PROMPTS)}")

    # Collect inputs
    print(f"\n{'='*80}")
    print("PHASE 1: COLLECTING ACTIVATIONS")
    print("="*80)

    X = collect_inputs_at_layer(model, tokenizer, DIVERSE_PROMPTS, args.layer)
    print(f"Collected shape: {X.shape}")

    # Analyze rank
    print(f"\n{'='*80}")
    print("PHASE 2: RANK ANALYSIS")
    print("="*80)

    analysis = analyze_rank(X)

    print(f"\nInput Distribution Statistics:")
    print(f"  Samples collected: {analysis['n_samples']}")
    print(f"  Hidden dimension: {analysis['hidden_dim']}")
    print(f"  Max possible rank: {analysis['max_possible_rank']}")
    print(f"\nEffective Rank:")
    print(f"  80% variance: {analysis['rank_80']}")
    print(f"  90% variance: {analysis['rank_90']}")
    print(f"  95% variance: {analysis['rank_95']}")
    print(f"  99% variance: {analysis['rank_99']}")
    print(f"  99.9% variance: {analysis['rank_999']}")
    print(f"\nDimensionality Measures:")
    print(f"  Shannon effective dim: {analysis['shannon_effective_dim']:.1f}")
    print(f"  Participation ratio: {analysis['participation_ratio']:.1f}")

    print(f"\nTop 20 singular values: {analysis['singular_values'][:20]}")

    # Test multiple layers
    print(f"\n{'='*80}")
    print("PHASE 3: RANK BY LAYER")
    print("="*80)

    print(f"\n{'Layer':>6} | {'Rank90':>8} | {'Rank95':>8} | {'Rank99':>8} | {'Shannon':>10} | {'Participation':>12}")
    print("-" * 70)

    for layer_idx in range(0, n_layers, max(1, n_layers // 10)):
        X_layer = collect_inputs_at_layer(model, tokenizer, DIVERSE_PROMPTS[:50], layer_idx)
        stats = analyze_rank(X_layer)
        print(f"{layer_idx:>6} | {stats['rank_90']:>8} | {stats['rank_95']:>8} | {stats['rank_99']:>8} | {stats['shannon_effective_dim']:>10.1f} | {stats['participation_ratio']:>12.1f}")

    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print("="*80)

    actual_rank_99 = analysis['rank_99']
    print(f"""
INPUT DISTRIBUTION ANALYSIS:

With {analysis['n_samples']} diverse prompts:
  - 99% of variance captured by rank {actual_rank_99}
  - Hidden dim is {hidden_dim}
  - Ratio: {actual_rank_99}/{hidden_dim} = {actual_rank_99/hidden_dim:.1%}

IMPLICATIONS FOR LIE ALGEBRA COMPRESSION:

If the input distribution has intrinsic rank R = {actual_rank_99}:
  1. We need at least R calibration samples (not {hidden_dim})
  2. T can be rank R and still capture the transformation exactly
  3. "Out-of-span" inputs are outside the natural distribution

NEXT STEP:
  - Collect {actual_rank_99 * 2} samples to fully span rank-{actual_rank_99} subspace
  - Test Lie algebra compression with this calibration set
  - Verify generalization on diverse held-out prompts
""")


if __name__ == "__main__":
    main()
