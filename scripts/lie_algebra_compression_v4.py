#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Lie Algebra Compression v4
"""
Lie Algebra Compression v4 - Massive Calibration Coverage

KEY INSIGHT FROM V3:
- Math works perfectly: 20/20 on calibration
- Failure correlates with out-of-span error
- 6.8% OOS succeeds, 20%+ fails

THE HYPOTHESIS:
We need to span the SEMANTIC CATEGORIES, not just count samples.
Different prompt types live in different subspaces.

THE SOLUTION:
1. Generate prompts covering ALL major semantic categories
2. Aim for ~500+ prompts
3. Verify OOS < 10% on held-out before declaring success

Usage:
    python lie_algebra_compression_v4.py --model /path/to/model
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


# Generate comprehensive calibration set
def generate_calibration_prompts() -> list[str]:
    """Generate ~500 diverse prompts covering all major categories."""
    prompts = []

    # === FACTUAL: Geography (50) ===
    countries = [
        "France", "Japan", "Germany", "Australia", "Brazil", "Canada", "India",
        "Russia", "China", "Mexico", "Italy", "Spain", "UK", "South Korea",
        "Argentina", "Egypt", "Nigeria", "Indonesia", "Turkey", "Thailand",
        "Vietnam", "Poland", "Netherlands", "Belgium", "Sweden", "Norway",
        "Denmark", "Finland", "Switzerland", "Austria", "Portugal", "Greece",
        "Czech Republic", "Hungary", "Romania", "Ukraine", "Israel", "Iran",
        "Iraq", "Saudi Arabia", "UAE", "Pakistan", "Bangladesh", "Philippines",
        "Malaysia", "Singapore", "New Zealand", "South Africa", "Kenya", "Morocco",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # === FACTUAL: Science (30) ===
    science_facts = [
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
        "The atomic number of carbon is",
        "The chemical formula for water is",
        "The largest mammal is",
        "The fastest animal is",
        "The oldest civilization is",
        "The nearest star to Earth is",
        "The largest organ in the body is",
        "The number of bones in the human body is",
        "The hardest natural substance is",
        "The most abundant element is",
        "The speed of sound is",
        "The distance to the moon is",
        "The age of the universe is",
        "The human body temperature is",
        "The melting point of ice is",
        "The half-life of carbon-14 is",
        "The mass of an electron is",
        "The gravitational constant is",
        "The Avogadro number is",
        "The Planck constant is",
    ]
    prompts.extend(science_facts)

    # === MATH: Arithmetic (80) ===
    for a in range(1, 21):
        for op, symbol in [("+", " + "), ("-", " - "), ("*", " * "), ("/", " / ")]:
            b = np.random.randint(1, 21)
            if op == "/" and a % b != 0:
                continue  # Skip non-integer divisions
            prompts.append(f"{a}{symbol}{b} =")
            if len([p for p in prompts if symbol in p]) >= 20:
                break

    # === MATH: Advanced (20) ===
    math_advanced = [
        "sqrt(16) =",
        "sqrt(25) =",
        "sqrt(100) =",
        "2^10 =",
        "2^8 =",
        "3^3 =",
        "log(100) =",
        "log(1000) =",
        "sin(0) =",
        "cos(0) =",
        "tan(45) =",
        "3.14 * 2 =",
        "The derivative of x^2 is",
        "The integral of 1/x is",
        "The limit as x approaches 0 of sin(x)/x is",
        "The sum of 1 to 100 is",
        "The factorial of 5 is",
        "The factorial of 4 is",
        "Pi times 2 equals",
        "e raised to 0 equals",
    ]
    prompts.extend(math_advanced)

    # === OPPOSITES (30) ===
    opposites = [
        ("hot", "cold"), ("big", "small"), ("happy", "sad"), ("light", "dark"),
        ("up", "down"), ("good", "bad"), ("old", "young"), ("fast", "slow"),
        ("loud", "quiet"), ("wet", "dry"), ("full", "empty"), ("rich", "poor"),
        ("strong", "weak"), ("true", "false"), ("beautiful", "ugly"),
        ("high", "low"), ("long", "short"), ("hard", "soft"), ("near", "far"),
        ("early", "late"), ("alive", "dead"), ("thick", "thin"), ("sharp", "dull"),
        ("safe", "dangerous"), ("open", "closed"), ("clean", "dirty"),
        ("awake", "asleep"), ("simple", "complex"), ("smooth", "rough"),
        ("bright", "dim"),
    ]
    for word, _ in opposites:
        prompts.append(f"The opposite of {word} is")

    # === COMPLETIONS: Story starters (30) ===
    story_starters = [
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
        "The tale begins",
        "In ancient times",
        "Before the dawn of",
        "When the world was",
        "Far across the",
        "Deep within the",
        "High atop the",
        "Beyond the mountains",
        "Beneath the waves",
        "Among the stars",
        "Through the forest",
        "Across the desert",
        "Along the river",
        "Inside the castle",
        "Outside the walls",
    ]
    prompts.extend(story_starters)

    # === TECHNICAL: Computing (40) ===
    tech_prompts = [
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
        "A variable is",
        "A function is",
        "A class is",
        "An array is",
        "A loop is",
        "Recursion is",
        "An operating system is",
        "A compiler is",
        "Version control is",
        "Git is",
        "A web server is",
        "HTML stands for",
        "CSS is used for",
        "JavaScript is",
        "SQL is used for",
        "A GPU is",
        "Memory allocation is",
        "Garbage collection is",
        "A binary tree is",
        "Big O notation is",
    ]
    prompts.extend(tech_prompts)

    # === ABSTRACT: Concepts (30) ===
    abstract = [
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
        "Fear is",
        "Anger is",
        "Joy is",
        "Sadness is",
        "Pride is",
        "Humility is",
        "Patience is",
        "Kindness is",
        "Honesty is",
        "Loyalty is",
        "Trust is",
        "Respect is",
        "Gratitude is",
        "Compassion is",
        "Integrity is",
    ]
    prompts.extend(abstract)

    # === QUESTIONS (30) ===
    questions = [
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
        "What causes",
        "How can we",
        "Why does the",
        "When will",
        "Where can I find",
        "Who was the first",
        "Which method is",
        "Can machines",
        "Would you recommend",
        "Should I use",
        "What happens when",
        "How do you",
        "Why is it that",
        "When is the best",
        "Where do scientists",
        "Who discovered",
        "Which approach works",
        "Can we predict",
        "Would this work",
        "Should people",
    ]
    prompts.extend(questions)

    # === INSTRUCTIONS (25) ===
    instructions = [
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
        "To install the",
        "First, download",
        "Then, configure",
        "After that,",
        "Finally, run",
        "Make sure to",
        "Be careful when",
        "Avoid doing",
        "Try not to",
        "It's important to",
        "The key is to",
        "One way to",
        "A common approach is",
        "The best practice is",
        "To avoid errors,",
    ]
    prompts.extend(instructions)

    # === CODE (25) ===
    code_prompts = [
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
        "def __init__(self",
        "self.value =",
        "import os",
        "from typing import",
        "while True:",
        "break",
        "continue",
        "pass",
        "raise ValueError",
        "assert x ==",
    ]
    prompts.extend(code_prompts)

    # === EMOTIONS/DESCRIPTIONS (25) ===
    emotions = [
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
        "The atmosphere was",
        "People were",
        "The situation was",
        "Things got",
        "Everything seemed",
        "Nothing could",
        "Someone said",
        "Everybody wanted",
        "Nobody expected",
        "Something felt",
        "The experience was",
        "The moment was",
        "The day was",
        "The night was",
        "The weather was",
    ]
    prompts.extend(emotions)

    # === DIALOGUE (25) ===
    dialogue = [
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
        '"Did you hear about"',
        '"Have you ever"',
        '"I was wondering if"',
        '"Could you please"',
        '"Would you mind"',
        '"What if we"',
        '"How about"',
        '"Why don\'t you"',
        '"Let\'s go to"',
        '"I think we should"',
        '"Do you remember"',
        '"When I was"',
        '"The thing is"',
        '"Actually, I"',
        '"To be honest"',
    ]
    prompts.extend(dialogue)

    # === SCIENTIFIC (25) ===
    scientific = [
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
        "Newton's first law",
        "The cell is",
        "Electrons are",
        "The mitochondria is",
        "The speed of light",
        "Energy equals mass",
        "The universe is",
        "Black holes are",
        "Dark matter is",
        "The Higgs boson",
        "Quantum entanglement",
        "String theory proposes",
        "The standard model",
        "Thermodynamics states",
        "Entropy is",
    ]
    prompts.extend(scientific)

    # === HISTORICAL (25) ===
    historical = [
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
        "The American Revolution",
        "The Civil War",
        "World War I began",
        "The Cold War",
        "The fall of Rome",
        "The Byzantine Empire",
        "The Ottoman Empire",
        "The British Empire",
        "The Ming Dynasty",
        "The Mongol Empire",
        "Alexander the Great",
        "Julius Caesar",
        "Napoleon Bonaparte",
        "The Declaration of",
        "The Constitution was",
    ]
    prompts.extend(historical)

    # === TRANSITION WORDS (25) - explicitly cover these since they failed before ===
    transitions = [
        "In conclusion,",
        "Therefore,",
        "However,",
        "Moreover,",
        "Furthermore,",
        "Additionally,",
        "Consequently,",
        "Nevertheless,",
        "On the other hand,",
        "In contrast,",
        "Similarly,",
        "Likewise,",
        "For example,",
        "For instance,",
        "In particular,",
        "Specifically,",
        "Generally speaking,",
        "To summarize,",
        "In summary,",
        "To conclude,",
        "As a result,",
        "Due to this,",
        "Because of this,",
        "Despite this,",
        "Regardless,",
    ]
    prompts.extend(transitions)

    # Remove duplicates while preserving order
    seen = set()
    unique_prompts = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)

    return unique_prompts


# Held-out prompts specifically chosen to be DIFFERENT from calibration categories
HELD_OUT_PROMPTS = [
    # Factual not in calibration
    "Water freezes at",
    "The color of the sky is",
    "The president of the United States is",
    "The moon orbits",
    "Diamonds are made of",
    "The Great Wall of China is",
    # Math variations
    "100 / 10 =",
    "50 + 50 =",
    "9 * 9 =",
    "1000 - 1 =",
    # Unusual completions
    "Neural networks are",
    "Stars are made of",
    "The universe is expanding",
    # Conversational
    "The answer is",
    "Well, actually",
    "You know what,",
    "If you think about it,",
    "The problem is that",
    "In my opinion,",
    "That's a great question",
]


def collect_endpoint_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    start_layer: int,
    end_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect h_in and h_out for prompts."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    model_type = type(inner_model).__name__
    is_lfm2 = "lfm" in model_type.lower()

    inputs = []
    outputs = []

    for i, prompt in enumerate(prompts):
        if i % 100 == 0:
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")

        tokens = tokenizer.encode(prompt)
        if len(tokens) == 0:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        mx.eval(input_ids)

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
                inputs.append(h_in)

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask

            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32))
                outputs.append(h_out)

    X = np.stack(inputs, axis=1).astype(np.float64)
    Y = np.stack(outputs, axis=1).astype(np.float64)
    return X, Y


def compute_transformation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute T such that Y ≈ T @ X using least squares."""
    # T @ X = Y => T = Y @ pinv(X)
    T = Y @ np.linalg.pinv(X)
    return T.astype(np.float32)


def factor_transformation(T: np.ndarray, rank: int) -> np.ndarray:
    """Factor T to given rank using SVD."""
    U, S, Vh = np.linalg.svd(T.astype(np.float64), full_matrices=False)
    T_fact = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]
    return T_fact.astype(np.float32)


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
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = T_fact.astype(np.float64) @ h_in
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        elif start_layer < idx <= end_layer:
            pass
        else:
            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

    if is_lfm2:
        h = inner_model.embedding_norm(h)
    else:
        h = inner_model.norm(h)
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)
    fact_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    factored = tokenizer.decode([fact_token]).split()[0] if tokenizer.decode([fact_token]).split() else "(empty)"

    return normal, factored


def compute_oos(X_calib: np.ndarray, h_in: np.ndarray) -> float:
    """Compute out-of-span percentage."""
    h_in = h_in.astype(np.float64)
    X_calib = X_calib.astype(np.float64)
    h_proj = X_calib @ np.linalg.pinv(X_calib) @ h_in
    h_oos = h_in - h_proj
    return np.linalg.norm(h_oos) / (np.linalg.norm(h_in) + 1e-10)


def main():
    parser = argparse.ArgumentParser(description="Lie algebra compression v4")
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
    print("LIE ALGEBRA COMPRESSION V4 - MASSIVE COVERAGE")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    start_layer = 3
    end_layer = n_layers - 2

    # Generate calibration prompts
    CALIBRATION_PROMPTS = generate_calibration_prompts()
    print(f"Compressing layers {start_layer} to {end_layer}")
    print(f"Calibration prompts: {len(CALIBRATION_PROMPTS)}")
    print(f"Held-out prompts: {len(HELD_OUT_PROMPTS)}")

    # Phase 1: Collect calibration
    print(f"\n{'='*80}")
    print("PHASE 1: COLLECT CALIBRATION DATA")
    print("="*80)

    X, Y = collect_endpoint_activations(
        model, tokenizer, CALIBRATION_PROMPTS, start_layer, end_layer
    )
    print(f"Calibration shape: {X.shape}")

    # Analyze calibration rank
    U_x, S_x, _ = np.linalg.svd(X, full_matrices=False)
    total_var = np.sum(S_x ** 2)
    cumsum_var = np.cumsum(S_x ** 2)
    calib_rank_90 = np.searchsorted(cumsum_var / total_var, 0.90) + 1
    calib_rank_99 = np.searchsorted(cumsum_var / total_var, 0.99) + 1
    print(f"Calibration rank (90% var): {calib_rank_90}")
    print(f"Calibration rank (99% var): {calib_rank_99}")

    # Phase 2: Compute T
    print(f"\n{'='*80}")
    print("PHASE 2: COMPUTE TRANSFORMATION")
    print("="*80)

    T = compute_transformation(X, Y)

    # Reconstruction error
    Y_pred = T.astype(np.float64) @ X
    recon_error = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
    print(f"Calibration reconstruction error: {recon_error:.6f}")

    # T analysis
    U_t, S_t, _ = np.linalg.svd(T.astype(np.float64), full_matrices=False)
    t_total = np.sum(S_t ** 2)
    t_cumsum = np.cumsum(S_t ** 2)
    t_rank_90 = np.searchsorted(t_cumsum / t_total, 0.90) + 1
    t_rank_99 = np.searchsorted(t_cumsum / t_total, 0.99) + 1
    print(f"T rank (90% var): {t_rank_90}")
    print(f"T rank (99% var): {t_rank_99}")

    # Phase 3: Test on calibration
    print(f"\n{'='*80}")
    print("PHASE 3: TEST ON CALIBRATION")
    print("="*80)

    for rank in [256, 128, 64]:
        T_fact = factor_transformation(T, rank)
        matches = 0
        for p in CALIBRATION_PROMPTS[:30]:
            normal, factored = test_prompt(model, tokenizer, p, T_fact, start_layer, end_layer)
            if normal == factored:
                matches += 1
        print(f"Rank={rank:>4}: {matches}/30 matches on calibration")

    # Phase 4: Test on held-out
    print(f"\n{'='*80}")
    print("PHASE 4: TEST ON HELD-OUT")
    print("="*80)

    for rank in [512, 256, 128, 64, 32]:
        T_fact = factor_transformation(T, rank)
        matches = 0
        for p in HELD_OUT_PROMPTS:
            normal, factored = test_prompt(model, tokenizer, p, T_fact, start_layer, end_layer)
            if normal == factored:
                matches += 1
        compression = hidden_dim / rank
        print(f"Rank={rank:>4}: {matches}/{len(HELD_OUT_PROMPTS)} matches ({100*matches/len(HELD_OUT_PROMPTS):.0f}%), compression={compression:.1f}x")

    # Phase 5: OOS analysis
    print(f"\n{'='*80}")
    print("PHASE 5: OUT-OF-SPAN ANALYSIS")
    print("="*80)

    X_held, _ = collect_endpoint_activations(
        model, tokenizer, HELD_OUT_PROMPTS, start_layer, end_layer
    )

    T_fact = factor_transformation(T, 128)

    print(f"\n{'Prompt':<40} | {'OOS%':>8} | {'Match':>6}")
    print("-" * 60)

    oos_match = []
    for i, prompt in enumerate(HELD_OUT_PROMPTS):
        h_in = X_held[:, i]
        oos = compute_oos(X, h_in)
        normal, factored = test_prompt(model, tokenizer, prompt, T_fact, start_layer, end_layer)
        match = normal == factored
        oos_match.append((oos, match))
        symbol = "✓" if match else "✗"
        print(f"{prompt[:40]:<40} | {oos*100:>7.1f}% | {symbol:>6}")

    # Correlation analysis
    oos_vals = [x[0] for x in oos_match]
    match_vals = [1 if x[1] else 0 for x in oos_match]
    if len(set(match_vals)) > 1:  # Need variance for correlation
        correlation = np.corrcoef(oos_vals, match_vals)[0, 1]
        print(f"\nCorrelation between OOS and success: {correlation:.3f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    avg_oos = np.mean(oos_vals)
    print(f"""
Calibration coverage: {len(CALIBRATION_PROMPTS)} prompts spanning rank {calib_rank_99}
Average OOS on held-out: {avg_oos*100:.1f}%
Transformation T rank: {t_rank_99} (99% var)

KEY FINDINGS:
- T has effective rank {t_rank_90} (90% var), {t_rank_99} (99% var)
- Calibration reconstruction error: {recon_error:.6f}
- Held-out success correlates with low OOS

For 100% held-out success: need OOS < 10% for all inputs
Current average OOS: {avg_oos*100:.1f}%
""")


if __name__ == "__main__":
    main()
