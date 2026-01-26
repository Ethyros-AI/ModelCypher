#!/usr/bin/env python3
"""Experiment 70: Prime Mechanism Analysis.

Nonsense primes work just as well as semantic primes.
What is actually happening?

Hypotheses:
1. Position matters - being further into context changes predictions
2. Period/sentence structure creates "answer mode"
3. Specific tokens trigger arithmetic circuits
4. Any prefix changes the probability distribution away from "?"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_digit_token(tokenizer, digit_str):
    """Get the actual digit token ID."""
    tokens = tokenizer.encode(digit_str)
    if len(tokens) > 1 and tokens[0] == 1:
        return tokens[1]
    return tokens[0] if tokens else -1


def analyze_prediction(model, tokenizer, prompt, expected):
    """Detailed analysis of a single prediction."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)

    logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    # Get probabilities
    probs = np.exp(logits_np - logits_np.max())
    probs = probs / probs.sum()

    # Top 5 predictions
    top_indices = np.argsort(probs)[::-1][:5]
    top_probs = probs[top_indices]
    top_tokens = [tokenizer.decode([i]).strip() for i in top_indices]

    # Target info
    target_id = get_digit_token(tokenizer, expected)
    target_prob = probs[target_id] if target_id >= 0 else 0.0
    target_rank = int((np.argsort(probs)[::-1] == target_id).nonzero()[0][0]) if target_id >= 0 else -1

    # Question mark token
    q_tokens = tokenizer.encode("?")
    q_id = q_tokens[1] if len(q_tokens) > 1 else q_tokens[0]
    q_prob = probs[q_id]
    q_rank = int((np.argsort(probs)[::-1] == q_id).nonzero()[0][0])

    return {
        "num_tokens": len(tokens),
        "top_5": list(zip(top_tokens, [float(p) for p in top_probs])),
        "target_prob": float(target_prob),
        "target_rank": int(target_rank),
        "question_prob": float(q_prob),
        "question_rank": int(q_rank),
        "predicted": top_tokens[0],
        "correct": expected in top_tokens[0] or top_tokens[0] == expected,
    }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 70: PRIME MECHANISM ANALYSIS")
    logger.info("=" * 60)

    test_problem = "4+1="
    expected = "5"

    results = {}

    # Hypothesis 1: Position/length matters
    logger.info("\n=== HYPOTHESIS 1: POSITION/LENGTH ===")
    logger.info("Does adding more tokens before the problem help?")

    length_primes = {
        "0_tokens": "",
        "1_token": "X",
        "2_tokens": "X Y",
        "3_tokens": "X Y Z",
        "5_tokens": "X Y Z A B",
        "10_tokens": "X Y Z A B C D E F G",
        "period_1": "X.",
        "period_3": "X Y Z.",
    }

    logger.info(f"\n{'Prime':<20} {'Tokens':>8} {'P(5)':>10} {'P(?)':>10} {'Pred':>8}")
    logger.info("-" * 60)

    for name, prime in length_primes.items():
        prompt = f"{prime} {test_problem}" if prime else test_problem
        analysis = analyze_prediction(model, tokenizer, prompt, expected)
        results[f"length_{name}"] = {"prime": prime, **analysis}

        logger.info(f"{name:<20} {analysis['num_tokens']:>8} {analysis['target_prob']:>10.1%} "
                   f"{analysis['question_prob']:>10.1%} {analysis['predicted']:>8}")

    # Hypothesis 2: Period creates "answer mode"
    logger.info("\n=== HYPOTHESIS 2: SENTENCE STRUCTURE ===")
    logger.info("Does punctuation matter?")

    punct_primes = {
        "no_punct": "Hello world",
        "period": "Hello world.",
        "question": "Hello world?",
        "exclaim": "Hello world!",
        "colon": "Hello world:",
        "comma": "Hello world,",
        "just_period": ".",
        "double_period": "..",
    }

    logger.info(f"\n{'Prime':<20} {'P(5)':>10} {'P(?)':>10} {'Pred':>8}")
    logger.info("-" * 45)

    for name, prime in punct_primes.items():
        prompt = f"{prime} {test_problem}"
        analysis = analyze_prediction(model, tokenizer, prompt, expected)
        results[f"punct_{name}"] = {"prime": prime, **analysis}

        logger.info(f"{name:<20} {analysis['target_prob']:>10.1%} "
                   f"{analysis['question_prob']:>10.1%} {analysis['predicted']:>8}")

    # Hypothesis 3: Specific tokens matter
    logger.info("\n=== HYPOTHESIS 3: TOKEN CONTENT ===")
    logger.info("Do math-related tokens help more?")

    content_primes = {
        "random_words": "Purple elephant dancing.",
        "math_words": "Calculate compute solve.",
        "numbers": "One two three four five.",
        "digits": "1 2 3 4 5.",
        "operations": "Plus minus times equals.",
        "equations": "2=2. 3=3.",
        "wrong_math": "2+2=5. 3+3=7.",  # Incorrect math
        "right_math": "2+2=4. 3+3=6.",  # Correct math
    }

    logger.info(f"\n{'Prime':<25} {'P(5)':>10} {'P(?)':>10} {'Pred':>8}")
    logger.info("-" * 55)

    for name, prime in content_primes.items():
        prompt = f"{prime} {test_problem}"
        analysis = analyze_prediction(model, tokenizer, prompt, expected)
        results[f"content_{name}"] = {"prime": prime, **analysis}

        logger.info(f"{name:<25} {analysis['target_prob']:>10.1%} "
                   f"{analysis['question_prob']:>10.1%} {analysis['predicted']:>8}")

    # Hypothesis 4: What gets suppressed?
    logger.info("\n=== HYPOTHESIS 4: SUPPRESSION ANALYSIS ===")
    logger.info("What happens to '?' probability?")

    key_primes = {
        "none": "",
        "nonsense": "Xyz abc qwerty.",
        "semantic": "Adding 1 means the next number.",
    }

    logger.info(f"\n{'Prime':<15} Top 5 predictions")
    logger.info("-" * 70)

    for name, prime in key_primes.items():
        prompt = f"{prime} {test_problem}" if prime else test_problem
        analysis = analyze_prediction(model, tokenizer, prompt, expected)
        results[f"key_{name}"] = {"prime": prime, **analysis}

        top5_str = ", ".join([f"{t}:{p:.1%}" for t, p in analysis["top_5"]])
        logger.info(f"{name:<15} {top5_str}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Check what correlates with success
    successful = [k for k, v in results.items() if v.get("correct", False)]
    failed = [k for k, v in results.items() if not v.get("correct", False)]

    if successful:
        avg_tokens_success = np.mean([results[k]["num_tokens"] for k in successful])
        avg_tokens_fail = np.mean([results[k]["num_tokens"] for k in failed]) if failed else 0

        logger.info(f"\nSuccessful primes: {len(successful)}")
        logger.info(f"Failed primes: {len(failed)}")
        logger.info(f"Avg tokens (success): {avg_tokens_success:.1f}")
        logger.info(f"Avg tokens (fail): {avg_tokens_fail:.1f}")

        # Check question mark suppression
        avg_q_success = np.mean([results[k]["question_prob"] for k in successful])
        avg_q_fail = np.mean([results[k]["question_prob"] for k in failed]) if failed else 0

        logger.info(f"Avg P(?) (success): {avg_q_success:.1%}")
        logger.info(f"Avg P(?) (fail): {avg_q_fail:.1%}")

        if avg_q_fail > avg_q_success * 2:
            logger.info("\n*** KEY FINDING: Primes SUPPRESS '?' predictions ***")
            logger.info("Without prime, '?' dominates. With prime, digits win.")

    # Save results
    output_path = "data/experiments/prime_mechanism_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
