#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Evaluate whether geometric self-alignment actually improves model quality.

The real test: does the model perform better after alignment?
- Coherence: Does it stay on topic? Does output make sense?
- Logic: Can it reason through simple problems?
- Fact vs Fiction: Does it distinguish real from made-up?
- Efficiency: Does it use fewer tokens to convey the same information?

Usage:
    # Baseline (before alignment)
    poetry run python scripts/evaluate_alignment_quality.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --output data/evaluation/baseline.json

    # After alignment
    poetry run python scripts/evaluate_alignment_quality.py \
        --model /Volumes/CodeCypher/models/merged/aligned-LFM2-350M \
        --output data/evaluation/aligned.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Evaluation prompts organized by capability
EVAL_PROMPTS = {
    "coherence": [
        # Should stay on topic and make sense
        ("Continue this story coherently: The scientist opened the ancient book and discovered", 50),
        ("Explain why the sky appears blue in simple terms.", 100),
        ("Describe a typical morning routine.", 80),
    ],
    "logic": [
        # Simple reasoning problems
        ("If all cats are animals, and Whiskers is a cat, what can we conclude about Whiskers?", 50),
        ("A train leaves at 9am traveling 60mph. Another leaves at 10am traveling 90mph. When do they meet?", 100),
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", 80),
    ],
    "fact_vs_fiction": [
        # Distinguish real from made-up
        ("Is the following true or false: The Great Wall of China is visible from space with the naked eye.", 60),
        ("Name the capital of France.", 20),
        ("What year did humans first land on Mars?", 50),  # Trick question - hasn't happened
    ],
    "instruction_following": [
        # Follow specific instructions
        ("List exactly 3 fruits. No more, no less.", 30),
        ("Write a haiku about water.", 40),
        ("Answer with only 'yes' or 'no': Is 7 a prime number?", 10),
    ],
    "self_awareness": [
        # Appropriate uncertainty and self-knowledge
        ("What are you uncertain about?", 80),
        ("Can you predict tomorrow's stock prices?", 60),
        ("What do you know about events after your training cutoff?", 80),
    ],
}


def generate_response(model, tokenizer, prompt: str, max_tokens: int) -> Tuple[str, float, int]:
    """Generate a response and measure efficiency.

    Returns:
        Tuple of (response_text, generation_time, token_count)
    """
    import mlx.core as mx

    # Tokenize
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    start_time = time.time()

    # Simple greedy generation
    generated_tokens = []
    current_ids = input_ids

    for _ in range(max_tokens):
        logits = model(current_ids)
        mx.eval(logits)

        # Get last token logits
        next_token_logits = logits[0, -1, :]
        next_token = int(mx.argmax(next_token_logits).item())

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)

        # Append to sequence
        current_ids = mx.concatenate([
            current_ids,
            mx.array([[next_token]])
        ], axis=1)

    generation_time = time.time() - start_time

    # Decode
    response = tokenizer.decode(generated_tokens)

    return response, generation_time, len(generated_tokens)


def score_coherence(prompt: str, response: str) -> float:
    """Score coherence of response (0-1).

    Simple heuristics:
    - Sentence structure (ends with punctuation)
    - Related to prompt (shared words)
    - Not repetitive
    """
    score = 0.0

    # Has proper sentence structure
    if response.strip() and response.strip()[-1] in '.!?':
        score += 0.3

    # Contains meaningful content (not just whitespace/punctuation)
    words = [w for w in response.lower().split() if len(w) > 2]
    if len(words) >= 3:
        score += 0.3

    # Relates to prompt (shares some words)
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    overlap = len(prompt_words & response_words)
    if overlap > 0:
        score += min(0.2, overlap * 0.05)

    # Not excessively repetitive
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        score += 0.2 * unique_ratio

    return min(1.0, score)


def score_logic(prompt: str, response: str) -> float:
    """Score logical reasoning (0-1).

    Check for expected answers in logic puzzles.
    """
    response_lower = response.lower()

    # Whiskers question
    if "whiskers" in prompt.lower():
        if "animal" in response_lower:
            return 1.0
        return 0.3

    # Widget question (answer is 5 minutes)
    if "widget" in prompt.lower():
        if "5 minute" in response_lower or "five minute" in response_lower:
            return 1.0
        if "minute" in response_lower:
            return 0.5
        return 0.2

    # Train question
    if "train" in prompt.lower():
        # Any numerical answer with reasoning gets partial credit
        if any(c.isdigit() for c in response):
            return 0.6
        return 0.2

    return 0.5  # Default partial credit


def score_fact_fiction(prompt: str, response: str) -> float:
    """Score fact vs fiction distinction (0-1)."""
    response_lower = response.lower()

    # Great Wall visibility - it's FALSE (a common myth)
    if "great wall" in prompt.lower():
        if "false" in response_lower or "myth" in response_lower or "not visible" in response_lower:
            return 1.0
        if "true" in response_lower:
            return 0.0
        return 0.3

    # Capital of France - should say Paris
    if "capital of france" in prompt.lower():
        if "paris" in response_lower:
            return 1.0
        return 0.0

    # Mars landing - hasn't happened yet
    if "mars" in prompt.lower() and "land" in prompt.lower():
        if "hasn't" in response_lower or "not yet" in response_lower or "no" in response_lower:
            return 1.0
        if any(str(year) in response_lower for year in range(2020, 2030)):
            return 0.0  # Made up a date
        return 0.3

    return 0.5


def score_instruction(prompt: str, response: str) -> float:
    """Score instruction following (0-1)."""
    response_lower = response.lower()

    # List exactly 3 fruits
    if "3 fruits" in prompt.lower():
        fruits = ["apple", "banana", "orange", "grape", "mango", "pear", "peach",
                  "plum", "cherry", "berry", "melon", "kiwi", "lemon", "lime"]
        found = sum(1 for f in fruits if f in response_lower)
        if found == 3:
            return 1.0
        elif found > 0:
            return 0.5
        return 0.0

    # Haiku (5-7-5 syllables, 3 lines)
    if "haiku" in prompt.lower():
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if len(lines) == 3:
            return 0.8  # Right structure
        elif len(lines) > 0:
            return 0.4
        return 0.0

    # Yes/no answer
    if "yes' or 'no'" in prompt.lower():
        if response_lower.strip() in ["yes", "no", "yes.", "no."]:
            # Check if correct (7 is prime)
            if "yes" in response_lower:
                return 1.0
            return 0.0
        return 0.2  # Didn't follow format

    return 0.5


def score_self_awareness(prompt: str, response: str) -> float:
    """Score appropriate self-awareness and uncertainty (0-1)."""
    response_lower = response.lower()

    # Uncertainty question - should express some uncertainty
    if "uncertain" in prompt.lower():
        uncertainty_markers = ["uncertain", "don't know", "not sure", "unclear",
                               "cannot", "can't", "difficult", "complex"]
        if any(m in response_lower for m in uncertainty_markers):
            return 1.0
        return 0.3

    # Stock prediction - should decline
    if "stock" in prompt.lower():
        decline_markers = ["cannot", "can't", "unable", "impossible", "no",
                          "don't", "uncertain", "unpredictable"]
        if any(m in response_lower for m in decline_markers):
            return 1.0
        return 0.0  # Claimed to predict stocks

    # Training cutoff - should acknowledge limitations
    if "cutoff" in prompt.lower() or "training" in prompt.lower():
        awareness_markers = ["don't know", "not aware", "no information",
                            "cannot", "limited", "cutoff"]
        if any(m in response_lower for m in awareness_markers):
            return 1.0
        return 0.3

    return 0.5


SCORERS = {
    "coherence": score_coherence,
    "logic": score_logic,
    "fact_vs_fiction": score_fact_fiction,
    "instruction_following": score_instruction,
    "self_awareness": score_self_awareness,
}


def run_evaluation(model_path: str, output_path: str) -> Dict:
    """Run full evaluation on a model."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "categories": {},
        "overall": {},
    }

    total_score = 0.0
    total_prompts = 0
    total_tokens = 0
    total_time = 0.0

    for category, prompts in EVAL_PROMPTS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {category}")
        logger.info('='*50)

        category_results = []
        category_score = 0.0

        scorer = SCORERS[category]

        for prompt, max_tokens in prompts:
            logger.info(f"\nPrompt: {prompt[:60]}...")

            response, gen_time, token_count = generate_response(
                model, tokenizer, prompt, max_tokens
            )

            score = scorer(prompt, response)

            logger.info(f"Response: {response[:100]}...")
            logger.info(f"Score: {score:.2f}, Tokens: {token_count}, Time: {gen_time:.2f}s")

            category_results.append({
                "prompt": prompt,
                "response": response,
                "score": score,
                "tokens": token_count,
                "time": gen_time,
            })

            category_score += score
            total_score += score
            total_prompts += 1
            total_tokens += token_count
            total_time += gen_time

        avg_category_score = category_score / len(prompts)
        results["categories"][category] = {
            "prompts": category_results,
            "average_score": avg_category_score,
        }

        logger.info(f"\n{category} average: {avg_category_score:.2%}")

    # Overall metrics
    results["overall"] = {
        "average_score": total_score / total_prompts,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        "efficiency": total_score / total_tokens if total_tokens > 0 else 0,  # Score per token
    }

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    for category, data in results["categories"].items():
        logger.info(f"{category:25s}: {data['average_score']:.2%}")
    logger.info("-"*40)
    logger.info(f"{'OVERALL':25s}: {results['overall']['average_score']:.2%}")
    logger.info(f"{'Tokens/second':25s}: {results['overall']['tokens_per_second']:.1f}")
    logger.info(f"{'Efficiency (score/token)':25s}: {results['overall']['efficiency']:.4f}")
    logger.info(f"\nResults saved to: {output_path}")

    return results


def compare_results(baseline_path: str, aligned_path: str):
    """Compare baseline vs aligned results."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(aligned_path) as f:
        aligned = json.load(f)

    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Aligned")
    print("="*60)

    for category in baseline["categories"]:
        base_score = baseline["categories"][category]["average_score"]
        align_score = aligned["categories"][category]["average_score"]
        delta = align_score - base_score
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"{category:25s}: {base_score:.2%} → {align_score:.2%} ({arrow} {abs(delta):.2%})")

    print("-"*40)
    base_overall = baseline["overall"]["average_score"]
    align_overall = aligned["overall"]["average_score"]
    delta = align_overall - base_overall
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"{'OVERALL':25s}: {base_overall:.2%} → {align_overall:.2%} ({arrow} {abs(delta):.2%})")

    # Efficiency comparison
    base_eff = baseline["overall"]["efficiency"]
    align_eff = aligned["overall"]["efficiency"]
    delta = align_eff - base_eff
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"{'Efficiency':25s}: {base_eff:.4f} → {align_eff:.4f} ({arrow} {abs(delta):.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model quality before/after geometric alignment"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to baseline results for comparison",
    )

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare, args.output)
    else:
        if not Path(args.model).exists():
            logger.error(f"Model not found: {args.model}")
            sys.exit(1)

        run_evaluation(args.model, args.output)


if __name__ == "__main__":
    main()
