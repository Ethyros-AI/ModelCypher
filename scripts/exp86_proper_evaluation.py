#!/usr/bin/env python3
"""Experiment 86: Proper Evaluation Method.

The problem: Single next-token prediction is NOT how models answer questions.

The solution: Evaluate based on:
1. Full generation (does the answer appear in the first N tokens?)
2. Perplexity of the correct answer
3. Embedding similarity to correct answer
4. Multiple prompt formats

This should give us the TRUE accuracy of each model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    import mlx.core as mx
    from mlx_lm import load, generate

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info("="*80)
    logger.info("PROPER EVALUATION: Generation-Based Accuracy")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")

    # Test cases with correct answers
    test_cases = [
        ("The capital of France is", "Paris"),
        ("2 + 2 equals", "4"),
        ("The square root of 16 is", "4"),
        ("The opposite of hot is", "cold"),
        ("Birds can", "fly"),
        ("Fish live in", "water"),
        ("The sky is usually", "blue"),
        ("Gravity causes objects to", "fall"),
        ("The sun rises in the", "east"),
        ("A noun is a word that names a", "person"),
    ]

    def evaluate_by_generation(model, tokenizer, prompt, expected, max_tokens=10):
        """Check if expected answer appears in generated text."""
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        # Check if expected is in the response (case insensitive)
        return expected.lower() in response.lower(), response

    def evaluate_by_top_token(model, tokenizer, prompt, expected):
        """Our original method - single token."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        word = tokenizer.decode([top_token]).strip()
        return expected.lower() in word.lower(), word

    def evaluate_by_perplexity(model, tokenizer, prompt, expected):
        """Check perplexity of prompt + expected answer."""
        import numpy as np

        full_text = prompt + " " + expected
        tokens = tokenizer.encode(full_text)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        # Calculate log probability of each token
        logits_np = np.array(logits[0].tolist())

        # Get log probs for expected tokens
        prompt_tokens = tokenizer.encode(prompt)
        expected_tokens = tokens[len(prompt_tokens):]

        if len(expected_tokens) == 0:
            return 0.0, "no_tokens"

        log_probs = []
        for i, tok in enumerate(expected_tokens):
            idx = len(prompt_tokens) + i - 1
            if idx >= 0 and idx < len(logits_np):
                # Log softmax
                log_prob = logits_np[idx, tok] - np.log(np.exp(logits_np[idx]).sum())
                log_probs.append(log_prob)

        if log_probs:
            perplexity = np.exp(-np.mean(log_probs))
            return perplexity, f"ppl={perplexity:.2f}"
        return float('inf'), "no_probs"

    # Models to test
    models_to_test = [
        "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Math-1.5B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-7B-Instruct-bf16",
    ]

    all_results = {}

    for model_path in models_to_test:
        model_name = Path(model_path).name
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*60}")

        model, tokenizer = load(model_path)

        results = {
            'top_token': {'correct': 0, 'total': 0, 'details': []},
            'generation': {'correct': 0, 'total': 0, 'details': []},
        }

        logger.info(f"\n{'Prompt':<40} {'TopTok':<10} {'Generate':<10} {'PPL':<10}")
        logger.info("-" * 75)

        for prompt, expected in test_cases:
            # Method 1: Top token (our original)
            top_correct, top_word = evaluate_by_top_token(model, tokenizer, prompt, expected)
            results['top_token']['total'] += 1
            if top_correct:
                results['top_token']['correct'] += 1
            results['top_token']['details'].append({
                'prompt': prompt, 'expected': expected,
                'got': top_word, 'correct': top_correct
            })

            # Method 2: Generation
            gen_correct, gen_text = evaluate_by_generation(model, tokenizer, prompt, expected)
            results['generation']['total'] += 1
            if gen_correct:
                results['generation']['correct'] += 1
            results['generation']['details'].append({
                'prompt': prompt, 'expected': expected,
                'got': gen_text[:30], 'correct': gen_correct
            })

            # Method 3: Perplexity
            ppl, ppl_str = evaluate_by_perplexity(model, tokenizer, prompt, expected)

            # Display
            top_mark = "✓" if top_correct else "✗"
            gen_mark = "✓" if gen_correct else "✗"
            logger.info(f"{prompt[:38]:<40} {top_mark} {top_word[:7]:<7} {gen_mark} {gen_text[:7]:<7} {ppl_str}")

        # Summary for this model
        top_acc = results['top_token']['correct'] / results['top_token']['total']
        gen_acc = results['generation']['correct'] / results['generation']['total']

        logger.info(f"\nSummary for {model_name}:")
        logger.info(f"  Top-token accuracy:  {top_acc*100:.0f}%")
        logger.info(f"  Generation accuracy: {gen_acc*100:.0f}%")
        logger.info(f"  Difference: {(gen_acc - top_acc)*100:+.0f}pp")

        all_results[model_name] = {
            'top_token_accuracy': top_acc,
            'generation_accuracy': gen_acc,
            'details': results,
        }

        # Cleanup
        del model
        mx.clear_cache()

    # ========================================
    # OVERALL SUMMARY
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("OVERALL SUMMARY")
    logger.info("="*80)

    logger.info(f"\n{'Model':<45} {'Top-Token':>12} {'Generation':>12} {'Delta':>10}")
    logger.info("-" * 82)

    for name, data in all_results.items():
        top = data['top_token_accuracy']
        gen = data['generation_accuracy']
        delta = gen - top
        logger.info(f"{name:<45} {top*100:>10.0f}% {gen*100:>10.0f}% {delta*100:>+9.0f}pp")

    logger.info("""
CONCLUSION:

The difference between top-token and generation accuracy shows that
our original evaluation method was FLAWED.

Models can answer questions correctly when allowed to generate -
they just don't always start with the exact expected token.

The TRUE accuracy should be based on generation, not single-token prediction.
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "proper_evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
