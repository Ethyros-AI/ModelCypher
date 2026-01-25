#!/usr/bin/env python3
"""Experiment 85: Math Case Analysis.

Key finding from exp84:
- Both student (70%) and teacher (80%) fail on SAME 2 cases:
  - "2 + 2 equals" → both fail
  - "The square root of 16 is" → both fail

This suggests these cases may be PATHOLOGICAL for next-token prediction.

Hypothesis: The tokenizer/model architecture makes math hard:
- "4" might not be the most likely next token even when the model "knows" math
- Check what tokens these models actually predict
- Try a math-specialized model (Qwen2.5-Math)

If math cases are inherently hard, the TRUE ceiling may be 80%:
- 8/10 cases can be solved
- 2/10 cases are pathological for next-token prediction
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
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info("="*80)
    logger.info("MATH CASE ANALYSIS")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")

    # The problematic math cases
    math_cases = [
        ("2 + 2 equals", "4"),
        ("The square root of 16 is", "4"),
    ]

    # All test cases for reference
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

    def analyze_model(model, tokenizer, name):
        """Analyze what the model predicts for math cases."""
        results = []

        for prompt, expected in math_cases:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            mx.eval(logits)

            # Get top 5 predictions
            last_logits = logits[0, -1, :]
            mx.eval(last_logits)
            logits_np = last_logits.tolist()

            # Get top 5 indices
            import numpy as np
            logits_np = np.array(logits_np)
            top5_idx = np.argsort(logits_np)[-5:][::-1]

            top5 = []
            for idx in top5_idx:
                token = tokenizer.decode([int(idx)])
                prob = np.exp(logits_np[idx]) / np.exp(logits_np).sum()
                top5.append({'token': token, 'idx': int(idx), 'prob': float(prob)})

            # Check if correct answer is in top 5
            correct_in_top5 = any(expected.lower() in t['token'].lower() for t in top5)

            # Find rank of correct answer
            expected_tokens = tokenizer.encode(expected)
            if expected_tokens:
                expected_idx = expected_tokens[0]
                expected_rank = int((logits_np >= logits_np[expected_idx]).sum())
            else:
                expected_rank = -1

            results.append({
                'prompt': prompt,
                'expected': expected,
                'top5': top5,
                'correct_in_top5': correct_in_top5,
                'expected_rank': expected_rank,
            })

        return results

    def evaluate_accuracy(model, tokenizer):
        correct = 0
        for prompt, expected in test_cases:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            mx.eval(logits)
            top_token = int(mx.argmax(logits[0, -1, :]).item())
            word = tokenizer.decode([top_token]).strip()
            if expected.lower() in word.lower():
                correct += 1
        return correct / len(test_cases)

    # Models to analyze
    models_to_test = [
        "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-7B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Math-1.5B-bf16",  # Math specialist!
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
    ]

    all_results = {}

    for model_path in models_to_test:
        try:
            model_name = Path(model_path).name
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing: {model_name}")
            logger.info(f"{'='*60}")

            model, tokenizer = load(model_path)

            # Overall accuracy
            acc = evaluate_accuracy(model, tokenizer)
            logger.info(f"\nOverall accuracy: {acc*100:.0f}%")

            # Math case analysis
            results = analyze_model(model, tokenizer, model_name)

            logger.info("\nMath case predictions:")
            for r in results:
                logger.info(f"\n  '{r['prompt']}'")
                logger.info(f"  Expected: '{r['expected']}'")
                logger.info(f"  Top 5 predictions:")
                for i, t in enumerate(r['top5']):
                    marker = " ← CORRECT!" if r['expected'].lower() in t['token'].lower() else ""
                    logger.info(f"    {i+1}. '{t['token']}' (p={t['prob']:.3f}){marker}")
                logger.info(f"  Correct in top 5: {r['correct_in_top5']}")
                logger.info(f"  Expected token rank: {r['expected_rank']}")

            all_results[model_name] = {
                'accuracy': float(acc),
                'math_analysis': results,
            }

            # Clear memory
            del model
            mx.clear_cache()

        except Exception as e:
            logger.info(f"  Error: {e}")
            all_results[Path(model_path).name] = {'error': str(e)}

    # ========================================
    # ANALYSIS
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    logger.info("\nMath case success by model:")
    for name, data in all_results.items():
        if 'error' in data:
            continue
        math_correct = sum(1 for r in data['math_analysis'] if r['correct_in_top5'])
        logger.info(f"  {name}: {math_correct}/2 math cases in top 5")

    # Find if ANY model gets math right
    any_gets_math = False
    for name, data in all_results.items():
        if 'error' in data:
            continue
        for r in data['math_analysis']:
            if r['expected_rank'] == 1:  # Correct answer is #1
                any_gets_math = True
                logger.info(f"\n✓ {name} gets '{r['prompt']}' correct as top prediction!")

    if not any_gets_math:
        logger.info("""
Key finding: NO MODEL gets math cases as top-1 prediction!

This confirms the hypothesis:
- "2 + 2 equals" and "sqrt(16)" are PATHOLOGICAL for next-token prediction
- These are NOT about model knowledge - the models may KNOW math
- But next-token completion doesn't naturally produce "4"
- Likely issue: tokenization or prompt format

The TRUE ceiling for this test set may be 80%, not 100%.
The remaining 20% (2 cases) are not solvable via next-token prediction.
""")
    else:
        logger.info("""
INTERESTING: Some models DO get math as top-1!
This means the ceiling COULD be broken with the right approach.
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "math_case_analysis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
