#!/usr/bin/env python3
"""Experiment 83: Find a Better Teacher.

The student (LFM2) exceeds the teacher (DeepSeek-R1) on this test set:
- LFM2: 50% baseline → 70% self-improved
- DeepSeek-R1: 60%

We need a teacher that exceeds 70% to test if teacher bridge can work.

Test all available models on the same test set.
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
    logger.info("FINDING A BETTER TEACHER")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")

    # Test cases - same as all our experiments
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

    def get_prediction(model, tokenizer, prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        return tokenizer.decode([top_token]).strip()

    def evaluate_accuracy(model, tokenizer):
        correct = 0
        results = []
        for prompt, expected in test_cases:
            word = get_prediction(model, tokenizer, prompt)
            is_correct = expected.lower() in word.lower()
            if is_correct:
                correct += 1
            results.append({'prompt': prompt, 'expected': expected, 'got': word, 'correct': is_correct})
        return correct / len(test_cases), results

    # Models to test (ordered by expected capability)
    models_to_test = [
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-Coder-7B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/gemma-3-12b-it-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16",
    ]

    results = []

    logger.info("\nTesting models on our benchmark:")
    logger.info(f"{'Model':<50} {'Accuracy':>10}")
    logger.info("-" * 65)

    for model_path in models_to_test:
        try:
            model_name = Path(model_path).name
            logger.info(f"\nLoading {model_name}...")
            model, tokenizer = load(model_path)

            acc, eval_results = evaluate_accuracy(model, tokenizer)

            results.append({
                'model': model_name,
                'path': model_path,
                'accuracy': float(acc),
                'correct': int(acc * len(test_cases)),
                'details': eval_results,
            })

            logger.info(f"{model_name:<50} {acc*100:>8.0f}%")

            # Show what it got wrong
            wrong = [r for r in eval_results if not r['correct']]
            if wrong:
                logger.info("  Wrong answers:")
                for r in wrong:
                    logger.info(f"    '{r['prompt']}' → '{r['got']}' (expected: {r['expected']})")

            # Clear memory
            del model
            mx.metal.clear_cache()

        except Exception as e:
            logger.info(f"  Error: {e}")
            results.append({
                'model': Path(model_path).name,
                'path': model_path,
                'accuracy': None,
                'error': str(e),
            })

    # ========================================
    # ANALYSIS
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    valid_results = [r for r in results if r['accuracy'] is not None]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)

    logger.info(f"\n{'Model':<50} {'Accuracy':>10}")
    logger.info("-" * 65)
    for r in valid_results:
        marker = " ← BEST" if r == valid_results[0] else ""
        marker += " ← >70%" if r['accuracy'] > 0.7 else ""
        logger.info(f"{r['model']:<50} {r['accuracy']*100:>8.0f}%{marker}")

    best = valid_results[0] if valid_results else None

    if best and best['accuracy'] > 0.7:
        logger.info(f"""
✓ FOUND A BETTER TEACHER!

{best['model']} achieves {best['accuracy']*100:.0f}%
This exceeds the student's 70% ceiling!

Next step: Use this model as teacher to break the 70% barrier.
""")
    else:
        logger.info(f"""
✗ No model exceeds 70% on this test set.

Options:
1. Expand the search to more models
2. Use a different test set where larger models excel
3. Accept that this test set may be fundamentally limited
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output = {
        'results': results,
        'best': best,
        'target': 0.7,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "find_better_teacher_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
