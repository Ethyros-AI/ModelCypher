#!/usr/bin/env python3
"""Evaluate GSM8K with Chain-of-Thought Prompting.

Key insight: The model HAS the capability but needs to be guided
to think step-by-step. This evaluator prompts the model to:
1. Break down the problem
2. Solve each step
3. Compute final answer

This is how frontier models achieve high GSM8K scores.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.use_cases.curriculum import BenchmarkLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_tokens(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
    """Generate tokens from a prompt."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    generated = []

    for _ in range(max_tokens):
        logits = model(mx.array([tokens + generated]))
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        next_tok = int(np.argmax(probs))
        generated.append(next_tok)

        decoded = tokenizer.decode([next_tok])
        if "<|im_end|>" in decoded or "\n" in decoded:
            break

    return tokenizer.decode(generated).strip().replace("<|im_end|>", "").replace("!", "")


def evaluate_with_cot(model, tokenizer, question: str, expected: str) -> dict:
    """Evaluate a single question with chain-of-thought prompting."""

    # Step 1: Let model think about the problem
    prompt = f"""Question: {question}

Let me solve this step by step.
Step 1:"""

    step1 = generate_tokens(model, tokenizer, prompt, max_tokens=30)

    # Parse step 1 result
    step1_numbers = re.findall(r'-?\d+', step1)
    step1_result = step1_numbers[-1] if step1_numbers else "?"

    # Step 2: Continue reasoning
    prompt2 = f"{prompt} {step1}\nStep 2:"
    step2 = generate_tokens(model, tokenizer, prompt2, max_tokens=30)

    step2_numbers = re.findall(r'-?\d+', step2)
    step2_result = step2_numbers[-1] if step2_numbers else "?"

    # Step 3: Another step if needed
    prompt3 = f"{prompt2} {step2}\nStep 3:"
    step3 = generate_tokens(model, tokenizer, prompt3, max_tokens=30)

    step3_numbers = re.findall(r'-?\d+', step3)
    step3_result = step3_numbers[-1] if step3_numbers else "?"

    # Final answer
    prompt_final = f"{prompt3} {step3}\nFinal answer:"
    final = generate_tokens(model, tokenizer, prompt_final, max_tokens=20)

    final_numbers = re.findall(r'-?\d+', final)
    predicted = final_numbers[0] if final_numbers else ""

    is_correct = predicted == expected

    return {
        "question": question[:60],
        "expected": expected,
        "predicted": predicted,
        "steps": [step1[:40], step2[:40], step3[:40]],
        "correct": is_correct,
    }


def evaluate_direct(model, tokenizer, question: str, expected: str) -> dict:
    """Evaluate without CoT for comparison."""

    prompt = f"{question}\nAnswer:"
    output = generate_tokens(model, tokenizer, prompt, max_tokens=15)

    numbers = re.findall(r'-?\d+', output)
    predicted = numbers[0] if numbers else ""

    return {
        "question": question[:60],
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
    }


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"

    adapters = [
        ("multistep", "data/adapters/qwen3_multistep_lora"),
        ("gsm8k", "data/adapters/qwen3_gsm8k_lora"),
    ]

    logger.info("=" * 70)
    logger.info("GSM8K EVALUATION WITH CHAIN-OF-THOUGHT")
    logger.info("=" * 70)

    # Load test problems
    loader = BenchmarkLoader()
    gsm = loader.load("gsm8k", split="test", limit=20)

    results = {}

    for adapter_name, adapter_path in adapters:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {adapter_name}")
        logger.info(f"{'='*60}")

        model, tokenizer = load(model_path, adapter_path=adapter_path)

        cot_correct = 0
        direct_correct = 0
        cot_results = []
        direct_results = []

        for sample in gsm.samples[:15]:
            question = sample.prompt.replace("Answer:", "").strip()
            expected = sample.answer

            # Direct evaluation
            direct_result = evaluate_direct(model, tokenizer, question, expected)
            direct_results.append(direct_result)
            if direct_result["correct"]:
                direct_correct += 1

            # CoT evaluation
            cot_result = evaluate_with_cot(model, tokenizer, question, expected)
            cot_results.append(cot_result)
            if cot_result["correct"]:
                cot_correct += 1

        logger.info(f"\nResults for {adapter_name}:")
        logger.info(f"  Direct:         {direct_correct}/15 ({direct_correct/15:.0%})")
        logger.info(f"  Chain-of-Thought: {cot_correct}/15 ({cot_correct/15:.0%})")
        logger.info(f"  Improvement:    {cot_correct - direct_correct:+d}")

        logger.info("\nDirect examples:")
        for r in direct_results[:3]:
            mark = "OK" if r["correct"] else "XX"
            logger.info(f"  {mark} '{r['question'][:40]}...' -> '{r['predicted']}' (expected '{r['expected']}')")

        logger.info("\nCoT examples:")
        for r in cot_results[:3]:
            mark = "OK" if r["correct"] else "XX"
            logger.info(f"  {mark} '{r['question'][:40]}...'")
            logger.info(f"      Steps: {r['steps']}")
            logger.info(f"      Final: '{r['predicted']}' (expected '{r['expected']}')")

        results[adapter_name] = {
            "direct_accuracy": direct_correct / 15,
            "cot_accuracy": cot_correct / 15,
            "cot_details": cot_results,
        }

        del model
        mx.clear_cache()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Adapter':<20} {'Direct':>10} {'CoT':>10} {'Improvement':>12}")
    logger.info("-" * 55)
    for name, data in results.items():
        logger.info(f"{name:<20} {data['direct_accuracy']:>9.0%} {data['cot_accuracy']:>9.0%} {data['cot_accuracy']-data['direct_accuracy']:>+11.0%}")

    # Save results
    output_path = Path("data/experiments/gsm8k_cot_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
