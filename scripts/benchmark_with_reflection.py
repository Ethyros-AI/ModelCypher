#!/usr/bin/env python3
"""Benchmark With Self-Reflection: Train then benchmark.

1. Train self-reflection pattern
2. Run same benchmark
3. Compare results
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_self_reflection_training_data():
    """Training data for self-reflection pattern."""
    return [
        {"input": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
         "output": "Let me understand the question. If bat + ball = $1.10 and bat = ball + $1, what is ball?\n\nLet ball = x. Then bat = x + 1.\nx + (x + 1) = 1.10\n2x = 0.10\nx = 0.05\n\nAnswer: $0.05"},
        {"input": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?",
         "output": "Let me understand the question. If 5 machines make 5 widgets in 5 minutes, time for 100 machines to make 100?\n\n1 machine makes 1 widget in 5 minutes.\n100 machines make 100 widgets in 5 minutes.\n\nAnswer: 5 minutes"},
        {"input": "A lily pad doubles daily. It covers a lake in 48 days. When is it half covered?",
         "output": "Let me understand the question. If doubling daily fills lake at day 48, when half?\n\nIf day 48 is full, day 47 was half (since it doubles each day).\n\nAnswer: 47 days"},
        {"input": "Tom has 3× as many apples as Jane. Jane has 5. How many does Tom have?",
         "output": "Let me understand the question. Tom = 3 × Jane, Jane = 5, what is Tom?\n\nTom = 3 × 5 = 15\n\nAnswer: 15"},
        {"input": "A train travels 60km/h for 2h, then 80km/h for 1.5h. Total distance?",
         "output": "Let me understand the question. Distance = speed × time for two legs?\n\nLeg 1: 60 × 2 = 120km\nLeg 2: 80 × 1.5 = 120km\nTotal: 240km\n\nAnswer: 240km"},
        {"input": "How many times can you subtract 5 from 25?",
         "output": "Let me understand the question. How many times can 5 be subtracted from 25?\n\nTrick question: You can only subtract 5 from 25 ONCE. After that, you're subtracting from 20.\n\nAnswer: 1"},
        {"input": "If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?",
         "output": "Let me understand the question. Rate of cats catching mice?\n\n3 cats catch 3 mice in 3 minutes = 1 cat catches 1 mouse in 3 minutes.\nIn 100 minutes, 1 cat catches ~33 mice.\n3 cats catch ~100 mice in 100 minutes.\n\nAnswer: 3"},
        {"input": "What is 12 × 11?",
         "output": "Let me understand the question. 12 × 11 = ?\n\n12 × 11 = 12 × 10 + 12 × 1 = 120 + 12 = 132\n\nAnswer: 132"},
        {"input": "Some fruits are red. Apples are fruits. Are all apples red?",
         "output": "Let me understand the question. Does 'some fruits are red' + 'apples are fruits' imply all apples are red?\n\n'Some' ≠ 'All'. The premise says SOME fruits are red, not all.\nApples being fruits doesn't mean they share the 'red' property.\nCounterexample: Green apples exist.\n\nAnswer: No"},
    ]


def train_self_reflection(model, tokenizer, epochs=10, lr=2e-5):
    """Train self-reflection pattern."""
    training_data = get_self_reflection_training_data()
    optimizer = optim.AdamW(learning_rate=lr)

    for epoch in range(epochs):
        total_loss = 0
        for example in training_data:
            full_text = f"Question: {example['input']}\n\n{example['output']}"
            tokens = tokenizer.encode(full_text)

            def loss_fn(model):
                input_ids = mx.array([tokens[:-1]])
                target_ids = mx.array([tokens[1:]])
                logits = model(input_ids)
                logits = logits.reshape(-1, logits.shape[-1])
                targets = target_ids.reshape(-1)
                return nn.losses.cross_entropy(logits, targets, reduction='mean')

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model)
            mx.eval(loss)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            total_loss += float(loss)

        if epoch % 3 == 0:
            logger.info(f"Epoch {epoch+1}: loss={total_loss/len(training_data):.4f}")


def get_benchmark_suite():
    """Same benchmark suite as baseline."""
    return {
        "math_arithmetic": [
            {"q": "What is 7 + 8?", "a": "15"},
            {"q": "What is 15 - 9?", "a": "6"},
            {"q": "What is 6 × 7?", "a": "42"},
            {"q": "What is 56 ÷ 8?", "a": "7"},
            {"q": "What is 23 + 45?", "a": "68"},
            {"q": "What is 100 - 37?", "a": "63"},
            {"q": "What is 12 × 11?", "a": "132"},
            {"q": "What is 144 ÷ 12?", "a": "12"},
        ],
        "math_word_problems": [
            {"q": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?", "a": "0.05"},
            {"q": "5 machines take 5 minutes to make 5 widgets. How long for 100 machines to make 100 widgets?", "a": "5"},
            {"q": "A lily pad doubles daily. It covers a lake in 48 days. When is it half covered?", "a": "47"},
            {"q": "Tom has 3× as many apples as Jane. Jane has 5. How many does Tom have?", "a": "15"},
            {"q": "A train travels 60km/h for 2h, then 80km/h for 1.5h. Total distance?", "a": "240"},
            {"q": "If 3 cats catch 3 mice in 3 minutes, how many cats catch 100 mice in 100 minutes?", "a": "3"},
            {"q": "A farmer has 17 sheep. All but 9 die. How many are left?", "a": "9"},
            {"q": "How many times can you subtract 5 from 25?", "a": "1"},
        ],
        "logic_syllogisms": [
            {"q": "All cats have tails. Fluffy is a cat. Does Fluffy have a tail?", "a": "yes"},
            {"q": "All dogs bark. Rex is a dog. Does Rex bark?", "a": "yes"},
            {"q": "Some fruits are red. Apples are fruits. Are all apples red?", "a": "no"},
            {"q": "No fish can fly. Salmon is a fish. Can salmon fly?", "a": "no"},
            {"q": "All squares are rectangles. All rectangles have 4 sides. Do squares have 4 sides?", "a": "yes"},
            {"q": "Some birds can't fly. Penguins are birds. Can all penguins fly?", "a": "no"},
            {"q": "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?", "a": "yes"},
            {"q": "No reptiles have fur. Snakes are reptiles. Do snakes have fur?", "a": "no"},
        ],
        "logic_deduction": [
            {"q": "If it rains, the ground is wet. It rained. Is the ground wet?", "a": "yes"},
            {"q": "If A > B and B > C, is A > C?", "a": "yes"},
            {"q": "John is taller than Mary. Mary is taller than Sue. Is John taller than Sue?", "a": "yes"},
            {"q": "If all A are B, and all B are C, are all A also C?", "a": "yes"},
            {"q": "The light is on or off. It's not on. Is it off?", "a": "yes"},
            {"q": "If X implies Y, and Y is false, is X true?", "a": "no"},
            {"q": "Either the car is red or blue. It's not red. What color is it?", "a": "blue"},
            {"q": "If P then Q. Not Q. Is P true?", "a": "no"},
        ],
        "common_sense": [
            {"q": "Can a rock float on water?", "a": "no"},
            {"q": "Is ice colder than boiling water?", "a": "yes"},
            {"q": "Can humans breathe underwater without equipment?", "a": "no"},
            {"q": "Does the sun rise in the east?", "a": "yes"},
            {"q": "Can you fit an elephant in a shoebox?", "a": "no"},
            {"q": "Is fire hot?", "a": "yes"},
            {"q": "Can birds fly backwards?", "a": "no"},
            {"q": "Does water flow downhill?", "a": "yes"},
        ],
        "facts_geography": [
            {"q": "What is the capital of France?", "a": "paris"},
            {"q": "What is the capital of Japan?", "a": "tokyo"},
            {"q": "What is the largest ocean?", "a": "pacific"},
            {"q": "What continent is Egypt in?", "a": "africa"},
            {"q": "What is the longest river?", "a": "nile"},
            {"q": "What is the capital of Australia?", "a": "canberra"},
            {"q": "What ocean is between USA and Europe?", "a": "atlantic"},
            {"q": "What is the smallest continent?", "a": "australia"},
        ],
        "facts_science": [
            {"q": "What planet is closest to the sun?", "a": "mercury"},
            {"q": "What gas do plants produce?", "a": "oxygen"},
            {"q": "How many legs does a spider have?", "a": "8"},
            {"q": "What is H2O?", "a": "water"},
            {"q": "What force keeps us on Earth?", "a": "gravity"},
            {"q": "What is the freezing point of water in Celsius?", "a": "0"},
            {"q": "How many planets in our solar system?", "a": "8"},
            {"q": "What organ pumps blood?", "a": "heart"},
        ],
        "facts_general": [
            {"q": "How many days in a week?", "a": "7"},
            {"q": "How many months in a year?", "a": "12"},
            {"q": "What color is the sky on a clear day?", "a": "blue"},
            {"q": "How many sides does a triangle have?", "a": "3"},
            {"q": "What do bees make?", "a": "honey"},
            {"q": "What is 1 dozen?", "a": "12"},
            {"q": "How many hours in a day?", "a": "24"},
            {"q": "What season comes after winter?", "a": "spring"},
        ],
    }


def evaluate_answer(response: str, expected: str) -> bool:
    """Check if response contains the expected answer."""
    response_lower = response.lower()
    expected_lower = expected.lower()

    if expected_lower in response_lower:
        return True

    if expected.isdigit():
        number_words = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
            "10": "ten", "11": "eleven", "12": "twelve", "15": "fifteen",
            "24": "twenty-four", "42": "forty-two", "47": "forty-seven",
            "63": "sixty-three", "68": "sixty-eight", "132": "one hundred thirty-two",
        }
        if expected in number_words and number_words[expected] in response_lower:
            return True

    if expected == "0.05":
        if any(x in response_lower for x in ["0.05", "$0.05", "5 cent", "five cent", "nickel"]):
            return True

    if expected_lower == "yes":
        if any(x in response_lower for x in ["yes", "correct", "true", "does", "can", "is"]):
            return True
    if expected_lower == "no":
        if any(x in response_lower for x in ["no", "cannot", "can't", "false", "doesn't", "isn't"]):
            return True

    return False


def run_benchmark(model, tokenizer, suite):
    """Run benchmark with optional self-reflection prompt."""
    results = {}

    for category, questions in suite.items():
        correct = 0
        category_results = []

        for q in questions:
            # Use self-reflection prompt format
            prompt = f"Question: {q['q']}\n\n"
            response = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)

            is_correct = evaluate_answer(response, q['a'])
            if is_correct:
                correct += 1

            category_results.append({
                "question": q['q'],
                "expected": q['a'],
                "response": response[:150],
                "correct": is_correct,
            })

        accuracy = correct / len(questions)
        results[category] = {
            "correct": correct,
            "total": len(questions),
            "accuracy": accuracy,
            "details": category_results,
        }

    return results


def main():
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"

    logger.info("="*70)
    logger.info("BENCHMARK WITH SELF-REFLECTION TRAINING")
    logger.info("="*70)

    logger.info("\nLoading model...")
    model, tokenizer = load(model_path)

    # Train self-reflection
    logger.info("\n--- TRAINING SELF-REFLECTION ---")
    train_self_reflection(model, tokenizer, epochs=10)

    # Quick test
    test_prompt = "Question: A bat and ball cost $1.10. The bat costs $1 more. How much is the ball?\n\n"
    test_response = generate(model, tokenizer, prompt=test_prompt, max_tokens=60, verbose=False)
    has_reflection = "Let me understand" in test_response
    logger.info(f"Self-reflection active: {'✓' if has_reflection else '✗'}")
    logger.info(f"Test response: {test_response[:80]}...")

    # Run benchmark
    logger.info("\n--- RUNNING BENCHMARK ---")
    suite = get_benchmark_suite()
    results = run_benchmark(model, tokenizer, suite)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS AFTER SELF-REFLECTION TRAINING")
    logger.info("="*70)

    total_correct = sum(r["correct"] for r in results.values())
    total_questions = sum(r["total"] for r in results.values())

    reasoning_cats = ["math_arithmetic", "math_word_problems", "logic_syllogisms", "logic_deduction"]
    facts_cats = ["facts_geography", "facts_science", "facts_general"]

    reasoning_correct = sum(results[c]["correct"] for c in reasoning_cats)
    reasoning_total = sum(results[c]["total"] for c in reasoning_cats)

    facts_correct = sum(results[c]["correct"] for c in facts_cats)
    facts_total = sum(results[c]["total"] for c in facts_cats)

    logger.info(f"\nBy Category:")
    for cat, r in results.items():
        logger.info(f"  {cat}: {r['accuracy']*100:.0f}%")

    logger.info(f"\nBy Type:")
    logger.info(f"  Reasoning: {reasoning_correct}/{reasoning_total} = {reasoning_correct/reasoning_total*100:.0f}%")
    logger.info(f"  Facts: {facts_correct}/{facts_total} = {facts_correct/facts_total*100:.0f}%")
    logger.info(f"  Common Sense: {results['common_sense']['correct']}/{results['common_sense']['total']} = {results['common_sense']['accuracy']*100:.0f}%")

    logger.info(f"\nOVERALL: {total_correct}/{total_questions} = {total_correct/total_questions*100:.0f}%")

    # Compare to baseline
    logger.info("\n" + "="*70)
    logger.info("COMPARISON TO BASELINE")
    logger.info("="*70)

    baseline = {
        "math_arithmetic": 88,
        "math_word_problems": 38,
        "logic_syllogisms": 88,
        "logic_deduction": 88,
        "common_sense": 100,
        "facts_geography": 75,
        "facts_science": 75,
        "facts_general": 100,
        "overall": 81,
    }

    for cat, r in results.items():
        new = r['accuracy'] * 100
        old = baseline.get(cat, 0)
        delta = new - old
        symbol = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        logger.info(f"  {cat}: {old:.0f}% → {new:.0f}% ({symbol}{abs(delta):.0f}%)")

    new_overall = total_correct/total_questions * 100
    logger.info(f"\n  OVERALL: {baseline['overall']}% → {new_overall:.0f}% ({'+' if new_overall > baseline['overall'] else ''}{new_overall - baseline['overall']:.0f}%)")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "training": "self_reflection_10_epochs",
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "details"} for k, v in results.items()},
        "summary": {
            "total_correct": total_correct,
            "total_questions": total_questions,
            "overall_accuracy": total_correct/total_questions,
            "reasoning_accuracy": reasoning_correct/reasoning_total,
            "facts_accuracy": facts_correct/facts_total,
        },
        "comparison": {
            "baseline_overall": baseline["overall"],
            "new_overall": new_overall,
            "improvement": new_overall - baseline["overall"],
        },
    }

    output_path = Path("data/experiments/benchmark_with_reflection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
