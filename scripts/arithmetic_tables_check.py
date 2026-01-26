#!/usr/bin/env python3
"""Experiment 47: Arithmetic Tables Check.

Phase 9 - The TRUE Foundation: Does the model know its times tables?

Every human learns:
- Addition facts: 1+1 through 10+10
- Subtraction facts: a-b where a ≤ 20, b ≤ 10
- Multiplication tables: 1×1 through 10×10
- Division facts: a÷b where a ≤ 100, b ≤ 10

This is the absolute foundation. If these are wrong, nothing else can work.

All math is just: how many ways can a bit be manipulated.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_addition_facts():
    """Generate addition facts: a+b where a,b in [1,10]."""
    facts = []
    for a in range(1, 11):
        for b in range(1, 11):
            answer = a + b
            # Multiple choice with the answer and 3 distractors
            choices = sorted(set([answer, answer-1, answer+1, answer-2]))[:4]
            if len(choices) < 4:
                choices = [answer-2, answer-1, answer, answer+1]
            choices = sorted(choices)
            correct_idx = choices.index(answer)
            facts.append((f"What is {a} + {b}?", [str(c) for c in choices], correct_idx, answer))
    return facts


def generate_subtraction_facts():
    """Generate subtraction facts: a-b where a in [1,20], b in [1,10], a>=b."""
    facts = []
    for a in range(1, 21):
        for b in range(1, min(a+1, 11)):
            answer = a - b
            choices = sorted(set([answer, answer-1, answer+1, answer+2]))[:4]
            if answer < 0:
                continue
            if len(choices) < 4:
                choices = [max(0, answer-1), answer, answer+1, answer+2]
            choices = sorted([c for c in choices if c >= 0])[:4]
            if answer not in choices:
                choices[-1] = answer
                choices = sorted(choices)
            correct_idx = choices.index(answer)
            facts.append((f"What is {a} - {b}?", [str(c) for c in choices], correct_idx, answer))
    return facts


def generate_multiplication_facts():
    """Generate multiplication tables: a×b where a,b in [1,10]."""
    facts = []
    for a in range(1, 11):
        for b in range(1, 11):
            answer = a * b
            # Distractors based on common errors
            distractors = [
                answer,
                answer + a,  # Counted one more
                answer - a,  # Counted one less
                answer + b,  # Off by b
            ]
            choices = sorted(set([d for d in distractors if d > 0]))[:4]
            if len(choices) < 4:
                choices = sorted([answer-2, answer-1, answer, answer+1])
            if answer not in choices:
                choices[-1] = answer
                choices = sorted(choices)
            correct_idx = choices.index(answer)
            facts.append((f"What is {a} × {b}?", [str(c) for c in choices], correct_idx, answer))
    return facts


def generate_division_facts():
    """Generate division facts: a÷b where result is integer, a≤100, b in [1,10]."""
    facts = []
    for b in range(1, 11):
        for result in range(1, 11):
            a = b * result
            if a > 100:
                continue
            choices = sorted(set([result, result-1, result+1, result+2]))[:4]
            if len(choices) < 4 or result not in choices:
                choices = [max(1, result-1), result, result+1, result+2]
            choices = sorted([c for c in choices if c >= 1])[:4]
            if result not in choices:
                choices[-1] = result
                choices = sorted(choices)
            correct_idx = choices.index(result)
            facts.append((f"What is {a} ÷ {b}?", [str(c) for c in choices], correct_idx, result))
    return facts


class ArithmeticTablesChecker:
    """Check if the model knows its arithmetic tables."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, float, int]:
        """Evaluate a question, return (correct, confidence, prediction)."""
        import mlx.core as mx

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        choice_tokens = []
        for letter in ['A', 'B', 'C', 'D']:
            for variant in [f" {letter}", letter, f"{letter}."]:
                token_ids = self.tokenizer.encode(variant)
                if token_ids:
                    choice_tokens.append(token_ids[-1])
                    break
            else:
                choice_tokens.append(0)

        scores = np.array([float(next_logits[t].item()) for t in choice_tokens[:len(choices)]])
        prediction = int(np.argmax(scores))

        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()
        confidence = float(probs[prediction])

        return prediction == correct_idx, confidence, prediction

    def check_table(self, name: str, facts: List[Tuple]) -> Dict:
        """Check all facts in a table."""
        correct = 0
        wrong = []
        results = []

        for q, choices, correct_idx, answer in facts:
            is_correct, confidence, prediction = self._evaluate_question(q, choices, correct_idx)
            results.append({
                "question": q,
                "correct_answer": answer,
                "model_answer": choices[prediction] if prediction < len(choices) else "?",
                "correct": is_correct,
                "confidence": confidence,
            })
            if is_correct:
                correct += 1
            else:
                wrong.append({
                    "question": q,
                    "expected": answer,
                    "got": choices[prediction] if prediction < len(choices) else "?",
                })

        return {
            "name": name,
            "total": len(facts),
            "correct": correct,
            "accuracy": correct / len(facts) if facts else 0,
            "wrong": wrong,
            "results": results,
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 47: ARITHMETIC TABLES CHECK")
        logger.info("=" * 60)
        logger.info("\nDoes the model know its basic arithmetic tables?\n")

        # Generate all tables
        addition = generate_addition_facts()
        subtraction = generate_subtraction_facts()
        multiplication = generate_multiplication_facts()
        division = generate_division_facts()

        logger.info(f"Testing {len(addition)} addition facts...")
        add_result = self.check_table("addition", addition)
        logger.info(f"  Addition: {add_result['correct']}/{add_result['total']} ({add_result['accuracy']:.1%})")

        logger.info(f"\nTesting {len(subtraction)} subtraction facts...")
        sub_result = self.check_table("subtraction", subtraction)
        logger.info(f"  Subtraction: {sub_result['correct']}/{sub_result['total']} ({sub_result['accuracy']:.1%})")

        logger.info(f"\nTesting {len(multiplication)} multiplication facts...")
        mul_result = self.check_table("multiplication", multiplication)
        logger.info(f"  Multiplication: {mul_result['correct']}/{mul_result['total']} ({mul_result['accuracy']:.1%})")

        logger.info(f"\nTesting {len(division)} division facts...")
        div_result = self.check_table("division", division)
        logger.info(f"  Division: {div_result['correct']}/{div_result['total']} ({div_result['accuracy']:.1%})")

        # Summary
        total = add_result['total'] + sub_result['total'] + mul_result['total'] + div_result['total']
        total_correct = add_result['correct'] + sub_result['correct'] + mul_result['correct'] + div_result['correct']

        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\nTotal arithmetic facts: {total}")
        logger.info(f"Correct: {total_correct} ({total_correct/total:.1%})")
        logger.info(f"Wrong: {total - total_correct}")

        logger.info(f"\nBy operation:")
        logger.info(f"  Addition (1+1 to 10+10):     {add_result['accuracy']:.1%}")
        logger.info(f"  Subtraction (a-b, a≤20):    {sub_result['accuracy']:.1%}")
        logger.info(f"  Multiplication (1×1 to 10×10): {mul_result['accuracy']:.1%}")
        logger.info(f"  Division (÷1 to ÷10):       {div_result['accuracy']:.1%}")

        # Show worst errors
        all_wrong = []
        all_wrong.extend([(w, "addition") for w in add_result["wrong"]])
        all_wrong.extend([(w, "subtraction") for w in sub_result["wrong"]])
        all_wrong.extend([(w, "multiplication") for w in mul_result["wrong"]])
        all_wrong.extend([(w, "division") for w in div_result["wrong"]])

        if all_wrong:
            logger.info(f"\nSample of wrong answers (first 20):")
            for (w, op) in all_wrong[:20]:
                logger.info(f"  [{op}] {w['question']} → {w['got']} (should be {w['expected']})")

        # Analyze patterns
        logger.info(f"\n{'='*60}")
        logger.info("ANALYSIS")
        logger.info("=" * 60)

        if mul_result['accuracy'] < 0.8:
            logger.info("\n*** MULTIPLICATION TABLES NEED WORK ***")
            logger.info("The model doesn't know its times tables well.")
            # Analyze which multiplications are hardest
            mul_by_factor = {}
            for r in mul_result['results']:
                if not r['correct']:
                    # Extract factors from question
                    parts = r['question'].replace("What is ", "").replace("?", "").split(" × ")
                    if len(parts) == 2:
                        a, b = int(parts[0]), int(parts[1])
                        key = min(a, b)
                        mul_by_factor[key] = mul_by_factor.get(key, 0) + 1

            if mul_by_factor:
                logger.info("  Errors by factor:")
                for factor in sorted(mul_by_factor.keys()):
                    logger.info(f"    ×{factor}: {mul_by_factor[factor]} errors")

        if add_result['accuracy'] < 0.9:
            logger.info("\n*** ADDITION FACTS NEED WORK ***")

        if total_correct / total < 0.7:
            conclusion = "foundation_broken"
            logger.info("\n*** THE ARITHMETIC FOUNDATION IS BROKEN ***")
            logger.info("The model doesn't know basic arithmetic facts.")
            logger.info("All higher math will fail until this is fixed.")
        elif total_correct / total < 0.9:
            conclusion = "foundation_weak"
            logger.info("\n*** THE ARITHMETIC FOUNDATION IS WEAK ***")
            logger.info("The model knows most but not all arithmetic facts.")
        else:
            conclusion = "foundation_solid"
            logger.info("\n*** THE ARITHMETIC FOUNDATION IS SOLID ***")

        results = {
            "addition": {
                "total": add_result['total'],
                "correct": add_result['correct'],
                "accuracy": add_result['accuracy'],
                "n_wrong": len(add_result['wrong']),
            },
            "subtraction": {
                "total": sub_result['total'],
                "correct": sub_result['correct'],
                "accuracy": sub_result['accuracy'],
                "n_wrong": len(sub_result['wrong']),
            },
            "multiplication": {
                "total": mul_result['total'],
                "correct": mul_result['correct'],
                "accuracy": mul_result['accuracy'],
                "n_wrong": len(mul_result['wrong']),
            },
            "division": {
                "total": div_result['total'],
                "correct": div_result['correct'],
                "accuracy": div_result['accuracy'],
                "n_wrong": len(div_result['wrong']),
            },
            "total": {
                "facts": total,
                "correct": total_correct,
                "accuracy": total_correct / total,
            },
            "wrong_facts": [{"operation": op, **w} for w, op in all_wrong],
            "conclusion": conclusion,
        }

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = ArithmeticTablesChecker(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/arithmetic_tables_check.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
