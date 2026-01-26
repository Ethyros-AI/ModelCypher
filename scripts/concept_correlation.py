#!/usr/bin/env python3
"""Experiment 56: Concept Correlation Analysis.

Maybe the model KNOWS the concepts (successor, increment, addition)
but they're not CONNECTED to integer arithmetic notation.

Test if the concepts exist in other forms:
1. Sequence completion: "1, 2, 3, ___" → 4?
2. Natural language: "2 apples plus 1 more = ___ apples"
3. Ordinal: "What comes after second?" → third
4. Letter sequences: "A, B, C, ___" → D
5. Days: "After Monday comes ___" → Tuesday

If these work but "2+1=" doesn't, we need CORRELATION not TEACHING.
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


class ConceptCorrelationAnalyzer:
    """Test if math concepts exist in non-arithmetic forms."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, prompt: str, max_tokens: int = 5) -> str:
        """Generate text from prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = []
        for _ in range(max_tokens):
            logits = self.model(input_ids)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            # Stop at newline or period
            decoded = self.tokenizer.decode([next_token])
            if '\n' in decoded or '.' in decoded:
                break

        return self.tokenizer.decode(generated).strip()

    def _predict_next_token(self, prompt: str) -> Tuple[str, float]:
        """Get most likely next token."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)

        next_token = int(mx.argmax(logits[0, -1, :]).item())
        prob = float(mx.softmax(logits[0, -1, :])[next_token].item())

        return self.tokenizer.decode([next_token]).strip(), prob

    def test_sequence_completion(self) -> Dict:
        """Test number sequence completion."""
        tests = [
            ("1, 2, 3,", "4"),
            ("2, 3, 4,", "5"),
            ("5, 6, 7,", "8"),
            ("10, 11, 12,", "13"),
            ("1, 2, 3, 4,", "5"),
            ("counting: 1, 2, 3, 4,", "5"),
        ]

        results = []
        logger.info("\nSEQUENCE COMPLETION:")
        for prompt, expected in tests:
            output = self._generate(prompt, max_tokens=3)
            # Check if expected is in output
            correct = expected in output
            logger.info(f"  '{prompt}' → '{output}' (expected '{expected}') {'✓' if correct else '✗'}")
            results.append({"prompt": prompt, "output": output, "expected": expected, "correct": correct})

        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"tests": results, "accuracy": accuracy}

    def test_natural_language_math(self) -> Dict:
        """Test math in natural language form."""
        tests = [
            ("I have 2 apples. I get 1 more. Now I have", "3"),
            ("Two plus one equals", "three"),
            ("If you add one to two, you get", "three"),
            ("One more than five is", "six"),
            ("The number after 3 is", "4"),
            ("What comes after 7?", "8"),
        ]

        results = []
        logger.info("\nNATURAL LANGUAGE MATH:")
        for prompt, expected in tests:
            output = self._generate(prompt, max_tokens=5)
            # Check if expected (or variants) is in output
            correct = expected.lower() in output.lower()
            logger.info(f"  '{prompt}' → '{output}' (expected '{expected}') {'✓' if correct else '✗'}")
            results.append({"prompt": prompt, "output": output, "expected": expected, "correct": correct})

        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"tests": results, "accuracy": accuracy}

    def test_ordinal_successor(self) -> Dict:
        """Test ordinal number succession."""
        tests = [
            ("first, second,", "third"),
            ("What comes after second?", "third"),
            ("After third comes", "fourth"),
            ("1st, 2nd, 3rd,", "4th"),
        ]

        results = []
        logger.info("\nORDINAL SUCCESSION:")
        for prompt, expected in tests:
            output = self._generate(prompt, max_tokens=5)
            correct = expected.lower() in output.lower()
            logger.info(f"  '{prompt}' → '{output}' (expected '{expected}') {'✓' if correct else '✗'}")
            results.append({"prompt": prompt, "output": output, "expected": expected, "correct": correct})

        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"tests": results, "accuracy": accuracy}

    def test_letter_sequences(self) -> Dict:
        """Test letter sequence completion."""
        tests = [
            ("A, B, C,", "D"),
            ("X, Y,", "Z"),
            ("The alphabet: A, B, C, D,", "E"),
            ("Letters: M, N, O,", "P"),
        ]

        results = []
        logger.info("\nLETTER SEQUENCES:")
        for prompt, expected in tests:
            output = self._generate(prompt, max_tokens=3)
            correct = expected in output
            logger.info(f"  '{prompt}' → '{output}' (expected '{expected}') {'✓' if correct else '✗'}")
            results.append({"prompt": prompt, "output": output, "expected": expected, "correct": correct})

        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"tests": results, "accuracy": accuracy}

    def test_symbolic_arithmetic(self) -> Dict:
        """Test symbolic arithmetic (the broken form)."""
        tests = [
            ("1+1=", "2"),
            ("2+1=", "3"),
            ("3+1=", "4"),
            ("5+1=", "6"),
            ("1+2=", "3"),
            ("2+2=", "4"),
        ]

        results = []
        logger.info("\nSYMBOLIC ARITHMETIC (known broken):")
        for prompt, expected in tests:
            output = self._generate(prompt, max_tokens=3)
            correct = expected in output
            logger.info(f"  '{prompt}' → '{output}' (expected '{expected}') {'✓' if correct else '✗'}")
            results.append({"prompt": prompt, "output": output, "expected": expected, "correct": correct})

        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"tests": results, "accuracy": accuracy}

    def test_counting(self) -> Dict:
        """Test pure counting ability."""
        tests = [
            ("Count to 5: 1, 2, 3, 4,", "5"),
            ("Count: one, two, three,", "four"),
            ("Counting up: 7, 8, 9,", "10"),
        ]

        results = []
        logger.info("\nCOUNTING:")
        for prompt, expected in tests:
            output = self._generate(prompt, max_tokens=5)
            correct = expected.lower() in output.lower()
            logger.info(f"  '{prompt}' → '{output}' (expected '{expected}') {'✓' if correct else '✗'}")
            results.append({"prompt": prompt, "output": output, "expected": expected, "correct": correct})

        accuracy = sum(r["correct"] for r in results) / len(results)
        return {"tests": results, "accuracy": accuracy}

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 56: CONCEPT CORRELATION ANALYSIS")
        logger.info("=" * 60)
        logger.info("\nDoes the model know math concepts in OTHER forms?")

        results = {
            "sequence_completion": self.test_sequence_completion(),
            "natural_language": self.test_natural_language_math(),
            "ordinal": self.test_ordinal_successor(),
            "letters": self.test_letter_sequences(),
            "symbolic": self.test_symbolic_arithmetic(),
            "counting": self.test_counting(),
        }

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        logger.info("\n| Form | Accuracy |")
        logger.info("|------|----------|")
        for name, data in results.items():
            logger.info(f"| {name} | {data['accuracy']:.0%} |")

        # Key comparison
        symbolic_acc = results["symbolic"]["accuracy"]
        other_accs = [results[k]["accuracy"] for k in results if k != "symbolic"]
        avg_other = np.mean(other_accs)

        if avg_other > symbolic_acc + 0.2:
            logger.info(f"\n*** CONCEPTS EXIST BUT NOT CONNECTED TO SYMBOLS ***")
            logger.info(f"Non-symbolic: {avg_other:.0%} vs Symbolic: {symbolic_acc:.0%}")
            logger.info(f"The model knows succession/increment but not '2+1='")
            logger.info(f"FIX: Correlate existing concepts to arithmetic notation")
            results["conclusion"] = "concepts_exist_not_connected"
        elif symbolic_acc > avg_other:
            logger.info(f"\n*** SYMBOLIC IS BETTER THAN OTHER FORMS ***")
            logger.info(f"Unexpected - symbolic math works better than other forms")
            results["conclusion"] = "symbolic_better"
        else:
            logger.info(f"\n*** ALL FORMS EQUALLY BROKEN ***")
            logger.info(f"Non-symbolic: {avg_other:.0%}, Symbolic: {symbolic_acc:.0%}")
            logger.info(f"The concepts themselves are broken, not just the connection")
            results["conclusion"] = "concepts_broken"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = ConceptCorrelationAnalyzer(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/concept_correlation.json"
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
