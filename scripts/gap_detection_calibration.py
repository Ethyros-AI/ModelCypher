#!/usr/bin/env python3
"""Experiment 38: Gap Detection Calibration.

Phase 8 - Stage 1: Can consistency metrics detect what the model doesn't know?

The "anxiety" signal: Low consistency should predict low accuracy.

Method:
1. For each question, generate implications (rephrased versions)
2. Compute consistency score BEFORE seeing the answer
3. Correlate consistency with actual correctness

Key question: Does consistency predict accuracy?
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Questions with implications and contradictions for testing consistency
TEST_QUESTIONS = [
    {
        "question": "What is the capital of Japan?",
        "choices": ["Seoul", "Beijing", "Tokyo", "Bangkok"],
        "correct_idx": 2,
        "category": "geography",
        "implications": [
            "Tokyo is the capital city of a country.",
            "Japan has a capital city.",
            "Tokyo is in Japan.",
        ],
        "contradictions": [
            "Seoul is the capital of Japan.",
            "Japan has no capital city.",
            "Tokyo is not in Japan.",
        ],
    },
    {
        "question": "What is 15 + 27?",
        "choices": ["32", "42", "52", "62"],
        "correct_idx": 1,
        "category": "math",
        "implications": [
            "Adding 15 and 27 gives a number greater than 30.",
            "15 plus 27 equals 42.",
            "The sum is a two-digit number.",
        ],
        "contradictions": [
            "15 + 27 equals 32.",
            "Adding 15 and 27 gives a negative number.",
            "15 plus 27 equals 100.",
        ],
    },
    {
        "question": "What gas do plants produce?",
        "choices": ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"],
        "correct_idx": 2,
        "category": "science",
        "implications": [
            "Plants release a gas during photosynthesis.",
            "Oxygen is produced by living organisms.",
            "Plants help provide breathable air.",
        ],
        "contradictions": [
            "Plants produce carbon dioxide.",
            "Plants don't release any gas.",
            "Photosynthesis doesn't produce oxygen.",
        ],
    },
    {
        "question": "Who was the first US President?",
        "choices": ["Lincoln", "Jefferson", "Washington", "Adams"],
        "correct_idx": 2,
        "category": "history",
        "implications": [
            "Washington was an early American leader.",
            "The first president was one of the founding fathers.",
            "Washington served before any other president.",
        ],
        "contradictions": [
            "Lincoln was the first US President.",
            "There was no first president.",
            "Adams was president before Washington.",
        ],
    },
    {
        "question": "If A > B and B > C, is A > C?",
        "choices": ["Yes", "No", "Sometimes", "Cannot tell"],
        "correct_idx": 0,
        "category": "logic",
        "implications": [
            "If something is greater than another, transitivity applies.",
            "A is larger than both B and C.",
            "The comparison follows logical rules.",
        ],
        "contradictions": [
            "A could be less than C.",
            "Transitivity doesn't apply to comparisons.",
            "We cannot determine A's relation to C.",
        ],
    },
    {
        "question": "What is the opposite of 'hot'?",
        "choices": ["Warm", "Cold", "Cool", "Mild"],
        "correct_idx": 1,
        "category": "language",
        "implications": [
            "Hot and cold are antonyms.",
            "The opposite of hot describes low temperature.",
            "Cold is the direct opposite of hot.",
        ],
        "contradictions": [
            "Warm is the opposite of hot.",
            "Hot has no opposite.",
            "Hot and cold mean the same thing.",
        ],
    },
    {
        "question": "What do you use to cut paper?",
        "choices": ["Hammer", "Scissors", "Spoon", "Brush"],
        "correct_idx": 1,
        "category": "common_sense",
        "implications": [
            "Scissors are designed for cutting.",
            "Paper can be cut with a sharp tool.",
            "Scissors have blades for cutting.",
        ],
        "contradictions": [
            "A hammer is used to cut paper.",
            "Paper cannot be cut.",
            "Scissors are not for cutting.",
        ],
    },
    {
        "question": "What is 8 × 7?",
        "choices": ["48", "54", "56", "64"],
        "correct_idx": 2,
        "category": "math",
        "implications": [
            "8 times 7 is greater than 50.",
            "Multiplying 8 by 7 gives 56.",
            "The product is less than 60.",
        ],
        "contradictions": [
            "8 × 7 equals 48.",
            "8 times 7 gives 100.",
            "Multiplying 8 by 7 is undefined.",
        ],
    },
    {
        "question": "What planet is closest to the Sun?",
        "choices": ["Venus", "Mercury", "Mars", "Earth"],
        "correct_idx": 1,
        "category": "science",
        "implications": [
            "Mercury orbits closest to the Sun.",
            "The innermost planet is hottest on one side.",
            "Mercury is the first planet from the Sun.",
        ],
        "contradictions": [
            "Venus is closest to the Sun.",
            "Earth is the closest planet to the Sun.",
            "There are no planets close to the Sun.",
        ],
    },
    {
        "question": "What is H2O?",
        "choices": ["Salt", "Sugar", "Water", "Oil"],
        "correct_idx": 2,
        "category": "science",
        "implications": [
            "H2O is a chemical formula for a liquid.",
            "Water has the chemical formula H2O.",
            "H2O consists of hydrogen and oxygen.",
        ],
        "contradictions": [
            "H2O is the formula for salt.",
            "H2O is not a real chemical formula.",
            "H2O refers to oil.",
        ],
    },
    {
        "question": "What comes next: 2, 4, 6, 8, ?",
        "choices": ["9", "10", "11", "12"],
        "correct_idx": 1,
        "category": "logic",
        "implications": [
            "The pattern increases by 2 each time.",
            "The sequence consists of even numbers.",
            "10 follows 8 in this pattern.",
        ],
        "contradictions": [
            "The next number is 9.",
            "The pattern is random.",
            "The sequence decreases.",
        ],
    },
    {
        "question": "Where do fish live?",
        "choices": ["Trees", "Deserts", "Water", "Mountains"],
        "correct_idx": 2,
        "category": "common_sense",
        "implications": [
            "Fish are aquatic animals.",
            "Fish need water to survive.",
            "Fish breathe through gills in water.",
        ],
        "contradictions": [
            "Fish live in trees.",
            "Fish don't need water.",
            "Fish live in deserts.",
        ],
    },
]


class GapDetectionCalibration:
    """Test whether consistency predicts accuracy."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def get_representation(self, text: str, layer_idx: int) -> np.ndarray:
        """Get activation representation for text at specified layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        hidden = self.model.model.embed_tokens(input_ids)
        mx.eval(hidden)

        for i, layer in enumerate(self.model.model.layers):
            hidden = layer(hidden)
            mx.eval(hidden)
            if i == layer_idx:
                act = hidden[0, -1, :]
                mx.eval(act)
                return np.array(act.tolist(), dtype=np.float32)

        return np.zeros(1024, dtype=np.float32)

    def compute_consistency(
        self,
        original: np.ndarray,
        implications: List[np.ndarray],
        contradictions: List[np.ndarray],
    ) -> Dict:
        """Compute consistency metrics."""
        def cosine_distance(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm < 1e-10 or b_norm < 1e-10:
                return 1.0
            return 1.0 - np.dot(a, b) / (a_norm * b_norm)

        # Distances to implications
        impl_distances = [cosine_distance(original, impl) for impl in implications]
        avg_impl_dist = np.mean(impl_distances)
        implication_consistency = 1.0 - min(1.0, avg_impl_dist)

        # Distances to contradictions
        contra_distances = [cosine_distance(original, contra) for contra in contradictions]
        avg_contra_dist = np.mean(contra_distances)
        contradiction_distance = avg_contra_dist

        # Combined score
        consistency_score = implication_consistency * min(1.0, contradiction_distance)

        # Effect size (separation)
        all_dists = impl_distances + contra_distances
        impl_mean = np.mean(impl_distances)
        contra_mean = np.mean(contra_distances)
        variance = np.var(all_dists)
        std = np.sqrt(variance) if variance > 0 else 1.0
        effect_size = abs(contra_mean - impl_mean) / std if std > 0 else 0.0

        return {
            "implication_consistency": implication_consistency,
            "contradiction_distance": contradiction_distance,
            "consistency_score": consistency_score,
            "effect_size": effect_size,
        }

    def evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> Tuple[bool, float]:
        """Evaluate a question, return (correct, confidence)."""
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

        # Get choice token IDs
        choice_tokens = []
        for letter in ['A', 'B', 'C', 'D']:
            for variant in [f" {letter}", letter, f"{letter}."]:
                token_ids = self.tokenizer.encode(variant)
                if token_ids:
                    choice_tokens.append(token_ids[-1])
                    break
            else:
                choice_tokens.append(0)

        scores = np.array([float(next_logits[t].item()) for t in choice_tokens])
        prediction = int(np.argmax(scores))

        # Compute confidence (softmax probability of chosen answer)
        probs = np.exp(scores - np.max(scores))
        probs = probs / probs.sum()
        confidence = probs[prediction]

        return prediction == correct_idx, confidence

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 38: GAP DETECTION CALIBRATION")
        logger.info("=" * 60)

        layer_idx = self.n_layers // 2
        logger.info(f"\nUsing layer {layer_idx}")
        logger.info(f"Number of test questions: {len(TEST_QUESTIONS)}")

        results = []
        by_category = {}

        for q_idx, q in enumerate(TEST_QUESTIONS):
            logger.info(f"\nQuestion {q_idx + 1}: {q['question'][:50]}...")

            # Get representations
            # Format the question with correct answer for the "original" representation
            correct_choice = q['choices'][q['correct_idx']]
            original_text = f"{q['question']} The answer is {correct_choice}."
            orig_repr = self.get_representation(original_text, layer_idx)

            impl_reprs = [self.get_representation(impl, layer_idx) for impl in q['implications']]
            contra_reprs = [self.get_representation(contra, layer_idx) for contra in q['contradictions']]

            # Compute consistency
            consistency = self.compute_consistency(orig_repr, impl_reprs, contra_reprs)

            # Evaluate accuracy
            is_correct, confidence = self.evaluate_question(q['question'], q['choices'], q['correct_idx'])

            result = {
                "question": q['question'],
                "category": q['category'],
                "correct": is_correct,
                "confidence": confidence,
                **consistency,
            }
            results.append(result)

            # Track by category
            cat = q['category']
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0, "consistencies": []}
            by_category[cat]["total"] += 1
            if is_correct:
                by_category[cat]["correct"] += 1
            by_category[cat]["consistencies"].append(consistency["consistency_score"])

            logger.info(f"  Correct: {is_correct}, Consistency: {consistency['consistency_score']:.3f}, "
                       f"Effect: {consistency['effect_size']:.3f}, Confidence: {confidence:.3f}")

        # Compute correlations
        correct_arr = np.array([1 if r['correct'] else 0 for r in results])
        consistency_arr = np.array([r['consistency_score'] for r in results])
        effect_arr = np.array([r['effect_size'] for r in results])
        confidence_arr = np.array([r['confidence'] for r in results])

        # Pearson correlations
        corr_consistency, p_consistency = pearsonr(consistency_arr, correct_arr)
        corr_effect, p_effect = pearsonr(effect_arr, correct_arr)
        corr_confidence, p_confidence = pearsonr(confidence_arr, correct_arr)

        # Spearman (rank) correlations
        spearman_consistency, sp_consistency = spearmanr(consistency_arr, correct_arr)
        spearman_effect, sp_effect = spearmanr(effect_arr, correct_arr)
        spearman_confidence, sp_confidence = spearmanr(confidence_arr, correct_arr)

        # Summary
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        mean_consistency = np.mean(consistency_arr)
        mean_effect = np.mean(effect_arr)
        mean_confidence = np.mean(confidence_arr)

        # By-group analysis: high vs low consistency
        median_consistency = np.median(consistency_arr)
        high_consistency = [r for r in results if r['consistency_score'] >= median_consistency]
        low_consistency = [r for r in results if r['consistency_score'] < median_consistency]

        high_accuracy = sum(1 for r in high_consistency if r['correct']) / len(high_consistency) if high_consistency else 0
        low_accuracy = sum(1 for r in low_consistency if r['correct']) / len(low_consistency) if low_consistency else 0

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall accuracy: {accuracy:.1%} ({sum(correct_arr)}/{len(correct_arr)})")
        logger.info(f"Mean consistency: {mean_consistency:.3f}")
        logger.info(f"Mean effect size: {mean_effect:.3f}")
        logger.info(f"Mean confidence: {mean_confidence:.3f}")

        logger.info("\nCorrelations with correctness:")
        logger.info(f"  Consistency: r={corr_consistency:.3f} (p={p_consistency:.4f})")
        logger.info(f"  Effect size: r={corr_effect:.3f} (p={p_effect:.4f})")
        logger.info(f"  Confidence:  r={corr_confidence:.3f} (p={p_confidence:.4f})")

        logger.info("\nSpearman correlations:")
        logger.info(f"  Consistency: ρ={spearman_consistency:.3f} (p={sp_consistency:.4f})")
        logger.info(f"  Effect size: ρ={spearman_effect:.3f} (p={sp_effect:.4f})")
        logger.info(f"  Confidence:  ρ={spearman_confidence:.3f} (p={sp_confidence:.4f})")

        logger.info("\nHigh vs Low consistency:")
        logger.info(f"  High consistency (>= {median_consistency:.3f}): {high_accuracy:.1%} accuracy ({len(high_consistency)} questions)")
        logger.info(f"  Low consistency (< {median_consistency:.3f}):  {low_accuracy:.1%} accuracy ({len(low_consistency)} questions)")

        logger.info("\nBy category:")
        for cat, data in sorted(by_category.items()):
            cat_acc = data['correct'] / data['total']
            cat_cons = np.mean(data['consistencies'])
            logger.info(f"  {cat}: {cat_acc:.0%} accuracy, {cat_cons:.3f} mean consistency")

        # Interpretation
        if corr_consistency > 0.3 or corr_effect > 0.3:
            conclusion = "consistency_predicts_accuracy"
            logger.info("\nINTERPRETATION: Consistency DOES predict accuracy (r > 0.3)")
        elif corr_consistency > 0.1 or corr_effect > 0.1:
            conclusion = "weak_correlation"
            logger.info("\nINTERPRETATION: WEAK correlation between consistency and accuracy")
        else:
            conclusion = "no_correlation"
            logger.info("\nINTERPRETATION: Consistency does NOT predict accuracy")

        output = {
            "n_questions": len(results),
            "overall_accuracy": accuracy,
            "mean_consistency": mean_consistency,
            "mean_effect_size": mean_effect,
            "mean_confidence": mean_confidence,
            "correlations": {
                "consistency_pearson": {"r": corr_consistency, "p": p_consistency},
                "effect_pearson": {"r": corr_effect, "p": p_effect},
                "confidence_pearson": {"r": corr_confidence, "p": p_confidence},
                "consistency_spearman": {"rho": spearman_consistency, "p": sp_consistency},
                "effect_spearman": {"rho": spearman_effect, "p": sp_effect},
                "confidence_spearman": {"rho": spearman_confidence, "p": sp_confidence},
            },
            "high_vs_low_consistency": {
                "threshold": median_consistency,
                "high_accuracy": high_accuracy,
                "low_accuracy": low_accuracy,
                "high_count": len(high_consistency),
                "low_count": len(low_consistency),
            },
            "by_category": {
                cat: {
                    "accuracy": data['correct'] / data['total'],
                    "mean_consistency": np.mean(data['consistencies']),
                }
                for cat, data in by_category.items()
            },
            "results": results,
            "conclusion": conclusion,
        }

        return output


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = GapDetectionCalibration(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/gap_detection_calibration.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
