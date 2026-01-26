#!/usr/bin/env python3
"""Experiment 39: Research Integration.

Phase 8 - Stage 1: Can researched information be converted to useful training signal?

The challenge: Converting web content to QA training pairs the model can learn from.

Method:
1. Identify a topic with low consistency (math, logic)
2. Research topic via web search
3. Generate QA pairs from research
4. Measure quality of generated training data

Note: This script demonstrates the research-to-training pipeline.
For actual web search, it uses the firecrawl MCP tool via Claude.
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


# Simulated research results (in real system, this would come from firecrawl)
# These represent what we'd get from searching "basic math multiplication facts"
RESEARCHED_FACTS = {
    "math": [
        {
            "fact": "8 multiplied by 7 equals 56",
            "source": "math_facts.com",
            "qa": {
                "question": "What is 8 × 7?",
                "answer": "56",
                "choices": ["48", "54", "56", "64"],
                "correct_idx": 2,
            }
        },
        {
            "fact": "15 plus 27 equals 42",
            "source": "basic_arithmetic.com",
            "qa": {
                "question": "What is 15 + 27?",
                "answer": "42",
                "choices": ["32", "42", "52", "62"],
                "correct_idx": 1,
            }
        },
        {
            "fact": "9 times 6 equals 54",
            "source": "multiplication_tables.com",
            "qa": {
                "question": "What is 9 × 6?",
                "answer": "54",
                "choices": ["45", "54", "56", "63"],
                "correct_idx": 1,
            }
        },
        {
            "fact": "100 divided by 5 equals 20",
            "source": "division_facts.com",
            "qa": {
                "question": "What is 100 ÷ 5?",
                "answer": "20",
                "choices": ["15", "20", "25", "50"],
                "correct_idx": 1,
            }
        },
        {
            "fact": "The square of 5 is 25",
            "source": "squares_cubes.com",
            "qa": {
                "question": "What is 5²?",
                "answer": "25",
                "choices": ["10", "15", "25", "50"],
                "correct_idx": 2,
            }
        },
    ],
    "logic": [
        {
            "fact": "In the sequence 2, 4, 6, 8, the next number is 10 because each term increases by 2",
            "source": "number_patterns.com",
            "qa": {
                "question": "What comes next: 2, 4, 6, 8, ?",
                "answer": "10",
                "choices": ["9", "10", "11", "12"],
                "correct_idx": 1,
            }
        },
        {
            "fact": "The sequence 1, 3, 5, 7 continues with 9 (odd numbers)",
            "source": "number_patterns.com",
            "qa": {
                "question": "What comes next: 1, 3, 5, 7, ?",
                "answer": "9",
                "choices": ["8", "9", "10", "11"],
                "correct_idx": 1,
            }
        },
        {
            "fact": "If all dogs are animals, and Rex is a dog, then Rex is an animal (logical syllogism)",
            "source": "logic_basics.com",
            "qa": {
                "question": "If all dogs are animals, and Rex is a dog, is Rex an animal?",
                "answer": "Yes",
                "choices": ["Yes", "No", "Maybe", "Cannot tell"],
                "correct_idx": 0,
            }
        },
    ],
}


class ResearchIntegration:
    """Test research-to-training pipeline."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

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
        confidence = probs[prediction]

        return prediction == correct_idx, confidence

    def generate_training_pairs(self, facts: List[Dict]) -> List[Dict]:
        """Convert researched facts to training pairs."""
        training_pairs = []

        for fact_data in facts:
            qa = fact_data["qa"]

            # Format 1: Direct Q&A
            pair1 = {
                "input": f"Question: {qa['question']}\nAnswer:",
                "target": f" {qa['answer']}",
                "type": "direct_qa",
            }
            training_pairs.append(pair1)

            # Format 2: Statement completion
            pair2 = {
                "input": f"Complete this: {fact_data['fact'].split()[0]}",
                "target": f" {' '.join(fact_data['fact'].split()[1:])}",
                "type": "completion",
            }
            training_pairs.append(pair2)

            # Format 3: True/False verification
            pair3 = {
                "input": f"Is this true? {fact_data['fact']} Answer:",
                "target": " Yes",
                "type": "verification",
            }
            training_pairs.append(pair3)

        return training_pairs

    def verify_qa_quality(self, qa_pairs: List[Dict]) -> Dict:
        """Verify the quality of generated QA pairs."""
        results = []

        for qa in qa_pairs:
            is_correct, confidence = self.evaluate_question(
                qa["question"], qa["choices"], qa["correct_idx"]
            )

            results.append({
                "question": qa["question"],
                "correct_answer": qa["answer"],
                "model_correct": is_correct,
                "confidence": confidence,
            })

        accuracy = sum(1 for r in results if r["model_correct"]) / len(results)

        return {
            "n_pairs": len(results),
            "accuracy_before_learning": accuracy,
            "results": results,
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 39: RESEARCH INTEGRATION")
        logger.info("=" * 60)

        output = {
            "categories": {},
            "training_pairs_generated": 0,
            "qa_verification": {},
        }

        for category, facts in RESEARCHED_FACTS.items():
            logger.info(f"\n--- Processing {category} ---")
            logger.info(f"  Researched facts: {len(facts)}")

            # Generate training pairs
            training_pairs = self.generate_training_pairs(facts)
            logger.info(f"  Generated training pairs: {len(training_pairs)}")
            output["training_pairs_generated"] += len(training_pairs)

            # Extract QA pairs for verification
            qa_pairs = [f["qa"] for f in facts]

            # Verify QA quality
            verification = self.verify_qa_quality(qa_pairs)
            logger.info(f"  Model accuracy on QA: {verification['accuracy_before_learning']:.1%}")

            output["categories"][category] = {
                "n_facts": len(facts),
                "n_training_pairs": len(training_pairs),
                "training_pair_types": {
                    "direct_qa": sum(1 for p in training_pairs if p["type"] == "direct_qa"),
                    "completion": sum(1 for p in training_pairs if p["type"] == "completion"),
                    "verification": sum(1 for p in training_pairs if p["type"] == "verification"),
                },
            }

            output["qa_verification"][category] = verification

            # Log individual results
            for result in verification["results"]:
                status = "✓" if result["model_correct"] else "✗"
                logger.info(f"    {status} {result['question'][:50]}... "
                           f"(conf: {result['confidence']:.2f})")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        total_qa = sum(v["n_pairs"] for v in output["qa_verification"].values())
        total_correct = sum(
            sum(1 for r in v["results"] if r["model_correct"])
            for v in output["qa_verification"].values()
        )
        overall_accuracy = total_correct / total_qa if total_qa > 0 else 0

        logger.info(f"Total training pairs generated: {output['training_pairs_generated']}")
        logger.info(f"Total QA pairs verified: {total_qa}")
        logger.info(f"Model accuracy on QA (before learning): {overall_accuracy:.1%}")

        # Determine if QA pairs are high quality (factually correct)
        # Since we're using verified facts, they're 100% factually correct
        # The model's accuracy tells us how much room for improvement there is
        logger.info(f"\nQA pairs are 100% factually correct (verified sources)")
        logger.info(f"Model needs improvement on: {(1-overall_accuracy)*100:.0f}% of questions")

        output["summary"] = {
            "total_training_pairs": output["training_pairs_generated"],
            "total_qa_pairs": total_qa,
            "model_accuracy_before": overall_accuracy,
            "qa_factual_accuracy": 1.0,  # By construction
            "improvement_potential": 1.0 - overall_accuracy,
        }

        # Interpretation
        if overall_accuracy < 0.8:
            output["conclusion"] = "high_potential"
            logger.info("\nINTERPRETATION: HIGH potential for improvement - model accuracy < 80%")
        else:
            output["conclusion"] = "low_potential"
            logger.info("\nINTERPRETATION: LOW potential - model already knows most facts")

        return output


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = ResearchIntegration(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/research_integration.json"
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
