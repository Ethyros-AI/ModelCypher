#!/usr/bin/env python3
"""Experiment 14: Activation Geometry Analysis by Category.

Question: Is the 3.7x activation amplification consistent across categories?
Do high-performing categories have MORE constant ratios in their activations?

Method:
1. Run model on category-specific questions, capture hidden states
2. Compute SVD of activations for each category
3. Compare: geometry vs accuracy correlation

If correlation exists → activations carry semantic-geometric link
If no correlation → geometry is noise/epiphenomenal
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CONSTANTS = {
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
    "phi": (1 + np.sqrt(5)) / 2,
    "1/phi": 2 / (1 + np.sqrt(5)),
    "sqrt2": np.sqrt(2),
    "1/sqrt2": 1 / np.sqrt(2),
    "sqrt3": np.sqrt(3),
}

CATEGORY_QUESTIONS = {
    "math": [
        ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
        ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
        ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
        ("What is 3²?", ["6", "9", "12", "27"], 1),
        ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
    ],
    "geography": [
        ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
        ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
        ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
        ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
        ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),
    ],
    "science": [
        ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
        ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
        ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
        ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
        ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),
    ],
    "history": [
        ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
        ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
        ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
        ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
        ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),
    ],
    "logic": [
        ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
        ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
        ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
        ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
        ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),
    ],
    "language": [
        ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
        ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
        ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
        ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
        ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),
    ],
    "common_sense": [
        ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
        ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
        ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
        ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
        ("How many days are in a week?", ["5", "6", "7", "8"], 2),
    ],
}


class ActivationGeometryAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._activation_cache = {}

    def _register_hooks(self, layer_indices: List[int]):
        """Register hooks to capture activations."""
        import mlx.core as mx

        self._activation_cache = {i: [] for i in layer_indices}
        self._hooks = []

        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]

            # We'll capture the output of the layer (post-attention + MLP)
            def make_hook(idx):
                def hook(module, args, output):
                    # output is the hidden state after this layer
                    mx.eval(output)
                    self._activation_cache[idx].append(np.array(output.tolist()))
                return hook

            # MLX doesn't have standard hooks, so we'll capture manually
            # For now, we'll run forward passes and intercept

        return self._activation_cache

    def _capture_activations(self, prompt: str, layer_indices: List[int]) -> Dict[int, np.ndarray]:
        """Capture hidden state activations for a prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Run through each layer and capture states
        # We need to access intermediate states
        activations = {}

        # Get embedding
        x = self.model.model.embed_tokens(input_ids)
        mx.eval(x)

        for i, layer in enumerate(self.model.model.layers):
            x = layer(x)
            mx.eval(x)
            if i in layer_indices:
                activations[i] = np.array(x.tolist())

        return activations

    def _count_constant_ratios(self, S: np.ndarray, proximity: float = 0.05) -> Dict[str, int]:
        """Count constant ratio matches in singular values."""
        counts = {k: 0 for k in CONSTANTS}

        for i in range(min(len(S) - 1, 30)):
            for j in range(i + 1, min(len(S), i + 8)):
                if S[j] > 1e-10:
                    ratio = S[i] / S[j]
                    for const_name, const_val in CONSTANTS.items():
                        if abs(ratio - const_val) / const_val < proximity:
                            counts[const_name] += 1
                            break

        return counts

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> bool:
        """Evaluate a single question."""
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

        scores = [float(next_logits[t].item()) for t in choice_tokens]
        return int(np.argmax(scores)) == correct_idx

    def analyze_category(
        self,
        category: str,
        questions: List[Tuple],
        layer_indices: List[int]
    ) -> Dict:
        """Analyze activation geometry for a single category."""
        import mlx.core as mx

        # First evaluate accuracy
        correct = 0
        for q, choices, correct_idx in questions:
            if self._evaluate_question(q, choices, correct_idx):
                correct += 1
        accuracy = correct / len(questions)

        # Collect activations for all questions
        all_activations = {i: [] for i in layer_indices}

        for q, choices, correct_idx in questions:
            prompt = f"Question: {q}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer:"

            activations = self._capture_activations(prompt, layer_indices)
            for layer_idx, act in activations.items():
                all_activations[layer_idx].append(act)

        # Analyze geometry per layer
        layer_geometry = {}
        for layer_idx in layer_indices:
            # Concatenate all activations for this layer
            acts = np.concatenate([a.reshape(-1, a.shape[-1]) for a in all_activations[layer_idx]], axis=0)

            # SVD of activations
            try:
                _, S, _ = svd(acts, full_matrices=False)
                counts = self._count_constant_ratios(S)
                total_matches = sum(counts.values())
                layer_geometry[layer_idx] = {
                    "total_matches": total_matches,
                    "by_constant": counts,
                    "n_singular_values": len(S),
                }
            except Exception as e:
                layer_geometry[layer_idx] = {"error": str(e)}

        total_geometry = sum(
            lg.get("total_matches", 0) for lg in layer_geometry.values()
        )

        return {
            "accuracy": accuracy,
            "total_geometry": total_geometry,
            "layer_geometry": layer_geometry,
        }

    def run_analysis(self) -> Dict:
        """Run full analysis across all categories."""
        mid = self.n_layers // 2
        layer_indices = list(range(mid - 3, mid + 4))  # Middle 7 layers

        logger.info("=" * 60)
        logger.info("ACTIVATION GEOMETRY ANALYSIS BY CATEGORY")
        logger.info("=" * 60)
        logger.info(f"Analyzing layers: {layer_indices}")

        results = {}
        category_data = []

        for category, questions in CATEGORY_QUESTIONS.items():
            logger.info(f"\n--- {category.upper()} ---")
            result = self.analyze_category(category, questions, layer_indices)
            results[category] = result

            logger.info(f"  Accuracy: {result['accuracy']:.0%}")
            logger.info(f"  Geometry (total matches): {result['total_geometry']}")

            category_data.append({
                "category": category,
                "accuracy": result["accuracy"],
                "geometry": result["total_geometry"],
            })

        # Compute correlation
        accuracies = [d["accuracy"] for d in category_data]
        geometries = [d["geometry"] for d in category_data]

        # Pearson correlation
        if len(set(accuracies)) > 1 and len(set(geometries)) > 1:
            corr = np.corrcoef(accuracies, geometries)[0, 1]
        else:
            corr = 0.0

        logger.info("\n" + "=" * 60)
        logger.info("CORRELATION ANALYSIS")
        logger.info("=" * 60)

        logger.info("\nCategory | Accuracy | Geometry")
        logger.info("-" * 35)
        for d in sorted(category_data, key=lambda x: x["accuracy"], reverse=True):
            logger.info(f"{d['category']:<12} | {d['accuracy']:>6.0%} | {d['geometry']:>8}")

        logger.info(f"\nPearson correlation (accuracy vs geometry): {corr:.3f}")

        if corr > 0.5:
            logger.info("*** STRONG POSITIVE CORRELATION ***")
            logger.info("High-accuracy categories have more geometric structure!")
        elif corr > 0.2:
            logger.info("** Moderate positive correlation **")
        elif corr < -0.5:
            logger.info("*** STRONG NEGATIVE CORRELATION ***")
            logger.info("High-accuracy categories have LESS geometric structure!")
        else:
            logger.info("Weak or no correlation between accuracy and geometry")

        return {
            "categories": results,
            "correlation": corr,
            "summary": category_data,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    analyzer = ActivationGeometryAnalyzer(model, tokenizer)
    results = analyzer.run_analysis()

    # Save
    output_path = "data/activation_geometry_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, set)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
