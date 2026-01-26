#!/usr/bin/env python3
"""Experiment 16: Semantic Direction Discovery.

Since Exp 14 showed NO correlation between accuracy and activation geometry,
we need to find WHERE semantics actually lives.

Method: Contrastive probing
1. Collect activations for different categories
2. Find directions that SEPARATE categories (difference of means)
3. Check if these semantic directions have geometric properties
4. Test if projecting onto these directions reveals structure

The key question: Do semantic concepts live in specific directions,
and do those directions have constant-ratio structure?
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


class SemanticDirectionDiscovery:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def _capture_activations(self, prompt: str, layer_indices: List[int]) -> Dict[int, np.ndarray]:
        """Capture hidden state activations for a prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        activations = {}
        x = self.model.model.embed_tokens(input_ids)
        mx.eval(x)

        for i, layer in enumerate(self.model.model.layers):
            x = layer(x)
            mx.eval(x)
            if i in layer_indices:
                activations[i] = np.array(x.tolist())

        return activations

    def _format_question(self, question: str, choices: List[str]) -> str:
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        return prompt

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

    def collect_category_activations(
        self,
        category: str,
        questions: List[Tuple],
        layer_indices: List[int]
    ) -> Dict[int, np.ndarray]:
        """Collect mean activations for a category across all questions."""
        all_activations = {i: [] for i in layer_indices}

        for q, choices, _ in questions:
            prompt = self._format_question(q, choices)
            activations = self._capture_activations(prompt, layer_indices)
            for layer_idx, act in activations.items():
                # Use the last token position (decision point)
                last_token_act = act[0, -1, :]  # [hidden_dim]
                all_activations[layer_idx].append(last_token_act)

        # Compute mean for each layer
        mean_activations = {}
        for layer_idx in layer_indices:
            mean_activations[layer_idx] = np.mean(all_activations[layer_idx], axis=0)

        return mean_activations

    def find_separation_direction(
        self,
        cat1_activations: Dict[int, np.ndarray],
        cat2_activations: Dict[int, np.ndarray],
        layer_idx: int
    ) -> Tuple[np.ndarray, float]:
        """Find the direction that separates two categories."""
        mean1 = cat1_activations[layer_idx]
        mean2 = cat2_activations[layer_idx]

        # Difference of means = separation direction
        diff = mean1 - mean2
        norm = np.linalg.norm(diff)

        if norm > 1e-10:
            direction = diff / norm
        else:
            direction = diff

        return direction, norm

    def analyze_direction_geometry(
        self,
        direction: np.ndarray,
        all_activations: List[np.ndarray]
    ) -> Dict:
        """Analyze if projections onto a semantic direction have geometric structure."""
        # Project all activations onto this direction
        projections = np.array([act @ direction for act in all_activations])

        # The projections are scalar values - analyze their distribution
        if len(projections) > 1:
            # Check ratios between consecutive projection magnitudes
            sorted_proj = np.sort(np.abs(projections))[::-1]
            ratios = []
            for i in range(min(len(sorted_proj) - 1, 10)):
                if sorted_proj[i+1] > 1e-10:
                    ratios.append(sorted_proj[i] / sorted_proj[i+1])

            # Count constant matches in these ratios
            const_matches = 0
            for ratio in ratios:
                for const_val in CONSTANTS.values():
                    if abs(ratio - const_val) / const_val < 0.10:
                        const_matches += 1
                        break

            return {
                "n_projections": len(projections),
                "projection_std": float(np.std(projections)),
                "projection_range": float(np.max(projections) - np.min(projections)),
                "ratios": [float(r) for r in ratios],
                "const_matches": const_matches,
            }

        return {"n_projections": len(projections)}

    def analyze_direction_in_weight_space(
        self,
        direction: np.ndarray,
        layer_idx: int
    ) -> Dict:
        """Check if the semantic direction aligns with weight geometry."""
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        W = np.array(w.tolist(), dtype=np.float32)

        # SVD of weight matrix
        U, S, Vt = svd(W, full_matrices=False)

        # Project semantic direction onto weight's right singular vectors
        # direction lives in input space, so project onto V
        V = Vt.T
        alignment_scores = np.abs(direction @ V)  # [n_sv]

        # Which singular vectors does the semantic direction align with?
        top_indices = np.argsort(alignment_scores)[::-1][:10]
        top_alignments = alignment_scores[top_indices]

        # Check if the aligned SVs have constant ratios
        aligned_svs = S[top_indices]
        sv_ratios = []
        for i in range(len(aligned_svs) - 1):
            if aligned_svs[i+1] > 1e-10:
                sv_ratios.append(aligned_svs[i] / aligned_svs[i+1])

        const_matches = 0
        for ratio in sv_ratios:
            for const_val in CONSTANTS.values():
                if abs(ratio - const_val) / const_val < 0.10:
                    const_matches += 1
                    break

        return {
            "top_sv_indices": [int(i) for i in top_indices],
            "top_alignments": [float(a) for a in top_alignments],
            "aligned_sv_ratios": [float(r) for r in sv_ratios],
            "const_matches_in_aligned": const_matches,
        }

    def run_analysis(self) -> Dict:
        """Run full semantic direction discovery."""
        mid = self.n_layers // 2
        layer_indices = [mid - 2, mid - 1, mid, mid + 1, mid + 2]

        logger.info("=" * 60)
        logger.info("SEMANTIC DIRECTION DISCOVERY")
        logger.info("=" * 60)
        logger.info(f"Analyzing layers: {layer_indices}")

        # Collect activations for all categories
        logger.info("\nCollecting category activations...")
        category_activations = {}
        for category, questions in CATEGORY_QUESTIONS.items():
            logger.info(f"  {category}...")
            category_activations[category] = self.collect_category_activations(
                category, questions, layer_indices
            )

        # Find separation directions between all category pairs
        logger.info("\nFinding separation directions...")
        categories = list(CATEGORY_QUESTIONS.keys())
        separation_results = {}

        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                key = f"{cat1}_vs_{cat2}"
                logger.info(f"\n--- {key} ---")

                layer_results = {}
                for layer_idx in layer_indices:
                    direction, separation_norm = self.find_separation_direction(
                        category_activations[cat1],
                        category_activations[cat2],
                        layer_idx
                    )

                    # Collect all activations for both categories at this layer
                    all_acts = []
                    for q, choices, _ in CATEGORY_QUESTIONS[cat1]:
                        prompt = self._format_question(q, choices)
                        acts = self._capture_activations(prompt, [layer_idx])
                        all_acts.append(acts[layer_idx][0, -1, :])
                    for q, choices, _ in CATEGORY_QUESTIONS[cat2]:
                        prompt = self._format_question(q, choices)
                        acts = self._capture_activations(prompt, [layer_idx])
                        all_acts.append(acts[layer_idx][0, -1, :])

                    # Analyze this direction
                    proj_analysis = self.analyze_direction_geometry(direction, all_acts)
                    weight_analysis = self.analyze_direction_in_weight_space(direction, layer_idx)

                    layer_results[layer_idx] = {
                        "separation_norm": float(separation_norm),
                        "projection_analysis": proj_analysis,
                        "weight_alignment": weight_analysis,
                    }

                    logger.info(f"  Layer {layer_idx}: sep_norm={separation_norm:.3f}, "
                               f"weight_const_matches={weight_analysis['const_matches_in_aligned']}")

                separation_results[key] = layer_results

        # Summary analysis
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY: Do Semantic Directions Have Geometric Structure?")
        logger.info("=" * 60)

        # Count total constant matches in weight-aligned SVs
        total_matches = 0
        total_pairs = 0
        for key, layers in separation_results.items():
            for layer_idx, data in layers.items():
                total_matches += data["weight_alignment"]["const_matches_in_aligned"]
                total_pairs += 1

        avg_matches = total_matches / total_pairs if total_pairs > 0 else 0
        logger.info(f"\nAverage const matches per semantic direction: {avg_matches:.2f}")

        # Find pairs with highest geometric structure
        high_geometry_pairs = []
        for key, layers in separation_results.items():
            avg_layer_matches = np.mean([
                data["weight_alignment"]["const_matches_in_aligned"]
                for data in layers.values()
            ])
            high_geometry_pairs.append((key, avg_layer_matches))

        high_geometry_pairs.sort(key=lambda x: x[1], reverse=True)

        logger.info("\nCategory pairs ranked by geometric structure in separation direction:")
        for key, matches in high_geometry_pairs[:5]:
            logger.info(f"  {key}: {matches:.2f} avg const matches")

        # Key finding
        logger.info("\n" + "=" * 60)
        logger.info("KEY FINDING")
        logger.info("=" * 60)

        if avg_matches > 2:
            logger.info("*** SEMANTIC DIRECTIONS ALIGN WITH GEOMETRIC STRUCTURE ***")
            logger.info("The directions that separate categories DO have constant ratios")
            logger.info("in the weight singular values they align with.")
        elif avg_matches > 1:
            logger.info("** Moderate alignment between semantic and geometric structure **")
        else:
            logger.info("Weak/no alignment - semantic directions don't preferentially")
            logger.info("align with geometrically-structured weight components.")

        return {
            "layer_indices": layer_indices,
            "separation_results": separation_results,
            "summary": {
                "avg_const_matches": avg_matches,
                "ranked_pairs": high_geometry_pairs,
            },
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    analyzer = SemanticDirectionDiscovery(model, tokenizer)
    results = analyzer.run_analysis()

    # Save
    output_path = "data/semantic_direction_discovery.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
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
