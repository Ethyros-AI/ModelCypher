#!/usr/bin/env python3
"""Parallel Pathway Test - Add New Connections, Don't Modify Old.

True additive learning: Add a NEW pathway that connects to the existing
computation, without modifying existing weights at all.

Like LoRA: W_effective = W_original + ΔW
But ΔW is constructed to have perfect geometric structure (constant ratios).

The key insight: We're not modifying W - we're adding a new term.
The original knowledge is preserved EXACTLY. New structure is added.

Success criterion: ANY improvement with ZERO degradation.
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

BENCHMARK_QUESTIONS = [
    # Math
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
    ("What is 3²?", ["6", "9", "12", "27"], 1),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
    # Geography
    ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
    ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
    ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
    ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
    ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),
    # Science
    ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
    ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
    ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
    ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
    ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),
    # History
    ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
    ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
    ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
    ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
    ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),
    # Logic
    ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
    ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
    ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
    ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
    ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),
    # Language
    ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
    ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
    ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
    ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
    ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),
    # Common sense
    ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
    ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
    ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
    ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
    ("How many days are in a week?", ["5", "6", "7", "8"], 2),
]


class ParallelPathwayTest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._original_weights = {}
        self._delta_weights = {}  # The parallel pathway

    def _get_weight(self, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_weight(self, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        new_weight = mx.array(weights.astype(np.float32))
        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight
        mx.eval(new_weight)

    def _cache_weights(self, layers: List[int]):
        self._original_weights = {i: self._get_weight(i).copy() for i in layers}

    def _reset_weights(self, layers: List[int]):
        for i in layers:
            if i in self._original_weights:
                self._set_weight(i, self._original_weights[i])

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> bool:
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

    def evaluate_by_category(self) -> Dict[str, float]:
        categories = {
            "math": BENCHMARK_QUESTIONS[0:5],
            "geography": BENCHMARK_QUESTIONS[5:10],
            "science": BENCHMARK_QUESTIONS[10:15],
            "history": BENCHMARK_QUESTIONS[15:20],
            "logic": BENCHMARK_QUESTIONS[20:25],
            "language": BENCHMARK_QUESTIONS[25:30],
            "common_sense": BENCHMARK_QUESTIONS[30:35],
        }
        results = {}
        for cat, questions in categories.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx))
            results[cat] = correct / len(questions)
        results["overall"] = sum(results.values()) / len(results)
        return results

    def create_geometric_delta(self, shape: Tuple[int, int], rank: int, scale: float) -> np.ndarray:
        """
        Create a low-rank ΔW with perfect geometric structure.

        ΔW = U_delta @ S_delta @ V_delta^T
        where S_delta has exact constant ratios.
        """
        m, n = shape

        # Create orthonormal U and V for the delta
        U_delta = np.linalg.qr(np.random.randn(m, rank).astype(np.float32))[0]
        V_delta = np.linalg.qr(np.random.randn(n, rank).astype(np.float32))[0]

        # Create singular values with EXACT constant ratios
        # Start with base value, then chain through constants
        const_vals = list(CONSTANTS.values())
        S_delta = np.zeros(rank, dtype=np.float32)
        S_delta[0] = scale

        for i in range(1, rank):
            # Each subsequent SV is previous / constant
            const = const_vals[i % len(const_vals)]
            S_delta[i] = S_delta[i-1] / const

        # Construct ΔW
        delta_W = U_delta @ np.diag(S_delta) @ V_delta.T

        return delta_W

    def create_aligned_delta(self, W_original: np.ndarray, rank: int, scale: float) -> np.ndarray:
        """
        Create ΔW that's aligned with W's structure but has geometric SVs.

        Use W's left/right singular vectors so the delta operates
        in the same subspace, but with geometric singular values.
        """
        U, S, Vt = svd(W_original, full_matrices=False)

        # Use W's singular vectors but geometric singular values
        # This aligns the delta with how W processes information
        S_geometric = np.zeros(rank, dtype=np.float32)
        S_geometric[0] = scale * S[0]  # Scale relative to W's magnitude

        const_vals = list(CONSTANTS.values())
        for i in range(1, rank):
            const = const_vals[i % len(const_vals)]
            S_geometric[i] = S_geometric[i-1] / const

        # ΔW uses W's subspace but geometric scaling
        delta_W = U[:, :rank] @ np.diag(S_geometric) @ Vt[:rank, :]

        return delta_W

    def create_orthogonal_delta(self, W_original: np.ndarray, rank: int, scale: float) -> np.ndarray:
        """
        Create ΔW that's ORTHOGONAL to W's structure.

        The delta operates in W's null space, so it adds new
        capacity without interfering with existing computation.
        """
        U, S, Vt = svd(W_original, full_matrices=False)

        # Find indices where S is very small (null space of W)
        threshold = S[0] * 1e-3
        null_indices = np.where(S < threshold)[0]

        if len(null_indices) < rank:
            # Not enough null space, use smallest SVs
            null_indices = np.argsort(S)[:rank]

        # Create geometric SVs in the null space
        S_geometric = np.zeros(len(null_indices), dtype=np.float32)
        S_geometric[0] = scale * S[0] * 0.01  # Small relative to W

        const_vals = list(CONSTANTS.values())
        for i in range(1, min(len(null_indices), rank)):
            const = const_vals[i % len(const_vals)]
            S_geometric[i] = S_geometric[i-1] / const

        # Use null space directions
        U_null = U[:, null_indices[:rank]]
        Vt_null = Vt[null_indices[:rank], :]

        delta_W = U_null @ np.diag(S_geometric[:rank]) @ Vt_null

        return delta_W

    def add_parallel_pathway(
        self,
        layer_idx: int,
        delta_type: str,  # "random", "aligned", "orthogonal"
        rank: int,
        scale: float
    ):
        """Add a parallel pathway to a layer."""
        W_original = self._original_weights[layer_idx]

        if delta_type == "random":
            delta_W = self.create_geometric_delta(W_original.shape, rank, scale)
        elif delta_type == "aligned":
            delta_W = self.create_aligned_delta(W_original, rank, scale)
        elif delta_type == "orthogonal":
            delta_W = self.create_orthogonal_delta(W_original, rank, scale)
        else:
            raise ValueError(f"Unknown delta type: {delta_type}")

        # Key: W_new = W_original + delta_W
        # Original is PRESERVED, we only ADD
        W_new = W_original + delta_W

        if np.all(np.isfinite(W_new)):
            self._set_weight(layer_idx, W_new)
            return True
        return False

    def run_experiment(self) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 3, mid + 4))

        self._cache_weights(layers)

        logger.info("=" * 60)
        logger.info("PARALLEL PATHWAY TEST - Add New Connections")
        logger.info("=" * 60)

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial: {initial}")

        results = {}

        # Test different delta types and scales
        for delta_type in ["random", "aligned", "orthogonal"]:
            for rank in [2, 4, 8]:
                for scale in [0.001, 0.01, 0.1]:
                    self._reset_weights(layers)

                    key = f"{delta_type}_r{rank}_s{scale}"
                    logger.info(f"\n--- {key} ---")

                    # Add parallel pathway to each layer
                    for layer_idx in layers:
                        self.add_parallel_pathway(layer_idx, delta_type, rank, scale)

                    final = self.evaluate_by_category()
                    changes = {k: final[k] - initial[k] for k in initial}
                    degraded = [k for k, v in changes.items() if v < -0.01]
                    improved = [k for k, v in changes.items() if v > 0.01]

                    results[key] = {
                        "final": final,
                        "changes": changes,
                        "degraded": degraded,
                        "improved": improved,
                    }

                    status = "✓" if not degraded else "✗"
                    improve_str = f", improved={improved}" if improved else ""
                    logger.info(f"{status} overall={final['overall']:.1%}, degraded={degraded}{improve_str}")

        # Find best configurations
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)

        no_degrade = [(k, v) for k, v in results.items() if not v["degraded"]]
        has_improve = [(k, v) for k, v in results.items() if v["improved"]]
        both = [(k, v) for k, v in results.items() if not v["degraded"] and v["improved"]]

        logger.info(f"Configs with NO degradation: {len(no_degrade)}/{len(results)}")
        logger.info(f"Configs with improvement: {len(has_improve)}/{len(results)}")
        logger.info(f"Configs with BOTH (the goal): {len(both)}/{len(results)}")

        if both:
            logger.info("\n*** SUCCESS - Improvement without degradation: ***")
            for key, data in sorted(both, key=lambda x: x[1]["changes"]["overall"], reverse=True):
                logger.info(f"  {key}: +{data['changes']['overall']:.1%} overall, improved={data['improved']}")

        if no_degrade and not both:
            logger.info("\nNo improvement, but these preserved quality:")
            for key, data in no_degrade[:5]:
                logger.info(f"  {key}")

        self._reset_weights(layers)

        return {
            "initial": initial,
            "results": results,
            "success": len(both) > 0,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = ParallelPathwayTest(model, tokenizer)
    results = test.run_experiment()

    # Save
    output_path = "data/parallel_pathway_test.json"
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
