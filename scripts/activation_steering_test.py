#!/usr/bin/env python3
"""Experiment 15: Inference-Time Activation Steering.

Exp 16 showed semantic directions align with geometric structure.
Now test: Can we modify ACTIVATIONS during inference to improve quality?

Key insight: Instead of modifying weights (which caused degradation-before-improvement),
modify the hidden states during forward pass. Activations are input-specific,
so different inputs get different steering.

Methods:
1. Geometric steering: Nudge activation SVD ratios toward constants
2. Semantic-aware steering: Project onto separation directions and enhance
3. Residual scaling: Scale the contribution of geometrically-aligned components
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import partial

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


class ActivationSteeringTest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._steering_config = None
        self._hooked_layers = {}

    def _steer_activation_geometric(self, x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Steer activation by nudging SVD ratios toward constants.

        x: [batch, seq, hidden] activation tensor
        alpha: interpolation strength (0 = no change, 1 = full alignment)
        """
        original_shape = x.shape
        # Reshape to [seq, hidden] for SVD
        x_flat = x.reshape(-1, x.shape[-1])

        try:
            U, S, Vt = svd(x_flat, full_matrices=False)

            if len(S) < 2:
                return x

            S_modified = S.copy()
            min_sv = S[0] * 1e-6

            # Find near-constant ratios and nudge toward exact
            for i in range(min(len(S) - 1, 10)):
                for j in range(i + 1, min(len(S), i + 5)):
                    if S[j] > max(1e-10, min_sv):
                        ratio = S[i] / S[j]
                        for const_val in CONSTANTS.values():
                            if abs(ratio - const_val) / const_val < 0.15:  # Within 15%
                                # Nudge toward constant
                                target = const_val * S_modified[j]
                                S_modified[i] = alpha * target + (1 - alpha) * S_modified[i]
                                break

            # Reconstruct
            x_steered = U @ np.diag(S_modified) @ Vt
            x_steered = x_steered.reshape(original_shape)

            if np.all(np.isfinite(x_steered)):
                return x_steered
        except Exception:
            pass

        return x

    def _steer_activation_scale_geometric(self, x: np.ndarray, scale: float = 1.1) -> np.ndarray:
        """
        Scale up components that already have geometric structure.

        Find which singular value pairs match constants, amplify those components.
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        try:
            U, S, Vt = svd(x_flat, full_matrices=False)

            if len(S) < 2:
                return x

            # Find which indices participate in constant ratios
            geometric_indices = set()
            for i in range(min(len(S) - 1, 15)):
                for j in range(i + 1, min(len(S), i + 5)):
                    if S[j] > 1e-10:
                        ratio = S[i] / S[j]
                        for const_val in CONSTANTS.values():
                            if abs(ratio - const_val) / const_val < 0.10:
                                geometric_indices.add(i)
                                geometric_indices.add(j)
                                break

            if not geometric_indices:
                return x

            # Scale up those components
            S_modified = S.copy()
            for idx in geometric_indices:
                S_modified[idx] *= scale

            x_steered = U @ np.diag(S_modified) @ Vt
            x_steered = x_steered.reshape(original_shape)

            if np.all(np.isfinite(x_steered)):
                return x_steered
        except Exception:
            pass

        return x

    def _steer_activation_suppress_noise(self, x: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Suppress components that don't participate in geometric structure.

        Keep geometric components, reduce noise components.
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        try:
            U, S, Vt = svd(x_flat, full_matrices=False)

            if len(S) < 2:
                return x

            # Find which indices are NOT geometric
            geometric_indices = set()
            for i in range(min(len(S) - 1, 15)):
                for j in range(i + 1, min(len(S), i + 5)):
                    if S[j] > 1e-10:
                        ratio = S[i] / S[j]
                        for const_val in CONSTANTS.values():
                            if abs(ratio - const_val) / const_val < 0.10:
                                geometric_indices.add(i)
                                geometric_indices.add(j)
                                break

            # Suppress non-geometric components
            S_modified = S.copy()
            for idx in range(len(S)):
                if idx not in geometric_indices:
                    S_modified[idx] *= (1 - threshold)

            x_steered = U @ np.diag(S_modified) @ Vt
            x_steered = x_steered.reshape(original_shape)

            if np.all(np.isfinite(x_steered)):
                return x_steered
        except Exception:
            pass

        return x

    def _evaluate_with_steering(
        self,
        question: str,
        choices: List[str],
        correct_idx: int,
        steering_fn: Optional[callable],
        steering_layers: List[int]
    ) -> bool:
        """Evaluate a question with optional activation steering."""
        import mlx.core as mx

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Manual forward pass with steering
        # LFM2 architecture: embed -> layers -> embedding_norm -> tied_embedding_logits
        x = self.model.model.embed_tokens(input_ids)
        mx.eval(x)

        # Create masks for attention/conv layers
        from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
        attn_mask = create_attention_mask(x, None)
        conv_mask = create_ssm_mask(x, None)

        for layer_idx, layer in enumerate(self.model.model.layers):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            x = layer(x, mask, cache=None)
            mx.eval(x)

            if steering_fn is not None and layer_idx in steering_layers:
                # Convert to numpy, steer, convert back
                x_np = np.array(x.tolist(), dtype=np.float32)
                x_np = steering_fn(x_np)
                x = mx.array(x_np)
                mx.eval(x)

        # Final norm (LFM2 uses embedding_norm)
        x = self.model.model.embedding_norm(x)
        # Logits via tied embeddings
        logits = self.model.model.embed_tokens.as_linear(x)
        mx.eval(logits)

        next_logits = logits[0, -1, :]

        # Get choice token scores
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

    def evaluate_by_category(
        self,
        steering_fn: Optional[callable] = None,
        steering_layers: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Evaluate all categories with optional steering."""
        if steering_layers is None:
            steering_layers = []

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
            correct = sum(
                1 for q, c, idx in questions
                if self._evaluate_with_steering(q, c, idx, steering_fn, steering_layers)
            )
            results[cat] = correct / len(questions)
        results["overall"] = sum(results.values()) / len(results)
        return results

    def run_experiment(self) -> Dict:
        """Test different steering methods."""
        mid = self.n_layers // 2
        steering_layers = list(range(mid - 2, mid + 3))  # Middle 5 layers

        logger.info("=" * 60)
        logger.info("ACTIVATION STEERING TEST")
        logger.info("=" * 60)
        logger.info(f"Steering layers: {steering_layers}")

        # Baseline (no steering)
        logger.info("\n--- Baseline (no steering) ---")
        baseline = self.evaluate_by_category(None, [])
        logger.info(f"Baseline: {baseline}")

        results = {"baseline": baseline}

        # Test different steering methods
        steering_configs = [
            ("geometric_a0.05", partial(self._steer_activation_geometric, alpha=0.05)),
            ("geometric_a0.10", partial(self._steer_activation_geometric, alpha=0.10)),
            ("geometric_a0.20", partial(self._steer_activation_geometric, alpha=0.20)),
            ("scale_1.05", partial(self._steer_activation_scale_geometric, scale=1.05)),
            ("scale_1.10", partial(self._steer_activation_scale_geometric, scale=1.10)),
            ("scale_1.20", partial(self._steer_activation_scale_geometric, scale=1.20)),
            ("suppress_0.01", partial(self._steer_activation_suppress_noise, threshold=0.01)),
            ("suppress_0.05", partial(self._steer_activation_suppress_noise, threshold=0.05)),
            ("suppress_0.10", partial(self._steer_activation_suppress_noise, threshold=0.10)),
        ]

        for name, fn in steering_configs:
            logger.info(f"\n--- {name} ---")
            result = self.evaluate_by_category(fn, steering_layers)

            changes = {k: result[k] - baseline[k] for k in baseline}
            degraded = [k for k, v in changes.items() if v < -0.01]
            improved = [k for k, v in changes.items() if v > 0.01]

            results[name] = {
                "scores": result,
                "changes": changes,
                "degraded": degraded,
                "improved": improved,
            }

            status = "+" if improved and not degraded else ("~" if not degraded else "-")
            logger.info(f"{status} overall={result['overall']:.1%}, degraded={degraded}, improved={improved}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)

        no_degrade = [(k, v) for k, v in results.items()
                      if k != "baseline" and not v.get("degraded", [])]
        has_improve = [(k, v) for k, v in results.items()
                       if k != "baseline" and v.get("improved", [])]
        both = [(k, v) for k, v in results.items()
                if k != "baseline" and not v.get("degraded", []) and v.get("improved", [])]

        logger.info(f"Methods with NO degradation: {len(no_degrade)}/{len(steering_configs)}")
        logger.info(f"Methods with improvement: {len(has_improve)}/{len(steering_configs)}")
        logger.info(f"Methods with BOTH: {len(both)}/{len(steering_configs)}")

        if both:
            logger.info("\n*** SUCCESS - Improvement without degradation: ***")
            for name, data in sorted(both, key=lambda x: x[1]["changes"]["overall"], reverse=True):
                logger.info(f"  {name}: +{data['changes']['overall']:.1%} overall")
        else:
            logger.info("\nNo method achieved improvement without degradation")
            if no_degrade:
                logger.info("Methods that preserved quality (but no improvement):")
                for name, _ in no_degrade:
                    logger.info(f"  {name}")

        return {
            "steering_layers": steering_layers,
            "results": results,
            "success": len(both) > 0,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = ActivationSteeringTest(model, tokenizer)
    results = test.run_experiment()

    # Save
    output_path = "data/activation_steering_test.json"
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
