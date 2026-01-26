#!/usr/bin/env python3
"""Residual Connection Test - Connect Through The Stream.

The residual stream is how layers communicate. It's the actual "connection."

x_out = x_in + MLP(LayerNorm(x_in))

What if geometric structure should be in how we BLEND the residual,
not in the MLP weights themselves?

Method 1: Scale the residual with geometric ratios
  x_out = α * x_in + (1-α) * MLP(x_in)  where α = π/e / (1 + π/e)

Method 2: Add a geometric "router" that modulates layer contributions
  x_out = router(x_in) * x_in + (1 - router(x_in)) * MLP(x_in)

Method 3: Cross-layer connections with geometric weights
  x_layer_n = Σ (geometric_weight_i * x_layer_i)

The key: Connection is about information FLOW, not static weights.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from functools import partial

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Geometric blending coefficients derived from constants
PI_E = np.pi / np.e
PHI = (1 + np.sqrt(5)) / 2
SQRT2 = np.sqrt(2)

# Blending ratios: α = const / (1 + const) maps [0,∞] → [0,1]
BLEND_RATIOS = {
    "pi_e": PI_E / (1 + PI_E),       # ≈ 0.536
    "phi": PHI / (1 + PHI),           # ≈ 0.618  (golden ratio in [0,1]!)
    "sqrt2": SQRT2 / (1 + SQRT2),     # ≈ 0.586
    "equal": 0.5,                      # baseline
    "residual_heavy": 0.7,             # favor skip connection
    "mlp_heavy": 0.3,                  # favor MLP output
}


BENCHMARK_QUESTIONS = [
    ("What is 15 + 27?", ["32", "42", "52", "62"], 1),
    ("What is 8 × 7?", ["48", "54", "56", "64"], 2),
    ("What is 100 ÷ 4?", ["20", "25", "30", "40"], 1),
    ("What is 3²?", ["6", "9", "12", "27"], 1),
    ("What is 50% of 80?", ["30", "40", "50", "60"], 1),
    ("What is the capital of Japan?", ["Seoul", "Beijing", "Tokyo", "Bangkok"], 2),
    ("Which continent is Brazil in?", ["Africa", "Europe", "Asia", "South America"], 3),
    ("What is the largest ocean?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
    ("Which country has the most people?", ["USA", "India", "China", "Russia"], 2),
    ("What river flows through Egypt?", ["Amazon", "Nile", "Yangtze", "Mississippi"], 1),
    ("What planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1),
    ("What gas do plants produce?", ["Carbon dioxide", "Nitrogen", "Oxygen", "Hydrogen"], 2),
    ("What is H2O?", ["Salt", "Sugar", "Water", "Oil"], 2),
    ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
    ("What is the hardest natural substance?", ["Gold", "Iron", "Diamond", "Platinum"], 2),
    ("Who was the first US President?", ["Lincoln", "Jefferson", "Washington", "Adams"], 2),
    ("In what year did WW2 end?", ["1943", "1944", "1945", "1946"], 2),
    ("What ancient wonder was in Egypt?", ["Colosseum", "Pyramids", "Parthenon", "Great Wall"], 1),
    ("Who wrote Romeo and Juliet?", ["Dickens", "Austen", "Shakespeare", "Hemingway"], 2),
    ("What empire built the Colosseum?", ["Greek", "Persian", "Roman", "Ottoman"], 2),
    ("If all cats have tails, and Fluffy is a cat, does Fluffy have a tail?", ["Yes", "No", "Maybe", "Unknown"], 0),
    ("What comes next: 2, 4, 6, 8, ?", ["9", "10", "11", "12"], 1),
    ("If A > B and B > C, is A > C?", ["Yes", "No", "Sometimes", "Cannot tell"], 0),
    ("Which is heavier: 1kg of feathers or 1kg of steel?", ["Feathers", "Steel", "Same weight", "Cannot compare"], 2),
    ("If today is Monday, what day was yesterday?", ["Tuesday", "Sunday", "Saturday", "Wednesday"], 1),
    ("What is the opposite of 'hot'?", ["Warm", "Cold", "Cool", "Mild"], 1),
    ("Which word is a verb?", ["Beautiful", "Running", "Quickly", "Happiness"], 1),
    ("What is the plural of 'child'?", ["Childs", "Children", "Childes", "Childern"], 1),
    ("Which is a synonym for 'happy'?", ["Sad", "Angry", "Joyful", "Tired"], 2),
    ("What punctuation ends a question?", ["Period", "Comma", "Question mark", "Exclamation"], 2),
    ("What do you use to cut paper?", ["Hammer", "Scissors", "Spoon", "Brush"], 1),
    ("Where do fish live?", ["Trees", "Deserts", "Water", "Mountains"], 2),
    ("What meal do people eat in the morning?", ["Lunch", "Dinner", "Breakfast", "Snack"], 2),
    ("What color is grass usually?", ["Blue", "Red", "Green", "Yellow"], 2),
    ("How many days are in a week?", ["5", "6", "7", "8"], 2),
]


class ResidualConnectionTest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._original_forwards = {}
        self._hooks = []

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

    def modify_layer_residual_scale(self, layer_idx: int, residual_alpha: float):
        """
        Modify how a layer blends residual and MLP output.

        Standard: x_out = x_in + MLP(LN(x_in))
        Modified: x_out = α * x_in + (1-α) * MLP(LN(x_in))

        Note: This modifies the MLP output scale, which effectively
        changes the residual contribution ratio.
        """
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp

        # Get the down projection weight and scale it
        if hasattr(mlp, 'down_proj'):
            w = mlp.down_proj.weight
        elif hasattr(mlp, 'w2'):
            w = mlp.w2.weight
        else:
            return False

        # Scale the output weight by (1 - alpha) / 1.0
        # This effectively changes x_out from:
        #   x_in + MLP_output
        # to:
        #   x_in + (1-α)/1 * MLP_output
        # which approximates:
        #   α * x_in + (1-α) * (x_in + MLP_output)
        mx.eval(w)
        w_np = np.array(w.tolist(), dtype=np.float32)

        # Save original for reset
        if layer_idx not in self._original_forwards:
            self._original_forwards[layer_idx] = w_np.copy()

        # Scale down MLP contribution (effectively increases residual importance)
        w_scaled = w_np * (1 - residual_alpha + 0.5)  # Shift so alpha=0.5 is neutral

        new_w = mx.array(w_scaled.astype(np.float32))
        if hasattr(mlp, 'down_proj'):
            mlp.down_proj.weight = new_w
        else:
            mlp.w2.weight = new_w
        mx.eval(new_w)

        return True

    def reset_layers(self, layers: List[int]):
        """Reset modified layers to original weights."""
        import mlx.core as mx

        for layer_idx in layers:
            if layer_idx in self._original_forwards:
                layer = self.model.model.layers[layer_idx]
                mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp

                w = mx.array(self._original_forwards[layer_idx])
                if hasattr(mlp, 'down_proj'):
                    mlp.down_proj.weight = w
                else:
                    mlp.w2.weight = w
                mx.eval(w)

        self._original_forwards = {}

    def test_geometric_blend_pattern(self, layers: List[int], pattern: str) -> Dict:
        """
        Apply a geometric blending pattern across layers.

        Patterns:
        - "constant": Same α for all layers
        - "fibonacci": α values follow Fibonacci ratios
        - "geometric": α values decay geometrically by φ
        - "wave": α values oscillate with period related to π
        """
        import mlx.core as mx

        if pattern == "constant_phi":
            alphas = [BLEND_RATIOS["phi"]] * len(layers)
        elif pattern == "constant_pi_e":
            alphas = [BLEND_RATIOS["pi_e"]] * len(layers)
        elif pattern == "fibonacci":
            # Fibonacci ratios approach φ
            fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
            alphas = [fib[i] / fib[i+1] if i+1 < len(fib) else PHI / (1 + PHI)
                     for i in range(len(layers))]
        elif pattern == "geometric_decay":
            # Start at φ/(1+φ), decay by φ each layer
            base = BLEND_RATIOS["phi"]
            alphas = [base * (1/PHI)**(i/len(layers)) for i in range(len(layers))]
        elif pattern == "wave_pi":
            # Oscillate around 0.5 with period π
            alphas = [0.5 + 0.1 * np.sin(i * np.pi / len(layers))
                     for i in range(len(layers))]
        elif pattern == "progressive":
            # Gradually shift from residual-heavy to MLP-heavy
            alphas = np.linspace(0.3, 0.7, len(layers)).tolist()
        else:
            alphas = [0.5] * len(layers)

        for layer_idx, alpha in zip(layers, alphas):
            self.modify_layer_residual_scale(layer_idx, alpha)

        return {"pattern": pattern, "alphas": alphas}

    def run_experiment(self) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 3, mid + 4))

        logger.info("=" * 60)
        logger.info("RESIDUAL CONNECTION TEST - Geometric Information Flow")
        logger.info("=" * 60)

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial: {initial}")

        results = {}

        # Test different blending patterns
        patterns = [
            "constant_phi",
            "constant_pi_e",
            "fibonacci",
            "geometric_decay",
            "wave_pi",
            "progressive",
        ]

        for pattern in patterns:
            self.reset_layers(layers)
            logger.info(f"\n--- Pattern: {pattern} ---")

            blend_info = self.test_geometric_blend_pattern(layers, pattern)
            logger.info(f"  Alphas: {[f'{a:.3f}' for a in blend_info['alphas']]}")

            final = self.evaluate_by_category()
            changes = {k: final[k] - initial[k] for k in initial}
            degraded = [k for k, v in changes.items() if v < -0.01]
            improved = [k for k, v in changes.items() if v > 0.01]

            results[pattern] = {
                "alphas": blend_info["alphas"],
                "final": final,
                "changes": changes,
                "degraded": degraded,
                "improved": improved,
            }

            status = "✓" if not degraded else "✗"
            improve_str = f", improved={improved}" if improved else ""
            logger.info(f"{status} overall={final['overall']:.1%}, degraded={degraded}{improve_str}")

        self.reset_layers(layers)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)

        no_degrade = [(k, v) for k, v in results.items() if not v["degraded"]]
        has_improve = [(k, v) for k, v in results.items() if v["improved"]]
        both = [(k, v) for k, v in results.items() if not v["degraded"] and v["improved"]]

        logger.info(f"Patterns with NO degradation: {len(no_degrade)}/{len(results)}")
        logger.info(f"Patterns with improvement: {len(has_improve)}/{len(results)}")
        logger.info(f"Patterns with BOTH: {len(both)}/{len(results)}")

        if both:
            logger.info("\n*** SUCCESS - Improvement without degradation: ***")
            for key, data in sorted(both, key=lambda x: x[1]["changes"]["overall"], reverse=True):
                logger.info(f"  {key}: +{data['changes']['overall']:.1%} overall")

        return {
            "initial": initial,
            "results": results,
            "success": len(both) > 0,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = ResidualConnectionTest(model, tokenizer)
    results = test.run_experiment()

    # Save
    output_path = "data/residual_connection_test.json"
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
