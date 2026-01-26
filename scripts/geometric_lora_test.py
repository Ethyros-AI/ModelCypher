#!/usr/bin/env python3
"""Experiment 17: Geometric LoRA Test.

Previous experiments showed:
- Pure additive (W + ΔW) preserves but doesn't improve
- Additive wasn't "connected" to the computation

Question: Can a LoRA-style adapter with geometric structure achieve integration?

Method: Instead of training a LoRA, we construct one that:
1. Uses activation-derived directions (connects to computation)
2. Has exact constant ratios in its structure
3. Is added to the model in a targeted way

This bridges:
- Parallel pathway (preserves original)
- Semantic direction discovery (activation-derived)
- Geometric structure (constant ratios)
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


class GeometricLoRATest:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)
        self._original_weights = {}

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

    def _capture_activations(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Capture activation at a specific layer."""
        import mlx.core as mx
        from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        x = self.model.model.embed_tokens(input_ids)
        mx.eval(x)

        attn_mask = create_attention_mask(x, None)
        conv_mask = create_ssm_mask(x, None)

        for i, layer in enumerate(self.model.model.layers):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            x = layer(x, mask, cache=None)
            mx.eval(x)
            if i == layer_idx:
                return np.array(x.tolist(), dtype=np.float32)

        return np.array(x.tolist(), dtype=np.float32)

    def _format_question(self, question: str, choices: List[str]) -> str:
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        return prompt

    def _evaluate_question(self, question: str, choices: List[str], correct_idx: int) -> bool:
        import mlx.core as mx

        prompt = self._format_question(question, choices)
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
        results = {}
        for cat, questions in CATEGORY_QUESTIONS.items():
            correct = sum(1 for q, c, idx in questions if self._evaluate_question(q, c, idx))
            results[cat] = correct / len(questions)
        results["overall"] = sum(results.values()) / len(results)
        return results

    def collect_activation_principal_directions(
        self,
        layer_idx: int,
        category: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect activations for a category and return principal directions.

        Returns (Vt, S) from SVD of activations - Vt contains the directions
        in activation space that capture the most variance.
        Vt has shape [rank, hidden_dim] which is what we need for LoRA A matrix.
        """
        questions = CATEGORY_QUESTIONS[category]
        all_acts = []

        for q, choices, _ in questions:
            prompt = self._format_question(q, choices)
            act = self._capture_activations(prompt, layer_idx)
            # Use last token position
            all_acts.append(act[0, -1, :])

        acts_matrix = np.array(all_acts)  # [n_questions, hidden_dim]
        _, S, Vt = svd(acts_matrix, full_matrices=False)
        # Vt has shape [min(n_questions, hidden_dim), hidden_dim]
        # = [5, 1024] for 5 questions

        return Vt, S

    def create_activation_aligned_lora(
        self,
        W_original: np.ndarray,
        activation_Vt: np.ndarray,
        rank: int,
        scale: float
    ) -> np.ndarray:
        """
        Create a LoRA-style delta that's aligned with activation directions.

        ΔW = B @ diag(S) @ A where:
        - A uses activation principal directions (connects to computation)
        - B is from weight's left singular vectors
        - S has geometric ratios

        W is [m, n] where m=output_dim, n=input_dim
        For gate_proj: m=4608, n=1024
        """
        m, n = W_original.shape

        # Get the weight's SVD
        U_w, S_w, _ = svd(W_original, full_matrices=False)

        # activation_Vt has shape [rank_act, hidden_dim] where rank_act = min(n_questions, hidden_dim)
        # hidden_dim = n (input dimension)
        # We use the top rows as A: [rank, n]
        actual_rank = min(rank, activation_Vt.shape[0])
        A_lora = activation_Vt[:actual_rank, :].astype(np.float32)  # [rank, n]

        # B: [m, rank] - from weight's left singular vectors
        B_lora = U_w[:, :actual_rank].astype(np.float32)  # [m, rank]

        # Create geometric singular values
        const_vals = list(CONSTANTS.values())
        S_lora = np.zeros(actual_rank, dtype=np.float32)
        S_lora[0] = scale * S_w[0]  # Scale relative to W

        for i in range(1, actual_rank):
            const = const_vals[i % len(const_vals)]
            S_lora[i] = S_lora[i-1] / const

        # ΔW = B @ diag(S) @ A has shape [m, n]
        delta_W = B_lora @ np.diag(S_lora) @ A_lora

        return delta_W

    def create_null_space_lora(
        self,
        W_original: np.ndarray,
        rank: int,
        scale: float
    ) -> np.ndarray:
        """
        Create a LoRA in W's null space - purely additive capacity.

        Uses W's smallest singular directions to add new capacity
        without interfering with existing computation.
        """
        U, S, Vt = svd(W_original, full_matrices=False)

        # Find the smallest singular value indices (null space approximation)
        null_indices = np.argsort(S)[:rank]

        # Create geometric singular values
        const_vals = list(CONSTANTS.values())
        S_lora = np.zeros(rank, dtype=np.float32)
        S_lora[0] = scale * S[0] * 0.001  # Very small

        for i in range(1, rank):
            const = const_vals[i % len(const_vals)]
            S_lora[i] = S_lora[i-1] / const

        # Construct ΔW using null space directions
        U_null = U[:, null_indices]
        Vt_null = Vt[null_indices, :]

        delta_W = U_null @ np.diag(S_lora) @ Vt_null

        return delta_W

    def create_category_specialized_lora(
        self,
        W_original: np.ndarray,
        layer_idx: int,
        target_category: str,
        rank: int,
        scale: float
    ) -> np.ndarray:
        """
        Create a LoRA specialized for a specific category.

        Uses activation directions from that category to create
        a delta that enhances processing for those inputs.
        """
        # Get activation directions for the target category
        act_Vt, act_S = self.collect_activation_principal_directions(layer_idx, target_category)

        # The activation principal directions indicate how this category
        # is processed. Create a delta that strengthens these directions.

        m, n = W_original.shape

        # A: [rank, n] from activation directions
        # act_Vt is [rank_act, hidden_dim=n]
        actual_rank = min(rank, act_Vt.shape[0])
        A_lora = act_Vt[:actual_rank, :].astype(np.float32)  # [rank, n]

        # B: [m, rank] - use W's left SVs
        U_w, _, _ = svd(W_original, full_matrices=False)
        B_lora = U_w[:, :actual_rank].astype(np.float32)  # [m, rank]

        # Geometric singular values
        const_vals = list(CONSTANTS.values())
        S_lora = np.zeros(actual_rank, dtype=np.float32)
        S_lora[0] = scale * act_S[0]

        for i in range(1, actual_rank):
            const = const_vals[i % len(const_vals)]
            S_lora[i] = S_lora[i-1] / const

        # ΔW = B @ diag(S) @ A
        delta_W = B_lora @ np.diag(S_lora) @ A_lora

        return delta_W

    def run_experiment(self) -> Dict:
        mid = self.n_layers // 2
        layers = list(range(mid - 2, mid + 3))  # 5 middle layers

        self._cache_weights(layers)

        logger.info("=" * 60)
        logger.info("GEOMETRIC LoRA TEST")
        logger.info("=" * 60)

        initial = self.evaluate_by_category()
        logger.info(f"\nInitial: {initial}")

        results = {}

        # Test different LoRA strategies
        lora_configs = [
            ("activation_aligned", 4, 0.001),
            ("activation_aligned", 4, 0.01),
            ("activation_aligned", 8, 0.001),
            ("null_space", 4, 0.001),
            ("null_space", 4, 0.01),
            ("null_space", 8, 0.001),
            ("category_language", 4, 0.001),
            ("category_language", 4, 0.01),
        ]

        for lora_type, rank, scale in lora_configs:
            key = f"{lora_type}_r{rank}_s{scale}"
            logger.info(f"\n--- {key} ---")

            self._reset_weights(layers)

            for layer_idx in layers:
                W = self._original_weights[layer_idx]

                if lora_type == "activation_aligned":
                    # Use mixed category activations
                    act_Vt, _ = self.collect_activation_principal_directions(layer_idx, "language")
                    delta_W = self.create_activation_aligned_lora(W, act_Vt, rank, scale)
                elif lora_type == "null_space":
                    delta_W = self.create_null_space_lora(W, rank, scale)
                elif lora_type == "category_language":
                    delta_W = self.create_category_specialized_lora(W, layer_idx, "language", rank, scale)
                else:
                    continue

                if np.all(np.isfinite(delta_W)):
                    W_new = W + delta_W
                    if np.all(np.isfinite(W_new)):
                        self._set_weight(layer_idx, W_new)

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

            status = "+" if improved and not degraded else ("~" if not degraded else "-")
            logger.info(f"{status} overall={final['overall']:.1%}, degraded={degraded}, improved={improved}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)

        no_degrade = [(k, v) for k, v in results.items() if not v.get("degraded", [])]
        has_improve = [(k, v) for k, v in results.items() if v.get("improved", [])]
        both = [(k, v) for k, v in results.items()
                if not v.get("degraded", []) and v.get("improved", [])]

        logger.info(f"Configs with NO degradation: {len(no_degrade)}/{len(lora_configs)}")
        logger.info(f"Configs with improvement: {len(has_improve)}/{len(lora_configs)}")
        logger.info(f"Configs with BOTH: {len(both)}/{len(lora_configs)}")

        if both:
            logger.info("\n*** SUCCESS - Improvement without degradation: ***")
            for key, data in sorted(both, key=lambda x: x[1]["changes"]["overall"], reverse=True):
                logger.info(f"  {key}: +{data['changes']['overall']:.1%} overall")
        else:
            logger.info("\nNo configuration achieved improvement without degradation")

        self._reset_weights(layers)

        return {
            "layers": layers,
            "initial": initial,
            "results": results,
            "success": len(both) > 0,
        }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    test = GeometricLoRATest(model, tokenizer)
    results = test.run_experiment()

    # Save
    output_path = "data/geometric_lora_test.json"
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
