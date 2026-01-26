#!/usr/bin/env python3
"""Experiment 52: Full Manifold Compression Test.

Experiment 51 showed 40% → 80% on 10 math questions.
Now test on ALL 455 arithmetic facts to see if compression scales.

The hypothesis: If manifold compression works by fixing the relational structure,
it should improve accuracy across ALL arithmetic operations, not just the 10 test cases.
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


def generate_arithmetic_tables() -> Dict[str, List[Tuple[str, int]]]:
    """Generate all basic arithmetic facts."""
    tables = {
        "addition": [],
        "subtraction": [],
        "multiplication": [],
        "division": [],
    }

    # Addition: a + b for a,b in 1-12
    for a in range(1, 13):
        for b in range(1, 13):
            tables["addition"].append((f"{a}+{b}=", a + b))

    # Subtraction: a - b for a in 1-12, b in 1-a (no negatives)
    for a in range(1, 13):
        for b in range(1, a + 1):
            tables["subtraction"].append((f"{a}-{b}=", a - b))

    # Multiplication: a × b for a,b in 1-12
    for a in range(1, 13):
        for b in range(1, 13):
            tables["multiplication"].append((f"{a}×{b}=", a * b))

    # Division: a ÷ b where a is divisible by b
    for b in range(1, 13):
        for multiplier in range(1, 13):
            a = b * multiplier
            if a <= 144:
                tables["division"].append((f"{a}÷{b}=", multiplier))

    return tables


class FullManifoldCompressor:
    """Test manifold compression on all arithmetic facts."""

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

    def _evaluate_fact(self, prompt: str, expected: int) -> Tuple[bool, int]:
        """Evaluate a single arithmetic fact."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        # Get probabilities for numbers 0-144
        probs = []
        for num in range(145):
            num_str = str(num)
            token_ids = self.tokenizer.encode(num_str)
            if token_ids:
                prob = float(next_logits[token_ids[-1]].item())
                probs.append((num, prob))

        probs.sort(key=lambda x: x[1], reverse=True)
        predicted = probs[0][0] if probs else -1

        return predicted == expected, predicted

    def _evaluate_all(self, tables: Dict[str, List[Tuple[str, int]]]) -> Dict:
        """Evaluate all arithmetic facts."""
        results = {}
        for op_name, facts in tables.items():
            correct = 0
            errors = []
            for prompt, expected in facts:
                is_correct, predicted = self._evaluate_fact(prompt, expected)
                if is_correct:
                    correct += 1
                else:
                    errors.append({
                        "prompt": prompt,
                        "expected": expected,
                        "predicted": predicted,
                        "error": predicted - expected,
                    })
            results[op_name] = {
                "correct": correct,
                "total": len(facts),
                "accuracy": correct / len(facts),
                "errors": errors[:10],  # Keep only first 10 errors for brevity
                "error_count": len(errors),
            }
        return results

    def compute_compression_transform(self, layer_idx: int, target_concentration: float = 0.9) -> np.ndarray:
        """Compress the weight matrix by concentrating singular values."""
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        S_new = S.copy()
        rest_sum = S[1:].sum()
        if rest_sum > 0:
            alpha = S[0] * (1 - target_concentration) / (target_concentration * rest_sum)
            alpha = max(0.01, min(alpha, 1.0))
            S_new[1:] = S[1:] * alpha

        W_new = U @ np.diag(S_new) @ Vt
        return W_new

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 52: FULL MANIFOLD COMPRESSION TEST")
        logger.info("=" * 60)
        logger.info("\nDoes 40%→80% on 10 questions scale to 455 facts?\n")

        tables = generate_arithmetic_tables()
        total_facts = sum(len(facts) for facts in tables.values())
        logger.info(f"Testing {total_facts} arithmetic facts")

        mid = self.n_layers // 2
        layers = [mid]
        self._cache_weights(layers)

        # Baseline evaluation
        logger.info("\nEvaluating baseline...")
        baseline = self._evaluate_all(tables)
        baseline_correct = sum(r["correct"] for r in baseline.values())
        baseline_acc = baseline_correct / total_facts

        logger.info(f"\nBaseline Results:")
        logger.info(f"  Total: {baseline_correct}/{total_facts} ({baseline_acc:.1%})")
        for op, data in baseline.items():
            logger.info(f"  {op.capitalize()}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")

        results = {
            "baseline": {
                "total_correct": baseline_correct,
                "total_facts": total_facts,
                "accuracy": baseline_acc,
                "by_operation": baseline,
            },
            "compressions": {},
        }

        # Test different compression levels
        target_concentrations = [0.85, 0.9, 0.95]

        for target in target_concentrations:
            logger.info(f"\n{'='*60}")
            logger.info(f"TARGET CONCENTRATION: {target}")
            logger.info("=" * 60)

            self._reset_weights(layers)
            W_compressed = self.compute_compression_transform(mid, target)

            if np.all(np.isfinite(W_compressed)):
                self._set_weight(mid, W_compressed)

                compressed = self._evaluate_all(tables)
                comp_correct = sum(r["correct"] for r in compressed.values())
                comp_acc = comp_correct / total_facts

                logger.info(f"\nResults at target={target}:")
                logger.info(f"  Total: {baseline_correct}/{total_facts} → {comp_correct}/{total_facts} ({baseline_acc:.1%} → {comp_acc:.1%})")

                for op in tables.keys():
                    b = baseline[op]
                    c = compressed[op]
                    delta = c["correct"] - b["correct"]
                    sign = "+" if delta > 0 else ""
                    logger.info(f"  {op.capitalize()}: {b['correct']} → {c['correct']} ({sign}{delta})")

                improvement = comp_acc - baseline_acc
                results["compressions"][str(target)] = {
                    "total_correct": comp_correct,
                    "accuracy": comp_acc,
                    "improvement": improvement,
                    "by_operation": compressed,
                }
            else:
                logger.info(f"  SKIPPED: Non-finite weights")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        best_target = None
        best_acc = baseline_acc
        for target, data in results["compressions"].items():
            if data["accuracy"] > best_acc:
                best_acc = data["accuracy"]
                best_target = target

        if best_target:
            improvement = best_acc - baseline_acc
            logger.info(f"\n*** COMPRESSION IMPROVED FULL ARITHMETIC ***")
            logger.info(f"Best: target={best_target}")
            logger.info(f"Accuracy: {baseline_acc:.1%} → {best_acc:.1%} (+{improvement:.1%})")
            results["conclusion"] = "compression_scales"
            results["best_target"] = best_target
            results["best_improvement"] = improvement
        else:
            logger.info(f"\n*** COMPRESSION DID NOT IMPROVE FULL ARITHMETIC ***")
            logger.info("The 40%→80% improvement on 10 questions may not generalize.")
            results["conclusion"] = "compression_does_not_scale"

        self._reset_weights(layers)
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = FullManifoldCompressor(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/full_manifold_compression.json"
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
