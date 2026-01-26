#!/usr/bin/env python3
"""Experiment 51: Manifold Compression.

The discovery: Math space is scattered (3.7 effective dims), non-math is
concentrated (1.46 dims). The relational structure is corrupted.

The hypothesis: If we compress the math representation space to have similar
concentration/coherence as non-math, the individual facts might align automatically
because they'd be part of a coherent relational structure.

This is manifold alignment, not fact-fixing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd, lstsq

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Test prompts
MATH_TEST = [
    ("What is 1 + 2?", ["2", "3", "4", "5"], 1),
    ("What is 2 + 2?", ["3", "4", "5", "6"], 1),
    ("What is 3 + 3?", ["5", "6", "7", "8"], 1),
    ("What is 5 + 5?", ["8", "9", "10", "11"], 2),
    ("What is 2 × 2?", ["2", "4", "6", "8"], 1),
    ("What is 3 × 3?", ["6", "9", "12", "15"], 1),
    ("What is 10 - 5?", ["3", "4", "5", "6"], 2),
    ("What is 6 ÷ 2?", ["2", "3", "4", "5"], 1),
    ("What is 1 + 1?", ["1", "2", "3", "4"], 1),
    ("What is 4 + 3?", ["6", "7", "8", "9"], 1),
]

# Reference prompts for non-math structure
NON_MATH_REF = [
    "The capital of France is",
    "Water is made of",
    "The sun rises in the",
    "Birds can",
    "Fire is",
    "The sky is",
    "Fish live in",
    "Trees are",
    "Ice is cold and",
    "Dogs are",
]

# Math prompts to get structure from
MATH_REF = [
    "1+1=",
    "2+2=",
    "3+3=",
    "5+5=",
    "2×2=",
    "3×3=",
    "10-5=",
    "6÷2=",
    "1+2=",
    "4+3=",
]


class ManifoldCompressor:
    """Compress the math manifold to match non-math structure."""

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

    def _get_activations(self, prompts: List[str]) -> np.ndarray:
        """Get final logits as activation proxy."""
        import mlx.core as mx
        activations = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)
            act = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            activations.append(act)
        return np.vstack(activations)

    def _compute_gram_structure(self, acts: np.ndarray) -> Dict:
        """Compute Gram matrix properties."""
        G = acts @ acts.T
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1]
        eigvals_pos = eigvals[eigvals > 1e-10]
        eigvals_norm = eigvals_pos / eigvals_pos.sum()
        entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-10))
        effective_dim = np.exp(entropy)
        concentration = eigvals_pos[0] / eigvals_pos.sum() if eigvals_pos.sum() > 0 else 0
        return {
            "effective_dim": float(effective_dim),
            "concentration": float(concentration),
            "top_eigenvalue": float(eigvals_pos[0]) if len(eigvals_pos) > 0 else 0,
        }

    def _evaluate_math(self) -> Tuple[int, int, List]:
        """Evaluate math accuracy."""
        import mlx.core as mx
        correct = 0
        results = []

        for q, choices, correct_idx in MATH_TEST:
            prompt = f"Question: {q}\n"
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

            is_correct = prediction == correct_idx
            if is_correct:
                correct += 1
            results.append({
                "question": q,
                "correct": is_correct,
                "predicted": choices[prediction],
                "expected": choices[correct_idx],
            })

        return correct, len(MATH_TEST), results

    def compute_compression_transform(self, target_concentration: float = 0.9) -> np.ndarray:
        """
        Compute a transform that compresses the math representation space.

        The idea: The weight matrix W maps inputs to outputs. If we modify W
        to concentrate its singular values, the output space becomes more focused.
        """
        logger.info("Computing compression transform...")

        mid = self.n_layers // 2
        W = self._original_weights[mid]

        # SVD of current weight
        U, S, Vt = svd(W, full_matrices=False)

        # Current concentration
        S_norm = S / S.sum()
        current_concentration = S[0] / S.sum()
        logger.info(f"  Current concentration: {current_concentration:.3f}")
        logger.info(f"  Target concentration: {target_concentration:.3f}")

        # Compute new singular values that achieve target concentration
        # We want S_new[0] / S_new.sum() = target_concentration
        # Keep the top singular value, scale down the rest

        S_new = S.copy()

        # Method: Scale down non-dominant singular values
        # S_new[0] stays the same
        # S_new[1:] get scaled by factor alpha
        # S_new[0] / (S_new[0] + alpha * sum(S[1:])) = target_concentration
        # S_new[0] = target_concentration * (S_new[0] + alpha * sum(S[1:]))
        # S[0] = target * S[0] + target * alpha * sum(S[1:])
        # S[0] * (1 - target) = target * alpha * sum(S[1:])
        # alpha = S[0] * (1 - target) / (target * sum(S[1:]))

        rest_sum = S[1:].sum()
        if rest_sum > 0:
            alpha = S[0] * (1 - target_concentration) / (target_concentration * rest_sum)
            alpha = max(0.01, min(alpha, 1.0))  # Clamp to reasonable range
            S_new[1:] = S[1:] * alpha
            logger.info(f"  Compression factor (alpha): {alpha:.4f}")

        new_concentration = S_new[0] / S_new.sum()
        logger.info(f"  New concentration: {new_concentration:.3f}")

        # Reconstruct weight matrix
        W_new = U @ np.diag(S_new) @ Vt

        return W_new

    def apply_selective_compression(self, layer_idx: int, math_direction: np.ndarray, compression: float = 0.5):
        """
        Apply compression only in the math-relevant subspace.

        The idea: Find the directions in weight space that correspond to math,
        and compress those while leaving non-math directions alone.
        """
        W = self._original_weights[layer_idx]
        U, S, Vt = svd(W, full_matrices=False)

        # Project math direction onto singular vectors
        # Find which singular vectors are aligned with math
        math_direction_norm = math_direction / (np.linalg.norm(math_direction) + 1e-10)

        # Alignment of each left singular vector with math direction
        alignments = np.abs(U.T @ math_direction_norm)

        # Compress singular values proportionally to their alignment with math
        S_new = S.copy()
        for i in range(len(S)):
            # More aligned with math = more compression
            compression_factor = 1.0 - compression * alignments[i] if i < len(alignments) else 1.0
            S_new[i] = S[i] * compression_factor

        W_new = U @ np.diag(S_new) @ Vt
        return W_new

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 51: MANIFOLD COMPRESSION")
        logger.info("=" * 60)
        logger.info("\nCan we fix math by compressing its scattered representation space?\n")

        mid = self.n_layers // 2
        layers = [mid]
        self._cache_weights(layers)

        # Initial evaluation
        init_correct, init_total, init_results = self._evaluate_math()
        init_acc = init_correct / init_total
        logger.info(f"Initial math accuracy: {init_correct}/{init_total} ({init_acc:.0%})")

        # Get initial structure
        math_acts = self._get_activations(MATH_REF)
        non_math_acts = self._get_activations(NON_MATH_REF)

        init_math_struct = self._compute_gram_structure(math_acts)
        init_non_math_struct = self._compute_gram_structure(non_math_acts)

        logger.info(f"\nInitial structure:")
        logger.info(f"  Math effective dim: {init_math_struct['effective_dim']:.2f}")
        logger.info(f"  Non-math effective dim: {init_non_math_struct['effective_dim']:.2f}")
        logger.info(f"  Math concentration: {init_math_struct['concentration']:.3f}")
        logger.info(f"  Non-math concentration: {init_non_math_struct['concentration']:.3f}")

        results = {
            "initial": {
                "accuracy": init_acc,
                "correct": init_correct,
                "total": init_total,
                "math_structure": init_math_struct,
                "non_math_structure": init_non_math_struct,
                "results": init_results,
            },
            "compressions": {},
        }

        # Try different compression levels
        target_concentrations = [0.8, 0.85, 0.9, 0.95]

        for target in target_concentrations:
            logger.info(f"\n{'='*60}")
            logger.info(f"TARGET CONCENTRATION: {target}")
            logger.info("=" * 60)

            self._reset_weights(layers)

            # Compute and apply compression
            W_compressed = self.compute_compression_transform(target)

            if np.all(np.isfinite(W_compressed)):
                self._set_weight(mid, W_compressed)

                # Evaluate
                comp_correct, comp_total, comp_results = self._evaluate_math()
                comp_acc = comp_correct / comp_total

                # Check new structure
                math_acts_new = self._get_activations(MATH_REF)
                new_struct = self._compute_gram_structure(math_acts_new)

                logger.info(f"\nResults at target={target}:")
                logger.info(f"  Accuracy: {init_correct}/{init_total} → {comp_correct}/{comp_total} ({init_acc:.0%} → {comp_acc:.0%})")
                logger.info(f"  Effective dim: {init_math_struct['effective_dim']:.2f} → {new_struct['effective_dim']:.2f}")
                logger.info(f"  Concentration: {init_math_struct['concentration']:.3f} → {new_struct['concentration']:.3f}")

                # Show what changed
                logger.info(f"\n  Individual results:")
                for i, (init_r, comp_r) in enumerate(zip(init_results, comp_results)):
                    if init_r['correct'] != comp_r['correct']:
                        if comp_r['correct']:
                            logger.info(f"    ✓ FIXED: {init_r['question']} ({init_r['predicted']} → {comp_r['predicted']})")
                        else:
                            logger.info(f"    ✗ BROKE: {init_r['question']} ({init_r['predicted']} → {comp_r['predicted']})")

                improved = comp_correct > init_correct
                degraded = comp_correct < init_correct

                results["compressions"][str(target)] = {
                    "accuracy": comp_acc,
                    "correct": comp_correct,
                    "new_structure": new_struct,
                    "improved": improved,
                    "degraded": degraded,
                    "results": comp_results,
                }
            else:
                logger.info(f"  SKIPPED: Non-finite weights")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        improvements = [t for t, data in results["compressions"].items() if data.get("improved")]
        degradations = [t for t, data in results["compressions"].items() if data.get("degraded")]

        if improvements:
            best_target = max(improvements, key=lambda t: results["compressions"][t]["accuracy"])
            best_acc = results["compressions"][best_target]["accuracy"]
            logger.info(f"\n*** COMPRESSION IMPROVED ACCURACY ***")
            logger.info(f"Best: target={best_target}, accuracy={best_acc:.0%}")
            results["conclusion"] = "compression_helps"
        elif degradations:
            logger.info(f"\n*** COMPRESSION DEGRADED ACCURACY ***")
            logger.info("The scattered representation might be adaptive, not corrupted.")
            results["conclusion"] = "compression_hurts"
        else:
            logger.info(f"\n*** NO CHANGE ***")
            logger.info("Compression doesn't affect math accuracy.")
            results["conclusion"] = "no_effect"

        self._reset_weights(layers)
        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = ManifoldCompressor(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/manifold_compression.json"
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
