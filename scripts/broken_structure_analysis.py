#!/usr/bin/env python3
"""Experiment 48: Broken Structure Analysis.

Phase 9 - The Shape of Broken.

The off-by-one error isn't 262 separate bugs - it's ONE structural misalignment.
Find the high-dimensional direction that represents "incrementing" and see if
it's systematically corrupted.

Hypothesis: There exists a single direction in weight/activation space that,
when aligned, would fix ALL off-by-one errors simultaneously.
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


# Generate pairs: (question, wrong_answer, correct_answer, error_type)
def generate_error_pairs():
    """Generate all the off-by-one error cases."""
    pairs = []

    # Addition: 1+n = n (should be n+1)
    for n in range(2, 11):
        pairs.append({
            "question": f"What is 1 + {n}?",
            "wrong": n,
            "correct": n + 1,
            "error": "ignore_plus_one",
            "operation": "addition",
        })

    # Subtraction: n-1 = n (should be n-1)
    for n in range(2, 20):
        pairs.append({
            "question": f"What is {n} - 1?",
            "wrong": n,
            "correct": n - 1,
            "error": "ignore_minus_one",
            "operation": "subtraction",
        })

    # Division: off by 1
    for divisor in range(2, 11):
        for result in range(2, 11):
            dividend = divisor * result
            if dividend <= 100:
                pairs.append({
                    "question": f"What is {dividend} ÷ {divisor}?",
                    "wrong": result - 1,
                    "correct": result,
                    "error": "off_by_one_division",
                    "operation": "division",
                })

    return pairs


class BrokenStructureAnalyzer:
    """Find the high-dimensional shape of the broken arithmetic structure."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(model.model.layers)

    def _get_activation_for_answer(self, question: str, answer: int) -> np.ndarray:
        """Get activation when model processes question with specific answer in mind."""
        import mlx.core as mx

        # Format: Question + Answer
        prompt = f"Question: {question}\nAnswer: {answer}"
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get hidden state from middle layer
        activations = []

        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                activations.append(outputs[0])
            else:
                activations.append(outputs)

        mid = self.n_layers // 2
        layer = self.model.model.layers[mid]

        # Forward pass
        _ = self.model(input_ids)
        mx.eval(_)

        # Get the weight as proxy for activation direction
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)

        return np.array(w.tolist(), dtype=np.float32)

    def _compute_error_direction(self, question: str, wrong: int, correct: int) -> np.ndarray:
        """Compute the direction from wrong answer to correct answer in weight space."""
        import mlx.core as mx

        # Get gradients for wrong vs correct answer
        wrong_prompt = f"Question: {question}\nAnswer: {wrong}"
        correct_prompt = f"Question: {question}\nAnswer: {correct}"

        wrong_tokens = self.tokenizer.encode(wrong_prompt)
        correct_tokens = self.tokenizer.encode(correct_prompt)

        wrong_ids = mx.array([wrong_tokens])
        correct_ids = mx.array([correct_tokens])

        # Get logits for both
        wrong_logits = self.model(wrong_ids)
        correct_logits = self.model(correct_ids)
        mx.eval(wrong_logits, correct_logits)

        # Difference in final logits as proxy for error direction
        diff = np.array(correct_logits[0, -1, :].tolist()) - np.array(wrong_logits[0, -1, :].tolist())

        return diff

    def _get_gradient_for_target(self, question: str, choices: List[str], target_idx: int) -> np.ndarray:
        """Get gradient of loss w.r.t. weights for a specific target answer."""
        import mlx.core as mx
        import mlx.nn as nn

        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Get middle layer weights
        mid = self.n_layers // 2
        layer = self.model.model.layers[mid]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)

        # Forward pass
        logits = self.model(input_ids)
        mx.eval(logits)

        # Get choice tokens
        choice_tokens = []
        for letter in ['A', 'B', 'C', 'D']:
            for variant in [f" {letter}", letter, f"{letter}."]:
                token_ids = self.tokenizer.encode(variant)
                if token_ids:
                    choice_tokens.append(token_ids[-1])
                    break
            else:
                choice_tokens.append(0)

        # Compute loss toward target
        next_logits = logits[0, -1, :]
        scores = mx.array([next_logits[t] for t in choice_tokens[:len(choices)]])
        mx.eval(scores)

        # Return weight as gradient proxy (actual gradient computation would need autograd)
        return np.array(w.tolist(), dtype=np.float32)

    def find_common_error_direction(self, pairs: List[Dict]) -> Dict:
        """Find the common direction across all off-by-one errors."""
        logger.info("Computing error directions for all off-by-one cases...")

        error_directions = []
        error_types = {"ignore_plus_one": [], "ignore_minus_one": [], "off_by_one_division": []}

        for pair in pairs[:50]:  # Sample for speed
            diff = self._compute_error_direction(
                pair["question"],
                pair["wrong"],
                pair["correct"]
            )
            error_directions.append(diff)
            error_types[pair["error"]].append(diff)

        # Stack all error directions
        error_matrix = np.vstack(error_directions)
        logger.info(f"Error matrix shape: {error_matrix.shape}")

        # PCA to find common direction
        mean_dir = error_matrix.mean(axis=0)
        centered = error_matrix - mean_dir

        # SVD of error directions
        U, S, Vt = svd(centered, full_matrices=False)

        # First principal component is the "main error direction"
        main_error_direction = Vt[0]

        # How much variance is explained by first component?
        total_var = (S ** 2).sum()
        first_var = S[0] ** 2
        variance_explained = first_var / total_var if total_var > 0 else 0

        logger.info(f"First PC explains {variance_explained:.1%} of error variance")

        # Check alignment of different error types with main direction
        alignments = {}
        for error_type, dirs in error_types.items():
            if dirs:
                type_mean = np.mean(dirs, axis=0)
                alignment = np.abs(np.dot(type_mean, main_error_direction)) / (
                    np.linalg.norm(type_mean) * np.linalg.norm(main_error_direction) + 1e-10
                )
                alignments[error_type] = float(alignment)
                logger.info(f"  {error_type} alignment with main direction: {alignment:.3f}")

        # Top singular values tell us the dimensionality of the error
        logger.info(f"\nTop 10 singular values of error matrix:")
        for i, s in enumerate(S[:10]):
            logger.info(f"  σ_{i+1}: {s:.3f} ({(s**2)/total_var:.1%} of variance)")

        # Effective dimensionality
        normalized_s = S / S.sum()
        entropy = -np.sum(normalized_s * np.log(normalized_s + 1e-10))
        effective_dim = np.exp(entropy)
        logger.info(f"\nEffective dimensionality of error: {effective_dim:.1f}")

        return {
            "n_samples": len(error_directions),
            "error_matrix_shape": list(error_matrix.shape),
            "variance_explained_by_first_pc": variance_explained,
            "effective_dimensionality": float(effective_dim),
            "error_type_alignments": alignments,
            "top_10_singular_values": S[:10].tolist(),
            "mean_direction_norm": float(np.linalg.norm(mean_dir)),
            "main_direction_norm": float(np.linalg.norm(main_error_direction)),
        }

    def analyze_increment_concept(self) -> Dict:
        """Analyze how the model represents the concept of incrementing."""
        logger.info("\n" + "=" * 60)
        logger.info("ANALYZING INCREMENT CONCEPT")
        logger.info("=" * 60)

        # Generate increment pairs: n → n+1
        increment_pairs = []
        for n in range(1, 20):
            increment_pairs.append((str(n), str(n+1)))

        # Get representation difference for each increment
        increment_directions = []

        for n_str, n_plus_1_str in increment_pairs[:15]:
            # Encode each number
            n_tokens = self.tokenizer.encode(n_str)
            n_plus_1_tokens = self.tokenizer.encode(n_plus_1_str)

            # Get embeddings
            import mlx.core as mx

            embed = self.model.model.embed_tokens
            n_embed = embed(mx.array([n_tokens[-1]]))
            n_plus_1_embed = embed(mx.array([n_plus_1_tokens[-1]]))
            mx.eval(n_embed, n_plus_1_embed)

            # Increment direction in embedding space
            diff = np.array(n_plus_1_embed[0].tolist()) - np.array(n_embed[0].tolist())
            increment_directions.append(diff)

        # Stack and analyze
        increment_matrix = np.vstack(increment_directions)
        logger.info(f"Increment matrix shape: {increment_matrix.shape}")

        # Mean increment direction
        mean_increment = increment_matrix.mean(axis=0)

        # Consistency: how aligned are individual increments with the mean?
        consistencies = []
        for i, inc_dir in enumerate(increment_directions):
            cos_sim = np.dot(inc_dir, mean_increment) / (
                np.linalg.norm(inc_dir) * np.linalg.norm(mean_increment) + 1e-10
            )
            consistencies.append(cos_sim)

        mean_consistency = np.mean(consistencies)
        logger.info(f"Mean consistency of increment direction: {mean_consistency:.3f}")

        # SVD of increment directions
        centered = increment_matrix - mean_increment
        U, S, Vt = svd(centered, full_matrices=False)

        total_var = (S ** 2).sum()
        first_var = S[0] ** 2
        variance_explained = first_var / total_var if total_var > 0 else 0

        logger.info(f"First PC explains {variance_explained:.1%} of increment variance")

        return {
            "n_pairs": len(increment_directions),
            "mean_consistency": float(mean_consistency),
            "individual_consistencies": [float(c) for c in consistencies],
            "variance_explained_by_first_pc": variance_explained,
            "mean_increment_norm": float(np.linalg.norm(mean_increment)),
        }

    def find_broken_dimension(self) -> Dict:
        """Find which dimension(s) are broken by comparing correct vs incorrect computations."""
        logger.info("\n" + "=" * 60)
        logger.info("FINDING BROKEN DIMENSIONS")
        logger.info("=" * 60)

        import mlx.core as mx

        # Get weight matrix
        mid = self.n_layers // 2
        layer = self.model.model.layers[mid]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        W = np.array(w.tolist(), dtype=np.float32)

        # SVD of weight matrix
        U, S, Vt = svd(W, full_matrices=False)
        logger.info(f"Weight matrix shape: {W.shape}")
        logger.info(f"SVD shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")

        # Analyze singular value spectrum
        logger.info(f"\nTop 20 singular values:")
        for i in range(min(20, len(S))):
            logger.info(f"  σ_{i+1}: {S[i]:.3f}")

        # Look for ratios near 1.0 (which might represent increment)
        logger.info(f"\nSingular value ratios near 1.0:")
        near_one = []
        for i in range(len(S) - 1):
            ratio = S[i] / S[i+1] if S[i+1] > 1e-10 else 0
            if 0.95 < ratio < 1.05:
                near_one.append((i, ratio))
                logger.info(f"  σ_{i+1}/σ_{i+2} = {ratio:.4f}")

        # The "broken" might be in how these near-1 ratios deviate
        deviations = [abs(r - 1.0) for _, r in near_one]
        mean_deviation = np.mean(deviations) if deviations else 0

        logger.info(f"\nMean deviation from 1.0: {mean_deviation:.4f}")

        return {
            "weight_shape": list(W.shape),
            "top_20_singular_values": S[:20].tolist(),
            "ratios_near_one": near_one,
            "mean_deviation_from_one": float(mean_deviation),
            "n_dimensions_near_one": len(near_one),
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 48: BROKEN STRUCTURE ANALYSIS")
        logger.info("=" * 60)
        logger.info("\nFinding the high-dimensional shape of broken arithmetic\n")

        # Generate error pairs
        pairs = generate_error_pairs()
        logger.info(f"Generated {len(pairs)} error pairs")

        # Find common error direction
        error_analysis = self.find_common_error_direction(pairs)

        # Analyze increment concept
        increment_analysis = self.analyze_increment_concept()

        # Find broken dimensions
        broken_dim_analysis = self.find_broken_dimension()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY: THE SHAPE OF BROKEN")
        logger.info("=" * 60)

        first_pc_var = error_analysis["variance_explained_by_first_pc"]
        effective_dim = error_analysis["effective_dimensionality"]
        increment_consistency = increment_analysis["mean_consistency"]

        logger.info(f"\n1. Error Structure:")
        logger.info(f"   - First PC explains {first_pc_var:.1%} of error variance")
        logger.info(f"   - Effective dimensionality: {effective_dim:.1f}")

        if first_pc_var > 0.5:
            logger.info(f"   → The error is CONCENTRATED in ~1 direction (fixable with single alignment)")
        else:
            logger.info(f"   → The error is DISTRIBUTED across {effective_dim:.0f} dimensions")

        logger.info(f"\n2. Increment Concept:")
        logger.info(f"   - Mean consistency: {increment_consistency:.3f}")
        if increment_consistency > 0.8:
            logger.info(f"   → Increment direction IS consistent (structure exists)")
        else:
            logger.info(f"   → Increment direction is INCONSISTENT (structure corrupted)")

        logger.info(f"\n3. Error Type Alignments:")
        for error_type, alignment in error_analysis["error_type_alignments"].items():
            logger.info(f"   - {error_type}: {alignment:.3f}")

        if all(a > 0.7 for a in error_analysis["error_type_alignments"].values()):
            conclusion = "unified_error"
            logger.info(f"\n*** ALL ERROR TYPES SHARE THE SAME DIRECTION ***")
            logger.info(f"This means ONE alignment could fix addition, subtraction, AND division!")
        else:
            conclusion = "multiple_errors"
            logger.info(f"\n*** ERROR TYPES HAVE DIFFERENT DIRECTIONS ***")
            logger.info(f"Each operation type may need separate alignment.")

        results = {
            "error_analysis": error_analysis,
            "increment_analysis": increment_analysis,
            "broken_dim_analysis": broken_dim_analysis,
            "conclusion": conclusion,
            "summary": {
                "first_pc_variance": first_pc_var,
                "effective_dimensionality": effective_dim,
                "increment_consistency": increment_consistency,
                "unified_error_direction": all(a > 0.7 for a in error_analysis["error_type_alignments"].values()),
            }
        }

        return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    experiment = BrokenStructureAnalyzer(model, tokenizer)
    results = experiment.run_experiment()

    output_path = "data/experiments/broken_structure_analysis.json"
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
