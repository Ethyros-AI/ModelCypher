#!/usr/bin/env python3
"""Experiment 49: Relational Structure Comparison.

The hypothesis: Math corruption is not about individual facts - it's about
the relational structure across all dimensions being misaligned.

If we compare Gram matrices (relational structure) between:
- Broken model (LFM2-350M, 18% addition)
- Working model (LFM2-1.2B, better math)

The CKA should be:
- Math operations: < 1.0 (misaligned relational structure)
- Non-math operations: closer to 1.0 (aligned relational structure)

This proves the corruption is in how the entire space is organized,
not in individual neurons or weights.
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


# Math prompts - same operations, will reveal relational structure
MATH_PROMPTS = [
    "1 + 1 =",
    "1 + 2 =",
    "2 + 2 =",
    "2 + 3 =",
    "3 + 3 =",
    "5 + 5 =",
    "1 - 1 =",
    "2 - 1 =",
    "3 - 1 =",
    "5 - 3 =",
    "10 - 5 =",
    "2 × 2 =",
    "3 × 3 =",
    "4 × 4 =",
    "2 × 3 =",
    "3 × 4 =",
    "4 ÷ 2 =",
    "6 ÷ 2 =",
    "9 ÷ 3 =",
    "10 ÷ 2 =",
]

# Non-math prompts - should have similar relational structure across models
NON_MATH_PROMPTS = [
    "The capital of France is",
    "The color of the sky is",
    "Water freezes at",
    "The sun rises in the",
    "Dogs are a type of",
    "Paris is a city in",
    "The ocean is full of",
    "Trees produce",
    "Birds can",
    "The Earth orbits the",
    "Cats like to",
    "Fire is",
    "Ice is",
    "The moon is",
    "Snow is",
    "Rain falls from",
    "Flowers bloom in",
    "Fish live in",
    "Stars are",
    "The sky is",
]


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Centered Kernel Alignment between two activation matrices.

    X, Y: (n_samples, n_features) - activations for same prompts from different models

    CKA measures similarity of relational structure, not individual values.
    """
    # Center the Gram matrices
    def center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # Compute linear kernels (Gram matrices)
    K_X = X @ X.T  # (n_samples, n_samples) - relational structure of X
    K_Y = Y @ Y.T  # (n_samples, n_samples) - relational structure of Y

    # Center them
    K_X_centered = center_gram(K_X)
    K_Y_centered = center_gram(K_Y)

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = np.sum(K_X_centered * K_Y_centered)
    hsic_xx = np.sum(K_X_centered * K_X_centered)
    hsic_yy = np.sum(K_Y_centered * K_Y_centered)

    # CKA
    if hsic_xx > 0 and hsic_yy > 0:
        return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    return 0.0


class RelationalStructureComparison:
    """Compare relational structure between broken and working models."""

    def __init__(self, broken_model, broken_tokenizer, working_model, working_tokenizer):
        self.broken_model = broken_model
        self.broken_tokenizer = broken_tokenizer
        self.working_model = working_model
        self.working_tokenizer = working_tokenizer
        self.broken_n_layers = len(broken_model.model.layers)
        self.working_n_layers = len(working_model.model.layers)

    def _get_activations(self, model, tokenizer, prompts: List[str], layer_frac: float = 0.5) -> np.ndarray:
        """Get activations from a specific layer fraction for all prompts."""
        import mlx.core as mx

        n_layers = len(model.model.layers)
        layer_idx = int(n_layers * layer_frac)

        activations = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Forward pass to get hidden states
            # We'll use the embedding output projected through the layer
            hidden = model.model.embed_tokens(input_ids)
            mx.eval(hidden)

            for i in range(layer_idx + 1):
                layer = model.model.layers[i]
                # Get attention output
                if hasattr(layer, 'input_layernorm'):
                    normed = layer.input_layernorm(hidden)
                else:
                    normed = hidden

                # Self attention
                attn_out = layer.self_attn(normed, mask=None, cache=None)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                hidden = hidden + attn_out
                mx.eval(hidden)

                # MLP
                if hasattr(layer, 'post_attention_layernorm'):
                    normed = layer.post_attention_layernorm(hidden)
                else:
                    normed = hidden

                mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
                mlp_out = mlp(normed)
                hidden = hidden + mlp_out
                mx.eval(hidden)

            # Take last token's activation
            act = np.array(hidden[0, -1, :].tolist(), dtype=np.float32)
            activations.append(act)

        return np.vstack(activations)

    def _get_activations_simple(self, model, tokenizer, prompts: List[str]) -> np.ndarray:
        """Simpler activation extraction - just use final logits as proxy for representation."""
        import mlx.core as mx

        activations = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Full forward pass
            logits = model(input_ids)
            mx.eval(logits)

            # Use last token's logits as representation proxy
            # This captures the full model's understanding of the prompt
            act = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            activations.append(act)

        return np.vstack(activations)

    def compare_relational_structure(self, prompts: List[str], category: str) -> Dict:
        """Compare relational structure between models for given prompts."""
        logger.info(f"\nGetting activations for {category}...")

        # Get activations from both models
        broken_acts = self._get_activations_simple(
            self.broken_model, self.broken_tokenizer, prompts
        )
        working_acts = self._get_activations_simple(
            self.working_model, self.working_tokenizer, prompts
        )

        logger.info(f"  Broken activations shape: {broken_acts.shape}")
        logger.info(f"  Working activations shape: {working_acts.shape}")

        # Handle dimension mismatch by projecting to common space
        if broken_acts.shape[1] != working_acts.shape[1]:
            logger.info(f"  Projecting to common dimension...")
            # Use PCA-like projection to match dimensions
            min_dim = min(broken_acts.shape[1], working_acts.shape[1])

            # SVD-based dimensionality alignment
            _, _, Vt_broken = np.linalg.svd(broken_acts, full_matrices=False)
            _, _, Vt_working = np.linalg.svd(working_acts, full_matrices=False)

            # Project to top-k dimensions
            k = min(min_dim, 512)
            broken_proj = broken_acts @ Vt_broken[:k].T
            working_proj = working_acts @ Vt_working[:k].T

            logger.info(f"  Projected to dimension {k}")
        else:
            broken_proj = broken_acts
            working_proj = working_acts

        # Compute CKA - this measures relational structure similarity
        cka = compute_cka(broken_proj, working_proj)
        logger.info(f"  CKA (relational structure alignment): {cka:.4f}")

        # Also compute the Gram matrices for analysis
        broken_gram = broken_proj @ broken_proj.T
        working_gram = working_proj @ working_proj.T

        # Normalize Gram matrices for comparison
        broken_gram_norm = broken_gram / (np.linalg.norm(broken_gram) + 1e-10)
        working_gram_norm = working_gram / (np.linalg.norm(working_gram) + 1e-10)

        # Frobenius norm of difference
        gram_diff = np.linalg.norm(broken_gram_norm - working_gram_norm, 'fro')
        logger.info(f"  Gram matrix difference (Frobenius): {gram_diff:.4f}")

        # Correlation between Gram matrices (flattened)
        gram_corr = np.corrcoef(
            broken_gram_norm.flatten(),
            working_gram_norm.flatten()
        )[0, 1]
        logger.info(f"  Gram matrix correlation: {gram_corr:.4f}")

        return {
            "category": category,
            "n_prompts": len(prompts),
            "cka": float(cka),
            "gram_difference": float(gram_diff),
            "gram_correlation": float(gram_corr),
            "broken_shape": list(broken_acts.shape),
            "working_shape": list(working_acts.shape),
        }

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 49: RELATIONAL STRUCTURE COMPARISON")
        logger.info("=" * 60)
        logger.info("\nComparing relational structure (Gram matrices) between models\n")

        # Compare math operations
        math_result = self.compare_relational_structure(MATH_PROMPTS, "math")

        # Compare non-math operations
        non_math_result = self.compare_relational_structure(NON_MATH_PROMPTS, "non_math")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY: RELATIONAL STRUCTURE ALIGNMENT")
        logger.info("=" * 60)

        logger.info(f"\n| Category | CKA | Gram Corr | Gram Diff |")
        logger.info(f"|----------|-----|-----------|-----------|")
        logger.info(f"| Math     | {math_result['cka']:.3f} | {math_result['gram_correlation']:.3f} | {math_result['gram_difference']:.3f} |")
        logger.info(f"| Non-Math | {non_math_result['cka']:.3f} | {non_math_result['gram_correlation']:.3f} | {non_math_result['gram_difference']:.3f} |")

        cka_diff = non_math_result['cka'] - math_result['cka']
        logger.info(f"\nCKA difference (non-math - math): {cka_diff:.3f}")

        if math_result['cka'] < non_math_result['cka']:
            conclusion = "math_misaligned"
            logger.info("\n*** MATH RELATIONAL STRUCTURE IS MORE MISALIGNED ***")
            logger.info("The Gram matrix (relational structure) of math operations")
            logger.info("is less aligned between models than non-math operations.")
            logger.info("This proves the corruption is in the relational structure itself.")
        elif math_result['cka'] < 0.9:
            conclusion = "both_misaligned"
            logger.info("\n*** BOTH CATEGORIES SHOW MISALIGNMENT ***")
            logger.info("Neither math nor non-math has CKA near 1.0")
        else:
            conclusion = "aligned"
            logger.info("\n*** RELATIONAL STRUCTURES ARE ALIGNED ***")

        results = {
            "math": math_result,
            "non_math": non_math_result,
            "cka_difference": cka_diff,
            "conclusion": conclusion,
        }

        return results


def main():
    from mlx_lm import load

    logger.info("Loading broken model (LFM2-350M)...")
    broken_model, broken_tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("Loading working model (LFM2-1.2B)...")
    working_model, working_tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16")

    experiment = RelationalStructureComparison(
        broken_model, broken_tokenizer,
        working_model, working_tokenizer
    )
    results = experiment.run_experiment()

    output_path = "data/experiments/relational_structure_comparison.json"
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
