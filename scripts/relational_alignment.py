#!/usr/bin/env python3
"""Experiment 53: Relational Alignment.

Compression was wrong. Compressing a misunderstanding just makes it
more confidently wrong.

The real problem: The model's relational structure for math POINTS THE
WRONG DIRECTION. 1+n should relate to (n+1), but the model has 1+n
relating to n.

The fix: Find the CORRECT relational structure from a working model,
compute the transform that rotates broken→correct, and apply it.

This is alignment, not compression.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import svd, lstsq, orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Arithmetic facts with KNOWN correct relationships
# These define what the relational structure SHOULD be
ARITHMETIC_PAIRS = [
    # Addition - these should have specific relationships
    ("1+1=", 2), ("1+2=", 3), ("1+3=", 4), ("1+4=", 5), ("1+5=", 6),
    ("2+1=", 3), ("2+2=", 4), ("2+3=", 5), ("2+4=", 6), ("2+5=", 7),
    ("3+1=", 4), ("3+2=", 5), ("3+3=", 6), ("3+4=", 7), ("3+5=", 8),
    ("4+1=", 5), ("4+2=", 6), ("4+3=", 7), ("4+4=", 8), ("4+5=", 9),
    ("5+1=", 6), ("5+2=", 7), ("5+3=", 8), ("5+4=", 9), ("5+5=", 10),
]


class RelationalAligner:
    """Align broken relational structure to match working model."""

    def __init__(self, broken_model, broken_tokenizer, working_model, working_tokenizer):
        self.broken_model = broken_model
        self.broken_tokenizer = broken_tokenizer
        self.working_model = working_model
        self.working_tokenizer = working_tokenizer
        self.broken_n_layers = len(broken_model.model.layers)
        self.working_n_layers = len(working_model.model.layers)
        self._original_weights = {}

    def _get_activations(self, model, tokenizer, prompts: List[str]) -> np.ndarray:
        """Get logit activations for prompts."""
        import mlx.core as mx

        activations = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            mx.eval(logits)
            act = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            activations.append(act)
        return np.vstack(activations)

    def _get_hidden_states(self, model, tokenizer, prompts: List[str], layer_frac: float = 0.5) -> np.ndarray:
        """Get hidden state activations at a specific layer."""
        import mlx.core as mx

        n_layers = len(model.model.layers)
        layer_idx = int(n_layers * layer_frac)

        activations = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            hidden = model.model.embed_tokens(input_ids)
            mx.eval(hidden)

            for i in range(layer_idx + 1):
                layer = model.model.layers[i]
                if hasattr(layer, 'input_layernorm'):
                    normed = layer.input_layernorm(hidden)
                else:
                    normed = hidden

                attn_out = layer.self_attn(normed, mask=None, cache=None)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                hidden = hidden + attn_out
                mx.eval(hidden)

                if hasattr(layer, 'post_attention_layernorm'):
                    normed = layer.post_attention_layernorm(hidden)
                else:
                    normed = hidden

                mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
                mlp_out = mlp(normed)
                hidden = hidden + mlp_out
                mx.eval(hidden)

            act = np.array(hidden[0, -1, :].tolist(), dtype=np.float32)
            activations.append(act)

        return np.vstack(activations)

    def _evaluate_math(self, model, tokenizer) -> Tuple[int, int, List]:
        """Evaluate math accuracy."""
        import mlx.core as mx
        correct = 0
        results = []

        for prompt, expected in ARITHMETIC_PAIRS:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            mx.eval(logits)
            next_logits = logits[0, -1, :]

            # Get probabilities for numbers 0-20
            probs = []
            for num in range(21):
                num_str = str(num)
                token_ids = tokenizer.encode(num_str)
                if token_ids:
                    prob = float(next_logits[token_ids[-1]].item())
                    probs.append((num, prob))

            probs.sort(key=lambda x: x[1], reverse=True)
            predicted = probs[0][0] if probs else -1

            is_correct = predicted == expected
            if is_correct:
                correct += 1
            results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })

        return correct, len(ARITHMETIC_PAIRS), results

    def compute_relational_structure(self, activations: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix (relational structure) of activations."""
        # Center activations
        centered = activations - activations.mean(axis=0)
        # Gram matrix captures relationships
        G = centered @ centered.T
        return G

    def compute_alignment_transform(self, broken_acts: np.ndarray, working_acts: np.ndarray) -> np.ndarray:
        """Compute the Procrustes transform that aligns broken→working structure.

        This finds the orthogonal matrix R such that broken @ R ≈ working
        """
        # Always reduce to manageable dimension for Procrustes
        k = min(broken_acts.shape[0], 256)  # Can't have more dims than samples

        # Project both to k dimensions using their own principal components
        _, _, Vt_broken = np.linalg.svd(broken_acts, full_matrices=False)
        _, _, Vt_working = np.linalg.svd(working_acts, full_matrices=False)

        broken_proj = broken_acts @ Vt_broken[:k].T
        working_proj = working_acts @ Vt_working[:k].T

        # Orthogonal Procrustes: find R that minimizes ||broken @ R - working||
        R, scale = orthogonal_procrustes(broken_proj, working_proj)

        return R, scale, broken_proj, working_proj

    def _get_weight(self, model, layer_idx: int) -> np.ndarray:
        import mlx.core as mx
        layer = model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        w = mlp.gate_proj.weight if hasattr(mlp, 'gate_proj') else (mlp.w1.weight if hasattr(mlp, 'w1') else mlp.weight)
        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_weight(self, model, layer_idx: int, weights: np.ndarray):
        import mlx.core as mx
        layer = model.model.layers[layer_idx]
        mlp = layer.feed_forward if hasattr(layer, 'feed_forward') else layer.mlp
        new_weight = mx.array(weights.astype(np.float32))
        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight
        mx.eval(new_weight)

    def _cache_weights(self, model, layers: List[int]):
        self._original_weights = {i: self._get_weight(model, i).copy() for i in layers}

    def _reset_weights(self, model, layers: List[int]):
        for i in layers:
            if i in self._original_weights:
                self._set_weight(model, i, self._original_weights[i])

    def run_experiment(self) -> Dict:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 53: RELATIONAL ALIGNMENT")
        logger.info("=" * 60)
        logger.info("\nCompression was wrong - we need to ROTATE the structure, not compress it.\n")

        prompts = [p for p, _ in ARITHMETIC_PAIRS]

        # Get baseline accuracy
        logger.info("Evaluating broken model baseline...")
        broken_correct, broken_total, broken_results = self._evaluate_math(
            self.broken_model, self.broken_tokenizer
        )
        broken_acc = broken_correct / broken_total
        logger.info(f"  Broken model: {broken_correct}/{broken_total} ({broken_acc:.0%})")

        logger.info("\nEvaluating working model baseline...")
        working_correct, working_total, working_results = self._evaluate_math(
            self.working_model, self.working_tokenizer
        )
        working_acc = working_correct / working_total
        logger.info(f"  Working model: {working_correct}/{working_total} ({working_acc:.0%})")

        # Get activations from both models
        logger.info("\nGetting activations...")
        broken_acts = self._get_activations(self.broken_model, self.broken_tokenizer, prompts)
        working_acts = self._get_activations(self.working_model, self.working_tokenizer, prompts)
        logger.info(f"  Broken: {broken_acts.shape}")
        logger.info(f"  Working: {working_acts.shape}")

        # Compute relational structures
        logger.info("\nComputing relational structures (Gram matrices)...")
        broken_gram = self.compute_relational_structure(broken_acts)
        working_gram = self.compute_relational_structure(working_acts)

        # Measure misalignment
        broken_gram_norm = broken_gram / (np.linalg.norm(broken_gram, 'fro') + 1e-10)
        working_gram_norm = working_gram / (np.linalg.norm(working_gram, 'fro') + 1e-10)
        gram_corr = np.corrcoef(broken_gram_norm.flatten(), working_gram_norm.flatten())[0, 1]
        logger.info(f"  Initial Gram correlation: {gram_corr:.4f}")

        # Compute alignment transform
        logger.info("\nComputing Procrustes alignment (broken → working)...")
        R, scale, broken_proj, working_proj = self.compute_alignment_transform(broken_acts, working_acts)
        logger.info(f"  Scale factor: {scale:.4f}")

        # Apply alignment to activations and measure new structure
        aligned_acts = broken_proj @ R
        aligned_gram = self.compute_relational_structure(aligned_acts)

        # Compare to working gram (need same dimension)
        working_gram_proj = self.compute_relational_structure(working_proj)
        aligned_gram_norm = aligned_gram / (np.linalg.norm(aligned_gram, 'fro') + 1e-10)
        working_gram_proj_norm = working_gram_proj / (np.linalg.norm(working_gram_proj, 'fro') + 1e-10)
        aligned_corr = np.corrcoef(aligned_gram_norm.flatten(), working_gram_proj_norm.flatten())[0, 1]
        logger.info(f"  Aligned Gram correlation: {aligned_corr:.4f}")

        # The key insight: what IS the transform R doing?
        # R rotates the broken representation space to match working
        logger.info("\nAnalyzing the alignment transform R...")
        U_R, S_R, Vt_R = svd(R)
        logger.info(f"  R is {R.shape[0]}×{R.shape[1]}")
        logger.info(f"  Top 5 singular values of R: {S_R[:5]}")
        logger.info(f"  R is {'orthogonal' if np.allclose(R @ R.T, np.eye(R.shape[0])) else 'not orthogonal'}")

        # Can we learn what R does to specific arithmetic relationships?
        logger.info("\nWhat does alignment change about specific facts?")
        for i, (prompt, expected) in enumerate(ARITHMETIC_PAIRS[:10]):
            broken_vec = broken_proj[i]
            working_vec = working_proj[i]
            aligned_vec = aligned_acts[i]

            # Similarity before and after
            sim_before = np.dot(broken_vec, working_vec) / (np.linalg.norm(broken_vec) * np.linalg.norm(working_vec) + 1e-10)
            sim_after = np.dot(aligned_vec, working_vec) / (np.linalg.norm(aligned_vec) * np.linalg.norm(working_vec) + 1e-10)

            logger.info(f"  {prompt}{expected}: sim {sim_before:.3f} → {sim_after:.3f}")

        results = {
            "broken_accuracy": broken_acc,
            "working_accuracy": working_acc,
            "initial_gram_correlation": float(gram_corr),
            "aligned_gram_correlation": float(aligned_corr),
            "alignment_scale": float(scale),
            "alignment_improves_structure": aligned_corr > gram_corr,
            "broken_results": broken_results,
            "working_results": working_results,
        }

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info("=" * 60)

        if aligned_corr > gram_corr:
            logger.info(f"\n*** ALIGNMENT IMPROVES RELATIONAL STRUCTURE ***")
            logger.info(f"Gram correlation: {gram_corr:.4f} → {aligned_corr:.4f}")
            logger.info(f"\nThe transform R rotates broken math toward working math.")
            logger.info(f"This is the CORRECT approach - not compression, but rotation.")
            results["conclusion"] = "alignment_works"
        else:
            logger.info(f"\n*** ALIGNMENT DID NOT IMPROVE STRUCTURE ***")
            results["conclusion"] = "alignment_failed"

        return results


def main():
    from mlx_lm import load

    logger.info("Loading broken model (LFM2-350M)...")
    broken_model, broken_tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("Loading working model (LFM2-1.2B)...")
    working_model, working_tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16")

    experiment = RelationalAligner(
        broken_model, broken_tokenizer,
        working_model, working_tokenizer
    )
    results = experiment.run_experiment()

    output_path = "data/experiments/relational_alignment.json"
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
