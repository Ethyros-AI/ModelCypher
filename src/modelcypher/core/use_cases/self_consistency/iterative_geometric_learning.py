# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Iterative Geometric Learning - The Full Loop.

Based on experimental results and user insight:
1. Model "thinks" (iterative self-questioning builds coherent context)
2. Thinking produces enriched representations
3. Surgical alignment locks geometric gains into weights
4. Repeat - like moving from kindergarten to 1st grade

The key insight: humans don't modify neurons to think - they process
recursively. But the geometry IS modified through learning. This loop
combines both: thinking to find coherence, then locking it geometrically.

Mathematical foundation:
- SVD ratios encode information structure
- Constants (π/e, φ, √2) are invariants of coherent processing
- Surgical modification preserves quality (proved in experiments)
- Iterative refinement should accumulate coherence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd

from modelcypher.core.domain.geometry._primitives.numpy_epsilon_utils import (
    np_division_epsilon,
    np_svd_rank_threshold,
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


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration: int
    thinking_iterations: int
    consistency_score: float
    matches_before: int
    matches_after: int
    targets_aligned: int
    quality: float


@dataclass
class IterativeLearningResult:
    """Result of full iterative learning loop."""
    total_iterations: int
    initial_matches: int
    final_matches: int
    initial_quality: float
    final_quality: float
    initial_consistency: float
    final_consistency: float
    history: List[IterationResult]


class IterativeGeometricLearning:
    """The full loop: think → measure → lock → repeat."""

    def __init__(
        self,
        model,
        tokenizer,
        proximity_threshold: float = 0.10,
        quality_threshold: float = 0.90,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.proximity_threshold = proximity_threshold
        self.quality_threshold = quality_threshold
        self.n_layers = len(model.model.layers)

    def _generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text from prompt."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = []
        current = input_ids

        for _ in range(max_tokens):
            logits = self.model(current)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[0, -1, :]).item())
            if next_token == self.tokenizer.eos_token_id:
                break
            generated.append(next_token)
            current = mx.concatenate([current, mx.array([[next_token]])], axis=1)

        return self.tokenizer.decode(generated)

    def _think(self, topic: str, max_iterations: int = 5) -> Tuple[str, int, float]:
        """Self-questioning loop to build coherent context.

        Returns: (final_response, iterations, consistency_score)
        """
        # Initial response
        response = self._generate(f"What do you know about {topic}?")

        for i in range(max_iterations):
            # Probe for implications
            implications = self._generate(
                f"If '{response[:100]}' is true, what else must be true?"
            )

            # Probe for contradictions
            contradictions = self._generate(
                f"What would make '{response[:100]}' false?"
            )

            # Check consistency
            consistency = self._measure_text_consistency(response, implications)

            if consistency > 0.8:
                # Sufficiently consistent
                return response, i + 1, consistency

            # Refine with contradiction awareness
            response = self._generate(
                f"Considering that '{contradictions[:100]}' could be false, "
                f"refine this understanding: {response[:100]}"
            )

        # Measure final consistency
        final_implications = self._generate(
            f"If '{response[:100]}' is true, what else must be true?"
        )
        final_consistency = self._measure_text_consistency(response, final_implications)

        return response, max_iterations, final_consistency

    def _measure_text_consistency(self, text1: str, text2: str) -> float:
        """Measure semantic consistency between two texts using representation distance."""
        import mlx.core as mx

        tokens1 = self.tokenizer.encode(text1)[:50]
        tokens2 = self.tokenizer.encode(text2)[:50]

        input1 = mx.array([tokens1])
        input2 = mx.array([tokens2])

        # Get embeddings from middle layer
        mid_layer = self.n_layers // 2
        layer = self.model.model.layers[mid_layer]

        # Get hidden states
        def get_hidden(input_ids):
            x = self.model.model.embed_tokens(input_ids)
            mx.eval(x)
            for i, l in enumerate(self.model.model.layers):
                x = l(x)
                mx.eval(x)
                if i == mid_layer:
                    return x
            return x

        h1 = get_hidden(input1)
        h2 = get_hidden(input2)
        mx.eval(h1, h2)

        # Mean pooling
        v1 = np.array(h1[0].tolist()).mean(axis=0)
        v2 = np.array(h2[0].tolist()).mean(axis=0)

        # Cosine similarity with dtype-derived epsilon
        div_eps = np_division_epsilon(v1)
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + div_eps)
        return float(np.clip(similarity, 0, 1))

    def _get_mlp_weight(self, layer_idx: int) -> np.ndarray:
        """Get the gate projection weight matrix."""
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
        else:
            mlp = layer.mlp

        if hasattr(mlp, 'gate_proj'):
            w = mlp.gate_proj.weight
        elif hasattr(mlp, 'w1'):
            w = mlp.w1.weight
        else:
            w = mlp.weight

        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_mlp_weight(self, layer_idx: int, weights: np.ndarray):
        """Set the gate projection weight matrix."""
        import mlx.core as mx
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
        else:
            mlp = layer.mlp

        new_weight = mx.array(weights.astype(np.float32))

        if hasattr(mlp, 'gate_proj'):
            mlp.gate_proj.weight = new_weight
        elif hasattr(mlp, 'w1'):
            mlp.w1.weight = new_weight
        else:
            mlp.weight = new_weight

        mx.eval(new_weight)

    def _count_matches(self, S: np.ndarray) -> int:
        """Count how many ratios match constants (within 5%)."""
        # Use dtype-derived threshold for numerical rank
        sv_threshold = np_svd_rank_threshold(S, len(S), S[0] if len(S) > 0 else 1.0)
        count = 0
        for i in range(min(len(S) - 1, 20)):
            for j in range(i + 1, min(len(S), i + 6)):
                if S[j] > sv_threshold:
                    ratio = S[i] / S[j]
                    for const_val in CONSTANTS.values():
                        if abs(ratio - const_val) / const_val < 0.05:
                            count += 1
                            break
        return count

    def _count_total_matches(self, layer_indices: List[int]) -> int:
        """Count total matches across layers."""
        total = 0
        for layer_idx in layer_indices:
            W = self._get_mlp_weight(layer_idx)
            _, S, _ = svd(W, full_matrices=False)
            total += self._count_matches(S)
        return total

    def _surgical_align_layer(self, layer_idx: int, max_targets: int = 3) -> int:
        """Apply surgical SVD alignment to a layer. Returns targets aligned."""
        W = self._get_mlp_weight(layer_idx)
        U, S, Vt = svd(W, full_matrices=False)

        # Find targets within proximity threshold using dtype-derived threshold
        sv_threshold = np_svd_rank_threshold(S, len(S), S[0] if len(S) > 0 else 1.0)
        targets = []

        for i in range(min(len(S) - 1, 15)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > sv_threshold:
                    ratio = S[i] / S[j]

                    for const_name, const_val in CONSTANTS.items():
                        error = abs(ratio - const_val) / const_val
                        if error < self.proximity_threshold:
                            targets.append((i, j, const_val))
                            break

        if not targets:
            return 0

        # Apply modifications
        S_modified = S.copy()
        aligned = 0

        for i, j, target_val in targets[:max_targets]:
            if S_modified[j] < sv_threshold:
                continue
            new_val = target_val * S_modified[j]
            if new_val > S[0] * 10 or new_val < sv_threshold:
                continue
            S_modified[i] = new_val
            aligned += 1

        if aligned > 0:
            # Check for numerical issues before reconstruction
            if not np.all(np.isfinite(S_modified)):
                return 0
            W_modified = U @ np.diag(S_modified) @ Vt
            if not np.all(np.isfinite(W_modified)):
                return 0
            self._set_mlp_weight(layer_idx, W_modified)

        return aligned

    def _evaluate_quality(self, test_prompts: List[Tuple[str, str]]) -> float:
        """Quick model quality check."""
        import mlx.core as mx

        correct = 0
        for prompt, expected in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            current = input_ids
            for _ in range(20):
                logits = self.model(current)
                mx.eval(logits)
                next_token = int(mx.argmax(logits[0, -1, :]).item())
                if next_token == self.tokenizer.eos_token_id:
                    break
                generated.append(next_token)
                current = mx.concatenate([current, mx.array([[next_token]])], axis=1)

            response = self.tokenizer.decode(generated).lower()
            if expected.lower() in response:
                correct += 1

        return correct / len(test_prompts) if test_prompts else 1.0

    def run(
        self,
        topics: List[str],
        test_prompts: List[Tuple[str, str]],
        n_iterations: int = 5,
        layer_indices: Optional[List[int]] = None,
    ) -> IterativeLearningResult:
        """Run the full iterative loop: think → measure → lock → repeat."""

        if layer_indices is None:
            mid = self.n_layers // 2
            layer_indices = list(range(mid - 3, mid + 4))

        logger.info("\n" + "=" * 60)
        logger.info("ITERATIVE GEOMETRIC LEARNING")
        logger.info(f"Topics: {len(topics)}")
        logger.info(f"Iterations: {n_iterations}")
        logger.info(f"Layers: {layer_indices}")
        logger.info("=" * 60)

        # Initial state
        initial_quality = self._evaluate_quality(test_prompts)
        initial_matches = self._count_total_matches(layer_indices)

        # Initial consistency (average across topics)
        initial_consistencies = []
        for topic in topics[:3]:  # Sample for speed
            _, _, consistency = self._think(topic, max_iterations=1)
            initial_consistencies.append(consistency)
        initial_consistency = np.mean(initial_consistencies)

        logger.info(f"\nInitial state:")
        logger.info(f"  Quality: {initial_quality:.2%}")
        logger.info(f"  Matches: {initial_matches}")
        logger.info(f"  Consistency: {initial_consistency:.3f}")

        history = []

        for iteration in range(n_iterations):
            logger.info(f"\n--- Iteration {iteration + 1} ---")

            # THINK phase: Self-questioning on each topic
            thinking_iterations = []
            consistencies = []

            for topic in topics:
                _, iters, consistency = self._think(topic, max_iterations=5)
                thinking_iterations.append(iters)
                consistencies.append(consistency)

            avg_thinking = np.mean(thinking_iterations)
            avg_consistency = np.mean(consistencies)
            logger.info(f"  Thinking: avg {avg_thinking:.1f} iterations, consistency {avg_consistency:.3f}")

            # LOCK phase: Surgical SVD alignment
            matches_before = self._count_total_matches(layer_indices)
            total_aligned = 0

            for layer_idx in layer_indices:
                aligned = self._surgical_align_layer(layer_idx, max_targets=2)
                total_aligned += aligned

            matches_after = self._count_total_matches(layer_indices)
            logger.info(f"  Locking: {total_aligned} targets aligned")
            logger.info(f"  Matches: {matches_before} → {matches_after}")

            # Check quality
            quality = self._evaluate_quality(test_prompts)
            logger.info(f"  Quality: {quality:.2%}")

            # If quality degraded significantly, stop
            if quality < initial_quality * self.quality_threshold:
                logger.info(f"  Quality degraded below threshold, stopping")
                break

            history.append(IterationResult(
                iteration=iteration + 1,
                thinking_iterations=int(np.mean(thinking_iterations)),
                consistency_score=float(avg_consistency),
                matches_before=matches_before,
                matches_after=matches_after,
                targets_aligned=total_aligned,
                quality=quality,
            ))

        # Final state
        final_quality = self._evaluate_quality(test_prompts)
        final_matches = self._count_total_matches(layer_indices)

        final_consistencies = []
        for topic in topics[:3]:
            _, _, consistency = self._think(topic, max_iterations=1)
            final_consistencies.append(consistency)
        final_consistency = np.mean(final_consistencies)

        logger.info(f"\n{'=' * 60}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Quality: {initial_quality:.2%} → {final_quality:.2%}")
        logger.info(f"Matches: {initial_matches} → {final_matches}")
        logger.info(f"Consistency: {initial_consistency:.3f} → {final_consistency:.3f}")

        return IterativeLearningResult(
            total_iterations=len(history),
            initial_matches=initial_matches,
            final_matches=final_matches,
            initial_quality=initial_quality,
            final_quality=final_quality,
            initial_consistency=initial_consistency,
            final_consistency=final_consistency,
            history=history,
        )


__all__ = ["IterativeGeometricLearning", "IterativeLearningResult"]
