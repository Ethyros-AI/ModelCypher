# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Geometric Learning - Direct manipulation of activation geometry.

The insight: tokens are shadows on the cave wall. Geometry is the fire.
We need to work directly with geometry, not with token outputs.

The model has:
1. Current weight geometry (SVD ratios)
2. Activation geometry for any input
3. The ability to predict what comes next (logits)

The hypothesis: if we can find the direction in weight space that:
- Preserves the model's predictions (logits stay similar)
- Improves geometric alignment (more SVD ratios match constants)

Then we can iteratively improve without degrading capabilities.

This is like gradient descent, but the loss is geometric alignment
and the constraint is prediction preservation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Fundamental constants
CONSTANTS = {
    "pi/e": 1.1557,
    "e/pi": 0.8653,
    "phi": 1.6180,
    "sqrt2": 1.4142,
}


@dataclass
class GeometricLearningResult:
    """Result of geometric learning."""

    n_iterations: int
    initial_matches: int
    final_matches: int
    initial_quality: float
    final_quality: float
    quality_preserved: bool
    trajectory: List[Dict]


class GeometricLearning:
    """Direct geometric learning through weight space exploration.

    The loop:
    1. Compute current SVD signature
    2. Find directions that would improve constant alignment
    3. Project out directions that would change predictions
    4. Take small step
    5. Verify quality preserved
    6. Repeat

    This is fundamentally different from token-based training:
    - We're not optimizing for next-token prediction
    - We're optimizing for geometric coherence
    - The constraint is: don't break what already works
    """

    def __init__(
        self,
        model,
        tokenizer,
        backend,
        layer_indices: Optional[List[int]] = None,
        step_size: float = 0.001,
        quality_threshold: float = 0.9,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.step_size = step_size
        self.quality_threshold = quality_threshold

        self.n_layers = len(model.model.layers)
        if layer_indices is None:
            mid = self.n_layers // 2
            layer_indices = [mid - 1, mid, mid + 1]
        self.layer_indices = layer_indices

    def _get_weights(self, layer_idx: int) -> np.ndarray:
        """Get MLP weights for a layer."""
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            if hasattr(layer.feed_forward, 'gate_proj'):
                w = layer.feed_forward.gate_proj.weight
            elif hasattr(layer.feed_forward, 'w1'):
                w = layer.feed_forward.w1.weight
            else:
                w = layer.feed_forward.weight
        else:
            if hasattr(layer.mlp, 'gate_proj'):
                w = layer.mlp.gate_proj.weight
            elif hasattr(layer.mlp, 'w1'):
                w = layer.mlp.w1.weight
            else:
                w = layer.mlp.weight

        mx.eval(w)
        return np.array(w.tolist(), dtype=np.float32)

    def _set_weights(self, layer_idx: int, weights: np.ndarray) -> None:
        """Set MLP weights for a layer."""
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]
        new_weight = mx.array(weights)

        if hasattr(layer, 'feed_forward'):
            if hasattr(layer.feed_forward, 'gate_proj'):
                layer.feed_forward.gate_proj.weight = new_weight
            elif hasattr(layer.feed_forward, 'w1'):
                layer.feed_forward.w1.weight = new_weight
            else:
                layer.feed_forward.weight = new_weight
        else:
            if hasattr(layer.mlp, 'gate_proj'):
                layer.mlp.gate_proj.weight = new_weight
            elif hasattr(layer.mlp, 'w1'):
                layer.mlp.w1.weight = new_weight
            else:
                layer.mlp.weight = new_weight

        mx.eval(new_weight)

    def _compute_svd_signature(self, W: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Compute SVD signature of weight matrix.

        Returns:
            (n_matches, mean_error, singular_values)
        """
        from scipy.linalg import svd

        try:
            _, S, _ = svd(W, full_matrices=False)
        except:
            return 0, 100.0, np.array([1.0])

        if len(S) < 2:
            return 0, 100.0, S

        n_matches = 0
        match_errors = []

        for i in range(min(len(S) - 1, 20)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > 1e-10:
                    ratio = S[i] / S[j]

                    min_error = float('inf')
                    for const_val in CONSTANTS.values():
                        error = abs(ratio - const_val) / const_val * 100
                        if error < min_error:
                            min_error = error

                    if min_error < 5.0:
                        n_matches += 1
                        match_errors.append(min_error)

        mean_error = sum(match_errors) / len(match_errors) if match_errors else 100.0

        return n_matches, mean_error, S

    def _compute_geometric_gradient(
        self,
        W: np.ndarray,
        target_constant: float = 0.8653,  # e/π by default
    ) -> np.ndarray:
        """Compute direction that would improve SVD ratio alignment.

        This finds the weight perturbation that would push the dominant
        SVD ratio toward a fundamental constant.

        Returns:
            Gradient direction (same shape as W)
        """
        from scipy.linalg import svd

        try:
            U, S, Vt = svd(W, full_matrices=False)
        except:
            return np.zeros_like(W)

        if len(S) < 2:
            return np.zeros_like(W)

        # Current ratio of first two singular values
        current_ratio = S[0] / S[1] if S[1] > 1e-10 else 1.0

        # How far are we from target?
        error = current_ratio - target_constant

        # To change the ratio S[0]/S[1]:
        # - Increasing S[0] increases ratio
        # - Increasing S[1] decreases ratio
        #
        # Gradient of ratio w.r.t. S[0] = 1/S[1]
        # Gradient of ratio w.r.t. S[1] = -S[0]/S[1]^2
        #
        # To decrease ratio (if current > target), we want to:
        # - Decrease S[0] or increase S[1]
        #
        # The direction that changes S[i] is: U[:, i] @ Vt[i, :]

        # Direction to decrease S[0]
        dir_s0 = np.outer(U[:, 0], Vt[0, :])

        # Direction to increase S[1]
        dir_s1 = np.outer(U[:, 1], Vt[1, :])

        if error > 0:
            # Ratio too high - decrease S[0] or increase S[1]
            gradient = -dir_s0 + dir_s1
        else:
            # Ratio too low - increase S[0] or decrease S[1]
            gradient = dir_s0 - dir_s1

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm

        return gradient

    def _get_logits(self, text: str) -> np.ndarray:
        """Get model's logit predictions for text."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        logits = self.model(input_ids)
        mx.eval(logits)

        # Return logits for last position
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    def _evaluate_quality(self, test_prompts: List[Tuple[str, str]]) -> float:
        """Evaluate model quality on test prompts."""
        import mlx.core as mx

        correct = 0
        for prompt, expected in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            current = input_ids
            for _ in range(30):
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

    def _project_out_prediction_change(
        self,
        gradient: np.ndarray,
        W: np.ndarray,
        reference_logits: List[np.ndarray],
        test_texts: List[str],
    ) -> np.ndarray:
        """Project out components of gradient that would change predictions.

        We want to move in weight space without changing what the model predicts.
        This finds the component of the gradient that's orthogonal to the
        prediction-preserving subspace.
        """
        # This is a simplification - full implementation would compute
        # Jacobian of logits w.r.t. weights and project out

        # For now, we just ensure the step is small enough that predictions
        # don't change much, which is controlled by step_size

        return gradient

    def run_iteration(
        self,
        test_prompts: List[Tuple[str, str]],
        iteration: int,
    ) -> Dict:
        """Run a single iteration of geometric learning."""

        logger.info(f"\n--- Iteration {iteration} ---")

        # Measure current state
        total_matches_before = 0
        layer_states = {}

        for layer_idx in self.layer_indices:
            W = self._get_weights(layer_idx)
            n_matches, mean_error, S = self._compute_svd_signature(W)
            total_matches_before += n_matches
            layer_states[layer_idx] = {
                'W': W,
                'n_matches': n_matches,
                'S': S,
            }

        quality_before = self._evaluate_quality(test_prompts)
        logger.info(f"Before: {total_matches_before} matches, quality={quality_before:.2%}")

        # Compute and apply geometric gradients
        for layer_idx in self.layer_indices:
            W = layer_states[layer_idx]['W']

            # Try multiple target constants and pick the one closest
            best_gradient = None
            best_target_error = float('inf')

            from scipy.linalg import svd
            try:
                _, S, _ = svd(W, full_matrices=False)
                if len(S) >= 2:
                    current_ratio = S[0] / S[1]

                    for const_val in CONSTANTS.values():
                        error = abs(current_ratio - const_val)
                        if error < best_target_error:
                            best_target_error = error
                            best_gradient = self._compute_geometric_gradient(W, const_val)
            except:
                pass

            if best_gradient is not None:
                # Apply small step
                new_W = W + self.step_size * best_gradient
                self._set_weights(layer_idx, new_W)

        # Measure new state
        total_matches_after = 0
        for layer_idx in self.layer_indices:
            W = self._get_weights(layer_idx)
            n_matches, _, _ = self._compute_svd_signature(W)
            total_matches_after += n_matches

        quality_after = self._evaluate_quality(test_prompts)
        logger.info(f"After: {total_matches_after} matches, quality={quality_after:.2%}")

        # Rollback if quality degraded
        if quality_after < quality_before * self.quality_threshold:
            logger.info("Quality degraded - rolling back")
            for layer_idx in self.layer_indices:
                self._set_weights(layer_idx, layer_states[layer_idx]['W'])
            total_matches_after = total_matches_before
            quality_after = quality_before
            rolled_back = True
        else:
            rolled_back = False

        return {
            'iteration': iteration,
            'matches_before': total_matches_before,
            'matches_after': total_matches_after,
            'quality_before': float(quality_before),
            'quality_after': float(quality_after),
            'rolled_back': rolled_back,
        }

    def run(
        self,
        test_prompts: List[Tuple[str, str]],
        n_iterations: int = 20,
    ) -> GeometricLearningResult:
        """Run geometric learning loop."""

        logger.info("\n" + "="*60)
        logger.info("GEOMETRIC LEARNING")
        logger.info(f"Layers: {self.layer_indices}")
        logger.info(f"Step size: {self.step_size}")
        logger.info("="*60)

        # Initial state
        initial_matches = 0
        for layer_idx in self.layer_indices:
            W = self._get_weights(layer_idx)
            n_matches, _, _ = self._compute_svd_signature(W)
            initial_matches += n_matches

        initial_quality = self._evaluate_quality(test_prompts)

        logger.info(f"\nInitial: {initial_matches} matches, quality={initial_quality:.2%}")

        # Run iterations
        trajectory = []
        no_progress_count = 0

        for i in range(n_iterations):
            result = self.run_iteration(test_prompts, i + 1)
            trajectory.append(result)

            # Check for progress
            if result['matches_after'] <= result['matches_before']:
                no_progress_count += 1
            else:
                no_progress_count = 0

            if no_progress_count >= 5:
                logger.info("No progress for 5 iterations - stopping")
                break

        # Final state
        final_matches = 0
        for layer_idx in self.layer_indices:
            W = self._get_weights(layer_idx)
            n_matches, _, _ = self._compute_svd_signature(W)
            final_matches += n_matches

        final_quality = self._evaluate_quality(test_prompts)

        logger.info(f"\n{'='*60}")
        logger.info("RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Matches: {initial_matches} → {final_matches}")
        logger.info(f"Quality: {initial_quality:.2%} → {final_quality:.2%}")

        return GeometricLearningResult(
            n_iterations=len(trajectory),
            initial_matches=initial_matches,
            final_matches=final_matches,
            initial_quality=float(initial_quality),
            final_quality=float(final_quality),
            quality_preserved=final_quality >= initial_quality * self.quality_threshold,
            trajectory=trajectory,
        )


__all__ = ["GeometricLearning", "GeometricLearningResult"]
