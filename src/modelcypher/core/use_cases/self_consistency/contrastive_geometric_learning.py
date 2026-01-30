# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Contrastive Geometric Learning - Let the model teach itself.

The key insight: the model already "knows" when it's processing coherent
vs incoherent information. Its own activations differ between:
- True statements it's confident about
- False/nonsensical statements
- Things it's uncertain about

We can use this contrastive signal to guide geometric learning:
1. Process COHERENT statements → measure geometry
2. Process INCOHERENT statements → measure geometry
3. Find the direction that makes coherent geometry more constant-aligned
4. Update weights to amplify this direction

This is self-supervised: the model's own confidence/coherence signals
guide the geometric update. No external labels needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from modelcypher.core.domain.geometry._primitives.numpy_epsilon_utils import (
    np_division_epsilon,
    np_svd_rank_threshold,
)

logger = logging.getLogger(__name__)


# Fundamental constants
CONSTANTS = {
    "pi/e": 1.1557,
    "e/pi": 0.8653,
    "phi": 1.6180,
    "sqrt2": 1.4142,
}

# Coherent statements - things the model should "know"
COHERENT_PROBES = [
    "Two plus two equals four.",
    "Water freezes at zero degrees Celsius.",
    "The Earth orbits the Sun.",
    "Paris is the capital of France.",
    "Triangles have three sides.",
    "Light travels faster than sound.",
]

# Incoherent statements - things that should show different geometry
INCOHERENT_PROBES = [
    "Colorless green ideas sleep furiously.",
    "The square root of purple is banana.",
    "Yesterday I will have gone tomorrow.",
    "France is the capital of Paris.",
    "Two plus two equals five.",
    "Water boils at negative temperatures.",
]


@dataclass
class ContrastiveLearningResult:
    """Result of contrastive geometric learning."""

    n_iterations: int
    initial_coherent_matches: int
    final_coherent_matches: int
    initial_incoherent_matches: int
    final_incoherent_matches: int
    initial_contrast: int  # coherent - incoherent
    final_contrast: int
    initial_quality: float
    final_quality: float
    trajectory: List[Dict]


class ContrastiveGeometricLearning:
    """Learn geometry by contrasting coherent vs incoherent processing.

    The loop:
    1. Process coherent probes → get activations → measure geometry
    2. Process incoherent probes → get activations → measure geometry
    3. Find direction that maximizes: coherent_geometry - incoherent_geometry
    4. Update weights in that direction
    5. Verify quality preserved
    6. Repeat

    The hypothesis: coherent processing naturally aligns with fundamental
    constants. By maximizing the contrast between coherent and incoherent,
    we push the model toward the invariant geometric structure.
    """

    def __init__(
        self,
        model,
        tokenizer,
        backend,
        layer_indices: Optional[List[int]] = None,
        step_size: float = 0.01,
        quality_threshold: float = 0.8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.step_size = step_size
        self.quality_threshold = quality_threshold

        self.n_layers = len(model.model.layers)
        if layer_indices is None:
            # Use middle layers
            mid = self.n_layers // 2
            layer_indices = list(range(mid - 2, mid + 3))
        self.layer_indices = layer_indices

    def _get_activations(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP activations for a layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        captured = {}

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = self.model(input_ids)
            mx.eval(captured['output'])
            return np.array(captured['output'][0].tolist(), dtype=np.float32)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

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

    def _compute_activation_geometry(
        self,
        probes: List[str],
        layer_idx: int,
    ) -> Tuple[int, float]:
        """Compute geometry metrics from activations.

        Returns:
            (n_constant_matches, mean_match_error)
        """
        from scipy.linalg import svd

        # Collect activations
        all_acts = []
        for probe in probes:
            act = self._get_activations(probe, layer_idx)
            all_acts.append(act)

        # Stack into matrix
        activations = np.vstack(all_acts)
        centered = activations - activations.mean(axis=0)

        try:
            _, S, _ = svd(centered, full_matrices=False)
        except:
            return 0, 100.0

        if len(S) < 2:
            return 0, 100.0

        # Count constant matches using dtype-derived threshold
        sv_threshold = np_svd_rank_threshold(S, len(S), S[0] if len(S) > 0 else 1.0)
        n_matches = 0
        match_errors = []

        for i in range(min(len(S) - 1, 15)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > sv_threshold:
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

        return n_matches, mean_error

    def _compute_contrastive_direction(
        self,
        layer_idx: int,
        coherent_probes: List[str],
        incoherent_probes: List[str],
    ) -> np.ndarray:
        """Find the direction that maximizes geometric contrast.

        The direction should:
        - Make coherent activations more constant-aligned
        - Make incoherent activations less constant-aligned

        This is the "learning signal" from the model itself.
        """
        from scipy.linalg import svd

        # Get activation covariances for both sets
        coherent_acts = []
        for probe in coherent_probes:
            act = self._get_activations(probe, layer_idx)
            coherent_acts.append(act.mean(axis=0))  # Mean across positions
        coherent_acts = np.vstack(coherent_acts)

        incoherent_acts = []
        for probe in incoherent_probes:
            act = self._get_activations(probe, layer_idx)
            incoherent_acts.append(act.mean(axis=0))
        incoherent_acts = np.vstack(incoherent_acts)

        # The direction we want amplifies coherent patterns
        # and diminishes incoherent patterns

        # Compute covariance matrices
        coh_cov = np.cov(coherent_acts.T)
        inc_cov = np.cov(incoherent_acts.T)

        # Contrastive direction: eigenvectors of (coh_cov - inc_cov)
        # This finds directions that are strong in coherent but weak in incoherent
        try:
            diff_cov = coh_cov - inc_cov
            eigenvalues, eigenvectors = np.linalg.eigh(diff_cov)

            # Take top eigenvector (largest positive eigenvalue = most contrastive)
            idx = np.argmax(eigenvalues)
            direction = eigenvectors[:, idx]

            # Normalize using dtype-derived epsilon
            norm = np.linalg.norm(direction)
            div_eps = np_division_epsilon(direction)
            if norm > div_eps:
                direction = direction / norm

            return direction
        except:
            return np.zeros(coherent_acts.shape[1])

    def _project_to_weight_update(
        self,
        direction: np.ndarray,
        layer_idx: int,
    ) -> np.ndarray:
        """Project activation direction to weight update.

        The direction is in activation space. We need to find the
        weight change that would amplify activations in this direction.
        """
        W = self._get_weights(layer_idx)

        # Simple approach: outer product of direction with itself
        # This amplifies the direction in the weight matrix
        if len(direction) == W.shape[0]:
            # direction is output dimension
            delta = self.step_size * np.outer(direction, np.random.randn(W.shape[1]))
        elif len(direction) == W.shape[1]:
            # direction is input dimension
            delta = self.step_size * np.outer(np.random.randn(W.shape[0]), direction)
        else:
            # Dimension mismatch - use SVD projection
            from scipy.linalg import svd
            try:
                U, S, Vt = svd(W, full_matrices=False)
                # Project direction into weight space via SVD
                # Amplify the component of W that aligns with direction
                if len(direction) <= len(S):
                    scale = np.zeros(len(S))
                    scale[:len(direction)] = direction[:len(S)] * S[:len(direction)]
                    delta = self.step_size * (U @ np.diag(scale) @ Vt)
                else:
                    delta = np.zeros_like(W)
            except:
                delta = np.zeros_like(W)

        return delta

    def _evaluate_quality(self, test_prompts: List[Tuple[str, str]]) -> float:
        """Evaluate model quality."""
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

    def run_iteration(
        self,
        test_prompts: List[Tuple[str, str]],
        iteration: int,
    ) -> Dict:
        """Run a single iteration."""
        logger.info(f"\n--- Iteration {iteration} ---")

        # Measure current geometry
        coherent_matches = 0
        incoherent_matches = 0

        for layer_idx in self.layer_indices:
            coh_m, _ = self._compute_activation_geometry(COHERENT_PROBES, layer_idx)
            inc_m, _ = self._compute_activation_geometry(INCOHERENT_PROBES, layer_idx)
            coherent_matches += coh_m
            incoherent_matches += inc_m

        quality_before = self._evaluate_quality(test_prompts)
        contrast_before = coherent_matches - incoherent_matches

        logger.info(f"Before: coherent={coherent_matches}, incoherent={incoherent_matches}, contrast={contrast_before}, quality={quality_before:.2%}")

        # Save weights for rollback
        original_weights = {
            layer_idx: self._get_weights(layer_idx)
            for layer_idx in self.layer_indices
        }

        # Compute and apply contrastive updates
        for layer_idx in self.layer_indices:
            direction = self._compute_contrastive_direction(
                layer_idx, COHERENT_PROBES, INCOHERENT_PROBES
            )

            # Use dtype-derived epsilon; fallback to float32 precision if array empty
            div_eps = np_division_epsilon(direction) if direction.size > 0 else np.sqrt(np.finfo(np.float32).eps)
            if np.linalg.norm(direction) > div_eps:
                delta = self._project_to_weight_update(direction, layer_idx)
                W = original_weights[layer_idx]
                new_W = W + delta
                self._set_weights(layer_idx, new_W)

        # Measure new state
        coherent_matches_after = 0
        incoherent_matches_after = 0

        for layer_idx in self.layer_indices:
            coh_m, _ = self._compute_activation_geometry(COHERENT_PROBES, layer_idx)
            inc_m, _ = self._compute_activation_geometry(INCOHERENT_PROBES, layer_idx)
            coherent_matches_after += coh_m
            incoherent_matches_after += inc_m

        quality_after = self._evaluate_quality(test_prompts)
        contrast_after = coherent_matches_after - incoherent_matches_after

        logger.info(f"After: coherent={coherent_matches_after}, incoherent={incoherent_matches_after}, contrast={contrast_after}, quality={quality_after:.2%}")

        # Rollback if quality degraded or contrast worsened
        if quality_after < quality_before * self.quality_threshold or contrast_after < contrast_before:
            logger.info("Rolling back")
            for layer_idx, W in original_weights.items():
                self._set_weights(layer_idx, W)
            rolled_back = True
            coherent_matches_after = coherent_matches
            incoherent_matches_after = incoherent_matches
            quality_after = quality_before
        else:
            rolled_back = False

        return {
            'iteration': iteration,
            'coherent_before': coherent_matches,
            'incoherent_before': incoherent_matches,
            'coherent_after': coherent_matches_after,
            'incoherent_after': incoherent_matches_after,
            'contrast_before': contrast_before,
            'contrast_after': coherent_matches_after - incoherent_matches_after,
            'quality_before': float(quality_before),
            'quality_after': float(quality_after),
            'rolled_back': rolled_back,
        }

    def run(
        self,
        test_prompts: List[Tuple[str, str]],
        n_iterations: int = 20,
    ) -> ContrastiveLearningResult:
        """Run contrastive geometric learning."""

        logger.info("\n" + "="*60)
        logger.info("CONTRASTIVE GEOMETRIC LEARNING")
        logger.info(f"Layers: {self.layer_indices}")
        logger.info(f"Step size: {self.step_size}")
        logger.info("="*60)

        # Initial state
        init_coh = 0
        init_inc = 0
        for layer_idx in self.layer_indices:
            coh_m, _ = self._compute_activation_geometry(COHERENT_PROBES, layer_idx)
            inc_m, _ = self._compute_activation_geometry(INCOHERENT_PROBES, layer_idx)
            init_coh += coh_m
            init_inc += inc_m

        init_quality = self._evaluate_quality(test_prompts)

        logger.info(f"\nInitial: coherent={init_coh}, incoherent={init_inc}, contrast={init_coh - init_inc}, quality={init_quality:.2%}")

        # Run iterations
        trajectory = []
        no_progress = 0

        for i in range(n_iterations):
            result = self.run_iteration(test_prompts, i + 1)
            trajectory.append(result)

            if result['rolled_back']:
                no_progress += 1
            else:
                no_progress = 0

            if no_progress >= 5:
                logger.info("No progress for 5 iterations - stopping")
                break

        # Final state
        final_coh = 0
        final_inc = 0
        for layer_idx in self.layer_indices:
            coh_m, _ = self._compute_activation_geometry(COHERENT_PROBES, layer_idx)
            inc_m, _ = self._compute_activation_geometry(INCOHERENT_PROBES, layer_idx)
            final_coh += coh_m
            final_inc += inc_m

        final_quality = self._evaluate_quality(test_prompts)

        logger.info(f"\n{'='*60}")
        logger.info("RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Coherent: {init_coh} → {final_coh}")
        logger.info(f"Incoherent: {init_inc} → {final_inc}")
        logger.info(f"Contrast: {init_coh - init_inc} → {final_coh - final_inc}")
        logger.info(f"Quality: {init_quality:.2%} → {final_quality:.2%}")

        return ContrastiveLearningResult(
            n_iterations=len(trajectory),
            initial_coherent_matches=init_coh,
            final_coherent_matches=final_coh,
            initial_incoherent_matches=init_inc,
            final_incoherent_matches=final_inc,
            initial_contrast=init_coh - init_inc,
            final_contrast=final_coh - final_inc,
            initial_quality=float(init_quality),
            final_quality=float(final_quality),
            trajectory=trajectory,
        )


__all__ = ["ContrastiveGeometricLearning", "ContrastiveLearningResult"]
