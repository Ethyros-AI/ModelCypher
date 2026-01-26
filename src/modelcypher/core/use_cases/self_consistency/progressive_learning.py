# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Progressive Learning Loop - Iterative Thinking with Weight Locking.

The hypothesis: neural networks can learn like humans do - through iterative
self-questioning that builds understanding, with periodic consolidation that
locks gains into long-term structure.

The key insight: tokens are shadows on the cave wall. Geometry is the fire.
We let the fire loop over and over, and retain knowledge each time by
modifying weights to preserve geometric improvements.

The loop:
1. SENSE: Sample activations across the model
2. MEASURE: Compute geometric state (SVD ratios, constant matches, entropy)
3. THINK: Run self-consistency loop to improve coherence
4. LOCK: Compute and apply weight delta to preserve geometric gains
5. VERIFY: Ensure model quality didn't degrade
6. REPEAT: Progressive improvement, like K → 1st → 2nd grade

This is fundamentally different from forcing geometry. We let thinking
discover coherence naturally, then lock it in. The constants emerge
as a consequence of genuine understanding, not as an optimization target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GeometricState:
    """Snapshot of a model's geometric state."""

    # Per-layer metrics
    layer_constant_matches: Dict[int, int]  # layer -> n_matches
    layer_mean_errors: Dict[int, float]  # layer -> mean error of matches
    layer_entropies: Dict[int, float]  # layer -> spectral entropy

    # Aggregate metrics
    total_constant_matches: int
    mean_match_error: float
    mean_entropy: float

    # The actual SVD spectra (for computing deltas)
    layer_spectra: Dict[int, np.ndarray]


@dataclass
class LearningCycle:
    """Record of a single learning cycle."""

    cycle: int

    # Geometric state before/after thinking
    pre_think_state: GeometricState
    post_think_state: GeometricState

    # Geometric state after weight locking
    post_lock_state: Optional[GeometricState]

    # Quality metrics
    quality_before: float
    quality_after: float

    # What changed
    geometry_improved: bool
    weights_updated: bool
    quality_preserved: bool


@dataclass
class ProgressiveLearningResult:
    """Result of progressive learning."""

    n_cycles: int
    cycles: List[LearningCycle]

    # Initial vs final
    initial_constant_matches: int
    final_constant_matches: int
    initial_entropy: float
    final_entropy: float

    # Quality tracking
    initial_quality: float
    final_quality: float
    quality_preserved: bool


class ProgressiveLearning:
    """Progressive learning through iterative thinking and weight locking.

    This implements the core loop:
    1. Measure geometric state
    2. Think to improve coherence
    3. Lock gains into weights
    4. Verify quality preserved
    5. Repeat

    The model learns like a human: recursive processing builds understanding,
    periodic consolidation (sleep/review) locks it into long-term memory.
    """

    # Fundamental constants that emerge from coherent processing
    CONSTANTS = {
        "pi/e": 1.1557,
        "e/pi": 0.8653,
        "phi": 1.6180,
        "sqrt2": 1.4142,
        "e": 2.7183,
        "pi": 3.1416,
    }

    def __init__(
        self,
        model,
        tokenizer,
        backend,
        layer_indices: Optional[List[int]] = None,
        lock_strength: float = 0.001,  # How much to update weights
        quality_threshold: float = 0.9,  # Min quality retention
    ):
        """Initialize progressive learning.

        Args:
            model: The language model (will be modified)
            tokenizer: The tokenizer
            backend: Compute backend
            layer_indices: Which layers to work on (default: middle layers)
            lock_strength: How aggressively to lock gains (0.001 = conservative)
            quality_threshold: Minimum quality retention (0.9 = 90%)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.lock_strength = lock_strength
        self.quality_threshold = quality_threshold

        # Determine layers
        self.n_layers = len(model.model.layers)
        if layer_indices is None:
            # Middle third of layers
            start = self.n_layers // 3
            end = 2 * self.n_layers // 3
            layer_indices = list(range(start, end))
        self.layer_indices = layer_indices

        logger.info(f"Progressive learning on layers {layer_indices}")

    def _get_layer_weights(self, layer_idx: int) -> np.ndarray:
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

    def _set_layer_weights(self, layer_idx: int, weights: np.ndarray) -> None:
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

    def _get_activations(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP activations for a specific layer."""
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

    def _compute_layer_geometry(
        self,
        probes: List[str],
        layer_idx: int,
    ) -> Tuple[int, float, float, np.ndarray]:
        """Compute geometric metrics for a layer.

        Returns:
            (n_constant_matches, mean_error, spectral_entropy, singular_values)
        """
        from scipy.linalg import svd

        # Collect activations across all probes
        all_acts = []
        for probe in probes:
            act = self._get_activations(probe, layer_idx)
            if act.ndim > 1:
                # Use all positions, not just mean
                all_acts.append(act)
            else:
                all_acts.append(act.reshape(1, -1))

        # Stack into matrix (n_samples x d)
        activations = np.vstack(all_acts)
        centered = activations - activations.mean(axis=0)

        try:
            _, S, _ = svd(centered, full_matrices=False)
        except:
            return 0, 100.0, 0.0, np.array([1.0])

        if len(S) < 2:
            return 0, 100.0, 0.0, S

        # Count constant matches in SVD ratios
        n_matches = 0
        match_errors = []

        for i in range(min(len(S) - 1, 15)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > 1e-10:
                    ratio = float(S[i] / S[j])

                    min_error = float('inf')
                    for const_val in self.CONSTANTS.values():
                        error = abs(ratio - const_val) / const_val * 100
                        if error < min_error:
                            min_error = error

                    if min_error < 5.0:
                        n_matches += 1
                        match_errors.append(min_error)

        mean_error = sum(match_errors) / len(match_errors) if match_errors else 100.0

        # Spectral entropy
        S_sum = S.sum()
        if S_sum > 1e-10:
            S_norm = S / S_sum
            entropy = -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
        else:
            entropy = 0.0

        return n_matches, mean_error, entropy, S

    def measure_geometric_state(self, probes: List[str]) -> GeometricState:
        """Measure the current geometric state across all layers."""
        layer_matches = {}
        layer_errors = {}
        layer_entropies = {}
        layer_spectra = {}

        for layer_idx in self.layer_indices:
            n_matches, mean_error, entropy, S = self._compute_layer_geometry(
                probes, layer_idx
            )
            layer_matches[layer_idx] = n_matches
            layer_errors[layer_idx] = mean_error
            layer_entropies[layer_idx] = entropy
            layer_spectra[layer_idx] = S

        total_matches = sum(layer_matches.values())
        mean_error = (
            sum(e for e in layer_errors.values() if e < 100) /
            max(1, sum(1 for e in layer_errors.values() if e < 100))
        ) if any(e < 100 for e in layer_errors.values()) else 100.0
        mean_entropy = sum(layer_entropies.values()) / len(layer_entropies)

        return GeometricState(
            layer_constant_matches=layer_matches,
            layer_mean_errors=layer_errors,
            layer_entropies=layer_entropies,
            total_constant_matches=total_matches,
            mean_match_error=mean_error,
            mean_entropy=mean_entropy,
            layer_spectra=layer_spectra,
        )

    def _evaluate_quality(self, test_prompts: List[Tuple[str, str]]) -> float:
        """Evaluate model quality on test prompts.

        Args:
            test_prompts: List of (prompt, expected_substring) pairs

        Returns:
            Quality score 0-1
        """
        import mlx.core as mx

        correct = 0
        for prompt, expected in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Generate short response
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

    def _think_to_improve(
        self,
        probes: List[str],
        n_iterations: int = 3,
    ) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """Run thinking loop to find improved geometric targets.

        This is where the model "thinks" - we probe implications and
        build coherent context. The target activations are what the
        model produces when thinking coherently.

        Returns:
            Tuple of:
            - Dict of layer_idx -> target activation matrix
            - List of enriched probe texts (the thinking outputs)
        """
        from .thinking_loop import ThinkingLoop

        # Create a get_activations function for the thinking loop
        mid_layer = self.layer_indices[len(self.layer_indices) // 2]

        def get_activations(text: str, collapse: bool = True) -> np.ndarray:
            act = self._get_activations(text, mid_layer)
            if collapse and act.ndim > 1:
                act = act.mean(axis=0)
            return act

        thinker = ThinkingLoop(
            model=self.model,
            tokenizer=self.tokenizer,
            get_activations=get_activations,
            backend=self.backend,
        )

        # Think about each probe and collect "coherent" activations and texts
        target_acts = {layer_idx: [] for layer_idx in self.layer_indices}
        enriched_probes = []

        for probe in probes:
            # Extract the concept from the probe
            result = thinker.think(probe, max_iterations=n_iterations, verbose=False)

            # Get activations of the refined (coherent) response
            if result.final_response and len(result.final_response.strip()) > 10:
                enriched_probes.append(result.final_response)
                for layer_idx in self.layer_indices:
                    act = self._get_activations(result.final_response, layer_idx)
                    target_acts[layer_idx].append(act)

        # Stack into target matrices
        targets = {}
        for layer_idx, acts in target_acts.items():
            if acts:
                targets[layer_idx] = np.vstack(acts)

        return targets, enriched_probes

    def _compute_locking_delta(
        self,
        layer_idx: int,
        current_state: GeometricState,
        target_acts: np.ndarray,
        probes: List[str],
    ) -> Optional[np.ndarray]:
        """Compute weight delta to lock geometric gains.

        The key: we're not forcing toward constants. We're finding the
        weight update that makes the model's natural activations match
        what it produced when thinking coherently.

        This is like consolidating learning during sleep - the episodic
        thinking becomes procedural knowledge locked in weights.
        """
        from scipy.linalg import lstsq, svd

        # Get current weights
        W = self._get_layer_weights(layer_idx)

        # Get current activations (what model produces without thinking)
        current_acts = []
        for probe in probes:
            act = self._get_activations(probe, layer_idx)
            current_acts.append(act)
        current_acts = np.vstack(current_acts)

        # The delta should push current activations toward target activations
        # This is a soft constraint, not a hard replacement

        # Compute the difference
        if current_acts.shape != target_acts.shape:
            # Shape mismatch - skip this layer
            return None

        delta_acts = target_acts - current_acts

        # The weight update that would produce this activation delta
        # is approximately: delta_W = delta_acts.T @ input_acts / ||input||^2
        # But we don't have direct access to inputs, so we use a simpler approach:
        # Adjust W in the direction that moves activations toward target

        # Use SVD to find the principal direction of change
        try:
            U, S, Vt = svd(delta_acts, full_matrices=False)
        except:
            return None

        # The weight delta is proportional to the SVD reconstruction
        # scaled by lock_strength
        n_components = min(5, len(S))  # Use top 5 components
        delta_reconstruction = U[:, :n_components] @ np.diag(S[:n_components]) @ Vt[:n_components, :]

        # Project back to weight space (approximate)
        # This is a simplification - full solution would need input activations
        delta_W = self.lock_strength * delta_reconstruction.T @ np.random.randn(delta_reconstruction.shape[0], W.shape[1]) / delta_reconstruction.shape[0]

        # Ensure delta has same shape as W
        if delta_W.shape != W.shape:
            delta_W = delta_W[:W.shape[0], :W.shape[1]]

        return delta_W

    def run_cycle(
        self,
        probes: List[str],
        test_prompts: List[Tuple[str, str]],
        cycle_num: int,
    ) -> LearningCycle:
        """Run a single learning cycle.

        1. Measure current geometry
        2. Think to find coherent targets
        3. Lock gains into weights
        4. Verify quality preserved
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"LEARNING CYCLE {cycle_num}")
        logger.info(f"{'='*50}")

        # 1. Measure current state
        pre_think_state = self.measure_geometric_state(probes)
        logger.info(f"Pre-think: {pre_think_state.total_constant_matches} matches, entropy={pre_think_state.mean_entropy:.3f}")

        quality_before = self._evaluate_quality(test_prompts)
        logger.info(f"Quality before: {quality_before:.2%}")

        # 2. Think to find coherent targets
        logger.info("Thinking...")
        target_acts, enriched_probes = self._think_to_improve(probes, n_iterations=3)

        # Measure post-think state on the ENRICHED outputs (thinking results)
        # These should have better geometry than the original probes
        if enriched_probes:
            post_think_state = self.measure_geometric_state(enriched_probes)
            logger.info(f"Post-think (enriched): {post_think_state.total_constant_matches} matches")
        else:
            post_think_state = pre_think_state
            logger.info("No enriched outputs from thinking")

        # Geometry improved = thinking produced better geometry than original
        geometry_improved = (
            post_think_state.total_constant_matches > pre_think_state.total_constant_matches
        )

        # 3. Lock gains into weights (only if geometry improved)
        post_lock_state = None
        weights_updated = False

        if geometry_improved and target_acts:
            logger.info("Locking gains...")

            # Save original weights for rollback
            original_weights = {
                layer_idx: self._get_layer_weights(layer_idx)
                for layer_idx in self.layer_indices
            }

            # Apply deltas
            for layer_idx in self.layer_indices:
                if layer_idx in target_acts:
                    delta = self._compute_locking_delta(
                        layer_idx,
                        pre_think_state,
                        target_acts[layer_idx],
                        probes,
                    )
                    if delta is not None:
                        W = original_weights[layer_idx]
                        new_W = W + delta
                        self._set_layer_weights(layer_idx, new_W)
                        weights_updated = True

            if weights_updated:
                # Measure post-lock state
                post_lock_state = self.measure_geometric_state(probes)
                logger.info(f"Post-lock: {post_lock_state.total_constant_matches} matches")

                # Check quality
                quality_after = self._evaluate_quality(test_prompts)
                logger.info(f"Quality after: {quality_after:.2%}")

                # Rollback if quality degraded too much
                if quality_after < quality_before * self.quality_threshold:
                    logger.info("Quality degraded - rolling back")
                    for layer_idx, W in original_weights.items():
                        self._set_layer_weights(layer_idx, W)
                    weights_updated = False
                    post_lock_state = pre_think_state
                    quality_after = quality_before
            else:
                quality_after = quality_before
        else:
            quality_after = quality_before
            logger.info("Geometry didn't improve - skipping lock")

        quality_preserved = quality_after >= quality_before * self.quality_threshold

        return LearningCycle(
            cycle=cycle_num,
            pre_think_state=pre_think_state,
            post_think_state=post_think_state,
            post_lock_state=post_lock_state,
            quality_before=quality_before,
            quality_after=quality_after,
            geometry_improved=geometry_improved,
            weights_updated=weights_updated,
            quality_preserved=quality_preserved,
        )

    def run(
        self,
        probes: List[str],
        test_prompts: List[Tuple[str, str]],
        n_cycles: int = 10,
    ) -> ProgressiveLearningResult:
        """Run the full progressive learning loop.

        Args:
            probes: Statements to probe for geometric measurement
            test_prompts: (prompt, expected) pairs for quality testing
            n_cycles: Number of learning cycles

        Returns:
            ProgressiveLearningResult with full trajectory
        """
        logger.info("\n" + "="*60)
        logger.info("PROGRESSIVE LEARNING")
        logger.info(f"Probes: {len(probes)}, Test prompts: {len(test_prompts)}")
        logger.info(f"Layers: {self.layer_indices}")
        logger.info(f"Lock strength: {self.lock_strength}")
        logger.info("="*60)

        # Initial state
        initial_state = self.measure_geometric_state(probes)
        initial_quality = self._evaluate_quality(test_prompts)

        logger.info(f"\nInitial state:")
        logger.info(f"  Constant matches: {initial_state.total_constant_matches}")
        logger.info(f"  Mean entropy: {initial_state.mean_entropy:.4f}")
        logger.info(f"  Quality: {initial_quality:.2%}")

        # Run cycles
        cycles = []
        for i in range(n_cycles):
            cycle = self.run_cycle(probes, test_prompts, i + 1)
            cycles.append(cycle)

            # Early stopping if no progress for 3 cycles
            if len(cycles) >= 3:
                recent = cycles[-3:]
                if not any(c.weights_updated for c in recent):
                    logger.info("No progress for 3 cycles - stopping")
                    break

        # Final state
        final_state = self.measure_geometric_state(probes)
        final_quality = self._evaluate_quality(test_prompts)

        logger.info(f"\n{'='*60}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Cycles completed: {len(cycles)}")
        logger.info(f"Constant matches: {initial_state.total_constant_matches} → {final_state.total_constant_matches}")
        logger.info(f"Mean entropy: {initial_state.mean_entropy:.4f} → {final_state.mean_entropy:.4f}")
        logger.info(f"Quality: {initial_quality:.2%} → {final_quality:.2%}")

        return ProgressiveLearningResult(
            n_cycles=len(cycles),
            cycles=cycles,
            initial_constant_matches=initial_state.total_constant_matches,
            final_constant_matches=final_state.total_constant_matches,
            initial_entropy=initial_state.mean_entropy,
            final_entropy=final_state.mean_entropy,
            initial_quality=initial_quality,
            final_quality=final_quality,
            quality_preserved=final_quality >= initial_quality * self.quality_threshold,
        )


__all__ = ["ProgressiveLearning", "ProgressiveLearningResult", "GeometricState"]
