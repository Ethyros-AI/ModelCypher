# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Iterative Thinking Loop for Self-Consistency.

The hypothesis: if a model engages in genuine self-questioning to achieve
internal consistency, fundamental constant signatures should emerge naturally.

This module implements the "thinking" process:
1. Generate initial response
2. Probe for implications and contradictions
3. Detect inconsistencies
4. Resolve by generating refined understanding
5. Measure geometry changes through iterations
6. Repeat until stable

The model isn't modifying weights. It's building coherent context through
self-questioning - similar to how humans achieve understanding through
introspection, not by modifying neurons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    pass

from .probing import SelfConsistencyProber
from .consistency_measure import ConsistencyMeasure, ConsistencyResult


@dataclass
class ThinkingIteration:
    """Record of a single thinking iteration."""

    iteration: int
    response: str
    implications: List[str]
    contradictions: List[str]
    inconsistencies: List[str]
    consistency: ConsistencyResult

    # Geometry metrics
    n_constant_matches: int
    mean_match_error: float
    spectral_entropy: float
    effective_rank: float


@dataclass
class ThinkingResult:
    """Result of the thinking loop."""

    # Input
    topic: str
    initial_response: str

    # Final state
    final_response: str
    n_iterations: int
    converged: bool  # Did consistency stabilize?

    # History
    iterations: List[ThinkingIteration]

    # Trajectory analysis
    initial_geometry: Dict[str, float] = field(default_factory=dict)
    final_geometry: Dict[str, float] = field(default_factory=dict)
    geometry_improved: bool = False
    consistency_improved: bool = False


class ThinkingLoop:
    """Let the model 'think' by iteratively questioning itself.

    The key insight: humans don't modify neurons to think. They process
    information recursively until coherence emerges. The geometry of
    understanding is a *result* of this process, not its goal.

    This class tests whether iterative self-questioning causes the
    fundamental constant signatures to emerge naturally.
    """

    def __init__(
        self,
        model,
        tokenizer,
        get_activations: Callable[[str, bool], np.ndarray],
        backend,
        convergence_threshold: float = 0.05,
    ):
        """Initialize the thinking loop.

        Args:
            model: The language model
            tokenizer: The tokenizer
            get_activations: Function to extract activations
                            (text, collapse) -> np.ndarray
            backend: Compute backend for consistency measurement
            convergence_threshold: Stop if consistency changes less than this
        """
        self.model = model
        self.tokenizer = tokenizer
        self.get_activations = get_activations
        self.backend = backend
        self.convergence_threshold = convergence_threshold

        self.prober = SelfConsistencyProber(model, tokenizer, get_activations)
        self.measure = ConsistencyMeasure(backend)

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate a response from the model."""
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

    def _compute_geometry(self, text: str) -> Tuple[int, float, float, float]:
        """Compute geometry metrics for a text.

        Returns:
            (n_constant_matches, mean_match_error, spectral_entropy, effective_rank)
        """
        from scipy.linalg import svd

        # Fundamental constants
        CONSTANTS = {
            "pi/e": 1.1557,
            "e/pi": 0.8653,
            "phi": 1.6180,
            "sqrt2": 1.4142,
            "e": 2.7183,
            "pi": 3.1416,
        }

        # Get full activation matrix (seq_len x d)
        activations = self.get_activations(text, collapse=False)

        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        centered = activations - activations.mean(axis=0)

        try:
            _, S, _ = svd(centered, full_matrices=False)
        except:
            return 0, 100.0, 0.0, 1.0

        if len(S) < 2:
            return 0, 100.0, 0.0, 1.0

        # Count constant matches
        n_matches = 0
        match_errors = []

        for i in range(min(len(S) - 1, 10)):
            for j in range(i + 1, min(len(S), i + 5)):
                if S[j] > 1e-10:
                    ratio = float(S[i] / S[j])

                    min_error = float('inf')
                    for const_val in CONSTANTS.values():
                        error = abs(ratio - const_val) / const_val * 100
                        if error < min_error:
                            min_error = error

                    if min_error < 5.0:
                        n_matches += 1
                        match_errors.append(min_error)

        mean_error = sum(match_errors) / len(match_errors) if match_errors else 100.0

        # Spectral entropy and effective rank
        S_sum = S.sum()
        if S_sum < 1e-10:
            return n_matches, mean_error, 0.0, 1.0

        S_norm = S / S_sum
        entropy = -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
        effective_rank = float(np.exp(entropy))

        return n_matches, mean_error, entropy, effective_rank

    def _find_inconsistencies(
        self,
        response: str,
        implications: List[str],
        contradictions: List[str],
    ) -> List[str]:
        """Identify inconsistencies between response and its implications.

        An inconsistency is when an implication contradicts the original
        response, or when the representation distances suggest incoherence.
        """
        inconsistencies = []

        # Get representations
        resp_act = self.get_activations(response, collapse=True)
        resp_arr = self.backend.array(resp_act)

        for impl in implications:
            if not impl:
                continue

            impl_act = self.get_activations(impl, collapse=True)
            impl_arr = self.backend.array(impl_act)

            # Compute cosine distance
            dist = self.measure.cosine_distance(resp_arr, impl_arr)

            # If implication is far from response, it's an inconsistency
            # (threshold based on empirical observation)
            if dist > 0.5:
                inconsistencies.append(
                    f"Implication '{impl[:50]}...' seems disconnected from response"
                )

        return inconsistencies

    def think(
        self,
        topic: str,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> ThinkingResult:
        """Run the thinking loop on a topic.

        Args:
            topic: The topic to think about
            max_iterations: Maximum iterations before stopping
            verbose: Whether to log progress

        Returns:
            ThinkingResult with the full trajectory
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"THINKING ABOUT: {topic}")
            print(f"{'='*60}")

        # Generate initial response
        initial_prompt = f"What do you know about {topic}?"
        response = self._generate(initial_prompt)

        if verbose:
            print(f"\nInitial response: {response[:200]}...")

        # Track iterations
        iterations = []
        prev_consistency = 0.0

        for i in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {i+1} ---")

            # Probe for implications and contradictions
            implications = self.prober.probe_implications(response, n=3)
            contradictions = self.prober.probe_contradictions(response, n=2)

            if verbose:
                print(f"  Implications: {len(implications)}")
                print(f"  Contradictions: {len(contradictions)}")

            # Measure consistency
            resp_act = self.get_activations(response, collapse=True)
            impl_acts = [self.get_activations(impl, collapse=True) for impl in implications if impl]
            contra_acts = [self.get_activations(c, collapse=True) for c in contradictions if c]

            resp_arr = self.backend.array(resp_act)
            impl_arrs = [self.backend.array(a) for a in impl_acts]
            contra_arrs = [self.backend.array(a) for a in contra_acts] if contra_acts else None

            if impl_arrs:
                consistency = self.measure.compute(resp_arr, impl_arrs, contra_arrs)
            else:
                from .consistency_measure import ConsistencyResult
                consistency = ConsistencyResult(
                    implication_consistency=0.5,
                    contradiction_distance=0.5,
                    consistency_score=0.25,
                    knowledge_confidence=0.5,
                    n_implications=0,
                    n_contradictions=0,
                    representation_distances=[],
                )

            # Measure geometry
            n_matches, mean_error, entropy, eff_rank = self._compute_geometry(response)

            if verbose:
                print(f"  Consistency: {consistency.consistency_score:.2%}")
                print(f"  Geometry: {n_matches} matches, {mean_error:.1f}% error")

            # Find inconsistencies
            inconsistencies = self._find_inconsistencies(response, implications, contradictions)

            # Record iteration
            iteration = ThinkingIteration(
                iteration=i + 1,
                response=response,
                implications=implications,
                contradictions=contradictions,
                inconsistencies=inconsistencies,
                consistency=consistency,
                n_constant_matches=n_matches,
                mean_match_error=float(mean_error),
                spectral_entropy=float(entropy),
                effective_rank=float(eff_rank),
            )
            iterations.append(iteration)

            # Check for convergence
            delta = abs(consistency.consistency_score - prev_consistency)
            if i > 0 and delta < self.convergence_threshold:
                if verbose:
                    print(f"  Converged (delta={delta:.4f})")
                break

            prev_consistency = consistency.consistency_score

            # If no inconsistencies, we've achieved coherence
            if not inconsistencies:
                if verbose:
                    print("  No inconsistencies found - coherent")
                break

            # Resolve by generating refined response
            resolution_prompt = f"""
Original statement: {response}

Implications: {', '.join(implications[:2])}

Inconsistency: {inconsistencies[0]}

Given this inconsistency, provide a more refined and coherent understanding:
"""
            response = self._generate(resolution_prompt, max_tokens=150)

            if verbose:
                print(f"  Refined: {response[:100]}...")

        # Build result
        initial_geom = iterations[0] if iterations else None
        final_geom = iterations[-1] if iterations else None

        return ThinkingResult(
            topic=topic,
            initial_response=iterations[0].response if iterations else "",
            final_response=response,
            n_iterations=len(iterations),
            converged=len(iterations) < max_iterations,
            iterations=iterations,
            initial_geometry={
                "n_matches": initial_geom.n_constant_matches if initial_geom else 0,
                "mean_error": initial_geom.mean_match_error if initial_geom else 100.0,
                "entropy": initial_geom.spectral_entropy if initial_geom else 0.0,
                "effective_rank": initial_geom.effective_rank if initial_geom else 1.0,
            },
            final_geometry={
                "n_matches": final_geom.n_constant_matches if final_geom else 0,
                "mean_error": final_geom.mean_match_error if final_geom else 100.0,
                "entropy": final_geom.spectral_entropy if final_geom else 0.0,
                "effective_rank": final_geom.effective_rank if final_geom else 1.0,
            },
            geometry_improved=(
                (final_geom.n_constant_matches > initial_geom.n_constant_matches)
                if final_geom and initial_geom else False
            ),
            consistency_improved=(
                (final_geom.consistency.consistency_score >
                 initial_geom.consistency.consistency_score)
                if final_geom and initial_geom else False
            ),
        )


__all__ = ["ThinkingLoop", "ThinkingResult", "ThinkingIteration"]
