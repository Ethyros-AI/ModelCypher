#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ModelCypher
"""Capability scanner for autonomous self-improvement.

The CapabilityScanner analyzes model capabilities using:
1. Geometric metrics (condition number κ of Gram matrix)
2. Behavioral metrics (accuracy on test problems)
3. Priming response (how accuracy changes with semantic primes)

This allows automatic classification of capabilities as:
- WORKING: Functions correctly without intervention
- DISCONNECTED: Exists but needs priming to activate
- TRUE_GAP: Missing entirely, requires training
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

from .types import (
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
    DEFAULT_ACCURACY_THRESHOLD,
    DEFAULT_PRIMES,
)

if TYPE_CHECKING:
    # Avoid import cycles - these are only for type hints
    pass


class CapabilityScanner:
    """Scan model capabilities via geometry (κ) and behavioral metrics.

    The scanner identifies three capability states:
    - WORKING: accuracy_raw >= threshold
    - DISCONNECTED: accuracy_primed >= threshold, accuracy_raw < threshold
    - TRUE_GAP: both accuracies < threshold

    Example:
        >>> scanner = CapabilityScanner(model, tokenizer)
        >>> capability = Capability.from_lists(
        ...     name="arithmetic",
        ...     prompts=["1+1=", "2+2=", "3+3="],
        ...     problems=[("1+1=", "2"), ("2+2=", "4")]
        ... )
        >>> analysis = scanner.scan(capability)
        >>> print(f"{analysis.status.value}: {analysis.accuracy_primed:.0%}")
    """

    PRIMES_TO_TRY: ClassVar[Tuple[str, ...]] = DEFAULT_PRIMES
    ACCURACY_THRESHOLD: ClassVar[float] = DEFAULT_ACCURACY_THRESHOLD

    def __init__(
        self,
        model,
        tokenizer,
        primes: Optional[Tuple[str, ...]] = None,
        accuracy_threshold: Optional[float] = None,
    ):
        """Initialize scanner.

        Args:
            model: The language model to scan
            tokenizer: The tokenizer for the model
            primes: Custom primes to try (defaults to DEFAULT_PRIMES)
            accuracy_threshold: Threshold for classification (default 0.7)
        """
        self.model = model
        self.tokenizer = tokenizer
        self._primes = primes if primes is not None else self.PRIMES_TO_TRY
        self._threshold = (
            accuracy_threshold
            if accuracy_threshold is not None
            else self.ACCURACY_THRESHOLD
        )

    def get_activations(self, prompts: List[str]) -> Any:
        """Get final-layer hidden state activations for prompts.

        Args:
            prompts: List of text prompts

        Returns:
            MLX array of shape (n_prompts, hidden_dim)
        """
        import mlx.core as mx

        activations = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            # Get hidden states from final layer
            hidden = self.model.model.embed_tokens(input_ids)
            for layer in self.model.model.layers:
                hidden = layer(hidden, mask=None, cache=None)

            mx.eval(hidden)
            # Take last token's hidden state
            activations.append(hidden[0, -1, :])

        return mx.stack(activations)

    def compute_kappa(self, activations: np.ndarray) -> float:
        """Compute condition number of Gram matrix.

        The condition number κ measures how well-conditioned the
        activation space is. High κ indicates poorly aligned
        representations (potential disconnection).

        Args:
            activations: Array of shape (n_samples, hidden_dim)

        Returns:
            Condition number κ (higher = worse alignment)
        """
        gram = activations @ activations.T
        try:
            return float(np.linalg.cond(gram))
        except Exception:
            return float("inf")

    def evaluate_accuracy(
        self,
        prime: str,
        problems: List[Tuple[str, str]],
    ) -> float:
        """Evaluate accuracy on test problems with optional prime.

        Args:
            prime: Prime to prepend (empty string for no priming)
            problems: List of (prompt, expected_answer) pairs

        Returns:
            Accuracy as float in [0, 1]
        """
        import mlx.core as mx

        if not problems:
            return 0.0

        correct = 0
        for problem, expected in problems:
            prompt = f"{prime} {problem}" if prime else problem

            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = self.model(input_ids)
            mx.eval(logits)

            # Get probabilities and top prediction
            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()

            top_token = int(np.argmax(probs))
            predicted = self.tokenizer.decode([top_token]).strip()

            # Check if prediction matches expected
            if expected in predicted or predicted == expected:
                correct += 1

        return correct / len(problems)

    def scan(self, capability: Capability) -> CapabilityAnalysis:
        """Scan a capability and classify its status.

        This is the main entry point for capability analysis.

        Args:
            capability: The capability to analyze

        Returns:
            CapabilityAnalysis with full metrics and classification
        """
        # Get raw metrics (no priming)
        prompts_list = list(capability.prompts)
        problems_list = [(p, e) for p, e in capability.problems]

        acts_raw = self.get_activations(prompts_list)
        kappa_raw = self.compute_kappa(acts_raw)
        accuracy_raw = self.evaluate_accuracy("", problems_list)

        # Try primes and find best
        best_accuracy = accuracy_raw
        best_kappa = kappa_raw
        best_prime = ""

        for prime in self._primes:
            # Get primed metrics
            primed_prompts = [f"{prime} {p}" for p in prompts_list]
            acts_primed = self.get_activations(primed_prompts)
            kappa_primed = self.compute_kappa(acts_primed)
            accuracy_primed = self.evaluate_accuracy(prime, problems_list)

            if accuracy_primed > best_accuracy:
                best_accuracy = accuracy_primed
                best_kappa = kappa_primed
                best_prime = prime

        # Classify
        if accuracy_raw >= self._threshold:
            status = CapabilityStatus.WORKING
        elif best_accuracy >= self._threshold:
            status = CapabilityStatus.DISCONNECTED
        else:
            status = CapabilityStatus.TRUE_GAP

        return CapabilityAnalysis(
            capability=capability,
            status=status,
            accuracy_raw=accuracy_raw,
            accuracy_primed=best_accuracy,
            kappa_raw=kappa_raw,
            kappa_primed=best_kappa,
            best_prime=best_prime,
        )

    def classify(
        self,
        capabilities: List[Capability],
    ) -> Dict[CapabilityStatus, List[CapabilityAnalysis]]:
        """Classify multiple capabilities by status.

        Args:
            capabilities: List of capabilities to analyze

        Returns:
            Dictionary mapping status to list of analyses
        """
        results: Dict[CapabilityStatus, List[CapabilityAnalysis]] = {
            CapabilityStatus.WORKING: [],
            CapabilityStatus.DISCONNECTED: [],
            CapabilityStatus.TRUE_GAP: [],
        }

        for cap in capabilities:
            analysis = self.scan(cap)
            results[analysis.status].append(analysis)

        return results


__all__ = ["CapabilityScanner"]
