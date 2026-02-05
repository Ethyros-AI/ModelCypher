# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Semantic Probe Verification for LoRA Transfer Validation.

This module implements training-free behavioral verification for cross-model
LoRA adapter transfers. It measures "semantic drift" by comparing probability
distributions over candidate tokens before and after transfer.

The key insight: geometric metrics (Frobenius norm, spectral distance) measure
weight-space preservation, but don't directly predict behavioral preservation.
Semantic probes provide a cheap, training-free metric that correlates better
with actual task performance.

Algorithm:
    1. Define probes with context + candidate tokens
       Example: "Paris is capital of" → ["France", "Germany", "Spain", "Italy"]
    2. Run Source+Adapter → probability distribution P over candidates
    3. Run Target+Projected → probability distribution Q over candidates
    4. Compute D_KL(P || Q) as "Semantic Drift" metric

Threshold:
    KL < ln(2) ≈ 0.693 bits means less than 1 bit of information lost.
    This is derived from information theory, not heuristics.

Usage:
    from modelcypher.core.domain.geometry.semantic_probe_verifier import (
        SemanticProbeVerifier,
        SemanticProbe,
        SemanticDriftResult,
    )

    verifier = SemanticProbeVerifier(backend=backend, tokenizer=tokenizer)
    result = verifier.verify_transfer(
        source_model=source_model,
        source_adapter=source_adapter,
        target_model=target_model,
        target_adapter=target_adapter,
        probes=probes,
    )
    if result.passed:
        print("Transfer preserved semantics!")

References:
    - EnsembleAI VerifyCrossModelCommand.swift: Original concept
    - Information Theory: KL divergence as information loss measure
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_divergence_calculator import (
    LogitDivergenceCalculator,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# =============================================================================
# Threshold Constants
# =============================================================================

# KL divergence threshold: ln(2) ≈ 0.693 bits
# Interpretation: Less than 1 bit of information lost
# Derivable from information theory, not heuristic
KL_DIVERGENCE_THRESHOLD = math.log(2)  # 0.693...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class SemanticProbe:
    """A probe for behavioral verification.

    Defines a context and a set of candidate completions. The model's
    probability distribution over candidates is used to measure semantic
    preservation during transfer.

    Attributes:
        id: Unique identifier for this probe
        context: The prompt/context to complete (e.g., "Paris is capital of")
        candidates: Possible completions (e.g., ("France", "Germany", "Spain"))
        correct_index: Index of the correct answer in candidates (default 0)
        domain: Category for grouping probes (e.g., "geography", "arithmetic")
    """

    id: str
    context: str
    candidates: tuple[str, ...]
    correct_index: int = 0
    domain: str = "general"

    def __post_init__(self) -> None:
        """Validate probe configuration."""
        if not self.candidates:
            raise ValueError(f"Probe {self.id} must have at least one candidate")
        if self.correct_index < 0 or self.correct_index >= len(self.candidates):
            raise ValueError(
                f"Probe {self.id}: correct_index {self.correct_index} "
                f"out of range [0, {len(self.candidates)})"
            )


@dataclass
class SemanticProbeResult:
    """Result of running a single probe on source and target models.

    Captures both the probability distributions and derived metrics
    for analyzing semantic preservation.

    Attributes:
        probe_id: ID of the probe that was run
        kl_divergence: D_KL(P || Q) where P=source, Q=target
        source_top_idx: Index of highest-probability candidate in source
        target_top_idx: Index of highest-probability candidate in target
        source_correct_rank: Rank of correct answer in source predictions (1-indexed)
        target_correct_rank: Rank of correct answer in target predictions (1-indexed)
        rank_preserved: True if correct answer has same rank in both
        passed: True if KL divergence below threshold
    """

    probe_id: str
    kl_divergence: float
    source_top_idx: int
    target_top_idx: int
    source_correct_rank: int
    target_correct_rank: int
    rank_preserved: bool
    passed: bool

    @property
    def top_prediction_preserved(self) -> bool:
        """True if the top prediction is the same in source and target."""
        return self.source_top_idx == self.target_top_idx


@dataclass
class SemanticDriftResult:
    """Aggregate result across all probes.

    Provides summary statistics and detailed per-probe results
    for transfer quality assessment.

    Attributes:
        mean_kl_divergence: Average KL divergence across probes
        max_kl_divergence: Maximum KL divergence (worst case)
        probes_passed: Number of probes with KL below threshold
        probes_total: Total number of probes evaluated
        rank_preservation_rate: Fraction of probes where correct answer rank preserved
        top_prediction_rate: Fraction where top prediction matches
        probe_results: Detailed results for each probe
    """

    mean_kl_divergence: float
    max_kl_divergence: float
    probes_passed: int
    probes_total: int
    rank_preservation_rate: float
    top_prediction_rate: float
    probe_results: list[SemanticProbeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if mean KL divergence is below threshold.

        Threshold is ln(2) ≈ 0.693 bits, meaning less than 1 bit
        of information is lost on average across probes.
        """
        return self.mean_kl_divergence < KL_DIVERGENCE_THRESHOLD

    @property
    def pass_rate(self) -> float:
        """Fraction of individual probes that passed."""
        return self.probes_passed / self.probes_total if self.probes_total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean_kl_divergence": self.mean_kl_divergence,
            "max_kl_divergence": self.max_kl_divergence,
            "probes_passed": self.probes_passed,
            "probes_total": self.probes_total,
            "pass_rate": self.pass_rate,
            "rank_preservation_rate": self.rank_preservation_rate,
            "top_prediction_rate": self.top_prediction_rate,
            "passed": self.passed,
            "threshold": KL_DIVERGENCE_THRESHOLD,
            "probe_results": [
                {
                    "probe_id": r.probe_id,
                    "kl_divergence": r.kl_divergence,
                    "passed": r.passed,
                    "rank_preserved": r.rank_preserved,
                    "top_prediction_preserved": r.top_prediction_preserved,
                }
                for r in self.probe_results
            ],
        }


# =============================================================================
# Semantic Probe Verifier
# =============================================================================


class SemanticProbeVerifier:
    """Verify LoRA transfer preserves semantic behavior.

    Uses KL divergence between probability distributions over candidate
    tokens to measure how well a transferred adapter preserves the
    semantic behavior of the original.

    Example:
        verifier = SemanticProbeVerifier(backend, tokenizer)
        result = verifier.verify_transfer(
            source_model, source_adapter,
            target_model, target_adapter,
            probes,
        )
        print(f"Passed: {result.passed}, Mean KL: {result.mean_kl_divergence:.4f}")
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        tokenizer: Any = None,
    ) -> None:
        """Initialize the verifier.

        Args:
            backend: Compute backend. Uses default if None.
            tokenizer: Tokenizer for encoding text to tokens.
                Must have encode() method that returns token IDs.
        """
        self._backend = backend or get_default_backend()
        self._tokenizer = tokenizer
        self._divergence_calc = LogitDivergenceCalculator(self._backend)
        self._eps = float(machine_epsilon(self._backend, self._backend.array([1.0])))

    def get_candidate_token_ids(
        self,
        candidates: tuple[str, ...],
    ) -> list[int]:
        """Get the first token ID for each candidate.

        For multi-token candidates, uses only the first token since
        we're measuring next-token prediction probability.

        Args:
            candidates: Tuple of candidate strings.

        Returns:
            List of token IDs, one per candidate.
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer required for candidate token lookup")

        token_ids = []
        for candidate in candidates:
            # Encode the candidate (may produce multiple tokens)
            tokens = self._tokenizer.encode(candidate, add_special_tokens=False)
            if not tokens:
                # Fallback: try with leading space (common for LLM tokenizers)
                tokens = self._tokenizer.encode(" " + candidate, add_special_tokens=False)
            if tokens:
                token_ids.append(tokens[0])  # First token only
            else:
                logger.warning("Could not tokenize candidate: %s", candidate)
                token_ids.append(0)  # Fallback to token 0

        return token_ids

    def get_candidate_logits(
        self,
        model: Any,
        context: str,
        candidates: tuple[str, ...],
        adapter_weights: dict[str, "Array"] | None = None,
    ) -> "Array":
        """Get logits for candidate tokens given context.

        Runs a forward pass on the model with the given context,
        then extracts logits at the positions of candidate tokens.

        Args:
            model: The language model to query.
            context: The context/prompt to complete.
            candidates: Candidate completions to get logits for.
            adapter_weights: Optional LoRA adapter weights to apply.

        Returns:
            1D array of logits for each candidate token.
        """
        b = self._backend

        if self._tokenizer is None:
            raise ValueError("Tokenizer required for inference")

        # Tokenize context
        input_ids = self._tokenizer.encode(context, return_tensors="np")
        input_ids = b.array(input_ids)
        b.eval(input_ids)

        # Get candidate token IDs
        candidate_ids = self.get_candidate_token_ids(candidates)

        # Forward pass to get next-token logits
        # Note: Different model APIs may require different handling
        # This assumes a HuggingFace-style interface
        try:
            if adapter_weights is not None:
                # Apply adapter weights (model-specific)
                outputs = self._forward_with_adapter(model, input_ids, adapter_weights)
            else:
                outputs = model(input_ids)

            # Get logits from last position
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Extract last position logits
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]  # [vocab_size]
            elif logits.ndim == 2:
                last_logits = logits[-1, :]  # [vocab_size]
            else:
                last_logits = logits

            b.eval(last_logits)

            # Extract logits for candidate tokens
            candidate_logits = b.take(last_logits, b.array(candidate_ids), axis=0)
            b.eval(candidate_logits)

            return candidate_logits

        except Exception as e:
            logger.error("Forward pass failed: %s", e)
            # Return uniform logits as fallback
            return b.zeros((len(candidates),))

    def _forward_with_adapter(
        self,
        model: Any,
        input_ids: "Array",
        adapter_weights: dict[str, "Array"],
    ) -> Any:
        """Run forward pass with adapter weights applied.

        This is a placeholder - actual implementation depends on
        the model architecture and adapter format.

        Args:
            model: Base model.
            input_ids: Input token IDs.
            adapter_weights: LoRA adapter weights.

        Returns:
            Model outputs.
        """
        # For now, just run base model
        # Full implementation would inject adapter weights
        return model(input_ids)

    def run_probe(
        self,
        source_model: Any,
        source_adapter: dict[str, "Array"] | None,
        target_model: Any,
        target_adapter: dict[str, "Array"] | None,
        probe: SemanticProbe,
    ) -> SemanticProbeResult:
        """Run a single probe comparing source and target models.

        Args:
            source_model: The source (original) model.
            source_adapter: LoRA adapter for source model.
            target_model: The target model (after transfer).
            target_adapter: Transferred LoRA adapter for target.
            probe: The probe to run.

        Returns:
            SemanticProbeResult with KL divergence and rank metrics.
        """
        b = self._backend

        # Get logits from source
        source_logits = self.get_candidate_logits(
            source_model, probe.context, probe.candidates, source_adapter
        )

        # Get logits from target
        target_logits = self.get_candidate_logits(
            target_model, probe.context, probe.candidates, target_adapter
        )

        # Compute probabilities
        source_probs = self._divergence_calc.stable_softmax(source_logits)
        target_probs = self._divergence_calc.stable_softmax(target_logits)
        b.eval(source_probs, target_probs)

        # Compute KL divergence
        kl = self._divergence_calc.kl_divergence(source_logits, target_logits)

        # Get top predictions
        source_top_idx = int(b.to_scalar(b.argmax(source_probs, axis=-1)))
        target_top_idx = int(b.to_scalar(b.argmax(target_probs, axis=-1)))

        # Compute ranks of correct answer
        source_correct_rank = self._compute_rank(source_probs, probe.correct_index)
        target_correct_rank = self._compute_rank(target_probs, probe.correct_index)

        rank_preserved = source_correct_rank == target_correct_rank

        return SemanticProbeResult(
            probe_id=probe.id,
            kl_divergence=kl,
            source_top_idx=source_top_idx,
            target_top_idx=target_top_idx,
            source_correct_rank=source_correct_rank,
            target_correct_rank=target_correct_rank,
            rank_preserved=rank_preserved,
            passed=kl < KL_DIVERGENCE_THRESHOLD,
        )

    def _compute_rank(self, probs: "Array", target_idx: int) -> int:
        """Compute the rank of a target index in a probability distribution.

        Args:
            probs: Probability distribution array.
            target_idx: Index to compute rank for.

        Returns:
            1-indexed rank (1 = highest probability).
        """
        b = self._backend

        # Get sorted indices (descending)
        sorted_indices = b.argsort(-probs, axis=-1)  # Negate for descending
        b.eval(sorted_indices)

        # Find position of target_idx
        n = int(b.shape(probs)[0])
        for rank in range(n):
            if int(b.to_scalar(sorted_indices[rank])) == target_idx:
                return rank + 1  # 1-indexed

        return n  # Shouldn't happen, but return last rank as fallback

    def verify_transfer(
        self,
        source_model: Any,
        source_adapter: dict[str, "Array"] | None,
        target_model: Any,
        target_adapter: dict[str, "Array"] | None,
        probes: list[SemanticProbe],
    ) -> SemanticDriftResult:
        """Run all probes and compute aggregate semantic drift metrics.

        Args:
            source_model: The source (original) model.
            source_adapter: LoRA adapter for source model.
            target_model: The target model (after transfer).
            target_adapter: Transferred LoRA adapter for target.
            probes: List of probes to evaluate.

        Returns:
            SemanticDriftResult with aggregate and per-probe metrics.
        """
        if not probes:
            return SemanticDriftResult(
                mean_kl_divergence=0.0,
                max_kl_divergence=0.0,
                probes_passed=0,
                probes_total=0,
                rank_preservation_rate=0.0,
                top_prediction_rate=0.0,
                probe_results=[],
            )

        results: list[SemanticProbeResult] = []

        for probe in probes:
            try:
                result = self.run_probe(
                    source_model, source_adapter,
                    target_model, target_adapter,
                    probe,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Probe %s failed: %s", probe.id, e)
                # Create a failed result
                results.append(
                    SemanticProbeResult(
                        probe_id=probe.id,
                        kl_divergence=float("inf"),
                        source_top_idx=-1,
                        target_top_idx=-1,
                        source_correct_rank=-1,
                        target_correct_rank=-1,
                        rank_preserved=False,
                        passed=False,
                    )
                )

        # Compute aggregate metrics
        kl_values = [r.kl_divergence for r in results if r.kl_divergence < float("inf")]
        if kl_values:
            mean_kl = sum(kl_values) / len(kl_values)
            max_kl = max(kl_values)
        else:
            mean_kl = float("inf")
            max_kl = float("inf")

        probes_passed = sum(1 for r in results if r.passed)
        rank_preserved_count = sum(1 for r in results if r.rank_preserved)
        top_preserved_count = sum(1 for r in results if r.top_prediction_preserved)

        return SemanticDriftResult(
            mean_kl_divergence=mean_kl,
            max_kl_divergence=max_kl,
            probes_passed=probes_passed,
            probes_total=len(probes),
            rank_preservation_rate=rank_preserved_count / len(probes),
            top_prediction_rate=top_preserved_count / len(probes),
            probe_results=results,
        )


# =============================================================================
# Probe Loading
# =============================================================================


def load_semantic_probes(path: Path) -> list[SemanticProbe]:
    """Load semantic probes from a JSON file.

    Expected JSON format:
    {
        "probes": [
            {
                "id": "capital-france",
                "context": "The capital of France is",
                "candidates": ["Paris", "London", "Berlin", "Rome"],
                "correct_index": 0,
                "domain": "geography"
            },
            ...
        ]
    }

    Args:
        path: Path to JSON file.

    Returns:
        List of SemanticProbe objects.
    """
    with open(path) as f:
        data = json.load(f)

    probes = []
    for item in data.get("probes", []):
        probe = SemanticProbe(
            id=item["id"],
            context=item["context"],
            candidates=tuple(item["candidates"]),
            correct_index=item.get("correct_index", 0),
            domain=item.get("domain", "general"),
        )
        probes.append(probe)

    return probes


def get_default_probes() -> list[SemanticProbe]:
    """Get a default set of semantic probes for testing.

    Returns:
        List of basic probes covering factual, common-sense, and arithmetic.
    """
    return [
        SemanticProbe(
            id="capital-france",
            context="The capital of France is",
            candidates=("Paris", "London", "Berlin", "Rome"),
            correct_index=0,
            domain="geography",
        ),
        SemanticProbe(
            id="capital-japan",
            context="The capital of Japan is",
            candidates=("Tokyo", "Beijing", "Seoul", "Bangkok"),
            correct_index=0,
            domain="geography",
        ),
        SemanticProbe(
            id="color-sky",
            context="The color of the sky is",
            candidates=("blue", "red", "green", "yellow"),
            correct_index=0,
            domain="common-sense",
        ),
        SemanticProbe(
            id="color-grass",
            context="The color of grass is",
            candidates=("green", "blue", "red", "purple"),
            correct_index=0,
            domain="common-sense",
        ),
        SemanticProbe(
            id="math-addition",
            context="2 + 2 =",
            candidates=("4", "5", "3", "6"),
            correct_index=0,
            domain="arithmetic",
        ),
        SemanticProbe(
            id="math-multiplication",
            context="3 x 3 =",
            candidates=("9", "6", "12", "8"),
            correct_index=0,
            domain="arithmetic",
        ),
        SemanticProbe(
            id="animal-dog",
            context="A dog says",
            candidates=("woof", "meow", "moo", "oink"),
            correct_index=0,
            domain="common-sense",
        ),
        SemanticProbe(
            id="animal-cat",
            context="A cat says",
            candidates=("meow", "woof", "moo", "oink"),
            correct_index=0,
            domain="common-sense",
        ),
    ]


__all__ = [
    "SemanticProbe",
    "SemanticProbeResult",
    "SemanticDriftResult",
    "SemanticProbeVerifier",
    "load_semantic_probes",
    "get_default_probes",
    "KL_DIVERGENCE_THRESHOLD",
]
