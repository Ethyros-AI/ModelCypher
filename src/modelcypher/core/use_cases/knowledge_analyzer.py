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

"""Knowledge Analyzer service for factual knowledge detection.

This service analyzes model representations to distinguish factual knowledge
from opinions using geometric metrics. It uses the ActivationProvider port
for model-agnostic activation collection, following hexagonal architecture.

Key Metric: Counterfactual Sensitivity (effect size 0.94)
- Facts: high sensitivity (~0.2+), representation changes when fact is violated
- Opinions: low sensitivity (~0.06), representation similar regardless of content

Design Principles:
- No heuristic defaults for layer selection - caller must specify
- Raw metrics only - no composite scores
- NaN for degenerate/undefined cases

Usage:
    from modelcypher.ports.activation_provider import get_activation_provider
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.use_cases.knowledge_analyzer import KnowledgeAnalyzer

    analyzer = KnowledgeAnalyzer(get_activation_provider(), get_default_backend())
    sig = analyzer.analyze_statement(
        model, tokenizer,
        statement="The capital of France is Paris",
        counterfactual="The capital of France is Madrid",
        layer_idx=12,  # REQUIRED - caller chooses layer
    )
    print(f"Counterfactual sensitivity: {sig.counterfactual_sensitivity}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
from modelcypher.core.domain.geometry.knowledge_metrics import (
    compute_kurtosis,
    counterfactual_sensitivity,
    layer_consistency,
    repetition_consistency,
)
from modelcypher.core.domain.geometry.knowledge_signature import (
    InvariantSignature,
    KnowledgeSignature,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Backend


class KnowledgeAnalyzer:
    """Analyze factual knowledge in model representations.

    Uses ActivationProvider port for model-agnostic activation collection.
    All tensor operations use the Backend port for GPU-native computation.

    Example:
        >>> analyzer = KnowledgeAnalyzer(provider, backend)
        >>> sig = analyzer.analyze_statement(
        ...     model, tokenizer,
        ...     statement="2 + 2 = 4",
        ...     counterfactual="2 + 2 = 5",
        ...     layer_idx=12,
        ... )
        >>> print(f"Sensitivity: {sig.counterfactual_sensitivity:.3f}")
        Sensitivity: 0.077  # High sensitivity = large activation difference
    """

    def __init__(
        self,
        activation_provider: "ActivationProvider",
        backend: "Backend",
    ) -> None:
        """Initialize the analyzer.

        Args:
            activation_provider: Provider for collecting model activations.
            backend: Compute backend for tensor operations.
        """
        self._provider = activation_provider
        self._backend = backend
        self._effective_rank = EffectiveRank(backend)

    def analyze_statement(
        self,
        model: Any,
        tokenizer: Any,
        statement: str,
        counterfactual: str,
        layer_idx: int,
    ) -> KnowledgeSignature:
        """Analyze a statement to determine if it represents factual knowledge.

        The primary metric is counterfactual sensitivity - how much the
        representation changes when the fact is violated.

        Args:
            model: The loaded model (e.g., mlx_lm model).
            tokenizer: The tokenizer for encoding text.
            statement: The original statement (e.g., "Paris is the capital of France").
            counterfactual: The fact-violating statement (e.g., "Madrid is the capital of France").
            layer_idx: REQUIRED - specific layer to analyze. Caller must choose.

        Returns:
            KnowledgeSignature containing raw geometric metrics for the statement.
            Metrics may be NaN for degenerate inputs.
        """
        # Get hidden activations for both statements
        hidden_orig = self._provider.collect_hidden_activations(
            model, tokenizer, statement
        )
        hidden_cf = self._provider.collect_hidden_activations(
            model, tokenizer, counterfactual
        )

        if layer_idx not in hidden_orig:
            raise ValueError(f"Layer {layer_idx} not in collected activations. Available: {sorted(hidden_orig.keys())}")
        if layer_idx not in hidden_cf:
            raise ValueError(f"Layer {layer_idx} not in counterfactual activations")

        # Get activations for the target layer
        act_orig = hidden_orig[layer_idx]
        act_cf = hidden_cf[layer_idx]

        # Compute counterfactual sensitivity (primary metric)
        cf_sens = counterfactual_sensitivity(act_orig, act_cf, self._backend)

        # Compute effective rank metrics for the original statement
        rank_result = self._effective_rank.compute(act_orig)

        return KnowledgeSignature(
            statement=statement,
            counterfactual=counterfactual,
            counterfactual_sensitivity=cf_sens,
            effective_rank=rank_result.shannon_effective_rank,
            spectral_entropy=rank_result.spectral_entropy,
        )

    def analyze_invariants(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        layer_indices: list[int],
        n_repetitions: int,
    ) -> InvariantSignature:
        """Analyze geometric invariants of a prompt's representation.

        Computes multiple geometric metrics to characterize representation stability.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            prompt: The prompt to analyze.
            layer_indices: REQUIRED - specific layers to analyze. Caller must choose.
            n_repetitions: REQUIRED - number of inference runs for consistency.

        Returns:
            InvariantSignature containing raw multi-metric geometric signature.
            Metrics may be NaN for degenerate/insufficient inputs.
        """
        if not layer_indices:
            raise ValueError("layer_indices must be non-empty")
        if n_repetitions < 1:
            raise ValueError("n_repetitions must be at least 1")

        # Collect activations for consistency analysis
        reps = []
        for _ in range(n_repetitions):
            hidden = self._provider.collect_hidden_activations(model, tokenizer, prompt)
            reps.append(hidden)

        if not reps or not reps[0]:
            raise ValueError("No activations collected")

        # Validate layer indices
        available_layers = set(reps[0].keys())
        missing = set(layer_indices) - available_layers
        if missing:
            raise ValueError(f"Layers {missing} not available. Available: {sorted(available_layers)}")

        # Use first specified layer for primary analysis
        primary_layer = layer_indices[0]
        primary_act = reps[0][primary_layer]

        # Compute kurtosis (may be NaN)
        kurt = compute_kurtosis(primary_act, self._backend)

        # Compute effective rank / spectral entropy
        rank_result = self._effective_rank.compute(primary_act)

        # Compute layer consistency across specified layers (may be NaN)
        layer_acts = {idx: reps[0][idx] for idx in layer_indices}
        layer_cons = layer_consistency(layer_acts, self._backend)

        # Compute repetition consistency at primary layer (may be NaN)
        rep_acts = [r[primary_layer] for r in reps if primary_layer in r]
        rep_cons = repetition_consistency(rep_acts, self._backend)

        return InvariantSignature(
            prompt=prompt,
            kurtosis=kurt,
            spectral_entropy=rank_result.spectral_entropy,
            effective_rank=rank_result.shannon_effective_rank,
            layer_consistency=layer_cons,
            repetition_consistency=rep_cons,
        )

    def batch_analyze_statements(
        self,
        model: Any,
        tokenizer: Any,
        statement_pairs: list[tuple[str, str]],
        layer_idx: int,
    ) -> list[KnowledgeSignature]:
        """Analyze multiple statement/counterfactual pairs.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for encoding text.
            statement_pairs: List of (statement, counterfactual) tuples.
            layer_idx: REQUIRED - specific layer to analyze.

        Returns:
            List of KnowledgeSignature for each statement pair.
        """
        return [
            self.analyze_statement(model, tokenizer, stmt, cf, layer_idx)
            for stmt, cf in statement_pairs
        ]
