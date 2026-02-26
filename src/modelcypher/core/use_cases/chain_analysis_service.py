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

"""Chain analysis service — unified causal chain diagnostic.

Orchestrates per-layer measurements of the validated causal chain:
    Entropy → Curvature → Cumulative curvature → Intrinsic Dimension → Phase

Combines sub-layer activation collection (injected callable) with
domain-level pure math (curvature decomposition, phase classification,
correlation analysis).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain.geometry.causal_chain import (
    ChainProfile,
    assemble_chain_profile,
    compute_layer_curvatures,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Backend

    # Type alias for the sublayer collector callable
    SublayerCollector = Callable[
        [Any, Any, list[str], int, Backend],  # model, tokenizer, prompts, num_layers, backend
        list[dict],  # returns list of sublayer activation dicts
    ]

logger = logging.getLogger(__name__)


class ChainAnalysisService:
    """Computes the full causal chain profile for a model.

    Delegates activation collection to an injected sublayer collector,
    entropy computation to BehavioralAnalyzer, and curvature/phase/correlation
    computation to the domain module.
    """

    def __init__(
        self,
        backend: Backend,
        activation_provider: ActivationProvider,
        sublayer_collector: SublayerCollector,
    ) -> None:
        self._backend = backend
        self._provider = activation_provider
        self._collect_sublayers = sublayer_collector

    def analyze_chain(
        self,
        model: Any,
        tokenizer: Any,
        probe_texts: list[str],
    ) -> ChainProfile:
        """Compute the full causal chain profile.

        Args:
            model: Loaded model (e.g., from ModelLoader).
            tokenizer: Tokenizer for the model.
            probe_texts: Texts to probe the model with.

        Returns:
            ChainProfile with per-layer measurements and cross-link correlations.
        """
        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model layers")
        num_layers = len(layers)

        # Get hidden dim from model config
        config = getattr(base_model, "args", getattr(base_model, "config", None))
        hidden_dim = 0
        if config is not None:
            hidden_dim = getattr(config, "hidden_size", 0)

        model_path = getattr(model, "_model_path", "unknown")

        # 1. Collect sublayer activations for curvature decomposition
        logger.info("Collecting sublayer activations (%d probes)...", len(probe_texts))
        sublayer_data = self._collect_sublayers(
            model, tokenizer, probe_texts, num_layers, self._backend
        )

        # Infer hidden_dim from activations if config didn't have it
        if hidden_dim == 0 and sublayer_data and sublayer_data[0]["h_in"]:
            hidden_dim = len(sublayer_data[0]["h_in"][0])

        # 2. Compute per-layer entropy via BehavioralAnalyzer
        logger.info("Computing per-layer entropy...")
        entropies = self._compute_entropies(model, tokenizer, probe_texts, num_layers)

        # 3. Compute curvature decomposition + ID (domain pure math)
        logger.info("Computing curvature decomposition and intrinsic dimension...")
        curvature_measurements = compute_layer_curvatures(sublayer_data)

        # 4. Assemble profile (phase classification + correlations in domain)
        logger.info("Classifying phases and computing correlations...")
        profile = assemble_chain_profile(
            model_path=str(model_path),
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            probe_count=len(probe_texts),
            curvature_measurements=curvature_measurements,
            entropies=entropies,
        )

        return profile

    def _compute_entropies(
        self,
        model: Any,
        tokenizer: Any,
        probe_texts: list[str],
        num_layers: int,
    ) -> list[float]:
        """Compute per-layer entropy using BehavioralAnalyzer.

        Falls back to zeros if entropy computation fails (e.g., model
        architecture not supported by LayerEntropyProjector).
        """
        try:
            from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer

            analyzer = BehavioralAnalyzer(self._provider, self._backend)
            result = analyzer.analyze_entropy_trajectory(
                model, tokenizer, tuple(probe_texts)
            )
            # Map layer_indices → entropies, filling gaps with 0.0
            entropy_map = dict(zip(result.layer_indices, result.layer_entropies))
            return [entropy_map.get(i, 0.0) for i in range(num_layers)]
        except Exception as exc:
            logger.warning("Entropy computation failed: %s. Using zeros.", exc)
            return [0.0] * num_layers
