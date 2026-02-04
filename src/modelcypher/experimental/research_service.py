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

"""Research service for experimental model analysis tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.ports.activation_provider import ActivationProvider
from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseRegion:
    """A sparse activation region in the model."""

    layer_name: str
    start_index: int
    end_index: int
    sparsity_ratio: float
    activation_pattern: str


@dataclass(frozen=True)
class SparseRegionResult:
    """Result of sparse region analysis."""

    model_path: str
    regions: list[SparseRegion]
    total_sparsity: float
    layer_count: int


@dataclass(frozen=True)
class ActivationMap:
    """Activation map for a layer."""

    layer_name: str
    activation_values: list[float]
    dominant_pattern: str
    mean_activation: float
    max_activation: float


@dataclass(frozen=True)
class AFMResult:
    """Result of activation function mapping analysis."""

    model_path: str
    activation_maps: dict[str, list[float]]
    dominant_patterns: list[str]
    layer_summaries: list[ActivationMap]


class ResearchService:
    """Service for experimental research tools on model analysis.

    Provides sparse activation region analysis and activation function mapping
    for understanding model internals and behavior patterns.
    """

    def __init__(
        self,
        activation_provider: ActivationProvider,
        model_loader: ModelLoaderPort,
    ) -> None:
        self._activation_provider = activation_provider
        self._model_loader = model_loader

    def sparse_region(self, model_path: str) -> SparseRegionResult:
        """Analyze sparse activation regions in a model.

        Identifies regions of the model where activations are sparse,
        which can indicate specialized functionality or potential
        optimization opportunities.

        Args:
            model_path: Path to the model directory.

        Returns:
            SparseRegionResult with identified sparse regions.

        Raises:
            ValueError: If model path does not exist or is invalid.
        """
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not resolved_path.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

        # Check for model config
        config_path = resolved_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"No config.json found in model directory: {model_path}")

        # Analyze model structure for sparse regions
        # In a full implementation, this would load the model and analyze activations
        # For now, we simulate the analysis based on model structure
        regions = self._analyze_sparse_regions(resolved_path)

        # Calculate total sparsity
        if regions:
            total_sparsity = sum(r.sparsity_ratio for r in regions) / len(regions)
        else:
            total_sparsity = 0.0

        return SparseRegionResult(
            model_path=str(resolved_path),
            regions=regions,
            total_sparsity=total_sparsity,
            layer_count=len(regions),
        )

    def _analyze_sparse_regions(self, model_path: Path) -> list[SparseRegion]:
        """Analyze model for sparse activation regions.

        Args:
            model_path: Path to the model directory.

        """
        backend = get_default_backend()
        layer_acts = self._collect_layer_activations(model_path)
        regions: list[SparseRegion] = []

        for layer_idx, vectors in layer_acts.items():
            if not vectors:
                continue
            stacked = backend.stack([backend.array(vec) for vec in vectors], axis=0)
            mean_abs = backend.mean(backend.abs(stacked), axis=0)
            backend.eval(mean_abs)
            eps = division_epsilon(backend, mean_abs)
            mask = mean_abs <= eps
            backend.eval(mask)
            mask_list = backend.tolist(mask)
            if not isinstance(mask_list, list):
                mask_list = [bool(mask_list)]

            start = None
            for idx, is_sparse in enumerate(mask_list):
                if is_sparse and start is None:
                    start = idx
                elif not is_sparse and start is not None:
                    end = idx
                    regions.append(
                        SparseRegion(
                            layer_name=f"layer_{layer_idx}",
                            start_index=start,
                            end_index=end,
                            sparsity_ratio=(end - start) / max(len(mask_list), 1),
                            activation_pattern="epsilon_sparse",
                        )
                    )
                    start = None
            if start is not None:
                end = len(mask_list)
                regions.append(
                    SparseRegion(
                        layer_name=f"layer_{layer_idx}",
                        start_index=start,
                        end_index=end,
                        sparsity_ratio=(end - start) / max(len(mask_list), 1),
                        activation_pattern="epsilon_sparse",
                    )
                )

        return regions

    def afm(self, model_path: str) -> AFMResult:
        """Run activation function mapping analysis.

        Analyzes how activation functions behave across the model,
        identifying dominant patterns and potential anomalies.

        Args:
            model_path: Path to the model directory.

        Returns:
            AFMResult with activation maps.

        Raises:
            ValueError: If model path does not exist or is invalid.
        """
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not resolved_path.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

        # Check for model config
        config_path = resolved_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"No config.json found in model directory: {model_path}")

        # Analyze activation functions
        layer_summaries = self._analyze_activation_functions(resolved_path)

        # Build activation maps dict
        activation_maps: dict[str, list[float]] = {}
        for summary in layer_summaries:
            activation_maps[summary.layer_name] = summary.activation_values

        # Extract dominant patterns
        dominant_patterns = list(set(s.dominant_pattern for s in layer_summaries))

        return AFMResult(
            model_path=str(resolved_path),
            activation_maps=activation_maps,
            dominant_patterns=dominant_patterns,
            layer_summaries=layer_summaries,
        )

    def _analyze_activation_functions(self, model_path: Path) -> list[ActivationMap]:
        """Analyze activation function behavior across model layers.

        Args:
            model_path: Path to the model directory.

        """
        backend = get_default_backend()
        layer_acts = self._collect_layer_activations(model_path)
        summaries: list[ActivationMap] = []

        for layer_idx, vectors in layer_acts.items():
            if not vectors:
                continue
            stacked = backend.stack([backend.array(vec) for vec in vectors], axis=0)
            mean_vec = backend.mean(stacked, axis=0)
            mean_val = backend.mean(mean_vec)
            max_val = backend.max(mean_vec)
            backend.eval(mean_vec, mean_val, max_val)

            mean_activation = float(backend.to_scalar(mean_val))
            max_activation = float(backend.to_scalar(max_val))
            eps = division_epsilon(backend, mean_vec)
            if abs(mean_activation) <= eps:
                pattern = "mean_zero"
            elif mean_activation > 0.0:
                pattern = "mean_positive"
            else:
                pattern = "mean_negative"

            values = backend.tolist(mean_vec)
            if not isinstance(values, list):
                values = [float(values)]

            summaries.append(
                ActivationMap(
                    layer_name=f"layer_{layer_idx}",
                    activation_values=[float(v) for v in values],
                    dominant_pattern=pattern,
                    mean_activation=mean_activation,
                    max_activation=max_activation,
                )
            )

        return summaries

    def _collect_layer_activations(self, model_path: Path) -> dict[int, list[Any]]:
        model, tokenizer = self._model_loader.load_model_for_training(str(model_path))
        if tokenizer is None:
            raise ValueError(f"Tokenizer missing for model: {model_path}")

        probe_texts = self._probe_texts()
        if not probe_texts:
            raise ValueError("No probe texts available for research analysis")

        layer_acts: dict[int, list[Any]] = {}
        for text in probe_texts:
            activations = self._activation_provider.collect_hidden_activations(
                model=model,
                tokenizer=tokenizer,
                text=text,
            )
            for layer_idx, vec in activations.items():
                layer_acts.setdefault(layer_idx, []).append(vec)

        return layer_acts

    @staticmethod
    def _probe_texts() -> list[str]:
        probes = UnifiedAtlasInventory.all_probes()
        texts: list[str] = []
        for probe in probes:
            if probe.support_texts:
                texts.append(probe.support_texts[0])
            elif probe.name:
                texts.append(probe.name)
            elif probe.description:
                texts.append(probe.description)
        return [text for text in texts if text.strip()]
