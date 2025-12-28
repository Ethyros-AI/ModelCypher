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

    def __init__(self) -> None:
        pass

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

        In a full implementation, this would:
        1. Load model weights
        2. Run sample inputs through the model
        3. Identify layers with sparse activations
        4. Characterize the sparsity patterns

        For now, we simulate based on typical transformer patterns.
        """
        import json

        config_path = model_path / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read model config: %s", exc)
            config = {}

        # Extract layer count from config
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 12))
        hidden_size = config.get("hidden_size", config.get("n_embd", 768))

        regions: list[SparseRegion] = []

        # Simulate sparse region detection across layers
        for i in range(num_layers):
            # Attention layers typically have moderate sparsity
            attn_sparsity = 0.3 + (i / num_layers) * 0.2  # Increases with depth
            regions.append(
                SparseRegion(
                    layer_name=f"layers.{i}.self_attn",
                    start_index=0,
                    end_index=hidden_size,
                    sparsity_ratio=min(0.8, attn_sparsity),
                    activation_pattern="attention_sparse"
                    if attn_sparsity > 0.4
                    else "attention_dense",
                )
            )

            # MLP layers often have higher sparsity due to ReLU/GELU
            mlp_sparsity = 0.4 + (i / num_layers) * 0.3
            regions.append(
                SparseRegion(
                    layer_name=f"layers.{i}.mlp",
                    start_index=0,
                    end_index=hidden_size * 4,  # Typical MLP expansion
                    sparsity_ratio=min(0.9, mlp_sparsity),
                    activation_pattern="mlp_sparse" if mlp_sparsity > 0.5 else "mlp_moderate",
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

        In a full implementation, this would:
        1. Load model weights
        2. Run sample inputs through each layer
        3. Record activation statistics
        4. Identify dominant patterns

        For now, we simulate based on typical patterns.
        """
        import json

        config_path = model_path / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read model config: %s", exc)
            config = {}

        num_layers = config.get("num_hidden_layers", config.get("n_layer", 12))
        hidden_act = config.get("hidden_act", config.get("activation_function", "gelu"))

        summaries: list[ActivationMap] = []

        for i in range(num_layers):
            # Simulate activation statistics for each layer
            # Early layers tend to have more uniform activations
            # Later layers tend to have more peaked/sparse activations
            depth_factor = i / max(1, num_layers - 1)

            # Generate simulated activation values (normalized)
            base_mean = 0.3 + depth_factor * 0.2
            base_max = 0.8 + depth_factor * 0.15

            # Determine dominant pattern based on activation function and depth
            if hidden_act in ("gelu", "gelu_new"):
                if depth_factor < 0.3:
                    pattern = "gelu_smooth"
                elif depth_factor < 0.7:
                    pattern = "gelu_peaked"
                else:
                    pattern = "gelu_sparse"
            elif hidden_act == "relu":
                pattern = "relu_sparse" if depth_factor > 0.5 else "relu_active"
            elif hidden_act == "silu":
                pattern = "silu_smooth" if depth_factor < 0.5 else "silu_peaked"
            else:
                pattern = f"{hidden_act}_standard"

            # Simulated activation values (sample of 10 values)
            activation_values = [base_mean * (1 + 0.1 * (j - 5)) for j in range(10)]

            summaries.append(
                ActivationMap(
                    layer_name=f"layers.{i}",
                    activation_values=activation_values,
                    dominant_pattern=pattern,
                    mean_activation=base_mean,
                    max_activation=base_max,
                )
            )

        return summaries
