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

"""Four-condition experiment setup for LoRA geometry validation.

Conditions:
1. Untrained: LoRA initialized, 0 training steps
2. Trained: Normal LoRA training, converged
3. Random Labels: Trained on shuffled labels
4. Pure Random: Random B, A matrices (no training)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.lora_geometry.measurements import (
    AdapterMeasurement,
    LayerMeasurement,
    collect_layer_measurements,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


class ConditionType(Enum):
    """Four experimental conditions for LoRA geometry study."""

    UNTRAINED = "untrained"  # Initialized, 0 training steps
    TRAINED = "trained"  # Normal training, converged
    RANDOM_LABELS = "random_labels"  # Trained on shuffled labels
    PURE_RANDOM = "pure_random"  # Random B, A matrices


@dataclass(frozen=True)
class SyntheticAdapter:
    """A synthetic LoRA adapter for controlled experiments.

    Attributes:
        condition: The experimental condition.
        delta_weights: Dict mapping layer keys to ΔW = B @ A.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA alpha (typically r).
        adapter_id: Unique identifier.
    """

    condition: ConditionType
    delta_weights: dict[str, "Array"]
    lora_rank: int
    lora_alpha: float
    adapter_id: str


def create_synthetic_adapter(
    base_weights: dict[str, "Array"],
    condition: ConditionType,
    lora_rank: int = 8,
    lora_alpha: float = 8.0,
    scale: float = 0.01,
    adapter_id: str | None = None,
    backend: "Backend | None" = None,
) -> SyntheticAdapter:
    """Create a synthetic LoRA adapter for a given condition.

    Args:
        base_weights: Dict mapping layer keys to base weight arrays.
        condition: The experimental condition to create.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA alpha scaling.
        scale: Scale factor for random initialization.
        adapter_id: Optional unique identifier.
        backend: Compute backend.

    Returns:
        SyntheticAdapter with delta weights for each layer.

    Note:
        - UNTRAINED: Standard Kaiming init for B, zeros for A (gives ΔW = 0)
        - TRAINED: Not created here - load from actual trained adapter
        - RANDOM_LABELS: Not created here - requires training infrastructure
        - PURE_RANDOM: Random Gaussian B and A matrices
    """
    if backend is None:
        backend = get_default_backend()

    if adapter_id is None:
        adapter_id = f"{condition.value}_{id(base_weights)}"

    delta_weights: dict[str, "Array"] = {}

    for key, base_w in base_weights.items():
        backend.eval(base_w)
        shape = backend.shape(base_w)
        out_features, in_features = int(shape[0]), int(shape[1])

        if condition == ConditionType.UNTRAINED:
            # Standard LoRA init: B has Kaiming init, A has zeros
            # This gives ΔW = B @ A = 0 at initialization
            B = backend.random_normal((out_features, lora_rank), dtype="float32")
            # Kaiming scaling: sqrt(2/fan_in)
            kaiming_scale = (2.0 / lora_rank) ** 0.5
            B = backend.multiply(B, kaiming_scale)
            A = backend.zeros((lora_rank, in_features), dtype="float32")
            delta_w = backend.matmul(B, A)

        elif condition == ConditionType.PURE_RANDOM:
            # Random Gaussian B and A
            B = backend.random_normal((out_features, lora_rank), dtype="float32")
            A = backend.random_normal((lora_rank, in_features), dtype="float32")
            # Scale to control magnitude
            B = backend.multiply(B, scale)
            delta_w = backend.matmul(B, A)

        elif condition == ConditionType.TRAINED:
            # Cannot create trained adapter synthetically
            # This should be loaded from actual trained weights
            raise ValueError(
                "TRAINED condition must be loaded from actual trained adapter, "
                "not created synthetically."
            )

        elif condition == ConditionType.RANDOM_LABELS:
            # Cannot create random-labels adapter synthetically
            # This requires actual training on shuffled labels
            raise ValueError(
                "RANDOM_LABELS condition requires actual training on shuffled labels, "
                "not synthetic creation."
            )

        backend.eval(delta_w)
        delta_weights[key] = delta_w

    return SyntheticAdapter(
        condition=condition,
        delta_weights=delta_weights,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        adapter_id=adapter_id,
    )


@dataclass
class FourConditionExperiment:
    """Container for four-condition experiment data and analysis.

    Attributes:
        base_model_id: Identifier of the base model.
        adapters_per_condition: Number of adapters per condition.
        measurements: Dict mapping condition -> list of AdapterMeasurements.
        prompt_set: Fixed prompts used for output-space metrics.
    """

    base_model_id: str
    adapters_per_condition: int = 8
    measurements: dict[ConditionType, list[AdapterMeasurement]] = field(
        default_factory=dict
    )
    prompt_set: list[str] = field(default_factory=list)

    def add_measurement(
        self, condition: ConditionType, measurement: AdapterMeasurement
    ) -> None:
        """Add a measurement for a condition."""
        if condition not in self.measurements:
            self.measurements[condition] = []
        self.measurements[condition].append(measurement)

    def get_metric_by_condition(
        self, metric_name: str
    ) -> dict[ConditionType, list[float]]:
        """Extract a metric across all conditions.

        Args:
            metric_name: One of:
                - "amplification_cv" (mean across layers)
                - "weyl_utilization" (mean across layers)
                - "delta_frobenius_norm" (total across layers)
                - "conflict_score"
                - "base_frontier_rate"

        Returns:
            Dict mapping condition to list of metric values.
        """
        result: dict[ConditionType, list[float]] = {}

        for condition, measurements in self.measurements.items():
            values: list[float] = []
            for m in measurements:
                if metric_name == "amplification_cv":
                    values.append(m.mean_amplification_cv())
                elif metric_name == "weyl_utilization":
                    values.append(m.mean_weyl_utilization())
                elif metric_name == "delta_frobenius_norm":
                    values.append(m.total_frobenius_norm())
                elif metric_name == "conflict_score":
                    if m.conflict_score is not None:
                        values.append(m.conflict_score)
                elif metric_name == "base_frontier_rate":
                    if m.base_frontier_rate is not None:
                        values.append(m.base_frontier_rate)
            result[condition] = values

        return result

    def get_per_layer_metrics(
        self, condition: ConditionType, metric_name: str
    ) -> dict[int, list[float]]:
        """Get per-layer metric values for a condition.

        Args:
            condition: The condition to extract.
            metric_name: One of "amplification_cv", "weyl_utilization", etc.

        Returns:
            Dict mapping layer_idx to list of metric values across adapters.
        """
        if condition not in self.measurements:
            return {}

        result: dict[int, list[float]] = {}

        for m in self.measurements[condition]:
            for layer_m in m.layer_measurements:
                idx = layer_m.layer_idx
                if idx not in result:
                    result[idx] = []

                if metric_name == "amplification_cv":
                    result[idx].append(layer_m.amplification_cv)
                elif metric_name == "weyl_utilization":
                    result[idx].append(layer_m.weyl_utilization)
                elif metric_name == "delta_frobenius_norm":
                    result[idx].append(layer_m.delta_frobenius_norm)
                elif metric_name == "delta_spectral_norm":
                    result[idx].append(layer_m.delta_spectral_norm)

        return result


def create_four_condition_synthetic(
    base_weights: dict[str, "Array"],
    base_model_id: str,
    adapters_per_condition: int = 8,
    lora_rank: int = 8,
    lora_alpha: float = 8.0,
    scale: float = 0.01,
    backend: "Backend | None" = None,
) -> FourConditionExperiment:
    """Create synthetic adapters for UNTRAINED and PURE_RANDOM conditions.

    TRAINED and RANDOM_LABELS must be added separately from actual training.

    Args:
        base_weights: Dict mapping layer keys to base weight arrays.
        base_model_id: Identifier of the base model.
        adapters_per_condition: Number of adapters per condition.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA alpha.
        scale: Scale for random initialization.
        backend: Compute backend.

    Returns:
        FourConditionExperiment with UNTRAINED and PURE_RANDOM populated.
    """
    if backend is None:
        backend = get_default_backend()

    experiment = FourConditionExperiment(
        base_model_id=base_model_id,
        adapters_per_condition=adapters_per_condition,
    )

    # Create UNTRAINED adapters
    for i in range(adapters_per_condition):
        adapter = create_synthetic_adapter(
            base_weights=base_weights,
            condition=ConditionType.UNTRAINED,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            scale=scale,
            adapter_id=f"untrained_{i}",
            backend=backend,
        )

        # Collect measurements
        measurements: list[LayerMeasurement] = []
        for key in sorted(base_weights.keys()):
            if key not in adapter.delta_weights:
                continue

            parts = key.split(".")
            layer_idx = -1
            proj_name = key
            for j, part in enumerate(parts):
                if part == "layers" and j + 1 < len(parts):
                    try:
                        layer_idx = int(parts[j + 1])
                    except ValueError:
                        pass
                if part in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ):
                    proj_name = part

            m = collect_layer_measurements(
                weight_original=base_weights[key],
                delta_w=adapter.delta_weights[key],
                layer_idx=layer_idx,
                projection_name=proj_name,
                backend=backend,
            )
            measurements.append(m)

        adapter_measurement = AdapterMeasurement(
            adapter_id=adapter.adapter_id,
            base_model_id=base_model_id,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_domain="synthetic_untrained",
            layer_measurements=measurements,
        )
        experiment.add_measurement(ConditionType.UNTRAINED, adapter_measurement)

    # Create PURE_RANDOM adapters
    for i in range(adapters_per_condition):
        adapter = create_synthetic_adapter(
            base_weights=base_weights,
            condition=ConditionType.PURE_RANDOM,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            scale=scale,
            adapter_id=f"pure_random_{i}",
            backend=backend,
        )

        measurements = []
        for key in sorted(base_weights.keys()):
            if key not in adapter.delta_weights:
                continue

            parts = key.split(".")
            layer_idx = -1
            proj_name = key
            for j, part in enumerate(parts):
                if part == "layers" and j + 1 < len(parts):
                    try:
                        layer_idx = int(parts[j + 1])
                    except ValueError:
                        pass
                if part in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ):
                    proj_name = part

            m = collect_layer_measurements(
                weight_original=base_weights[key],
                delta_w=adapter.delta_weights[key],
                layer_idx=layer_idx,
                projection_name=proj_name,
                backend=backend,
            )
            measurements.append(m)

        adapter_measurement = AdapterMeasurement(
            adapter_id=adapter.adapter_id,
            base_model_id=base_model_id,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_domain="synthetic_pure_random",
            layer_measurements=measurements,
        )
        experiment.add_measurement(ConditionType.PURE_RANDOM, adapter_measurement)

    return experiment
