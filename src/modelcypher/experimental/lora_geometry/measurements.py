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

"""Raw measurement collection for LoRA geometry experiments.

All computations use the Backend protocol. No NumPy/SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.lora_isometry import (
    SpectralSelectivity,
    WeylMetrics,
    compute_spectral_selectivity,
    compute_weyl_utilization,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class LayerMeasurement:
    """Per-layer geometric measurements for a LoRA adapter.

    Attributes:
        layer_idx: Layer index in the model.
        projection_name: Name of the weight projection (e.g., "q_proj", "v_proj").
        amplification_cv: Spectral selectivity CV metric.
        weyl_utilization: Weyl η metric (fraction of Weyl bound used).
        delta_frobenius_norm: ||ΔW||_F
        delta_spectral_norm: ||ΔW||_2 (largest singular value of ΔW)
        max_amplification: Max amplification ratio from spectral selectivity.
        min_amplification: Min amplification ratio from spectral selectivity.
        projection_depth: Dimensions of W needed to capture 50% of ΔW.
    """

    layer_idx: int
    projection_name: str
    amplification_cv: float
    weyl_utilization: float
    delta_frobenius_norm: float
    delta_spectral_norm: float
    max_amplification: float
    min_amplification: float
    projection_depth: int


@dataclass
class AdapterMeasurement:
    """Complete measurement for a LoRA adapter.

    Attributes:
        adapter_id: Unique identifier for the adapter.
        base_model_id: Identifier of the base model.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA alpha scaling parameter.
        training_domain: Description of training data domain.
        layer_measurements: Per-layer measurements.
        conflict_score: Output-space conflict score (if computed).
        mean_kl: Mean KL divergence from conflict analysis.
        base_frontier_rate: Fraction of tokens in base frontier.
        metadata: Additional metadata for confound control.
    """

    adapter_id: str
    base_model_id: str
    lora_rank: int
    lora_alpha: float
    training_domain: str
    layer_measurements: list[LayerMeasurement] = field(default_factory=list)
    conflict_score: float | None = None
    mean_kl: float | None = None
    base_frontier_rate: float | None = None
    metadata: dict = field(default_factory=dict)

    def mean_amplification_cv(self) -> float:
        """Compute mean amplification CV across all layers."""
        if not self.layer_measurements:
            return 0.0
        return sum(m.amplification_cv for m in self.layer_measurements) / len(
            self.layer_measurements
        )

    def mean_weyl_utilization(self) -> float:
        """Compute mean Weyl utilization across all layers."""
        if not self.layer_measurements:
            return 0.0
        return sum(m.weyl_utilization for m in self.layer_measurements) / len(
            self.layer_measurements
        )

    def total_frobenius_norm(self) -> float:
        """Compute total Frobenius norm across all layers."""
        return sum(m.delta_frobenius_norm for m in self.layer_measurements)


def collect_layer_measurements(
    weight_original: "Array",
    delta_w: "Array",
    layer_idx: int,
    projection_name: str,
    k_singular_vectors: int = 64,
    backend: "Backend | None" = None,
) -> LayerMeasurement:
    """Collect geometric measurements for a single layer.

    Args:
        weight_original: Base weight W [out_features, in_features].
        delta_w: LoRA perturbation ΔW = B @ A.
        layer_idx: Layer index.
        projection_name: Name of the projection (e.g., "q_proj").
        k_singular_vectors: Number of singular vectors for selectivity.
        backend: Compute backend.

    Returns:
        LayerMeasurement with all geometric metrics.
    """
    if backend is None:
        backend = get_default_backend()

    # Compute spectral selectivity
    selectivity: SpectralSelectivity = compute_spectral_selectivity(
        weight_original, delta_w, k=k_singular_vectors, backend=backend
    )

    # Compute Weyl metrics
    weyl: WeylMetrics = compute_weyl_utilization(weight_original, delta_w, backend)

    # Compute Frobenius norm of delta
    delta_frob = float(backend.to_scalar(backend.norm(delta_w)))

    return LayerMeasurement(
        layer_idx=layer_idx,
        projection_name=projection_name,
        amplification_cv=selectivity.amplification_cv,
        weyl_utilization=weyl.weyl_utilization,
        delta_frobenius_norm=delta_frob,
        delta_spectral_norm=weyl.delta_spectral_norm,
        max_amplification=selectivity.max_amplification,
        min_amplification=selectivity.min_amplification,
        projection_depth=weyl.projection_depth,
    )


def collect_adapter_measurement(
    adapter_id: str,
    base_model_id: str,
    base_weights: dict[str, "Array"],
    delta_weights: dict[str, "Array"],
    lora_rank: int,
    lora_alpha: float,
    training_domain: str = "unknown",
    k_singular_vectors: int = 64,
    backend: "Backend | None" = None,
) -> AdapterMeasurement:
    """Collect all geometric measurements for an adapter.

    Args:
        adapter_id: Unique identifier for the adapter.
        base_model_id: Identifier of the base model.
        base_weights: Dict mapping layer keys to base weight arrays.
        delta_weights: Dict mapping layer keys to delta weight arrays.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA alpha.
        training_domain: Description of training data domain.
        k_singular_vectors: Number of singular vectors for selectivity.
        backend: Compute backend.

    Returns:
        AdapterMeasurement with all layer measurements.

    Note:
        Weight keys should be in format "layers.{idx}.{proj_name}" or similar.
        The function parses layer index and projection name from keys.
    """
    if backend is None:
        backend = get_default_backend()

    measurements: list[LayerMeasurement] = []

    for key in sorted(base_weights.keys()):
        if key not in delta_weights:
            continue

        base_w = base_weights[key]
        delta_w = delta_weights[key]

        # Parse layer index and projection name from key
        # Expected formats: "layers.0.self_attn.q_proj.weight"
        #                   "model.layers.0.mlp.gate_proj.weight"
        parts = key.split(".")
        layer_idx = -1
        proj_name = key

        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
            # Find projection name (last meaningful part before .weight)
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

        # Ensure arrays are evaluated
        backend.eval(base_w, delta_w)

        measurement = collect_layer_measurements(
            weight_original=base_w,
            delta_w=delta_w,
            layer_idx=layer_idx,
            projection_name=proj_name,
            k_singular_vectors=k_singular_vectors,
            backend=backend,
        )
        measurements.append(measurement)

    return AdapterMeasurement(
        adapter_id=adapter_id,
        base_model_id=base_model_id,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        training_domain=training_domain,
        layer_measurements=measurements,
    )
