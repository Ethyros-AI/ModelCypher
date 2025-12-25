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

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from modelcypher.core.domain._backend import get_default_backend


@dataclass(frozen=True)
class AnchorActivation:
    anchor_id: str
    layer: int
    activation: list[float]
    norm: float = field(init=False)

    def __post_init__(self) -> None:
        norm = (
            math.sqrt(sum(float(value) * float(value) for value in self.activation))
            if self.activation
            else 0.0
        )
        object.__setattr__(self, "norm", float(norm))


@dataclass(frozen=True)
class LayerStatistics:
    layer: int
    anchor_count: int
    mean_activation_norm: float
    std_activation_norm: float
    hidden_dim: int


class AnchorCategory(str, Enum):
    semantic_prime = "prime"
    computational_gate = "gate"

    @property
    def prefix(self) -> str:
        return f"{self.value}:"


@dataclass(frozen=True)
class AnchorMetadata:
    total_count: int
    semantic_prime_count: int
    computational_gate_count: int
    anchor_ids: list[str]


@dataclass
class ConceptResponseMatrix:
    model_identifier: str
    layer_count: int
    hidden_dim: int
    anchor_metadata: AnchorMetadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activations: dict[int, dict[str, AnchorActivation]] = field(default_factory=dict)

    def record_activations(self, anchor_id: str, layer_states: dict[int, Any]) -> None:
        backend = get_default_backend()
        for layer, state in layer_states.items():
            pooled = _mean_pool_state(state, backend)
            activation = backend.to_numpy(pooled).reshape(-1).tolist()
            if layer not in self.activations:
                self.activations[layer] = {}
            self.activations[layer][anchor_id] = AnchorActivation(
                anchor_id=anchor_id,
                layer=int(layer),
                activation=activation,
            )

    def compute_layer_statistics(self) -> list[LayerStatistics]:
        stats: list[LayerStatistics] = []
        for layer in range(self.layer_count):
            layer_acts = self.activations.get(layer)
            if not layer_acts:
                continue
            norms = [activation.norm for activation in layer_acts.values()]
            anchor_count = len(norms)
            if anchor_count == 0:
                continue
            mean = sum(norms) / float(anchor_count)
            variance = sum((value - mean) ** 2 for value in norms) / float(anchor_count)
            std = math.sqrt(max(0.0, variance))
            hidden_dim = (
                next(iter(layer_acts.values())).activation.__len__()
                if layer_acts
                else self.hidden_dim
            )
            stats.append(
                LayerStatistics(
                    layer=layer,
                    anchor_count=anchor_count,
                    mean_activation_norm=float(mean),
                    std_activation_norm=float(std),
                    hidden_dim=int(hidden_dim),
                )
            )
        return sorted(stats, key=lambda item: item.layer)

    def activation_matrix(self, layer: int) -> list[list[float]] | None:
        layer_acts = self.activations.get(layer)
        if layer_acts is None:
            return None
        matrix: list[list[float]] = []
        for anchor_id in self.anchor_metadata.anchor_ids:
            activation = layer_acts.get(anchor_id)
            if activation is not None:
                matrix.append(activation.activation)
        return matrix or None

    def common_anchor_ids(self, other: "ConceptResponseMatrix") -> list[str]:
        return sorted(
            set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        )

    def activation_matrix_for_category(
        self, category: AnchorCategory, layer: int
    ) -> list[list[float]] | None:
        layer_acts = self.activations.get(layer)
        if layer_acts is None:
            return None
        prefix = category.prefix
        matrix: list[list[float]] = []
        for anchor_id in self.anchor_metadata.anchor_ids:
            if not anchor_id.startswith(prefix):
                continue
            activation = layer_acts.get(anchor_id)
            if activation is not None:
                matrix.append(activation.activation)
        return matrix or None

    def compute_cka_matrix(self, other: ConceptResponseMatrix) -> list[list[float]]:
        backend = get_default_backend()
        cka_matrix = [[0.0 for _ in range(other.layer_count)] for _ in range(self.layer_count)]
        common = set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        if not common:
            return cka_matrix
        sorted_anchors = sorted(common)
        source_grams: dict[int, tuple["Array", float]] = {}
        target_grams: dict[int, tuple["Array", float]] = {}

        for layer in range(self.layer_count):
            activations = self._extract_activations(layer, sorted_anchors)
            if activations is None:
                continue
            array = backend.array(activations)
            if array.size == 0:
                continue
            centered = array - backend.mean(array, axis=0, keepdims=True)
            gram = centered @ centered.T
            frob = float(backend.to_numpy(backend.sum(gram * gram)))
            source_grams[layer] = (gram, frob)

        for layer in range(other.layer_count):
            activations = other._extract_activations(layer, sorted_anchors)
            if activations is None:
                continue
            array = backend.array(activations)
            if array.size == 0:
                continue
            centered = array - backend.mean(array, axis=0, keepdims=True)
            gram = centered @ centered.T
            frob = float(backend.to_numpy(backend.sum(gram * gram)))
            target_grams[layer] = (gram, frob)

        for source_layer in range(self.layer_count):
            source_entry = source_grams.get(source_layer)
            if source_entry is None:
                continue
            source_gram, source_frob = source_entry
            if source_frob <= 1e-10:
                continue
            for target_layer in range(other.layer_count):
                target_entry = target_grams.get(target_layer)
                if target_entry is None:
                    continue
                target_gram, target_frob = target_entry
                denom = math.sqrt(source_frob * target_frob)
                if denom <= 1e-10:
                    continue
                hsic_xy = float(backend.to_numpy(backend.sum(source_gram * target_gram)))
                cka_matrix[source_layer][target_layer] = float(hsic_xy / denom)
        return cka_matrix

    def compute_layer_cka(
        self, source_layer: int, other: ConceptResponseMatrix, target_layer: int
    ) -> float | None:
        common = sorted(
            set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        )
        if not common:
            return None
        source = self._extract_activations(source_layer, common)
        target = other._extract_activations(target_layer, common)
        if source is None or target is None:
            return None
        return float(self.compute_linear_cka(source, target))

    def compare(self, other: ConceptResponseMatrix) -> "ComparisonReport":
        common = set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        cka_matrix = self.compute_cka_matrix(other)

        matches: list[ComparisonReport.LayerMatch] = []
        used_targets: set[int] = set()

        for source_layer in range(self.layer_count):
            best_target = -1
            best_cka = -1.0
            for target_layer in range(other.layer_count):
                if target_layer in used_targets:
                    continue
                cka = cka_matrix[source_layer][target_layer]
                if cka > best_cka:
                    best_cka = cka
                    best_target = target_layer
            if best_target >= 0:
                used_targets.add(best_target)
                matches.append(
                    ComparisonReport.LayerMatch(
                        source_layer=source_layer,
                        target_layer=best_target,
                        cka=float(best_cka),
                    )
                )

        overall_alignment = (
            sum(match.cka for match in matches) / float(len(matches)) if matches else 0.0
        )

        return ComparisonReport(
            source_model=self.model_identifier,
            target_model=other.model_identifier,
            common_anchor_count=len(common),
            cka_matrix=cka_matrix,
            layer_correspondence=matches,
            overall_alignment=float(overall_alignment),
        )

    def compute_transition_alignment(
        self, other: "ConceptResponseMatrix"
    ) -> TransitionExperiment | None:
        common = sorted(
            set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        )
        if len(common) < 3:
            return None

        max_layer = min(self.layer_count, other.layer_count) - 1
        if max_layer < 1:
            return None

        transitions: list[LayerTransitionResult] = []
        for layer in range(max_layer):
            next_layer = layer + 1
            source_current = self._extract_activations(layer, common)
            source_next = self._extract_activations(next_layer, common)
            target_current = other._extract_activations(layer, common)
            target_next = other._extract_activations(next_layer, common)
            if (
                source_current is None
                or source_next is None
                or target_current is None
                or target_next is None
            ):
                continue

            source_delta, source_norm = self._compute_layer_delta(source_current, source_next)
            target_delta, target_norm = self._compute_layer_delta(target_current, target_next)
            if not source_delta or not target_delta:
                continue

            transition_cka = self.compute_linear_cka(source_delta, target_delta)
            state_cka = self.compute_linear_cka(source_current, target_current)
            transitions.append(
                LayerTransitionResult(
                    from_layer=layer,
                    to_layer=next_layer,
                    transition_cka=float(transition_cka),
                    state_cka=float(state_cka),
                    source_delta_norm=float(source_norm),
                    target_delta_norm=float(target_norm),
                )
            )

        if not transitions:
            return None

        mean_transition = sum(item.transition_cka for item in transitions) / float(len(transitions))
        mean_state = sum(item.state_cka for item in transitions) / float(len(transitions))
        advantage = mean_transition / mean_state if mean_state > 0.001 else 0.0

        return TransitionExperiment(
            source_model=self.model_identifier,
            target_model=other.model_identifier,
            timestamp=datetime.now(timezone.utc),
            transitions=transitions,
            mean_transition_cka=float(mean_transition),
            mean_state_cka=float(mean_state),
            transition_better_than_state=mean_transition > mean_state,
            transition_advantage=float(advantage),
            anchor_count=len(common),
            layer_transition_count=len(transitions),
        )

    def compute_consistency_profile(
        self,
        other: "ConceptResponseMatrix",
        layer_sample_count: int = 6,
    ) -> ConsistencyProfile | None:
        backend = get_default_backend()
        common = sorted(
            set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        )
        if len(common) < 4:
            return None

        layer_count = min(self.layer_count, other.layer_count)
        if layer_count <= 0:
            return None

        sample_count = min(max(2, layer_sample_count), layer_count)
        sample_layers = _sample_layer_indices(layer_count, sample_count)

        source_sum: "Array | None" = None
        target_sum: "Array | None" = None
        sample_matrices: dict[int, tuple["Array", "Array"]] = {}

        for layer in sample_layers:
            source_act = self._extract_activations(layer, common)
            target_act = other._extract_activations(layer, common)
            if source_act is None or target_act is None:
                continue
            source_matrix = _cosine_similarity_matrix(source_act)
            target_matrix = _cosine_similarity_matrix(target_act)
            if source_matrix is None or target_matrix is None:
                continue

            if source_sum is None:
                source_sum = backend.zeros_like(source_matrix)
                target_sum = backend.zeros_like(target_matrix)
            source_sum = source_sum + source_matrix
            target_sum = target_sum + target_matrix
            sample_matrices[layer] = (source_matrix, target_matrix)

        if len(sample_matrices) < 2 or source_sum is None or target_sum is None:
            return None

        sampled = float(len(sample_matrices))
        source_mean = source_sum / sampled
        target_mean = target_sum / sampled
        reference = 0.5 * (source_mean + target_mean)

        source_distance_sum = 0.0
        target_distance_sum = 0.0
        target_weights: dict[int, float] = {}
        epsilon = 1e-6

        for layer, (source_matrix, target_matrix) in sample_matrices.items():
            source_distance = float(_mean_absolute_difference(source_matrix, reference))
            target_distance = float(_mean_absolute_difference(target_matrix, reference))
            source_distance_sum += source_distance
            target_distance_sum += target_distance

            max_distance = max(source_distance, target_distance)
            inv_source = max_distance - source_distance
            inv_target = max_distance - target_distance
            denom = inv_source + inv_target
            weight = inv_target / denom if denom > epsilon else 0.5
            target_weights[layer] = float(max(0.0, min(1.0, weight)))

        sampled_layers = sorted(target_weights.keys())
        full_weights = _interpolate_layer_weights(
            sample_layers=sampled_layers,
            sample_weights=target_weights,
            layer_count=layer_count,
        )

        return ConsistencyProfile(
            anchor_count=len(common),
            sample_layer_count=len(sample_matrices),
            mean_source_distance=source_distance_sum / sampled,
            mean_target_distance=target_distance_sum / sampled,
            target_weight_by_layer=full_weights,
        )

    def save(self, path: str) -> None:
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)

    @staticmethod
    def load(path: str) -> "ConceptResponseMatrix":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ConceptResponseMatrix.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelIdentifier": self.model_identifier,
            "createdAt": _encode_datetime(self.created_at),
            "layerCount": self.layer_count,
            "hiddenDim": self.hidden_dim,
            "anchorMetadata": {
                "totalCount": self.anchor_metadata.total_count,
                "semanticPrimeCount": self.anchor_metadata.semantic_prime_count,
                "computationalGateCount": self.anchor_metadata.computational_gate_count,
                "anchorIDs": self.anchor_metadata.anchor_ids,
            },
            "activations": {
                str(layer): {
                    anchor_id: {
                        "anchorID": activation.anchor_id,
                        "layer": activation.layer,
                        "activation": activation.activation,
                        "norm": activation.norm,
                    }
                    for anchor_id, activation in layer_acts.items()
                }
                for layer, layer_acts in self.activations.items()
            },
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ConceptResponseMatrix":
        anchor_meta = payload["anchorMetadata"]
        metadata = AnchorMetadata(
            total_count=int(anchor_meta["totalCount"]),
            semantic_prime_count=int(anchor_meta["semanticPrimeCount"]),
            computational_gate_count=int(anchor_meta["computationalGateCount"]),
            anchor_ids=[str(value) for value in anchor_meta["anchorIDs"]],
        )
        crm = ConceptResponseMatrix(
            model_identifier=str(payload["modelIdentifier"]),
            layer_count=int(payload["layerCount"]),
            hidden_dim=int(payload["hiddenDim"]),
            anchor_metadata=metadata,
            created_at=_decode_datetime(payload["createdAt"]),
        )
        activations: dict[int, dict[str, AnchorActivation]] = {}
        for layer_key, layer_values in payload.get("activations", {}).items():
            layer_index = int(layer_key)
            activations[layer_index] = {}
            for anchor_id, raw in layer_values.items():
                activation = AnchorActivation(
                    anchor_id=str(raw.get("anchorID", anchor_id)),
                    layer=int(raw.get("layer", layer_index)),
                    activation=[float(value) for value in raw.get("activation", [])],
                )
                activations[layer_index][anchor_id] = activation
        crm.activations = activations
        return crm

    def _extract_activations(self, layer: int, anchors: list[str]) -> list[list[float]] | None:
        layer_acts = self.activations.get(layer)
        if layer_acts is None:
            return None
        matrix: list[list[float]] = []
        for anchor_id in anchors:
            activation = layer_acts.get(anchor_id)
            if activation is None:
                return None
            matrix.append(activation.activation)
        return matrix

    @staticmethod
    def compute_linear_cka(x: list[list[float]], y: list[list[float]]) -> float:
        """Compute linear CKA between activation matrices.

        Delegates to the canonical CKA implementation in cka.py.
        """
        from modelcypher.core.domain.geometry.cka import compute_cka_from_lists

        return compute_cka_from_lists(x, y)

    @staticmethod
    def _compute_layer_delta(
        current: list[list[float]],
        next_layer: list[list[float]],
    ) -> tuple[list[list[float]], float]:
        if len(current) != len(next_layer) or not current:
            return ([], 0.0)

        delta: list[list[float]] = []
        total_norm = 0.0
        for curr, nxt in zip(current, next_layer):
            if len(curr) != len(nxt):
                continue
            diff = [float(nxt[idx] - curr[idx]) for idx in range(len(curr))]
            total_norm += math.sqrt(sum(value * value for value in diff))
            delta.append(diff)

        mean_norm = total_norm / float(len(delta)) if delta else 0.0
        return delta, float(mean_norm)


@dataclass(frozen=True)
class ComparisonReport:
    source_model: str
    target_model: str
    common_anchor_count: int
    cka_matrix: list[list[float]]
    layer_correspondence: list["ComparisonReport.LayerMatch"]
    overall_alignment: float

    @dataclass(frozen=True)
    class LayerMatch:
        source_layer: int
        target_layer: int
        cka: float


@dataclass(frozen=True)
class LayerTransitionResult:
    from_layer: int
    to_layer: int
    transition_cka: float
    state_cka: float
    delta_alignment: float
    source_delta_norm: float
    target_delta_norm: float

    def __init__(
        self,
        from_layer: int,
        to_layer: int,
        transition_cka: float,
        state_cka: float,
        source_delta_norm: float,
        target_delta_norm: float,
    ) -> None:
        object.__setattr__(self, "from_layer", int(from_layer))
        object.__setattr__(self, "to_layer", int(to_layer))
        object.__setattr__(self, "transition_cka", float(transition_cka))
        object.__setattr__(self, "state_cka", float(state_cka))
        delta_alignment = float(transition_cka) / float(state_cka) if state_cka > 0.001 else 0.0
        object.__setattr__(self, "delta_alignment", float(delta_alignment))
        object.__setattr__(self, "source_delta_norm", float(source_delta_norm))
        object.__setattr__(self, "target_delta_norm", float(target_delta_norm))


@dataclass(frozen=True)
class TransitionExperiment:
    source_model: str
    target_model: str
    timestamp: datetime
    transitions: list[LayerTransitionResult]
    mean_transition_cka: float
    mean_state_cka: float
    transition_better_than_state: bool
    transition_advantage: float
    anchor_count: int
    layer_transition_count: int


@dataclass(frozen=True)
class ConsistencyProfile:
    anchor_count: int
    sample_layer_count: int
    mean_source_distance: float
    mean_target_distance: float
    target_weight_by_layer: dict[int, float]


def _mean_pool_state(state: Any, backend: Any) -> "Array":
    array = backend.array(state)
    if array.ndim == 3:
        pooled = backend.mean(array, axis=(0, 1))
    elif array.ndim == 2:
        pooled = backend.mean(array, axis=0)
    else:
        pooled = array
    return pooled


def _cosine_similarity_matrix(activations: list[list[float]]) -> "Array | None":
    backend = get_default_backend()
    if not activations:
        return None
    arr = backend.array(activations)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    norms = backend.norm(arr, axis=1, keepdims=True)
    norms = backend.clip(norms, 1e-8, None)
    normalized = arr / norms
    return normalized @ normalized.T


def _mean_absolute_difference(a: "Array", b: "Array") -> float:
    backend = get_default_backend()
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(backend.to_numpy(backend.mean(backend.abs(a - b))))


def _sample_layer_indices(layer_count: int, sample_count: int) -> list[int]:
    if layer_count <= 0:
        return []
    if sample_count <= 1:
        return [layer_count // 2]
    if sample_count >= layer_count:
        return list(range(layer_count))

    stride = float(layer_count - 1) / float(sample_count - 1)
    indices = [
        min(layer_count - 1, max(0, int(round(idx * stride)))) for idx in range(sample_count)
    ]
    unique = sorted(set(indices))
    if 0 not in unique:
        unique.insert(0, 0)
    if (layer_count - 1) not in unique:
        unique.append(layer_count - 1)
    return unique


def _interpolate_layer_weights(
    sample_layers: list[int],
    sample_weights: dict[int, float],
    layer_count: int,
) -> dict[int, float]:
    if layer_count <= 0 or not sample_layers:
        return {}

    sorted_layers = sorted(sample_layers)
    weights: dict[int, float] = {}

    first_layer = sorted_layers[0]
    first_weight = sample_weights.get(first_layer, 0.5)
    for layer in range(0, first_layer):
        weights[layer] = float(first_weight)

    for idx in range(len(sorted_layers) - 1):
        left = sorted_layers[idx]
        right = sorted_layers[idx + 1]
        left_weight = sample_weights.get(left, 0.5)
        right_weight = sample_weights.get(right, left_weight)
        span = max(1, right - left)
        for layer in range(left, right + 1):
            t = float(layer - left) / float(span)
            weights[layer] = float(left_weight + (right_weight - left_weight) * t)

    last_layer = sorted_layers[-1]
    last_weight = sample_weights.get(last_layer, 0.5)
    if last_layer < layer_count - 1:
        for layer in range(last_layer + 1, layer_count):
            weights[layer] = float(last_weight)

    return weights


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    iso = value.isoformat().replace("+00:00", "Z")
    return iso


def _decode_datetime(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(raw)
