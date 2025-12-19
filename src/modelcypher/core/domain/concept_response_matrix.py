from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import json
import math

import numpy as np


@dataclass(frozen=True)
class AnchorActivation:
    anchor_id: str
    layer: int
    activation: list[float]
    norm: float = field(init=False)

    def __post_init__(self) -> None:
        norm = math.sqrt(sum(float(value) * float(value) for value in self.activation)) if self.activation else 0.0
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
        for layer, state in layer_states.items():
            pooled = _mean_pool_state(state)
            activation = pooled.astype(np.float32).reshape(-1).tolist()
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
            hidden_dim = next(iter(layer_acts.values())).activation.__len__() if layer_acts else self.hidden_dim
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

    def activation_matrix_for_category(self, category: AnchorCategory, layer: int) -> list[list[float]] | None:
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
        cka_matrix = [[0.0 for _ in range(other.layer_count)] for _ in range(self.layer_count)]
        common = set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        if not common:
            return cka_matrix
        sorted_anchors = sorted(common)
        for source_layer in range(self.layer_count):
            for target_layer in range(other.layer_count):
                source = self._extract_activations(source_layer, sorted_anchors)
                target = other._extract_activations(target_layer, sorted_anchors)
                if source is None or target is None:
                    continue
                cka_matrix[source_layer][target_layer] = float(self.compute_linear_cka(source, target))
        return cka_matrix

    def compute_layer_cka(self, source_layer: int, other: ConceptResponseMatrix, target_layer: int) -> float | None:
        common = sorted(set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids))
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
        if not x or not y or len(x) != len(y):
            return 0.0
        x_centered = _center_matrix(x)
        y_centered = _center_matrix(y)
        k = _gram_matrix(x_centered)
        l = _gram_matrix(y_centered)
        hsic_xy = _frobenius_inner_product(k, l)
        hsic_xx = _frobenius_inner_product(k, k)
        hsic_yy = _frobenius_inner_product(l, l)
        denom = math.sqrt(hsic_xx * hsic_yy)
        if denom <= 1e-10:
            return 0.0
        return float(hsic_xy / denom)


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


def _mean_pool_state(state: Any) -> np.ndarray:
    array = np.array(state, dtype=np.float32)
    if array.ndim == 3:
        pooled = array.mean(axis=(0, 1))
    elif array.ndim == 2:
        pooled = array.mean(axis=0)
    else:
        pooled = array
    return pooled


def _center_matrix(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix or not matrix[0]:
        return matrix
    array = np.array(matrix, dtype=np.float32)
    means = array.mean(axis=0, keepdims=True)
    centered = array - means
    return centered.tolist()


def _gram_matrix(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    array = np.array(matrix, dtype=np.float32)
    gram = array @ array.T
    return gram.tolist()


def _frobenius_inner_product(a: list[list[float]], b: list[list[float]]) -> float:
    if not a or not b:
        return 0.0
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        return 0.0
    return float(np.sum(arr_a * arr_b))


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    iso = value.isoformat().replace("+00:00", "Z")
    return iso


def _decode_datetime(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw.replace("Z", "+00:00")
    return datetime.fromisoformat(raw)
