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

"""Concept Response Matrix (CRM) for model fingerprinting.

Captures anchor activations across layers and supports layer correspondence
via CKA-based comparisons.

References:
    - Kornblith et al. (2019). "Similarity of Neural Network Representations
      Revisited." arXiv:1905.00414
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_finite,
    machine_epsilon,
    precision_dtype,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    geodesic_cosine_matrix,
    geodesic_norms,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array


@dataclass(frozen=True)
class AnchorActivation:
    anchor_id: str
    layer: int
    activation: list[float]
    norm: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.activation:
            norm = 0.0
        else:
            backend = get_default_backend()
            arr = backend.array(self.activation)
            norms = geodesic_norms(backend.reshape(arr, (1, -1)), backend)
            backend.eval(norms)
            norm = float(backend.to_scalar(norms[0]))
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
            pooled_flat = backend.reshape(pooled, (-1,))
            backend.eval(pooled_flat)
            activation_list = backend.tolist(pooled_flat)
            if not isinstance(activation_list, list):
                activation_list = [activation_list]
            activation = [float(value) for value in activation_list]
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
            std = sqrt_scalar(max(0.0, variance), get_default_backend())
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
        """Compute geodesic CKA matrix between native layer activations.

        Geodesic CKA near 1.0 suggests strong structural overlap; lower values
        indicate divergent structure or limited probe coverage.
        """
        from modelcypher.core.domain.geometry.cka import compute_cka

        backend = get_default_backend()
        cka_matrix = [[0.0 for _ in range(other.layer_count)] for _ in range(self.layer_count)]
        common = set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        if not common:
            return cka_matrix
        sorted_anchors = sorted(common)

        # Pre-extract all layer activations for efficiency
        source_acts_cache: dict[int, tuple["Array", "Array"]] = {}
        target_acts_cache: dict[int, tuple["Array", "Array"]] = {}

        for layer in range(self.layer_count):
            activations = self._extract_activations(layer, sorted_anchors)
            if activations is not None:
                arr = backend.array(activations)
                arr = backend.astype(arr, precision_dtype(backend, reference=arr))
                backend.eval(arr)
                source_mean = backend.mean(arr, axis=0, keepdims=True)
                source_c = arr - source_mean
                source_pinv = backend.pinv(source_c)
                backend.eval(source_mean, source_c, source_pinv)
                source_acts_cache[layer] = (arr, source_pinv)

        for layer in range(other.layer_count):
            activations = other._extract_activations(layer, sorted_anchors)
            if activations is not None:
                arr = backend.array(activations)
                arr = backend.astype(arr, precision_dtype(backend, reference=arr))
                backend.eval(arr)
                target_mean = backend.mean(arr, axis=0, keepdims=True)
                target_c = arr - target_mean
                backend.eval(target_mean, target_c)
                target_acts_cache[layer] = (arr, target_c)

        # Compute alignment for each layer pair
        for source_layer in range(self.layer_count):
            source_entry = source_acts_cache.get(source_layer)
            if source_entry is None:
                continue
            source_arr, source_pinv = source_entry

            for target_layer in range(other.layer_count):
                target_entry = target_acts_cache.get(target_layer)
                if target_entry is None:
                    continue
                target_arr, target_c = target_entry

                # Reuse source pseudoinverse across all target layers for this source layer.
                # This preserves the invariant F = pinv(source_c) @ target_c and avoids
                # recomputing pinv for every pair.
                F = backend.matmul(source_pinv, target_c)
                aligned = backend.matmul(source_arr, F)
                backend.eval(F, aligned)

                # Geodesic CKA is the discriminative score for correspondence.
                result = compute_cka(aligned, target_arr, backend)
                cka = max(0.0, min(1.0, float(result.cka)))
                cka_matrix[source_layer][target_layer] = cka

        return cka_matrix

    def compute_layer_cka(
        self, source_layer: int, other: ConceptResponseMatrix, target_layer: int
    ) -> float | None:
        """Compute CKA between specific layers using Gram alignment."""
        from modelcypher.core.domain.geometry.gram_aligner import find_alignment

        common = sorted(
            set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        )
        if not common:
            return None
        source = self._extract_activations(source_layer, common)
        target = other._extract_activations(target_layer, common)
        if source is None or target is None:
            return None

        backend = get_default_backend()
        source_arr = backend.array(source)
        target_arr = backend.array(target)

        result = find_alignment(source_arr, target_arr, backend)
        cka = max(0.0, min(1.0, 1.0 - result.numerical_deviation))
        return float(cka)

    def compute_alignment_transform(self, other: ConceptResponseMatrix) -> tuple[float, list[list[float]] | None]:
        """Compute the alignment transform between representation spaces.

        Aligns Gram matrices using geodesic RBF kernels and returns the
        transform along with a precision score.

        Returns:
            (numerical_precision, T) where:
            - numerical_precision: Alignment score (1.0 is best)
            - T: The Gram-space transform matrix [n_samples, n_samples]
        """
        from modelcypher.core.domain.geometry.cka import (
            _center_gram_matrix,
            _rbf_gram_from_sq_distances,
            _shared_rbf_sigma,
            geodesic_squared_distances,
        )

        backend = get_default_backend()
        common = sorted(
            set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        )
        if not common:
            return 0.0, None

        def _stack_all_layers(crm: ConceptResponseMatrix, anchors: list[str]) -> "Array":
            """Stack activations across all layers for each anchor."""
            rows: list[list[float]] = []
            for anchor_id in anchors:
                anchor_vec: list[float] = []
                for layer in range(crm.layer_count):
                    layer_acts = crm.activations.get(layer)
                    if layer_acts is None:
                        continue
                    activation = layer_acts.get(anchor_id)
                    if activation is not None:
                        anchor_vec.extend(activation.activation)
                rows.append(anchor_vec)
            arr = backend.array(rows)
            backend.eval(arr)
            return arr

        source_repr = _stack_all_layers(self, common)
        target_repr = _stack_all_layers(other, common)

        # Compute geodesic RBF Gram matrices (dimension-invariant)
        sq_dist_s = geodesic_squared_distances(source_repr, backend)
        sq_dist_t = geodesic_squared_distances(target_repr, backend)
        sigma = _shared_rbf_sigma(sq_dist_s, sq_dist_t, backend)
        K_s = _rbf_gram_from_sq_distances(sq_dist_s, sigma, backend)
        K_t = _rbf_gram_from_sq_distances(sq_dist_t, sigma, backend)
        backend.eval(K_s, K_t)

        # Center the Gram matrices (required for CKA)
        K_s_c = _center_gram_matrix(K_s, backend)
        K_t_c = _center_gram_matrix(K_t, backend)
        backend.eval(K_s_c, K_t_c)

        # GRAM-SPACE ALIGNMENT: T = K_t^{1/2} @ K_s^{-1/2}
        # By algebra: T @ K_s @ T^T = K_t^{1/2} @ K_s^{-1/2} @ K_s @ K_s^{-1/2} @ K_t^{1/2} = K_t
        # This is MATHEMATICALLY EXACT. Any deviation is numerical precision.
        from modelcypher.core.domain.geometry.numerical_stability import (
            power_iteration_eigh,
            regularization_epsilon,
        )

        n = int(backend.shape(K_s_c)[0])
        reg_eps = regularization_epsilon(backend, K_s_c)

        # Eigendecomposition of K_s_c and K_t_c
        eig_s, vec_s = power_iteration_eigh(backend, K_s_c, k=n)
        eig_t, vec_t = power_iteration_eigh(backend, K_t_c, k=n)
        backend.eval(eig_s, vec_s, eig_t, vec_t)

        # Clamp negative eigenvalues (numerical noise) and compute sqrt/inv_sqrt
        eig_s = backend.maximum(eig_s, backend.zeros_like(eig_s))
        eig_t = backend.maximum(eig_t, backend.zeros_like(eig_t))
        backend.eval(eig_s, eig_t)

        # K_s^{-1/2} = V @ diag(1/sqrt(λ)) @ V^T
        eig_s_safe = backend.maximum(eig_s, backend.full(backend.shape(eig_s), reg_eps))
        inv_sqrt_s = 1.0 / backend.sqrt(eig_s_safe)
        K_s_inv_sqrt = backend.matmul(
            backend.matmul(vec_s, backend.diag(inv_sqrt_s)),
            backend.transpose(vec_s)
        )
        backend.eval(K_s_inv_sqrt)

        # K_t^{1/2} = V @ diag(sqrt(λ)) @ V^T
        sqrt_t = backend.sqrt(eig_t + reg_eps)
        K_t_sqrt = backend.matmul(
            backend.matmul(vec_t, backend.diag(sqrt_t)),
            backend.transpose(vec_t)
        )
        backend.eval(K_t_sqrt)

        # T = K_t^{1/2} @ K_s^{-1/2}
        T = backend.matmul(K_t_sqrt, K_s_inv_sqrt)
        backend.eval(T)

        # VERIFY ALIGNMENT: T @ K_s_c @ T^T should equal K_t_c
        K_aligned = backend.matmul(backend.matmul(T, K_s_c), backend.transpose(T))
        backend.eval(K_aligned)

        # Compute precision: 1.0 - ||K_aligned - K_t_c||_F / ||K_t_c||_F
        diff = K_aligned - K_t_c
        diff_norm_sq = backend.sum(diff * diff)
        target_gram_norm_sq = backend.sum(K_t_c * K_t_c)
        backend.eval(diff_norm_sq, target_gram_norm_sq)

        diff_norm = sqrt_scalar(float(backend.to_scalar(diff_norm_sq)), backend)
        target_gram_norm = sqrt_scalar(float(backend.to_scalar(target_gram_norm_sq)), backend)

        # Relative error (numerical precision)
        eps = division_epsilon(backend, K_t_c)
        relative_error = diff_norm / (target_gram_norm + eps) if target_gram_norm > eps else 0.0
        precision = max(0.0, min(1.0, 1.0 - relative_error))

        return precision, backend.tolist(T)

    def compare(self, other: ConceptResponseMatrix) -> "ComparisonReport":
        common = set(self.anchor_metadata.anchor_ids).intersection(other.anchor_metadata.anchor_ids)
        cka_matrix = self.compute_cka_matrix(other)

        from modelcypher.core.domain.geometry.hungarian import hungarian_assignment

        matches: list[ComparisonReport.LayerMatch] = []
        source_count = self.layer_count
        target_count = other.layer_count
        backend = get_default_backend()

        cka_arr = backend.array(cka_matrix)
        cost_arr = 1.0 - cka_arr
        pad_cost = 1.0 + float(division_epsilon(backend, cka_arr))

        if source_count < target_count:
            pad_rows = backend.full((target_count - source_count, target_count), pad_cost)
            cost_arr = backend.concatenate([cost_arr, pad_rows], axis=0)
        if target_count < source_count:
            pad_cols = backend.full((cost_arr.shape[0], source_count - target_count), pad_cost)
            cost_arr = backend.concatenate([cost_arr, pad_cols], axis=1)

        assignment_arr = hungarian_assignment(cost_arr, backend)
        assignment = [int(x) for x in backend.tolist(assignment_arr)]

        for source_layer in range(source_count):
            target_layer = assignment[source_layer]
            if target_layer >= target_count:
                continue
            cka = cka_matrix[source_layer][target_layer]
            matches.append(
                ComparisonReport.LayerMatch(
                    source_layer=source_layer,
                    target_layer=target_layer,
                    cka=float(cka),
                )
            )

        mean_cka = (
            sum(match.cka for match in matches) / float(len(matches)) if matches else 0.0
        )

        # ALIGNMENT: Compute Gram-space transform (kernel alignment constraint)
        alignment_precision, _ = self.compute_alignment_transform(other)

        report = ComparisonReport(
            source_model=self.model_identifier,
            target_model=other.model_identifier,
            common_anchor_count=len(common),
            cka_matrix=cka_matrix,
            layer_correspondence=matches,
            mean_cka=float(mean_cka),
            alignment_precision=alignment_precision,
        )

        # Check numerical precision of the Gram alignment computation
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([alignment_precision]))
        threshold = backend.sqrt(backend.array([eps]))
        precision_tolerance = float(backend.to_scalar(threshold))

        if alignment_precision < 1.0 - precision_tolerance:
            logger.warning(
                "Alignment precision=%.4f (expected 1.0). "
                "This is a numerical precision issue in the algorithm, NOT model incompatibility. "
                "Kernel alignment in Gram space is exact by construction.",
                alignment_precision,
            )

        return report

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

            transition_cka = self.compute_geodesic_cka(source_delta, target_delta)
            state_cka = self.compute_geodesic_cka(source_current, target_current)
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
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([mean_state]))
        advantage = mean_transition / mean_state if mean_state > eps else 0.0

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
        target_alignment: dict[int, float] = {}
        epsilon = division_epsilon(backend, reference)

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
            target_alignment[layer] = float(max(0.0, min(1.0, weight)))

        sampled_layers = sorted(target_alignment.keys())
        full_alignment = _interpolate_layer_alignment(
            sample_layers=sampled_layers,
            sample_alignment=target_alignment,
            layer_count=layer_count,
        )

        return ConsistencyProfile(
            anchor_count=len(common),
            sample_layer_count=len(sample_matrices),
            mean_source_distance=source_distance_sum / sampled,
            mean_target_distance=target_distance_sum / sampled,
            target_alignment_by_layer=full_alignment,
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
    def compute_geodesic_cka(x: list[list[float]], y: list[list[float]]) -> float:
        """Compute geodesic CKA between activation matrices."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.cka import (
            HSICEstimator,
            compute_cka,
        )
        from modelcypher.core.domain.geometry.numerical_stability import precision_dtype

        backend = get_default_backend()
        arr_x = backend.array(x)
        arr_y = backend.array(y)
        arr_x = backend.astype(arr_x, precision_dtype(backend, reference=arr_x))
        arr_y = backend.astype(arr_y, precision_dtype(backend, reference=arr_y))

        result = compute_cka(
            arr_x,
            arr_y,
            backend=backend,
            estimator=HSICEstimator.AUTO,
            feature_bias_correction=True,
        )
        return result.best if result.is_valid else 0.0

    @staticmethod
    def compute_linear_cka(x: list[list[float]], y: list[list[float]]) -> float:
        """DEPRECATED: use compute_geodesic_cka."""
        return ConceptResponseMatrix.compute_geodesic_cka(x, y)

    @staticmethod
    def _compute_layer_delta(
        current: list[list[float]],
        next_layer: list[list[float]],
    ) -> tuple[list[list[float]], float]:
        if len(current) != len(next_layer) or not current:
            return ([], 0.0)

        backend = get_default_backend()

        # Vectorized: convert to arrays and compute delta in one operation
        current_arr = backend.array(current)
        next_arr = backend.array(next_layer)
        delta_arr = next_arr - current_arr
        backend.eval(delta_arr)

        # Compute all norms at once using geodesic_norms (handles all rows)
        norms_arr = geodesic_norms(delta_arr, backend)
        backend.eval(norms_arr)

        # Mean norm - single backend operation
        mean_norm_arr = backend.mean(norms_arr)
        backend.eval(mean_norm_arr)
        mean_norm = float(backend.to_scalar(mean_norm_arr))

        # Convert delta back to list for downstream CKA compatibility
        delta = backend.tolist(delta_arr)

        return delta, mean_norm


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison report between two ConceptResponseMatrices.

    alignment_precision: Numerical precision of the alignment computation.
        This field reports how close the Gram-space kernel alignment got to
        the ideal target of 1.0. Values < 1.0 may reflect numerical precision
        limits rather than model incompatibility.
    """

    source_model: str
    target_model: str
    common_anchor_count: int
    cka_matrix: list[list[float]]
    layer_correspondence: list["ComparisonReport.LayerMatch"]
    mean_cka: float
    alignment_precision: float = 1.0  # Numerical precision (1.0 = exact)

    @dataclass(frozen=True)
    class LayerMatch:
        source_layer: int
        target_layer: int
        cka: float

        @property
        def is_perfect(self) -> bool:
            """True if geodesic CKA is within sqrt(machine_epsilon) of 1.0.

            Uses a dtype-derived tolerance to account for numerical error in
            matrix operations (sqrt, pinv, matmul).
            """
            backend = get_default_backend()
            eps = machine_epsilon(backend, backend.array([self.cka]))
            threshold = backend.sqrt(backend.array([eps]))
            return self.cka >= 1.0 - float(backend.to_scalar(threshold))

    @property
    def is_perfect(self) -> bool:
        """True if all layer matches are within precision of 1.0."""
        return all(match.is_perfect for match in self.layer_correspondence)

    @property
    def imperfect_matches(self) -> list["ComparisonReport.LayerMatch"]:
        """Return layer matches below the CKA diagnostic threshold."""
        return [m for m in self.layer_correspondence if not m.is_perfect]


@dataclass(frozen=True)
class LayerTransitionResult:
    from_layer: int
    to_layer: int
    transition_cka: float
    state_cka: float
    transition_cka_ratio: float
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
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([state_cka]))
        transition_cka_ratio = float(transition_cka) / float(state_cka) if state_cka > eps else 0.0
        object.__setattr__(self, "transition_cka_ratio", float(transition_cka_ratio))
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
    target_alignment_by_layer: dict[int, float]


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
    return geodesic_cosine_matrix(arr, backend)


def _mean_absolute_difference(a: "Array", b: "Array") -> float:
    backend = get_default_backend()
    if a.shape != b.shape or a.size == 0:
        return 0.0
    mad_arr = backend.mean(backend.abs(a - b))
    backend.eval(mad_arr)
    return float(backend.to_scalar(mad_arr))


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


def _interpolate_layer_alignment(
    sample_layers: list[int],
    sample_alignment: dict[int, float],
    layer_count: int,
) -> dict[int, float]:
    if layer_count <= 0 or not sample_layers or not sample_alignment:
        return {}

    sorted_layers = sorted(sample_layers)
    weights: dict[int, float] = {}
    default_weight = sum(sample_alignment.values()) / float(len(sample_alignment))

    first_layer = sorted_layers[0]
    first_weight = sample_alignment.get(first_layer, default_weight)
    for layer in range(0, first_layer):
        weights[layer] = float(first_weight)

    for idx in range(len(sorted_layers) - 1):
        left = sorted_layers[idx]
        right = sorted_layers[idx + 1]
        left_weight = sample_alignment.get(left, default_weight)
        right_weight = sample_alignment.get(right, left_weight)
        span = max(1, right - left)
        for layer in range(left, right + 1):
            t = float(layer - left) / float(span)
            weights[layer] = float(left_weight + (right_weight - left_weight) * t)

    last_layer = sorted_layers[-1]
    last_weight = sample_alignment.get(last_layer, default_weight)
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
