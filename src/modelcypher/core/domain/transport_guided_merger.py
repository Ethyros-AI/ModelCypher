from __future__ import annotations

from dataclasses import dataclass, field
import math
from modelcypher.core.domain.concept_response_matrix import ConceptResponseMatrix
from modelcypher.core.domain.gromov_wasserstein import Config as GWConfig
from modelcypher.core.domain.gromov_wasserstein import GromovWassersteinDistance
from modelcypher.core.domain.manifold_stitcher import IntersectionMap


@dataclass(frozen=True)
class Config:
    coupling_threshold: float = 0.001
    normalize_rows: bool = True
    blend_alpha: float = 0.5
    use_intersection_confidence: bool = True
    min_samples: int = 5
    gw_config: GWConfig = field(default_factory=GWConfig)


@dataclass(frozen=True)
class Result:
    merged_weights: list[list[float]]
    gw_distance: float
    marginal_error: float
    effective_rank: int
    converged: bool
    iterations: int
    dimension_confidences: list[float]


@dataclass(frozen=True)
class BatchResult:
    layer_results: dict[str, Result]
    mean_gw_distance: float
    mean_marginal_error: float
    failed_layers: list[str]

    @property
    def quality_score(self) -> float:
        total_layers = len(self.layer_results) + len(self.failed_layers)
        success_rate = float(len(self.layer_results)) / float(max(1, total_layers))
        convergence_rate = float(sum(1 for result in self.layer_results.values() if result.converged)) / float(
            max(1, len(self.layer_results))
        )
        distance_score = max(0.0, 1.0 - self.mean_gw_distance)
        return (success_rate + convergence_rate + distance_score) / 3.0


class TransportGuidedMerger:
    @staticmethod
    def synthesize(
        source_weights: list[list[float]],
        target_weights: list[list[float]],
        transport_plan: list[list[float]],
        config: Config | None = None,
    ) -> list[list[float]] | None:
        cfg = config or Config()
        n = len(source_weights)
        m = len(target_weights)
        if n == 0 or m == 0:
            return None
        if len(transport_plan) != n or len(transport_plan[0]) != m:
            return None

        d_source = len(source_weights[0])
        d_target = len(target_weights[0])
        if any(len(row) != d_source for row in source_weights):
            return None

        processed = transport_plan
        if cfg.coupling_threshold > 0:
            processed = TransportGuidedMerger.apply_threshold(processed, cfg.coupling_threshold)
        if cfg.normalize_rows:
            processed = TransportGuidedMerger.normalize_rows(processed)

        transport_t = TransportGuidedMerger.transpose(processed)
        merged = [[0.0 for _ in range(d_source)] for _ in range(m)]
        for j in range(m):
            for i in range(n):
                coupling = transport_t[j][i]
                if coupling <= 0:
                    continue
                for d in range(d_source):
                    merged[j][d] += coupling * source_weights[i][d]

        if d_source == d_target and cfg.blend_alpha > 0:
            alpha = cfg.blend_alpha
            one_minus = 1.0 - alpha
            for j in range(m):
                for d in range(d_source):
                    merged[j][d] = one_minus * merged[j][d] + alpha * target_weights[j][d]

        return merged

    @staticmethod
    def synthesize_with_gw(
        source_activations: list[list[float]],
        target_activations: list[list[float]],
        source_weights: list[list[float]],
        target_weights: list[list[float]],
        config: Config | None = None,
    ) -> Result | None:
        cfg = config or Config()
        sample_count = len(source_activations)
        if sample_count < cfg.min_samples:
            return None
        if sample_count != len(target_activations):
            return None
        if not source_weights or not target_weights:
            return None

        source_dist = GromovWassersteinDistance.compute_pairwise_distances(source_activations)
        target_dist = GromovWassersteinDistance.compute_pairwise_distances(target_activations)
        gw_result = GromovWassersteinDistance.compute(
            source_distances=source_dist,
            target_distances=target_dist,
            config=cfg.gw_config,
        )
        if not gw_result.converged and gw_result.iterations <= 0:
            return None

        row_error, col_error = TransportGuidedMerger.compute_marginal_error(gw_result.coupling)
        marginal_error = max(row_error, col_error)
        effective_rank = TransportGuidedMerger.compute_effective_rank(
            gw_result.coupling, threshold=cfg.coupling_threshold
        )
        dimension_confidences = TransportGuidedMerger.compute_dimension_confidences(gw_result.coupling)

        merged = TransportGuidedMerger.synthesize(
            source_weights=source_weights,
            target_weights=target_weights,
            transport_plan=gw_result.coupling,
            config=cfg,
        )
        if merged is None:
            return None

        return Result(
            merged_weights=merged,
            gw_distance=gw_result.distance,
            marginal_error=marginal_error,
            effective_rank=effective_rank,
            converged=gw_result.converged,
            iterations=gw_result.iterations,
            dimension_confidences=dimension_confidences,
        )

    @staticmethod
    def synthesize_from_crms(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        source_weights: dict[int, list[list[float]]],
        target_weights: dict[int, list[list[float]]],
        config: Config | None = None,
    ) -> BatchResult:
        cfg = config or Config()
        layer_results: dict[str, Result] = {}
        failed_layers: list[str] = []
        total_gw = 0.0
        total_marginal = 0.0

        common_layers = sorted(set(source_weights.keys()) & set(target_weights.keys()))
        for layer in common_layers:
            layer_key = f"layer_{layer}"
            source_act = source_crm.activation_matrix(layer)
            target_act = target_crm.activation_matrix(layer)
            if source_act is None or target_act is None:
                failed_layers.append(layer_key)
                continue
            src_w = source_weights.get(layer)
            tgt_w = target_weights.get(layer)
            if src_w is None or tgt_w is None:
                failed_layers.append(layer_key)
                continue
            result = TransportGuidedMerger.synthesize_with_gw(
                source_activations=source_act,
                target_activations=target_act,
                source_weights=src_w,
                target_weights=tgt_w,
                config=cfg,
            )
            if result is None:
                failed_layers.append(layer_key)
                continue
            layer_results[layer_key] = result
            total_gw += result.gw_distance
            total_marginal += result.marginal_error

        count = float(max(1, len(layer_results)))
        return BatchResult(
            layer_results=layer_results,
            mean_gw_distance=total_gw / count,
            mean_marginal_error=total_marginal / count,
            failed_layers=failed_layers,
        )

    @staticmethod
    def apply_threshold(plan: list[list[float]], threshold: float) -> list[list[float]]:
        return [[0.0 if value < threshold else value for value in row] for row in plan]

    @staticmethod
    def normalize_rows(plan: list[list[float]]) -> list[list[float]]:
        normalized: list[list[float]] = []
        for row in plan:
            total = sum(row)
            if total <= 0:
                normalized.append(list(row))
            else:
                normalized.append([value / total for value in row])
        return normalized

    @staticmethod
    def normalize_cols(plan: list[list[float]]) -> list[list[float]]:
        transposed = TransportGuidedMerger.transpose(plan)
        normalized = TransportGuidedMerger.normalize_rows(transposed)
        return TransportGuidedMerger.transpose(normalized)

    @staticmethod
    def transpose(matrix: list[list[float]]) -> list[list[float]]:
        if not matrix:
            return []
        n = len(matrix)
        m = len(matrix[0])
        result = [[0.0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            for j in range(m):
                result[j][i] = matrix[i][j]
        return result

    @staticmethod
    def compute_marginal_error(coupling: list[list[float]]) -> tuple[float, float]:
        n = len(coupling)
        if n == 0:
            return 0.0, 0.0
        m = len(coupling[0]) if coupling[0] else 0
        if m == 0:
            return 0.0, 0.0
        expected_row = 1.0 / float(n)
        expected_col = 1.0 / float(m)
        max_row = 0.0
        for row in coupling:
            max_row = max(max_row, abs(sum(row) - expected_row))
        max_col = 0.0
        for j in range(m):
            col_sum = 0.0
            for i in range(n):
                col_sum += coupling[i][j]
            max_col = max(max_col, abs(col_sum - expected_col))
        return max_row, max_col

    @staticmethod
    def compute_effective_rank(coupling: list[list[float]], threshold: float) -> int:
        return sum(1 for row in coupling for value in row if value >= threshold)

    @staticmethod
    def compute_dimension_confidences(coupling: list[list[float]]) -> list[float]:
        n = len(coupling)
        if n == 0:
            return []
        confidences = [0.0] * n
        for i in range(n):
            row = coupling[i]
            total = sum(row)
            if total <= 0:
                continue
            entropy = 0.0
            for value in row:
                p = value / total
                if p > 0:
                    entropy -= p * math.log(p)
            m = float(len(row))
            max_entropy = math.log(m) if m > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            confidences[i] = max(0.0, 1.0 - normalized_entropy)
        return confidences

    @staticmethod
    def extract_hard_assignment(coupling: list[list[float]]) -> list[int]:
        assignment: list[int] = []
        for row in coupling:
            if not row:
                assignment.append(0)
                continue
            max_idx = 0
            max_val = row[0]
            for idx, value in enumerate(row):
                if value > max_val:
                    max_val = value
                    max_idx = idx
            assignment.append(max_idx)
        return assignment

    @staticmethod
    def modulate_alpha_with_intersection(
        base_alpha: float,
        intersection_map: IntersectionMap,
        layer: int,
    ) -> float:
        confidence = None
        for entry in intersection_map.layer_confidences:
            if entry.layer == layer:
                confidence = entry.confidence
                break
        if confidence is None:
            return base_alpha
        correlation = max(0.0, min(1.0, confidence))
        modulated = base_alpha * (1.0 - correlation * 0.5)
        return max(0.0, min(1.0, modulated))
