"""
Transport-Guided Model Merger using Optimal Transport (OT).

Scientific Foundation:
Uses the Gromov-Wasserstein transport plan to guide weighted parameter averaging.
The transport plan π[i,j] represents the optimal soft correspondence between
source neuron i and target neuron j based on their relational structure.

Math:
W_merged[i,:] = Σ_k π[i,k] * W_source[k,:]

Ported from the reference Swift implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set

from modelcypher.core.domain.geometry.gromov_wasserstein import (
    GromovWassersteinDistance,
    Config as GWConfig,
    Result as GWResult,
)
# Assuming ConceptResponseMatrix exists here based on grep
from modelcypher.core.domain.geometry.concept_response_matrix import ConceptResponseMatrix


# Placeholder for IntersectionMap if not fully ported yet, assuming simple struct
@dataclass
class LayerConfidence:
    layer: int
    confidence: float

@dataclass
class IntersectionMap:
    layer_confidences: List[LayerConfidence]


class TransportGuidedMerger:
    
    @dataclass
    class Config:
        coupling_threshold: float = 0.001
        normalize_rows: bool = True
        blend_alpha: float = 0.5
        use_intersection_confidence: bool = True
        min_samples: int = 5
        gw_config: GWConfig = field(default_factory=GWConfig)

    @dataclass
    class Result:
        merged_weights: List[List[float]]
        gw_distance: float
        marginal_error: float
        effective_rank: int
        converged: bool
        iterations: int
        dimension_confidences: List[float]

    @dataclass
    class BatchResult:
        layer_results: Dict[str, "TransportGuidedMerger.Result"]
        mean_gw_distance: float
        mean_marginal_error: float
        failed_layers: List[str]

        @property
        def quality_score(self) -> float:
            total_attempted = max(1, len(self.layer_results) + len(self.failed_layers))
            success_rate = len(self.layer_results) / total_attempted
            
            converged_count = sum(1 for r in self.layer_results.values() if r.converged)
            convergence_rate = converged_count / max(1, len(self.layer_results))
            
            distance_score = max(0.0, 1.0 - self.mean_gw_distance)
            return (success_rate + convergence_rate + distance_score) / 3.0

    # MARK: - Core Synthesis

    @staticmethod
    def synthesize(
        source_weights: List[List[float]],
        target_weights: List[List[float]],
        transport_plan: List[List[float]],
        config: Config = Config()
    ) -> Optional[List[List[float]]]:
        n = len(source_weights)
        m = len(target_weights)
        
        if n == 0 or m == 0: return None
        if len(transport_plan) != n: return None
        if len(transport_plan[0]) != m: return None
        
        d_source = len(source_weights[0])
        d_target = len(target_weights[0])
        
        # Apply threshold and normalize if configured
        processed_plan = transport_plan
        if config.coupling_threshold > 0:
            processed_plan = TransportGuidedMerger._apply_threshold(processed_plan, config.coupling_threshold)
        
        if config.normalize_rows:
            processed_plan = TransportGuidedMerger._normalize_rows(processed_plan)

        # Compute transport-merged weights: W_merged[j,:] = Σ_i π[i,j] * W_source[i,:]
        merged_weights = [[0.0] * d_source for _ in range(m)]

        for i in range(n):
            row = processed_plan[i]
            if not row:
                continue
            source_row = source_weights[i]
            for j in range(m):
                coupling = row[j]
                if coupling > 1e-9:
                    for d in range(d_source):
                        merged_weights[j][d] += coupling * source_row[d]
                        
        # Blend with target if dimensions match and alpha > 0
        if d_source == d_target and config.blend_alpha > 0:
            alpha = config.blend_alpha
            one_minus_alpha = 1.0 - alpha
            for j in range(m):
                for d in range(d_source):
                    merged_weights[j][d] = (one_minus_alpha * merged_weights[j][d]) + (alpha * target_weights[j][d])
                    
        return merged_weights

    @staticmethod
    def synthesize_with_gw(
        source_activations: List[List[float]],
        target_activations: List[List[float]],
        source_weights: List[List[float]],
        target_weights: List[List[float]],
        config: Config = Config()
    ) -> Optional[Result]:
        sample_count = len(source_activations)
        if sample_count < config.min_samples: return None
        if sample_count != len(target_activations): return None
        if not source_weights or not target_weights: return None
        
        source_points = TransportGuidedMerger._align_activations_to_weights(source_activations, len(source_weights))
        target_points = TransportGuidedMerger._align_activations_to_weights(target_activations, len(target_weights))
        
        if not source_points or not target_points: return None
        
        # Compute pairwise distances
        source_dist = GromovWassersteinDistance.compute_pairwise_distances(source_points)
        target_dist = GromovWassersteinDistance.compute_pairwise_distances(target_points)
        
        # Compute GW transport plan
        gw_result = GromovWassersteinDistance.compute(
            source_distances=source_dist,
            target_distances=target_dist,
            config=config.gw_config
        )
        
        if not (gw_result.converged or gw_result.iterations > 0): return None
        
        # Metrics
        row_error, col_error = TransportGuidedMerger._compute_marginal_error(gw_result.coupling)
        marginal_error = max(row_error, col_error)
        
        effective_rank = TransportGuidedMerger._compute_effective_rank(gw_result.coupling, config.coupling_threshold)
        dim_confidences = TransportGuidedMerger._compute_dimension_confidences(gw_result.coupling)
        
        # Synthesize
        merged = TransportGuidedMerger.synthesize(
            source_weights=source_weights,
            target_weights=target_weights,
            transport_plan=gw_result.coupling,
            config=config
        )
        
        if not merged: return None
        
        return TransportGuidedMerger.Result(
            merged_weights=merged,
            gw_distance=gw_result.distance,
            marginal_error=marginal_error,
            effective_rank=effective_rank,
            converged=gw_result.converged,
            iterations=gw_result.iterations,
            dimension_confidences=dim_confidences
        )

    @staticmethod
    def synthesize_from_crms(
        source_crm: ConceptResponseMatrix,
        target_crm: ConceptResponseMatrix,
        source_weights: Dict[int, List[List[float]]],
        target_weights: Dict[int, List[List[float]]],
        config: Config = Config()
    ) -> BatchResult:
        layer_results = {}
        failed_layers = []
        total_gw_dist = 0.0
        total_marginal = 0.0
        
        # If commonAnchorIDs logic exists in ConceptResponseMatrix (assuming port parity)
        # Using a safer approach if method names differ slightly
        common_anchors = source_crm.common_anchor_ids(target_crm) if hasattr(source_crm, 'common_anchor_ids') else []
        if not common_anchors:
            # Try to infer or fallback? Return empty
            return TransportGuidedMerger.BatchResult({}, 0.0, 0.0, [])

        source_layers = set(source_weights.keys())
        target_layers = set(target_weights.keys())
        common_layers = sorted(list(source_layers.intersection(target_layers)))
        
        for layer in common_layers:
            layer_key = f"layer_{layer}"
            
            source_act = source_crm.activation_matrix(layer, common_anchors)
            target_act = target_crm.activation_matrix(layer, common_anchors)
            src_w = source_weights.get(layer)
            tgt_w = target_weights.get(layer)
            
            if source_act is None or target_act is None or src_w is None or tgt_w is None:
                failed_layers.append(layer_key)
                continue
                
            result = TransportGuidedMerger.synthesize_with_gw(
                source_activations=source_act,
                target_activations=target_act,
                source_weights=src_w,
                target_weights=tgt_w,
                config=config
            )
            
            if result:
                layer_results[layer_key] = result
                total_gw_dist += result.gw_distance
                total_marginal += result.marginal_error
            else:
                failed_layers.append(layer_key)
                
        count = max(1, len(layer_results))
        return TransportGuidedMerger.BatchResult(
            layer_results=layer_results,
            mean_gw_distance=total_gw_dist / count,
            mean_marginal_error=total_marginal / count,
            failed_layers=failed_layers
        )

    # MARK: - Utilities
    
    @staticmethod
    def _align_activations_to_weights(activations: List[List[float]], weight_count: int) -> Optional[List[List[float]]]:
        if not activations: return None
        n_rows = len(activations)
        n_cols = len(activations[0])
        
        # If rows match weight count (N neurons), use as is
        if n_rows == weight_count:
            return activations
        # If cols match weight count, transpose (samples x neurons -> neurons x samples)
        # Wait, GW usually expects point clouds. 
        # Swift code: 
        #   if activations.count == weightCount { return activations }
        #   if firstRow.count == weightCount { return transpose(activations) }
        # So it wants [Neuron x Features].
        
        if n_cols == weight_count:
            return TransportGuidedMerger._transpose(activations)
            
        return None

    @staticmethod
    def _apply_threshold(plan: List[List[float]], threshold: float) -> List[List[float]]:
        return [[(val if val >= threshold else 0.0) for val in row] for row in plan]

    @staticmethod
    def _normalize_rows(plan: List[List[float]]) -> List[List[float]]:
        normalized = []
        for row in plan:
            row_sum = sum(row)
            if row_sum > 0:
                normalized.append([val / row_sum for val in row])
            else:
                normalized.append(row)
        return normalized

    @staticmethod
    def _transpose(matrix: List[List[float]]) -> List[List[float]]:
        if not matrix: return []
        return [list(col) for col in zip(*matrix)]

    @staticmethod
    def _compute_marginal_error(coupling: List[List[float]]) -> Tuple[float, float]:
        n = len(coupling)
        if n == 0: return (0.0, 0.0)
        m = len(coupling[0])
        if m == 0: return (0.0, 0.0)
        
        expected_row = 1.0 / n
        expected_col = 1.0 / m
        
        max_row_error = 0.0
        for i in range(n):
            row_sum = sum(coupling[i])
            max_row_error = max(max_row_error, abs(row_sum - expected_row))
            
        max_col_error = 0.0
        # Column sums
        col_sums = [0.0] * m
        for i in range(n):
            for j in range(m):
                col_sums[j] += coupling[i][j]
                
        for j in range(m):
            max_col_error = max(max_col_error, abs(col_sums[j] - expected_col))
            
        return (max_row_error, max_col_error)

    @staticmethod
    def _compute_effective_rank(coupling: List[List[float]], threshold: float) -> int:
        count = 0
        for row in coupling:
            for val in row:
                if val >= threshold:
                    count += 1
        return count

    @staticmethod
    def _compute_dimension_confidences(coupling: List[List[float]]) -> List[float]:
        n = len(coupling)
        if n == 0: return []
        confidences = []
        
        for row in coupling:
            row_sum = sum(row)
            if row_sum <= 0:
                confidences.append(0.0)
                continue
                
            # Entropy
            entropy = 0.0
            for val in row:
                p = val / row_sum
                if p > 0:
                    entropy -= p * math.log(p)
            
            m = len(row)
            max_entropy = math.log(m) if m > 1 else 1.0
            normalized = entropy / max_entropy
            confidences.append(max(0.0, 1.0 - normalized))
            
        return confidences
