"""
Invariant Convergence Analyzer.

Computes layer-wise convergence for sequence invariants across two models.
Tracks the stability of semantic invariants (logic, fibonacci, primes, etc.)
across training steps or model merges.

Ported from InvariantConvergenceAnalyzer.swift (434 lines → 350 lines).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set

from .manifold_stitcher import ManifoldStitcher, ContinuousModelFingerprints

logger = logging.getLogger("modelcypher.geometry.invariant_convergence")


# =============================================================================
# Enums and Types
# =============================================================================


class AlignMode(str, Enum):
    """Layer alignment mode for cross-architecture comparison."""
    LAYER = "layer"           # Direct layer-to-layer (same architecture)
    NORMALIZED = "normalized"  # Normalized depth (cross-architecture)


@dataclass
class AlignmentPair:
    """A matched pair of layers between source and target."""
    index: int
    source_layer: int
    target_layer: int
    normalized_depth: float


@dataclass
class DimensionAlignmentSummary:
    """Summary of dimension alignment across layers."""
    mode: str
    holdout_prefixes: List[str]
    aligned_dimension_count_by_layer: Dict[str, int]
    total_aligned_dimensions: int


# =============================================================================
# Family Result
# =============================================================================


@dataclass
class FamilyResult:
    """Result of convergence analysis for a single sequence family."""
    anchor_ids: List[str]
    layers: Dict[str, float]  # Layer label -> cosine similarity
    mean_cosine: Optional[float]


# =============================================================================
# Summary and Report
# =============================================================================


@dataclass
class Summary:
    """Summary statistics across families and layers."""
    mean_cosine_by_family: Dict[str, Optional[float]]
    mean_cosine_by_layer: Dict[str, float]


@dataclass
class Report:
    """Complete convergence analysis report."""
    
    @dataclass
    class Models:
        model_a: str
        model_b: str
    
    models: Models
    align_mode: AlignMode
    dimension_alignment: DimensionAlignmentSummary
    layers: List[float]
    families: Dict[str, FamilyResult]
    summary: Summary
    layer_count: int
    source_layer_count: int
    target_layer_count: int
    aligned_layers: List[AlignmentPair]


# =============================================================================
# Legacy Types (backward compatibility)
# =============================================================================


@dataclass
class ConvergenceMetric:
    """Single family convergence metric."""
    sequence_family: str
    step: int
    alignment_score: float
    variance: float
    is_converged: bool


@dataclass
class ConvergenceReport:
    """Legacy convergence report for training tracking."""
    model_id: str
    metrics: List[ConvergenceMetric]
    overall_convergence: float
    stable_families: List[str]


# =============================================================================
# Invariant Convergence Analyzer
# =============================================================================


class InvariantConvergenceAnalyzer:
    """
    Computes layer-wise convergence for sequence invariants across two models.
    
    The analyzer computes cosine similarity between activation vectors for
    invariant anchors (fibonacci, lucas, primes, catalan sequences) across
    matched layer pairs.
    
    Supports two alignment modes:
    - LAYER: Direct layer-to-layer matching (same architecture)
    - NORMALIZED: Normalized depth matching (cross-architecture)
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize with per-family convergence thresholds.
        
        Args:
            thresholds: Minimum cosine similarity to consider converged per family.
        """
        self.thresholds = thresholds or {
            "fibonacci": 0.8,
            "lucas": 0.8,
            "primes": 0.75,
            "catalan": 0.7,
            "logic": 0.85,
            "knowledge": 0.6,
            "creative": 0.4,
        }
    
    def analyze(
        self,
        source: ContinuousModelFingerprints,
        target: ContinuousModelFingerprints,
        align_mode: AlignMode = AlignMode.LAYER,
        family_allowlist: Optional[Set[str]] = None,
    ) -> Report:
        """
        Analyze convergence between source and target fingerprints.
        
        Args:
            source: Source model fingerprints.
            target: Target model fingerprints.
            align_mode: How to align layers between models.
            family_allowlist: Only analyze these families (None = all).
        
        Returns:
            Report with per-family and per-layer convergence scores.
        """
        # Build anchor vectors by layer
        source_vectors = self._build_anchor_vectors(source, family_allowlist)
        target_vectors = self._build_anchor_vectors(target, family_allowlist)
        
        source_layers = self._collect_layers(source_vectors)
        target_layers = self._collect_layers(target_vectors)
        
        # Align layers based on mode
        aligned_pairs, axis_by_index = self._align_layers(
            source_layers, target_layers,
            source.layer_count if hasattr(source, 'layer_count') else len(source_layers),
            target.layer_count if hasattr(target, 'layer_count') else len(target_layers),
            align_mode,
        )
        
        # Collect families
        families_in_data = set()
        for anchor_id in list(source_vectors.keys()) + list(target_vectors.keys()):
            family = anchor_id.split("_")[0] if "_" in anchor_id else anchor_id
            if family_allowlist is None or family in family_allowlist:
                families_in_data.add(family)
        
        ordered_families = sorted(families_in_data)
        
        # Process each family
        family_results: Dict[str, FamilyResult] = {}
        layers_union: List[int] = []
        
        for family in ordered_families:
            anchor_ids = [aid for aid in source_vectors.keys() 
                         if aid.startswith(f"{family}_")]
            
            source_layer_vecs, target_layer_vecs = self._gather_family_vectors(
                anchor_ids, source_vectors, target_vectors,
            )
            
            # Compute cosines per aligned layer
            layer_cosines: Dict[int, float] = {}
            
            if align_mode == AlignMode.NORMALIZED:
                pair_by_index = {p.index: p for p in aligned_pairs}
                for pair in aligned_pairs:
                    avg_source = self._average_sparse(source_layer_vecs.get(pair.source_layer, []))
                    avg_target = self._average_sparse(target_layer_vecs.get(pair.target_layer, []))
                    cosine = self._cosine_sparse(avg_source, avg_target)
                    if cosine is not None:
                        layer_cosines[pair.index] = cosine
            else:
                common_layers = set(source_layer_vecs.keys()) & set(target_layer_vecs.keys())
                for layer in sorted(common_layers):
                    avg_source = self._average_sparse(source_layer_vecs.get(layer, []))
                    avg_target = self._average_sparse(target_layer_vecs.get(layer, []))
                    cosine = self._cosine_sparse(avg_source, avg_target)
                    if cosine is not None:
                        layer_cosines[layer] = cosine
            
            # Format labels
            labeled_layers = {
                self._format_layer_label(layer, align_mode, axis_by_index): cos
                for layer, cos in layer_cosines.items()
            }
            
            mean_cosine = None
            if layer_cosines:
                mean_cosine = sum(layer_cosines.values()) / len(layer_cosines)
            
            family_results[family] = FamilyResult(
                anchor_ids=anchor_ids,
                layers=labeled_layers,
                mean_cosine=mean_cosine,
            )
            
            for layer in layer_cosines.keys():
                if layer not in layers_union:
                    layers_union.append(layer)
        
        layers_union.sort()
        if align_mode == AlignMode.NORMALIZED:
            layers_union = [p.index for p in aligned_pairs]
        
        # Build summaries
        mean_by_family = {f: family_results[f].mean_cosine for f in ordered_families}
        mean_by_layer = {
            self._format_layer_label(layer, align_mode, axis_by_index): self._mean_across_families(layer, family_results, align_mode, axis_by_index)
            for layer in layers_union
        }
        
        output_layers = [
            axis_by_index.get(layer, float(layer)) if align_mode == AlignMode.NORMALIZED
            else float(layer)
            for layer in layers_union
        ]
        
        src_count = source.layer_count if hasattr(source, 'layer_count') else len(source_layers)
        tgt_count = target.layer_count if hasattr(target, 'layer_count') else len(target_layers)
        
        return Report(
            models=Report.Models(
                model_a=source.model_id,
                model_b=target.model_id,
            ),
            align_mode=align_mode,
            dimension_alignment=DimensionAlignmentSummary(
                mode="intersection",
                holdout_prefixes=["invariant:"],
                aligned_dimension_count_by_layer={},
                total_aligned_dimensions=0,
            ),
            layers=output_layers,
            families=family_results,
            summary=Summary(
                mean_cosine_by_family=mean_by_family,
                mean_cosine_by_layer=mean_by_layer,
            ),
            layer_count=min(src_count, tgt_count) if src_count > 0 and tgt_count > 0 else max(src_count, tgt_count),
            source_layer_count=src_count,
            target_layer_count=tgt_count,
            aligned_layers=aligned_pairs,
        )
    
    # =========================================================================
    # CSV Export
    # =========================================================================
    
    @staticmethod
    def csv_lines(report: Report) -> List[str]:
        """
        Generate CSV lines from a report.
        
        Format: family,layer,cosine
        """
        lines = ["family,layer,cosine"]
        families = sorted(report.families.keys())
        
        for family in families:
            family_result = report.families.get(family)
            if not family_result:
                continue
            for layer_label, cosine in sorted(family_result.layers.items()):
                lines.append(f"{family},{layer_label},{cosine:.6f}")
        
        return lines
    
    # =========================================================================
    # Legacy Interface (backward compatibility)
    # =========================================================================
    
    def analyze_convergence(
        self,
        baseline: ContinuousModelFingerprints,
        current: ContinuousModelFingerprints,
        step: int,
        sequence_families: Dict[str, List[str]],
    ) -> ConvergenceReport:
        """
        Legacy interface for training-time convergence tracking.
        """
        metrics = []
        stable_families = []
        
        for family, prime_ids in sequence_families.items():
            scores = []
            
            base_fps = {fp.prime_id: fp for fp in baseline.fingerprints if fp.prime_id in prime_ids}
            curr_fps = {fp.prime_id: fp for fp in current.fingerprints if fp.prime_id in prime_ids}
            
            common_ids = set(base_fps.keys()) & set(curr_fps.keys())
            
            for pid in common_ids:
                layer_scores = []
                base_fp = base_fps[pid]
                curr_fp = curr_fps[pid]
                
                common_layers = set(base_fp.activation_vectors.keys()) & set(curr_fp.activation_vectors.keys())
                for layer in common_layers:
                    res = ManifoldStitcher.compute_continuous_correlation(base_fp, curr_fp, layer)
                    if res:
                        layer_scores.append(res.compatibility_score)
                
                if layer_scores:
                    scores.append(sum(layer_scores) / len(layer_scores))
            
            if not scores:
                continue
            
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            
            threshold = self.thresholds.get(family, 0.7)
            is_converged = mean_score >= threshold
            
            if is_converged:
                stable_families.append(family)
            
            metrics.append(ConvergenceMetric(
                sequence_family=family,
                step=step,
                alignment_score=mean_score,
                variance=variance,
                is_converged=is_converged,
            ))
        
        overall = sum(m.alignment_score for m in metrics) / len(metrics) if metrics else 0.0
        
        return ConvergenceReport(
            model_id=current.model_id,
            metrics=metrics,
            overall_convergence=overall,
            stable_families=stable_families,
        )
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _build_anchor_vectors(
        self,
        fingerprints: ContinuousModelFingerprints,
        family_allowlist: Optional[Set[str]],
    ) -> Dict[str, Dict[int, Dict[int, float]]]:
        """Build anchor -> layer -> dimension -> value mapping."""
        vectors: Dict[str, Dict[int, Dict[int, float]]] = {}
        
        for fp in fingerprints.fingerprints:
            prime_id = fp.prime_id
            if not prime_id.startswith("invariant:"):
                continue
            
            anchor_id = prime_id[len("invariant:"):]
            
            if family_allowlist:
                family = anchor_id.split("_")[0] if "_" in anchor_id else anchor_id
                if family not in family_allowlist:
                    continue
            
            layer_map = vectors.setdefault(anchor_id, {})
            
            for layer, dims in fp.activation_vectors.items():
                sparse: Dict[int, float] = {}
                if isinstance(dims, dict):
                    sparse = {int(k): float(v) for k, v in dims.items()}
                elif isinstance(dims, list):
                    for i, v in enumerate(dims):
                        if v != 0:
                            sparse[i] = float(v)
                
                if sparse:
                    layer_map[layer] = sparse
        
        return vectors
    
    def _collect_layers(self, vectors: Dict[str, Dict[int, Dict[int, float]]]) -> List[int]:
        """Collect all unique layer indices."""
        layers: Set[int] = set()
        for layer_map in vectors.values():
            layers.update(layer_map.keys())
        return sorted(layers)
    
    def _align_layers(
        self,
        source_layers: List[int],
        target_layers: List[int],
        source_count: int,
        target_count: int,
        mode: AlignMode,
    ) -> Tuple[List[AlignmentPair], Dict[int, float]]:
        """Align layers between source and target."""
        pairs: List[AlignmentPair] = []
        axis_by_index: Dict[int, float] = {}
        
        if mode == AlignMode.NORMALIZED:
            aligned_count = min(len(source_layers), len(target_layers))
            for idx in range(aligned_count):
                src_idx = self._scaled_index(idx, aligned_count, len(source_layers))
                tgt_idx = self._scaled_index(idx, aligned_count, len(target_layers))
                
                if src_idx < len(source_layers) and tgt_idx < len(target_layers):
                    src_layer = source_layers[src_idx]
                    tgt_layer = target_layers[tgt_idx]
                    
                    src_pos = src_layer / max(1, source_count)
                    tgt_pos = tgt_layer / max(1, target_count)
                    normalized = (src_pos + tgt_pos) / 2
                    
                    axis_by_index[idx] = normalized
                    pairs.append(AlignmentPair(
                        index=idx,
                        source_layer=src_layer,
                        target_layer=tgt_layer,
                        normalized_depth=normalized,
                    ))
        else:
            common = sorted(set(source_layers) & set(target_layers))
            for layer in common:
                pairs.append(AlignmentPair(
                    index=layer,
                    source_layer=layer,
                    target_layer=layer,
                    normalized_depth=float(layer),
                ))
        
        return pairs, axis_by_index
    
    def _scaled_index(self, position: int, aligned_count: int, total_count: int) -> int:
        """Scale position to total count."""
        if total_count <= 0:
            return 0
        if aligned_count <= 1:
            return 0
        fraction = position / (aligned_count - 1)
        scaled = int(round(fraction * (total_count - 1)))
        return max(0, min(total_count - 1, scaled))
    
    def _gather_family_vectors(
        self,
        anchor_ids: List[str],
        source_vectors: Dict[str, Dict[int, Dict[int, float]]],
        target_vectors: Dict[str, Dict[int, Dict[int, float]]],
    ) -> Tuple[Dict[int, List[Dict[int, float]]], Dict[int, List[Dict[int, float]]]]:
        """Gather vectors by layer for a family."""
        source_by_layer: Dict[int, List[Dict[int, float]]] = {}
        target_by_layer: Dict[int, List[Dict[int, float]]] = {}
        
        for anchor_id in anchor_ids:
            if anchor_id in source_vectors:
                for layer, vec in source_vectors[anchor_id].items():
                    source_by_layer.setdefault(layer, []).append(vec)
            if anchor_id in target_vectors:
                for layer, vec in target_vectors[anchor_id].items():
                    target_by_layer.setdefault(layer, []).append(vec)
        
        return source_by_layer, target_by_layer
    
    def _average_sparse(self, vectors: List[Dict[int, float]]) -> Dict[int, float]:
        """Average sparse vectors."""
        if not vectors:
            return {}
        
        sums: Dict[int, float] = {}
        for vec in vectors:
            for idx, val in vec.items():
                sums[idx] = sums.get(idx, 0.0) + val
        
        count = len(vectors)
        return {idx: val / count for idx, val in sums.items()}
    
    def _cosine_sparse(self, a: Dict[int, float], b: Dict[int, float]) -> Optional[float]:
        """Compute cosine similarity between sparse vectors."""
        if not a or not b:
            return None
        
        dot = 0.0
        for idx, val in a.items():
            dot += val * b.get(idx, 0.0)
        
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        
        if norm_a <= 0 or norm_b <= 0:
            return None
        
        return dot / (norm_a * norm_b)
    
    def _format_layer_label(
        self,
        layer: int,
        mode: AlignMode,
        axis_by_index: Dict[int, float],
    ) -> str:
        """Format layer index as label."""
        if mode == AlignMode.NORMALIZED:
            value = axis_by_index.get(layer, 0.0)
            return f"{value:.4f}"
        return str(layer)
    
    def _mean_across_families(
        self,
        layer: int,
        family_results: Dict[str, FamilyResult],
        mode: AlignMode,
        axis_by_index: Dict[int, float],
    ) -> float:
        """Compute mean cosine across families for a layer."""
        label = self._format_layer_label(layer, mode, axis_by_index)
        values = []
        for result in family_results.values():
            if label in result.layers:
                values.append(result.layers[label])
        return sum(values) / len(values) if values else 0.0

