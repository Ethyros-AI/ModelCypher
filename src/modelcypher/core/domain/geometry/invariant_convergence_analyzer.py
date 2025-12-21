"""
Invariant Convergence Analyzer.

Tracks the stability of semantic invariants (logic vs knowledge sequences) 
across training steps or model merges.

Ported from InvariantConvergenceAnalyzer.swift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import mlx.core as mx

from .manifold_stitcher import ManifoldStitcher, ContinuousModelFingerprints

@dataclass
class ConvergenceMetric:
    sequence_family: str
    step: int
    alignment_score: float
    variance: float
    is_converged: bool

@dataclass
class ConvergenceReport:
    model_id: str
    metrics: List[ConvergenceMetric]
    overall_convergence: float
    stable_families: List[str]

class InvariantConvergenceAnalyzer:
    """
    Analyzes the convergence of geometric invariants.
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "logic": 0.85,    # Logic requires high stability
            "knowledge": 0.6, # Knowledge allows more drift
            "creative": 0.4   # Creative allowing drift
        }

    def analyze_convergence(
        self,
        baseline: ContinuousModelFingerprints,
        current: ContinuousModelFingerprints,
        step: int,
        sequence_families: Dict[str, List[str]] # Family -> [PrimeIDs]
    ) -> ConvergenceReport:
        """
        Compare current fingerprints to baseline per sequence family.
        """
        metrics = []
        stable_families = []
        
        # Check per family
        for family, prime_ids in sequence_families.items():
            scores = []
            
            # Filter fingerprints by family
            base_fps = {fp.prime_id: fp for fp in baseline.fingerprints if fp.prime_id in prime_ids}
            curr_fps = {fp.prime_id: fp for fp in current.fingerprints if fp.prime_id in prime_ids}
            
            common_ids = set(base_fps.keys()) & set(curr_fps.keys())
            
            for pid in common_ids:
                # Compute correlation for each shared layer
                # We average across layers for a single scalar score per prime
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
            variance = sum((s - mean_score)**2 for s in scores) / len(scores)
            
            threshold = self.thresholds.get(family, 0.7)
            is_converged = mean_score >= threshold
            
            if is_converged:
                stable_families.append(family)
                
            metrics.append(ConvergenceMetric(
                sequence_family=family,
                step=step,
                alignment_score=mean_score,
                variance=variance,
                is_converged=is_converged
            ))
            
        overall = sum(m.alignment_score for m in metrics) / len(metrics) if metrics else 0.0
        
        return ConvergenceReport(
            model_id=current.model_id,
            metrics=metrics,
            overall_convergence=overall,
            stable_families=stable_families
        )
