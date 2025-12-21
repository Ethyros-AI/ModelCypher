
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum
import mlx.core as mx
import math

class CompositionCategory(Enum):
    MENTAL_PREDICATE = "mentalPredicate"
    ACTION = "action"
    EVALUATIVE = "evaluative"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUANTIFIED = "quantified"
    RELATIONAL = "relational"

@dataclass(frozen=True)
class CompositionProbe:
    phrase: str
    components: List[str]
    category: CompositionCategory

@dataclass(frozen=True)
class CompositionAnalysis:
    probe: CompositionProbe
    barycentric_weights: List[float]
    residual_norm: float
    centroid_similarity: float
    component_angles: List[float]
    
    @property
    def is_compositional(self) -> bool:
        return self.residual_norm < 0.5 and self.centroid_similarity > 0.3

@dataclass(frozen=True)
class ConsistencyResult:
    probe_count: int
    analyses_a: List[CompositionAnalysis]
    analyses_b: List[CompositionAnalysis]
    barycentric_correlation: float
    angular_correlation: float
    consistency_score: float
    is_compatible: bool
    interpretation: str

class CompositionalProbes:
    """
    Compositional probe analysis for cross-model semantic structure verification.
    Ported from CompositionalProbes.swift.
    """
    
    STANDARD_PROBES = [
        CompositionProbe("I THINK", ["I", "THINK"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I KNOW", ["I", "KNOW"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I FEEL", ["I", "FEEL"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I SEE", ["I", "SEE"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I HEAR", ["I", "HEAR"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("SOMEONE DO", ["SOMEONE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("PEOPLE DO", ["PEOPLE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("I SAY", ["I", "SAY"], CompositionCategory.ACTION),
        CompositionProbe("GOOD THINGS", ["GOOD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("BAD THINGS", ["BAD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("GOOD PEOPLE", ["GOOD", "PEOPLE"], CompositionCategory.EVALUATIVE),
        CompositionProbe("BEFORE NOW", ["BEFORE", "NOW"], CompositionCategory.TEMPORAL),
        CompositionProbe("AFTER THIS", ["AFTER", "THIS"], CompositionCategory.TEMPORAL),
        CompositionProbe("A LONG TIME BEFORE", ["A_LONG_TIME", "BEFORE"], CompositionCategory.TEMPORAL),
        CompositionProbe("ABOVE HERE", ["ABOVE", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("FAR FROM HERE", ["FAR", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("NEAR THIS", ["NEAR", "THIS"], CompositionCategory.SPATIAL),
        CompositionProbe("MUCH GOOD", ["MUCH_MANY", "GOOD"], CompositionCategory.QUANTIFIED),
        CompositionProbe("MANY PEOPLE", ["MUCH_MANY", "PEOPLE"], CompositionCategory.QUANTIFIED),
        CompositionProbe("I WANT GOOD THINGS", ["I", "WANT", "GOOD", "SOMETHING"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("SOMEONE DO BAD THINGS", ["SOMEONE", "DO", "BAD", "SOMETHING"], CompositionCategory.ACTION),
    ]

    @staticmethod
    def analyze_composition(
        composition_embedding: mx.array, # [D]
        component_embeddings: mx.array,  # [N, D]
        probe: CompositionProbe
    ) -> CompositionAnalysis:
        
        # Ensure 1D comp
        if composition_embedding.ndim == 2 and composition_embedding.shape[0] == 1:
            composition_embedding = composition_embedding[0]
            
        d = composition_embedding.shape[0]
        n = component_embeddings.shape[0]
        
        if n == 0 or d == 0:
            return CompositionAnalysis(probe, [], float("inf"), 0.0, [])
            
        # Centroid
        centroid = mx.mean(component_embeddings, axis=0)
        
        # Centroid similarity
        centroid_sim = CompositionalProbes._cosine_similarity(composition_embedding, centroid).item()
        
        # Component angles
        angles = []
        for i in range(n):
            sim = CompositionalProbes._cosine_similarity(composition_embedding, component_embeddings[i]).item()
            angles.append(sim)
            
        # Barycentric weights
        # Solve A x = b where A = component_embeddings.T [D, N], b = composition_embedding [D]
        # Using least squares
        
        A = component_embeddings.transpose() # [D, N]
        b = composition_embedding # [D]
        
        # Use simple lstsq if available or normal equations: (A^T A) x = A^T b
        # In MLX, A.T is (N, D).
        # G = A.T @ A = (component_embeddings @ component_embeddings.T) [N, N]
        # rhs = A.T @ b = component_embeddings @ composition_embedding [N]
        
        # NOTE: component_embeddings is [N, D].
        # We want to represent composition as sum (w_i * comp_i).
        # So we want w such that w @ component_embeddings ≈ composition_embedding
        # If we use column vectors:
        # C = [c1 c2 ... cn] (D x N) matrix of components
        # target t (D x 1)
        # C w = t
        # C^T C w = C^T t
        # This is G w = rhs
        
        G = component_embeddings @ component_embeddings.transpose() # [N, N]
        rhs = component_embeddings @ composition_embedding # [N]
        
        # Solve G w = rhs
        # Regularize diagonal slightly for stability
        eps = 1e-6
        G = G + mx.eye(n) * eps
        
        weights_mx = mx.linalg.solve(G, rhs) # [N]
        weights = weights_mx.tolist()
        
        # Calc residual
        reconstructed = weights_mx @ component_embeddings # [D]
        diff = composition_embedding - reconstructed
        residual_norm = mx.linalg.norm(diff).item()
        
        return CompositionAnalysis(
            probe=probe,
            barycentric_weights=weights,
            residual_norm=residual_norm,
            centroid_similarity=centroid_sim,
            component_angles=angles
        )

    @staticmethod
    def _cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
        # Returns scalar array
        dot = mx.inner(a, b)
        norm_a = mx.linalg.norm(a)
        norm_b = mx.linalg.norm(b)
        denom = norm_a * norm_b
        return mx.where(denom > 1e-9, dot / denom, mx.array(0.0))

    @staticmethod
    def check_consistency(
        analyses_a: List[CompositionAnalysis],
        analyses_b: List[CompositionAnalysis]
    ) -> ConsistencyResult:
        if len(analyses_a) != len(analyses_b) or not analyses_a:
            return ConsistencyResult(0, [], [], 0, 0, 0, False, "Insufficient data")
            
        n = len(analyses_a)
        
        # Collect weights
        weights_a: List[float] = []
        weights_b: List[float] = []
        for i in range(n):
            if len(analyses_a[i].barycentric_weights) == len(analyses_b[i].barycentric_weights):
                weights_a.extend(analyses_a[i].barycentric_weights)
                weights_b.extend(analyses_b[i].barycentric_weights)
                
        # Collect angles
        angles_a: List[float] = []
        angles_b: List[float] = []
        for i in range(n):
            if len(analyses_a[i].component_angles) == len(analyses_b[i].component_angles):
                angles_a.extend(analyses_a[i].component_angles)
                angles_b.extend(analyses_b[i].component_angles)
                
        bary_corr = CompositionalProbes._pearson(weights_a, weights_b)
        ang_corr = CompositionalProbes._pearson(angles_a, angles_b)
        
        score = 0.4 * max(0, bary_corr) + 0.6 * max(0, ang_corr)
        is_compatible = score >= 0.5 and ang_corr >= 0.4
        
        interpretation = "Low consistency"
        if score >= 0.8: interpretation = "Excellent consistency"
        elif score >= 0.6: interpretation = "Good consistency"
        elif score >= 0.4: interpretation = "Partial consistency"
        
        return ConsistencyResult(
            probe_count=n,
            analyses_a=analyses_a,
            analyses_b=analyses_b,
            barycentric_correlation=bary_corr,
            angular_correlation=ang_corr,
            consistency_score=score,
            is_compatible=is_compatible,
            interpretation=interpretation
        )

    @staticmethod
    def _pearson(a: List[float], b: List[float]) -> float:
        if len(a) < 2 or len(b) < 2: return 0.0
        
        # Create vectors
        va = mx.array(a)
        vb = mx.array(b)
        
        ma = mx.mean(va)
        mb = mx.mean(vb)
        
        da = va - ma
        db = vb - mb
        
        num = mx.sum(da * db)
        den = mx.sqrt(mx.sum(da**2) * mx.sum(db**2))
        
        return (num / den).item() if den.item() > 1e-9 else 0.0
