from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
import math
import mlx.core as mx

class CompositionCategory(str, Enum):
    MENTAL_PREDICATE = "mentalPredicate"
    ACTION = "action"
    EVALUATIVE = "evaluative"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUANTIFIED = "quantified"
    RELATIONAL = "relational"

@dataclass
class CompositionProbe:
    """
    A compositional probe: a phrase and its component primes.
    """
    phrase: str
    components: List[str]
    category: CompositionCategory

@dataclass
class CompositionAnalysis:
    """
    Result of compositional structure analysis.
    """
    probe: CompositionProbe
    barycentric_weights: List[float]
    residual_norm: float
    centroid_similarity: float
    component_angles: List[float]
    
    @property
    def is_compositional(self) -> bool:
        return self.residual_norm < 0.5 and self.centroid_similarity > 0.3

@dataclass
class ConsistencyResult:
    """
    Result of cross-model compositional consistency check.
    """
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
    """
    
    STANDARD_PROBES = [
        # Mental predicates
        CompositionProbe("I THINK", ["I", "THINK"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I KNOW", ["I", "KNOW"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I FEEL", ["I", "FEEL"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I SEE", ["I", "SEE"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("I HEAR", ["I", "HEAR"], CompositionCategory.MENTAL_PREDICATE),
        
        # Actions
        CompositionProbe("SOMEONE DO", ["SOMEONE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("PEOPLE DO", ["PEOPLE", "DO"], CompositionCategory.ACTION),
        CompositionProbe("I SAY", ["I", "SAY"], CompositionCategory.ACTION),
        
        # Evaluatives
        CompositionProbe("GOOD THINGS", ["GOOD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("BAD THINGS", ["BAD", "SOMETHING"], CompositionCategory.EVALUATIVE),
        CompositionProbe("GOOD PEOPLE", ["GOOD", "PEOPLE"], CompositionCategory.EVALUATIVE),
        
        # Temporal
        CompositionProbe("BEFORE NOW", ["BEFORE", "NOW"], CompositionCategory.TEMPORAL),
        CompositionProbe("AFTER THIS", ["AFTER", "THIS"], CompositionCategory.TEMPORAL),
        CompositionProbe("A LONG TIME BEFORE", ["A_LONG_TIME", "BEFORE"], CompositionCategory.TEMPORAL),
        
        # Spatial
        CompositionProbe("ABOVE HERE", ["ABOVE", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("FAR FROM HERE", ["FAR", "HERE"], CompositionCategory.SPATIAL),
        CompositionProbe("NEAR THIS", ["NEAR", "THIS"], CompositionCategory.SPATIAL),
        
        # Quantified
        CompositionProbe("MUCH GOOD", ["MUCH_MANY", "GOOD"], CompositionCategory.QUANTIFIED),
        CompositionProbe("MANY PEOPLE", ["MUCH_MANY", "PEOPLE"], CompositionCategory.QUANTIFIED),
        
        # Complex
        CompositionProbe("I WANT GOOD THINGS", ["I", "WANT", "GOOD", "SOMETHING"], CompositionCategory.MENTAL_PREDICATE),
        CompositionProbe("SOMEONE DO BAD THINGS", ["SOMEONE", "DO", "BAD", "SOMETHING"], CompositionCategory.ACTION),
    ]

    @staticmethod
    def analyze_composition(
        composition_embedding: List[float],
        component_embeddings: List[List[float]],
        probe: CompositionProbe
    ) -> CompositionAnalysis:
        n = len(component_embeddings)
        d = len(composition_embedding)
        
        if n == 0 or d == 0:
            return CompositionAnalysis(probe, [], float('inf'), 0.0, [])
            
        # Vectors using MLX for speed
        target = mx.array(composition_embedding)
        basis = mx.array(component_embeddings)
        
        # Centroid
        centroid = mx.mean(basis, axis=0)
        
        # Centroid similarity
        centroid_sim = CompositionalProbes.cosine_similarity(target, centroid)
        
        # Component angles
        angles = []
        for i in range(n):
            sim = CompositionalProbes.cosine_similarity(target, basis[i])
            angles.append(sim)
            
        # Barycentric weights
        weights, residual = CompositionalProbes.compute_barycentric_weights(target, basis)
        
        return CompositionAnalysis(
            probe=probe,
            barycentric_weights=weights,
            residual_norm=residual,
            centroid_similarity=centroid_sim,
            component_angles=angles
        )

    @staticmethod
    def compute_barycentric_weights(target: mx.array, basis: mx.array) -> Tuple[List[float], float]:
        # Solve A.T A x = A.T b
        # A = basis.T (d x n matrix, columns are basis vectors)
        # But here basis is (n, d).
        # So A.T @ A = basis @ basis.T ?? No.
        # Least squares: min || B.T w - t || where B is (d, n) matrix of basis vectors as cols.
        # Or here basis is (n, d) -> rows are basis vectors.
        # composition approx linear combo of rows: target = w1*b1 + w2*b2... = w @ basis
        # target (d,)  basis (n, d)  weights (n,)
        # target = weights @ basis
        # Transpose: target.T = basis.T @ weights.T
        # B = basis.T (d, n). y = target.T (d, 1). x = weights (n, 1)
        # min || Bx - y ||
        # Normal eq: B.T B x = B.T y
        
        B = basis.T # (d, n)
        Gram = B.T @ B # (n, n)
        RHS = B.T @ target # (n,)
        
        # Check det/invertibility. Use pseudo-inverse approach or solve with jitter.
        # n is small (2-4), so inversion is cheap.
        # Add simpler jitter for stability
        Gram_reg = Gram + mx.eye(Gram.shape[0]) * 1e-6
        
        try:
            # Manually invert since solve might not be exposed or robust?
            # mx.linalg.inv is standard.
            Gram_inv = mx.linalg.inv(Gram_reg)
            weights_vec = Gram_inv @ RHS
        except Exception:
            # Fallback to equal weights
            weights_vec = mx.ones((basis.shape[0],)) / basis.shape[0]
            
        weights = weights_vec.tolist()
        
        # Residual
        reconstructed = weights_vec @ basis # (n,) @ (n, d) -> (d,)
        diff = target - reconstructed
        residual = float(mx.linalg.norm(diff).item())
        
        return (weights, residual)

    @staticmethod
    def cosine_similarity(a: mx.array, b: mx.array) -> float:
        dot = mx.dot(a, b).item()
        norm_a = mx.linalg.norm(a).item()
        norm_b = mx.linalg.norm(b).item()
        if norm_a < 1e-9 or norm_b < 1e-9: return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def check_consistency(
        analyses_a: List[CompositionAnalysis],
        analyses_b: List[CompositionAnalysis]
    ) -> ConsistencyResult:
        if len(analyses_a) != len(analyses_b) or not analyses_a:
            return ConsistencyResult(0, [], [], 0.0, 0.0, 0.0, False, "Insufficient data")
            
        n = len(analyses_a)
        
        all_weights_a = []
        all_weights_b = []
        for i in range(n):
            all_weights_a.extend(analyses_a[i].barycentric_weights)
            all_weights_b.extend(analyses_b[i].barycentric_weights)
            
        all_angles_a = []
        all_angles_b = []
        for i in range(n):
            all_angles_a.extend(analyses_a[i].component_angles)
            all_angles_b.extend(analyses_b[i].component_angles)
            
        bary_corr = CompositionalProbes.pearson_correlation(all_weights_a, all_weights_b)
        ang_corr = CompositionalProbes.pearson_correlation(all_angles_a, all_angles_b)
        
        score = 0.4 * max(0.0, bary_corr) + 0.6 * max(0.0, ang_corr)
        is_compatible = score >= 0.5 and ang_corr >= 0.4
        
        if score >= 0.8: interp = "Excellent compositional consistency."
        elif score >= 0.6: interp = "Good compositional consistency."
        elif score >= 0.4: interp = "Partial compositional consistency."
        else: interp = "Low compositional consistency."
        
        return ConsistencyResult(
            probe_count=n,
            analyses_a=analyses_a,
            analyses_b=analyses_b,
            barycentric_correlation=bary_corr,
            angular_correlation=ang_corr,
            consistency_score=score,
            is_compatible=is_compatible,
            interpretation=interp
        )

    @staticmethod
    def pearson_correlation(a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or len(a) < 2: return 0.0
        
        arr_a = mx.array(a)
        arr_b = mx.array(b)
        
        mean_a = mx.mean(arr_a)
        mean_b = mx.mean(arr_b)
        
        da = arr_a - mean_a
        db = arr_b - mean_b
        
        sum_ab = mx.dot(da, db).item()
        sum_a2 = mx.dot(da, da).item()
        sum_b2 = mx.dot(db, db).item()
        
        denom = math.sqrt(sum_a2 * sum_b2)
        if denom > 1e-10:
            return sum_ab / denom
        return 0.0

    @staticmethod
    def analyze_all_probes(
        prime_embeddings: Dict[str, List[float]],
        composition_embeddings: Dict[str, List[float]],
        probes: List[CompositionProbe] = STANDARD_PROBES
    ) -> List[CompositionAnalysis]:
        analyses = []
        for probe in probes:
            if probe.phrase not in composition_embeddings:
                continue
                
            comp_embed = composition_embeddings[probe.phrase]
            
            component_embeds = []
            all_found = True
            for c in probe.components:
                if c in prime_embeddings:
                    component_embeds.append(prime_embeddings[c])
                else:
                    all_found = False
                    break
            
            if not all_found: continue
            
            analysis = CompositionalProbes.analyze_composition(comp_embed, component_embeds, probe)
            analyses.append(analysis)
            
        return analyses
