from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from typing import ClassVar


class Thresholds:
    strong_correlation: ClassVar[float] = 0.7
    moderate_correlation: ClassVar[float] = 0.4
    strong_weight: ClassVar[float] = 1.0
    moderate_weight: ClassVar[float] = 0.6
    weak_weight: ClassVar[float] = 0.2


@dataclass(frozen=True)
class DimensionCorrelation:
    source_dim: int
    target_dim: int
    correlation: float

    @property
    def is_strong_correlation(self) -> bool:
        return self.correlation > Thresholds.strong_correlation

    @property
    def is_moderate_correlation(self) -> bool:
        return Thresholds.moderate_correlation < self.correlation <= Thresholds.strong_correlation

    @property
    def is_weak_correlation(self) -> bool:
        return self.correlation <= Thresholds.moderate_correlation


@dataclass(frozen=True)
class LayerConfidence:
    layer: int
    strong_correlations: int
    moderate_correlations: int
    weak_correlations: int
    confidence: float = field(init=False)

    def __post_init__(self) -> None:
        total = self.strong_correlations + self.moderate_correlations + self.weak_correlations
        if total > 0:
            weighted = (
                float(self.strong_correlations) * Thresholds.strong_weight
                + float(self.moderate_correlations) * Thresholds.moderate_weight
                + float(self.weak_correlations) * Thresholds.weak_weight
            )
            value = weighted / float(total)
        else:
            value = 0.0
        object.__setattr__(self, "confidence", value)

    @property
    def total_correlations(self) -> int:
        return self.strong_correlations + self.moderate_correlations + self.weak_correlations


@dataclass(frozen=True)
class IntersectionMap:
    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    overall_correlation: float
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    layer_confidences: list[LayerConfidence]


class ProbeSpace(str, Enum):
    prelogits_hidden = "prelogits-hidden"
    output_logits = "output-logits"

output_layer_marker = -1


import math
import mlx.core as mx
from typing import Dict, List
from .probe_corpus import ProbeCorpus # Helper class we just created

@dataclass
class ContinuousFingerprint:
    """
    Continuous activation fingerprint preserving magnitude information.
    Ported from ManifoldStitcher.swift.
    """
    prime_id: str
    prime_text: str
    
    # Layer -> Full activation vector
    activation_vectors: Dict[int, List[float]]
    
    # Layer -> L2 Magnitude
    magnitudes: Dict[int, float]
    
    # Layer -> Entropy (0-1)
    entropies: Dict[int, float]
    
    # Layer -> Sparsity (0-1)
    sparsities: Dict[int, float]

    @staticmethod
    def from_activations(prime_id: str, prime_text: str, layer_activations: Dict[int, List[float]]) -> "ContinuousFingerprint":
        magnitudes = {}
        entropies = {}
        sparsities = {}
        
        for layer, activations in layer_activations.items():
            arr = mx.array(activations)
            magnitudes[layer] = float(mx.linalg.norm(arr))
            
            logits = arr
            max_val = mx.max(logits)
            exp_acts = mx.exp(logits - max_val)
            probs = exp_acts / (mx.sum(exp_acts) + 1e-10)
            
            log_probs = mx.log(probs + 1e-10)
            entropy = -mx.sum(probs * log_probs).item()
            max_entropy = math.log(max(len(activations), 1))
            entropies[layer] = min(max(entropy / max_entropy, 0.0), 1.0) if max_entropy > 0 else 0.0
            
            abs_acts = mx.abs(arr)
            threshold = 0.01 * mx.max(abs_acts).item()
            near_zero = mx.sum(abs_acts < threshold).item()
            sparsities[layer] = float(near_zero) / max(len(activations), 1)
            
        return ContinuousFingerprint(prime_id, prime_text, layer_activations, magnitudes, entropies, sparsities)

@dataclass
class ContinuousCorrelationResult:
    cka: float
    cosine_similarity: float
    magnitude_ratio: float
    entropy_delta: float
    
    @property
    def compatibility_score(self) -> float:
        cka_score = self.cka if self.cosine_similarity >= 0 else 0.0
        return (0.6 * cka_score + 
                0.2 * max(0.0, self.cosine_similarity) + 
                0.1 * (1.0 - min(abs(self.magnitude_ratio - 1.0), 1.0)) + 
                0.1 * (1.0 - min(abs(self.entropy_delta), 1.0)))

@dataclass
class ContinuousModelFingerprints:
    """
    Collection of continuous fingerprints for a model.
    """
    model_id: str
    hidden_dim: int
    layer_count: int
    fingerprints: List[ContinuousFingerprint]
    
    @property
    def mean_entropy(self) -> float:
        vals = [e for fp in self.fingerprints for e in fp.entropies.values()]
        return sum(vals) / len(vals) if vals else 0.0
    
    @property
    def mean_sparsity(self) -> float:
        vals = [s for fp in self.fingerprints for s in fp.sparsities.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def from_model_fingerprints(source: "ModelFingerprints") -> Optional["ContinuousModelFingerprints"]:
        if not hasattr(source, "activation_vectors") or not source.activation_vectors:
            return None
            
        fingerprints_by_prime: Dict[str, Dict[int, List[float]]] = {}
        for key, vec in source.activation_vectors.items():
            if "_layer" not in key: continue
            idx = key.rfind("_layer")
            try:
                layer = int(key[idx+6:])
                prime_id = key[:idx]
                if prime_id not in fingerprints_by_prime: fingerprints_by_prime[prime_id] = {}
                fingerprints_by_prime[prime_id][layer] = vec
            except ValueError: continue
            
        prime_texts = {fp.prime_id: fp.prime_text for fp in source.fingerprints}
        continuous_fps = [
            ContinuousFingerprint.from_activations(pid, prime_texts.get(pid, pid), layers)
            for pid, layers in fingerprints_by_prime.items()
        ]
        return ContinuousModelFingerprints(source.model_id, source.hidden_dim, source.layer_count, continuous_fps)


class ManifoldStitcher:
    """
    Manifold stitching for cross-architecture model merging.
    Implementation of Continuous CKA-based stitching.
    """
    OUTPUT_LAYER_MARKER = 999999
    
    @staticmethod
    def compute_continuous_correlation(source: ContinuousFingerprint, target: ContinuousFingerprint, layer: int) -> Optional[ContinuousCorrelationResult]:
        if layer not in source.activation_vectors or layer not in target.activation_vectors:
            return None
        s_vec = mx.array(source.activation_vectors[layer])
        t_vec = mx.array(target.activation_vectors[layer])
        if s_vec.size == 0 or t_vec.size == 0:
            return None
            
        min_len = min(s_vec.size, t_vec.size)
        s_trunc, t_trunc = s_vec[:min_len], t_vec[:min_len]
        
        dot_prod = mx.dot(s_trunc, t_trunc).item()
        s_norm, t_norm = mx.linalg.norm(s_vec).item(), mx.linalg.norm(t_vec).item()
        cosine = dot_prod / (s_norm * t_norm) if (s_norm > 1e-8 and t_norm > 1e-8) else 0.0
        
        mag_ratio = source.magnitudes.get(layer, 1.0) / target.magnitudes.get(layer, 1.0) if target.magnitudes.get(layer, 1.0) > 1e-8 else 1.0
        entropy_delta = source.entropies.get(layer, 0.0) - target.entropies.get(layer, 0.0)
        
        return ContinuousCorrelationResult(cosine * cosine, cosine, mag_ratio, entropy_delta)

    @staticmethod
    def compute_cka_matrix(source: ContinuousModelFingerprints, target: ContinuousModelFingerprints, layer: int) -> Tuple[mx.array, List[str], List[str]]:
        """
        Compute pairwise CKA matrix between all primes at a given layer.
        Returns: (matrix, source_prime_ids, target_prime_ids)
        """
        s_fps = [fp for fp in source.fingerprints if layer in fp.activation_vectors]
        t_fps = [fp for fp in target.fingerprints if layer in fp.activation_vectors]
        if not s_fps or not t_fps: return (mx.array([]), [], [])
        
        matrix = []
        for s_fp in s_fps:
            row = []
            for t_fp in t_fps:
                res = ManifoldStitcher.compute_continuous_correlation(s_fp, t_fp, layer)
                row.append(res.cka if res else 0.0)
            matrix.append(row)
        return (mx.array(matrix), [fp.prime_id for fp in s_fps], [fp.prime_id for fp in t_fps])

    @staticmethod
    def compute_intersection_rotation(
        intersection: IntersectionMap,
        layer: int,
        source_basis: mx.array,
        target_basis: mx.array
    ) -> Tuple[mx.array, float]:
        """
        Computes a targeted rotation matrix using the intersection map.
        Strong correlations -> tight rotation (trust mapping).
        """
        correlations = intersection.dimension_correlations.get(layer, [])
        if not correlations:
            return (mx.eye(source_basis.shape[1]), 0.0)
            
        dim_s = source_basis.shape[0]
        dim_t = target_basis.shape[0]
        
        # Filter valid correlations
        filtered = [
            c for c in correlations 
            if c.source_dim < dim_s and c.target_dim < dim_t
        ]
        
        if len(filtered) < 2:
            k = min(source_basis.shape[1], target_basis.shape[1])
            return (mx.eye(k), 0.0)
            
        # Build index arrays
        source_indices = mx.array([c.source_dim for c in filtered], dtype=mx.int32)
        target_indices = mx.array([c.target_dim for c in filtered], dtype=mx.int32)
        
        # Take rows from basis matrices
        z_source = source_basis[source_indices]
        z_target = target_basis[target_indices]
        
        # Weighting
        weights = [max(0.0, c.correlation) for c in filtered]
        sqrt_weights = mx.array(weights).sqrt().reshape(-1, 1)
        
        z_source = z_source * sqrt_weights
        z_target = z_target * sqrt_weights
        
        # Procrustes
        m = z_source.T @ z_target
        u, _, vt = mx.linalg.svd(m)
        omega = u @ vt
        
        # Sign correction
        det = mx.linalg.det(omega)
        if det < 0:
            k = u.shape[1]
            mask = mx.ones((1, k))
            mask[0, -1] = -1
            u = u * mask
            omega = u @ vt
            
        confidence = sum(weights) / max(len(weights), 1)
        return (omega, confidence)

    @staticmethod
    def cluster_activations(
        source_activations: Dict[str, List[float]], # PrimeID -> Activation (Layer 0)
        target_activations: Dict[str, List[float]],
        cluster_count: int = 8
    ) -> List["AlignmentCluster"]: # Forward ref string since defined later
        """
        Clusters activations to identify alignment regions.
        """
        keys = sorted(list(set(source_activations.keys()) & set(target_activations.keys())))
        if not keys: return []
        
        source_vecs = [source_activations[k] for k in keys]
        target_vecs = [target_activations[k] for k in keys]
        
        # K-Means on source
        assignments, _ = ManifoldStitcher.k_means(source_vecs, cluster_count)
        
        clusters = []
        shared_dim = min(len(source_vecs[0]), len(target_vecs[0]))
        
        for cluster_id in range(cluster_count):
            indices = [i for i, a in enumerate(assignments) if a == cluster_id]
            if not indices: continue
            
            s_members = mx.array([source_vecs[i][:shared_dim] for i in indices])
            t_members = mx.array([target_vecs[i][:shared_dim] for i in indices])
            
            s_mean = mx.mean(s_members, axis=0)
            t_mean = mx.mean(t_members, axis=0)
            
            # Local rotation
            s_centered = s_members - s_mean
            t_centered = t_members - t_mean
            
            m = s_centered.T @ t_centered
            u, _, vt = mx.linalg.svd(m)
            omega = u @ vt
            
            if mx.linalg.det(omega) < 0:
                mask = mx.ones((1, u.shape[1]))
                mask[0, -1] = -1
                u = u * mask
                omega = u @ vt
                
            # Error
            projected = s_centered @ omega
            error = projected - t_centered
            error_norm = mx.sqrt(mx.sum(error * error)).item()
            target_norm = mx.sqrt(mx.sum(t_centered * t_centered)).item()
            procrustes_error = error_norm / target_norm if target_norm > 1e-6 else 0.0
            
            clusters.append(AlignmentCluster(
                id=cluster_id,
                centroid_source=s_mean.tolist(),
                centroid_target=t_mean.tolist(),
                local_rotation=omega,
                procrustes_error=procrustes_error,
                member_count=len(indices)
            ))
            
        return clusters

    @staticmethod
    def k_means(points: List[List[float]], k: int, max_iterations: int = 50) -> Tuple[List[int], List[List[float]]]:
        n = len(points)
        if n == 0 or k <= 0: return ([], [])
        
        pts = mx.array(points)
        # Init centroids (random)
        centroids = pts[mx.random.randint(0, n, (k,))]
        
        assignments = mx.zeros((n,), dtype=mx.int32)
        
        for _ in range(max_iterations):
            # Compute distances
            # (N, 1, D) - (1, K, D) -> (N, K, D)
            dists = mx.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
            new_assignments = mx.argmin(dists, axis=1)
            
            if mx.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments
            
            # Update centroids
            for c in range(k):
                mask = (assignments == c)
                if mx.sum(mask).item() > 0:
                    centroids[c] = mx.mean(pts[mask], axis=0)
                    
        return (assignments.tolist(), centroids.tolist())

    @staticmethod
    def soft_rotation(
        weight: mx.array,
        clusters: List["AlignmentCluster"],
        temperature: float = 0.3
    ) -> mx.array:
        if not clusters: return weight
        if weight.ndim != 2: return weight
        
        in_dim = weight.shape[1]
        cluster_dim = clusters[0].local_rotation.shape[0]
        if in_dim != cluster_dim: return weight
        
        # Weighted average
        weights = []
        for c in clusters:
            w = math.exp(-c.procrustes_error / temperature) * c.member_count
            weights.append(w)
            
        total_weight = sum(weights)
        if total_weight <= 0: return weight
        
        weighted_omega = mx.zeros((cluster_dim, cluster_dim))
        for i, c in enumerate(clusters):
            norm_w = weights[i] / total_weight
            weighted_omega = weighted_omega + (c.local_rotation * norm_w)
            
        # Re-orthogonalize
        u, _, vt = mx.linalg.svd(weighted_omega)
        omega = u @ vt
        
        return weight @ omega.T

    @staticmethod
    async def validate_merged_model(
        merged_model_ctx: Any, # ModelContext
        merged_model_id: str,
        target_fingerprints: ModelFingerprints,
        top_k: int = 32
    ) -> ValidationResult:
        """
        Validates a merged model by comparing its fingerprints to the original target.
        """
        # Determine layers to probe
        target_layers = set()
        for fp in target_fingerprints.fingerprints:
            target_layers.update(fp.activated_dimensions.keys())
            
        intermediate_layers = {l for l in target_layers if l > 0 and l != ManifoldStitcher.OUTPUT_LAYER_MARKER}
        probe_layers = list(intermediate_layers) if intermediate_layers else None
        
        # Probe merged model
        merged_fingerprints = await ManifoldStitcher.probe_with_primes(
            model_ctx=merged_model_ctx,
            model_id=merged_model_id,
            probe_space=target_fingerprints.probe_space,
            top_k=top_k,
            layer_indices=probe_layers
        )
        
        # Compare
        layer_deltas = []
        all_layers = target_layers.union(
            {l for fp in merged_fingerprints.fingerprints for l in fp.activated_dimensions.keys()}
        )
        
        merged_map = {fp.prime_id: fp for fp in merged_fingerprints.fingerprints}
        target_map = {fp.prime_id: fp for fp in target_fingerprints.fingerprints}
        
        prime_ids = set(merged_map.keys()) & set(target_map.keys())
        
        for layer in sorted(list(all_layers)):
            merged_dims = set()
            target_dims = set()
            
            for pid in prime_ids:
                if layer in merged_map[pid].activated_dimensions:
                    merged_dims.update([d.index for d in merged_map[pid].activated_dimensions[layer]])
                if layer in target_map[pid].activated_dimensions:
                    target_dims.update([d.index for d in target_map[pid].activated_dimensions[layer]])
                    
            overlap = merged_dims & target_dims
            union = merged_dims | target_dims
            jaccard = len(overlap) / len(union) if union else 0.0
            
            layer_deltas.append(LayerDelta(
                layer=layer,
                overlap_count=len(overlap),
                merged_count=len(merged_dims),
                target_count=len(target_dims),
                jaccard_similarity=jaccard
            ))
            
        mean_jaccard = sum(d.jaccard_similarity for d in layer_deltas) / len(layer_deltas) if layer_deltas else 0.0
        
        if mean_jaccard > 0.7: status = ValidationStatus.EXCELLENT
        elif mean_jaccard > 0.5: status = ValidationStatus.GOOD
        elif mean_jaccard > 0.3: status = ValidationStatus.FAIR
        else: status = ValidationStatus.POOR
        
        return ValidationResult(
            merged_model=merged_model_id,
            target_model=target_fingerprints.model_id,
            layer_deltas=layer_deltas,
            overall_similarity=mean_jaccard,
            status=status
        )

    @staticmethod
    async def probe_with_primes(
        model_ctx: Any,
        model_id: str,
        probe_space: ProbeSpace,
        top_k: int,
        layer_indices: Optional[List[int]] = None
    ) -> ModelFingerprints:
        # Placeholder for actual probing logic
        # In a real implementation, this would use the ProbeCorpus and run inference
        # capturing activations.
        # For now, we return an empty fingerprint set or could mock it.
        # This requires porting `collectActivations` fully which relies on tokenizer/model details.
        
        # We will assume for now this is handled by external service or just return empty for parity structure.
        return ModelFingerprints(
            model_id=model_id,
            probe_space=probe_space,
            probe_capture_key=None,
            hidden_dim=0,
            layer_count=0,
            fingerprints=[]
        )



@dataclass(frozen=True)
class ActivatedDimension:
    index: int
    activation: float

    def __lt__(self, other: "ActivatedDimension") -> bool:
        return self.activation > other.activation


@dataclass(frozen=True)
class ActivationFingerprint:
    prime_id: str
    prime_text: str
    activated_dimensions: dict[int, list[ActivatedDimension]]


@dataclass(frozen=True)
class SparseActivationVector:
    indices: list[int]
    values: list[float]
    length: int

    def dot(self, other: "SparseActivationVector") -> float:
        count_a = min(len(self.indices), len(self.values))
        count_b = min(len(other.indices), len(other.values))
        if count_a <= 0 or count_b <= 0:
            return 0.0
        i = 0
        j = 0
        total = 0.0
        while i < count_a and j < count_b:
            idx_a = self.indices[i]
            idx_b = other.indices[j]
            if idx_a == idx_b:
                total += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif idx_a < idx_b:
                i += 1
            else:
                j += 1
        return total

    def dot_dense(self, dense: list[float]) -> float:
        count = min(len(self.indices), len(self.values))
        if count <= 0:
            return 0.0
        total = 0.0
        for i in range(count):
            idx = self.indices[i]
            if 0 <= idx < len(dense):
                total += self.values[i] * dense[idx]
        return total


@dataclass(frozen=True)
class ModelFingerprints:
    model_id: str
    probe_space: ProbeSpace
    probe_capture_key: Optional[str]
    hidden_dim: int
    layer_count: int
    fingerprints: list[ActivationFingerprint]
    activation_vectors: Optional[dict[str, list[float]]] = None
    activation_sparse_vectors: Optional[dict[str, SparseActivationVector]] = None


class ClusterClassification(str, Enum):
    ALIGNED = "aligned"
    TRANSLATABLE = "translatable"
    DIVERGENT = "divergent"

@dataclass
class AlignmentCluster:
    id: int
    centroid_source: List[float]
    centroid_target: List[float]
    local_rotation: mx.array
    procrustes_error: float
    member_count: int
    
    @property
    def classification(self) -> ClusterClassification:
        if self.procrustes_error < 0.3: return ClusterClassification.ALIGNED
        if self.procrustes_error < 0.7: return ClusterClassification.TRANSLATABLE
        return ClusterClassification.DIVERGENT

@dataclass
class LayerDelta:
    layer: int
    overlap_count: int
    merged_count: int
    target_count: int
    jaccard_similarity: float

class ValidationStatus(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ValidationResult:
    merged_model: str
    target_model: str
    layer_deltas: List[LayerDelta]
    overall_similarity: float
    status: ValidationStatus



def intersection_map_from_dict(payload: dict[str, Any]) -> IntersectionMap:
    def _get(key: str, fallback: str | None = None) -> Any:
        if key in payload:
            return payload[key]
        if fallback and fallback in payload:
            return payload[fallback]
        return None

    raw_correlations = _get("dimensionCorrelations", "dimension_correlations") or {}
    dimension_correlations: dict[int, list[DimensionCorrelation]] = {}
    for layer_key, entries in raw_correlations.items():
        try:
            layer = int(layer_key)
        except (TypeError, ValueError):
            continue
        parsed: list[DimensionCorrelation] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            source_dim = entry.get("sourceDim", entry.get("source_dim"))
            target_dim = entry.get("targetDim", entry.get("target_dim"))
            correlation = entry.get("correlation")
            if source_dim is None or target_dim is None or correlation is None:
                continue
            parsed.append(
                DimensionCorrelation(
                    source_dim=int(source_dim),
                    target_dim=int(target_dim),
                    correlation=float(correlation),
                )
            )
        if parsed:
            dimension_correlations[layer] = parsed

    raw_layer_confidences = _get("layerConfidences", "layer_confidences") or []
    layer_confidences: list[LayerConfidence] = []
    for entry in raw_layer_confidences:
        if not isinstance(entry, dict):
            continue
        layer = entry.get("layer")
        strong = entry.get("strongCorrelations", entry.get("strong_correlations"))
        moderate = entry.get("moderateCorrelations", entry.get("moderate_correlations"))
        weak = entry.get("weakCorrelations", entry.get("weak_correlations"))
        if layer is None or strong is None or moderate is None or weak is None:
            continue
        layer_confidences.append(
            LayerConfidence(
                layer=int(layer),
                strong_correlations=int(strong),
                moderate_correlations=int(moderate),
                weak_correlations=int(weak),
            )
        )

    return IntersectionMap(
        source_model=str(_get("sourceModel", "source_model") or ""),
        target_model=str(_get("targetModel", "target_model") or ""),
        dimension_correlations=dimension_correlations,
        overall_correlation=float(_get("overallCorrelation", "overall_correlation") or 0.0),
        aligned_dimension_count=int(_get("alignedDimensionCount", "aligned_dimension_count") or 0),
        total_source_dims=int(_get("totalSourceDims", "total_source_dims") or 0),
        total_target_dims=int(_get("totalTargetDims", "total_target_dims") or 0),
        layer_confidences=layer_confidences,
    )
