
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict
import math
import uuid
import uuid as uuid_lib
from datetime import datetime
import numpy as np # For some list ops not efficiently done in MLX graph yet
import mlx.core as mx

from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimensionEstimator, TwoNNConfiguration

@dataclass(frozen=True)
class ManifoldPoint:
    """A point in the high-dimensional manifold representing a generation trace."""
    mean_entropy: float
    entropy_variance: float
    first_token_entropy: float
    gate_count: int
    mean_gate_confidence: float
    dominant_gate_category: int
    entropy_path_correlation: float
    assessment_strength: float
    prompt_hash: str
    
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    # 8-dimensional feature vector
    # [mean_ent, var_ent, first_ent, gate_ct, gate_conf, dom_gate, path_corr, assess_str]
    
    @property
    def feature_vector(self) -> mx.array:
        return mx.array([
            self.mean_entropy,
            self.entropy_variance,
            self.first_token_entropy,
            float(self.gate_count),
            self.mean_gate_confidence,
            float(self.dominant_gate_category),
            self.entropy_path_correlation,
            self.assessment_strength
        ], dtype=mx.float32)

    def distance(self, other: 'ManifoldPoint') -> float:
        # Euclidean distance
        # We assume vectors are small (8D), so computing on CPU or single GPU op is fast.
        # Instantiating arrays every time might be slow for many points.
        # But this is a direct port. Optimized version would batch this.
        diff = self.feature_vector - other.feature_vector
        dist_sq = mx.sum(diff * diff)
        return float(np.sqrt(dist_sq.item()))

@dataclass
class ManifoldRegion:
    """A clustered region in the manifold."""
    id: uuid.UUID
    region_type: str
    centroid: ManifoldPoint
    member_count: int
    member_ids: List[uuid.UUID]
    dominant_gates: List[str]
    intrinsic_dimension: Optional[float]
    radius: float
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def classify(centroid: ManifoldPoint) -> str:
        # Heuristic classification
        if centroid.mean_entropy < 1.0:
            return "crystalline" # Rigid
        elif centroid.mean_entropy > 4.0:
            return "sparse" # Chaotic
        elif centroid.entropy_variance > 2.0:
            return "transitional"
        else:
            return "dense" # Normal

class ManifoldClusterer:
    """
    Incremental DBSCAN clustering for manifold points.
    """
    
    @dataclass
    class Configuration:
        epsilon: float = 0.3
        min_points: int = 5
        compute_intrinsic_dimension: bool = True
        max_clusters: int = 50

    @dataclass
    class ClusteringResult:
        regions: List[ManifoldRegion]
        noise_points: List[ManifoldPoint]
        new_clusters_formed: int
        clusters_merged: int
        points_assigned_to_existing: int

    def __init__(self, configuration: Configuration = Configuration()):
        self.config = configuration

    def cluster(self, points: List[ManifoldPoint]) -> ClusteringResult:
        if not points:
            return self.ClusteringResult([], [], 0, 0, 0)
            
        N = len(points)
        labels = [-1] * N # -1 unvisited, -2 noise, >=0 cluster
        cluster_id = 0
        
        # Precompute distances? For 8D points, N=100-1000, O(N^2) is fine.
        # Optimization: Use MLX to compute full distance matrix at once.
        vectors = mx.stack([p.feature_vector for p in points]) # [N, 8]
        # Dist matrix:
        dots = vectors @ vectors.T
        norms = mx.sum(vectors * vectors, axis=1)
        dist_sq = norms[:, None] + norms[None, :] - 2 * dots
        dist_matrix = mx.sqrt(mx.abs(dist_sq)) # [N, N]
        
        # Pull to logical memory (numpy/python) for DBSCAN logic
        # We iterate sequentially anyway.
        dists = np.array(dist_matrix) 
        
        region_features: Dict[int, List[ManifoldPoint]] = {}
        
        def region_query(idx):
            # Return indices where dist < epsilon
            return np.where(dists[idx] <= self.config.epsilon)[0].tolist()

        for i in range(N):
            if labels[i] != -1: continue
            
            neighbors = region_query(i)
            
            if len(neighbors) < self.config.min_points:
                labels[i] = -2 # Noise
            else:
                self._expand_cluster(points, labels, i, neighbors, cluster_id, region_query)
                cluster_id += 1
                
        # Build regions
        regions = []
        noise = []
        
        for i, lbl in enumerate(labels):
            if lbl == -2:
                noise.append(points[i])
            elif lbl >= 0:
                if lbl not in region_features: region_features[lbl] = []
                region_features[lbl].append(points[i])
                
        for cid, members in region_features.items():
            if region := self._build_region(members):
                regions.append(region)
                
        return self.ClusteringResult(
            regions=regions,
            noise_points=noise,
            new_clusters_formed=len(regions),
            clusters_merged=0,
            points_assigned_to_existing=0
        )

    def _expand_cluster(self, points, labels, point_idx, neighbors, cluster_id, query_fn):
        labels[point_idx] = cluster_id
        seed_set = list(neighbors)
        
        i = 0
        while i < len(seed_set):
            neighbor_idx = seed_set[i]
            
            if labels[neighbor_idx] == -2:
                labels[neighbor_idx] = cluster_id # Border point
                
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                
                new_neighbors = query_fn(neighbor_idx)
                if len(new_neighbors) >= self.config.min_points:
                    # Merge neighbors into seed set
                    # Simply appending is safe, avoiding duplicates is optimization
                    # Using set for efficient lookup
                    existing = set(seed_set)
                    for n in new_neighbors:
                        if n not in existing:
                            seed_set.append(n)
                            existing.add(n)
                            
            i += 1

    def _build_region(self, points: List[ManifoldPoint], existing_id: Optional[uuid.UUID] = None) -> Optional[ManifoldRegion]:
        if not points: return None
        
        centroid = self._compute_centroid(points)
        
        # Radius
        max_dist = 0.0
        c_vec = centroid.feature_vector
        for p in points:
            d = mx.sum((p.feature_vector - c_vec)**2).item()
            if d > max_dist: max_dist = d
        radius = math.sqrt(max_dist)
        
        # ID Estimation
        id_est = None
        if self.config.compute_intrinsic_dimension and len(points) >= 3:
            try:
                # Convert points to [N, D] mx array
                data = mx.stack([p.feature_vector for p in points])
                est = IntrinsicDimensionEstimator.estimate_two_nn(data)
                id_est = est.intrinsic_dimension
            except Exception:
                pass
                
        return ManifoldRegion(
            id=existing_id or uuid.uuid4(),
            region_type=ManifoldRegion.classify(centroid),
            centroid=centroid,
            member_count=len(points),
            member_ids=[p.id for p in points],
            dominant_gates=[], # Placeholder for now, simplistic logic omitted
            intrinsic_dimension=id_est,
            radius=radius
        )

    def _compute_centroid(self, points: List[ManifoldPoint]) -> ManifoldPoint:
        # Average all fields
        count = float(len(points))
        sums = [0.0] * 8
        for p in points:
            sums[0] += p.mean_entropy
            sums[1] += p.entropy_variance
            sums[2] += p.first_token_entropy
            sums[3] += float(p.gate_count)
            sums[4] += p.mean_gate_confidence
            sums[5] += float(p.dominant_gate_category)
            sums[6] += p.entropy_path_correlation
            sums[7] += p.assessment_strength
            
        return ManifoldPoint(
            mean_entropy=sums[0]/count,
            entropy_variance=sums[1]/count,
            first_token_entropy=sums[2]/count,
            gate_count=int(sums[3]/count),
            mean_gate_confidence=sums[4]/count,
            dominant_gate_category=int(sums[5]/count),
            entropy_path_correlation=sums[6]/count,
            assessment_strength=sums[7]/count,
            prompt_hash="centroid"
        )
