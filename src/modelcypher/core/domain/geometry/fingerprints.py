
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import mlx.core as mx
import operator

# Data Structures mimicking Swift ManifoldStitcher.ModelFingerprints

@dataclass(frozen=True)
class ActivatedDimension:
    index: int
    activation: float

@dataclass(frozen=True)
class Fingerprint:
    prime_id: str
    prime_text: str
    # layer_index -> list of ActivatedDimension
    activated_dimensions: Dict[int, List[ActivatedDimension]]

@dataclass(frozen=True)
class ModelFingerprints:
    model_id: str
    fingerprints: List[Fingerprint]

# Projection Logic

class ProjectionMethod(Enum):
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"

class ProjectionError(Exception):
    pass

@dataclass
class ProjectionFeature:
    layer: int
    dimension: int
    frequency: int
    
    @property
    def key(self) -> str:
        return f"{self.layer}:{self.dimension}"

@dataclass
class ProjectionPoint:
    id: str
    prime_id: str
    prime_text: str
    x: float
    y: float

@dataclass
class Projection:
    model_id: str
    method: ProjectionMethod
    max_features: int
    included_layers: Optional[List[List[int]]]
    features: List[ProjectionFeature]
    points: List[ProjectionPoint]

class ModelFingerprintsProjection:
    """
    Project model fingerprints to 2D for visualization.
    Ported from ModelFingerprintsProjection.swift.
    """
    
    @staticmethod
    def project_2d(
        fingerprints: ModelFingerprints,
        method: ProjectionMethod = ProjectionMethod.PCA,
        max_features: int = 1200,
        layers: Optional[Set[int]] = None,
        seed: int = 42
    ) -> Projection:
        
        if len(fingerprints.fingerprints) < 2:
            raise ProjectionError(f"Insufficient samples: {len(fingerprints.fingerprints)}")
            
        if method != ProjectionMethod.PCA:
            raise ProjectionError(f"Unsupported method: {method}")

        # 1. Feature Selection
        feature_list = ModelFingerprintsProjection._select_features(
            fingerprints, max_features, layers
        )
        
        if len(feature_list) < 2:
            raise ProjectionError(f"Insufficient features: {len(feature_list)}")
            
        # 2. Build Matrix
        n = len(fingerprints.fingerprints)
        d = len(feature_list)
        
        # Mapping (layer, dim) -> col_index
        feature_index = {
            (f.layer, f.dimension): i for i, f in enumerate(feature_list)
        }
        
        # Create dense matrix on CPU first (list of lists) or numpy
        # MLX construction: flat array then reshape?
        import numpy as np
        matrix_np = np.zeros((n, d), dtype=np.float32)
        
        for row, fp in enumerate(fingerprints.fingerprints):
            for layer, dims in fp.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    key = (layer, dim.index)
                    if key in feature_index:
                        col = feature_index[key]
                        matrix_np[row, col] = dim.activation
                        
        # Move to MLX
        X = mx.array(matrix_np)
        
        # 3. Normalize & Center
        # L2 Normalize rows
        norms = mx.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        X = mx.where(norms > 1e-9, X / norms, X)
        
        # Center columns (Mean centering)
        means = mx.mean(X, axis=0, keepdims=True)
        X = X - means
        
        # 4. PCA via SVD
        # X = U S V^T
        # Principal components are V columns.
        # But we want projection of X onto components, which is U @ S (or X @ V)
        # For 2D: take first 2 components.
        
        # mx.linalg.svd returns U, S, Vt
        # "full_matrices=False" is default behavior or need check? 
        # MLX svd always returns thin SVD I think? Let's assume standard behavior.
        # Note: if n < d, U is (n, n). 
        # Projection = U[:, :2] * S[:2]? 
        # Yes, standard PCA coordinates are T = X W = U S V^T V = U S.
        
        U, S, Vt = mx.linalg.svd(X)
        
        # Take top 2
        # U is (N, K), S is (K,), Vt is (K, D)
        # Coordinates = U * S
        
        coords = U[:, :2] * S[:2]
        
        # Extract points
        points = []
        # Convert back to python list
        coords_list = coords.tolist() 
        
        for i, fp in enumerate(fingerprints.fingerprints):
            pt = coords_list[i]
            points.append(ProjectionPoint(
                id=fp.prime_id,
                prime_id=fp.prime_id,
                prime_text=fp.prime_text,
                x=float(pt[0]),
                y=float(pt[1])
            ))
            
        included_layers = [sorted(list(layers))] if layers else None
        
        return Projection(
            model_id=fingerprints.model_id,
            method=method,
            max_features=max_features,
            included_layers=included_layers,
            features=feature_list,
            points=points
        )

    @staticmethod
    def _select_features(
        fingerprints: ModelFingerprints, 
        max_features: int, 
        layers: Optional[Set[int]]
    ) -> List[ProjectionFeature]:
        
        freq_map: Dict[Tuple[int, int], int] = {}
        
        for fp in fingerprints.fingerprints:
            for layer, dims in fp.activated_dimensions.items():
                if layers and layer not in layers:
                    continue
                for dim in dims:
                    key = (layer, dim.index)
                    freq_map[key] = freq_map.get(key, 0) + 1
                    
        # Sort by frequency desc, then layer asc, then dim asc
        sorted_items = sorted(
            freq_map.items(), 
            key=lambda x: (-x[1], x[0][0], x[0][1])
        )
        
        limit = max(1, max_features)
        selected = sorted_items[:limit]
        
        return [
            ProjectionFeature(layer=k[0], dimension=k[1], frequency=v)
            for k, v in selected
        ]
