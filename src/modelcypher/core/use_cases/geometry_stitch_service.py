"""Geometry stitch service for manifold stitching operations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from modelcypher.core.domain.geometry.affine_stitching_layer import (
    AffineStitchingLayer,
    AnchorPair,
    Config as StitchConfig,
    Result as StitchResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StitchPoint:
    """A stitching point between manifolds."""
    layer_name: str
    source_dim: int
    target_dim: int
    quality_score: float


@dataclass(frozen=True)
class StitchAnalysisResult:
    """Result of analyzing manifold stitching between checkpoints."""
    manifold_distance: float
    stitching_points: list[StitchPoint]
    recommended_config: dict
    interpretation: str


@dataclass(frozen=True)
class StitchApplyResult:
    """Result of applying stitching operation."""
    output_path: str
    stitched_layers: int
    quality_score: float


class GeometryStitchService:
    """Service for manifold stitching analysis and operations."""

    def analyze(self, checkpoints: list[str]) -> StitchAnalysisResult:
        """Analyze manifold stitching between checkpoints.
        
        Args:
            checkpoints: List of paths to checkpoint directories.
            
        Returns:
            StitchAnalysisResult with manifold distance and stitching points.
        """
        if len(checkpoints) < 2:
            raise ValueError("At least two checkpoints are required for stitching analysis")
        
        # Load weights from first two checkpoints
        path_a = Path(checkpoints[0]).expanduser().resolve()
        path_b = Path(checkpoints[1]).expanduser().resolve()
        
        if not path_a.exists():
            raise ValueError(f"Checkpoint path does not exist: {path_a}")
        if not path_b.exists():
            raise ValueError(f"Checkpoint path does not exist: {path_b}")
        
        weights_a = self._load_weights(path_a)
        weights_b = self._load_weights(path_b)
        
        # Find common layers
        common_layers = set(weights_a.keys()) & set(weights_b.keys())
        if not common_layers:
            return StitchAnalysisResult(
                manifold_distance=1.0,
                stitching_points=[],
                recommended_config={},
                interpretation="No common layers found between checkpoints.",
            )
        
        # Analyze each layer for stitching potential
        stitching_points = []
        total_distance = 0.0
        
        for layer_name in sorted(common_layers):
            tensor_a = weights_a[layer_name]
            tensor_b = weights_b[layer_name]
            
            if tensor_a.shape != tensor_b.shape:
                continue
            
            # Compute manifold distance for this layer
            distance = self._compute_manifold_distance(tensor_a, tensor_b)
            total_distance += distance
            
            # Determine if this is a good stitching point
            quality = 1.0 - distance
            if quality > 0.5:  # Only include good stitching points
                stitching_points.append(StitchPoint(
                    layer_name=layer_name,
                    source_dim=tensor_a.shape[-1] if len(tensor_a.shape) > 1 else tensor_a.shape[0],
                    target_dim=tensor_b.shape[-1] if len(tensor_b.shape) > 1 else tensor_b.shape[0],
                    quality_score=quality,
                ))
        
        num_layers = len(common_layers)
        avg_distance = total_distance / num_layers if num_layers > 0 else 1.0
        manifold_distance = min(1.0, max(0.0, avg_distance))
        
        # Generate recommended config
        recommended_config = {
            "learning_rate": 0.01 if manifold_distance < 0.3 else 0.001,
            "max_iterations": 500 if manifold_distance < 0.3 else 1000,
            "use_procrustes_warm_start": manifold_distance < 0.5,
        }
        
        if manifold_distance < 0.2:
            interpretation = "Checkpoints are closely aligned. Stitching should be straightforward."
        elif manifold_distance < 0.5:
            interpretation = "Checkpoints show moderate divergence. Stitching is feasible with care."
        else:
            interpretation = "Checkpoints are significantly divergent. Stitching may be challenging."
        
        return StitchAnalysisResult(
            manifold_distance=manifold_distance,
            stitching_points=stitching_points,
            recommended_config=recommended_config,
            interpretation=interpretation,
        )


    def apply(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        config: dict | None = None,
    ) -> StitchApplyResult:
        """Apply stitching operation between source and target.
        
        Args:
            source_path: Path to source checkpoint.
            target_path: Path to target checkpoint.
            output_path: Path for output stitched model.
            config: Optional stitching configuration.
            
        Returns:
            StitchApplyResult with output path and quality metrics.
        """
        source = Path(source_path).expanduser().resolve()
        target = Path(target_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        
        if not source.exists():
            raise ValueError(f"Source path does not exist: {source}")
        if not target.exists():
            raise ValueError(f"Target path does not exist: {target}")
        
        # Create output directory
        output.mkdir(parents=True, exist_ok=True)
        
        # Load weights
        source_weights = self._load_weights(source)
        target_weights = self._load_weights(target)
        
        # Parse config
        stitch_config = StitchConfig(
            learning_rate=config.get("learning_rate", 0.01) if config else 0.01,
            max_iterations=config.get("max_iterations", 500) if config else 500,
            use_procrustes_warm_start=config.get("use_procrustes_warm_start", True) if config else True,
        )
        
        # Find common layers and stitch
        common_layers = set(source_weights.keys()) & set(target_weights.keys())
        stitched_weights = {}
        stitched_count = 0
        total_quality = 0.0
        
        for layer_name in common_layers:
            source_tensor = source_weights[layer_name]
            target_tensor = target_weights[layer_name]
            
            if source_tensor.shape != target_tensor.shape:
                # Keep source weights for incompatible layers
                stitched_weights[layer_name] = source_tensor
                continue
            
            # Simple linear interpolation for stitching
            # In a full implementation, this would use AffineStitchingLayer
            alpha = 0.5
            stitched = (1 - alpha) * source_tensor + alpha * target_tensor
            stitched_weights[layer_name] = stitched.astype(np.float32)
            stitched_count += 1
            
            # Compute quality
            quality = 1.0 - self._compute_manifold_distance(source_tensor, target_tensor)
            total_quality += quality
        
        # Add any layers only in source
        for layer_name in source_weights:
            if layer_name not in stitched_weights:
                stitched_weights[layer_name] = source_weights[layer_name]
        
        # Save stitched weights
        output_file = output / "model.safetensors"
        save_file(stitched_weights, output_file)
        
        # Copy config.json if exists
        source_config = source / "config.json"
        if source_config.exists():
            (output / "config.json").write_text(
                source_config.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        
        avg_quality = total_quality / stitched_count if stitched_count > 0 else 0.0
        
        return StitchApplyResult(
            output_path=str(output),
            stitched_layers=stitched_count,
            quality_score=avg_quality,
        )

    def _load_weights(self, path: Path) -> dict:
        """Load weights from safetensors files."""
        weights = {}
        
        safetensor_files = list(path.glob("*.safetensors"))
        for st_file in safetensor_files:
            try:
                with safe_open(st_file, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            except Exception as exc:
                logger.warning("Failed to read safetensors file %s: %s", st_file, exc)
        
        return weights

    def _compute_manifold_distance(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
        """Compute normalized manifold distance between two tensors."""
        diff = tensor_a.astype(np.float32) - tensor_b.astype(np.float32)
        norm_diff = np.linalg.norm(diff.flatten())
        norm_a = np.linalg.norm(tensor_a.astype(np.float32).flatten())
        norm_b = np.linalg.norm(tensor_b.astype(np.float32).flatten())
        
        max_norm = max(norm_a, norm_b, 1e-8)
        relative_distance = norm_diff / max_norm
        
        # Normalize to [0, 1]
        return float(min(1.0, max(0.0, 1.0 - np.exp(-relative_distance))))
