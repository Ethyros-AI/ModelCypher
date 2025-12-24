"""Adapter service for LoRA adapter operations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerAdapterInfo:
    """Information about adapter weights for a single layer."""
    name: str
    rank: int
    alpha: float
    parameters: int


@dataclass(frozen=True)
class AdapterInspectResult:
    """Result of inspecting an adapter."""
    rank: int
    alpha: float
    target_modules: list[str]
    sparsity: float
    parameter_count: int
    layer_analysis: list[LayerAdapterInfo]


@dataclass(frozen=True)
class ProjectResult:
    """Result of projecting an adapter."""
    output_path: str
    projected_layers: int


@dataclass(frozen=True)
class WrapResult:
    """Result of wrapping an adapter for MLX."""
    output_path: str
    wrapped_layers: int


@dataclass(frozen=True)
class SmoothResult:
    """Result of smoothing adapter weights."""
    output_path: str
    smoothed_layers: int
    variance_reduction: float


@dataclass(frozen=True)
class AdapterMergeResult:
    """Result of merging multiple adapters using geometric alignment."""
    output_path: str
    merged_modules: int
    procrustes_error: float
    permutation_quality: float
    merge_confidence: float
    ensemble_recommendation: Optional[dict]


class AdapterService:
    """Service for LoRA adapter operations."""

    def inspect(self, adapter_path: str) -> AdapterInspectResult:
        """Return detailed adapter analysis.
        
        Args:
            adapter_path: Path to adapter directory.
            
        Returns:
            AdapterInspectResult with adapter details.
        """
        path = Path(adapter_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {path}")
        
        # Load adapter config
        config_path = path / "adapter_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            rank = config.get("r", config.get("rank", 8))
            alpha = config.get("lora_alpha", config.get("alpha", 16.0))
            target_modules = config.get("target_modules", [])
        else:
            rank = 8
            alpha = 16.0
            target_modules = []
        
        # Analyze weights
        weights = self._load_weights(path)
        layer_analysis = []
        total_params = 0
        zero_count = 0
        total_elements = 0
        
        for name, tensor in weights.items():
            params = tensor.size
            total_params += params
            total_elements += params
            zero_count += np.sum(np.abs(tensor) < 1e-8)
            
            layer_analysis.append(LayerAdapterInfo(
                name=name,
                rank=rank,
                alpha=alpha,
                parameters=params,
            ))
        
        sparsity = zero_count / total_elements if total_elements > 0 else 0.0
        
        return AdapterInspectResult(
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            sparsity=float(sparsity),
            parameter_count=total_params,
            layer_analysis=layer_analysis,
        )

    def project(self, adapter_path: str, target_space: str, output_path: str) -> ProjectResult:
        """Project adapter to target space.
        
        Args:
            adapter_path: Path to adapter directory.
            target_space: Target space identifier.
            output_path: Output path for projected adapter.
            
        Returns:
            ProjectResult with output path.
        """
        path = Path(adapter_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        
        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {path}")
        
        output.mkdir(parents=True, exist_ok=True)
        
        weights = self._load_weights(path)
        projected_weights = {}
        
        for name, tensor in weights.items():
            # Simple projection: normalize weights
            norm = np.linalg.norm(tensor)
            if norm > 0:
                projected_weights[name] = (tensor / norm).astype(np.float32)
            else:
                projected_weights[name] = tensor.astype(np.float32)
        
        save_file(projected_weights, output / "adapter_model.safetensors")
        
        # Copy config
        config_path = path / "adapter_config.json"
        if config_path.exists():
            (output / "adapter_config.json").write_text(
                config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        
        return ProjectResult(
            output_path=str(output),
            projected_layers=len(projected_weights),
        )

    def wrap_mlx(self, adapter_path: str, output_path: str) -> WrapResult:
        """Wrap adapter for MLX compatibility.
        
        Args:
            adapter_path: Path to adapter directory.
            output_path: Output path for wrapped adapter.
            
        Returns:
            WrapResult with output path.
        """
        path = Path(adapter_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        
        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {path}")
        
        output.mkdir(parents=True, exist_ok=True)
        
        weights = self._load_weights(path)
        wrapped_weights = {}
        
        for name, tensor in weights.items():
            # MLX expects [out, in] layout
            wrapped_weights[name] = tensor.astype(np.float32)
        
        save_file(wrapped_weights, output / "adapters.safetensors")
        
        return WrapResult(
            output_path=str(output),
            wrapped_layers=len(wrapped_weights),
        )

    def smooth(self, adapter_path: str, output_path: str, strength: float = 0.1) -> SmoothResult:
        """Apply smoothing to adapter weights.
        
        Args:
            adapter_path: Path to adapter directory.
            output_path: Output path for smoothed adapter.
            strength: Smoothing strength (0.0 to 1.0).
            
        Returns:
            SmoothResult with variance reduction.
        """
        path = Path(adapter_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        
        if not path.exists():
            raise ValueError(f"Adapter path does not exist: {path}")
        
        output.mkdir(parents=True, exist_ok=True)
        
        weights = self._load_weights(path)
        smoothed_weights = {}
        original_variance = 0.0
        smoothed_variance = 0.0
        
        for name, tensor in weights.items():
            original_variance += np.var(tensor)
            
            # Apply smoothing: blend towards mean
            mean = np.mean(tensor)
            smoothed = tensor * (1 - strength) + mean * strength
            smoothed_weights[name] = smoothed.astype(np.float32)
            
            smoothed_variance += np.var(smoothed)
        
        save_file(smoothed_weights, output / "adapter_model.safetensors")
        
        # Copy config
        config_path = path / "adapter_config.json"
        if config_path.exists():
            (output / "adapter_config.json").write_text(
                config_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        
        variance_reduction = 1.0 - (smoothed_variance / original_variance) if original_variance > 0 else 0.0
        
        return SmoothResult(
            output_path=str(output),
            smoothed_layers=len(smoothed_weights),
            variance_reduction=float(variance_reduction),
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

    def merge(
        self,
        adapter_paths: list[str],
        output_dir: str,
        recommend_ensemble: bool = False,
    ) -> AdapterMergeResult:
        """Merge multiple LoRA adapters using geometric alignment.

        Uses the ONE correct merge method:
        1. Permutation alignment (re-basin neurons)
        2. Procrustes rotation (align weight spaces)
        3. Confidence-weighted blending

        Args:
            adapter_paths: List of paths to adapter directories.
            output_dir: Output directory for merged adapter.
            recommend_ensemble: Whether to compute ensemble routing recommendation.

        Returns:
            AdapterMergeResult with output path and merge metrics.
        """
        from modelcypher.core.domain.merging.lora_adapter_merger import LoRAAdapterMerger

        if len(adapter_paths) < 2:
            raise ValueError("At least 2 adapters are required for merging")

        # Resolve paths
        resolved_paths = []
        for adapter_path in adapter_paths:
            path = Path(adapter_path).expanduser().resolve()
            if not path.exists():
                raise ValueError(f"Adapter path does not exist: {path}")
            resolved_paths.append(path)

        output = Path(output_dir).expanduser().resolve()

        # Delegate to unified geometric merger
        report = LoRAAdapterMerger.merge(
            adapter_directories=resolved_paths,
            output_directory=output,
        )

        # Compute ensemble recommendation if requested
        ensemble_recommendation = None
        if recommend_ensemble:
            all_weights = [self._load_weights(p) for p in resolved_paths]
            common_keys = set(all_weights[0].keys())
            for weights in all_weights[1:]:
                common_keys &= set(weights.keys())
            ensemble_recommendation = self._compute_ensemble_recommendation(
                all_weights, common_keys, resolved_paths
            )

        return AdapterMergeResult(
            output_path=report.output_directory,
            merged_modules=report.total_merged_parameters,
            procrustes_error=report.mean_procrustes_error,
            permutation_quality=report.mean_permutation_quality,
            merge_confidence=report.merge_confidence,
            ensemble_recommendation=ensemble_recommendation,
        )

    def _compute_ensemble_recommendation(
        self,
        all_weights: list[dict],
        common_keys: set[str],
        paths: list[Path],
    ) -> dict:
        """Compute ensemble routing recommendation based on adapter characteristics."""
        # Compute per-adapter statistics
        adapter_stats = []
        for i, weights in enumerate(all_weights):
            total_norm = 0.0
            total_params = 0
            for key in common_keys:
                tensor = weights[key]
                total_norm += np.linalg.norm(tensor)
                total_params += tensor.size
            
            adapter_stats.append({
                "path": str(paths[i]),
                "total_norm": float(total_norm),
                "total_params": total_params,
            })
        
        # Compute weights based on inverse norm (smaller norm = more specialized)
        total_inv_norm = sum(1.0 / (s["total_norm"] + 1e-8) for s in adapter_stats)
        weights = []
        for s in adapter_stats:
            inv_norm = 1.0 / (s["total_norm"] + 1e-8)
            weight = inv_norm / total_inv_norm
            weights.append(round(weight, 4))
        
        return {
            "strategy": "weighted",
            "weights": weights,
            "adapter_stats": adapter_stats,
            "recommendation": "Use weighted ensemble with computed weights for balanced contribution",
        }
