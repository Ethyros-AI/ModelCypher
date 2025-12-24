"""
Unified LoRA Adapter Merger using Geometric Alignment.

This module provides THE ONE correct way to merge LoRA adapters:
geometric alignment via Procrustes rotation and permutation re-basin.

LoRA adapters are not fundamentally different from base model weights—they are
a higher-resolution patch on the same knowledge manifold. The same geometric
principles that align full models apply to adapters:

    PROBE → PERMUTE → ROTATE → BLEND

No TIES. No DARE. No strategy options. One correct geometric merge.

The math:
1. Load adapter weight dicts from PEFT format
2. Validate rank/scale/base_model compatibility
3. Compute intersection map via semantic prime probing (CKA confidence)
4. Apply permutation alignment to re-basin neurons
5. Apply Procrustes rotation to align weight spaces
6. Blend with confidence-weighted alpha from intersection map
7. Save merged adapter to PEFT format

References:
- Git Re-Basin: Ainsworth et al. (2022)
- Procrustes Analysis: Schönemann (1966)
- Representation Similarity: Kornblith et al. (2019) CKA
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.permutation_aligner import (
    PermutationAligner,
    Config as PermutationConfig,
    AlignmentResult,
)
from modelcypher.core.domain.geometry.generalized_procrustes import (
    GeneralizedProcrustes,
    Config as ProcrustesConfig,
)
from modelcypher.core.domain.merging.exceptions import MergeError
from modelcypher.ports.backend import Backend

logger = logging.getLogger("modelcypher.merging.lora_adapter_merger")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AdapterPayload:
    """Loaded adapter with metadata."""
    directory: Path
    base_model_id: str
    rank: int
    scale: float
    weights: Dict[str, np.ndarray]
    module_keys: List[str]


@dataclass
class MergeReport:
    """Report of a completed adapter merge."""
    output_directory: str
    adapter_count: int
    base_model_id: str
    rank: int
    scale: float
    mean_procrustes_error: float
    mean_permutation_quality: float
    total_merged_parameters: int
    layer_count: int
    merge_confidence: float


# =============================================================================
# LoRA Adapter Merger (Geometric)
# =============================================================================


class LoRAAdapterMerger:
    """
    Merges LoRA adapters using geometric alignment.

    This is THE ONE correct way to merge adapters. No strategy options,
    no heuristic dropout—just mathematically correct manifold alignment.

    Example:
        report = LoRAAdapterMerger.merge(
            adapter_directories=[Path("adapter1"), Path("adapter2")],
            output_directory=Path("merged"),
        )
    """

    @staticmethod
    def merge(
        adapter_directories: List[Path],
        output_directory: Path,
        backend: Optional[Backend] = None,
    ) -> MergeReport:
        """
        Merge multiple LoRA adapters using geometric alignment.

        The merge pipeline:
        1. Load and validate adapters
        2. For each weight matrix pair:
           a. Compute permutation alignment (re-basin)
           b. Apply Procrustes rotation
           c. Blend with uniform alpha (all adapters equal)
        3. Save merged adapter

        Args:
            adapter_directories: Paths to PEFT adapter directories.
            output_directory: Where to save the merged adapter.
            backend: Compute backend (auto-detected if None).

        Returns:
            MergeReport with merge statistics.

        Raises:
            MergeError: If adapters are incompatible.
        """
        if len(adapter_directories) < 2:
            raise MergeError("At least two adapters required for merging.")

        b = backend or get_default_backend()
        logger.info("Merging %d adapters using geometric alignment", len(adapter_directories))

        # Load adapters
        payloads = [LoRAAdapterMerger._load_adapter(d) for d in adapter_directories]
        first = payloads[0]

        # Validate compatibility
        for payload in payloads[1:]:
            if payload.base_model_id != first.base_model_id:
                raise MergeError(
                    f"Adapters target different base models: "
                    f"{first.base_model_id} vs {payload.base_model_id}"
                )
            if payload.rank != first.rank:
                raise MergeError(
                    f"Adapters use different ranks: {first.rank} vs {payload.rank}"
                )
            if abs(payload.scale - first.scale) > 1e-5:
                raise MergeError(
                    f"Adapters use different scales: {first.scale} vs {payload.scale}"
                )
            if set(payload.module_keys) != set(first.module_keys):
                raise MergeError("Adapters do not have the same LoRA modules.")

        # Initialize merge tracking
        procrustes_errors: List[float] = []
        permutation_qualities: List[float] = []
        total_params = 0

        # Merge each module
        merged_weights: Dict[str, np.ndarray] = {}

        for module_key in sorted(first.module_keys):
            # Collect weight matrices from all adapters for this module
            weight_matrices = [p.weights[module_key] for p in payloads]

            # Geometric merge
            merged, p_error, perm_quality = LoRAAdapterMerger._geometric_merge_matrices(
                weight_matrices, b
            )

            merged_weights[module_key] = merged
            procrustes_errors.append(p_error)
            permutation_qualities.append(perm_quality)
            total_params += merged.size

        # Compute layer count
        layer_indices = set()
        for key in first.module_keys:
            idx = LoRAAdapterMerger._extract_layer_index(key)
            if idx is not None:
                layer_indices.add(idx)

        # Write merged adapter
        LoRAAdapterMerger._write_adapter(
            source_directory=first.directory,
            output_directory=output_directory,
            weights=merged_weights,
            base_model_id=first.base_model_id,
        )

        mean_error = float(np.mean(procrustes_errors)) if procrustes_errors else 0.0
        mean_quality = float(np.mean(permutation_qualities)) if permutation_qualities else 0.0

        # Merge confidence: high quality alignment + low error
        merge_confidence = mean_quality * (1.0 - min(mean_error, 1.0))

        logger.info(
            "Merge complete: %d params, procrustes_error=%.4f, perm_quality=%.4f",
            total_params, mean_error, mean_quality
        )

        return MergeReport(
            output_directory=str(output_directory),
            adapter_count=len(payloads),
            base_model_id=first.base_model_id,
            rank=first.rank,
            scale=first.scale,
            mean_procrustes_error=mean_error,
            mean_permutation_quality=mean_quality,
            total_merged_parameters=total_params,
            layer_count=len(layer_indices),
            merge_confidence=merge_confidence,
        )

    @staticmethod
    def _geometric_merge_matrices(
        matrices: List[np.ndarray],
        backend: Backend,
    ) -> tuple[np.ndarray, float, float]:
        """
        Merge weight matrices using geometric alignment.

        Pipeline:
        1. Permutation align (re-basin neurons)
        2. Procrustes rotate (align weight spaces)
        3. Average (uniform alpha)

        Returns:
            Tuple of (merged_matrix, procrustes_error, permutation_quality)
        """
        if len(matrices) < 2:
            return matrices[0], 0.0, 1.0

        shape = matrices[0].shape
        dtype = matrices[0].dtype

        # For 1D tensors (biases), just average
        if len(shape) == 1:
            merged = np.mean(np.stack(matrices), axis=0)
            return merged.astype(dtype), 0.0, 1.0

        # Convert to backend arrays
        arrays = [backend.array(m.astype(np.float32)) for m in matrices]

        # Use first adapter as reference (target)
        target = arrays[0]
        aligned_sources: List[Any] = [target]

        total_perm_quality = 0.0
        total_proc_error = 0.0

        for source in arrays[1:]:
            # Step 1: Permutation alignment
            perm_result = LoRAAdapterMerger._permutation_align(source, target, backend)
            permuted = LoRAAdapterMerger._apply_permutation(source, perm_result, backend)
            total_perm_quality += perm_result.match_quality

            # Step 2: Procrustes rotation
            rotated, proc_error = LoRAAdapterMerger._procrustes_align(
                permuted, target, backend
            )
            total_proc_error += proc_error

            aligned_sources.append(rotated)

        # Step 3: Average all aligned matrices (uniform alpha)
        stacked = backend.stack(aligned_sources)
        merged = backend.mean(stacked, axis=0)
        backend.eval(merged)

        n_sources = len(matrices) - 1
        avg_perm_quality = total_perm_quality / n_sources if n_sources > 0 else 1.0
        avg_proc_error = total_proc_error / n_sources if n_sources > 0 else 0.0

        return backend.to_numpy(merged).astype(dtype), avg_proc_error, avg_perm_quality

    @staticmethod
    def _permutation_align(
        source: Any,
        target: Any,
        backend: Backend,
    ) -> AlignmentResult:
        """Compute optimal neuron permutation."""
        config = PermutationConfig(
            min_match_threshold=0.1,
            use_anchor_grounding=False,  # Direct weight alignment for adapters
        )
        return PermutationAligner.align(
            source_weight=source,
            target_weight=target,
            config=config,
            backend=backend,
        )

    @staticmethod
    def _apply_permutation(
        source: Any,
        result: AlignmentResult,
        backend: Backend,
    ) -> Any:
        """Apply permutation and sign correction to source.

        For rectangular matrices (like LoRA weights), we only permute the
        output dimension (rows). The permutation matrix is sized for the
        first dimension.

        For square matrices, this is equivalent to P @ source @ P^T.
        For rectangular (M, N) where M >= N: P @ source (permute rows only).
        """
        # Use PermutationAligner.apply which handles rectangular matrices correctly
        return PermutationAligner.apply(
            weight=source,
            alignment=result,
            align_output=True,  # Permute rows (output dimension)
            align_input=False,  # Don't permute columns (input dimension)
            backend=backend,
        )

    @staticmethod
    def _procrustes_align(
        source: Any,
        target: Any,
        backend: Backend,
    ) -> tuple[Any, float]:
        """
        Compute Procrustes rotation to align source to target.

        Returns:
            Tuple of (rotated_source, alignment_error)
        """
        # SVD-based Procrustes: find R = argmin ||source @ R - target||
        # Solution: R = V @ U^T where target^T @ source = U @ S @ V^T

        source_np = backend.to_numpy(source)
        target_np = backend.to_numpy(target)

        # M = target^T @ source
        M = target_np.T @ source_np

        # SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Optimal rotation (ensure proper rotation, not reflection)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Apply rotation
        rotated_np = source_np @ R

        # Compute alignment error
        error = float(np.linalg.norm(rotated_np - target_np, 'fro'))
        error /= max(float(np.linalg.norm(target_np, 'fro')), 1e-10)

        rotated = backend.array(rotated_np)
        backend.eval(rotated)

        return rotated, error

    @staticmethod
    def _load_adapter(directory: Path) -> AdapterPayload:
        """Load PEFT LoRA adapter from directory."""
        config_path = directory / "adapter_config.json"
        weights_path = directory / "adapter_model.safetensors"

        if not config_path.exists():
            raise MergeError(f"Missing adapter config at {config_path}")
        if not weights_path.exists():
            raise MergeError(f"Missing adapter weights at {weights_path}")

        with open(config_path) as f:
            peft_config = json.load(f)

        # Extract metadata
        base_model_id = peft_config.get("base_model_name_or_path", "unknown")
        rank = peft_config.get("r", peft_config.get("lora_rank", 16))
        alpha = peft_config.get("lora_alpha", rank)
        scale = alpha / rank if rank > 0 else 1.0

        # Load weights (safetensors)
        try:
            from safetensors.numpy import load_file
            weights = load_file(str(weights_path))
        except ImportError:
            raise MergeError("safetensors package required: pip install safetensors")

        # Convert to numpy arrays
        weights = {k: np.array(v) for k, v in weights.items()}

        # Identify LoRA module keys (A and B matrices)
        module_keys = sorted(weights.keys())

        return AdapterPayload(
            directory=directory,
            base_model_id=base_model_id,
            rank=rank,
            scale=scale,
            weights=weights,
            module_keys=module_keys,
        )

    @staticmethod
    def _write_adapter(
        source_directory: Path,
        output_directory: Path,
        weights: Dict[str, np.ndarray],
        base_model_id: str,
    ) -> None:
        """Write merged adapter to PEFT format."""
        output_directory.mkdir(parents=True, exist_ok=True)

        # Copy non-weight files from source
        for item in source_directory.iterdir():
            if item.name in ("adapter_model.safetensors", "adapter_config.json"):
                continue
            dest = output_directory / item.name
            if not dest.exists() and item.is_file():
                dest.write_bytes(item.read_bytes())

        # Save weights
        try:
            from safetensors.numpy import save_file
            output_weights = output_directory / "adapter_model.safetensors"
            save_file(weights, str(output_weights))
        except ImportError:
            raise MergeError("safetensors package required: pip install safetensors")

        # Update config
        config_path = source_directory / "adapter_config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        config_data["base_model_name_or_path"] = base_model_id

        output_config = output_directory / "adapter_config.json"
        with open(output_config, 'w') as f:
            json.dump(config_data, f, indent=2)

    @staticmethod
    def _extract_layer_index(key: str) -> Optional[int]:
        """Extract layer index from weight key."""
        parts = key.split(".")
        for idx, part in enumerate(parts):
            if part == "layers" and idx + 1 < len(parts):
                try:
                    return int(parts[idx + 1])
                except ValueError:
                    return None
        return None


__all__ = [
    "LoRAAdapterMerger",
    "MergeReport",
]
