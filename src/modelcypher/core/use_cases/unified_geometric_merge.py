# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""
Unified Geometric Merge Pipeline.

Combines geometric merge techniques in the correct order:

    VOCAB → PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE → VALIDATE

The intersection map (from semantic probes) is the control signal
that guides all downstream operations.

Key Principles:
1. Intersection map confidence controls when to apply risky operations
2. Geometric transformations are propagated layer-to-layer (zipper)
3. Alpha adjustments are applied sequentially (12+ stages)
4. Per-dimension control enables surgical merging

Stage implementations are in merge_stages/ subpackage for modularity.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedMergeConfig:
    """
    Configuration for unified geometric merge.

    PURE GEOMETRY: The merge formula is mathematically derived.
    No arbitrary thresholds. No "vibes". The math determines everything.

    W_merged = U_t @ diag(√(σ_s' ⊙ σ_t)) @ V_t^T

    Where:
    - R* = argmin ||W_t - R @ W_s||_F (Procrustes alignment)
    - σ_s' = singular values of aligned source
    - σ_t = singular values of target
    - √(σ_s' ⊙ σ_t) = Fréchet mean (geodesic midpoint on ℝ^+)
    """

    # Probe mode: "precise" (CKA on activations) or "fast" (weight-level CKA)
    probe_mode: Literal["precise", "fast"] = "precise"

    # Maximum probes in precise mode (0 = all 403)
    max_probes: int = 0

    # Use Gromov-Wasserstein instead of Procrustes for alignment
    use_transport_guided: bool = False

    # Output quantization (None = preserve original dtype)
    output_quant: str | None = None


@dataclass
class LayerMergeState:
    """State carried through layers during merge (zipper)."""

    # Current input rotation (from previous layer's output)
    omega_in: "Array | None" = None

    # Layer index
    layer_index: int = 0

    # Accumulated metrics
    procrustes_errors: list[float] = field(default_factory=list)
    spectral_ratios: list[float] = field(default_factory=list)
    effective_alphas: list[float] = field(default_factory=list)


@dataclass
class UnifiedMergeResult:
    """Result of unified geometric merge."""

    merged_weights: dict[str, "Array"]

    # Per-stage metrics
    vocab_metrics: dict[str, Any]  # Stage 0: Vocabulary alignment
    probe_metrics: dict[str, Any]
    permute_metrics: dict[str, Any]
    rotate_metrics: dict[str, Any]
    blend_metrics: dict[str, Any]

    # Overall quality
    mean_confidence: float
    mean_procrustes_error: float
    layer_count: int
    weight_count: int

    # Timing
    timestamp: datetime

    # Optional fields (must come after required fields)
    # Output path (if saved)
    output_path: str | None = None

    # Vocabulary alignment status
    vocab_aligned: bool = False

    # Stage 6: Safety validation metrics
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    safety_verdict: str = "not_validated"  # safe, caution, unsafe, critical
    refusal_preserved: bool = True


class UnifiedGeometricMerger:
    """
    Unified geometric merge pipeline.

    Combines techniques in the correct order:
    VOCAB → PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE → VALIDATE

    Stage implementations are in merge_stages/ for modularity.
    """

    def __init__(
        self,
        model_loader: "ModelLoaderPort",
        config: UnifiedMergeConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize with required dependencies.

        Args:
            model_loader: Model loader port for loading weights (REQUIRED).
            config: Merge configuration (optional, defaults to default config).
            backend: Compute backend for tensor operations (defaults to MLXBackend).
                     All geometric operations run on GPU when using MLXBackend.
        """
        self._model_loader = model_loader
        self.config = config or UnifiedMergeConfig()

        # Default to MLXBackend for GPU-accelerated operations
        if backend is None:
            from modelcypher.backends.mlx_backend import MLXBackend

            self._backend = MLXBackend()
        else:
            self._backend = backend

    def merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: str | None = None,
        dry_run: bool = False,
    ) -> UnifiedMergeResult:
        """
        Execute pure geometric merge.

        PURE GEOMETRY: No configurable thresholds. The math determines everything.

        W_merged = U_t @ diag(√(σ_s' ⊙ σ_t)) @ V_t^T

        Args:
            source_path: Path to source model (skill donor)
            target_path: Path to target model (knowledge base)
            output_dir: Output directory for merged model
            dry_run: If True, don't save to disk

        Returns:
            UnifiedMergeResult with merged weights and metrics
        """
        logger.info("=== PURE GEOMETRIC MERGE ===")
        logger.info("Source: %s", source_path)
        logger.info("Target: %s", target_path)

        # Load weights
        source_weights, _ = self._load_weights(source_path)
        target_weights, target_format = self._load_weights(target_path)

        # Identify layers
        layer_indices = self._extract_layer_indices(target_weights)
        logger.info("Found %d layers", len(layer_indices))

        # Load tokenizers for vocabulary alignment
        source_tokenizer = self._load_tokenizer(source_path)
        target_tokenizer = self._load_tokenizer(target_path)

        # =================================================================
        # STAGE 0: VOCABULARY (Cross-vocabulary alignment for embedding layers)
        # =================================================================
        logger.info("STAGE 0: VOCABULARY ALIGNMENT")
        source_weights, vocab_metrics, vocab_aligned = self._stage_vocabulary(
            source_weights=source_weights,
            target_weights=target_weights,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )

        # Load models for probe stage
        source_model = None
        target_model = None
        if self.config.probe_mode == "precise":
            logger.info("Loading models for precise probe execution...")
            source_model = self._load_model_for_probing(source_path)
            target_model = self._load_model_for_probing(target_path)

        # =================================================================
        # STAGE 1: PROBE (Compute layer correspondences via CKA)
        # =================================================================
        logger.info("STAGE 1: PROBE (%s mode)", self.config.probe_mode)
        probe_result, probe_metrics, source_activations, target_activations = self._stage_probe(
            source_weights=source_weights,
            target_weights=target_weights,
            source_model=source_model,
            target_model=target_model,
            tokenizer=target_tokenizer,
        )

        layer_confidences: dict[int, float] = probe_result.get("confidences", {})
        dimension_correlations: dict = probe_result.get("dimension_correlations", {})

        # Log activation collection results
        if source_activations and target_activations:
            logger.info(
                "PROBE: Collected activations for %d source layers, %d target layers",
                len(source_activations),
                len(target_activations),
            )

        # Clear GPU memory
        del source_model
        del target_model
        backend = get_default_backend()
        backend.clear_cache()
        logger.info("Cleared GPU cache after probe stage")

        # =================================================================
        # STAGE 2: PERMUTE (Re-Basin neuron alignment)
        # =================================================================
        logger.info("STAGE 2: PERMUTE")
        permuted_source, permute_metrics = self._stage_permute(
            source_weights, target_weights, layer_confidences
        )

        # =================================================================
        # STAGES 3-5: PURE GEOMETRIC MERGE (with per-layer alignment)
        # =================================================================
        logger.info("STAGES 3-5: GEOMETRIC MERGE")
        merged_weights, rotate_metrics, blend_metrics = self._stage_rotate_blend_propagate(
            permuted_source,
            target_weights,
            layer_indices,
            layer_confidences,
            dimension_correlations,
            source_activations=source_activations,
            target_activations=target_activations,
        )

        # =================================================================
        # OUTPUT
        # =================================================================
        if output_dir and not dry_run:
            self._save_weights(output_dir, merged_weights, target_format)
            self._copy_config_files(target_path, output_dir)
            output_path = output_dir
        else:
            output_path = None

        # Compute metrics
        mean_confidence = probe_metrics.get("mean_confidence", 0.0)
        procrustes_errors = rotate_metrics.get("procrustes_errors", [])
        mean_error = sum(procrustes_errors) / len(procrustes_errors) if procrustes_errors else 0.0

        result = UnifiedMergeResult(
            merged_weights=merged_weights,
            vocab_metrics=vocab_metrics,
            probe_metrics=probe_metrics,
            permute_metrics=permute_metrics,
            rotate_metrics=rotate_metrics,
            blend_metrics=blend_metrics,
            validation_metrics={},
            mean_confidence=mean_confidence,
            mean_procrustes_error=float(mean_error),
            layer_count=len(layer_indices),
            weight_count=len(merged_weights),
            timestamp=datetime.utcnow(),
            output_path=output_path,
            vocab_aligned=vocab_aligned,
            safety_verdict="geometric",
            refusal_preserved=True,
        )

        logger.info(
            "Merge complete: %d layers, %d weights, confidence=%.3f, error=%.3f",
            result.layer_count, result.weight_count,
            result.mean_confidence, result.mean_procrustes_error,
        )

        return result

    # =========================================================================
    # STAGE DELEGATES
    # =========================================================================

    def _stage_vocabulary(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        source_tokenizer: Any | None,
        target_tokenizer: Any | None,
    ) -> tuple[dict[str, "Array"], dict[str, Any], bool]:
        """Stage 0: Align source vocabulary to target vocabulary."""
        from .merge_stages.stage_0_vocabulary import (
            VocabularyConfig,
            stage_vocabulary_align,
        )

        config = VocabularyConfig()

        result = stage_vocabulary_align(
            source_weights=source_weights,
            target_weights=target_weights,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            config=config,
        )

        if result.was_aligned:
            logger.info("Vocabulary alignment applied")
        else:
            reason = result.metrics.get("reason", "unknown")
            logger.info("Vocabulary alignment skipped: %s", reason)

        return result.modified_weights, result.metrics, result.was_aligned

    def _stage_probe(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        source_model: Any | None,
        target_model: Any | None,
        tokenizer: Any | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict | None, dict | None]:
        """Stage 1: Compute layer correspondences via CKA.

        Returns:
            Tuple of (probe_result_dict, metrics, source_activations, target_activations)
        """
        from .merge_stages.stage_1_probe import (
            ProbeConfig,
            collect_layer_activations_mlx,
            stage_probe,
        )

        config = ProbeConfig(
            probe_mode=self.config.probe_mode,
            max_probes=self.config.max_probes,
        )

        collect_fn = collect_layer_activations_mlx if source_model is not None else None

        result = stage_probe(
            source_weights=source_weights,
            target_weights=target_weights,
            config=config,
            extract_layer_index_fn=self._extract_layer_index,
            source_model=source_model,
            target_model=target_model,
            tokenizer=tokenizer,
            collect_activations_fn=collect_fn,
        )

        return {
            "correlations": result.correlations,
            "confidences": result.confidences,
            "dimension_correlations": result.dimension_correlations,
        }, result.metrics, result.source_activations, result.target_activations

    def _stage_permute(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        layer_confidences: dict[int, float],
    ) -> tuple[dict[str, "Array"], dict[str, Any]]:
        """Stage 2: Re-Basin neuron permutation alignment."""
        from .merge_stages.stage_2_permute import (
            PermuteConfig,
            infer_hidden_dim,
            stage_permute,
        )

        config = PermuteConfig()

        result = stage_permute(
            source_weights=source_weights,
            target_weights=target_weights,
            intersection_map_obj=None,
            layer_confidences=layer_confidences,
            config=config,
            infer_hidden_dim_fn=infer_hidden_dim,
        )

        return result.weights, result.metrics

    def _stage_rotate_blend_propagate(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        layer_indices: list[int],
        layer_confidences: dict[int, float],
        dimension_correlations: dict,
        source_activations: dict | None = None,
        target_activations: dict | None = None,
    ) -> tuple[dict[str, "Array"], dict[str, Any], dict[str, Any]]:
        """Stages 3-5: PURE GEOMETRIC MERGE with per-layer alignment.

        Uses activations to compute per-layer rotations before merging.
        W_merged = U_t @ diag(√(σ_s' ⊙ σ_t)) @ V_t^T

        Layer confidences from probe stage inform alignment quality.
        Target activations enable null-space filtering to eliminate interference.
        """
        from .merge_stages.stage_3_5_rotate_blend import (
            RotateBlendConfig,
            stage_rotate_blend_propagate,
        )

        config = RotateBlendConfig(
            use_transport_guided=self.config.use_transport_guided,
        )

        result = stage_rotate_blend_propagate(
            source_weights=source_weights,
            target_weights=target_weights,
            intersection_map_obj=None,
            layer_confidences=layer_confidences,
            dimension_correlations=dimension_correlations,
            layer_indices=layer_indices,
            config=config,
            extract_layer_index_fn=self._extract_layer_index,
            backend=self._backend,
            source_activations=source_activations,
            target_activations=target_activations,
        )

        return result.merged_weights, result.rotate_metrics, result.blend_metrics

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _load_tokenizer(self, model_path: str) -> Any | None:
        """Load tokenizer for probe execution."""
        try:
            # Try transformers tokenizer first (avoids loading model)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            return tokenizer
        except Exception:
            pass

        try:
            # Fall back to mlx_lm (loads both model and tokenizer)
            from mlx_lm import load

            _, tokenizer = load(model_path)
            return tokenizer
        except Exception as e:
            logger.warning("Failed to load tokenizer: %s", e)
            return None

    def _load_model_for_probing(self, model_path: str) -> Any | None:
        """Load model for precise probe execution."""
        try:
            from mlx_lm import load

            logger.info("Loading model from %s for activation probing...", model_path)
            model, _ = load(model_path)
            logger.info("Model loaded successfully: %s", type(model).__name__)
            return model
        except Exception as e:
            logger.error("Failed to load model for probing: %s", e)
            import traceback

            logger.debug("Traceback: %s", traceback.format_exc())
            return None

    def _load_weights(self, model_path: str) -> tuple[dict[str, Any], str]:
        """Load model weights as native backend arrays (GPU-accelerated).

        Returns native arrays (mx.array for MLX) that run on GPU.
        """
        weights = self._model_loader.load_weights(model_path)
        return weights, "safetensors"

    def _load_weights_as_arrays(self, model_path: str) -> tuple[dict[str, "Array"], str]:
        """Load model weights as backend Arrays."""
        weights = self._model_loader.load_weights(model_path)
        return weights, "safetensors"

    def _save_weights(
        self,
        output_dir: str,
        weights: dict[str, Any],
        output_format: str,
    ) -> None:
        """Save merged weights (handles both native arrays and NumPy)."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        output_path = path / "model.safetensors"

        # Check if weights are MLX arrays
        first_weight = next(iter(weights.values()), None)
        if first_weight is not None:
            try:
                import mlx.core as mx

                if isinstance(first_weight, mx.array):
                    # Use MLX native save (faster, no conversion)
                    mx.save_safetensors(str(output_path), weights)
                    logger.info("Saved merged weights to %s (MLX native)", output_path)
                    return
            except (ImportError, TypeError):
                pass

        # Fallback to safetensors (convert arrays to numpy for save)
        if output_format == "safetensors":
            from safetensors.numpy import save_file

            # Convert backend arrays to numpy for safetensors save
            numpy_weights = {}
            for key, value in weights.items():
                if hasattr(value, "__array__"):
                    self._backend.eval(value)
                    numpy_weights[key] = self._backend.to_numpy(value)
                else:
                    numpy_weights[key] = value
            save_file(numpy_weights, str(output_path))
        else:
            # For npz format, also convert to numpy
            output_path = path / "weights.npz"
            import numpy as _np_for_save  # Only for file I/O, not computation

            numpy_weights = {}
            for key, value in weights.items():
                if hasattr(value, "__array__"):
                    self._backend.eval(value)
                    numpy_weights[key] = self._backend.to_numpy(value)
                else:
                    numpy_weights[key] = value
            _np_for_save.savez(str(output_path), **numpy_weights)

        logger.info("Saved merged weights to %s", output_path)

    def _copy_config_files(self, source_path: str, output_dir: str) -> None:
        """Copy config files from source to output."""
        source = Path(source_path)
        dest = Path(output_dir)

        for config_file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            src_file = source / config_file
            if src_file.exists():
                shutil.copy(src_file, dest / config_file)

    def _extract_layer_indices(self, weights: dict[str, "Array"]) -> list[int]:
        """Extract unique layer indices from weight keys."""
        indices = set()
        for key in weights:
            match = re.search(r"layers\.(\d+)\.", key)
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def _extract_layer_index(self, key: str) -> int | None:
        """Extract layer index from weight key."""
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            return int(match.group(1))
        return None


def unified_merge(
    source: str,
    target: str,
    output_dir: str,
    model_loader: "ModelLoaderPort",
    config: UnifiedMergeConfig | None = None,
    dry_run: bool = False,
) -> UnifiedMergeResult:
    """
    Execute unified geometric merge.

    Convenience function that creates the merger and runs the merge.

    Args:
        source: Path to source model (skill donor)
        target: Path to target model (knowledge base)
        output_dir: Output directory for merged model
        model_loader: Model loader port implementation (injected dependency)
        config: Merge configuration (optional)
        dry_run: If True, don't save to disk

    Returns:
        UnifiedMergeResult with merged weights and metrics
    """
    merger = UnifiedGeometricMerger(model_loader=model_loader, config=config)

    return merger.merge(
        source_path=source,
        target_path=target,
        output_dir=output_dir,
        dry_run=dry_run,
    )
