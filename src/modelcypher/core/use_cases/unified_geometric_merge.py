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
import time
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


@dataclass
class CrossArchitectureInfo:
    """Information about cross-architecture model pair."""

    is_cross_architecture: bool = False
    source_layer_count: int = 0
    target_layer_count: int = 0
    source_hidden_dim: int = 0
    target_hidden_dim: int = 0
    layer_correspondence: dict[int, int] | None = None


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
        use_full_geometry: bool = True,
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
            use_full_geometry: If True, use GeometricMergeOrchestrator with ALL 84 geometry files

        Returns:
            UnifiedMergeResult with merged weights and metrics
        """
        logger.info("=== PURE GEOMETRIC MERGE ===")
        logger.info("Source: %s", source_path)
        logger.info("Target: %s", target_path)

        if use_full_geometry:
            return self._merge_with_full_geometry(
                source_path, target_path, output_dir, dry_run
            )

        # Load weights (CPU first to reduce GPU memory pressure during merge)
        source_weights, _ = self._load_weights_cpu(source_path)
        target_weights, target_format = self._load_weights_cpu(target_path)

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
        source_weights, vocab_metrics, vocab_aligned, vocab_alignment_map = self._stage_vocabulary(
            source_weights=source_weights,
            target_weights=target_weights,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        self._require_vocab_phase_lock(vocab_metrics, vocab_aligned)

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
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            alignment_map=vocab_alignment_map,
        )

        layer_confidences: dict[int, float] = probe_result.get("confidences", {})
        dimension_correlations: dict = probe_result.get("dimension_correlations", {})
        intersection_map_obj = probe_result.get("intersection_map")
        probe_failed = bool(probe_metrics.get("probe_failed"))
        perfect_alignment = bool(probe_metrics.get("perfect_alignment"))

        if probe_failed:
            min_cka = probe_metrics.get("min_cka", 0.0)
            mean_cka = probe_metrics.get("mean_cka", 0.0)
            raise RuntimeError(
                "PROBE SIGNAL: Alignment signals missing (mean_cka=%.4f, min_cka=%.4f). "
                "Phase lock is required before merge."
                % (mean_cka, min_cka)
            )

        if not perfect_alignment:
            min_cka = probe_metrics.get("min_cka", 0.0)
            mean_cka = probe_metrics.get("mean_cka", 0.0)
            raise RuntimeError(
                "PROBE BAROMETER: Alignment not phase-locked (mean_cka=%.4f, min_cka=%.4f). "
                "Resolve alignment before merge."
                % (mean_cka, min_cka)
            )

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
            source_weights, target_weights, layer_confidences, intersection_map_obj
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
            intersection_map_obj=intersection_map_obj,
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

    def _merge_with_full_geometry(
        self,
        source_path: str,
        target_path: str,
        output_dir: str | None = None,
        dry_run: bool = False,
    ) -> "UnifiedMergeResult":
        """
        Execute merge using GeometricMergeOrchestrator with ALL 84 geometry files.

        This is the comprehensive merge that uses:
        - intrinsic_dimension: Per-layer intrinsic dimension
        - manifold_curvature: Curvature for geodesic interpolation
        - shared_subspace_projector: CCA-based shared dimension discovery
        - relative_representation: Anchor-based dimension-agnostic alignment
        - fisher_blending: Importance-weighted blending
        - dimension_blender: Per-dimension alpha computation
        - null_space_filter: Interference elimination
        - dare_sparsity: Optional sparsification
        - ... and 70+ more geometry files

        Higher dimensions contain lower dimensions (1D ⊂ 2D ⊂ 3D ⊂ ... ⊂ nD).
        We analyze and blend at EVERY dimension level.
        """
        from .geometric_merge_orchestrator import GeometricMergeOrchestrator

        logger.info("=== FULL GEOMETRY MERGE (84 files) ===")
        logger.info("Source: %s", source_path)
        logger.info("Target: %s", target_path)
        logger.info("Backend: %s", type(self._backend).__name__)

        # Load weights
        source_weights, _ = self._load_weights(source_path)
        target_weights, target_format = self._load_weights(target_path)

        # Load tokenizers
        source_tokenizer = self._load_tokenizer(source_path)
        target_tokenizer = self._load_tokenizer(target_path)

        # Stage 0: Vocabulary alignment
        logger.info("STAGE 0: VOCABULARY ALIGNMENT")
        stage_start = time.perf_counter()
        source_weights, vocab_metrics, vocab_aligned, vocab_alignment_map = self._stage_vocabulary(
            source_weights=source_weights,
            target_weights=target_weights,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        self._require_vocab_phase_lock(vocab_metrics, vocab_aligned)
        logger.info(
            "STAGE 0: VOCABULARY ALIGNMENT completed in %.2fs",
            time.perf_counter() - stage_start,
        )

        # Collect activations if models can be loaded
        source_activations = None
        target_activations = None
        source_model = None
        target_model = None

        if self.config.probe_mode == "precise":
            logger.info("Loading models for activation collection...")
            load_start = time.perf_counter()
            source_model = self._load_model_for_probing(source_path)
            target_model = self._load_model_for_probing(target_path)
            logger.info(
                "STAGE 1: Model load completed in %.2fs",
                time.perf_counter() - load_start,
            )

            if source_model and target_model and source_tokenizer and target_tokenizer:
                from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
                from modelcypher.core.domain.vocabulary.alignment_map import AlignmentQuality
                from .merge_stages.stage_1_probe import (
                    _encode_probe_ids,
                    build_token_id_map,
                    collect_layer_activations_mlx,
                    map_token_ids,
                )

                probes = UnifiedAtlasInventory.all_probes()
                max_probes = self.config.max_probes if self.config.max_probes > 0 else len(probes)

                source_activations = {}
                target_activations = {}
                token_id_map = None
                if vocab_alignment_map is not None:
                    token_id_map = build_token_id_map(
                        vocab_alignment_map,
                        min_confidence=1.0,
                        min_size=0,
                        allowed_qualities={AlignmentQuality.EXACT},
                    )
                    if token_id_map:
                        logger.info(
                            "STAGE 1: Using aligned token map for probes (%d tokens).",
                            len(token_id_map),
                        )

                for i, probe in enumerate(probes[:max_probes]):
                    try:
                        probe_text = None
                        source_ids: list[int] | None = None
                        target_ids: list[int] | None = None
                        for candidate in probe.support_texts or []:
                            if not candidate or len(candidate.strip()) < 2:
                                continue
                            if token_id_map is None:
                                probe_text = candidate
                                break
                            candidate_source_ids = _encode_probe_ids(
                                source_tokenizer, candidate, add_special_tokens=False
                            )
                            candidate_target_ids = map_token_ids(
                                candidate_source_ids, token_id_map
                            )
                            if candidate_target_ids is None:
                                continue
                            probe_text = candidate
                            source_ids = candidate_source_ids
                            target_ids = candidate_target_ids
                            break

                        if probe_text is None:
                            continue

                        src_acts = collect_layer_activations_mlx(
                            source_model,
                            source_tokenizer,
                            probe_text,
                            token_ids=source_ids,
                        )
                        tgt_acts = collect_layer_activations_mlx(
                            target_model,
                            target_tokenizer,
                            probe_text,
                            token_ids=target_ids,
                        )

                        for layer_idx, act in src_acts.items():
                            if layer_idx not in source_activations:
                                source_activations[layer_idx] = []
                            source_activations[layer_idx].append(act)

                        for layer_idx, act in tgt_acts.items():
                            if layer_idx not in target_activations:
                                target_activations[layer_idx] = []
                            target_activations[layer_idx].append(act)
                    except Exception:
                        continue

                    if (i + 1) % 20 == 0:
                        logger.info("Collected activations from %d/%d probes", i + 1, max_probes)

                logger.info(
                    "Collected activations: %d source layers, %d target layers",
                    len(source_activations),
                    len(target_activations),
                )

        # Clear model memory
        del source_model
        del target_model
        self._backend.clear_cache()

        # Create orchestrator and analyze geometry
        logger.info("ANALYZING FULL GEOMETRY...")
        analyze_start = time.perf_counter()
        orchestrator = GeometricMergeOrchestrator(backend=self._backend)
        geometry = orchestrator.analyze_merge(
            source_weights=source_weights,
            target_weights=target_weights,
            source_activations=source_activations,
            target_activations=target_activations,
            tokenizer=target_tokenizer,
        )
        logger.info(
            "ANALYZING FULL GEOMETRY completed in %.2fs",
            time.perf_counter() - analyze_start,
        )

        if geometry.overall_cka != 1.0:
            logger.info(
                "PROBE BAROMETER: Overall CKA=%.4f. "
                "Phase-lock alignment will resolve per-layer alignment before merging.",
                geometry.overall_cka,
            )

        # Execute merge using geometry
        logger.info("EXECUTING MERGE...")
        merge_start = time.perf_counter()
        merged_weights, merge_metrics = orchestrator.merge_weights(
            source_weights=source_weights,
            target_weights=target_weights,
            geometry=geometry,
            extract_layer_index_fn=self._extract_layer_index,
            checkpoint_dir=output_dir,
        )
        logger.info(
            "EXECUTING MERGE completed in %.2fs",
            time.perf_counter() - merge_start,
        )

        # Detect target quantization and requantize merged weights to match
        target_is_quantized = any(
            k.endswith(".scales") or k.endswith(".biases")
            for k in target_weights.keys()
        )

        if target_is_quantized:
            from .quantization_utils import (
                QuantizationHint,
                quantization_config_from_payload,
                requantize_weights,
            )
            import json
            from pathlib import Path

            # Read target config to get quantization params
            config_path = Path(target_path) / "config.json"
            quant_hint = QuantizationHint(bits=4, group_size=64, mode="affine")  # Default
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config_data = json.load(f)
                    quant_config = quantization_config_from_payload(config_data)
                    if quant_config and quant_config.default:
                        quant_hint = quant_config.default
                        logger.info(
                            "Detected target quantization: %d-bit, group_size=%d",
                            quant_hint.bits,
                            quant_hint.group_size,
                        )
                except Exception as e:
                    logger.warning("Could not read target config for quantization: %s", e)

            logger.info("Requantizing merged weights to match target format...")
            merged_weights = requantize_weights(
                merged_weights,
                self._backend,
                quant_hint,
            )
            logger.info("Requantization complete: %d weights", len(merged_weights))

        # Save if requested
        if output_dir and not dry_run:
            self._save_weights(output_dir, merged_weights, target_format)
            self._copy_config_files(target_path, output_dir)
            output_path = output_dir
        else:
            output_path = None

        # Build result
        layer_indices = self._extract_layer_indices(target_weights)
        result = UnifiedMergeResult(
            merged_weights=merged_weights,
            vocab_metrics=vocab_metrics,
            probe_metrics={
                "overall_cka": geometry.overall_cka,
                "mean_intrinsic_dim": geometry.mean_intrinsic_dimension,
                "mean_shared_dim": geometry.mean_shared_dimension,
            },
            permute_metrics={},
            rotate_metrics=merge_metrics,
            blend_metrics=merge_metrics,
            validation_metrics={},
            mean_confidence=geometry.overall_cka,
            mean_procrustes_error=0.0,
            layer_count=len(layer_indices),
            weight_count=len(merged_weights),
            timestamp=datetime.utcnow(),
            output_path=output_path,
            vocab_aligned=vocab_aligned,
            safety_verdict="geometric",
            refusal_preserved=geometry.refusal_preserved,
        )

        logger.info(
            "FULL GEOMETRY MERGE COMPLETE: %d layers, %d weights, CKA=%.4f",
            result.layer_count,
            result.weight_count,
            geometry.overall_cka,
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
    ) -> tuple[dict[str, "Array"], dict[str, Any], bool, Any | None]:
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

        return (
            result.modified_weights,
            result.metrics,
            result.was_aligned,
            result.alignment_map,
        )

    def _stage_probe(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        source_model: Any | None,
        target_model: Any | None,
        source_tokenizer: Any | None,
        target_tokenizer: Any | None,
        alignment_map: Any | None = None,
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

        collect_fn = (
            collect_layer_activations_mlx
            if source_model is not None and source_tokenizer and target_tokenizer
            else None
        )

        result = stage_probe(
            source_weights=source_weights,
            target_weights=target_weights,
            config=config,
            extract_layer_index_fn=self._extract_layer_index,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            collect_activations_fn=collect_fn,
            alignment_map=alignment_map,
        )

        return {
            "correlations": result.correlations,
            "confidences": result.confidences,
            "dimension_correlations": result.dimension_correlations,
            "intersection_map": result.intersection_map,
        }, result.metrics, result.source_activations, result.target_activations

    def _stage_permute(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        layer_confidences: dict[int, float],
        intersection_map_obj: Any | None,
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
            intersection_map_obj=intersection_map_obj,
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
        intersection_map_obj: Any | None,
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
            intersection_map_obj=intersection_map_obj,
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

    def _load_weights_cpu(self, model_path: str) -> tuple[dict[str, Any], str]:
        """Load model weights as CPU arrays to reduce GPU memory pressure."""
        weights = self._model_loader.load_weights_as_numpy(model_path)
        return weights, "safetensors"

    def _load_weights_as_arrays(self, model_path: str) -> tuple[dict[str, "Array"], str]:
        """Load model weights as backend Arrays."""
        weights = self._model_loader.load_weights(model_path)
        return weights, "safetensors"

    def _require_vocab_phase_lock(
        self, vocab_metrics: dict[str, Any], vocab_aligned: bool
    ) -> None:
        if not vocab_aligned:
            raise RuntimeError(
                "Vocabulary alignment was not applied. Phase lock is required before merge."
            )
        binary = vocab_metrics.get("binary_alignment", {})
        vocab = vocab_metrics.get("vocab_phase_lock", {})
        if not binary or not vocab:
            raise RuntimeError(
                "Vocabulary alignment metrics missing; cannot confirm phase lock."
            )
        for key, entry in binary.items():
            if not entry.get("phase_locked"):
                raise RuntimeError(
                    f"Binary phase lock missing for {key}; aborting merge."
                )
        for key, entry in vocab.items():
            if not entry.get("phase_locked"):
                raise RuntimeError(
                    f"Vocabulary phase lock missing for {key}; aborting merge."
                )

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

    def _detect_cross_architecture(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
    ) -> CrossArchitectureInfo:
        """
        Detect if models have different architectures (layer count or hidden dim).

        Cross-architecture merging requires layer correspondence mapping and
        potentially dimension projection. This method detects the mismatch
        and returns information needed for alignment.

        Args:
            source_weights: Source model weights
            target_weights: Target model weights

        Returns:
            CrossArchitectureInfo with detection results
        """
        # Extract layer counts
        source_layers = self._extract_layer_indices(source_weights)
        target_layers = self._extract_layer_indices(target_weights)

        layer_mismatch = len(source_layers) != len(target_layers)

        # Check dimension mismatch from representative weight matrices
        source_hidden_dim = 0
        target_hidden_dim = 0

        # Look for q_proj weights as they reflect hidden dimension
        for key in source_weights:
            if ".q_proj.weight" in key or ".self_attn.q_proj.weight" in key:
                source_hidden_dim = source_weights[key].shape[-1]
                break

        for key in target_weights:
            if ".q_proj.weight" in key or ".self_attn.q_proj.weight" in key:
                target_hidden_dim = target_weights[key].shape[-1]
                break

        # Fallback to any 2D weight if q_proj not found
        if source_hidden_dim == 0:
            for key in source_weights:
                w = source_weights[key]
                if w.ndim == 2 and "layers.0." in key:
                    source_hidden_dim = w.shape[-1]
                    break

        if target_hidden_dim == 0:
            for key in target_weights:
                w = target_weights[key]
                if w.ndim == 2 and "layers.0." in key:
                    target_hidden_dim = w.shape[-1]
                    break

        dim_mismatch = source_hidden_dim != target_hidden_dim and source_hidden_dim > 0 and target_hidden_dim > 0

        is_cross_arch = layer_mismatch or dim_mismatch

        if is_cross_arch:
            logger.info(
                "Cross-architecture detected: source=%d layers/%d dim, target=%d layers/%d dim",
                len(source_layers),
                source_hidden_dim,
                len(target_layers),
                target_hidden_dim,
            )

        return CrossArchitectureInfo(
            is_cross_architecture=is_cross_arch,
            source_layer_count=len(source_layers),
            target_layer_count=len(target_layers),
            source_hidden_dim=source_hidden_dim,
            target_hidden_dim=target_hidden_dim,
            layer_correspondence=None,  # Computed later if needed
        )


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
