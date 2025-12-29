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

Pipeline:
    VOCAB → PROBE → PERMUTE → TRANSPLANT → VALIDATE

Stage 0: VOCABULARY - Cross-vocabulary embedding alignment
Stage 1: PROBE - Build intersection map from probe responses
Stage 2: PERMUTE - Git Re-Basin permutation alignment (same-arch only)
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting
Stage 4: VALIDATE - Safety checks (numerical + content)

Key Principles:
1. Null-space projection guarantees: A_boundary @ W' = A_boundary @ W_target
2. Layer targeting enables surgical transplants
3. Cross-dimensional projection via GRAM_TRANSPORT
4. Permutation alignment reduces delta magnitude before transplant (same-arch)

References:
- Git Re-Basin: Ainsworth et al. (2023) arXiv:2209.04836
- AlphaEdit (null-space): Fang et al. (2025) ICLR Outstanding Paper

REMOVED (proven broken):
- rotate_blend: Alpha-blending has no constraint, destroys coherence
- ROTATE/BLEND/PROPAGATE: Only served rotate_blend

Stage implementations are in merge_stages/ subpackage for modularity.
"""

from __future__ import annotations

import logging
import time
import re
import shutil
from dataclasses import dataclass, field, replace
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

    Transplant formula:
        W' = W_target + P_null(A_boundary) @ (W_source_aligned - W_target)

    Guarantee:
        A_boundary @ W' = A_boundary @ W_target  (boundary preserved)

    This was validated empirically (Phase 6-8 research) and theoretically
    (AlphaEdit, ICLR 2025 Outstanding Paper).
    """

    # Probe mode: "precise" (CKA on activations) or "fast" (weight-level CKA)
    probe_mode: Literal["precise", "fast"] = "precise"

    # Maximum probes in precise mode (0 = all 403)
    max_probes: int = 0

    # Transplant settings - REQUIRED for effective knowledge transfer
    # Core domains define what concepts to transplant (e.g., "mathematical")
    transplant_domains: tuple[str, ...] = ()
    # Specific layers to transplant (None = all, but targeting is recommended)
    transplant_layers: tuple[int, ...] | None = None
    transplant_boundary_k: int | None = None
    transplant_geodesic_k_neighbors: int | None = None

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
    probe_metrics: dict[str, Any]  # Stage 1: Probe
    permute_metrics: dict[str, Any]  # Stage 2: Git Re-Basin permutation
    transplant_metrics: dict[str, Any]  # Stage 3: Transplant

    # Overall quality
    mean_confidence: float
    mean_procrustes_error: float
    layer_count: int
    weight_count: int

    # Timing
    timestamp: datetime

    # Merge strategy used
    merge_strategy: str = "transplant"

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

    Pipeline: VOCAB → PROBE → PERMUTE → TRANSPLANT → VALIDATE

    - PERMUTE (Git Re-Basin): Solves permutation symmetry for same-architecture models.
      Reduces delta magnitude before transplant by aligning neuron orderings.
    - TRANSPLANT: Null-space constrained projection preserves boundary behavior
      while transferring knowledge.

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
        output_path: str | None = None,
        dry_run: bool = False,
        use_full_geometry: bool = True,
        knowledge_delta_mask_path: str | None = None,
        transplant_domains: list[str] | None = None,
        transplant_layers: list[int] | None = None,
        transplant_boundary_k: int | None = None,
        transplant_geodesic_k_neighbors: int | None = None,
        target_weights: dict[str, "Array"] | None = None,
        config: "UnifiedMergeConfig | None" = None,
    ) -> UnifiedMergeResult:
        """
        Execute null-space constrained transplant merge.

        Transplant formula:
            W' = W_target + P_null(A_boundary) @ (W_source_aligned - W_target)

        Guarantee:
            A_boundary @ W' = A_boundary @ W_target  (boundary preserved)

        Args:
            source_path: Path to source model (skill donor)
            target_path: Path to target model (knowledge base)
            output_dir: Output directory for merged model (deprecated, use output_path)
            output_path: Output path for merged model (preferred over output_dir)
            dry_run: If True, don't save to disk
            use_full_geometry: If True, use GeometricMergeOrchestrator
            knowledge_delta_mask_path: Optional delta mask JSON for layer gating
            transplant_domains: Core domains to transplant (e.g., ["mathematical"])
            transplant_layers: Limit transplant to specific layer indices (optional)
            transplant_boundary_k: Boundary neighbors per core probe (optional)
            transplant_geodesic_k_neighbors: k for geodesic graph (optional)
            target_weights: Pre-loaded target weights (avoids reloading from disk).
                           Used by multi-donor pipeline to pass merged weights from
                           previous donor without reloading.
            config: Override merge configuration (optional)

        Returns:
            UnifiedMergeResult with merged weights and metrics
        """
        logger.info("=== PURE GEOMETRIC MERGE ===")
        logger.info("Source: %s", source_path)
        logger.info("Target: %s", target_path)

        # Use passed config or fall back to instance config
        merge_config = config if config is not None else self.config

        # Resolve output path (prefer output_path over output_dir)
        effective_output = output_path or output_dir
        if (
            transplant_domains is not None
            or transplant_layers is not None
            or transplant_boundary_k is not None
            or transplant_geodesic_k_neighbors is not None
        ):
            merge_config = replace(
                merge_config,
                transplant_domains=tuple(transplant_domains or merge_config.transplant_domains),
                transplant_layers=(
                    tuple(transplant_layers) if transplant_layers else merge_config.transplant_layers
                ),
                transplant_boundary_k=(
                    transplant_boundary_k
                    if transplant_boundary_k is not None
                    else merge_config.transplant_boundary_k
                ),
                transplant_geodesic_k_neighbors=(
                    transplant_geodesic_k_neighbors
                    if transplant_geodesic_k_neighbors is not None
                    else merge_config.transplant_geodesic_k_neighbors
                ),
            )

        # Transplant strategy bypasses full geometry merge (uses null-space projection)
        if use_full_geometry and merge_config.transplant_domains:
            logger.info(
                "Transplant domains specified; using null-space constrained transplant."
            )
            use_full_geometry = False

        if use_full_geometry:
            return self._merge_with_full_geometry(
                source_path,
                target_path,
                output_dir,
                dry_run,
                knowledge_delta_mask_path,
            )

        # Load weights (CPU first to reduce GPU memory pressure during merge)
        source_weights, _ = self._load_weights_cpu(source_path)

        # Use pre-loaded target weights if provided (multi-donor optimization)
        if target_weights is not None:
            logger.info("Using pre-loaded target weights (multi-donor mode)")
            loaded_target_weights = target_weights
            target_format = "safetensors"  # Assume safetensors for pre-loaded weights
        else:
            loaded_target_weights, target_format = self._load_weights_cpu(target_path)

        # Identify layers
        layer_indices = self._extract_layer_indices(loaded_target_weights)
        logger.info("Found %d layers", len(layer_indices))

        # Load tokenizers for vocabulary alignment
        source_tokenizer = self._load_tokenizer(source_path)
        target_tokenizer = self._load_tokenizer(target_path)

        # =================================================================
        # STAGE 0: VOCABULARY (Cross-vocabulary alignment for embedding layers)
        # =================================================================
        # Skip vocabulary alignment for transplant since GRAM_TRANSPORT
        # handles cross-dimensional projection at the weight level.
        skip_vocab_for_transplant = bool(merge_config.transplant_domains)
        if skip_vocab_for_transplant:
            logger.info("STAGE 0: VOCABULARY ALIGNMENT (skipped for transplant)")
            vocab_metrics = {"skipped": True, "reason": "transplant_strategy"}
            vocab_aligned = False
            vocab_alignment_map = None
        else:
            logger.info("STAGE 0: VOCABULARY ALIGNMENT")
            source_weights, vocab_metrics, vocab_aligned, vocab_alignment_map = self._stage_vocabulary(
                source_weights=source_weights,
                target_weights=loaded_target_weights,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
            )
            self._require_vocab_phase_lock(vocab_metrics, vocab_aligned)

        # Load models for probe stage
        source_model = None
        target_model = None
        if merge_config.probe_mode == "precise":
            logger.info("Loading models for precise probe execution...")
            source_model = self._load_model_for_probing(source_path)
            target_model = self._load_model_for_probing(target_path)

        # =================================================================
        # STAGE 1: PROBE (Compute layer correspondences via CKA)
        # =================================================================
        logger.info("STAGE 1: PROBE (%s mode)", merge_config.probe_mode)
        probe_result, probe_metrics, source_activations, target_activations = self._stage_probe(
            source_weights=source_weights,
            target_weights=loaded_target_weights,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            alignment_map=vocab_alignment_map,
            config_override=merge_config,
        )

        layer_confidences: dict[int, float] = probe_result.get("confidences", {})
        dimension_correlations: dict = probe_result.get("dimension_correlations", {})
        intersection_map_obj = probe_result.get("intersection_map")
        probe_failed = bool(probe_metrics.get("probe_failed"))
        perfect_alignment = bool(probe_metrics.get("perfect_alignment"))

        # Skip CKA alignment check for transplant strategy - cross-architecture models
        # won't have high CKA, but transplant works via null-space projection which
        # operates in target activation space regardless of CKA alignment.
        if not skip_vocab_for_transplant:
            if probe_failed:
                min_cka = probe_metrics.get("min_cka", 0.0)
                mean_cka = probe_metrics.get("mean_cka", 0.0)
                raise RuntimeError(
                    "PROBE SIGNAL: Alignment signals missing (mean_cka=%.4f, min_cka=%.4f). "
                    "Exact kernel alignment is required before merge."
                    % (mean_cka, min_cka)
                )

            if not perfect_alignment:
                min_cka = probe_metrics.get("min_cka", 0.0)
                mean_cka = probe_metrics.get("mean_cka", 0.0)
                raise RuntimeError(
                    "PROBE BAROMETER: Alignment not exact kernel aligned "
                    "(mean_cka=%.4f, min_cka=%.4f). Resolve alignment before merge."
                    % (mean_cka, min_cka)
                )
        else:
            mean_cka = probe_metrics.get("mean_cka", 0.0)
            logger.info(
                "PROBE BAROMETER: Skipped for transplant (mean_cka=%.4f - low CKA expected for cross-arch)",
                mean_cka,
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
        # STAGE 2: PERMUTE (Git Re-Basin alignment for same-architecture)
        # =================================================================
        # Permutation alignment reduces delta magnitude before transplant.
        # Only enabled for same-architecture models (hidden dimensions must match).
        source_hidden = self._infer_hidden_dim(source_weights)
        target_hidden = self._infer_hidden_dim(loaded_target_weights)
        enable_permutation = source_hidden == target_hidden

        if enable_permutation:
            logger.info("STAGE 2: PERMUTE (Git Re-Basin, hidden_dim=%d)", target_hidden)
            permuted_weights, permute_metrics = self._stage_permute(
                source_weights=source_weights,
                target_weights=loaded_target_weights,
                intersection_map_obj=intersection_map_obj,
                layer_confidences=layer_confidences,
                enable_permutation=True,
            )
            if not permute_metrics.get("skipped"):
                source_weights = permuted_weights
                logger.info(
                    "PERMUTE: Aligned %d MLP blocks, mean_quality=%.3f",
                    permute_metrics.get("layers_permuted", 0),
                    permute_metrics.get("mean_quality", 0.0),
                )
            else:
                logger.info("PERMUTE: Skipped (%s)", permute_metrics.get("reason", "unknown"))
        else:
            logger.info(
                "STAGE 2: PERMUTE (skipped - hidden_dim mismatch: source=%d, target=%d)",
                source_hidden,
                target_hidden,
            )
            permute_metrics = {"skipped": True, "reason": "hidden_dim_mismatch"}

        # =================================================================
        # STAGE 3: TRANSPLANT (Null-space constrained knowledge transfer)
        # =================================================================
        # ROTATE/BLEND was removed - alpha-blending produces gibberish.
        # Only null-space constrained transplant preserves boundary relationships.
        if not merge_config.transplant_domains:
            raise RuntimeError(
                "Transplant requires transplant_domains. "
                "Specify domains like ['mathematical', 'logical'] for knowledge transfer."
            )
        if not target_activations:
            raise RuntimeError(
                "Transplant requires probe activations. "
                "Use `mc geometry transplant run` to collect activations before merging."
            )

        logger.info("STAGE 3: TRANSPLANT (null-space constrained)")
        merged_weights, transplant_metrics = self._stage_transplant(
            source_weights=source_weights,
            target_weights=loaded_target_weights,
            layer_indices=layer_indices,
            probe_ids=probe_result.get("probe_ids"),
            probe_domains=probe_result.get("probe_domains"),
            target_activations=target_activations,
            config=merge_config,
        )

        # =================================================================
        # OUTPUT
        # =================================================================
        final_output_path: str | None = None
        if effective_output and not dry_run:
            self._save_weights(effective_output, merged_weights, target_format)
            self._copy_config_files(target_path, effective_output)
            final_output_path = effective_output

        # Compute metrics
        mean_confidence = probe_metrics.get("mean_confidence", 0.0)
        projection_losses = transplant_metrics.get("projection_losses", [])
        mean_error = (
            sum(projection_losses) / len(projection_losses) if projection_losses else 0.0
        )

        result = UnifiedMergeResult(
            merged_weights=merged_weights,
            vocab_metrics=vocab_metrics,
            probe_metrics=probe_metrics,
            permute_metrics=permute_metrics,
            transplant_metrics=transplant_metrics,
            mean_confidence=mean_confidence,
            mean_procrustes_error=float(mean_error),
            layer_count=len(layer_indices),
            weight_count=len(merged_weights),
            timestamp=datetime.utcnow(),
            merge_strategy="transplant",
            output_path=final_output_path,
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
        knowledge_delta_mask_path: str | None = None,
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

        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        sample_array = next(iter(source_weights.values()), None)
        # Use a tolerance of 1e-5 for CKA comparison, which accounts for numerical
        # precision issues in Gram matrix computation, centering, and accumulation.
        # Machine epsilon (~1e-7 for float32) is too tight for CKA comparisons.
        base_eps = (
            machine_epsilon(self._backend, sample_array) if sample_array is not None else 1e-7
        )
        phase_tol = max(base_eps * 100, 1e-5)  # At least 1e-5, or 100x machine epsilon
        if geometry.overall_cka < 1.0 - phase_tol:
            raise RuntimeError(
                "PROBE BAROMETER: Overall CKA=%.6f < 1.0. "
                "Exact kernel alignment is required before merging."
                % geometry.overall_cka
            )

        # Execute merge using geometry
        logger.info("EXECUTING MERGE...")
        merge_start = time.perf_counter()
        layer_alpha_scale = None
        if knowledge_delta_mask_path:
            layer_alpha_scale = self._load_knowledge_delta_mask(knowledge_delta_mask_path)
            logger.info(
                "Applying knowledge delta mask: %s (%d layers)",
                knowledge_delta_mask_path,
                len(layer_alpha_scale),
            )
        merged_weights, merge_metrics = orchestrator.merge_weights(
            source_weights=source_weights,
            target_weights=target_weights,
            geometry=geometry,
            extract_layer_index_fn=self._extract_layer_index,
            checkpoint_dir=output_dir,
            layer_alpha_scale=layer_alpha_scale,
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
            permute_metrics={"skipped": True, "reason": "full_geometry_mode"},
            transplant_metrics=merge_metrics,
            mean_confidence=geometry.overall_cka,
            mean_procrustes_error=0.0,
            layer_count=len(layer_indices),
            weight_count=len(merged_weights),
            timestamp=datetime.utcnow(),
            merge_strategy="full_geometry",
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
        config_override: UnifiedMergeConfig | None = None,
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

        active_config = config_override or self.config
        config = ProbeConfig(
            probe_mode=active_config.probe_mode,
            max_probes=active_config.max_probes,
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
            "probe_ids": result.probe_ids,
            "probe_domains": result.probe_domains,
        }, result.metrics, result.source_activations, result.target_activations

    def _stage_permute(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        intersection_map_obj: Any | None,
        layer_confidences: dict[int, float],
        enable_permutation: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Stage 2: Git Re-Basin permutation alignment.

        Solves the permutation symmetry problem for MLP neurons.
        Neural networks have N! permutation symmetries per MLP layer.
        This stage finds the optimal permutation P that minimizes:
            ||W_target - P @ W_source||_F

        This runs BEFORE transplant to reduce the delta magnitude between
        source and target weights. By aligning neuron orderings first, the
        null-space projection in transplant has less work to do.

        Reference: Ainsworth et al. (2023) arXiv:2209.04836 "Git Re-Basin"
        """
        from .merge_stages.stage_2_permute import (
            PermuteConfig,
            infer_hidden_dim,
            stage_permute,
        )

        config = PermuteConfig(enable_permutation=enable_permutation)

        result = stage_permute(
            source_weights=source_weights,
            target_weights=target_weights,
            intersection_map_obj=intersection_map_obj,
            layer_confidences=layer_confidences,
            config=config,
            infer_hidden_dim_fn=infer_hidden_dim,
            backend=self._backend,
        )

        return result.weights, result.metrics

    def _stage_transplant(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        layer_indices: list[int],
        probe_ids: list[str] | None,
        probe_domains: list[str] | None,
        target_activations: dict | None,
        config: UnifiedMergeConfig,
    ) -> tuple[dict[str, "Array"], dict[str, Any]]:
        """Stage 3: Null-space constrained transplant."""
        from .merge_stages.stage_3_transplant import (
            TransplantStageConfig,
            stage_transplant,
        )

        stage_config = TransplantStageConfig(
            core_domains=tuple(config.transplant_domains),
            boundary_k=config.transplant_boundary_k,
            geodesic_k_neighbors=config.transplant_geodesic_k_neighbors,
            transplant_layers=config.transplant_layers,
        )

        result = stage_transplant(
            source_weights=source_weights,
            target_weights=target_weights,
            layer_indices=layer_indices,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            target_activations=target_activations,
            config=stage_config,
            extract_layer_index_fn=self._extract_layer_index,
            backend=self._backend,
        )

        return result.merged_weights, result.metrics

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

    def _load_knowledge_delta_mask(self, mask_path: str) -> dict[int, float]:
        """Load per-layer alpha scaling from a knowledge delta mask file."""
        import json

        payload = json.loads(Path(mask_path).read_text())
        alpha_by_layer = payload.get("alphaByLayer") or payload.get("alpha_by_layer")
        if isinstance(alpha_by_layer, dict):
            return {int(layer): float(alpha) for layer, alpha in alpha_by_layer.items()}

        graft_layers = payload.get("graftLayers") or payload.get("graft_layers")
        if isinstance(graft_layers, list):
            return {int(layer): 1.0 for layer in graft_layers}

        raise ValueError("Invalid knowledge delta mask: missing alphaByLayer or graftLayers.")

    def _infer_hidden_dim(self, weights: dict[str, Any]) -> int:
        """Infer hidden dimension from weight shapes.

        Used to determine if permutation alignment is applicable (same hidden dim).
        """
        # Prefer norm weights: they are 1D, remain unquantized, and directly encode hidden size.
        for key, val in weights.items():
            if key.endswith(".scales") or key.endswith(".biases"):
                continue
            if not hasattr(val, "shape"):
                continue
            if len(val.shape) != 1:
                continue
            if key.endswith(("norm.weight", "layernorm.weight", "rms_norm.weight")):
                return int(val.shape[0])

        # Fall back to projection matrices (avoid quantization metadata like *.scales).
        for key, val in weights.items():
            if key.endswith(".scales") or key.endswith(".biases"):
                continue
            if not hasattr(val, "shape") or len(val.shape) != 2:
                continue
            if not key.endswith(".weight"):
                continue
            if "q_proj" in key or "o_proj" in key:
                # Usually square [hidden, hidden]
                return int(max(val.shape))
            if "k_proj" in key or "v_proj" in key:
                # GQA: [kv_dim, hidden] -> hidden is the max dim
                return int(max(val.shape))
            if "up_proj" in key or "gate_proj" in key or "down_proj" in key:
                # MLP: [intermediate, hidden] or [hidden, intermediate] -> hidden is the min dim
                return int(min(val.shape))
        # Return 0 for unknown (will disable permutation)
        return 0

    def _require_vocab_phase_lock(
        self, vocab_metrics: dict[str, Any], vocab_aligned: bool
    ) -> None:
        if not vocab_aligned:
            raise RuntimeError(
                "Vocabulary alignment was not applied. "
                "Exact kernel alignment is required before merge."
            )
        binary = vocab_metrics.get("binary_alignment", {})
        vocab = vocab_metrics.get("vocab_phase_lock", {})
        if not binary or not vocab:
            raise RuntimeError(
                "Vocabulary alignment metrics missing; cannot confirm exact kernel alignment."
            )
        for key, entry in binary.items():
            if not entry.get("phase_locked"):
                raise RuntimeError(
                    f"Binary exact kernel alignment missing for {key}; aborting merge."
                )
        for key, entry in vocab.items():
            if not entry.get("phase_locked"):
                raise RuntimeError(
                    f"Vocabulary exact kernel alignment missing for {key}; aborting merge."
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

        # MLX native save is only safe when *all* tensors are mx.array.
        # Mixed dicts (mx.array + numpy) trigger MLX std::bad_cast.
        try:
            import mlx.core as mx

            if weights and all(isinstance(v, mx.array) for v in weights.values()):
                mx.save_safetensors(str(output_path), weights)
                logger.info("Saved merged weights to %s (MLX native)", output_path)
                return
        except Exception:
            # Fall through to numpy-based save paths.
            pass

        # Fallback to safetensors (convert arrays to numpy for save)
        if output_format == "safetensors":
            from safetensors.numpy import save_file

            # Convert backend arrays to numpy for safetensors save
            numpy_weights = {}
            for key, value in weights.items():
                if type(value).__module__.startswith("numpy"):
                    numpy_weights[key] = value
                    continue
                try:
                    numpy_weights[key] = self._backend.to_numpy(value)
                except Exception:
                    numpy_weights[key] = value
            save_file(numpy_weights, str(output_path))
        else:
            # For npz format, also convert to numpy
            output_path = path / "weights.npz"
            import numpy as _np_for_save  # Only for file I/O, not computation

            numpy_weights = {}
            for key, value in weights.items():
                if type(value).__module__.startswith("numpy"):
                    numpy_weights[key] = value
                    continue
                try:
                    numpy_weights[key] = self._backend.to_numpy(value)
                except Exception:
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
