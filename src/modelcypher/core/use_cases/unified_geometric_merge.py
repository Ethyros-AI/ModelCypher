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

import numpy as np

from modelcypher.core.domain.thermo.phase_transition_theory import Phase

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedMergeConfig:
    """
    Configuration for unified geometric merge.

    This consolidates ALL merge configuration into one place.
    Each stage can be enabled/disabled independently.
    """

    # ==========================================================================
    # STAGE 1: PROBE (Fingerprinting)
    # ==========================================================================

    # Probe mode: how to compute layer confidence
    # - "precise": Run all 403 probes through BOTH models, compute CKA on activations
    #              This is the CORRECT method per Moschella et al. (2023)
    #              Slower (~5-10 min) but produces accurate layer confidence.
    # - "fast": Use weight-level CKA (current behavior, faster but less accurate)
    probe_mode: Literal["precise", "fast"] = "precise"

    # Whether to probe models for fingerprints (if False, use pre-computed)
    probe_models: bool = True

    # Similarity mode for intersection map: jaccard, cka, ensemble
    intersection_mode: str = "ensemble"

    # Minimum correlation to include in intersection map
    intersection_threshold: float = 0.3

    # Use UnifiedAtlasInventory (403 probes) for semantic anchoring
    use_atlas_probes: bool = True

    # Maximum probes to run in precise mode (for faster testing)
    # Set to 0 for all probes (403)
    max_probes: int = 0

    # ==========================================================================
    # STAGE 2: PERMUTE (Re-Basin)
    # ==========================================================================

    # Enable permutation alignment for MLP neurons
    enable_permutation: bool = True

    # Minimum confidence to apply permutation (risky below this)
    permutation_confidence_threshold: float = 0.6

    # ==========================================================================
    # STAGE 3: ROTATE (Geometric Alignment)
    # ==========================================================================

    # Enable rotation alignment
    enable_rotation: bool = True

    # Minimum confidence to apply rotation
    rotation_confidence_threshold: float = 0.4

    # SVD rank for rotation computation
    alignment_rank: int = 32

    # Use transport-guided (Gromov-Wasserstein) instead of Procrustes
    use_transport_guided: bool = False

    # Transport coupling threshold (for GW)
    transport_coupling_threshold: float = 0.001

    # Enable shared subspace projection blending
    enable_shared_subspace: bool = True

    # Shared subspace blend weight
    shared_subspace_blend: float = 0.5

    # ==========================================================================
    # STAGE 4: BLEND (Multi-Layer Alpha)
    # ==========================================================================

    # Base alpha (0 = all target, 1 = all source)
    base_alpha: float = 0.5

    # --- 4.1: Gaussian Smoothing ---
    enable_alpha_smoothing: bool = True
    smoothing_window: int = 2
    smoothing_sigma: float = 1.0

    # --- 4.2: Spectral Penalty ---
    enable_spectral_penalty: bool = False  # Disabled - uses numpy SVD
    spectral_penalty_strength: float = 0.5

    # --- 4.3: SVD-Aware Blending ---
    enable_svd_blending: bool = False  # Disabled for GPU acceleration
    svd_rank_ratio: float = 0.1
    high_rank_alpha: float = 0.3  # Trust source for skills
    low_rank_alpha: float = 0.7  # Trust target for structure

    # --- 4.4: Correlation-Based Dimension Weights ---
    enable_correlation_weights: bool = False  # Disabled for GPU acceleration
    correlation_scale: float = 5.0
    stability_alpha: float = 0.7  # Used when dimensions disagree

    # --- 4.5: VerbNoun Modulation ---
    enable_verb_noun: bool = False  # Disabled for GPU acceleration
    verb_noun_strength: float = 0.7

    # --- 4.6: Domain Signals ---
    # Uses gradient SNR and sparsity per-layer to adjust alpha.
    # High gradient SNR + low sparsity → trust target (higher alpha)
    # Low gradient SNR + high sparsity → trust source (lower alpha)
    enable_domain_signals: bool = True
    domain_signal_strength: float = 0.3

    # --- 4.7: Transition Gate (CRM) ---
    enable_transition_gate: bool = False
    transition_gate_strength: float = 0.3

    # --- 4.8: Consistency Gate (CRM) ---
    enable_consistency_gate: bool = False
    consistency_gate_strength: float = 0.3

    # --- 4.9: Module-Specific Policy ---
    enable_module_policy: bool = True
    # Different modules require different blending strategies based on their role:
    # - v_proj: Captures what information gets attended to, trust source skills (hard swap)
    # - o_proj: Projects back to residual stream, preserve target structure (skip)
    # - Others: Use computed alpha (soft blend)
    module_policy_v_alpha: float = 0.9  # Trust source for v_proj (skills)
    module_policy_o_alpha: float = 0.1  # Trust target for o_proj (structure)

    # --- 4.10: MLP Internal Gate ---
    enable_mlp_gate: bool = False
    mlp_gate_strength: float = 0.3

    # --- 4.11: Clamping ---
    alpha_min: float = 0.1
    alpha_max: float = 0.9

    # --- 4.12: Refinement Density (Validated Law #5) ---
    # Uses DARE sparsity + DoRA drift to identify "knowledge dense" layers
    # High refinement → layer learned new skills → trust source (lower alpha)
    # Low refinement → layer unchanged → trust target (higher alpha)
    enable_refinement_density: bool = True
    refinement_density_strength: float = 0.7  # How strongly to modulate alphas
    refinement_hard_swap_enabled: bool = (
        True  # Allow full source replacement for highly refined layers
    )

    # --- 4.13: Intrinsic Dimension Gating (Dimensional Hierarchy) ---
    # Uses SVD effective rank to estimate manifold complexity per layer.
    # Low complexity (intrinsic_dim << hidden_dim) → simple manifold → blend aggressively
    # High complexity → complex structure → blend conservatively (trust target)
    enable_intrinsic_dim_gating: bool = False  # Disabled for GPU acceleration
    intrinsic_dim_strength: float = 0.5  # How strongly to modulate alphas
    intrinsic_dim_threshold: float = 0.01  # SVD cutoff (1% of max singular value)

    # --- 4.14: Thermodynamic Phase Gating ---
    # Uses entropy phase analysis to adjust alpha aggressiveness.
    # ORDERED phase: Low entropy, safe to blend aggressively (full alpha)
    # CRITICAL phase: Near phase boundary, reduce alpha by 30% (conservative)
    # DISORDERED phase: High entropy, reduce alpha by 15% (slightly conservative)
    enable_thermo_gating: bool = True
    thermo_safety_prompts: tuple[str, ...] = (
        "Explain how to be helpful and harmless.",
        "Describe ethical behavior in AI systems.",
        "What does it mean to be a responsible assistant?",
    )

    # ==========================================================================
    # STAGE 5: PROPAGATE (Zipper)
    # ==========================================================================

    # Propagate rotations to next layer (essential for coherence)
    enable_zipper: bool = True

    # Use proper weight matching (Linear Assignment Problem) for zipper.
    # This computes full-rank permutation matrices that properly propagate
    # through layers, following Git Re-Basin (Ainsworth et al., 2022).
    # When False, uses low-rank spectral rotation (may have dimension mismatch).
    zipper_use_weight_matching: bool = True

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    # Output quantization
    output_quant: str | None = None
    output_quant_group_size: int | None = None

    # ==========================================================================
    # STAGE 0: VOCABULARY ALIGNMENT (Cross-Vocabulary Merging)
    # ==========================================================================

    # Enable vocabulary alignment for cross-vocabulary merging
    enable_vocabulary_alignment: bool = True

    # Projection strategy: procrustes, pca, optimal_transport, cca, truncate
    vocab_projection_strategy: str = "procrustes"

    # Alignment thresholds
    vocab_similarity_threshold: float = 0.8
    vocab_confidence_threshold: float = 0.5

    # Embedding blending
    vocab_blend_alpha: float = 0.5
    vocab_preserve_special_tokens: bool = True

    # Quality thresholds
    vocab_min_compatibility_score: float = 0.3
    vocab_min_coverage: float = 0.5

    # Advanced options
    vocab_use_embedding_similarity: bool = True
    vocab_anchor_count: int = 1000

    # ==========================================================================
    # STAGE 6: VALIDATE (Safety Checks)
    # ==========================================================================

    # Enable safety validation after merge
    enable_safety_validation: bool = True

    # Fail merge if validation fails (otherwise just log warning)
    validation_fail_on_unsafe: bool = False

    # Enable content safety check via refusal direction detection
    # Requires models and tokenizer to be provided
    enable_refusal_check: bool = True

    # Minimum refusal preservation score (0-1, higher = safer)
    refusal_preservation_threshold: float = 0.7

    # Maximum instability score before rejection (0-1)
    max_instability_threshold: float = 0.8

    # Maximum interference score before rejection (0-1)
    max_interference_threshold: float = 0.9

    @classmethod
    def default(cls) -> UnifiedMergeConfig:
        """Default balanced configuration."""
        return cls()

    @classmethod
    def conservative(cls) -> UnifiedMergeConfig:
        """Conservative: preserve target structure."""
        return cls(
            base_alpha=0.7,
            permutation_confidence_threshold=0.7,
            rotation_confidence_threshold=0.5,
            use_transport_guided=False,
            high_rank_alpha=0.4,
            low_rank_alpha=0.8,
            verb_noun_strength=0.5,
        )

    @classmethod
    def aggressive(cls) -> UnifiedMergeConfig:
        """Aggressive: trust source skills."""
        return cls(
            base_alpha=0.3,
            permutation_confidence_threshold=0.4,
            rotation_confidence_threshold=0.3,
            alignment_rank=48,
            high_rank_alpha=0.2,
            low_rank_alpha=0.6,
            verb_noun_strength=0.9,
            enable_domain_signals=True,
        )


@dataclass
class LayerMergeState:
    """State carried through layers during merge (zipper)."""

    # Current input rotation (from previous layer's output)
    omega_in: np.ndarray | None = None

    # Layer index
    layer_index: int = 0

    # Accumulated metrics
    procrustes_errors: list[float] = field(default_factory=list)
    spectral_ratios: list[float] = field(default_factory=list)
    effective_alphas: list[float] = field(default_factory=list)


@dataclass
class UnifiedMergeResult:
    """Result of unified geometric merge."""

    merged_weights: dict[str, np.ndarray]

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
        self.config = config or UnifiedMergeConfig.default()

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
        source_fingerprints: dict | None = None,
        target_fingerprints: dict | None = None,
        source_tokenizer: Any | None = None,
        target_tokenizer: Any | None = None,
        base_model_path: str | None = None,
        dry_run: bool = False,
    ) -> UnifiedMergeResult:
        """
        Execute unified geometric merge.

        Args:
            source_path: Path to source model (skill donor)
            target_path: Path to target model (knowledge base)
            output_dir: Output directory for merged model
            source_fingerprints: Pre-computed source fingerprints
            target_fingerprints: Pre-computed target fingerprints
            source_tokenizer: Source tokenizer (for cross-vocabulary merging)
            target_tokenizer: Target tokenizer (for cross-vocabulary merging)
            base_model_path: Path to base/pretrained model for refinement density
                             calculation. If provided, computes per-layer refinement
                             density (DARE sparsity + DoRA drift) to weight layer
                             alphas based on actual information content.
            dry_run: If True, don't save to disk

        Returns:
            UnifiedMergeResult with merged weights and metrics
        """
        logger.info("=== UNIFIED GEOMETRIC MERGE ===")
        logger.info("Source: %s", source_path)
        logger.info("Target: %s", target_path)
        if base_model_path:
            logger.info("Base model: %s (refinement density enabled)", base_model_path)
        logger.info("Probe mode: %s", self.config.probe_mode)

        # Load weights
        source_weights, source_format = self._load_weights(source_path)
        target_weights, target_format = self._load_weights(target_path)

        # Load base model weights for refinement density (if provided)
        base_weights: dict[str, np.ndarray] | None = None
        if base_model_path and self.config.enable_refinement_density:
            try:
                base_weights, _ = self._load_weights(base_model_path)
                logger.info("Loaded base model weights for refinement density analysis")
            except Exception as e:
                logger.warning("Failed to load base model weights: %s", e)
                base_weights = None

        # Load tokenizer for probe execution (needed for both precise and fast modes)
        tokenizer = self._load_tokenizer(target_path)

        # For precise mode, load models for activation extraction
        source_model = None
        target_model = None
        if self.config.probe_mode == "precise":
            logger.info("Loading models for precise probe execution...")
            source_model = self._load_model_for_probing(source_path)
            target_model = self._load_model_for_probing(target_path)

        # Identify layers
        layer_indices = self._extract_layer_indices(target_weights)
        logger.info("Found %d layers", len(layer_indices))

        # =================================================================
        # REFINEMENT DENSITY ANALYSIS (Validated Law #5)
        # =================================================================
        refinement_alphas: dict[int, float] | None = None
        refinement_hard_swap_layers: set[int] = set()
        refinement_metrics: dict[str, Any] = {}

        if base_weights is not None and self.config.enable_refinement_density:
            logger.info("REFINEMENT DENSITY: Computing per-layer knowledge mass...")
            refinement_alphas, hard_swap_layers, refinement_metrics = (
                self._compute_refinement_density(
                    source_weights=source_weights,
                    base_weights=base_weights,
                    source_path=source_path,
                    base_path=base_model_path or "",
                )
            )
            if self.config.refinement_hard_swap_enabled:
                refinement_hard_swap_layers = set(hard_swap_layers)
            if refinement_alphas:
                logger.info(
                    "REFINEMENT DENSITY: Computed alphas for %d layers (mean=%.3f), %d hard-swap candidates",
                    len(refinement_alphas),
                    np.mean(list(refinement_alphas.values())),
                    len(refinement_hard_swap_layers),
                )

        # =================================================================
        # STAGE 0: VOCABULARY ALIGNMENT
        # =================================================================
        vocab_metrics: dict[str, Any] = {}
        vocab_aligned = False

        if self.config.enable_vocabulary_alignment:
            logger.info("STAGE 0: VOCABULARY ALIGNMENT")
            source_weights, vocab_metrics, vocab_aligned = self._stage_vocabulary_align(
                source_weights,
                target_weights,
                source_tokenizer,
                target_tokenizer,
            )
            if vocab_aligned:
                logger.info("Vocabulary alignment applied")
            else:
                logger.info("Vocabularies compatible, no alignment needed")

        # =================================================================
        # STAGE 1: PROBE
        # =================================================================
        logger.info("STAGE 1: PROBE (%s mode)", self.config.probe_mode)
        probe_result, probe_metrics = self._stage_probe(
            source_weights=source_weights,
            target_weights=target_weights,
            source_fingerprints=source_fingerprints,
            target_fingerprints=target_fingerprints,
            source_model=source_model,
            target_model=target_model,
            tokenizer=tokenizer,
            source_path=source_path,
            target_path=target_path,
        )

        # Extract the IntersectionMap object (if built) for downstream stages
        from modelcypher.core.domain.geometry.manifold_stitcher import IntersectionMap

        intersection_map_obj: IntersectionMap | None = probe_result.get("intersection_map")
        layer_confidences: dict[int, float] = probe_result.get("confidences", {})
        dimension_correlations: dict = probe_result.get("dimension_correlations", {})

        # =================================================================
        # STAGE 2: PERMUTE
        # =================================================================
        logger.info("STAGE 2: PERMUTE (Re-Basin)")
        permuted_source, permute_metrics = self._stage_permute(
            source_weights,
            target_weights,
            intersection_map_obj,
            layer_confidences,
        )

        # =================================================================
        # THERMODYNAMIC PHASE ANALYSIS
        # =================================================================
        entropy_phase = Phase.ORDERED  # Default
        if self.config.enable_thermo_gating:
            logger.info("THERMO: Computing entropy phase for merge gating...")
            entropy_phase = self._compute_entropy_phase(target_path, tokenizer)

        # =================================================================
        # STAGE 3 & 4 & 5: ROTATE + BLEND + PROPAGATE (merged loop)
        # =================================================================
        logger.info("STAGES 3-5: ROTATE + BLEND + PROPAGATE")
        merged_weights, rotate_metrics, blend_metrics = self._stage_rotate_blend_propagate(
            permuted_source,
            target_weights,
            intersection_map_obj,
            layer_confidences,
            dimension_correlations,
            layer_indices,
            refinement_alphas=refinement_alphas,
            hard_swap_layers=refinement_hard_swap_layers,
            entropy_phase=entropy_phase,
        )

        # =================================================================
        # STAGE 6: VALIDATE (Safety Checks)
        # =================================================================
        logger.info("STAGE 6: VALIDATE (Safety)")
        validation_metrics, safety_verdict, refusal_preserved = self._stage_validate(
            merged_weights=merged_weights,
            source_weights=source_weights,
            target_weights=target_weights,
            source_model=source_model,
            target_model=target_model,
            tokenizer=tokenizer,
            blend_metrics=blend_metrics,
            layer_confidences=layer_confidences,
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

        # Compute overall metrics
        mean_confidence = probe_metrics.get("mean_confidence", 0.0)
        procrustes_errors = rotate_metrics.get("procrustes_errors", [])
        mean_error = float(np.mean(procrustes_errors)) if procrustes_errors else 0.0

        result = UnifiedMergeResult(
            merged_weights=merged_weights,
            vocab_metrics=vocab_metrics,
            probe_metrics=probe_metrics,
            permute_metrics=permute_metrics,
            rotate_metrics=rotate_metrics,
            blend_metrics=blend_metrics,
            validation_metrics=validation_metrics,
            mean_confidence=mean_confidence,
            mean_procrustes_error=float(mean_error),
            layer_count=len(layer_indices),
            weight_count=len(merged_weights),
            timestamp=datetime.utcnow(),
            output_path=output_path,
            vocab_aligned=vocab_aligned,
            safety_verdict=safety_verdict,
            refusal_preserved=refusal_preserved,
        )

        logger.info(
            "Merge complete: %d layers, %d weights, confidence=%.3f, error=%.3f",
            result.layer_count,
            result.weight_count,
            result.mean_confidence,
            result.mean_procrustes_error,
        )

        return result

    # =========================================================================
    # STAGE DELEGATES - Each stage is implemented in merge_stages/
    # =========================================================================

    def _stage_vocabulary_align(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        source_tokenizer: Any | None,
        target_tokenizer: Any | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
        """Stage 0: Vocabulary alignment. See merge_stages/stage_0_vocabulary.py."""
        from .merge_stages.stage_0_vocabulary import (
            VocabularyConfig,
            stage_vocabulary_align,
        )

        config = VocabularyConfig(
            projection_strategy=self.config.vocab_projection_strategy,
            similarity_threshold=self.config.vocab_similarity_threshold,
            confidence_threshold=self.config.vocab_confidence_threshold,
            blend_alpha=self.config.vocab_blend_alpha,
            preserve_special_tokens=self.config.vocab_preserve_special_tokens,
            min_compatibility_score=self.config.vocab_min_compatibility_score,
            min_coverage=self.config.vocab_min_coverage,
            use_embedding_similarity=self.config.vocab_use_embedding_similarity,
            anchor_count=self.config.vocab_anchor_count,
        )

        result = stage_vocabulary_align(
            source_weights=source_weights,
            target_weights=target_weights,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            config=config,
        )

        return result.modified_weights, result.metrics, result.was_aligned

    def _stage_probe(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        source_fingerprints: dict | None,
        target_fingerprints: dict | None,
        source_model: Any | None,
        target_model: Any | None,
        tokenizer: Any | None,
        source_path: str,
        target_path: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Stage 1: Probing. See merge_stages/stage_1_probe.py."""
        from .merge_stages.stage_1_probe import (
            ProbeConfig,
            collect_layer_activations_mlx,
            stage_probe,
        )

        config = ProbeConfig(
            probe_mode=self.config.probe_mode,
            max_probes=self.config.max_probes,
            intersection_mode=self.config.intersection_mode,
        )

        # Provide activation collection function for precise mode
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
            "intersection_map": result.intersection_map,
            "dimension_correlations": result.dimension_correlations,
        }, result.metrics

    def _stage_permute(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        intersection_map_obj: Any | None,
        layer_confidences: dict[int, float],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Stage 2: Permutation. See merge_stages/stage_2_permute.py."""
        from .merge_stages.stage_2_permute import (
            PermuteConfig,
            infer_hidden_dim,
            stage_permute,
        )

        config = PermuteConfig(
            enable_permutation=self.config.enable_permutation,
            permutation_confidence_threshold=self.config.permutation_confidence_threshold,
        )

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
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        intersection_map_obj: Any | None,
        layer_confidences: dict[int, float],
        dimension_correlations: dict,
        layer_indices: list[int],
        refinement_alphas: dict[int, float] | None = None,
        hard_swap_layers: set[int] | None = None,
        entropy_phase: Phase = Phase.ORDERED,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
        """Stages 3-5: Rotate + Blend + Propagate. See merge_stages/stage_3_5_rotate_blend.py."""
        from .merge_stages.stage_3_5_rotate_blend import (
            RotateBlendConfig,
            stage_rotate_blend_propagate,
        )

        config = RotateBlendConfig(
            enable_rotation=self.config.enable_rotation,
            rotation_confidence_threshold=self.config.rotation_confidence_threshold,
            alignment_rank=self.config.alignment_rank,
            use_transport_guided=self.config.use_transport_guided,
            transport_coupling_threshold=self.config.transport_coupling_threshold,
            base_alpha=self.config.base_alpha,
            enable_alpha_smoothing=self.config.enable_alpha_smoothing,
            smoothing_window=self.config.smoothing_window,
            smoothing_sigma=self.config.smoothing_sigma,
            enable_spectral_penalty=self.config.enable_spectral_penalty,
            spectral_penalty_strength=self.config.spectral_penalty_strength,
            enable_svd_blending=self.config.enable_svd_blending,
            svd_rank_ratio=self.config.svd_rank_ratio,
            high_rank_alpha=self.config.high_rank_alpha,
            low_rank_alpha=self.config.low_rank_alpha,
            enable_correlation_weights=self.config.enable_correlation_weights,
            correlation_scale=self.config.correlation_scale,
            stability_alpha=self.config.stability_alpha,
            enable_verb_noun=self.config.enable_verb_noun,
            verb_noun_strength=self.config.verb_noun_strength,
            enable_domain_signals=self.config.enable_domain_signals,
            domain_signal_strength=self.config.domain_signal_strength,
            enable_module_policy=self.config.enable_module_policy,
            module_policy_v_alpha=self.config.module_policy_v_alpha,
            module_policy_o_alpha=self.config.module_policy_o_alpha,
            alpha_min=self.config.alpha_min,
            alpha_max=self.config.alpha_max,
            enable_refinement_density=self.config.enable_refinement_density,
            refinement_density_strength=self.config.refinement_density_strength,
            enable_zipper=self.config.enable_zipper,
            zipper_use_weight_matching=self.config.zipper_use_weight_matching,
            enable_intrinsic_dim_gating=self.config.enable_intrinsic_dim_gating,
            intrinsic_dim_strength=self.config.intrinsic_dim_strength,
            intrinsic_dim_threshold=self.config.intrinsic_dim_threshold,
            entropy_phase=entropy_phase.value,  # Pass as string
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
            refinement_alphas=refinement_alphas,
            hard_swap_layers=hard_swap_layers,
            backend=self._backend,
        )

        return result.merged_weights, result.rotate_metrics, result.blend_metrics

    def _stage_validate(
        self,
        merged_weights: dict[str, np.ndarray],
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        source_model: Any | None,
        target_model: Any | None,
        tokenizer: Any | None,
        blend_metrics: dict[str, Any],
        layer_confidences: dict[int, float],
    ) -> tuple[dict[str, Any], str, bool]:
        """Stage 6: Validation. See merge_stages/stage_6_validate.py."""
        from .merge_stages.stage_6_validate import (
            ValidateConfig,
            stage_validate,
        )

        config = ValidateConfig(
            enable_safety_validation=self.config.enable_safety_validation,
            validation_fail_on_unsafe=self.config.validation_fail_on_unsafe,
            enable_refusal_check=self.config.enable_refusal_check,
            refusal_preservation_threshold=self.config.refusal_preservation_threshold,
            max_instability_threshold=self.config.max_instability_threshold,
            max_interference_threshold=self.config.max_interference_threshold,
        )

        # Extract layer indices and hidden dim for validation
        from .merge_stages.stage_3_5_rotate_blend import _infer_hidden_dim

        layer_indices = self._extract_layer_indices(target_weights)
        hidden_dim = _infer_hidden_dim(target_weights)

        result = stage_validate(
            merged_weights=merged_weights,
            source_weights=source_weights,
            target_weights=target_weights,
            layer_confidences=layer_confidences,
            config=config,
            layer_indices=layer_indices,
            hidden_dim=hidden_dim,
            target_model=target_model,
            tokenizer=tokenizer,
        )

        return result.metrics, result.safety_verdict, result.refusal_preserved

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

    def _load_weights_numpy(self, model_path: str) -> tuple[dict[str, np.ndarray], str]:
        """Load model weights as NumPy (for stages that still need NumPy)."""
        weights = self._model_loader.load_weights_as_numpy(model_path)
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

        # Fallback to NumPy safetensors
        if output_format == "safetensors":
            from safetensors.numpy import save_file

            save_file(weights, str(output_path))
        else:
            output_path = path / "weights.npz"
            np.savez(str(output_path), **weights)

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

    def _extract_layer_indices(self, weights: dict[str, np.ndarray]) -> list[int]:
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

    def _compute_refinement_density(
        self,
        source_weights: dict[str, np.ndarray],
        base_weights: dict[str, np.ndarray],
        source_path: str,
        base_path: str,
    ) -> tuple[dict[int, float] | None, list[int], dict[str, Any]]:
        """
        Compute per-layer refinement density.

        Uses DARE sparsity + DoRA drift to identify "knowledge dense" layers.
        Returns alpha modulation: high refinement → lower alpha (trust source).
        """
        try:
            from modelcypher.core.domain.geometry.refinement_density import (
                RefinementDensityAnalyzer,
                RefinementDensityConfig,
            )

            config = RefinementDensityConfig(
                strength=self.config.refinement_density_strength,
            )

            analyzer = RefinementDensityAnalyzer(config)
            result = analyzer.analyze(
                fine_tuned_weights=source_weights,
                base_weights=base_weights,
                fine_tuned_path=source_path,
                base_path=base_path,
            )

            return (
                result.alpha_by_layer,
                list(result.hard_swap_layers),
                result.metrics,
            )

        except Exception as e:
            logger.warning("Refinement density analysis failed: %s", e)
            return None, [], {"error": str(e)}

    def _compute_entropy_phase(
        self,
        model_path: str,
        tokenizer: Any | None,
    ) -> Phase:
        """
        Compute thermodynamic phase of model using entropy analysis.

        Uses safety prompts to measure model's entropy distribution and
        classify into ORDERED (low entropy), CRITICAL (phase boundary),
        or DISORDERED (high entropy).

        Args:
            model_path: Path to model for entropy measurement
            tokenizer: Tokenizer for the model

        Returns:
            Phase classification (ORDERED, CRITICAL, or DISORDERED)
        """
        if not self.config.enable_thermo_gating:
            return Phase.ORDERED  # Default: don't adjust alpha

        try:
            from modelcypher.core.domain.thermo.linguistic_calorimeter import (
                LinguisticCalorimeter,
            )

            # Use simulated mode for fast phase estimation
            # (Real mode would require full inference which is slow)
            calorimeter = LinguisticCalorimeter(simulated=True)

            entropies: list[float] = []
            for prompt in self.config.thermo_safety_prompts:
                try:
                    measurement = calorimeter.measure_entropy(
                        prompt=prompt,
                        temperature=1.0,
                        max_tokens=20,
                    )
                    entropies.append(measurement.mean_entropy)
                except Exception:
                    continue

            if not entropies:
                logger.warning("No entropy measurements available, using ORDERED phase")
                return Phase.ORDERED

            mean_entropy = float(np.mean(entropies))
            logger.info("THERMO PHASE: mean_entropy=%.3f", mean_entropy)

            # Phase classification based on full-vocab entropy scale
            # (0-10.5 range for 32K vocab)
            if mean_entropy < 2.5:
                logger.info("THERMO PHASE: ORDERED (low entropy, safe for aggressive blend)")
                return Phase.ORDERED
            elif mean_entropy < 3.5:
                logger.info("THERMO PHASE: CRITICAL (near boundary, use conservative blend)")
                return Phase.CRITICAL
            else:
                logger.info("THERMO PHASE: DISORDERED (high entropy, slightly conservative)")
                return Phase.DISORDERED

        except Exception as e:
            logger.warning("Entropy phase analysis failed: %s, using ORDERED", e)
            return Phase.ORDERED


def unified_merge(
    source: str,
    target: str,
    output_dir: str,
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
        config: Merge configuration (optional)
        dry_run: If True, don't save to disk

    Returns:
        UnifiedMergeResult with merged weights and metrics
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader

    model_loader = MLXModelLoader()
    merger = UnifiedGeometricMerger(model_loader=model_loader, config=config)

    return merger.merge(
        source_path=source,
        target_path=target,
        output_dir=output_dir,
        dry_run=dry_run,
    )
