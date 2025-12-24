"""
Unified Geometric Merge Pipeline.

This is THE ONE merge method that combines ALL geometric techniques
in the correct order. Based on the 5-stage pipeline:

    PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE

The intersection map (from semantic probes) is the PRIMARY CONTROL SIGNAL
that guides all downstream operations.

Key Principles:
1. Intersection map confidence controls when to apply risky operations
2. Geometric transformations are propagated layer-to-layer (zipper)
3. Alpha adjustments are applied sequentially (12+ stages)
4. Per-dimension control enables surgical merging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from modelcypher.core.domain.agents.unified_atlas import (
    UnifiedAtlasInventory,
    MultiAtlasTriangulationScorer,
    AtlasProbe,
)

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
    enable_spectral_penalty: bool = True
    spectral_penalty_strength: float = 0.5

    # --- 4.3: SVD-Aware Blending ---
    enable_svd_blending: bool = True
    svd_rank_ratio: float = 0.1
    high_rank_alpha: float = 0.3  # Trust source for skills
    low_rank_alpha: float = 0.7   # Trust target for structure

    # --- 4.4: Correlation-Based Dimension Weights ---
    enable_correlation_weights: bool = True
    correlation_scale: float = 5.0
    stability_alpha: float = 0.7  # Used when dimensions disagree

    # --- 4.5: VerbNoun Modulation ---
    enable_verb_noun: bool = True
    verb_noun_strength: float = 0.7

    # --- 4.6: Domain Signals ---
    enable_domain_signals: bool = False
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
    module_policy_v_alpha: float = 0.9   # Trust source for v_proj (skills)
    module_policy_o_alpha: float = 0.1   # Trust target for o_proj (structure)

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
    refinement_hard_swap_enabled: bool = True  # Allow full source replacement for highly refined layers

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
    output_quant: Optional[str] = None
    output_quant_group_size: Optional[int] = None

    # ==========================================================================
    # STAGE 0: VOCABULARY ALIGNMENT (Cross-Vocabulary Merging)
    # ==========================================================================

    # Enable vocabulary alignment for cross-vocabulary merging
    enable_vocabulary_alignment: bool = True

    # Method selection: "auto" picks based on overlap, or "fvt", "procrustes", "affine"
    vocab_bridge_method: str = "auto"

    # Minimum alignment quality to proceed (0.0-1.0)
    vocab_quality_threshold: float = 0.5

    # Overlap ratio above which vocabulary is considered compatible (skip bridge)
    vocab_compatible_threshold: float = 0.95

    # Minimum anchor pairs needed for Procrustes/Affine bridge
    vocab_min_anchor_pairs: int = 10

    # Use semantic primes as anchors for vocabulary alignment
    vocab_use_semantic_primes: bool = True

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
    omega_in: Optional[np.ndarray] = None

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

    # Output path (if saved)
    output_path: Optional[str] = None

    # Vocabulary alignment status
    vocab_aligned: bool = False


class UnifiedGeometricMerger:
    """
    The ONE geometric merge pipeline.

    Combines all techniques in the correct order:
    PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE
    """

    def __init__(self, config: Optional[UnifiedMergeConfig] = None):
        self.config = config or UnifiedMergeConfig.default()

    def merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: Optional[str] = None,
        source_fingerprints: Optional[dict] = None,
        target_fingerprints: Optional[dict] = None,
        source_tokenizer: Optional[Any] = None,
        target_tokenizer: Optional[Any] = None,
        base_model_path: Optional[str] = None,
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
        base_weights: Optional[dict[str, np.ndarray]] = None
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
        # Computes per-layer "knowledge mass" using:
        # - DARE sparsity: What fraction of weights are essential
        # - DoRA drift: How much the feature space rotated from base
        # High refinement density → layer learned new capabilities → trust source
        refinement_alphas: Optional[dict[int, float]] = None
        refinement_hard_swap_layers: set[int] = set()
        refinement_metrics: dict[str, Any] = {}

        if base_weights is not None and self.config.enable_refinement_density:
            logger.info("REFINEMENT DENSITY: Computing per-layer knowledge mass...")
            refinement_alphas, hard_swap_layers, refinement_metrics = self._compute_refinement_density(
                source_weights=source_weights,
                base_weights=base_weights,
                source_path=source_path,
                base_path=base_model_path or "",
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
        # STAGE 0: VOCABULARY ALIGNMENT (if enabled and tokenizers provided)
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
        )

        # Extract the IntersectionMap object (if built) for downstream stages
        from modelcypher.core.domain.geometry.manifold_stitcher import IntersectionMap
        intersection_map_obj: Optional[IntersectionMap] = probe_result.get("intersection_map")
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
            layer_indices,
        )

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
            mean_confidence=mean_confidence,
            mean_procrustes_error=float(mean_error),
            layer_count=len(layer_indices),
            weight_count=len(merged_weights),
            timestamp=datetime.utcnow(),
            output_path=output_path,
            vocab_aligned=vocab_aligned,
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
    # STAGE 0: VOCABULARY ALIGNMENT
    # =========================================================================

    def _stage_vocabulary_align(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        source_tokenizer: Optional[Any],
        target_tokenizer: Optional[Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
        """
        Stage 0: Align source vocabulary to target vocabulary.

        Detects vocabulary mismatch and applies embedding bridge if needed.
        Uses intelligent method selection: FVT → Procrustes → Affine.

        Returns:
            Tuple of (modified_source_weights, metrics, was_aligned)
        """
        metrics: dict[str, Any] = {
            "enabled": True,
            "tokenizers_provided": source_tokenizer is not None and target_tokenizer is not None,
        }

        # Skip if tokenizers not provided
        if source_tokenizer is None or target_tokenizer is None:
            logger.info("Tokenizers not provided, skipping vocabulary alignment")
            metrics["skipped"] = True
            metrics["reason"] = "tokenizers_not_provided"
            return source_weights, metrics, False

        # Find embedding layer keys
        embed_keys = [k for k in source_weights if "embed" in k.lower() and "weight" in k.lower()]
        if not embed_keys:
            logger.info("No embedding layer found, skipping vocabulary alignment")
            metrics["skipped"] = True
            metrics["reason"] = "no_embedding_layer"
            return source_weights, metrics, False

        # Import vocabulary alignment modules
        try:
            from modelcypher.core.domain.merging.vocabulary_alignment import (
                VocabularyAligner,
                VocabularyAlignmentConfig,
            )
            from modelcypher.core.domain.merging.embedding_bridge import (
                EmbeddingBridgeBuilder,
                EmbeddingBridgeConfig,
                BridgeMethod,
            )
        except ImportError as e:
            logger.warning("Vocabulary alignment modules not available: %s", e)
            metrics["skipped"] = True
            metrics["reason"] = f"import_error: {e}"
            return source_weights, metrics, False

        # Align vocabularies
        aligner = VocabularyAligner()
        alignment = aligner.align(source_tokenizer, target_tokenizer)

        metrics["source_vocab_size"] = alignment.source_vocab_size
        metrics["target_vocab_size"] = alignment.target_vocab_size
        metrics["overlap_count"] = alignment.overlap_count
        metrics["overlap_ratio"] = alignment.overlap_ratio
        metrics["coverage"] = alignment.coverage
        metrics["recommended_method"] = alignment.recommended_method
        metrics["merge_feasibility"] = alignment.merge_feasibility

        # Check if vocabulary is already compatible
        if alignment.overlap_ratio >= self.config.vocab_compatible_threshold:
            logger.info(
                "Vocabulary overlap %.1f%% >= %.1f%% threshold, no bridge needed",
                alignment.overlap_ratio * 100,
                self.config.vocab_compatible_threshold * 100,
            )
            metrics["bridge_applied"] = False
            metrics["reason"] = "compatible_vocabulary"
            return source_weights, metrics, False

        # Check feasibility
        if alignment.merge_feasibility == "infeasible":
            logger.warning("Vocabulary merge marked as infeasible (coverage: %.1f%%)", alignment.coverage * 100)
            metrics["bridge_applied"] = False
            metrics["reason"] = "infeasible_vocabulary"
            return source_weights, metrics, False

        # Configure bridge builder
        bridge_config = EmbeddingBridgeConfig(
            auto_select=(self.config.vocab_bridge_method == "auto"),
            quality_threshold=self.config.vocab_quality_threshold,
            min_anchor_pairs=self.config.vocab_min_anchor_pairs,
            use_semantic_primes=self.config.vocab_use_semantic_primes,
        )

        if self.config.vocab_bridge_method != "auto":
            bridge_config = EmbeddingBridgeConfig(
                auto_select=False,
                fallback_chain=(self.config.vocab_bridge_method,),
                quality_threshold=self.config.vocab_quality_threshold,
                min_anchor_pairs=self.config.vocab_min_anchor_pairs,
            )

        bridge_builder = EmbeddingBridgeBuilder(bridge_config)

        # Apply bridge to each embedding layer
        modified_weights = source_weights.copy()
        bridged_layers = 0

        for embed_key in embed_keys:
            source_embed = source_weights.get(embed_key)
            target_embed = target_weights.get(embed_key)

            if source_embed is None or target_embed is None:
                continue

            if source_embed.shape[1] != target_embed.shape[1]:
                logger.warning(
                    "Hidden dimension mismatch for %s: %d vs %d",
                    embed_key,
                    source_embed.shape[1],
                    target_embed.shape[1],
                )
                continue

            # Build embedding bridge
            result = bridge_builder.build(
                source_embed,
                target_embed,
                alignment,
            )

            if result.alignment_quality < self.config.vocab_quality_threshold:
                logger.warning(
                    "Bridge quality %.3f below threshold %.3f for %s",
                    result.alignment_quality,
                    self.config.vocab_quality_threshold,
                    embed_key,
                )
                metrics[f"{embed_key}_quality"] = result.alignment_quality
                metrics[f"{embed_key}_warnings"] = result.warnings
                continue

            modified_weights[embed_key] = result.bridged_embeddings
            bridged_layers += 1

            metrics[f"{embed_key}_method"] = result.method_used.value
            metrics[f"{embed_key}_quality"] = result.alignment_quality
            logger.info(
                "Applied %s bridge to %s (quality: %.3f)",
                result.method_used.value.upper(),
                embed_key,
                result.alignment_quality,
            )

        metrics["bridge_applied"] = bridged_layers > 0
        metrics["bridged_layers"] = bridged_layers

        return modified_weights, metrics, bridged_layers > 0

    # =========================================================================
    # STAGE 1: PROBE
    # =========================================================================

    def _stage_probe(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        source_fingerprints: Optional[dict],
        target_fingerprints: Optional[dict],
        source_model: Optional[Any] = None,
        target_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> tuple[dict, dict]:
        """
        Stage 1: Build intersection map from probe responses.

        The intersection map is the PRIMARY CONTROL SIGNAL for all
        downstream operations.

        Two modes:
        - "precise": Run 403 probes through BOTH models, compute CKA on activations
                     This is the CORRECT method per the theoretical foundation.
        - "fast": Use weight-level CKA (faster but less accurate)

        Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations"
        Reference: Moschella et al. (2023) "Relative Representations Enable Zero-Shot Transfer"
        """
        if self.config.probe_mode == "precise" and source_model is not None and target_model is not None:
            return self._stage_probe_precise(
                source_model=source_model,
                target_model=target_model,
                tokenizer=tokenizer,
                source_weights=source_weights,
                target_weights=target_weights,
            )
        else:
            if self.config.probe_mode == "precise":
                logger.warning(
                    "Precise mode requested but models not loaded. "
                    "Falling back to fast mode (weight-level CKA)."
                )
            return self._stage_probe_fast(
                source_weights=source_weights,
                target_weights=target_weights,
            )

    def _stage_probe_precise(
        self,
        source_model: Any,
        target_model: Any,
        tokenizer: Any,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        source_path: str = "",
        target_path: str = "",
    ) -> tuple[dict, dict]:
        """
        Precise probe mode: Run 403 probes through BOTH models.

        This is THE CORRECT method per the theoretical foundation:
        1. Get all probes from UnifiedAtlasInventory (403 probes)
        2. For each probe text:
           - Tokenize → run through source model → collect per-layer activations
           - Tokenize → run through target model → collect per-layer activations
           - Sparsify to top-32 dimensions → ActivationFingerprint
        3. Build IntersectionMap using ManifoldStitcher (dimension-level correlations)
        4. Extract layer confidences for downstream stages
        """
        import mlx.core as mx

        from modelcypher.core.domain.geometry.cka import compute_cka
        from modelcypher.core.domain.geometry.manifold_stitcher import (
            ActivatedDimension,
            ActivationFingerprint,
            IntersectionMap,
            IntersectionSimilarityMode,
            build_intersection_map,
        )

        probes = UnifiedAtlasInventory.all_probes()
        num_probes = len(probes)

        # Limit probes if configured (for faster testing)
        if self.config.max_probes > 0 and self.config.max_probes < num_probes:
            probes = probes[: self.config.max_probes]
            logger.info(
                "PROBE PRECISE: Limited to %d/%d probes (max_probes=%d)",
                len(probes),
                num_probes,
                self.config.max_probes,
            )

        logger.info(
            "PROBE PRECISE: Running %d probes through source and target models...",
            len(probes),
        )

        # Build ActivationFingerprints for each probe
        source_fingerprints: list[ActivationFingerprint] = []
        target_fingerprints: list[ActivationFingerprint] = []

        # Also collect raw activations for CKA (fallback/validation)
        source_layer_activations: dict[int, list[np.ndarray]] = {}
        target_layer_activations: dict[int, list[np.ndarray]] = {}

        probes_processed = 0
        probes_failed = 0

        for probe in probes:
            # Use the first support text as the probe input
            probe_texts = probe.support_texts
            if not probe_texts:
                probes_failed += 1
                continue

            probe_text = probe_texts[0]
            if not probe_text or len(probe_text.strip()) < 2:
                probes_failed += 1
                continue

            try:
                # Get activations from source model
                source_acts = self._collect_layer_activations(
                    source_model, tokenizer, probe_text
                )
                # Get activations from target model
                target_acts = self._collect_layer_activations(
                    target_model, tokenizer, probe_text
                )

                # Build ActivationFingerprint for this probe (sparsify to top-32)
                source_activated: dict[int, list[ActivatedDimension]] = {}
                target_activated: dict[int, list[ActivatedDimension]] = {}

                for layer_idx, act in source_acts.items():
                    source_activated[layer_idx] = self._extract_top_k_dims(act, k=32)
                    # Also aggregate for CKA fallback
                    if layer_idx not in source_layer_activations:
                        source_layer_activations[layer_idx] = []
                    source_layer_activations[layer_idx].append(act)

                for layer_idx, act in target_acts.items():
                    target_activated[layer_idx] = self._extract_top_k_dims(act, k=32)
                    if layer_idx not in target_layer_activations:
                        target_layer_activations[layer_idx] = []
                    target_layer_activations[layer_idx].append(act)

                # Create ActivationFingerprint objects
                source_fingerprints.append(ActivationFingerprint(
                    prime_id=probe.probe_id,
                    prime_text=probe.name,
                    activated_dimensions=source_activated,
                ))
                target_fingerprints.append(ActivationFingerprint(
                    prime_id=probe.probe_id,
                    prime_text=probe.name,
                    activated_dimensions=target_activated,
                ))

                probes_processed += 1

                if probes_processed % 50 == 0:
                    logger.info(
                        "PROBE PRECISE: Processed %d/%d probes...",
                        probes_processed,
                        len(probes),
                    )

            except Exception as e:
                logger.debug("Probe '%s' failed: %s", probe.probe_id, e)
                probes_failed += 1
                continue

        logger.info(
            "PROBE PRECISE: Completed %d probes (%d failed), built %d fingerprints",
            probes_processed,
            probes_failed,
            len(source_fingerprints),
        )

        # =================================================================
        # BUILD INTERSECTION MAP using ManifoldStitcher
        # =================================================================
        # This is the REAL geometric signal - dimension-level correlations
        intersection_map_obj: Optional[IntersectionMap] = None
        dimension_correlations: dict = {}

        if source_fingerprints and target_fingerprints:
            try:
                intersection_map_obj = build_intersection_map(
                    source_fingerprints=source_fingerprints,
                    target_fingerprints=target_fingerprints,
                    source_model=source_path or "source",
                    target_model=target_path or "target",
                    mode=IntersectionSimilarityMode.ENSEMBLE,
                    correlation_threshold=0.3,
                )
                dimension_correlations = intersection_map_obj.dimension_correlations
                logger.info(
                    "PROBE PRECISE: Built IntersectionMap with overall_correlation=%.3f, %d layers",
                    intersection_map_obj.overall_correlation,
                    len(intersection_map_obj.layer_confidences),
                )
            except Exception as e:
                logger.warning("Failed to build IntersectionMap: %s", e)
                intersection_map_obj = None

        # Extract layer confidences from IntersectionMap (or compute from CKA as fallback)
        layer_confidences: dict[int, float] = {}
        layer_cka_scores: dict[int, float] = {}

        if intersection_map_obj is not None:
            # Use IntersectionMap layer confidences (THE CORRECT SOURCE)
            for lc in intersection_map_obj.layer_confidences:
                layer_confidences[lc.layer] = lc.confidence
        else:
            # Fallback: Compute CKA between source and target activations per layer
            common_layers = set(source_layer_activations.keys()) & set(
                target_layer_activations.keys()
            )

            for layer_idx in sorted(common_layers):
                source_acts_list = source_layer_activations[layer_idx]
                target_acts_list = target_layer_activations[layer_idx]

                # Ensure same number of probe responses
                n_samples = min(len(source_acts_list), len(target_acts_list))
                if n_samples < 10:
                    logger.warning(
                        "Layer %d has only %d probe responses, skipping",
                        layer_idx,
                        n_samples,
                    )
                    continue

                # Stack activations: [n_probes, hidden_dim]
                source_stack = np.stack(source_acts_list[:n_samples], axis=0)
                target_stack = np.stack(target_acts_list[:n_samples], axis=0)

                # Compute CKA between source and target activations for this layer
                try:
                    cka_result = compute_cka(
                        source_stack, target_stack, use_linear_kernel=True
                    )
                    cka_score = cka_result.cka if cka_result.is_valid else 0.0
                    layer_cka_scores[layer_idx] = float(cka_score)
                    layer_confidences[layer_idx] = float(cka_score)
                except Exception as e:
                    logger.debug("CKA failed for layer %d: %s", layer_idx, e)
                    layer_confidences[layer_idx] = 0.0

        # Build per-weight correlation map from layer confidences
        weight_correlations: dict[str, float] = {}
        for key in target_weights:
            if key not in source_weights:
                continue
            layer_idx = self._extract_layer_index(key)
            if layer_idx is not None and layer_idx in layer_confidences:
                weight_correlations[key] = layer_confidences[layer_idx]
            else:
                weight_correlations[key] = 0.0

        mean_confidence = (
            float(np.mean(list(layer_confidences.values())))
            if layer_confidences
            else 0.0
        )
        mean_cka = (
            float(np.mean(list(layer_cka_scores.values())))
            if layer_cka_scores
            else 0.0
        )

        metrics = {
            "probe_mode": "precise",
            "probes_total": len(probes),
            "probes_processed": probes_processed,
            "probes_failed": probes_failed,
            "fingerprints_built": len(source_fingerprints),
            "layers_analyzed": len(layer_confidences),
            "layer_confidences": layer_confidences,
            "layer_cka_scores": layer_cka_scores,
            "mean_confidence": mean_confidence,
            "mean_cka": mean_cka,
            "min_confidence": (
                min(layer_confidences.values()) if layer_confidences else 0.0
            ),
            "max_confidence": (
                max(layer_confidences.values()) if layer_confidences else 0.0
            ),
            "atlas_sources": list(set(p.source.value for p in probes)),
            "atlas_domains": list(set(p.domain.value for p in probes)),
            "intersection_map_built": intersection_map_obj is not None,
            "overall_correlation": (
                intersection_map_obj.overall_correlation
                if intersection_map_obj
                else 0.0
            ),
        }

        logger.info(
            "PROBE PRECISE: %d layers, mean_confidence=%.3f, overall_correlation=%.3f",
            len(layer_confidences),
            mean_confidence,
            metrics["overall_correlation"],
        )

        return {
            "correlations": weight_correlations,
            "confidences": layer_confidences,
            "intersection_map": intersection_map_obj,  # Full IntersectionMap object
            "dimension_correlations": dimension_correlations,  # Per-layer dimension correlations
        }, metrics

    def _extract_top_k_dims(
        self,
        activation_vector: np.ndarray,
        k: int = 32,
        threshold: float = 0.01,
    ) -> list:
        """
        Extract top-k activated dimensions by magnitude.

        This sparsifies the dense activation vector into ActivatedDimension objects
        for use in ManifoldStitcher's dimension-level correlation analysis.

        Args:
            activation_vector: Dense activation vector [hidden_dim]
            k: Number of top dimensions to extract
            threshold: Minimum activation magnitude to include

        Returns:
            List of ActivatedDimension objects sorted by index
        """
        from modelcypher.core.domain.geometry.manifold_stitcher import ActivatedDimension

        abs_vals = np.abs(activation_vector)
        top_indices = np.argsort(-abs_vals)[:k]

        return [
            ActivatedDimension(
                index=int(idx),
                activation=float(activation_vector[idx]),
            )
            for idx in sorted(top_indices)
            if abs_vals[idx] > threshold
        ]

    def _stage_probe_fast(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
    ) -> tuple[dict, dict]:
        """
        Fast probe mode: Use weight-level CKA (no model inference).

        This is faster but less accurate than precise mode.
        Computes similarity directly on weight matrices.
        """
        from modelcypher.core.domain.geometry.cka import (
            compute_layer_cka,
            ensemble_similarity,
        )

        intersection_map = {}
        layer_confidences: dict[int, list[float]] = {}
        cka_scores = {}
        cosine_scores = {}

        for key in target_weights:
            if key not in source_weights:
                continue

            source_w = source_weights[key]
            target_w = target_weights[key]

            if source_w.shape != target_w.shape:
                continue

            layer_idx = self._extract_layer_index(key)
            if layer_idx is None:
                continue

            # Compute CKA for 2D weight matrices
            max_cka_dim = 512
            can_compute_cka = (
                self.config.intersection_mode != "jaccard"
                and source_w.ndim == 2
                and source_w.shape[0] >= 2
                and source_w.shape[0] <= max_cka_dim
            )
            if can_compute_cka:
                try:
                    cka_result = compute_layer_cka(source_w, target_w)
                    cka_score = cka_result.cka if cka_result.is_valid else 0.0
                except Exception:
                    cka_score = 0.0
            else:
                cka_score = 0.0

            # Compute cosine similarity
            s_flat = source_w.flatten().astype(np.float32)
            t_flat = target_w.flatten().astype(np.float32)
            s_norm = np.linalg.norm(s_flat)
            t_norm = np.linalg.norm(t_flat)

            if s_norm > 1e-8 and t_norm > 1e-8:
                cosine = float(np.dot(s_flat, t_flat) / (s_norm * t_norm))
            else:
                cosine = 0.0

            # Approximate Jaccard from weight overlap
            threshold = 0.01 * max(np.abs(source_w).max(), np.abs(target_w).max())
            s_active = np.abs(source_w) > threshold
            t_active = np.abs(target_w) > threshold
            intersection = np.sum(s_active & t_active)
            union = np.sum(s_active | t_active)
            jaccard = float(intersection / max(union, 1))

            # Ensemble similarity
            if self.config.intersection_mode == "cka":
                confidence = cka_score
            elif self.config.intersection_mode == "jaccard":
                confidence = jaccard
            else:  # ensemble (default)
                confidence = ensemble_similarity(
                    jaccard=jaccard,
                    cka=cka_score,
                    cosine=cosine,
                    jaccard_weight=0.6,
                    cka_weight=0.4,
                )

            intersection_map[key] = float(confidence)
            cka_scores[key] = cka_score
            cosine_scores[key] = cosine

            if layer_idx not in layer_confidences:
                layer_confidences[layer_idx] = []
            layer_confidences[layer_idx].append(float(confidence))

        # Compute per-layer confidence (mean of all weights in layer)
        layer_confidences_final: dict[int, float] = {}
        for layer_idx in layer_confidences:
            layer_confidences_final[layer_idx] = float(
                np.mean(layer_confidences[layer_idx])
            )

        mean_confidence = (
            float(np.mean(list(layer_confidences_final.values())))
            if layer_confidences_final
            else 0.0
        )
        mean_cka = float(np.mean(list(cka_scores.values()))) if cka_scores else 0.0

        metrics = {
            "probe_mode": "fast",
            "weight_count": len(intersection_map),
            "layer_confidences": layer_confidences_final,
            "mean_confidence": mean_confidence,
            "mean_cka": mean_cka,
            "min_confidence": (
                min(layer_confidences_final.values()) if layer_confidences_final else 0.0
            ),
            "max_confidence": (
                max(layer_confidences_final.values()) if layer_confidences_final else 0.0
            ),
            "intersection_mode": self.config.intersection_mode,
        }

        logger.info(
            "PROBE FAST: %d weights, mean_confidence=%.3f, mean_cka=%.3f",
            len(intersection_map),
            mean_confidence,
            mean_cka,
        )

        return {"correlations": intersection_map, "confidences": layer_confidences_final}, metrics

    def _collect_layer_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
    ) -> dict[int, np.ndarray]:
        """
        Collect per-layer hidden state activations for a text input.

        Runs the text through the model and extracts the final hidden state
        (mean-pooled over sequence length) at each layer.

        Args:
            model: Loaded MLX model with forward_with_cache or similar
            tokenizer: Tokenizer for encoding text
            text: Input text to process

        Returns:
            Dict mapping layer_idx -> activation vector [hidden_dim]
        """
        import mlx.core as mx

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            input_ids = mx.array([tokens])
        else:
            input_ids = mx.array([tokens.ids])

        # Run through model and collect hidden states
        activations: dict[int, np.ndarray] = {}

        try:
            # Try model.forward_with_hidden_states if available
            if hasattr(model, "forward_with_hidden_states"):
                _, hidden_states = model.forward_with_hidden_states(input_ids)
                for layer_idx, hidden in enumerate(hidden_states):
                    # Mean pool over sequence length: [batch, seq_len, hidden] -> [hidden]
                    pooled = mx.mean(hidden, axis=(0, 1))
                    mx.eval(pooled)
                    activations[layer_idx] = np.array(pooled)
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                # Manual collection through transformer layers
                # Get embeddings
                if hasattr(model.model, "embed_tokens"):
                    h = model.model.embed_tokens(input_ids)
                elif hasattr(model.model, "wte"):
                    h = model.model.wte(input_ids)
                else:
                    # Fallback: use the model's embedding method
                    h = model.embed(input_ids) if hasattr(model, "embed") else None

                if h is not None:
                    # Pass through each layer
                    for layer_idx, layer in enumerate(model.model.layers):
                        h, _ = layer(h)  # Most transformers return (hidden, cache)
                        # Mean pool and store
                        pooled = mx.mean(h, axis=(0, 1))
                        mx.eval(pooled)
                        activations[layer_idx] = np.array(pooled)
            else:
                # Last resort: just run forward and hope for the best
                output = model(input_ids)
                mx.eval(output)
                # Use final output as single "layer"
                pooled = mx.mean(output, axis=(0, 1))
                mx.eval(pooled)
                activations[0] = np.array(pooled)

        except Exception as e:
            logger.debug("Activation collection failed for text '%s...': %s", text[:30], e)

        return activations

    # =========================================================================
    # STAGE 2: PERMUTE
    # =========================================================================

    def _stage_permute(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        intersection_map_obj: Optional[Any],
        layer_confidences: dict[int, float],
        layer_indices: list[int],
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        Stage 2: Permutation alignment for MLP neurons.

        Uses PermutationAligner to solve the permutation symmetry problem.
        Only applies when layer confidence >= threshold.

        Mathematical Foundation:
            Neural networks have N! permutation symmetries per MLP layer.
            We find P, S such that W_aligned = S @ P @ W @ P^T @ S^T
            This aligns source neurons to target neuron ordering.

        Args:
            source_weights: Source model weights
            target_weights: Target model weights
            intersection_map_obj: IntersectionMap object (dimension-level correlations)
            layer_confidences: Per-layer confidence scores from probing
            layer_indices: List of layer indices to process

        Reference: Ainsworth et al. (2022) "Git Re-Basin"
        """
        import mlx.core as mx
        from modelcypher.core.domain.geometry.permutation_aligner import (
            PermutationAligner,
            Config as PAConfig,
        )

        if not self.config.enable_permutation:
            logger.info("PERMUTE: Disabled")
            return source_weights, {"skipped": True}

        # Use IntersectionMap dimension correlations for targeted permutation if available
        dimension_correlations = {}
        if intersection_map_obj is not None:
            dimension_correlations = intersection_map_obj.dimension_correlations

        # Convert numpy weights to MLX arrays
        source_mx: dict[str, mx.array] = {}
        target_mx: dict[str, mx.array] = {}

        for key, val in source_weights.items():
            source_mx[key] = mx.array(val.astype(np.float32))
        for key, val in target_weights.items():
            target_mx[key] = mx.array(val.astype(np.float32))

        # Build anchor embeddings from model's embedding layer
        # Anchors ground neuron signatures for alignment
        anchor_key = None
        for key in target_weights:
            if "embed_tokens" in key or "wte" in key or "embedding" in key.lower():
                anchor_key = key
                break

        if anchor_key is not None:
            # Use embedding rows as semantic anchors
            embed = target_weights[anchor_key]
            # Sample subset of embeddings (e.g., first 128 tokens)
            num_anchors = min(128, embed.shape[0])
            anchors = mx.array(embed[:num_anchors].astype(np.float32))
            logger.info("PERMUTE: Using %d embedding anchors from %s", num_anchors, anchor_key)
        else:
            # Fallback: create random anchors matching hidden dimension
            hidden_dim = self._infer_hidden_dim(target_weights)
            anchors = mx.random.normal((64, hidden_dim)) * 0.1
            logger.warning("PERMUTE: No embedding found, using random anchors (dim=%d)", hidden_dim)

        # Check mean confidence to decide if alignment is worthwhile
        mean_confidence = np.mean(list(layer_confidences.values())) if layer_confidences else 0.0
        if mean_confidence < self.config.permutation_confidence_threshold:
            logger.info("PERMUTE: Skipped (mean confidence %.3f < threshold %.3f)",
                       mean_confidence, self.config.permutation_confidence_threshold)
            return source_weights, {
                "skipped": True,
                "reason": "low_confidence",
                "mean_confidence": float(mean_confidence),
            }

        # Configure aligner
        pa_config = PAConfig(
            min_match_threshold=0.1,
            use_anchor_grounding=True,
        )

        # Run MLP re-basin alignment
        try:
            aligned_mx, mean_quality, blocks_aligned = PermutationAligner.rebasin_mlp_with_activations(
                source_mx,
                target_mx,
                anchors,
                anchor_activations=None,  # No per-layer activations in unified mode
                config=pa_config,
            )
            mx.eval(aligned_mx)

            # Convert back to numpy
            permuted: dict[str, np.ndarray] = {}
            for key, val in aligned_mx.items():
                permuted[key] = np.asarray(val)

            logger.info(
                "PERMUTE: Aligned %d MLP blocks, mean quality=%.3f",
                blocks_aligned, mean_quality,
            )

            metrics = {
                "layers_permuted": blocks_aligned,
                "mean_quality": float(mean_quality),
                "threshold": self.config.permutation_confidence_threshold,
                "mean_confidence": float(mean_confidence),
            }

            return permuted, metrics

        except Exception as e:
            logger.warning("PERMUTE: Alignment failed (%s), returning original weights", e)
            return source_weights, {
                "skipped": True,
                "reason": "alignment_failed",
                "error": str(e),
            }

    def _infer_hidden_dim(self, weights: dict[str, np.ndarray]) -> int:
        """Infer hidden dimension from weight shapes."""
        for key, val in weights.items():
            if "q_proj" in key or "k_proj" in key:
                # Attention projection: [heads*head_dim, hidden_dim]
                return val.shape[1]
            if "up_proj" in key or "gate_proj" in key:
                # MLP projection: [intermediate, hidden_dim]
                return val.shape[1]
        # Fallback
        return 4096

    # =========================================================================
    # STAGES 3-5: ROTATE + BLEND + PROPAGATE
    # =========================================================================

    def _stage_rotate_blend_propagate(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        intersection_map_obj: Optional[Any],
        layer_confidences: dict[int, float],
        dimension_correlations: dict,
        layer_indices: list[int],
        refinement_alphas: Optional[dict[int, float]] = None,
        hard_swap_layers: Optional[set[int]] = None,
    ) -> tuple[dict[str, np.ndarray], dict, dict]:
        """
        Stages 3-5 merged into single loop for efficiency.

        For each layer:
        1. ROTATE: Compute/apply geometric alignment (Procrustes or GW Transport)
        2. BLEND: Compute multi-layer alpha with all adjustments
        3. PROPAGATE: Carry rotation to next layer (zipper)

        When use_transport_guided=True, uses Gromov-Wasserstein optimal transport
        instead of Procrustes rotation. GW computes soft correspondence between
        neurons based on their relational structure, then blends using the coupling.

        Args:
            source_weights: Permuted source model weights
            target_weights: Target model weights
            intersection_map_obj: IntersectionMap object with dimension-level correlations
            layer_confidences: Per-layer confidence scores from probing
            dimension_correlations: Per-layer dimension correlation data
            layer_indices: List of layer indices to process
            refinement_alphas: Per-layer alphas from refinement density (Law #5)
            hard_swap_layers: Layers to hard-swap (full source) due to high refinement

        Blend adjustments applied in sequence:
        1. Base alpha from intersection confidence
        2. Gaussian smoothing across layers
        3. Spectral penalty for ill-conditioned weights
        4. SVD-aware blending (different alpha for high/low rank)
        5. Correlation-based dimension weights (from IntersectionMap)
        6. VerbNoun modulation
        """
        from modelcypher.core.domain.geometry.alpha_smoothing import (
            AlphaSmoothingConfig,
            gaussian_smooth_alpha_profile,
        )
        from modelcypher.core.domain.geometry.spectral_analysis import (
            SpectralConfig,
            compute_spectral_metrics,
            apply_spectral_penalty,
        )
        from modelcypher.core.domain.geometry.task_singular_vectors import (
            SVDBlendConfig,
            blend_with_svd_awareness,
        )
        from modelcypher.core.domain.geometry.verb_noun_classifier import (
            VerbNounConfig,
        )
        from modelcypher.core.domain.geometry.transport_guided_merger import (
            TransportGuidedMerger,
        )

        # Use IntersectionMap for dimension-aware blending if available
        has_dimension_correlations = intersection_map_obj is not None and bool(dimension_correlations)

        # Pre-compute smoothed alpha profile
        # If refinement_alphas provided (from Validated Law #5), use them as base.
        # Otherwise, compute from intersection confidence.
        raw_alphas = {}
        for layer_idx in layer_indices:
            if refinement_alphas is not None and layer_idx in refinement_alphas:
                # Use refinement density alpha (DARE + DoRA analysis)
                # Low alpha = trust source more (layer is refined)
                # High alpha = trust target more (layer unchanged)
                base_alpha = refinement_alphas[layer_idx]
            else:
                # Fallback: compute from intersection confidence
                confidence = layer_confidences.get(layer_idx, 0.0)
                # High confidence → trust source more → lower alpha
                base_alpha = 1.0 - (confidence * 0.7)

            # Apply refinement density modulation strength
            if refinement_alphas is not None and layer_idx in refinement_alphas:
                # Blend between original confidence-based and refinement-based
                confidence = layer_confidences.get(layer_idx, 0.0)
                conf_alpha = 1.0 - (confidence * 0.7)
                raw_alphas[layer_idx] = (
                    self.config.refinement_density_strength * base_alpha +
                    (1.0 - self.config.refinement_density_strength) * conf_alpha
                )
            else:
                raw_alphas[layer_idx] = base_alpha

        if self.config.enable_alpha_smoothing:
            smoothing_config = AlphaSmoothingConfig(
                smoothing_window=self.config.smoothing_window,
                sigma=self.config.smoothing_sigma,
            )
            smoothed_alphas = gaussian_smooth_alpha_profile(raw_alphas, smoothing_config)
        else:
            smoothed_alphas = raw_alphas

        # SVD config
        svd_config = SVDBlendConfig(
            rank_ratio=self.config.svd_rank_ratio,
            high_rank_alpha=self.config.high_rank_alpha,
            low_rank_alpha=self.config.low_rank_alpha,
        ) if self.config.enable_svd_blending else None

        # Spectral config
        spectral_config = SpectralConfig(
            penalty_strength=self.config.spectral_penalty_strength,
        )

        # VerbNoun config
        verb_noun_config = VerbNounConfig(
            verb_alpha=0.8,  # Trust Source for skills (high alpha)
            noun_alpha=0.2,  # Trust Target for knowledge (low alpha)
            modulation_strength=self.config.verb_noun_strength,
        ) if self.config.enable_verb_noun else None

        # Merge state (zipper)
        # Track omega rotations per layer for propagation
        # omega_layer[layer_idx] = rotation matrix from that layer's residual outputs
        omega_by_layer: dict[int, np.ndarray] = {}

        # Start with target weights
        merged = {k: np.asarray(v) for k, v in target_weights.items()}

        rotate_metrics = {
            "procrustes_errors": [],
            "rotations_applied": 0,
            "identity_used": 0,
            "transport_guided_applied": 0,
            "gw_distances": [],
            "zipper_propagations": 0,
            "zipper_applications": 0,
        }
        blend_metrics = {
            "effective_alphas": [],
            "spectral_adjustments": 0,
            "svd_blended": 0,
            "correlation_weighted": 0,
            "verb_noun_modulated": 0,
            "hard_swaps": 0,  # Layers fully replaced from source (Validated Law #5)
        }

        # Initialize hard_swap_layers if not provided
        if hard_swap_layers is None:
            hard_swap_layers = set()

        # Process each weight
        total_weights = len(target_weights)
        processed = 0
        for key in sorted(target_weights.keys()):
            if key not in source_weights:
                continue

            processed += 1
            if processed % 100 == 0:
                logger.info("BLEND: processed %d/%d weights", processed, total_weights)

            source_w = np.asarray(source_weights[key], dtype=np.float32)
            target_w = np.asarray(target_weights[key], dtype=np.float32)

            if source_w.shape != target_w.shape:
                continue

            layer_idx = self._extract_layer_index(key)
            confidence = layer_confidences.get(layer_idx, 0.0) if layer_idx is not None else 0.0

            # Get base alpha
            if layer_idx is not None and layer_idx in smoothed_alphas:
                effective_alpha = smoothed_alphas[layer_idx]
            else:
                effective_alpha = self.config.base_alpha

            # STAGE 3: ROTATE (Procrustes or GW Transport geometric alignment)
            omega_out = None
            procrustes_error = 0.0
            transport_blended = None  # If GW transport produces blended weights

            # Only rotate 2D weights with sufficient dimensions
            can_rotate = (
                self.config.enable_rotation
                and confidence >= self.config.rotation_confidence_threshold
                and source_w.ndim == 2
                and target_w.ndim == 2
                and min(source_w.shape) >= self.config.alignment_rank
            )

            # Check for transport-guided merge (GW optimal transport)
            can_transport = (
                self.config.use_transport_guided
                and can_rotate
                and source_w.shape[0] <= 512  # GW is O(n²m²), limit size
            )

            if can_transport:
                # Use Gromov-Wasserstein transport instead of Procrustes
                # GW computes soft correspondence π[i,j] between neurons
                # based on their relational structure (distance matrices)
                gw_result = self._compute_transport_guided_blend(
                    source_w, target_w, effective_alpha
                )
                if gw_result is not None:
                    transport_blended, gw_distance = gw_result
                    rotate_metrics["transport_guided_applied"] += 1
                    rotate_metrics["gw_distances"].append(gw_distance)
                else:
                    # Fallback to Procrustes if GW fails
                    can_transport = False

            if can_rotate and not can_transport:
                # Compute Procrustes rotation for this weight
                omega_out, procrustes_error = self._compute_procrustes_rotation(
                    source_w, target_w, rank=self.config.alignment_rank
                )
                rotate_metrics["procrustes_errors"].append(procrustes_error)
                rotate_metrics["rotations_applied"] += 1

                # Note: This low-rank Procrustes rotation aligns individual weights.
                # Cross-layer propagation is handled separately in STAGE 5 (zipper)
                # using full-rank permutations or rotations for residual outputs.

            elif not can_transport:
                rotate_metrics["identity_used"] += 1

            # STAGE 5: PROPAGATE (zipper - track transformation for next layer)
            # The geometric zipper propagates permutations/rotations layer-to-layer:
            # 1. Residual outputs (o_proj, down_proj) compute and apply P or R
            # 2. The transformation is stored and applied to next layer's inputs
            #
            # Mathematical Foundation (Git Re-Basin, Ainsworth et al., 2022):
            #   W_ℓ' = P @ W_ℓ           (permute output neurons)
            #   W_{ℓ+1}' = W_{ℓ+1} @ P^T  (permute input neurons of next layer)
            #
            # For permutation P: P^T = P^{-1}, so this maintains functional equivalence.
            # For orthogonal R: R^T = R^{-1}, generalizing to continuous rotations.

            if self.config.enable_zipper and layer_idx is not None:
                is_residual = self._is_residual_output(key)
                is_input_proj = self._is_attention_input(key) or self._is_mlp_input(key)

                # For residual outputs: compute and apply transformation, store for next layer
                if is_residual and source_w.ndim == 2 and source_w.shape[0] == target_w.shape[0]:
                    out_dim = source_w.shape[0]

                    if self.config.zipper_use_weight_matching:
                        # Weight matching: compute permutation matrix P
                        P = self._compute_weight_matching_permutation(source_w, target_w)
                        # Apply: source' = P @ source (permute output neurons)
                        source_w = P @ source_w
                        omega_by_layer[layer_idx] = P
                        rotate_metrics["zipper_propagations"] += 1
                        logger.debug("ZIPPER: layer %d residual - permutation computed (dim=%d)", layer_idx, out_dim)
                    else:
                        # Full-rank orthogonal rotation
                        R, error = self._compute_full_rank_rotation(source_w, target_w)
                        source_w = R @ source_w
                        omega_by_layer[layer_idx] = R
                        rotate_metrics["zipper_propagations"] += 1
                        rotate_metrics["procrustes_errors"].append(error)
                        logger.debug("ZIPPER: layer %d residual - rotation computed (dim=%d, error=%.4f)", layer_idx, out_dim, error)

                # For input projections: apply transformation from previous layer
                elif is_input_proj and source_w.ndim == 2:
                    prev_layer = layer_idx - 1
                    if prev_layer in omega_by_layer:
                        omega_in = omega_by_layer[prev_layer]
                        # Apply input rotation: W' = W @ P^T (or R^T)
                        # Check dimension compatibility
                        if omega_in.shape[0] == source_w.shape[1]:
                            source_w = source_w @ omega_in.T
                            rotate_metrics["zipper_applications"] += 1
                            logger.debug("ZIPPER: layer %d input - transformation applied", layer_idx)
                        else:
                            logger.warning(
                                "ZIPPER: dimension mismatch at layer %d: omega=%s, weight input=%d",
                                layer_idx, omega_in.shape, source_w.shape[1]
                            )

            # STAGE 4: BLEND

            # 4.0: Hard swap check (Validated Law #5: Refinement Density)
            # If layer is marked for hard swap, take entirely from source
            if layer_idx is not None and layer_idx in hard_swap_layers:
                # Full source replacement - layer is highly refined
                merged[key] = source_w.astype(target_w.dtype)
                blend_metrics["hard_swaps"] += 1
                blend_metrics["effective_alphas"].append(0.0)  # 0 = all source
                continue

            # 4.1: Module-specific policy
            # Different modules require different blending strategies
            if self.config.enable_module_policy:
                if self._is_v_proj(key):
                    # v_proj: Captures what gets attended to, trust source for skills
                    effective_alpha = self.config.module_policy_v_alpha
                    blend_metrics.setdefault("module_policy_v", 0)
                    blend_metrics["module_policy_v"] += 1
                elif self._is_o_proj(key):
                    # o_proj: Projects back to residual, preserve target structure
                    effective_alpha = self.config.module_policy_o_alpha
                    blend_metrics.setdefault("module_policy_o", 0)
                    blend_metrics["module_policy_o"] += 1

            # 4.2: Spectral penalty
            if self.config.enable_spectral_penalty and source_w.ndim >= 1:
                spectral = compute_spectral_metrics(source_w, target_w, spectral_config)
                effective_alpha = apply_spectral_penalty(
                    effective_alpha,
                    spectral.spectral_confidence,
                    self.config.spectral_penalty_strength,
                )
                blend_metrics["spectral_adjustments"] += 1

            # 4.3: SVD-aware blending (or use transport-blended if available)
            if transport_blended is not None:
                # GW transport already produced blended weights
                blended = transport_blended
            elif svd_config is not None and source_w.ndim == 2:
                blended = blend_with_svd_awareness(
                    source_w, target_w, effective_alpha, svd_config
                )
                blend_metrics["svd_blended"] += 1
            else:
                # Simple linear blend
                blended = (1.0 - effective_alpha) * target_w + effective_alpha * source_w

            # 4.4: Correlation-based dimension weighting
            # For 2D weights, compute per-dimension alpha adjustments
            if self.config.enable_correlation_weights and source_w.ndim == 2:
                # Use weight rows as activation proxies
                from modelcypher.core.domain.geometry.dimension_blender import (
                    CorrelationWeightConfig,
                    compute_dimension_correlations,
                    compute_correlation_weights,
                )

                corr_config = CorrelationWeightConfig(
                    correlation_scale=self.config.correlation_scale,
                    stability_alpha=self.config.stability_alpha,
                )

                # Sample rows to compute correlation (limit for efficiency)
                sample_rows = min(source_w.shape[0], 256)
                source_sample = source_w[:sample_rows, :]
                target_sample = target_w[:sample_rows, :]

                if source_sample.shape == target_sample.shape and source_sample.shape[0] > 1:
                    try:
                        correlations = compute_dimension_correlations(
                            source_sample.T, target_sample.T, corr_config
                        )
                        corr_weights = compute_correlation_weights(correlations, corr_config)

                        # Apply correlation-based modulation per-dimension
                        # corr_weights are per-row, apply to blend
                        if len(corr_weights) == blended.shape[0]:
                            # Modulate effective_alpha per row
                            row_alphas = (
                                (1.0 - corr_weights) * effective_alpha +
                                corr_weights * self.config.stability_alpha
                            )
                            # Re-blend with per-row alphas
                            blended = (
                                (1.0 - row_alphas[:, np.newaxis]) * target_w +
                                row_alphas[:, np.newaxis] * source_w
                            )
                            blend_metrics["correlation_weighted"] += 1
                    except Exception as e:
                        logger.debug("Correlation weighting failed for %s: %s", key, e)

            # 4.5: VerbNoun modulation
            # Apply verb/noun alpha modulation based on dimension classification
            if verb_noun_config is not None and source_w.ndim == 2:
                # Simple heuristic: use variance ratio as verb/noun proxy
                # High variance dimensions → verb-like → trust source
                # Low variance dimensions → noun-like → trust target
                source_var = np.var(source_w, axis=1)
                target_var = np.var(target_w, axis=1)

                # Ratio > 1 means source has more variance (verb-like)
                var_ratio = source_var / (target_var + 1e-8)

                # Create per-dimension alpha based on verb/noun
                verb_mask = var_ratio > 2.0  # High variance → verb
                noun_mask = var_ratio < 0.5  # Low variance → noun

                vn_alphas = np.full(source_w.shape[0], effective_alpha, dtype=np.float32)
                vn_alphas[verb_mask] = verb_noun_config.verb_alpha
                vn_alphas[noun_mask] = verb_noun_config.noun_alpha

                # Modulate: blend current alpha with VN alpha
                modulated_alphas = (
                    (1.0 - verb_noun_config.modulation_strength) * effective_alpha +
                    verb_noun_config.modulation_strength * vn_alphas
                )

                # Re-blend with VN-modulated alphas
                blended = (
                    (1.0 - modulated_alphas[:, np.newaxis]) * target_w +
                    modulated_alphas[:, np.newaxis] * source_w
                )
                blend_metrics["verb_noun_modulated"] += 1

            # Clamp alpha and record
            effective_alpha = max(self.config.alpha_min, min(self.config.alpha_max, effective_alpha))
            blend_metrics["effective_alphas"].append(effective_alpha)

            # Store merged weight
            merged[key] = blended.astype(target_w.dtype)

        # Summarize metrics
        rotate_metrics["rotations_applied"] = int(rotate_metrics["rotations_applied"])
        rotate_metrics["identity_used"] = int(rotate_metrics["identity_used"])
        rotate_metrics["transport_guided_applied"] = int(rotate_metrics["transport_guided_applied"])

        if rotate_metrics["gw_distances"]:
            rotate_metrics["mean_gw_distance"] = float(np.mean(rotate_metrics["gw_distances"]))

        if blend_metrics["effective_alphas"]:
            blend_metrics["mean_alpha"] = float(np.mean(blend_metrics["effective_alphas"]))
            blend_metrics["min_alpha"] = float(np.min(blend_metrics["effective_alphas"]))
            blend_metrics["max_alpha"] = float(np.max(blend_metrics["effective_alphas"]))

        logger.info(
            "ROTATE: %d procrustes, %d transport, %d identity",
            rotate_metrics["rotations_applied"],
            rotate_metrics["transport_guided_applied"],
            rotate_metrics["identity_used"],
        )
        if rotate_metrics["transport_guided_applied"] > 0:
            logger.info(
                "GW TRANSPORT: mean_distance=%.4f",
                rotate_metrics.get("mean_gw_distance", 0),
            )
        if rotate_metrics["zipper_propagations"] > 0 or rotate_metrics["zipper_applications"] > 0:
            logger.info(
                "ZIPPER: %d propagations, %d applications",
                rotate_metrics["zipper_propagations"],
                rotate_metrics["zipper_applications"],
            )
        logger.info(
            "BLEND: mean_alpha=%.3f, spectral=%d, svd=%d, corr=%d, vn=%d, hard_swap=%d",
            blend_metrics.get("mean_alpha", 0),
            blend_metrics["spectral_adjustments"],
            blend_metrics["svd_blended"],
            blend_metrics["correlation_weighted"],
            blend_metrics["verb_noun_modulated"],
            blend_metrics["hard_swaps"],
        )
        if blend_metrics.get("module_policy_v", 0) > 0 or blend_metrics.get("module_policy_o", 0) > 0:
            logger.info(
                "MODULE POLICY: v_proj=%d (α=%.1f), o_proj=%d (α=%.1f)",
                blend_metrics.get("module_policy_v", 0),
                self.config.module_policy_v_alpha,
                blend_metrics.get("module_policy_o", 0),
                self.config.module_policy_o_alpha,
            )

        return merged, rotate_metrics, blend_metrics

    # =========================================================================
    # REFINEMENT DENSITY (Validated Law #5)
    # =========================================================================

    def _compute_refinement_density(
        self,
        source_weights: dict[str, np.ndarray],
        base_weights: dict[str, np.ndarray],
        source_path: str,
        base_path: str,
    ) -> tuple[Optional[dict[int, float]], list[int], dict[str, Any]]:
        """
        Compute per-layer refinement density using DARE sparsity + DoRA drift.

        Refinement density measures how much each layer has been "refined" in
        the source model compared to the base model:
        - High refinement → layer learned new capabilities → trust source (low alpha)
        - Low refinement → layer unchanged → trust target (high alpha)

        This implements Validated Law #5: Not all layers have equal knowledge mass.

        Args:
            source_weights: Source (fine-tuned) model weights
            base_weights: Base (pretrained) model weights
            source_path: Path to source model (for logging)
            base_path: Path to base model (for logging)

        Returns:
            Tuple of (alpha_by_layer dict, hard_swap_layers list, metrics dict)
            Returns (None, [], empty_metrics) on error
        """
        from modelcypher.core.domain.geometry.refinement_density import (
            RefinementDensityAnalyzer,
            RefinementDensityConfig,
        )

        try:
            # Convert numpy weights to format expected by analyzer
            base_dict: dict[str, Any] = {}
            adapted_dict: dict[str, Any] = {}

            for name in source_weights:
                if name not in base_weights:
                    continue
                source_w = source_weights[name]
                base_w = base_weights[name]

                # Only analyze weights with matching shapes
                if source_w.shape != base_w.shape:
                    continue

                base_dict[name] = base_w
                adapted_dict[name] = source_w

            if not base_dict:
                logger.warning("REFINEMENT DENSITY: No matching weights between source and base")
                return None, [], {"error": "no_matching_weights"}

            # Configure analyzer (use aggressive for model merging - we want
            # to identify and trust refined layers)
            config = RefinementDensityConfig.aggressive()
            analyzer = RefinementDensityAnalyzer(config)

            # Compute refinement density
            result = analyzer.analyze_from_weights(
                source_model=source_path,
                target_model=base_path,
                base_weights=base_dict,
                adapted_weights=adapted_dict,
            )

            # Extract alpha_by_layer and hard_swap_layers
            alpha_by_layer = result.alpha_by_layer
            hard_swap_layers = result.hard_swap_layers

            # Build metrics
            metrics = {
                "mean_composite_score": result.mean_composite_score,
                "max_composite_score": result.max_composite_score,
                "layers_above_hard_swap": result.layers_above_hard_swap,
                "layers_above_high_alpha": result.layers_above_high_alpha,
                "hard_swap_layer_indices": hard_swap_layers,
                "has_sparsity_data": result.has_sparsity_data,
                "has_directional_data": result.has_directional_data,
            }

            logger.info("REFINEMENT DENSITY: %s", result.interpretation.split("\n")[1])

            return alpha_by_layer, hard_swap_layers, metrics

        except ImportError as e:
            logger.warning("REFINEMENT DENSITY: Required modules not available: %s", e)
            return None, [], {"error": f"import_error: {e}"}
        except Exception as e:
            logger.warning("REFINEMENT DENSITY: Analysis failed: %s", e)
            return None, [], {"error": str(e)}

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _load_tokenizer(self, model_path: str) -> Any:
        """Load tokenizer from model path."""
        from tokenizers import Tokenizer

        path = Path(model_path)
        tokenizer_path = path / "tokenizer.json"

        if tokenizer_path.exists():
            return Tokenizer.from_file(str(tokenizer_path))

        # Try to find tokenizer in parent or alternate locations
        for alt_name in ["tokenizer.json", "tokenizer_config.json"]:
            alt_path = path / alt_name
            if alt_path.exists():
                try:
                    return Tokenizer.from_file(str(alt_path))
                except Exception:
                    pass

        logger.warning("No tokenizer found at %s, using fallback", model_path)
        return None

    def _load_model_for_probing(self, model_path: str) -> Any:
        """
        Load MLX model for activation extraction in precise probe mode.

        Uses mlx-lm's load function to get a full model that can
        be used for forward passes and activation collection.
        """
        try:
            from mlx_lm import load as mlx_load

            logger.info("Loading model from %s for probing...", model_path)
            model, _ = mlx_load(model_path)
            logger.info("Model loaded successfully")
            return model

        except ImportError:
            logger.warning(
                "mlx-lm not installed. Cannot load models for precise probe mode. "
                "Install with: pip install mlx-lm"
            )
            return None
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", model_path, e)
            return None

    def _load_weights(self, model_path: str) -> tuple[dict[str, np.ndarray], str]:
        """Load model weights from path."""
        from safetensors import safe_open

        path = Path(model_path)
        weights = {}

        # Try safetensors first
        safetensor_files = list(path.glob("*.safetensors"))
        if safetensor_files:
            for sf_path in safetensor_files:
                with safe_open(str(sf_path), framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            return weights, "safetensors"

        raise ValueError(f"No weights found at {model_path}")

    def _save_weights(
        self,
        output_dir: str,
        weights: dict[str, np.ndarray],
        format: str,
    ) -> None:
        """Save merged weights."""
        from safetensors.numpy import save_file

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        save_file(weights, str(path / "model.safetensors"))

    def _copy_config_files(self, source_path: str, output_dir: str) -> None:
        """Copy config files from source to output."""
        import shutil

        source = Path(source_path)
        dest = Path(output_dir)

        for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
            src = source / config_file
            if src.exists():
                shutil.copy2(src, dest / config_file)

    def _extract_layer_indices(self, weights: dict) -> list[int]:
        """Extract unique layer indices from weight keys."""
        indices = set()
        for key in weights:
            idx = self._extract_layer_index(key)
            if idx is not None:
                indices.add(idx)
        return sorted(indices)

    def _extract_layer_index(self, key: str) -> Optional[int]:
        """Extract layer index from weight key."""
        import re
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            return int(match.group(1))
        return None

    def _compute_procrustes_rotation(
        self,
        source_w: np.ndarray,
        target_w: np.ndarray,
        rank: int = 32,
    ) -> tuple[np.ndarray, float]:
        """
        Compute optimal rotation matrix using Procrustes analysis.

        This finds the rotation that best aligns source to target in spectral space.

        Args:
            source_w: Source weight matrix [out_dim, in_dim]
            target_w: Target weight matrix [out_dim, in_dim]
            rank: Truncation rank for SVD

        Returns:
            Tuple of (rotation_matrix, procrustes_error)
        """
        # Compute truncated SVD of both matrices
        min_dim = min(source_w.shape[0], source_w.shape[1], rank)
        if min_dim < 2:
            return np.eye(rank, dtype=np.float32), 0.0

        try:
            # Randomized SVD for efficiency
            from scipy.linalg import svd

            # Compute SVD of source and target
            u_s, s_s, vt_s = svd(source_w.astype(np.float32), full_matrices=False)
            u_t, s_t, vt_t = svd(target_w.astype(np.float32), full_matrices=False)

            # Truncate to rank
            k = min(min_dim, len(s_s), len(s_t))
            u_s = u_s[:, :k]
            u_t = u_t[:, :k]

            # Compute optimal rotation: argmin ||source @ R - target||
            # Using Procrustes: R = V @ U^T where source^T @ target = U @ S @ V^T
            m = u_s.T @ u_t
            u_m, _, vt_m = np.linalg.svd(m, full_matrices=False)

            # Ensure proper rotation (det = +1)
            omega = u_m @ vt_m
            if np.linalg.det(omega) < 0:
                u_m[:, -1] *= -1
                omega = u_m @ vt_m

            # Compute Procrustes error
            projected = omega @ u_s.T @ source_w
            error = np.linalg.norm(projected - u_t.T @ target_w) / (np.linalg.norm(u_t.T @ target_w) + 1e-8)

            # Pad rotation matrix to requested rank
            if omega.shape[0] < rank:
                padded = np.eye(rank, dtype=np.float32)
                padded[:omega.shape[0], :omega.shape[1]] = omega
                omega = padded

            return omega.astype(np.float32), float(error)

        except Exception as e:
            logger.debug("Procrustes computation failed: %s", e)
            return np.eye(rank, dtype=np.float32), 0.0

    def _compute_transport_guided_blend(
        self,
        source_w: np.ndarray,
        target_w: np.ndarray,
        alpha: float,
    ) -> Optional[tuple[np.ndarray, float]]:
        """
        Compute transport-guided blend using Gromov-Wasserstein optimal transport.

        GW computes soft correspondence π[i,j] between source neuron i and
        target neuron j based on their relational structure (pairwise distances).

        The transport plan is then used to blend:
            W_merged[j] = Σ_i π[i,j] * W_source[i]

        Finally, we blend with target using alpha:
            W_final = (1-α) * W_merged + α * W_target

        Mathematical Foundation:
            - Gromov-Wasserstein finds optimal coupling minimizing relational distortion
            - Unlike Procrustes which assumes bijective alignment, GW allows soft matching
            - Better for models with different neuron orderings or permutations

        References:
            - Peyré & Cuturi (2019) "Computational Optimal Transport"
            - Mémoli (2011) "Gromov-Wasserstein distances and the metric approach"

        Args:
            source_w: Source weight matrix [out_dim, in_dim]
            target_w: Target weight matrix [out_dim, in_dim]
            alpha: Blend factor (0 = all target, 1 = all source)

        Returns:
            Tuple of (blended_weights, gw_distance) or None if failed
        """
        from modelcypher.core.domain.geometry.gromov_wasserstein import (
            GromovWassersteinDistance,
            Config as GWConfig,
        )
        from modelcypher.core.domain.geometry.transport_guided_merger import (
            TransportGuidedMerger,
        )

        try:
            # Use weight rows as point clouds for distance computation
            # Each row represents a neuron's connectivity pattern
            source_points = source_w.tolist()
            target_points = target_w.tolist()

            # Compute pairwise distances within each space
            source_dist = GromovWassersteinDistance.compute_pairwise_distances(source_points)
            target_dist = GromovWassersteinDistance.compute_pairwise_distances(target_points)

            # Configure GW for reasonable speed/quality tradeoff
            gw_config = GWConfig(
                epsilon=0.1,  # Higher entropy for faster convergence
                max_outer_iterations=30,
                convergence_threshold=1e-4,
            )

            # Compute GW transport plan
            gw_result = GromovWassersteinDistance.compute(
                source_distances=source_dist,
                target_distances=target_dist,
                config=gw_config,
            )

            if not gw_result.converged and gw_result.iterations == 0:
                return None

            # Configure transport-guided merger
            merge_config = TransportGuidedMerger.Config(
                coupling_threshold=self.config.transport_coupling_threshold,
                normalize_rows=True,
                blend_alpha=alpha,  # (1-α)*transport + α*target
            )

            # Synthesize blended weights using transport plan
            merged = TransportGuidedMerger.synthesize(
                source_weights=source_points,
                target_weights=target_points,
                transport_plan=gw_result.coupling,
                config=merge_config,
            )

            if merged is None:
                return None

            # Convert back to numpy
            blended = np.array(merged, dtype=source_w.dtype)

            return blended, gw_result.distance

        except Exception as e:
            logger.debug("Transport-guided blend failed: %s", e)
            return None

    def _apply_rotation(
        self,
        weight: np.ndarray,
        omega_in: Optional[np.ndarray],
        omega_out: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Apply input and output rotations to a weight matrix.

        Weight layout is [out_dim, in_dim].
        omega_in rotates the input space, omega_out rotates the output space.

        Args:
            weight: Weight matrix [out_dim, in_dim]
            omega_in: Input rotation [rank, rank] or None
            omega_out: Output rotation [rank, rank] or None

        Returns:
            Rotated weight matrix
        """
        result = weight.copy()

        # Output rotation: W' = omega_out @ W (rotate rows)
        if omega_out is not None and omega_out.shape[0] == weight.shape[0]:
            result = omega_out @ result

        # Input rotation: W' = W @ omega_in^T (rotate columns)
        if omega_in is not None and omega_in.shape[0] == weight.shape[1]:
            result = result @ omega_in.T

        return result

    def _is_residual_output(self, key: str) -> bool:
        """Check if weight is a residual stream output (o_proj, down_proj)."""
        lower = key.lower()
        return any(token in lower for token in ("o_proj", "wo", "out_proj", "down_proj", "w2"))

    def _is_attention_input(self, key: str) -> bool:
        """Check if weight is an attention input projection (q_proj, k_proj, v_proj)."""
        lower = key.lower()
        return any(token in lower for token in ("q_proj", "k_proj", "v_proj", "wq", "wk", "wv", "query", "key", "value"))

    def _is_mlp_input(self, key: str) -> bool:
        """Check if weight is an MLP input projection (gate_proj, up_proj)."""
        lower = key.lower()
        return any(token in lower for token in ("gate_proj", "up_proj", "w1", "w3", "fc1"))

    def _is_v_proj(self, key: str) -> bool:
        """Check if weight is the value projection (v_proj)."""
        lower = key.lower()
        return any(token in lower for token in ("v_proj", "wv", ".value"))

    def _is_o_proj(self, key: str) -> bool:
        """Check if weight is the output projection (o_proj)."""
        lower = key.lower()
        return any(token in lower for token in ("o_proj", "wo.", "out_proj"))

    def _compute_weight_matching_permutation(
        self,
        source_w: np.ndarray,
        target_w: np.ndarray,
    ) -> np.ndarray:
        """
        Compute optimal permutation matrix using weight matching (LAP).

        This implements the weight matching algorithm from Git Re-Basin
        (Ainsworth et al., 2022). For each output neuron i in source, we find
        the best matching neuron j in target based on weight similarity.

        The Linear Assignment Problem maximizes:
            Σ_i ⟨source_w[i,:], target_w[π(i),:]⟩

        Args:
            source_w: Source weight matrix [out_dim, in_dim]
            target_w: Target weight matrix [out_dim, in_dim]

        Returns:
            P: Permutation matrix [out_dim, out_dim] such that
               P @ source_w aligns neurons to target_w
        """
        n = source_w.shape[0]  # output dimension

        # Similarity matrix: S[i,j] = ⟨source_w[i,:], target_w[j,:]⟩
        # This measures how similar output neuron i of source is to
        # output neuron j of target, based on their weight patterns.
        S = source_w @ target_w.T  # [n, n]

        try:
            # Optimal: Hungarian algorithm via scipy
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-S)
        except ImportError:
            # Fallback: Greedy matching (not optimal but fast)
            # For each source neuron, greedily pick the best available target
            logger.debug("scipy not available, using greedy weight matching")
            row_ind = np.arange(n)
            col_ind = np.zeros(n, dtype=np.int64)
            available = set(range(n))

            for i in range(n):
                # Find best available target for source[i]
                best_j = max(available, key=lambda j: S[i, j])
                col_ind[i] = best_j
                available.remove(best_j)

        # Build permutation matrix: P[row_ind[k], col_ind[k]] = 1
        # This means: permuted[col_ind[k]] = original[row_ind[k]]
        # Or: permuted = P @ original
        P = np.zeros((n, n), dtype=np.float32)
        P[col_ind, row_ind] = 1.0  # P[j, i] = 1 means output j comes from input i

        return P

    def _compute_full_rank_rotation(
        self,
        source_w: np.ndarray,
        target_w: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute full-rank orthogonal rotation using Procrustes.

        Finds R ∈ O(n) that minimizes ||R @ source_w - target_w||_F.
        This is the continuous relaxation of the permutation problem.

        Args:
            source_w: Source weight matrix [out_dim, in_dim]
            target_w: Target weight matrix [out_dim, in_dim]

        Returns:
            Tuple of (R: rotation matrix [out_dim, out_dim], error: float)
        """
        # Procrustes solution: R = U @ V^T where M = target @ source^T = U Σ V^T
        M = target_w @ source_w.T  # [out_dim, out_dim]

        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=True)

            # Ensure proper rotation (det = +1, not reflection)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            # Compute alignment error
            aligned = R @ source_w
            error = np.linalg.norm(aligned - target_w) / (np.linalg.norm(target_w) + 1e-8)

            return R.astype(np.float32), float(error)

        except np.linalg.LinAlgError:
            # SVD failed, return identity
            n = source_w.shape[0]
            return np.eye(n, dtype=np.float32), 1.0


# =============================================================================
# PUBLIC API
# =============================================================================

def unified_merge(
    source: str,
    target: str,
    output_dir: str,
    config: Optional[UnifiedMergeConfig] = None,
    base_model: Optional[str] = None,
    dry_run: bool = False,
) -> UnifiedMergeResult:
    """
    Execute unified geometric merge.

    This is THE ONE merge function that combines all techniques.

    Args:
        source: Path to source model (skill donor)
        target: Path to target model (knowledge base)
        output_dir: Output directory
        config: Merge configuration
        base_model: Path to base/pretrained model for refinement density.
                    If provided, computes per-layer "knowledge mass" using
                    DARE sparsity + DoRA drift to weight layer alphas.
                    (Validated Law #5: Not all layers have equal knowledge mass)
        dry_run: If True, don't save to disk

    Returns:
        UnifiedMergeResult with merged weights and metrics
    """
    merger = UnifiedGeometricMerger(config)
    return merger.merge(source, target, output_dir, base_model_path=base_model, dry_run=dry_run)
