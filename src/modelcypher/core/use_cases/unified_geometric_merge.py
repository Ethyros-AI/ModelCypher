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
from typing import Any, Optional

import numpy as np

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

    # Whether to probe models for fingerprints (if False, use pre-computed)
    probe_models: bool = True

    # Similarity mode for intersection map: jaccard, cka, ensemble
    intersection_mode: str = "ensemble"

    # Minimum correlation to include in intersection map
    intersection_threshold: float = 0.3

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
            dry_run: If True, don't save to disk

        Returns:
            UnifiedMergeResult with merged weights and metrics
        """
        logger.info("=== UNIFIED GEOMETRIC MERGE ===")
        logger.info("Source: %s", source_path)
        logger.info("Target: %s", target_path)

        # Load weights
        source_weights, source_format = self._load_weights(source_path)
        target_weights, target_format = self._load_weights(target_path)

        # Identify layers
        layer_indices = self._extract_layer_indices(target_weights)
        logger.info("Found %d layers", len(layer_indices))

        # =================================================================
        # STAGE 1: PROBE
        # =================================================================
        logger.info("STAGE 1: PROBE (Fingerprinting)")
        intersection_map, probe_metrics = self._stage_probe(
            source_weights,
            target_weights,
            source_fingerprints,
            target_fingerprints,
        )

        # =================================================================
        # STAGE 2: PERMUTE
        # =================================================================
        logger.info("STAGE 2: PERMUTE (Re-Basin)")
        permuted_source, permute_metrics = self._stage_permute(
            source_weights,
            target_weights,
            intersection_map,
            layer_indices,
        )

        # =================================================================
        # STAGE 3 & 4 & 5: ROTATE + BLEND + PROPAGATE (merged loop)
        # =================================================================
        logger.info("STAGES 3-5: ROTATE + BLEND + PROPAGATE")
        merged_weights, rotate_metrics, blend_metrics = self._stage_rotate_blend_propagate(
            permuted_source,
            target_weights,
            intersection_map,
            layer_indices,
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
    # STAGE 1: PROBE
    # =========================================================================

    def _stage_probe(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        source_fingerprints: Optional[dict],
        target_fingerprints: Optional[dict],
    ) -> tuple[dict, dict]:
        """
        Stage 1: Build intersection map from fingerprints.

        The intersection map is the PRIMARY CONTROL SIGNAL for all
        downstream operations.

        Uses ensemble similarity combining:
        1. CKA (Centered Kernel Alignment) - rotation/scale invariant
        2. Cosine similarity - direction alignment
        3. Jaccard (approximated) - sparse overlap

        Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations"
        """
        from modelcypher.core.domain.geometry.cka import (
            compute_layer_cka,
            ensemble_similarity,
        )

        intersection_map = {}
        layer_confidences = {}
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

            # Compute CKA for 2D weight matrices (skip if jaccard-only mode or too large)
            # CKA is O(n²) so we limit to small matrices
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
            # (fraction of dimensions where both are non-negligible)
            threshold = 0.01 * max(np.abs(source_w).max(), np.abs(target_w).max())
            s_active = np.abs(source_w) > threshold
            t_active = np.abs(target_w) > threshold
            intersection = np.sum(s_active & t_active)
            union = np.sum(s_active | t_active)
            jaccard = float(intersection / max(union, 1))

            # Ensemble similarity (CKA-weighted)
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
        for layer_idx in layer_confidences:
            layer_confidences[layer_idx] = float(np.mean(layer_confidences[layer_idx]))

        mean_confidence = float(np.mean(list(layer_confidences.values()))) if layer_confidences else 0.0
        mean_cka = float(np.mean(list(cka_scores.values()))) if cka_scores else 0.0

        metrics = {
            "weight_count": len(intersection_map),
            "layer_confidences": layer_confidences,
            "mean_confidence": mean_confidence,
            "mean_cka": mean_cka,
            "min_confidence": min(layer_confidences.values()) if layer_confidences else 0.0,
            "max_confidence": max(layer_confidences.values()) if layer_confidences else 0.0,
            "intersection_mode": self.config.intersection_mode,
        }

        logger.info(
            "PROBE: %d weights, mean_confidence=%.3f, mean_cka=%.3f",
            len(intersection_map),
            mean_confidence,
            mean_cka,
        )

        return {"correlations": intersection_map, "confidences": layer_confidences}, metrics

    # =========================================================================
    # STAGE 2: PERMUTE
    # =========================================================================

    def _stage_permute(
        self,
        source_weights: dict[str, np.ndarray],
        target_weights: dict[str, np.ndarray],
        intersection_map: dict,
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

        layer_confidences = intersection_map.get("confidences", {})

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
        intersection_map: dict,
        layer_indices: list[int],
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

        Blend adjustments applied in sequence:
        1. Base alpha from intersection confidence
        2. Gaussian smoothing across layers
        3. Spectral penalty for ill-conditioned weights
        4. SVD-aware blending (different alpha for high/low rank)
        5. Correlation-based dimension weights
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

        layer_confidences = intersection_map.get("confidences", {})

        # Pre-compute smoothed alpha profile
        raw_alphas = {}
        for layer_idx in layer_indices:
            confidence = layer_confidences.get(layer_idx, 0.0)
            # Base alpha from confidence: high confidence → trust source more
            raw_alphas[layer_idx] = 1.0 - (confidence * 0.7)

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
        }

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
            "BLEND: mean_alpha=%.3f, spectral=%d, svd=%d, corr=%d, vn=%d",
            blend_metrics.get("mean_alpha", 0),
            blend_metrics["spectral_adjustments"],
            blend_metrics["svd_blended"],
            blend_metrics["correlation_weighted"],
            blend_metrics["verb_noun_modulated"],
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
    # HELPERS
    # =========================================================================

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
        dry_run: If True, don't save to disk

    Returns:
        UnifiedMergeResult with merged weights and metrics
    """
    merger = UnifiedGeometricMerger(config)
    return merger.merge(source, target, output_dir, dry_run=dry_run)
