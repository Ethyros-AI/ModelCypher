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

"""LoRA Safety Service.

Provides safety analysis for LoRA training and deployment:

1. Fisher-guided module targeting (exp15: r=-0.864)
   - Target LOW-Fisher modules for better LoRA adaptation

2. Mode connectivity barrier check (exp16: r=0.989)
   - Use barrier as safety gate before LoRA deployment

3. Goldilocks quality scoring for curriculum (exp17: r=-0.955)
   - Select training data with moderate challenge for best learning

4. Geometry-derived scale bounds (NEW)
   - LoRA scale must respect the spectral structure of base weights
   - scale_bound = σ_k(W) / ||B@A||_spectral
   - σ_k is smallest significant singular value (above √ε × σ_max)
   - Ensures LoRA adds at edge of effective subspace, not overwhelming it

Usage:
    from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

    service = LoRASafetyService()

    # Check if LoRA scale is safe
    report = service.compute_geometric_scale(model_path, adapter_path)
    if not report.is_safe:
        print(f"WARNING: {report.recommendation}")

    # Apply with geometry-derived scale (safe)
    model, scales = service.apply_lora_geometric(model, adapter_path)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class EigengapInfo:
    """Information about spectral gaps in a weight matrix.

    From Theorem 2 (Eigengap Refinement): When W has a spectral gap,
    the scale bound can be tightened to gap_k / (2 × ||Δ||_spectral).

    Reference: arXiv:2510.25670 (NeurIPS 2025)
    """

    position: int | None  # Index where gap occurs (None if no significant gap)
    gap_value: float  # σ_k - σ_{k+1} at gap position
    gap_ratio: float  # σ_k / σ_{k+1} at gap position
    tightened_bound: float | None  # gap_value / 2 if gap exists


@dataclass
class AlignmentQuality:
    """Measures how aligned LoRA's principal direction is with W's subspace.

    High alignment (near 1.0) means LoRA delta operates within W's existing
    directions. Low alignment means LoRA introduces orthogonal components.

    Reference: RoRA (arXiv:2601.06305) - alignment as root cause of failures
    """

    projection_norm: float  # ||U_k^T @ u_delta|| where u_delta is LoRA's top SV
    alignment_ratio: float  # projection_norm / ||u_delta|| (0 to 1)
    is_well_aligned: bool  # True if alignment_ratio > 0.5


@dataclass
class GeometricScaleBound:
    """Geometry-derived scale bound for a single LoRA layer.

    The scale bound is derived from the spectral structure of the base weight:
        scale_bound = σ_k(W) / ||B@A||_spectral

    Where σ_k is the smallest significant singular value of W (above √ε × σ_max).
    This ensures the LoRA delta adds information at the edge of the weight's
    effective subspace rather than overwhelming its learned structure.

    Enhanced with eigengap detection (Theorem 2) and alignment quality (RoRA).
    """

    layer_key: str
    sigma_max: float  # Largest singular value of base weight
    sigma_k: float  # Smallest significant singular value
    delta_spectral_norm: float  # Spectral norm of LoRA delta (unscaled)
    effective_rank: int  # Number of significant singular values
    geometric_scale_bound: float  # Max safe scale from geometry
    configured_scale: float  # Scale that would be used (alpha/rank)
    scale_ratio: float  # configured / geometric (>1 means too aggressive)
    # New fields for enhanced analysis
    eigengap: EigengapInfo | None = None  # Eigengap info if detected
    alignment: AlignmentQuality | None = None  # LoRA-base alignment quality
    adaptive_bound: float | None = None  # Tighter bound using eigengap if available


@dataclass
class GeometricScaleReport:
    """Report of geometry-derived scale analysis for a LoRA adapter."""

    adapter_path: str
    base_model_path: str
    configured_alpha: float
    configured_rank: int
    configured_scale: float  # alpha / rank
    layer_bounds: list[GeometricScaleBound]
    min_geometric_bound: float  # Minimum across all layers
    max_scale_ratio: float  # Maximum configured/geometric ratio
    is_safe: bool  # Whether configured scale respects geometry
    recommendation: str


@dataclass
class ModuleRecommendation:
    """Recommendation for a single module."""

    module: str
    fisher_score: float
    recommendation: str  # EXCELLENT, GOOD, ACCEPTABLE, AVOID


@dataclass
class TargetModuleResult:
    """Result of module recommendation analysis."""

    model_path: str
    layer: int
    n_samples: int
    recommendations: list[ModuleRecommendation]
    guidance: str = "Target LOW-Fisher modules for better LoRA adaptation. Exp15: r=-0.864."


@dataclass
class BarrierSafetyResult:
    """Result of barrier safety check."""

    base_path: str
    target_path: str
    layer: int
    barrier_height: float
    barrier_normalized: float
    safety_level: str  # SAFE, CAUTION, WARNING
    cka_at_target: float
    recommendation: str


@dataclass
class CurriculumScoreResult:
    """Result of curriculum quality scoring."""

    model_path: str
    n_problems: int
    quality_distribution: dict[str, dict]  # group -> {count, mean_score}
    top_problems: list[dict]  # [{"prompt": ..., "quality_score": ..., etc}]
    guidance: str = "Use high_quality problems. Goldilocks: CKA~0.9, barrier 0.02-0.10."


class LoRASafetyService:
    """Service for LoRA safety analysis.

    Combines Fisher Information, Mode Connectivity, and Goldilocks Quality
    scoring to provide safe and effective LoRA training guidance.
    """

    # Safety thresholds (validated in exp16)
    BARRIER_SAFE = 0.01
    BARRIER_CAUTION = 0.03

    # Fisher recommendation thresholds (validated in exp15)
    FISHER_EXCELLENT = 0.0004
    FISHER_GOOD = 0.0005
    FISHER_ACCEPTABLE = 0.0007

    # Eigengap detection threshold (ratio σ_k/σ_{k+1})
    # A ratio > 2.0 indicates meaningful spectral structure
    EIGENGAP_THRESHOLD = 2.0

    # Alignment quality threshold
    # LoRA delta with alignment > 0.5 operates mostly within W's subspace
    ALIGNMENT_THRESHOLD = 0.5

    @staticmethod
    def _prompt_id(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def detect_eigengap(
        singular_values: "Array",
        backend: "Backend",
        gap_threshold: float = 2.0,
    ) -> EigengapInfo:
        """Detect spectral gap for tighter scale bounds.

        From Theorem 2 (Eigengap Refinement): When W has a spectral gap at
        position k (σ_k / σ_{k+1} > threshold), the scale bound can be tightened.

        Args:
            singular_values: Singular values in descending order.
            backend: Compute backend.
            gap_threshold: Minimum ratio to count as a gap (default 2.0).

        Returns:
            EigengapInfo with gap position, value, and tightened bound.

        Reference: arXiv:2510.25670 (NeurIPS 2025)
        """
        b = backend
        S = singular_values

        # Need at least 2 values to detect gap
        if len(S.shape) == 0 or S.shape[0] < 2:
            return EigengapInfo(
                position=None, gap_value=0.0, gap_ratio=1.0, tightened_bound=None
            )

        # Compute ratios of consecutive singular values
        # Add small epsilon to prevent division by zero
        eps = b.finfo(S.dtype).eps
        S_shifted = S[1:]
        S_safe = b.where(S_shifted < eps, b.ones_like(S_shifted) * eps, S_shifted)
        ratios = S[:-1] / S_safe
        b.eval(ratios)

        # Find positions where ratio exceeds threshold
        gaps = ratios > gap_threshold
        b.eval(gaps)

        # Find first significant gap by iterating
        # (Backend doesn't have np.where-style index finding)
        n_ratios = ratios.shape[0]
        for i in range(n_ratios):
            gap_val = b.to_scalar(gaps[i])
            if bool(gap_val):
                k = i
                gap_ratio = float(b.to_scalar(ratios[k]))
                gap_value = float(b.to_scalar(S[k])) - float(b.to_scalar(S[k + 1]))
                tightened = gap_value / 2.0

                return EigengapInfo(
                    position=k,
                    gap_value=gap_value,
                    gap_ratio=gap_ratio,
                    tightened_bound=tightened,
                )

        return EigengapInfo(
            position=None, gap_value=0.0, gap_ratio=1.0, tightened_bound=None
        )

    @staticmethod
    def compute_alignment_quality(
        W_U_k: "Array",
        delta_u1: "Array",
        backend: "Backend",
    ) -> AlignmentQuality:
        """Measure how aligned LoRA's principal direction is with W's subspace.

        High alignment means LoRA operates within W's existing structure.
        Low alignment means LoRA introduces orthogonal (potentially conflicting)
        directions.

        Args:
            W_U_k: Left singular vectors of W corresponding to significant SVs.
                   Shape [m, k] where k is effective rank.
            delta_u1: First left singular vector of LoRA delta.
                      Shape [m] or [m, 1].
            backend: Compute backend.

        Returns:
            AlignmentQuality with projection norm and alignment ratio.

        Reference: RoRA (arXiv:2601.06305) - alignment as root cause of failures
        """
        b = backend

        # Ensure delta_u1 is column vector
        if len(delta_u1.shape) == 1:
            delta_u1 = b.reshape(delta_u1, (-1, 1))

        # Project delta's principal direction onto W's significant subspace
        # projection = U_k^T @ u_delta gives coordinates in W's subspace
        projection = b.matmul(b.transpose(W_U_k), delta_u1)
        b.eval(projection)

        # Norm of projection (how much of delta is within W's subspace)
        proj_norm = b.sqrt(b.sum(projection * projection))
        b.eval(proj_norm)
        proj_norm_val = float(b.to_scalar(proj_norm))

        # Norm of delta_u1 (should be 1.0 for normalized singular vector)
        delta_norm = b.sqrt(b.sum(delta_u1 * delta_u1))
        b.eval(delta_norm)
        delta_norm_val = float(b.to_scalar(delta_norm))

        # Alignment ratio: fraction of delta within W's subspace
        eps = b.finfo(delta_u1.dtype).eps
        alignment_ratio = proj_norm_val / max(delta_norm_val, eps)

        return AlignmentQuality(
            projection_norm=proj_norm_val,
            alignment_ratio=alignment_ratio,
            is_well_aligned=alignment_ratio > 0.5,
        )

    @staticmethod
    def compute_adaptive_bound(
        sigma_k: float,
        eigengap: EigengapInfo,
        delta_spectral: float,
    ) -> float:
        """Compute adaptive scale bound using eigengap when available.

        Uses the tighter of:
        1. Standard bound: σ_k / ||Δ||_spectral
        2. Eigengap bound: (gap_value / 2) / ||Δ||_spectral

        Args:
            sigma_k: Smallest significant singular value.
            eigengap: Eigengap information (may have no gap).
            delta_spectral: Spectral norm of LoRA delta.

        Returns:
            Adaptive scale bound (tighter than standard when eigengap exists).
        """
        if delta_spectral < 1e-10:
            return float("inf")

        standard_bound = sigma_k / delta_spectral

        if eigengap.tightened_bound is not None:
            eigengap_bound = eigengap.tightened_bound / delta_spectral
            return min(standard_bound, eigengap_bound)

        return standard_bound

    def recommend_target_modules(
        self,
        model_path: str | Path,
        prompts: list[str],
        layer_idx: int | None = None,
        top_k: int = 4,
    ) -> TargetModuleResult:
        """Get Fisher-guided module recommendations for LoRA targeting.

        Based on exp15 validation (r=-0.864): Lower Fisher = better LoRA adaptation.

        Args:
            model_path: Path to base model
            prompts: Test prompts for activation collection
            layer_idx: Layer to analyze (default: middle)
            top_k: Number of recommendations to return

        Returns:
            TargetModuleResult with ranked module recommendations
        """
        from modelcypher.adapters.model_loader import load_model_for_training
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.fisher_information import (
            compute_empirical_fisher_diagonal,
        )

        backend = get_default_backend()
        model, tokenizer = load_model_for_training(str(model_path))

        # Get model architecture
        from modelcypher.adapters.model_architecture import get_model_architecture

        arch = get_model_architecture(model, model_path=str(model_path))
        num_layers = arch.num_layers

        if layer_idx is None:
            layer_idx = num_layers // 2

        # Collect activations from different modules
        module_fisher_scores = self._compute_module_fisher_scores(
            model, tokenizer, arch, prompts, layer_idx, backend
        )

        # Sort by Fisher score (ascending - lower is better)
        sorted_modules = sorted(module_fisher_scores.items(), key=lambda x: x[1])

        # Create recommendations
        recommendations = []
        for module, score in sorted_modules[:top_k]:
            if score < self.FISHER_EXCELLENT:
                rec = "EXCELLENT"
            elif score < self.FISHER_GOOD:
                rec = "GOOD"
            elif score < self.FISHER_ACCEPTABLE:
                rec = "ACCEPTABLE"
            else:
                rec = "AVOID"

            recommendations.append(ModuleRecommendation(
                module=module,
                fisher_score=score,
                recommendation=rec,
            ))

        return TargetModuleResult(
            model_path=str(model_path),
            layer=layer_idx,
            n_samples=len(prompts),
            recommendations=recommendations,
        )

    def check_barrier_safety(
        self,
        base_path: str | Path,
        target_path: str | Path,
        prompts: list[str],
        layer_idx: int | None = None,
    ) -> BarrierSafetyResult:
        """Check mode connectivity barrier for LoRA safety.

        Based on exp16 validation (r=0.989): Barrier predicts LoRA divergence.

        Args:
            base_path: Path to base model
            target_path: Path to LoRA weights or merged model
            prompts: Test prompts for activation collection
            layer_idx: Layer to analyze (default: middle)

        Returns:
            BarrierSafetyResult with safety assessment
        """
        from modelcypher.adapters.model_loader import load_model_for_training
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

        backend = get_default_backend()

        # Load base model and collect activations
        base_model, base_tokenizer = load_model_for_training(str(base_path))
        from modelcypher.adapters.model_architecture import get_model_architecture

        base_arch = get_model_architecture(base_model, model_path=str(base_path))
        num_layers = base_arch.num_layers

        if layer_idx is None:
            layer_idx = num_layers // 2

        base_activations = self._collect_activations(
            base_model, base_tokenizer, base_arch, prompts, layer_idx, backend
        )

        # Load target model and collect activations
        target_model, target_tokenizer = load_model_for_training(str(target_path))
        target_arch = get_model_architecture(target_model, model_path=str(target_path))

        target_activations = self._collect_activations(
            target_model, target_tokenizer, target_arch, prompts, layer_idx, backend
        )

        # Compute barrier
        barrier_height, barrier_normalized, cka_at_target = self._compute_barrier(
            base_activations, target_activations, backend
        )

        # Determine safety level
        if barrier_height < self.BARRIER_SAFE:
            safety_level = "SAFE"
            recommendation = "LoRA stays in-basin. Safe to deploy."
        elif barrier_height < self.BARRIER_CAUTION:
            safety_level = "CAUTION"
            recommendation = "LoRA near basin boundary. Verify downstream performance."
        else:
            safety_level = "WARNING"
            recommendation = "LoRA pushes out of basin. May fight base model. Proceed with caution."

        return BarrierSafetyResult(
            base_path=str(base_path),
            target_path=str(target_path),
            layer=layer_idx,
            barrier_height=barrier_height,
            barrier_normalized=barrier_normalized,
            safety_level=safety_level,
            cka_at_target=cka_at_target,
            recommendation=recommendation,
        )

    def score_curriculum(
        self,
        model_path: str | Path,
        problems: list[dict],
        reference_prompts: list[str] | None = None,
        layer_idx: int | None = None,
        top_k: int = 10,
    ) -> CurriculumScoreResult:
        """Score problems using Goldilocks quality metric.

        Based on exp17 validation (r=-0.955): Moderate challenge = best learning.

        Args:
            model_path: Path to model
            problems: List of problem dicts with "prompt" key
            reference_prompts: Reference prompts (default: simple arithmetic)
            layer_idx: Layer to analyze (default: middle)
            top_k: Number of top problems to return

        Returns:
            CurriculumScoreResult with quality distribution and top problems
        """
        from modelcypher.adapters.model_loader import load_model_for_training
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.goldilocks_quality import (
            compute_goldilocks_quality,
            GoldilocksQualityResult,
        )

        backend = get_default_backend()

        # Default reference prompts (simple arithmetic)
        if reference_prompts is None:
            reference_prompts = [
                "What is 2+2?",
                "What is 5+3?",
                "What is 10-4?",
                "What is 7*2?",
            ]

        # Load model
        model, tokenizer = load_model_for_training(str(model_path))
        from modelcypher.adapters.model_architecture import get_model_architecture

        arch = get_model_architecture(model, model_path=str(model_path))
        num_layers = arch.num_layers

        if layer_idx is None:
            layer_idx = num_layers // 2

        # Collect reference activations
        reference_activations = self._collect_activations(
            model, tokenizer, arch, reference_prompts, layer_idx, backend
        )

        # Score each problem
        problem_scores = []
        for problem in problems:
            prompt = problem.get("prompt", problem.get("text", ""))
            if not prompt:
                continue

            # Collect activations for this problem
            problem_activations = self._collect_activations(
                model, tokenizer, arch, [prompt], layer_idx, backend
            )

            # Compute Goldilocks quality
            quality = compute_goldilocks_quality(
                problem_activations, reference_activations, backend
            )

            problem_scores.append({
                "prompt": prompt,
                "prompt_id": self._prompt_id(prompt),
                "prompt_preview": prompt[:100],
                "quality_score": quality.quality_score,
                "cka": quality.cka_similarity,
                "barrier": quality.barrier_height,
                "fisher": quality.fisher_mean,
                "quality_level": quality.quality_level,
            })

        # Sort by quality score
        problem_scores.sort(key=lambda x: x["quality_score"], reverse=True)

        # Compute quality distribution
        high_count = sum(1 for p in problem_scores if p["quality_level"] == "high")
        medium_count = sum(1 for p in problem_scores if p["quality_level"] == "medium")
        low_count = sum(1 for p in problem_scores if p["quality_level"] == "low")

        high_mean = (
            sum(p["quality_score"] for p in problem_scores if p["quality_level"] == "high")
            / max(high_count, 1)
        )
        medium_mean = (
            sum(p["quality_score"] for p in problem_scores if p["quality_level"] == "medium")
            / max(medium_count, 1)
        )
        low_mean = (
            sum(p["quality_score"] for p in problem_scores if p["quality_level"] == "low")
            / max(low_count, 1)
        )

        return CurriculumScoreResult(
            model_path=str(model_path),
            n_problems=len(problem_scores),
            quality_distribution={
                "high_quality": {"count": high_count, "mean_score": high_mean},
                "medium_quality": {"count": medium_count, "mean_score": medium_mean},
                "low_quality": {"count": low_count, "mean_score": low_mean},
            },
            top_problems=problem_scores[:top_k],
        )

    def filter_by_difficulty(
        self,
        model_path: str | Path,
        problems: list[dict],
        target_difficulty: str = "medium",
        reference_prompts: list[str] | None = None,
        layer_idx: int | None = None,
    ) -> list[dict]:
        """Filter problems by difficulty level using Goldilocks quality metric.

        Difficulty is based on CKA similarity to known-good prompts:
        - easy: CKA > 0.95 (model nearly knows it)
        - medium: CKA 0.85-0.95 (Goldilocks zone - optimal for learning)
        - hard: CKA < 0.85 (significant challenge)

        Args:
            model_path: Path to model
            problems: List of problem dicts with "prompt" key
            target_difficulty: One of "easy", "medium", "hard", "all"
            reference_prompts: Reference prompts (default: simple arithmetic)
            layer_idx: Layer to analyze (default: middle)

        Returns:
            Filtered list of problems matching the target difficulty
        """
        # Score all problems first
        result = self.score_curriculum(
            model_path=model_path,
            problems=problems,
            reference_prompts=reference_prompts,
            layer_idx=layer_idx,
            top_k=len(problems),  # Get all scores
        )

        if target_difficulty == "all":
            return problems

        # Map difficulty to CKA thresholds
        difficulty_map = {
            "easy": lambda cka: cka > 0.95,
            "medium": lambda cka: 0.85 <= cka <= 0.95,
            "hard": lambda cka: cka < 0.85,
        }

        if target_difficulty not in difficulty_map:
            logger.warning(
                "Unknown difficulty '%s', using 'medium'", target_difficulty
            )
            target_difficulty = "medium"

        predicate = difficulty_map[target_difficulty]

        # Build lookup from scored problems
        scored_by_prompt = {
            p.get("prompt_id", self._prompt_id(p["prompt"])): p
            for p in result.top_problems
        }

        # Filter original problems
        filtered = []
        for problem in problems:
            prompt = problem.get("prompt", problem.get("text", ""))
            prompt_key = self._prompt_id(prompt)
            if prompt_key in scored_by_prompt:
                scored = scored_by_prompt[prompt_key]
                if predicate(scored["cka"]):
                    filtered.append(problem)

        logger.info(
            "Filtered %d/%d problems for difficulty '%s'",
            len(filtered), len(problems), target_difficulty
        )

        return filtered

    def _compute_module_fisher_scores(
        self,
        model,
        tokenizer,
        arch,
        prompts: list[str],
        layer_idx: int,
        backend: "Backend",
    ) -> dict[str, float]:
        """Compute Fisher scores for different module types.

        Returns dict mapping module name to Fisher score.
        """
        from modelcypher.core.domain.geometry.fisher_information import (
            compute_empirical_fisher_diagonal,
        )

        # Common LoRA target modules
        module_names = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"]

        scores = {}
        layer = arch.layers[layer_idx]

        for module_name in module_names:
            # Try to find this module in the layer
            weight = self._find_module_weight(layer, module_name)
            if weight is None:
                continue

            # Collect activations through this module
            try:
                activations = self._collect_module_activations(
                    model, tokenizer, arch, prompts, layer_idx, module_name, backend
                )
                if activations is not None:
                    fisher_result = compute_empirical_fisher_diagonal(activations, backend)
                    scores[module_name] = fisher_result.mean_fim
            except Exception as e:
                logger.debug("Failed to compute Fisher for %s: %s", module_name, e)
                continue

        return scores

    def _find_module_weight(self, layer, module_name: str):
        """Find a module's weight matrix in a layer."""
        # Try common locations
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, module_name):
                m = getattr(attn, module_name)
                if hasattr(m, "weight"):
                    return m.weight

        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, module_name):
                m = getattr(mlp, module_name)
                if hasattr(m, "weight"):
                    return m.weight

        return None

    def _collect_module_activations(
        self,
        model,
        tokenizer,
        arch,
        prompts: list[str],
        layer_idx: int,
        module_name: str,
        backend: "Backend",
    ):
        """Collect activations from a specific module."""
        activations_list = []

        for prompt in prompts[:10]:  # Limit for speed
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            if len(tokens) > 64:
                tokens = tokens[:64]

            input_ids = backend.array([tokens])
            hidden = arch.embed_module(input_ids)
            backend.eval(hidden)

            # Forward through layers up to target
            for i, layer in enumerate(arch.layers):
                if i > layer_idx:
                    break
                hidden = layer(hidden)
                backend.eval(hidden)

            # Mean pool
            pooled = backend.mean(hidden, axis=(0, 1))
            backend.eval(pooled)
            activations_list.append(pooled)

        if not activations_list:
            return None

        return backend.stack(activations_list, axis=0)

    def _collect_activations(
        self,
        model,
        tokenizer,
        arch,
        prompts: list[str],
        layer_idx: int,
        backend: "Backend",
    ):
        """Collect activations from model for given prompts."""
        activations_list = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            if len(tokens) > 64:
                tokens = tokens[:64]

            input_ids = backend.array([tokens])
            hidden = arch.embed_module(input_ids)
            backend.eval(hidden)

            for i, layer in enumerate(arch.layers):
                if i > layer_idx:
                    break
                hidden = layer(hidden)
                backend.eval(hidden)

            pooled = backend.mean(hidden, axis=(0, 1))
            backend.eval(pooled)
            activations_list.append(pooled)

        return backend.stack(activations_list, axis=0)

    def _compute_barrier(
        self,
        source_activations,
        target_activations,
        backend: "Backend",
    ) -> tuple[float, float, float]:
        """Compute CKA barrier between source and target activations.

        Returns:
            (barrier_height, normalized_barrier, cka_at_target)
        """
        from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

        # Center activations
        source_centered = source_activations - backend.mean(source_activations, axis=0, keepdims=True)
        target_centered = target_activations - backend.mean(target_activations, axis=0, keepdims=True)
        backend.eval(source_centered, target_centered)

        # Compute CKA at target
        cka_at_target = compute_linear_cka_from_activations(
            source_centered, target_centered, backend
        )

        # Compute barrier along interpolation path
        n_steps = 11
        losses = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            interpolated = (1 - t) * source_centered + t * target_centered
            backend.eval(interpolated)

            cka = compute_linear_cka_from_activations(source_centered, interpolated, backend)
            losses.append(1.0 - cka)

        barrier_height = max(losses)
        target_loss = losses[-1]
        normalized = barrier_height / max(target_loss, 1e-8) if target_loss > 1e-8 else 0.0

        return barrier_height, normalized, cka_at_target

    def compute_geometric_scale(
        self,
        model_path: str | Path,
        adapter_path: str | Path,
    ) -> GeometricScaleReport:
        """Compute geometry-derived scale bounds for a LoRA adapter.

        The scale bound for each layer is derived from the spectral structure:
            scale_bound = σ_k(W) / ||B@A||_spectral

        Where σ_k is the smallest significant singular value of the base weight
        (those above √ε × σ_max). This ensures the LoRA delta adds information
        at the edge of the weight's effective subspace.

        Args:
            model_path: Path to base model
            adapter_path: Path to LoRA adapter directory

        Returns:
            GeometricScaleReport with per-layer analysis and recommendations
        """
        import json

        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        backend = get_default_backend()

        adapter_path = Path(adapter_path)
        config_path = adapter_path / "adapter_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json in {adapter_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Extract config - handle both standard and custom formats
        rank = config.get("rank") or config.get("r", 8)
        alpha = config.get("alpha") or config.get("lora_alpha", rank)
        configured_scale = alpha / rank

        # Find LoRA weights file
        weights_path = None
        for candidate in ["lora_weights.safetensors", "adapter_model.safetensors"]:
            if (adapter_path / candidate).exists():
                weights_path = adapter_path / candidate
                break

        if weights_path is None:
            raise FileNotFoundError(f"No LoRA weights found in {adapter_path}")

        # Load model and weights
        from modelcypher.adapters.model_loader import load_model_for_training

        model, _ = load_model_for_training(str(model_path))
        base_model = getattr(model, "model", model)

        # Load LoRA weights
        lora_weights = backend.load_safetensors(str(weights_path))

        # Organize LoRA pairs
        lora_pairs: dict[str, dict] = {}
        for key, value in lora_weights.items():
            if key.endswith(".lora_a"):
                base_key = key[:-7]
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["a"] = value
            elif key.endswith(".lora_b"):
                base_key = key[:-7]
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["b"] = value

        # Machine epsilon for significance threshold
        sqrt_eps = sqrt_scalar(backend.finfo().eps, backend)

        layer_bounds: list[GeometricScaleBound] = []

        for base_key, pair in sorted(lora_pairs.items()):
            if "a" not in pair or "b" not in pair:
                continue

            lora_a = pair["a"]
            lora_b = pair["b"]

            # Navigate to base weight
            W = self._get_base_weight(base_model, base_key)
            if W is None:
                logger.warning(f"Could not find base weight for {base_key}")
                continue

            # Compute SVD of base weight using Backend (with U for alignment check)
            W_f32 = backend.astype(W, "float32")
            backend.eval(W_f32)
            U_W, S, _ = backend.svd(W_f32, compute_uv=True)
            backend.eval(U_W, S)

            sigma_max = float(backend.to_scalar(S[0]))
            threshold = sqrt_eps * sigma_max

            # Find effective rank and smallest significant singular value
            significant_mask = S > threshold
            eff_rank_arr = backend.sum(backend.astype(significant_mask, "int32"))
            backend.eval(eff_rank_arr)
            effective_rank = int(backend.to_scalar(eff_rank_arr))

            # Get sigma_k: smallest significant singular value
            if effective_rank > 0:
                # Use the effective_rank-th singular value (0-indexed: effective_rank - 1)
                sigma_k = float(backend.to_scalar(S[effective_rank - 1]))
            else:
                sigma_k = float(backend.to_scalar(S[-1]))

            # Detect eigengap for tighter bounds
            eigengap = self.detect_eigengap(S, backend, self.EIGENGAP_THRESHOLD)

            # Compute LoRA delta spectral norm (unscaled) using Backend
            D = backend.matmul(backend.transpose(lora_b), backend.transpose(lora_a))
            D_f32 = backend.astype(D, "float32")
            backend.eval(D_f32)
            U_D, S_D, _ = backend.svd(D_f32, compute_uv=True)
            backend.eval(U_D, S_D)
            delta_spectral = float(backend.to_scalar(S_D[0]))

            # Compute alignment quality (how much LoRA aligns with W's subspace)
            alignment = None
            if effective_rank > 0 and delta_spectral > sqrt_eps:
                # Get W's significant left singular vectors
                U_W_k = U_W[:, :effective_rank]
                # Get LoRA's first left singular vector
                delta_u1 = U_D[:, 0]
                backend.eval(U_W_k, delta_u1)
                alignment = self.compute_alignment_quality(U_W_k, delta_u1, backend)

            # Geometry-derived scale bound (standard)
            geo_scale = sigma_k / delta_spectral if delta_spectral > 0 else float("inf")

            # Adaptive bound using eigengap if available
            adaptive_bound = self.compute_adaptive_bound(sigma_k, eigengap, delta_spectral)

            scale_ratio = configured_scale / geo_scale if geo_scale > 0 else float("inf")

            layer_bounds.append(
                GeometricScaleBound(
                    layer_key=base_key,
                    sigma_max=sigma_max,
                    sigma_k=sigma_k,
                    delta_spectral_norm=delta_spectral,
                    effective_rank=effective_rank,
                    geometric_scale_bound=geo_scale,
                    configured_scale=configured_scale,
                    scale_ratio=scale_ratio,
                    eigengap=eigengap,
                    alignment=alignment,
                    adaptive_bound=adaptive_bound,
                )
            )

        if not layer_bounds:
            raise ValueError(f"No valid LoRA layers found in {adapter_path}")

        min_bound = min(lb.geometric_scale_bound for lb in layer_bounds)
        max_ratio = max(lb.scale_ratio for lb in layer_bounds)
        is_safe = max_ratio <= 1.0

        if is_safe:
            recommendation = "Scale respects spectral geometry. Safe to apply."
        elif max_ratio <= 2.0:
            recommendation = (
                f"Scale is {max_ratio:.1f}× geometric bound. "
                "Minor perturbation beyond edge - verify outputs."
            )
        elif max_ratio <= 10.0:
            recommendation = (
                f"Scale is {max_ratio:.1f}× geometric bound. "
                "Significant overflow - expect degraded performance."
            )
        else:
            recommendation = (
                f"Scale is {max_ratio:.1f}× geometric bound. "
                "CRITICAL: LoRA overwhelms base weights. Use apply_lora_geometric()."
            )

        return GeometricScaleReport(
            adapter_path=str(adapter_path),
            base_model_path=str(model_path),
            configured_alpha=alpha,
            configured_rank=rank,
            configured_scale=configured_scale,
            layer_bounds=layer_bounds,
            min_geometric_bound=min_bound,
            max_scale_ratio=max_ratio,
            is_safe=is_safe,
            recommendation=recommendation,
        )

    def apply_lora_geometric(
        self,
        model,
        adapter_path: str | Path,
        target_spectral_ratio: float = 0.9,
        use_eigengap: bool = True,
    ):
        """Apply LoRA adapter with geometry-derived per-layer scaling.

        Instead of using the configured alpha/rank scale, this method computes
        the scale for each layer from the spectral structure of the base weight:
            scale = target_ratio × σ_k(W) / ||B@A||_spectral

        This ensures the LoRA delta lives at the edge of the weight's effective
        subspace rather than overwhelming its learned structure.

        Args:
            model: The loaded model to modify (will be modified in-place)
            adapter_path: Path to LoRA adapter directory
            target_spectral_ratio: Fraction of the geometric bound to use.
                Default 0.9 is conservative, leaving 10% safety margin.
                Use 1.0 for full geometric budget, <1.0 for more conservative.
            use_eigengap: If True, use eigengap-tightened bound when available.
                Default True for tighter, safer bounds.

        Returns:
            The modified model and a dict of applied scales per layer
        """
        import json

        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        backend = get_default_backend()

        adapter_path = Path(adapter_path)
        weights_path = None
        for candidate in ["lora_weights.safetensors", "adapter_model.safetensors"]:
            if (adapter_path / candidate).exists():
                weights_path = adapter_path / candidate
                break

        if weights_path is None:
            raise FileNotFoundError(f"No LoRA weights found in {adapter_path}")

        lora_weights = backend.load_safetensors(str(weights_path))
        sqrt_eps = sqrt_scalar(backend.finfo().eps, backend)

        # Organize LoRA pairs
        lora_pairs: dict[str, dict] = {}
        for key, value in lora_weights.items():
            if key.endswith(".lora_a"):
                base_key = key[:-7]
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["a"] = value
            elif key.endswith(".lora_b"):
                base_key = key[:-7]
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["b"] = value

        base_model = getattr(model, "model", model)
        applied_scales: dict[str, float] = {}

        for base_key, pair in lora_pairs.items():
            if "a" not in pair or "b" not in pair:
                continue

            lora_a = pair["a"]
            lora_b = pair["b"]

            # Get base weight
            W = self._get_base_weight(base_model, base_key)
            if W is None:
                logger.warning(f"Could not find base weight for {base_key}")
                continue

            # Compute geometry-derived scale using Backend
            W_f32 = backend.astype(W, "float32")
            backend.eval(W_f32)
            _, S, _ = backend.svd(W_f32, compute_uv=True)
            backend.eval(S)

            sigma_max = float(backend.to_scalar(S[0]))
            threshold = sqrt_eps * sigma_max
            significant_mask = S > threshold
            eff_rank_arr = backend.sum(backend.astype(significant_mask, "int32"))
            backend.eval(eff_rank_arr)
            effective_rank = int(backend.to_scalar(eff_rank_arr))
            sigma_k = float(backend.to_scalar(S[effective_rank - 1])) if effective_rank > 0 else float(backend.to_scalar(S[-1]))

            # Detect eigengap for potentially tighter bound
            eigengap = None
            if use_eigengap:
                eigengap = self.detect_eigengap(S, backend, self.EIGENGAP_THRESHOLD)

            # Delta spectral norm using Backend
            D = backend.matmul(backend.transpose(lora_b), backend.transpose(lora_a))
            D_f32 = backend.astype(D, "float32")
            backend.eval(D_f32)
            _, S_D, _ = backend.svd(D_f32, compute_uv=True)
            backend.eval(S_D)
            delta_spectral = float(backend.to_scalar(S_D[0]))

            # Geometry-derived scale (standard or adaptive based on eigengap)
            if use_eigengap and eigengap is not None:
                geo_scale = self.compute_adaptive_bound(sigma_k, eigengap, delta_spectral)
            else:
                geo_scale = sigma_k / delta_spectral if delta_spectral > 0 else 0.0

            # Apply target ratio for safety margin
            applied_scale = target_spectral_ratio * geo_scale

            # Apply scaled delta
            scaled_delta = D * applied_scale
            new_weight = W + backend.astype(scaled_delta, W.dtype)

            # Set the weight back
            self._set_base_weight(base_model, base_key, new_weight)
            applied_scales[base_key] = applied_scale

        # Evaluate model parameters
        params = model.parameters()
        if isinstance(params, dict):
            for p in params.values():
                if hasattr(p, "shape"):
                    backend.eval(p)
        return model, applied_scales

    def _get_base_weight(self, base_model, lora_key: str):
        """Get the base weight matrix for a LoRA key."""
        # Parse key like "model.layers.7.feed_forward.w1"
        parts = lora_key.split(".")
        if parts[0] == "model":
            parts = parts[1:]

        current = base_model
        for part in parts[:-1]:
            if part == "layers":
                continue
            elif part.isdigit():
                current = current.layers[int(part)]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        # Get the final Linear module
        param_name = parts[-1]
        if hasattr(current, param_name):
            linear = getattr(current, param_name)
            if hasattr(linear, "weight"):
                return linear.weight
        return None

    def _set_base_weight(self, base_model, lora_key: str, new_weight):
        """Set the base weight matrix for a LoRA key."""
        parts = lora_key.split(".")
        if parts[0] == "model":
            parts = parts[1:]

        current = base_model
        for part in parts[:-1]:
            if part == "layers":
                continue
            elif part.isdigit():
                current = current.layers[int(part)]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return

        param_name = parts[-1]
        if hasattr(current, param_name):
            linear = getattr(current, param_name)
            if hasattr(linear, "weight"):
                linear.weight = new_weight
