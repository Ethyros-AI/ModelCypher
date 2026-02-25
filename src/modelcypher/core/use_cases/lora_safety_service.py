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
   - σ_k is smallest precision-significant singular value
     (above max(m,n) × ε × σ_max)
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
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.cayley_lora import PerDirectionBoundResult
    from modelcypher.ports.backend import Array, Backend


logger = logging.getLogger(__name__)


@dataclass
class EigengapInfo:
    """Information about spectral gaps in a weight matrix.

    Weyl no-crossing condition: singular value crossing at rank k occurs
    when ||E||_2 > gap_k / 2, where gap_k = σ_k - σ_{k+1}. When a gap
    exists, the scale bound tightens to gap_k / (2 × ||Δ||_spectral).

    Reference: Weyl (1912); see also Tran et al. arXiv:2510.25670.
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
    is_well_aligned: bool  # True if alignment exceeds random subspace baseline


@dataclass
class GeometricScaleBound:
    """Geometry-derived scale bound for a single LoRA layer.

    The scale bound is derived from the spectral structure of the base weight:
        scale_bound = σ_k(W) / ||B@A||_spectral

    Where σ_k is the smallest precision-significant singular value of W
    (above max(m,n) × ε × σ_max).
    This ensures the LoRA delta adds information at the edge of the weight's
    effective subspace rather than overwhelming its learned structure.

    Enhanced with Weyl no-crossing detection and alignment quality (RoRA).
    """

    layer_key: str
    sigma_max: float  # Largest singular value of base weight
    sigma_k: float  # Smallest precision-significant singular value
    delta_spectral_norm: float  # Spectral norm of LoRA delta (unscaled)
    effective_rank: int  # Number of significant singular values
    geometric_scale_bound: float  # Max safe scale from geometry
    configured_scale: float  # Scale that would be used (alpha/rank)
    scale_ratio: float  # configured / geometric (>1 means too aggressive)
    # New fields for enhanced analysis
    eigengap: EigengapInfo | None = None  # Eigengap info if detected
    alignment: AlignmentQuality | None = None  # LoRA-base alignment quality
    adaptive_bound: float | None = None  # Tighter bound using eigengap if available
    per_direction: "PerDirectionBoundResult" | None = None  # Per-direction verification result



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
    guidance: str = "Use high_quality problems and inspect measured CKA/barrier distributions."


class LoRASafetyService:
    """Service for LoRA safety analysis.

    Combines Fisher Information, Mode Connectivity, and Goldilocks Quality
    scoring to provide safe and effective LoRA training guidance.
    """

    @staticmethod
    def _prompt_id(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def _percentile(values: list[float], numerator: int, denominator: int) -> float:
        """Compute integer-indexed percentile from measured values."""
        if not values:
            raise ValueError("percentile requires non-empty values")
        ordered = sorted(values)
        idx = ((len(ordered) - 1) * numerator) // denominator
        return ordered[idx]

    @staticmethod
    def detect_eigengap(
        singular_values: "Array",
        backend: "Backend",
        gap_threshold: float | None = None,
    ) -> EigengapInfo:
        """Detect spectral gap for tighter scale bounds.

        Weyl no-crossing condition: when W has a spectral gap at position k
        (σ_k / σ_{k+1} > threshold), singular value crossing requires
        ||E||_2 > gap_k / 2. This tightens the scale bound beyond the
        standard Weyl bound σ_k / ||Δ||_spectral.

        If no threshold is provided, uses the numerical distinguishability
        floor (1 + machine_epsilon) for the singular-value dtype.

        Args:
            singular_values: Singular values in descending order.
            backend: Compute backend.
            gap_threshold: Minimum ratio to count as a gap. If None, uses
                1 + machine_epsilon for the singular-value dtype.

        Returns:
            EigengapInfo with gap position, value, and tightened bound.

        Reference: Weyl (1912); see also Tran et al. arXiv:2510.25670.
        """
        b = backend
        S = singular_values

        # Need at least 2 values to detect gap
        if len(S.shape) == 0 or S.shape[0] < 2:
            return EigengapInfo(
                position=None, gap_value=0.0, gap_ratio=1.0, tightened_bound=None
            )

        eps = float(b.finfo(S.dtype).eps)
        if gap_threshold is None:
            gap_threshold = 1.0 + eps

        # Compute ratios of consecutive singular values
        # Add dtype-derived epsilon to prevent division by zero.
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
        eps = float(b.finfo(delta_u1.dtype).eps)
        alignment_ratio = proj_norm_val / max(delta_norm_val, eps)

        # Random baseline for a unit vector projected to a k-dim subspace in R^m.
        # E[||P_k u||_2^2] = k / m  =>  E[||P_k u||_2] ~ sqrt(k / m)
        m_dim = int(W_U_k.shape[0]) if len(W_U_k.shape) >= 1 else 1
        k_dim = int(W_U_k.shape[1]) if len(W_U_k.shape) >= 2 else m_dim
        random_baseline = (k_dim / max(m_dim, 1)) ** 0.5

        return AlignmentQuality(
            projection_norm=proj_norm_val,
            alignment_ratio=alignment_ratio,
            is_well_aligned=alignment_ratio > (random_baseline + eps),
        )

    @staticmethod
    def compute_adaptive_bound(
        sigma_k: float,
        eigengap: EigengapInfo,
        delta_spectral: float,
        min_delta: float = 0.0,
    ) -> float:
        """Compute adaptive scale bound using eigengap when available.

        Uses the tighter of:
        1. Standard bound: σ_k / ||Δ||_spectral
        2. Eigengap bound: (gap_value / 2) / ||Δ||_spectral

        Args:
            sigma_k: Smallest precision-significant singular value.
            eigengap: Eigengap information (may have no gap).
            delta_spectral: Spectral norm of LoRA delta.

        Returns:
            Adaptive scale bound (tighter than standard when eigengap exists).
        """
        if delta_spectral <= min_delta:
            return float("inf")

        standard_bound = sigma_k / delta_spectral

        if eigengap.tightened_bound is not None:
            eigengap_bound = eigengap.tightened_bound / delta_spectral
            return min(standard_bound, eigengap_bound)

        return standard_bound

    @staticmethod
    def verify_per_direction_bounds(
        B: "Array",
        A: "Array",
        W: "Array",
        backend: "Backend",
        rtol: float = 1.01,
    ) -> "PerDirectionBoundResult":
        """Verify that LoRA delta respects per-direction spectral bounds.

        Computes U^T @ (B@A) @ V and checks diagonal entries against singular values of W.
        Directly uses NBLoRALayer.verify_per_direction_bounds logic but for explicit B, A.

        Args:
            B: LoRA output matrix [out, rank]
            A: LoRA input matrix [rank, in]
            W: Base weight matrix [out, in]
            backend: Compute backend
            rtol: Relative tolerance (default 1.01)

        Returns:
            PerDirectionBoundResult with verification details
        """
        from modelcypher.core.domain.geometry.cayley_lora import NBLoRALayer

        # Create a temporary NBLoRALayer wrapper to reuse verification logic
        # We don't need Cayley transform here since we have explicit B, A
        # But we can override get_effective_delta

        class ExplicitLoRAWrapper(NBLoRALayer):
            def __init__(self, B, A, backend):
                self._B = B
                self._A = A
                self._backend = backend

            def get_effective_delta(self):
                # Simple product B @ A
                return self._backend.matmul(self._B, self._A)

        # Mock config (not used for verification logic)
        wrapper = ExplicitLoRAWrapper(B, A, backend)

        return wrapper.verify_per_direction_bounds(W, rtol=rtol)


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
        fisher_values = [score for _, score in sorted_modules]
        if fisher_values:
            excellent_max = self._percentile(fisher_values, 1, 4)
            good_max = self._percentile(fisher_values, 2, 4)
            acceptable_max = self._percentile(fisher_values, 3, 4)
        else:
            excellent_max = good_max = acceptable_max = 0.0

        # Create recommendations
        recommendations = []
        for module, score in sorted_modules[:top_k]:
            if score <= excellent_max:
                rec = "EXCELLENT"
            elif score <= good_max:
                rec = "GOOD"
            elif score <= acceptable_max:
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
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

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

        # Safety is derived from numerical excess above a monotonic path.
        # normalized == 1 means no additional barrier beyond endpoint loss.
        sqrt_eps = sqrt_scalar(backend.finfo().eps, backend)
        excess = barrier_normalized - 1.0
        if excess <= sqrt_eps:
            safety_level = "SAFE"
            recommendation = "No measurable barrier beyond endpoint loss."
        elif cka_at_target > sqrt_eps:
            safety_level = "CAUTION"
            recommendation = (
                f"Measured barrier excess={excess:.6f}; verify downstream behavior."
            )
        else:
            safety_level = "WARNING"
            recommendation = (
                f"Large barrier excess={excess:.6f} with weak endpoint CKA; "
                "target may leave source basin."
            )

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
                "cka": quality.cka_similarity,
                "barrier": quality.barrier_height,
                "fisher": quality.fisher_mean,
            })

        # Sort by CKA similarity (raw measurement, no composite score)
        problem_scores.sort(key=lambda x: x["cka"], reverse=True)

        # Compute tertile distribution from raw CKA (equal-size groups)
        n = len(problem_scores)
        tercile_size = max(1, n // 3)
        high_group = problem_scores[:tercile_size]
        mid_group = problem_scores[tercile_size:2 * tercile_size]
        low_group = problem_scores[2 * tercile_size:]

        high_mean_cka = sum(p["cka"] for p in high_group) / max(len(high_group), 1)
        mid_mean_cka = sum(p["cka"] for p in mid_group) / max(len(mid_group), 1)
        low_mean_cka = sum(p["cka"] for p in low_group) / max(len(low_group), 1)

        return CurriculumScoreResult(
            model_path=str(model_path),
            n_problems=len(problem_scores),
            quality_distribution={
                "high_cka_tercile": {"count": len(high_group), "mean_cka": high_mean_cka},
                "mid_cka_tercile": {"count": len(mid_group), "mean_cka": mid_mean_cka},
                "low_cka_tercile": {"count": len(low_group), "mean_cka": low_mean_cka},
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

        Difficulty bands are derived from the measured CKA distribution of
        the provided problems:
        - hard: lower tercile
        - medium: middle tercile
        - easy: upper tercile

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

        cka_values = [float(p["cka"]) for p in result.top_problems]
        if not cka_values:
            return []

        hard_upper = self._percentile(cka_values, 1, 3)
        medium_upper = self._percentile(cka_values, 2, 3)

        # Map difficulty to data-derived CKA bands.
        difficulty_map = {
            "easy": lambda cka: cka > medium_upper,
            "medium": lambda cka: hard_upper <= cka <= medium_upper,
            "hard": lambda cka: cka < hard_upper,
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
        eps = float(backend.finfo(source_centered.dtype).eps)
        normalized = barrier_height / max(target_loss, eps)

        return barrier_height, normalized, cka_at_target

    def compute_geometric_scale(
        self,
        model_path: str | Path,
        adapter_path: str | Path,
    ) -> GeometricScaleReport:
        """Compute geometry-derived scale bounds for a LoRA adapter.

        The scale bound for each layer is derived from the spectral structure:
            scale_bound = σ_k(W) / ||B@A||_spectral

        Where σ_k is the smallest precision-significant singular value of the
        base weight (those above max(m,n) × ε × σ_max). This ensures the LoRA
        delta adds information at the edge of the weight's effective subspace.

        Args:
            model_path: Path to base model
            adapter_path: Path to LoRA adapter directory

        Returns:
            GeometricScaleReport with per-layer analysis and recommendations
        """
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

        backend = get_default_backend()

        adapter_path = Path(adapter_path)
        config_path = adapter_path / "adapter_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json in {adapter_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Extract config - rank must be explicitly present in adapter metadata.
        rank = config.get("rank", config.get("r"))
        if rank is None:
            raise ValueError(
                "Adapter config must include rank (rank or r). "
                "Heuristic rank fallback has been removed."
            )
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f"Adapter rank must be positive, got {rank}.")
        alpha = config.get("alpha", config.get("lora_alpha", rank))
        alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError(f"Adapter alpha must be positive, got {alpha}.")
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

            # Navigate to base weight (dequantized when base module is quantized).
            W = self._get_weight_for_svd(base_model, base_key, backend)
            if W is None:
                logger.warning(f"Could not find base weight for {base_key}")
                continue

            # Compute SVD of base weight using Backend (with U for alignment check)
            W_f32 = backend.astype(W, "float32")
            backend.eval(W_f32)
            U_W, S, _ = backend.svd(W_f32, compute_uv=True)
            backend.eval(U_W, S)

            sigma_max = float(backend.to_scalar(S[0]))
            max_dim = max(int(W_f32.shape[0]), int(W_f32.shape[1]))
            eps_svd = float(backend.finfo(S.dtype).eps)
            threshold = float(max_dim) * eps_svd * sigma_max

            # Find effective rank and smallest precision-significant singular value
            significant_mask = S > threshold
            eff_rank_arr = backend.sum(backend.astype(significant_mask, "int32"))
            backend.eval(eff_rank_arr)
            effective_rank = int(backend.to_scalar(eff_rank_arr))

            # Get sigma_k: smallest precision-significant singular value
            if effective_rank > 0:
                # Use the effective_rank-th singular value (0-indexed: effective_rank - 1)
                sigma_k = float(backend.to_scalar(S[effective_rank - 1]))
            else:
                sigma_k = float(backend.to_scalar(S[-1]))

            # Detect eigengap for tighter bounds
            eigengap = self.detect_eigengap(S, backend)

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
            adaptive_bound = self.compute_adaptive_bound(
                sigma_k,
                eigengap,
                delta_spectral,
                min_delta=sqrt_eps,
            )

            # Per-direction verification
            per_direction = self.verify_per_direction_bounds(lora_b, lora_a, W, backend)

            ratio_bound = adaptive_bound if adaptive_bound is not None else geo_scale
            scale_ratio = configured_scale / ratio_bound if ratio_bound > 0 else float("inf")


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
                    per_direction=per_direction,
                )
            )


        if not layer_bounds:
            raise ValueError(f"No valid LoRA layers found in {adapter_path}")

        min_bound = min(
            lb.adaptive_bound
            if lb.adaptive_bound is not None
            else lb.geometric_scale_bound
            for lb in layer_bounds
        )
        max_ratio = max(lb.scale_ratio for lb in layer_bounds)
        is_safe = max_ratio <= (1.0 + sqrt_eps)

        if is_safe:
            recommendation = (
                f"Scale respects measured spectral bound "
                f"(max_ratio={max_ratio:.6f}, tolerance={sqrt_eps:.6f})."
            )
        else:
            overflow = max_ratio - 1.0
            recommendation = (
                f"Scale exceeds measured spectral bound by {overflow:.6f} "
                f"(max_ratio={max_ratio:.6f}). Use apply_lora_geometric()."
            )

        # Check for per-direction violations (which might happen even if global norm is safe-ish)
        # We only flag this if we haven't already flagged a critical global violation
        if is_safe:
            violation_count = sum(len(lb.per_direction.violations) for lb in layer_bounds if lb.per_direction)
            if violation_count > 0:
                is_safe = False
                recommendation += f" WARNING: {violation_count} per-direction violations detected."


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
        target_spectral_ratio: float = 1.0,
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
                Default 1.0 applies the measured geometric bound directly.
                Use <1.0 only if you explicitly want extra conservatism.
            use_eigengap: If True, use eigengap-tightened bound when available.
                Default True for tighter, safer bounds.

        Returns:
            The modified model and a dict of applied scales per layer
        """

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

            # Get raw base weight for update and dequantized weight for SVD.
            W_update = self._get_base_weight(base_model, base_key)
            W_svd = self._get_weight_for_svd(base_model, base_key, backend)
            if W_update is None or W_svd is None:
                logger.warning(f"Could not find base weight for {base_key}")
                continue

            # Compute geometry-derived scale using Backend
            W_f32 = backend.astype(W_svd, "float32")
            backend.eval(W_f32)
            _, S, _ = backend.svd(W_f32, compute_uv=True)
            backend.eval(S)

            sigma_max = float(backend.to_scalar(S[0]))
            max_dim = max(int(W_f32.shape[0]), int(W_f32.shape[1]))
            eps_svd = float(backend.finfo(S.dtype).eps)
            threshold = float(max_dim) * eps_svd * sigma_max
            significant_mask = S > threshold
            eff_rank_arr = backend.sum(backend.astype(significant_mask, "int32"))
            backend.eval(eff_rank_arr)
            effective_rank = int(backend.to_scalar(eff_rank_arr))
            sigma_k = (
                float(backend.to_scalar(S[effective_rank - 1]))
                if effective_rank > 0
                else float(backend.to_scalar(S[-1]))
            )

            # Detect eigengap for potentially tighter bound
            eigengap = None
            if use_eigengap:
                eigengap = self.detect_eigengap(S, backend)

            # Delta spectral norm using Backend
            D = backend.matmul(backend.transpose(lora_b), backend.transpose(lora_a))
            D_f32 = backend.astype(D, "float32")
            backend.eval(D_f32)
            _, S_D, _ = backend.svd(D_f32, compute_uv=True)
            backend.eval(S_D)
            delta_spectral = float(backend.to_scalar(S_D[0]))

            # Geometry-derived scale (standard or adaptive based on eigengap)
            if use_eigengap and eigengap is not None:
                geo_scale = self.compute_adaptive_bound(
                    sigma_k,
                    eigengap,
                    delta_spectral,
                    min_delta=sqrt_eps,
                )
            else:
                geo_scale = sigma_k / delta_spectral if delta_spectral > sqrt_eps else 0.0

            # Apply target ratio for safety margin
            applied_scale = target_spectral_ratio * geo_scale

            # Apply scaled delta
            scaled_delta = D * applied_scale
            if tuple(int(x) for x in W_update.shape) != tuple(int(x) for x in scaled_delta.shape):
                logger.warning(
                    "Shape mismatch when applying LoRA to %s (%s vs %s); "
                    "using dequantized base weight for update.",
                    base_key,
                    tuple(int(x) for x in W_update.shape),
                    tuple(int(x) for x in scaled_delta.shape),
                )
                W_update = W_svd
            new_weight = W_update + backend.astype(scaled_delta, W_update.dtype)

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

    def _resolve_base_linear(self, base_model, lora_key: str):
        """Resolve a LoRA key to its target projection module."""
        parts = lora_key.split(".")
        if parts[0] == "model":
            parts = parts[1:]

        current = base_model
        for part in parts[:-1]:
            if part == "layers":
                continue
            if part.isdigit():
                current = current.layers[int(part)]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        param_name = parts[-1]
        if hasattr(current, param_name):
            return getattr(current, param_name)
        return None

    @staticmethod
    def _is_quantized_linear(linear) -> bool:
        return (
            hasattr(linear, "weight")
            and hasattr(linear, "scales")
            and hasattr(linear, "group_size")
            and hasattr(linear, "bits")
        )

    @staticmethod
    def _dequantize_linear_weight(linear, backend: "Backend"):
        """Dequantize a quantized linear module weight via Backend protocol."""
        biases = getattr(linear, "biases", None)
        bits = int(getattr(linear, "bits"))
        group_size = int(getattr(linear, "group_size"))

        mode = "affine"
        if biases is None and bits == 4 and group_size == 32:
            mode = "mxfp4"

        try:
            dequantized = backend.dequantize(
                linear.weight,
                linear.scales,
                biases=biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            backend.eval(dequantized)
            return dequantized
        except Exception as exc:
            fallback_mode = "affine" if mode == "mxfp4" else "mxfp4"
            try:
                dequantized = backend.dequantize(
                    linear.weight,
                    linear.scales,
                    biases=biases,
                    group_size=group_size,
                    bits=bits,
                    mode=fallback_mode,
                )
                backend.eval(dequantized)
                logger.debug(
                    "Dequantized %s with fallback mode=%s after mode=%s failed: %s",
                    getattr(linear, "__class__", type(linear)).__name__,
                    fallback_mode,
                    mode,
                    exc,
                )
                return dequantized
            except Exception:
                logger.warning(
                    "Failed to dequantize %s with modes (%s, %s).",
                    getattr(linear, "__class__", type(linear)).__name__,
                    mode,
                    fallback_mode,
                )
                return None

    def _get_weight_for_svd(self, base_model, lora_key: str, backend: "Backend"):
        """Get base weight for spectral analysis, dequantizing when needed."""
        linear = self._resolve_base_linear(base_model, lora_key)
        if linear is None or not hasattr(linear, "weight"):
            return None
        if self._is_quantized_linear(linear):
            dequantized = self._dequantize_linear_weight(linear, backend)
            if dequantized is not None:
                return dequantized
        return linear.weight

    def _get_base_weight(self, base_model, lora_key: str):
        """Get the base weight matrix for a LoRA key."""
        linear = self._resolve_base_linear(base_model, lora_key)
        if linear is not None and hasattr(linear, "weight"):
            return linear.weight
        return None

    def _set_base_weight(self, base_model, lora_key: str, new_weight):
        """Set the base weight matrix for a LoRA key."""
        linear = self._resolve_base_linear(base_model, lora_key)
        if linear is not None and hasattr(linear, "weight"):
            linear.weight = new_weight
