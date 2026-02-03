# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Curriculum Profiler: Geometric difficulty measurement for training problems.

This module measures difficulty geometrically - no heuristics, only measurements.

Available signals:
    - CKA similarity to reference (from goldilocks_quality)
    - Activation barrier height (from goldilocks_quality)
    - Fisher Information (from goldilocks_quality)
    - Trajectory curvature (from trajectory_complexity)
    - Local density (from density_estimator)
    - Intrinsic dimension (from intrinsic_dimension)

Usage:
    profiler = CurriculumProfiler(model, tokenizer, backend)
    
    # Profile a set of problems
    profiles = profiler.profile_problems(problems, reference_prompts)
    
    # Get correlation-ready output
    df = profiles.to_dataframe()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.density_estimator import DensityEstimator
from modelcypher.core.domain.geometry.goldilocks_quality import compute_goldilocks_quality
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.trajectory_complexity import TrajectoryComplexity

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class ProblemProfile:
    """Geometric profile for a single problem.
    
    All fields are raw measurements with no interpretation.
    """
    
    problem_id: str
    prompt: str
    
    # Goldilocks metrics (validated r=-0.955)
    cka_similarity: float
    barrier_height: float
    fisher_mean: float
    goldilocks_score: float
    
    # Trajectory metrics
    trajectory_curvature_mean: float
    trajectory_curvature_max: float
    trajectory_path_length_ratio: float
    trajectory_spectral_entropy: float
    
    # Density metrics
    local_density: float
    density_percentile: float  # Relative to corpus
    
    # Dimension metrics
    intrinsic_dimension: float
    
    # Layer used for profiling
    layer_idx: int
    
    # Composite difficulty score (Fisher-dominant)
    difficulty_score: float = 0.0
    
    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "problem_id": self.problem_id,
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "cka_similarity": self.cka_similarity,
            "barrier_height": self.barrier_height,
            "fisher_mean": self.fisher_mean,
            "goldilocks_score": self.goldilocks_score,
            "trajectory_curvature_mean": self.trajectory_curvature_mean,
            "trajectory_curvature_max": self.trajectory_curvature_max,
            "trajectory_path_length_ratio": self.trajectory_path_length_ratio,
            "trajectory_spectral_entropy": self.trajectory_spectral_entropy,
            "local_density": self.local_density,
            "density_percentile": self.density_percentile,
            "intrinsic_dimension": self.intrinsic_dimension,
            "layer_idx": self.layer_idx,
            "difficulty_score": self.difficulty_score,
        }


@dataclass
class CurriculumProfiles:
    """Collection of problem profiles with analysis methods."""
    
    profiles: list[ProblemProfile] = field(default_factory=list)
    model_id: str = ""
    reference_count: int = 0
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis."""
        try:
            import pandas as pd
            return pd.DataFrame([p.as_dict() for p in self.profiles])
        except ImportError:
            logger.warning("pandas not available, returning list of dicts")
            return [p.as_dict() for p in self.profiles]
    
    def filter_by_goldilocks(
        self,
        cka_min: float = 0.85,
        cka_max: float = 0.95,
    ) -> list[ProblemProfile]:
        """Filter problems in the Goldilocks zone."""
        return [
            p for p in self.profiles
            if cka_min <= p.cka_similarity <= cka_max
        ]
    
    def filter_by_density_percentile(
        self,
        min_percentile: float = 0.0,
        max_percentile: float = 0.25,
    ) -> list[ProblemProfile]:
        """Filter sparse region problems (low density)."""
        return [
            p for p in self.profiles
            if min_percentile <= p.density_percentile <= max_percentile
        ]
    
    def sort_by_difficulty(
        self,
        metric: str = "cka_similarity",
        ascending: bool = True,
    ) -> list[ProblemProfile]:
        """Sort problems by difficulty metric.
        
        Default: ascending CKA = hardest first (lowest similarity to known).
        """
        return sorted(
            self.profiles,
            key=lambda p: getattr(p, metric),
            reverse=not ascending,
        )
    
    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "reference_count": self.reference_count,
            "problem_count": len(self.profiles),
            "profiles": [p.as_dict() for p in self.profiles],
        }
    
    def compute_difficulty_scores(self) -> None:
        """Compute composite difficulty scores for all profiles.
        
        Uses Fisher-dominant weighting based on Experiment 18 findings:
        - Fisher is 9% higher for incorrect answers (direct uncertainty signal)
        - CKA measures syntax, not computational complexity
        - Barrier captures activation divergence
        - Curvature captures processing complexity
        """
        for profile in self.profiles:
            profile.difficulty_score = self._compute_single_difficulty(profile)
    
    @staticmethod
    def _compute_single_difficulty(profile: ProblemProfile) -> float:
        """Compute composite difficulty for a single profile.
        
        Fisher-dominant weighting:
        - 40% Fisher (computational uncertainty)
        - 30% Barrier (activation divergence)
        - 15% CKA (syntactic distance)
        - 15% Curvature (processing complexity)
        """
        import math
        
        # Scale Fisher to [0, 1] range (typically 0.0003-0.0005)
        fisher_score = min(profile.fisher_mean * 2000, 1.0)
        
        # Barrier is already in reasonable range
        barrier_score = profile.barrier_height
        
        # Invert CKA (lower similarity = harder)
        syntax_score = 1 - profile.cka_similarity
        
        # Normalize curvature (typical range 1.5-2.5, cap at 3)
        curvature = profile.trajectory_curvature_mean
        if math.isnan(curvature):
            curvature_score = 0.5  # Default if unavailable
        else:
            curvature_score = min(curvature / 3.0, 1.0)
        
        # Weighted combination (Fisher-dominant)
        return (
            0.40 * fisher_score +
            0.30 * barrier_score +
            0.15 * syntax_score +
            0.15 * curvature_score
        )
    
    def filter_by_difficulty_score(
        self,
        min_score: float = 0.3,
        max_score: float = 0.7,
    ) -> list[ProblemProfile]:
        """Filter problems in the optimal difficulty range.
        
        Default range (0.3-0.7) selects moderate difficulty problems
        where learning is most effective (Goldilocks zone).
        """
        return [
            p for p in self.profiles
            if min_score <= p.difficulty_score <= max_score
        ]
    
    def select_curriculum(
        self,
        n_samples: int = 100,
        strategy: str = "balanced",
    ) -> list[ProblemProfile]:
        """Select training curriculum from profiled problems.
        
        Strategies:
        - 'balanced': Mix of easy, medium, hard (recommended)
        - 'hardest': Focus on highest difficulty
        - 'goldilocks': Moderate difficulty only
        """
        # Ensure scores are computed
        self.compute_difficulty_scores()
        
        if strategy == "hardest":
            sorted_profiles = sorted(
                self.profiles,
                key=lambda p: p.difficulty_score,
                reverse=True,
            )
            return sorted_profiles[:n_samples]
        
        elif strategy == "goldilocks":
            goldilocks = self.filter_by_difficulty_score(0.3, 0.7)
            return goldilocks[:n_samples]
        
        else:  # balanced
            # Split into thirds
            sorted_profiles = sorted(
                self.profiles,
                key=lambda p: p.difficulty_score,
            )
            n = len(sorted_profiles)
            easy = sorted_profiles[:n//3]
            medium = sorted_profiles[n//3:2*n//3]
            hard = sorted_profiles[2*n//3:]
            
            # Sample proportionally (20% easy, 60% medium, 20% hard)
            n_easy = max(1, n_samples // 5)
            n_hard = max(1, n_samples // 5)
            n_medium = n_samples - n_easy - n_hard
            
            selected = (
                easy[:n_easy] +
                medium[:n_medium] +
                hard[:n_hard]
            )
            return selected[:n_samples]


class CurriculumProfiler:
    """Profile problems geometrically for curriculum design.
    
    Combines multiple geometric signals to measure problem difficulty
    without heuristics - all values are direct measurements.
    
    Example:
        >>> profiler = CurriculumProfiler(model, tokenizer)
        >>> profiles = profiler.profile_problems(
        ...     problems=["What is 2+2?", "Prove Fermat's Last Theorem"],
        ...     reference_prompts=["What is 1+1?", "What is 3+3?"],
        ... )
        >>> for p in profiles.sort_by_difficulty():
        ...     print(f"{p.prompt}: CKA={p.cka_similarity:.3f}")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        backend: "Backend | None" = None,
        layer_idx: int | None = None,
    ):
        """Initialize the profiler.
        
        Args:
            model: The model to profile (must support get_activations or forward hooks)
            tokenizer: Tokenizer for the model
            backend: Backend for tensor operations
            layer_idx: Which layer to extract activations from.
                       If None, uses middle layer (n_layers // 2).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend or get_default_backend()
        
        # Determine layer to profile
        if layer_idx is not None:
            self.layer_idx = layer_idx
        else:
            # Default to middle layer
            n_layers = self._get_n_layers()
            self.layer_idx = n_layers // 2
        
        # Initialize geometry components
        self._density_estimator = DensityEstimator(backend=self.backend)
        self._trajectory_complexity = TrajectoryComplexity(backend=self.backend)
        self._intrinsic_dimension = IntrinsicDimension(backend=self.backend)
        
        logger.info(f"CurriculumProfiler initialized, profiling layer {self.layer_idx}")
    
    def _get_n_layers(self) -> int:
        """Get number of layers in model."""
        # Try different model architectures
        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "num_hidden_layers"):
                return config.num_hidden_layers
            if hasattr(config, "n_layers"):
                return config.n_layers
        
        # Try to count model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return len(self.model.layers)
        
        # Default fallback
        logger.warning("Could not determine layer count, defaulting to 16")
        return 16
    
    def _get_activations(self, prompts: list[str], layer_idx: int) -> "Array":
        """Extract activations for prompts at specified layer.
        
        Returns:
            Array of shape [n_prompts, hidden_dim]
        """
        from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
        
        provider = MLXActivationProvider()
        
        # Extract activations for all prompts
        all_activations = []
        for prompt in prompts:
            try:
                # Get hidden activations for all layers
                layer_acts = provider.collect_hidden_activations(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    text=prompt,
                )
                # Get activation for target layer
                if layer_idx in layer_acts:
                    act = layer_acts[layer_idx]
                    all_activations.append(act)
            except Exception as e:
                logger.warning(f"Failed to extract activations for prompt: {e}")
                continue
        
        if not all_activations:
            raise ValueError("No activations extracted")
        
        return self.backend.stack(all_activations, axis=0)
    
    def _get_layer_activations(self, prompt: str) -> dict[int, "Array"]:
        """Get activations from all layers for trajectory analysis."""
        from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider
        
        provider = MLXActivationProvider()
        
        # Get hidden activations for all layers
        layer_acts = provider.collect_hidden_activations(
            model=self.model,
            tokenizer=self.tokenizer,
            text=prompt,
        )
        
        # Sample every other layer to reduce computation
        n_layers = self._get_n_layers()
        sampled_layers = list(range(0, n_layers, 2))
        
        return {k: v for k, v in layer_acts.items() if k in sampled_layers}
    
    def profile_problems(
        self,
        problems: list[str],
        reference_prompts: list[str] | None = None,
        problem_ids: list[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> CurriculumProfiles:
        """Profile a set of problems geometrically.
        
        Args:
            problems: List of problem prompts to profile
            reference_prompts: Reference prompts for CKA comparison.
                              If None, uses first 10% of problems as reference.
            problem_ids: Optional IDs for each problem.
            progress_callback: Optional callback(current, total).
        
        Returns:
            CurriculumProfiles with geometric measurements.
        """
        logger.info(f"Profiling {len(problems)} problems at layer {self.layer_idx}")
        
        # Generate problem IDs if not provided
        if problem_ids is None:
            problem_ids = [f"p{i}" for i in range(len(problems))]
        
        # Use first 10% as reference if not provided
        if reference_prompts is None:
            n_ref = max(1, len(problems) // 10)
            reference_prompts = problems[:n_ref]
            logger.info(f"Using first {n_ref} problems as reference")
        
        # Get reference activations
        reference_activations = self._get_activations(reference_prompts, self.layer_idx)
        logger.info(f"Reference activations shape: {reference_activations.shape}")
        
        # Get all problem activations for density calculation
        all_problem_activations = self._get_activations(problems, self.layer_idx)
        
        # Compute density for all problems at once
        density_result = self._density_estimator.compute(all_problem_activations)
        # Force evaluation and convert to list
        import mlx.core as mx
        mx.eval(density_result.densities)
        densities = density_result.densities
        
        # Compute density percentiles
        density_percentiles = self._compute_percentiles(densities)
        
        # Profile each problem
        profiles = []
        for i, (prompt, pid) in enumerate(zip(problems, problem_ids)):
            if progress_callback:
                progress_callback(i + 1, len(problems))
            
            try:
                profile = self._profile_single(
                    prompt=prompt,
                    problem_id=pid,
                    problem_activation=all_problem_activations[i],
                    reference_activations=reference_activations,
                    local_density=float(densities[i]),
                    density_percentile=density_percentiles[i],
                )
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to profile problem {pid}: {e}")
                continue
        
        logger.info(f"Successfully profiled {len(profiles)}/{len(problems)} problems")
        
        return CurriculumProfiles(
            profiles=profiles,
            model_id=getattr(self.model, "name_or_path", "unknown"),
            reference_count=len(reference_prompts),
        )
    
    def _profile_single(
        self,
        prompt: str,
        problem_id: str,
        problem_activation: "Array",
        reference_activations: "Array",
        local_density: float,
        density_percentile: float,
    ) -> ProblemProfile:
        """Profile a single problem."""
        # Expand dimensions for goldilocks computation
        problem_act_2d = self.backend.expand_dims(problem_activation, axis=0)
        
        # Goldilocks quality (validated metrics)
        goldilocks = compute_goldilocks_quality(
            activations=problem_act_2d,
            reference_activations=reference_activations,
            backend=self.backend,
        )
        
        # Trajectory complexity (curvature through layers)
        try:
            layer_activations = self._get_layer_activations(prompt)
            trajectory = self._trajectory_complexity.compute(layer_activations)
            traj_curvature_mean = trajectory.mean_curvature
            traj_curvature_max = trajectory.max_curvature
            traj_path_ratio = trajectory.path_length_ratio
            traj_spectral_entropy = trajectory.trajectory_spectral_entropy
        except Exception as e:
            logger.debug(f"Trajectory computation failed: {e}")
            traj_curvature_mean = float("nan")
            traj_curvature_max = float("nan")
            traj_path_ratio = float("nan")
            traj_spectral_entropy = float("nan")
        
        # Intrinsic dimension (local complexity)
        try:
            id_result = self._intrinsic_dimension.compute(problem_act_2d)
            intrinsic_dim = id_result.intrinsic_dimension
        except Exception as e:
            logger.debug(f"ID computation failed: {e}")
            intrinsic_dim = float("nan")
        
        return ProblemProfile(
            problem_id=problem_id,
            prompt=prompt,
            cka_similarity=goldilocks.cka_similarity,
            barrier_height=goldilocks.barrier_height,
            fisher_mean=goldilocks.fisher_mean,
            goldilocks_score=goldilocks.quality_score,
            trajectory_curvature_mean=traj_curvature_mean,
            trajectory_curvature_max=traj_curvature_max,
            trajectory_path_length_ratio=traj_path_ratio,
            trajectory_spectral_entropy=traj_spectral_entropy,
            local_density=local_density,
            density_percentile=density_percentile,
            intrinsic_dimension=intrinsic_dim,
            layer_idx=self.layer_idx,
        )
    
    def _compute_percentiles(self, values: "Array") -> list[float]:
        """Compute percentile rank for each value."""
        values_list = [float(v) for v in values]
        n = len(values_list)
        
        # Sort and get ranks
        sorted_indices = sorted(range(n), key=lambda i: values_list[i])
        ranks = [0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank / (n - 1) if n > 1 else 0.5
        
        return ranks


__all__ = [
    "CurriculumProfiler",
    "CurriculumProfiles",
    "ProblemProfile",
]
