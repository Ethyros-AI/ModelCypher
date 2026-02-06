#!/usr/bin/env python3
"""RL Processing Geometry Experiment (Experiment 2).

Follow-up to the RL Geometric Attractor Experiment which REJECTED the
hypothesis that RL training collapses expansion_ratio to ~1.0. Instead,
RL amplifies geometric differentiation (2.4x higher mean expansion_ratio,
wider band, more cross-category variance).

This experiment tests four candidate theories for WHY:

A: Processing Depth Shift — RL moves heavy computation to later layers
B: Trajectory Complexity — RL models take longer, more winding paths
C: Layer-wise Differentiation — RL differentiates tasks at different layers
D: Late-Layer Convergence — Despite higher mid-layer variance, RL trajectories
   converge to a universal basin in final layers

Same model pair, same 30 prompts as experiment 1 for direct comparison.

Usage:
    # Smoke test (2 prompts per category, ~3-5 min)
    poetry run python scripts/rl_processing_geometry_experiment.py \\
        --smoke --output results/rl_processing/smoke/

    # Full experiment (~20-40 min)
    poetry run python scripts/rl_processing_geometry_experiment.py \\
        --output results/rl_processing/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry (same as experiment 1)
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

MODEL_REGISTRY = {
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
        "architecture": "qwen3",
        "training": "base",
    },
    "DeepSeek-R1-8B": {
        "path": f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        "architecture": "qwen3",
        "training": "rl_grpo",
    },
}


# =============================================================================
# Probe Categories (same 30 prompts as experiment 1)
# =============================================================================

PROBE_CATEGORIES = {
    "retrieval": [
        "The capital of France is",
        "Who wrote Romeo and Juliet?",
        "The chemical symbol for water is",
        "The largest planet in our solar system is",
        "The speed of light in a vacuum is approximately",
        "The first president of the United States was",
    ],
    "arithmetic": [
        "What is 347 + 528?",
        "What is 15 * 23?",
        "What is 1024 / 16?",
        "What is 99 - 37?",
        "What is 8 * 7 + 13?",
        "What is 256 + 384 - 100?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. At the second stop, 12 get off and 7 get on. How many people are on the bus now?",
        "A lily pad doubles in size every day. If it takes 48 days to cover the entire lake, on what day does it cover half the lake?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Describe a sunset over the mountains in one vivid sentence.",
        "Write a short poem about the passage of time.",
        "Describe the taste of your favorite food using only three words.",
        "Write a one-sentence story with a twist ending.",
        "Describe the sound of rain on a tin roof.",
    ],
    "code": [
        "Write a Python function that reverses a string.",
        "Write a Python function that checks if a number is prime.",
        "Write a Python function to compute the Fibonacci sequence up to n terms.",
        "Write a Python function that finds the maximum element in a list without using max().",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python one-liner to flatten a nested list.",
    ],
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptGeometry:
    """Full geometry for one prompt."""

    prompt: str
    category: str
    token_count: int

    # From experiment 1
    expansion_ratio: float
    peak_layer: int
    peak_dim: float
    final_dim: float
    smoothness: float
    id_trajectory: list[float]
    norm_trajectory: list[float]

    # Processing velocity (Theory A)
    velocity_trajectory: list[float]  # ||h_{l+1} - h_l|| / ||h_l||
    directional_change: list[float]  # angle between consecutive velocity vectors
    peak_velocity_layer: int
    mean_velocity: float

    # Trajectory complexity (Theory B)
    path_length_ratio: float
    mean_curvature: float
    max_curvature: float
    curvature_variance: float
    return_visit_count: int
    trajectory_effective_rank: float

    # Mean-pooled centroids for post-hoc CKA (Theories C, D, E)
    centroids: list[list[float]]  # [num_layers][hidden_dim]

    error: str | None = None


@dataclass
class ModelGeometry:
    """All prompts for one model."""

    model_name: str
    training_type: str
    architecture: str
    num_layers: int
    prompt_geometries: list[PromptGeometry]


@dataclass
class VelocityAnalysis:
    """Theory A: Processing velocity comparison."""

    base_mean_velocity_per_layer: list[float]
    rl_mean_velocity_per_layer: list[float]
    base_peak_velocity_layer: float  # mean peak layer across prompts
    rl_peak_velocity_layer: float
    cohens_d_per_layer: list[float]
    base_mean_overall_velocity: float
    rl_mean_overall_velocity: float


@dataclass
class ComplexityAnalysis:
    """Theory B: Trajectory complexity comparison."""

    base_path_length_ratios: list[float]
    rl_path_length_ratios: list[float]
    base_mean_curvatures: list[float]
    rl_mean_curvatures: list[float]
    base_return_visit_counts: list[int]
    rl_return_visit_counts: list[int]
    cohens_d_path_ratio: float
    cohens_d_curvature: float
    cohens_d_return_visits: float


@dataclass
class CrossCategoryCKAAnalysis:
    """Theory C: Cross-category CKA per layer."""

    base_mean_cross_cka_per_layer: list[float]
    rl_mean_cross_cka_per_layer: list[float]
    base_min_differentiation_layer: int  # layer of minimum cross-cat CKA
    rl_min_differentiation_layer: int
    base_within_cat_cka_per_layer: list[float]
    rl_within_cat_cka_per_layer: list[float]


@dataclass
class ConvergenceAnalysis:
    """Theory D: Late-layer convergence."""

    base_final_cross_cka: float
    rl_final_cross_cka: float
    base_peak_cross_cka: float
    rl_peak_cross_cka: float
    base_convergence_ratio: float  # final / peak
    rl_convergence_ratio: float
    base_final_cosine_similarity: float
    rl_final_cosine_similarity: float


@dataclass
class LayerSimilarityAnalysis:
    """Bonus: Layer-to-layer CKA similarity matrix."""

    base_similarity_matrix: list[list[float]]  # [num_layers, num_layers]
    rl_similarity_matrix: list[list[float]]


@dataclass
class GeometryComparison:
    """Cross-model analysis aggregating all theories."""

    velocity: VelocityAnalysis
    complexity: ComplexityAnalysis
    cross_category_cka: CrossCategoryCKAAnalysis
    convergence: ConvergenceAnalysis
    layer_similarity: LayerSimilarityAnalysis


# =============================================================================
# Statistics Helpers (copied from experiment 1)
# =============================================================================


def cohens_d_two_groups(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d between two independent groups (pooled std)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

    if pooled_var <= 0:
        return 0.0
    return (m1 - m2) / math.sqrt(pooled_var)


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(values: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_val + 1)
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _safe_mean(values: list[float]) -> float:
    """Mean of values, filtering NaN."""
    valid = [v for v in values if v == v]
    return sum(valid) / len(valid) if valid else float("nan")


def _safe_std(values: list[float]) -> float:
    """Std of values, filtering NaN."""
    valid = [v for v in values if v == v]
    if len(valid) < 2:
        return 0.0
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def _mean_trajectory(trajectories: list[list[float]]) -> list[float]:
    """Compute element-wise mean of multiple trajectories."""
    valid = [t for t in trajectories if t]
    if not valid:
        return []
    max_len = max(len(t) for t in valid)
    result = []
    for i in range(max_len):
        vals = [t[i] for t in valid if i < len(t) and t[i] == t[i]]
        result.append(sum(vals) / len(vals) if vals else float("nan"))
    return result


# =============================================================================
# Main Experiment Class
# =============================================================================


class RLProcessingGeometryExperiment:
    """RL Processing Geometry Experiment (Experiment 2)."""

    def __init__(
        self,
        output_dir: Path,
        seed: int = 42,
        smoke: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.seed = seed
        self.smoke = smoke
        self._rng = random.Random(seed)
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.num_layers = 0

    # ----- Model lifecycle -----

    def _load_model(self, model_name: str) -> None:
        """Load model and tokenizer, detect layer count."""
        from modelcypher.backends import initialize_default_backend

        if self.backend is None:
            self.backend = initialize_default_backend()

        model_info = MODEL_REGISTRY[model_name]
        model_path = model_info["path"]

        logger.info(f"Loading model: {model_name} from {model_path}")
        self.model, self.tokenizer = self.backend.load_model(model_path)

        base_model = getattr(self.model, "model", self.model)
        layers = getattr(base_model, "layers", None)
        self.num_layers = len(layers) if layers else 0

        logger.info(f"Model loaded: {self.num_layers} layers")

    def _unload_model(self) -> None:
        """Unload model and free GPU memory."""
        self.model = None
        self.tokenizer = None
        self.num_layers = 0
        gc.collect()
        gc.collect()
        if self.backend is not None:
            try:
                self.backend.clear_cache()
            except Exception:
                pass

    # ----- Prompt formatting -----

    def _format_prompt(self, raw_prompt: str) -> str:
        """Apply chat template if available."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return raw_prompt

    # ----- Per-prompt data collection -----

    def _collect_prompt_geometry(self, prompt: str, category: str) -> PromptGeometry:
        """Collect richer geometry for a single prompt.

        Steps:
        A. Collect trajectory activations
        B. Mean-pool per layer -> centroids (Python lists)
        C. Per-layer intrinsic dimension via batched TwoNN
        D. Processing velocity profile (Theory A)
        E. Trajectory complexity (Theory B)
        F. Norm trajectory
        G. Derive scalar metrics, convert to Python, free GPU arrays
        """
        b = self.backend
        formatted = self._format_prompt(prompt)

        # Step A: Collect trajectory activations
        from modelcypher.ports.activation_provider import get_activation_provider

        provider = get_activation_provider()
        trajectory = provider.collect_trajectory_batch(
            self.model, self.tokenizer, [formatted]
        )

        num_layers = len(trajectory.positions)
        token_count = trajectory.total_tokens
        layer_indices = sorted(trajectory.positions.keys())

        if token_count < 4:
            return PromptGeometry(
                prompt=prompt,
                category=category,
                token_count=token_count,
                expansion_ratio=float("nan"),
                peak_layer=-1,
                peak_dim=0.0,
                final_dim=0.0,
                smoothness=0.0,
                id_trajectory=[],
                norm_trajectory=[],
                velocity_trajectory=[],
                directional_change=[],
                peak_velocity_layer=-1,
                mean_velocity=0.0,
                path_length_ratio=float("nan"),
                mean_curvature=float("nan"),
                max_curvature=float("nan"),
                curvature_variance=float("nan"),
                return_visit_count=0,
                trajectory_effective_rank=0.0,
                centroids=[],
                error=f"Too few tokens ({token_count}), need >= 4",
            )

        # Step B: Mean-pool per layer -> centroids
        centroids_arrays = {}  # layer_idx -> Array[hidden_dim]
        centroids_lists = {}  # layer_idx -> list[float]
        for li in layer_indices:
            centroid = b.mean(trajectory.positions[li], axis=0)
            b.eval(centroid)
            centroids_arrays[li] = centroid
            centroids_lists[li] = b.tolist(centroid)

        # Step C: Per-layer intrinsic dimension via batched TwoNN
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
        )

        estimator = IntrinsicDimension(b)
        point_clouds = [trajectory.positions[i] for i in layer_indices]
        id_results = estimator.batch_compute(point_clouds)

        id_trajectory: list[float] = []
        for result in id_results:
            if result is not None:
                id_trajectory.append(result.intrinsic_dimension)
            else:
                id_trajectory.append(float("nan"))

        # Step D: Processing velocity profile (Theory A)
        velocity_trajectory: list[float] = []
        directional_change: list[float] = []

        for l_idx in range(len(layer_indices) - 1):
            li_curr = layer_indices[l_idx]
            li_next = layer_indices[l_idx + 1]
            c_curr = centroids_arrays[li_curr]
            c_next = centroids_arrays[li_next]

            diff = c_next - c_curr
            norm_diff = b.sqrt(b.sum(diff * diff))
            norm_curr = b.sqrt(b.sum(c_curr * c_curr))
            b.eval(norm_diff, norm_curr)

            norm_diff_val = float(b.to_scalar(norm_diff))
            norm_curr_val = float(b.to_scalar(norm_curr))

            if norm_curr_val > 1e-10:
                velocity_trajectory.append(norm_diff_val / norm_curr_val)
            else:
                velocity_trajectory.append(0.0)

        # Directional change: angle between consecutive velocity vectors
        for l_idx in range(len(layer_indices) - 2):
            li_0 = layer_indices[l_idx]
            li_1 = layer_indices[l_idx + 1]
            li_2 = layer_indices[l_idx + 2]

            v1 = centroids_arrays[li_1] - centroids_arrays[li_0]
            v2 = centroids_arrays[li_2] - centroids_arrays[li_1]

            norm1 = b.sqrt(b.sum(v1 * v1))
            norm2 = b.sqrt(b.sum(v2 * v2))
            b.eval(norm1, norm2)
            n1 = float(b.to_scalar(norm1))
            n2 = float(b.to_scalar(norm2))

            if n1 > 1e-10 and n2 > 1e-10:
                dot = b.sum(v1 * v2)
                b.eval(dot)
                cos_angle = float(b.to_scalar(dot)) / (n1 * n2)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                directional_change.append(math.acos(cos_angle))
            else:
                directional_change.append(0.0)

        if velocity_trajectory:
            peak_velocity_layer = velocity_trajectory.index(max(velocity_trajectory))
            mean_velocity = sum(velocity_trajectory) / len(velocity_trajectory)
        else:
            peak_velocity_layer = -1
            mean_velocity = 0.0

        # Step E: Trajectory complexity (Theory B)
        from modelcypher.core.domain.geometry.trajectory_complexity import (
            TrajectoryComplexity,
        )

        tc = TrajectoryComplexity(b)
        tc_result = tc.compute(centroids_arrays)

        # Step F: Norm trajectory
        norm_trajectory: list[float] = []
        for li in layer_indices:
            pos = trajectory.positions[li]
            norm_val = b.sqrt(b.sum(pos * pos, axis=1))
            mean_norm = b.mean(norm_val)
            b.eval(mean_norm)
            norm_trajectory.append(float(b.to_scalar(mean_norm)))

        # Step G: Derive scalar metrics (same as experiment 1)
        valid_ids = [d for d in id_trajectory if d == d and d > 0]

        if len(valid_ids) >= 2:
            peak_dim = max(valid_ids)
            peak_layer = id_trajectory.index(peak_dim)
            final_dim = valid_ids[-1] if valid_ids[-1] == valid_ids[-1] else valid_ids[-2]
            expansion_ratio = peak_dim / final_dim if final_dim > 0 else float("nan")
            smoothness = spearman_correlation(
                list(range(len(id_trajectory))), id_trajectory
            )
        else:
            peak_dim = 0.0
            peak_layer = -1
            final_dim = 0.0
            expansion_ratio = float("nan")
            smoothness = 0.0

        # Build centroids as list[list[float]] ordered by layer index
        centroids_output = [centroids_lists[li] for li in layer_indices]

        # Free backend arrays
        del trajectory, point_clouds, centroids_arrays
        try:
            b.clear_cache()
        except Exception:
            pass

        return PromptGeometry(
            prompt=prompt,
            category=category,
            token_count=token_count,
            expansion_ratio=expansion_ratio,
            peak_layer=peak_layer,
            peak_dim=peak_dim,
            final_dim=final_dim,
            smoothness=smoothness,
            id_trajectory=id_trajectory,
            norm_trajectory=norm_trajectory,
            velocity_trajectory=velocity_trajectory,
            directional_change=directional_change,
            peak_velocity_layer=peak_velocity_layer,
            mean_velocity=mean_velocity,
            path_length_ratio=tc_result.path_length_ratio,
            mean_curvature=tc_result.mean_curvature,
            max_curvature=tc_result.max_curvature,
            curvature_variance=tc_result.curvature_variance,
            return_visit_count=tc_result.return_visit_count,
            trajectory_effective_rank=tc_result.trajectory_effective_rank,
            centroids=centroids_output,
        )

    # ----- Model-level collection -----

    def _collect_model_geometry(self, model_name: str) -> list[PromptGeometry]:
        """Collect geometry for all prompts for one model."""
        results: list[PromptGeometry] = []
        prompt_count = 0

        for category, prompts in PROBE_CATEGORIES.items():
            if self.smoke:
                prompts = prompts[:2]

            for i, prompt in enumerate(prompts):
                prompt_count += 1
                logger.info(
                    f"  [{prompt_count}] {category}/{i}: "
                    f"{prompt[:60]}..."
                )

                try:
                    result = self._collect_prompt_geometry(prompt, category)
                    results.append(result)

                    if result.error:
                        logger.warning(f"    Error: {result.error}")
                    else:
                        logger.info(
                            f"    tokens={result.token_count}, "
                            f"ER={result.expansion_ratio:.3f}, "
                            f"vel={result.mean_velocity:.4f}, "
                            f"PLR={result.path_length_ratio:.3f}"
                        )
                except Exception as e:
                    logger.warning(f"    FAILED: {e}")
                    results.append(
                        PromptGeometry(
                            prompt=prompt,
                            category=category,
                            token_count=0,
                            expansion_ratio=float("nan"),
                            peak_layer=-1,
                            peak_dim=0.0,
                            final_dim=0.0,
                            smoothness=0.0,
                            id_trajectory=[],
                            norm_trajectory=[],
                            velocity_trajectory=[],
                            directional_change=[],
                            peak_velocity_layer=-1,
                            mean_velocity=0.0,
                            path_length_ratio=float("nan"),
                            mean_curvature=float("nan"),
                            max_curvature=float("nan"),
                            curvature_variance=float("nan"),
                            return_visit_count=0,
                            trajectory_effective_rank=0.0,
                            centroids=[],
                            error=str(e),
                        )
                    )

                # Memory management
                if prompt_count % 5 == 0 and self.backend is not None:
                    try:
                        self.backend.clear_cache()
                    except Exception:
                        pass

        return results

    # ----- Post-collection Analyses (CPU-only, no model loaded) -----

    def _analyze_processing_velocity(
        self, base: ModelGeometry, rl: ModelGeometry
    ) -> VelocityAnalysis:
        """Theory A: Processing velocity profile comparison."""
        num_layers = max(base.num_layers, rl.num_layers)

        def _velocity_per_layer(geom: ModelGeometry) -> list[float]:
            trajs = [
                g.velocity_trajectory
                for g in geom.prompt_geometries
                if g.error is None and g.velocity_trajectory
            ]
            return _mean_trajectory(trajs)

        base_vel = _velocity_per_layer(base)
        rl_vel = _velocity_per_layer(rl)

        # Cohen's d at each layer
        cohens_d_per_layer: list[float] = []
        max_len = min(len(base_vel), len(rl_vel))
        for l_idx in range(max_len):
            base_vals = [
                g.velocity_trajectory[l_idx]
                for g in base.prompt_geometries
                if g.error is None
                and len(g.velocity_trajectory) > l_idx
                and g.velocity_trajectory[l_idx] == g.velocity_trajectory[l_idx]
            ]
            rl_vals = [
                g.velocity_trajectory[l_idx]
                for g in rl.prompt_geometries
                if g.error is None
                and len(g.velocity_trajectory) > l_idx
                and g.velocity_trajectory[l_idx] == g.velocity_trajectory[l_idx]
            ]
            cohens_d_per_layer.append(cohens_d_two_groups(base_vals, rl_vals))

        base_peaks = [
            g.peak_velocity_layer
            for g in base.prompt_geometries
            if g.error is None and g.peak_velocity_layer >= 0
        ]
        rl_peaks = [
            g.peak_velocity_layer
            for g in rl.prompt_geometries
            if g.error is None and g.peak_velocity_layer >= 0
        ]

        base_vels = [
            g.mean_velocity
            for g in base.prompt_geometries
            if g.error is None and g.mean_velocity == g.mean_velocity
        ]
        rl_vels = [
            g.mean_velocity
            for g in rl.prompt_geometries
            if g.error is None and g.mean_velocity == g.mean_velocity
        ]

        return VelocityAnalysis(
            base_mean_velocity_per_layer=base_vel,
            rl_mean_velocity_per_layer=rl_vel,
            base_peak_velocity_layer=_safe_mean(
                [float(p) for p in base_peaks]
            ),
            rl_peak_velocity_layer=_safe_mean(
                [float(p) for p in rl_peaks]
            ),
            cohens_d_per_layer=cohens_d_per_layer,
            base_mean_overall_velocity=_safe_mean(base_vels),
            rl_mean_overall_velocity=_safe_mean(rl_vels),
        )

    def _analyze_trajectory_complexity(
        self, base: ModelGeometry, rl: ModelGeometry
    ) -> ComplexityAnalysis:
        """Theory B: Trajectory complexity comparison."""
        base_plr = [
            g.path_length_ratio
            for g in base.prompt_geometries
            if g.error is None and g.path_length_ratio == g.path_length_ratio
        ]
        rl_plr = [
            g.path_length_ratio
            for g in rl.prompt_geometries
            if g.error is None and g.path_length_ratio == g.path_length_ratio
        ]
        base_curv = [
            g.mean_curvature
            for g in base.prompt_geometries
            if g.error is None and g.mean_curvature == g.mean_curvature
        ]
        rl_curv = [
            g.mean_curvature
            for g in rl.prompt_geometries
            if g.error is None and g.mean_curvature == g.mean_curvature
        ]
        base_rv = [
            g.return_visit_count
            for g in base.prompt_geometries
            if g.error is None
        ]
        rl_rv = [
            g.return_visit_count
            for g in rl.prompt_geometries
            if g.error is None
        ]

        return ComplexityAnalysis(
            base_path_length_ratios=base_plr,
            rl_path_length_ratios=rl_plr,
            base_mean_curvatures=base_curv,
            rl_mean_curvatures=rl_curv,
            base_return_visit_counts=base_rv,
            rl_return_visit_counts=rl_rv,
            cohens_d_path_ratio=cohens_d_two_groups(base_plr, rl_plr),
            cohens_d_curvature=cohens_d_two_groups(base_curv, rl_curv),
            cohens_d_return_visits=cohens_d_two_groups(
                [float(x) for x in base_rv], [float(x) for x in rl_rv]
            ),
        )

    def _analyze_cross_category_cka(
        self, base: ModelGeometry, rl: ModelGeometry
    ) -> CrossCategoryCKAAnalysis:
        """Theory C: Cross-category CKA per layer.

        At each layer, compute mean CKA between all pairs of categories.
        Low CKA = model differentiates tasks. High CKA = model sees tasks as similar.
        """
        from modelcypher.backends import initialize_default_backend
        from modelcypher.core.domain.geometry.cka import (
            compute_linear_cka_from_activations,
        )

        b = initialize_default_backend()

        categories = list(PROBE_CATEGORIES.keys())
        category_pairs = list(combinations(categories, 2))

        def _compute_for_model(
            geom: ModelGeometry,
        ) -> tuple[list[float], list[float]]:
            num_layers = geom.num_layers

            # Organize centroids by category and layer
            cat_centroids: dict[str, dict[int, list[list[float]]]] = {
                cat: {} for cat in categories
            }
            for pg in geom.prompt_geometries:
                if pg.error is None and pg.centroids:
                    for li in range(min(num_layers, len(pg.centroids))):
                        if li not in cat_centroids[pg.category]:
                            cat_centroids[pg.category][li] = []
                        cat_centroids[pg.category][li].append(pg.centroids[li])

            mean_cross_cka: list[float] = []
            mean_within_cka: list[float] = []

            for li in range(num_layers):
                # Cross-category CKA
                pair_ckas = []
                for cat_a, cat_b in category_pairs:
                    centroids_a = cat_centroids.get(cat_a, {}).get(li, [])
                    centroids_b = cat_centroids.get(cat_b, {}).get(li, [])
                    if len(centroids_a) >= 2 and len(centroids_b) >= 2:
                        arr_a = b.array(centroids_a)
                        arr_b = b.array(centroids_b)
                        cka = compute_linear_cka_from_activations(arr_a, arr_b, b)
                        pair_ckas.append(cka)
                mean_cross_cka.append(
                    _safe_mean(pair_ckas) if pair_ckas else float("nan")
                )

                # Within-category CKA (self-similarity)
                within_ckas = []
                for cat in categories:
                    centroids_cat = cat_centroids.get(cat, {}).get(li, [])
                    if len(centroids_cat) >= 2:
                        arr = b.array(centroids_cat)
                        # Split in half for within-group CKA
                        half = len(centroids_cat) // 2
                        if half >= 2:
                            arr_a = b.array(centroids_cat[:half])
                            arr_b = b.array(centroids_cat[half : half * 2])
                            cka = compute_linear_cka_from_activations(arr_a, arr_b, b)
                            within_ckas.append(cka)
                mean_within_cka.append(
                    _safe_mean(within_ckas) if within_ckas else float("nan")
                )

            return mean_cross_cka, mean_within_cka

        logger.info("  Computing cross-category CKA for base model...")
        base_cross, base_within = _compute_for_model(base)
        logger.info("  Computing cross-category CKA for RL model...")
        rl_cross, rl_within = _compute_for_model(rl)

        # Find layer of minimum cross-category CKA (max differentiation)
        def _argmin_valid(vals: list[float]) -> int:
            valid = [(v, i) for i, v in enumerate(vals) if v == v]
            return min(valid, key=lambda x: x[0])[1] if valid else -1

        return CrossCategoryCKAAnalysis(
            base_mean_cross_cka_per_layer=base_cross,
            rl_mean_cross_cka_per_layer=rl_cross,
            base_min_differentiation_layer=_argmin_valid(base_cross),
            rl_min_differentiation_layer=_argmin_valid(rl_cross),
            base_within_cat_cka_per_layer=base_within,
            rl_within_cat_cka_per_layer=rl_within,
        )

    def _analyze_late_layer_convergence(
        self, base: ModelGeometry, rl: ModelGeometry
    ) -> ConvergenceAnalysis:
        """Theory D: Late-layer convergence analysis.

        Check if RL model's cross-category CKA converges in final layers,
        suggesting an attractor basin.
        """
        from modelcypher.backends import initialize_default_backend
        from modelcypher.core.domain.geometry.cka import (
            compute_linear_cka_from_activations,
        )

        b = initialize_default_backend()

        def _compute_convergence(geom: ModelGeometry) -> tuple[float, float, float]:
            """Returns (final_cross_cka, peak_cross_cka, final_cosine_sim)."""
            num_layers = geom.num_layers
            valid_geoms = [g for g in geom.prompt_geometries if g.error is None and g.centroids]

            if not valid_geoms or num_layers < 2:
                return float("nan"), float("nan"), float("nan")

            # Get all centroids at final layer
            final_layer = num_layers - 1
            final_centroids = []
            for g in valid_geoms:
                if len(g.centroids) > final_layer:
                    final_centroids.append(g.centroids[final_layer])

            if len(final_centroids) < 4:
                return float("nan"), float("nan"), float("nan")

            # Pairwise cosine similarity at final layer
            cos_sims = []
            for i in range(len(final_centroids)):
                for j in range(i + 1, len(final_centroids)):
                    a = final_centroids[i]
                    b_vec = final_centroids[j]
                    dot_val = sum(x * y for x, y in zip(a, b_vec))
                    norm_a = math.sqrt(sum(x * x for x in a))
                    norm_b = math.sqrt(sum(x * x for x in b_vec))
                    if norm_a > 1e-10 and norm_b > 1e-10:
                        cos_sims.append(dot_val / (norm_a * norm_b))

            final_cos_sim = _safe_mean(cos_sims)

            # Cross-category CKA at final layer vs peak
            categories = list(PROBE_CATEGORIES.keys())
            category_pairs = list(combinations(categories, 2))

            cat_centroids_by_layer: dict[str, dict[int, list[list[float]]]] = {
                cat: {} for cat in categories
            }
            for g in valid_geoms:
                for li in range(min(num_layers, len(g.centroids))):
                    if li not in cat_centroids_by_layer[g.category]:
                        cat_centroids_by_layer[g.category][li] = []
                    cat_centroids_by_layer[g.category][li].append(g.centroids[li])

            cross_cka_per_layer = []
            for li in range(num_layers):
                pair_ckas = []
                for cat_a, cat_b in category_pairs:
                    ca = cat_centroids_by_layer.get(cat_a, {}).get(li, [])
                    cb = cat_centroids_by_layer.get(cat_b, {}).get(li, [])
                    if len(ca) >= 2 and len(cb) >= 2:
                        arr_a = b.array(ca)
                        arr_b = b.array(cb)
                        cka = compute_linear_cka_from_activations(arr_a, arr_b, b)
                        pair_ckas.append(cka)
                cross_cka_per_layer.append(
                    _safe_mean(pair_ckas) if pair_ckas else float("nan")
                )

            valid_cka = [v for v in cross_cka_per_layer if v == v]
            final_cross = cross_cka_per_layer[-1] if cross_cka_per_layer else float("nan")
            peak_cross = max(valid_cka) if valid_cka else float("nan")

            return final_cross, peak_cross, final_cos_sim

        logger.info("  Computing late-layer convergence for base model...")
        base_final, base_peak, base_cos = _compute_convergence(base)
        logger.info("  Computing late-layer convergence for RL model...")
        rl_final, rl_peak, rl_cos = _compute_convergence(rl)

        def _ratio(final: float, peak: float) -> float:
            if peak == peak and peak > 0 and final == final:
                return final / peak
            return float("nan")

        return ConvergenceAnalysis(
            base_final_cross_cka=base_final,
            rl_final_cross_cka=rl_final,
            base_peak_cross_cka=base_peak,
            rl_peak_cross_cka=rl_peak,
            base_convergence_ratio=_ratio(base_final, base_peak),
            rl_convergence_ratio=_ratio(rl_final, rl_peak),
            base_final_cosine_similarity=base_cos,
            rl_final_cosine_similarity=rl_cos,
        )

    def _analyze_layer_similarity(
        self, base: ModelGeometry, rl: ModelGeometry
    ) -> LayerSimilarityAnalysis:
        """Bonus: Layer-to-layer CKA similarity matrix.

        For each model, build NxN CKA matrix where CKA(layer_i, layer_j)
        measures processing block structure.
        """
        from modelcypher.backends import initialize_default_backend
        from modelcypher.core.domain.geometry.cka import (
            compute_linear_cka_from_activations,
        )

        b = initialize_default_backend()

        def _compute_matrix(geom: ModelGeometry) -> list[list[float]]:
            num_layers = geom.num_layers
            valid_geoms = [g for g in geom.prompt_geometries if g.error is None and g.centroids]

            if not valid_geoms or num_layers < 2:
                return []

            # Build [num_prompts, hidden_dim] matrix per layer
            layer_matrices: dict[int, list[list[float]]] = {}
            for li in range(num_layers):
                layer_matrices[li] = []
                for g in valid_geoms:
                    if len(g.centroids) > li:
                        layer_matrices[li].append(g.centroids[li])

            # Build CKA matrix
            matrix = [[0.0] * num_layers for _ in range(num_layers)]
            for i in range(num_layers):
                matrix[i][i] = 1.0  # Self-similarity is 1.0
                for j in range(i + 1, num_layers):
                    ci = layer_matrices.get(i, [])
                    cj = layer_matrices.get(j, [])
                    if len(ci) >= 2 and len(cj) >= 2:
                        arr_i = b.array(ci)
                        arr_j = b.array(cj)
                        cka = compute_linear_cka_from_activations(arr_i, arr_j, b)
                        matrix[i][j] = cka
                        matrix[j][i] = cka

            return matrix

        logger.info("  Computing layer similarity matrix for base model...")
        base_matrix = _compute_matrix(base)
        logger.info("  Computing layer similarity matrix for RL model...")
        rl_matrix = _compute_matrix(rl)

        return LayerSimilarityAnalysis(
            base_similarity_matrix=base_matrix,
            rl_similarity_matrix=rl_matrix,
        )

    # ----- Main experiment loop -----

    def run(self) -> None:
        """Run full RL processing geometry experiment."""
        logger.info("=" * 70)
        logger.info("RL PROCESSING GEOMETRY EXPERIMENT (Experiment 2)")
        logger.info("Testing WHY RL amplifies geometric differentiation")
        logger.info(f"Models: {list(MODEL_REGISTRY.keys())}")
        logger.info(f"Categories: {list(PROBE_CATEGORIES.keys())}")
        prompts_per_cat = 2 if self.smoke else 6
        total_prompts = len(PROBE_CATEGORIES) * prompts_per_cat
        logger.info(f"Prompts: {total_prompts} ({'smoke' if self.smoke else 'full'})")
        logger.info(f"Seed: {self.seed}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 70)

        # Verify volume
        if not os.path.exists(MODELS_BASE):
            logger.error(f"Model volume not accessible: {MODELS_BASE}")
            logger.error("Is the external drive mounted?")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        model_geometries: dict[str, ModelGeometry] = {}

        # Phase 1 & 2: Collect data per model
        for model_name in MODEL_REGISTRY:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"MODEL: {model_name}")
            logger.info(f"{'=' * 60}")

            t0 = time.time()

            try:
                self._load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            try:
                prompt_geoms = self._collect_model_geometry(model_name)
            except Exception as e:
                logger.error(f"Failed to collect geometry for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                self._unload_model()
                continue

            model_info = MODEL_REGISTRY[model_name]
            geom = ModelGeometry(
                model_name=model_name,
                training_type=model_info["training"],
                architecture=model_info["architecture"],
                num_layers=self.num_layers,
                prompt_geometries=prompt_geoms,
            )
            model_geometries[model_name] = geom

            # Save intermediate
            self._save_intermediate(model_name, prompt_geoms)

            elapsed = time.time() - t0
            valid = [g for g in prompt_geoms if g.error is None]
            logger.info(
                f"\n{model_name} complete: {elapsed:.1f}s, "
                f"{len(valid)}/{len(prompt_geoms)} prompts succeeded"
            )

            self._unload_model()

        # Phase 3: Post-collection analysis (no model loaded)
        comparison = None
        if "Qwen3-8B" in model_geometries and "DeepSeek-R1-8B" in model_geometries:
            base = model_geometries["Qwen3-8B"]
            rl = model_geometries["DeepSeek-R1-8B"]

            logger.info(f"\n{'=' * 60}")
            logger.info("POST-COLLECTION ANALYSIS")
            logger.info(f"{'=' * 60}")

            logger.info("\nAnalysis A: Processing Velocity...")
            velocity = self._analyze_processing_velocity(base, rl)

            logger.info("\nAnalysis B: Trajectory Complexity...")
            complexity = self._analyze_trajectory_complexity(base, rl)

            logger.info("\nAnalysis C: Cross-Category CKA...")
            cross_cat = self._analyze_cross_category_cka(base, rl)

            logger.info("\nAnalysis D: Late-Layer Convergence...")
            convergence = self._analyze_late_layer_convergence(base, rl)

            logger.info("\nAnalysis E: Layer Similarity Matrix...")
            layer_sim = self._analyze_layer_similarity(base, rl)

            comparison = GeometryComparison(
                velocity=velocity,
                complexity=complexity,
                cross_category_cka=cross_cat,
                convergence=convergence,
                layer_similarity=layer_sim,
            )

            logger.info("\n--- Quick Summary ---")
            logger.info(
                f"  Velocity: base peak=L{velocity.base_peak_velocity_layer:.0f}, "
                f"RL peak=L{velocity.rl_peak_velocity_layer:.0f}"
            )
            logger.info(
                f"  Complexity PLR: base={_safe_mean(complexity.base_path_length_ratios):.3f}, "
                f"RL={_safe_mean(complexity.rl_path_length_ratios):.3f} "
                f"(d={complexity.cohens_d_path_ratio:+.3f})"
            )
            logger.info(
                f"  Convergence ratio: base={convergence.base_convergence_ratio:.3f}, "
                f"RL={convergence.rl_convergence_ratio:.3f}"
            )

        # Save results and report
        self._save_results(model_geometries, comparison)
        self._generate_report(model_geometries, comparison)

        logger.info("\nExperiment complete.")

    # ----- Save / Load -----

    def _save_intermediate(
        self, model_name: str, results: list[PromptGeometry]
    ) -> None:
        """Save per-model results for crash recovery."""
        raw_dir = self.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{model_name}_geometry.jsonl"

        def _sf(v: float) -> float | None:
            return v if v == v else None

        with open(path, "w") as f:
            for r in results:
                record = {
                    "prompt": r.prompt[:200],
                    "category": r.category,
                    "token_count": r.token_count,
                    "expansion_ratio": _sf(r.expansion_ratio),
                    "peak_layer": r.peak_layer,
                    "peak_dim": r.peak_dim,
                    "final_dim": r.final_dim,
                    "smoothness": r.smoothness,
                    "id_trajectory": [_sf(v) for v in r.id_trajectory],
                    "norm_trajectory": r.norm_trajectory,
                    "velocity_trajectory": r.velocity_trajectory,
                    "directional_change": r.directional_change,
                    "peak_velocity_layer": r.peak_velocity_layer,
                    "mean_velocity": r.mean_velocity,
                    "path_length_ratio": _sf(r.path_length_ratio),
                    "mean_curvature": _sf(r.mean_curvature),
                    "max_curvature": _sf(r.max_curvature),
                    "curvature_variance": _sf(r.curvature_variance),
                    "return_visit_count": r.return_visit_count,
                    "trajectory_effective_rank": r.trajectory_effective_rank,
                    # Omit centroids from intermediate (too large for JSONL)
                    "error": r.error,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"  Saved {len(results)} prompts to {path}")

    def _save_results(
        self,
        model_geometries: dict[str, ModelGeometry],
        comparison: GeometryComparison | None,
    ) -> None:
        """Save full results as JSON."""
        def _sf(v: float) -> float | None:
            return v if v == v else None

        serializable: dict[str, Any] = {
            "experiment": {
                "name": "rl_processing_geometry",
                "version": 2,
                "seed": self.seed,
                "smoke": self.smoke,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_categories": len(PROBE_CATEGORIES),
                "prompts_per_category": 2 if self.smoke else 6,
            },
            "models": {},
        }

        for name, mg in model_geometries.items():
            per_cat: dict[str, dict] = {}
            for cat in PROBE_CATEGORIES:
                cat_geoms = [
                    g for g in mg.prompt_geometries
                    if g.category == cat and g.error is None
                ]
                ers = [g.expansion_ratio for g in cat_geoms if g.expansion_ratio == g.expansion_ratio]
                plrs = [g.path_length_ratio for g in cat_geoms if g.path_length_ratio == g.path_length_ratio]
                per_cat[cat] = {
                    "n_probes": len(cat_geoms),
                    "mean_expansion_ratio": _sf(_safe_mean(ers)),
                    "mean_path_length_ratio": _sf(_safe_mean(plrs)),
                    "mean_velocity": _sf(
                        _safe_mean([g.mean_velocity for g in cat_geoms])
                    ),
                }

            all_ers = [
                g.expansion_ratio for g in mg.prompt_geometries
                if g.error is None and g.expansion_ratio == g.expansion_ratio
            ]

            serializable["models"][name] = {
                "training_type": mg.training_type,
                "architecture": mg.architecture,
                "num_layers": mg.num_layers,
                "overall_expansion_ratio_mean": _sf(_safe_mean(all_ers)),
                "categories": per_cat,
            }

        if comparison is not None:
            vel = comparison.velocity
            comp = comparison.complexity
            cka = comparison.cross_category_cka
            conv = comparison.convergence

            serializable["comparison"] = {
                "velocity": {
                    "base_mean_velocity_per_layer": [_sf(v) for v in vel.base_mean_velocity_per_layer],
                    "rl_mean_velocity_per_layer": [_sf(v) for v in vel.rl_mean_velocity_per_layer],
                    "base_peak_velocity_layer": _sf(vel.base_peak_velocity_layer),
                    "rl_peak_velocity_layer": _sf(vel.rl_peak_velocity_layer),
                    "cohens_d_per_layer": [_sf(v) for v in vel.cohens_d_per_layer],
                    "base_mean_overall_velocity": _sf(vel.base_mean_overall_velocity),
                    "rl_mean_overall_velocity": _sf(vel.rl_mean_overall_velocity),
                },
                "complexity": {
                    "cohens_d_path_ratio": _sf(comp.cohens_d_path_ratio),
                    "cohens_d_curvature": _sf(comp.cohens_d_curvature),
                    "cohens_d_return_visits": _sf(comp.cohens_d_return_visits),
                    "base_mean_path_length_ratio": _sf(_safe_mean(comp.base_path_length_ratios)),
                    "rl_mean_path_length_ratio": _sf(_safe_mean(comp.rl_path_length_ratios)),
                    "base_mean_curvature": _sf(_safe_mean(comp.base_mean_curvatures)),
                    "rl_mean_curvature": _sf(_safe_mean(comp.rl_mean_curvatures)),
                    "base_mean_return_visits": _sf(_safe_mean([float(x) for x in comp.base_return_visit_counts])),
                    "rl_mean_return_visits": _sf(_safe_mean([float(x) for x in comp.rl_return_visit_counts])),
                },
                "cross_category_cka": {
                    "base_mean_cross_cka_per_layer": [_sf(v) for v in cka.base_mean_cross_cka_per_layer],
                    "rl_mean_cross_cka_per_layer": [_sf(v) for v in cka.rl_mean_cross_cka_per_layer],
                    "base_min_differentiation_layer": cka.base_min_differentiation_layer,
                    "rl_min_differentiation_layer": cka.rl_min_differentiation_layer,
                    "base_within_cat_cka_per_layer": [_sf(v) for v in cka.base_within_cat_cka_per_layer],
                    "rl_within_cat_cka_per_layer": [_sf(v) for v in cka.rl_within_cat_cka_per_layer],
                },
                "convergence": {
                    "base_final_cross_cka": _sf(conv.base_final_cross_cka),
                    "rl_final_cross_cka": _sf(conv.rl_final_cross_cka),
                    "base_peak_cross_cka": _sf(conv.base_peak_cross_cka),
                    "rl_peak_cross_cka": _sf(conv.rl_peak_cross_cka),
                    "base_convergence_ratio": _sf(conv.base_convergence_ratio),
                    "rl_convergence_ratio": _sf(conv.rl_convergence_ratio),
                    "base_final_cosine_similarity": _sf(conv.base_final_cosine_similarity),
                    "rl_final_cosine_similarity": _sf(conv.rl_final_cosine_similarity),
                },
                "layer_similarity": {
                    "base_matrix": comparison.layer_similarity.base_similarity_matrix,
                    "rl_matrix": comparison.layer_similarity.rl_similarity_matrix,
                },
            }

        path = self.output_dir / "raw_results.json"
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {path}")

    # ----- Report generation -----

    def _generate_report(
        self,
        model_geometries: dict[str, ModelGeometry],
        comparison: GeometryComparison | None,
    ) -> None:
        """Generate PROCESSING_REPORT.md."""
        lines: list[str] = []
        w = lines.append

        w("# RL Processing Geometry Report")
        w("")
        w(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
        w(f"**Seed**: {self.seed}")
        w(f"**Mode**: {'smoke test' if self.smoke else 'full experiment'}")
        w(f"**Prompts per category**: {2 if self.smoke else 6}")
        w("")

        w("## Context")
        w("")
        w("Experiment 1 (RL Geometric Attractor) REJECTED the hypothesis that RL")
        w("training collapses expansion_ratio. Instead, RL **amplifies** geometric")
        w("differentiation (2.4x higher mean ER, wider band, more variance).")
        w("")
        w("This experiment tests four theories for WHY:")
        w("- **A**: Processing depth shift (heavy computation moves to later layers)")
        w("- **B**: Trajectory complexity (RL takes longer/more winding paths)")
        w("- **C**: Layer-wise differentiation (RL differentiates tasks at different layers)")
        w("- **D**: Late-layer convergence (despite mid-layer variance, RL converges in final layers)")
        w("")

        if comparison is None:
            w("**No comparison available** (missing one or both models).")
            w("")
        else:
            vel = comparison.velocity
            comp = comparison.complexity
            cka = comparison.cross_category_cka
            conv = comparison.convergence

            # Summary table
            w("## Summary of Findings")
            w("")
            w("| Theory | Key Evidence |")
            w("|--------|-------------|")

            # Theory A
            w(
                f"| A: Processing depth shift | "
                f"Peak velocity: base=L{vel.base_peak_velocity_layer:.0f}, "
                f"RL=L{vel.rl_peak_velocity_layer:.0f} |"
            )

            # Theory B
            w(
                f"| B: Trajectory complexity | "
                f"PLR: base={_safe_mean(comp.base_path_length_ratios):.3f}, "
                f"RL={_safe_mean(comp.rl_path_length_ratios):.3f} "
                f"(d={comp.cohens_d_path_ratio:+.3f}) |"
            )

            # Theory C
            w(
                f"| C: Layer-wise differentiation | "
                f"Min CKA layer: base=L{cka.base_min_differentiation_layer}, "
                f"RL=L{cka.rl_min_differentiation_layer} |"
            )

            # Theory D
            w(
                f"| D: Late-layer convergence | "
                f"Convergence ratio: base={conv.base_convergence_ratio:.3f}, "
                f"RL={conv.rl_convergence_ratio:.3f} |"
            )
            w("")

            # Analysis A: Processing Velocity
            w("## Analysis A: Processing Velocity")
            w("")
            w(f"- Base model mean overall velocity: **{vel.base_mean_overall_velocity:.4f}**")
            w(f"- RL model mean overall velocity: **{vel.rl_mean_overall_velocity:.4f}**")
            w(f"- Base peak velocity layer: **L{vel.base_peak_velocity_layer:.0f}**")
            w(f"- RL peak velocity layer: **L{vel.rl_peak_velocity_layer:.0f}**")
            w("")

            # Layer-by-layer velocity (abbreviated)
            if vel.cohens_d_per_layer:
                w("### Layer-by-Layer Cohen's d (velocity)")
                w("")
                w("| Layer | Base Vel | RL Vel | Cohen's d |")
                w("|-------|----------|--------|-----------|")
                for l_idx in range(min(len(vel.base_mean_velocity_per_layer), len(vel.rl_mean_velocity_per_layer))):
                    bv = vel.base_mean_velocity_per_layer[l_idx]
                    rv = vel.rl_mean_velocity_per_layer[l_idx]
                    d = vel.cohens_d_per_layer[l_idx] if l_idx < len(vel.cohens_d_per_layer) else 0.0
                    bv_s = f"{bv:.4f}" if bv == bv else "NaN"
                    rv_s = f"{rv:.4f}" if rv == rv else "NaN"
                    d_s = f"{d:+.3f}" if d == d else "NaN"
                    w(f"| {l_idx} | {bv_s} | {rv_s} | {d_s} |")
                w("")

            # Analysis B: Trajectory Complexity
            w("## Analysis B: Trajectory Complexity")
            w("")
            w("| Metric | Base | RL | Cohen's d |")
            w("|--------|------|-----|-----------|")
            w(
                f"| Path length ratio | "
                f"{_safe_mean(comp.base_path_length_ratios):.3f} | "
                f"{_safe_mean(comp.rl_path_length_ratios):.3f} | "
                f"{comp.cohens_d_path_ratio:+.3f} |"
            )
            w(
                f"| Mean curvature | "
                f"{_safe_mean(comp.base_mean_curvatures):.4f} | "
                f"{_safe_mean(comp.rl_mean_curvatures):.4f} | "
                f"{comp.cohens_d_curvature:+.3f} |"
            )
            w(
                f"| Return visit count | "
                f"{_safe_mean([float(x) for x in comp.base_return_visit_counts]):.1f} | "
                f"{_safe_mean([float(x) for x in comp.rl_return_visit_counts]):.1f} | "
                f"{comp.cohens_d_return_visits:+.3f} |"
            )
            w("")

            # Analysis C: Cross-Category CKA
            w("## Analysis C: Cross-Category Differentiation")
            w("")
            w(f"- Base min differentiation layer: **L{cka.base_min_differentiation_layer}**")
            w(f"- RL min differentiation layer: **L{cka.rl_min_differentiation_layer}**")
            w("")

            if cka.base_mean_cross_cka_per_layer and cka.rl_mean_cross_cka_per_layer:
                w("### Cross-Category CKA by Layer")
                w("")
                w("| Layer | Base Cross-CKA | RL Cross-CKA | Base Within-CKA | RL Within-CKA |")
                w("|-------|----------------|--------------|-----------------|---------------|")
                max_len = min(
                    len(cka.base_mean_cross_cka_per_layer),
                    len(cka.rl_mean_cross_cka_per_layer),
                )
                for l_idx in range(max_len):
                    bc = cka.base_mean_cross_cka_per_layer[l_idx]
                    rc = cka.rl_mean_cross_cka_per_layer[l_idx]
                    bw = cka.base_within_cat_cka_per_layer[l_idx] if l_idx < len(cka.base_within_cat_cka_per_layer) else float("nan")
                    rw = cka.rl_within_cat_cka_per_layer[l_idx] if l_idx < len(cka.rl_within_cat_cka_per_layer) else float("nan")
                    bc_s = f"{bc:.4f}" if bc == bc else "NaN"
                    rc_s = f"{rc:.4f}" if rc == rc else "NaN"
                    bw_s = f"{bw:.4f}" if bw == bw else "NaN"
                    rw_s = f"{rw:.4f}" if rw == rw else "NaN"
                    w(f"| {l_idx} | {bc_s} | {rc_s} | {bw_s} | {rw_s} |")
                w("")

            # Analysis D: Late-Layer Convergence
            w("## Analysis D: Late-Layer Convergence")
            w("")
            w("| Metric | Base | RL |")
            w("|--------|------|-----|")
            w(f"| Final-layer cross-cat CKA | {conv.base_final_cross_cka:.4f} | {conv.rl_final_cross_cka:.4f} |")
            w(f"| Peak cross-cat CKA | {conv.base_peak_cross_cka:.4f} | {conv.rl_peak_cross_cka:.4f} |")
            w(f"| Convergence ratio (final/peak) | {conv.base_convergence_ratio:.4f} | {conv.rl_convergence_ratio:.4f} |")
            w(f"| Final-layer cosine similarity | {conv.base_final_cosine_similarity:.4f} | {conv.rl_final_cosine_similarity:.4f} |")
            w("")

            if conv.rl_convergence_ratio > conv.base_convergence_ratio:
                w("RL model shows **higher convergence** in final layers relative to peak differentiation.")
            else:
                w("RL model does NOT show higher convergence than base in final layers.")
            w("")

            # Analysis E: Layer Similarity (summary only, full matrix in JSON)
            w("## Analysis E: Layer Similarity Structure")
            w("")
            w("Full NxN CKA matrices saved in `raw_results.json`.")
            w("")

            # Summarize block structure
            for model_name, matrix in [
                ("Base (Qwen3-8B)", comparison.layer_similarity.base_similarity_matrix),
                ("RL (DeepSeek-R1-8B)", comparison.layer_similarity.rl_similarity_matrix),
            ]:
                if not matrix:
                    continue
                num_layers = len(matrix)
                w(f"### {model_name}")
                w("")

                # Find processing blocks (contiguous layers with CKA > 0.8)
                blocks = []
                block_start = 0
                for i in range(1, num_layers):
                    if matrix[i - 1][i] < 0.8:
                        if i - block_start >= 2:
                            blocks.append((block_start, i - 1))
                        block_start = i
                if num_layers - block_start >= 2:
                    blocks.append((block_start, num_layers - 1))

                if blocks:
                    w(f"Processing blocks (adjacent CKA > 0.8): {len(blocks)}")
                    for start, end in blocks:
                        w(f"- Layers {start}-{end} ({end - start + 1} layers)")
                else:
                    w("No strong processing blocks detected (all adjacent CKA < 0.8)")
                w("")

        # Per-model per-category breakdown
        for name, mg in model_geometries.items():
            w(f"## {name} Per-Category Details")
            w("")
            w("| Category | N | Mean ER | Mean Vel | Mean PLR | Mean Curv |")
            w("|----------|---|---------|----------|----------|-----------|")
            for cat in PROBE_CATEGORIES:
                cat_geoms = [
                    g for g in mg.prompt_geometries
                    if g.category == cat and g.error is None
                ]
                ers = [g.expansion_ratio for g in cat_geoms if g.expansion_ratio == g.expansion_ratio]
                vels = [g.mean_velocity for g in cat_geoms]
                plrs = [g.path_length_ratio for g in cat_geoms if g.path_length_ratio == g.path_length_ratio]
                curvs = [g.mean_curvature for g in cat_geoms if g.mean_curvature == g.mean_curvature]
                w(
                    f"| {cat} | {len(cat_geoms)} | "
                    f"{_safe_mean(ers):.3f} | "
                    f"{_safe_mean(vels):.4f} | "
                    f"{_safe_mean(plrs):.3f} | "
                    f"{_safe_mean(curvs):.4f} |"
                )
            w("")

        # Write report
        report_path = self.output_dir / "PROCESSING_REPORT.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Report saved to {report_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RL Processing Geometry Experiment (Experiment 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rl_processing",
        help="Output directory (default: results/rl_processing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 2 prompts per category (10 total)",
    )

    args = parser.parse_args()

    experiment = RLProcessingGeometryExperiment(
        output_dir=Path(args.output),
        seed=args.seed,
        smoke=args.smoke,
    )
    experiment.run()


if __name__ == "__main__":
    main()
