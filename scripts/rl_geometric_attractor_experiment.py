#!/usr/bin/env python3
"""RL Geometric Attractor Experiment.

Tests the hypothesis that RL training creates stable geometric attractors
that freeze processing geometry into a single fixed point.

Controlled pair: Qwen3-8B (base) vs DeepSeek-R1-0528-Qwen3-8B (RL-trained via GRPO).
Same architecture, different training objectives.

Three analysis angles:
1. Distribution Comparison — expansion_ratio per model, per task type
2. Trajectory Overlay — mean ID trajectory per (model, task_type), layer-by-layer divergence
3. Variance Quantification — Levene's test, CV comparison (core hypothesis test)
4. Attractor Detection — clustering analysis of RL model's expansion_ratios

Methodology:
- GREEDY decoding only (temp=0) — temperature is noise, not signal
- Per-prompt trajectory capture via collect_trajectory_batch
- Per-layer intrinsic dimension via batched TwoNN
- Per-layer spectral entropy via ManifoldEntropy
- All backend arrays converted to Python lists immediately, freed after capture

Usage:
    # Smoke test (2 prompts per category, ~3-5 min)
    poetry run python scripts/rl_geometric_attractor_experiment.py \\
        --smoke --output results/rl_attractor/smoke/

    # Full experiment (~10-20 min)
    poetry run python scripts/rl_geometric_attractor_experiment.py \\
        --output results/rl_attractor/
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
from pathlib import Path
from typing import Any

from modelcypher.core.domain.geometry.id_trajectory_analysis import analyze_id_trajectory
from modelcypher.core.domain.statistics import (
    cohens_d_bootstrap_ci,
    cohens_d_two_groups,
    levene_test_statistic,
    mean_trajectory,
    permutation_test_p_value,
    spearman_correlation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry
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
# Probe Categories (30 prompts total, 6 per category)
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
class ProbeResult:
    """Per-prompt raw metrics."""

    prompt: str
    category: str
    token_count: int
    expansion_ratio: float
    peak_layer: int
    peak_dim: float
    final_dim: float
    smoothness: float  # Spearman of ID trajectory with layer order
    id_trajectory: list[float]  # per-layer intrinsic dimension
    spectral_entropy_trajectory: list[float]  # per-layer spectral entropy
    norm_trajectory: list[float]  # per-layer mean L2 norm
    error: str | None = None


@dataclass
class CategorySummary:
    """Aggregated per-task-type."""

    category: str
    n_probes: int
    mean_expansion_ratio: float
    std_expansion_ratio: float
    mean_smoothness: float
    mean_peak_layer: float
    expansion_ratios: list[float]
    mean_id_trajectory: list[float]
    mean_spectral_entropy_trajectory: list[float]
    mean_norm_trajectory: list[float]


@dataclass
class ModelResult:
    """All results for one model."""

    model_name: str
    training_type: str
    architecture: str
    num_layers: int
    probe_results: list[ProbeResult]
    category_summaries: dict[str, CategorySummary]
    overall_expansion_ratio_mean: float
    overall_expansion_ratio_std: float
    overall_expansion_ratio_cv: float  # coefficient of variation
    cross_category_cv: float  # CV of per-category mean expansion_ratios


@dataclass
class ComparisonResult:
    """Cross-model statistical analysis."""

    # Overall expansion_ratio comparison
    overall_cohens_d: float
    overall_cohens_d_ci: tuple[float, float]
    overall_permutation_p: float

    # Per-category expansion_ratio comparison
    per_category_cohens_d: dict[str, float]
    per_category_permutation_p: dict[str, float]

    # Variance comparison (core hypothesis)
    base_cross_category_cv: float
    rl_cross_category_cv: float
    levene_statistic: float
    levene_p_value: float

    # Trajectory divergence
    mean_trajectory_divergence: list[float]  # per-layer |base_id - rl_id|
    max_divergence_layer: int
    max_divergence_value: float

    # Attractor detection
    rl_expansion_ratio_band_width: float  # max - min of RL per-category means
    rl_mean_expansion_ratio: float
    base_expansion_ratio_band_width: float
    base_mean_expansion_ratio: float


# =============================================================================
# Main Experiment Class
# =============================================================================


class RLAttractorExperiment:
    """RL Geometric Attractor Experiment."""

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

    def _collect_prompt_geometry(self, prompt: str, category: str) -> ProbeResult:
        """Collect geometric data for a single prompt.

        Steps:
        A. Collect trajectory activations
        B. Per-layer intrinsic dimension via batched TwoNN
        C. Per-layer spectral entropy via ManifoldEntropy
        D. Norm trajectory from positions
        E. Derive scalar metrics (expansion_ratio, smoothness)
        F. Convert to Python lists, free backend arrays
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

        if token_count < 4:
            return ProbeResult(
                prompt=prompt,
                category=category,
                token_count=token_count,
                expansion_ratio=float("nan"),
                peak_layer=-1,
                peak_dim=0.0,
                final_dim=0.0,
                smoothness=0.0,
                id_trajectory=[],
                spectral_entropy_trajectory=[],
                norm_trajectory=[],
                error=f"Too few tokens ({token_count}), need >= 4",
            )

        # Step B: Per-layer intrinsic dimension via batched TwoNN
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

        estimator = IntrinsicDimension(b)
        layer_indices = sorted(trajectory.positions.keys())
        point_clouds = [trajectory.positions[i] for i in layer_indices]
        id_results = estimator.batch_compute(point_clouds)

        id_trajectory: list[float] = []
        for result in id_results:
            if result is not None:
                id_trajectory.append(result.intrinsic_dimension)
            else:
                id_trajectory.append(float("nan"))

        # Step C: Per-layer spectral entropy via ManifoldEntropy
        from modelcypher.core.domain.geometry.manifold_entropy import ManifoldEntropy

        entropy_analyzer = ManifoldEntropy(b)
        entropy_result = entropy_analyzer.compute_from_activations(trajectory.positions)

        spectral_entropy_trajectory: list[float] = []
        for li in layer_indices:
            if li in entropy_result.layer_entropies:
                spectral_entropy_trajectory.append(
                    entropy_result.layer_entropies[li].spectral_entropy
                )
            else:
                spectral_entropy_trajectory.append(float("nan"))

        # Step D: Norm trajectory
        norm_trajectory: list[float] = []
        for li in layer_indices:
            pos = trajectory.positions[li]
            norm_val = b.sqrt(b.sum(pos * pos, axis=1))
            mean_norm = b.mean(norm_val)
            b.eval(mean_norm)
            norm_trajectory.append(float(b.to_scalar(mean_norm)))

        # Step E: Derive scalar metrics
        id_analysis = analyze_id_trajectory(id_trajectory)
        expansion_ratio = id_analysis.expansion_ratio
        peak_layer = id_analysis.peak_layer
        peak_dim = id_analysis.peak_dim
        final_dim = id_analysis.final_dim
        smoothness = id_analysis.smoothness

        # Step F: Free backend arrays (trajectory positions are large)
        del trajectory, point_clouds, entropy_result
        try:
            b.clear_cache()
        except Exception:
            pass

        return ProbeResult(
            prompt=prompt,
            category=category,
            token_count=token_count,
            expansion_ratio=expansion_ratio,
            peak_layer=peak_layer,
            peak_dim=peak_dim,
            final_dim=final_dim,
            smoothness=smoothness,
            id_trajectory=id_trajectory,
            spectral_entropy_trajectory=spectral_entropy_trajectory,
            norm_trajectory=norm_trajectory,
        )

    # ----- Model-level collection -----

    def _collect_model_geometry(self, model_name: str) -> list[ProbeResult]:
        """Collect geometry for all prompts for one model."""
        results: list[ProbeResult] = []
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
                            f"expansion_ratio={result.expansion_ratio:.3f}, "
                            f"smoothness={result.smoothness:.3f}"
                        )
                except Exception as e:
                    logger.warning(f"    FAILED: {e}")
                    results.append(ProbeResult(
                        prompt=prompt,
                        category=category,
                        token_count=0,
                        expansion_ratio=float("nan"),
                        peak_layer=-1,
                        peak_dim=0.0,
                        final_dim=0.0,
                        smoothness=0.0,
                        id_trajectory=[],
                        spectral_entropy_trajectory=[],
                        norm_trajectory=[],
                        error=str(e),
                    ))

                # Memory management: clear cache every 5 prompts
                if prompt_count % 5 == 0 and self.backend is not None:
                    try:
                        self.backend.clear_cache()
                    except Exception:
                        pass

        return results

    # ----- Analysis helpers -----

    def _build_category_summary(
        self, results: list[ProbeResult], category: str
    ) -> CategorySummary:
        """Build aggregated summary for one category."""
        cat_results = [r for r in results if r.category == category and r.error is None]
        expansion_ratios = [
            r.expansion_ratio for r in cat_results
            if r.expansion_ratio == r.expansion_ratio  # filter NaN
        ]

        n = len(expansion_ratios)
        mean_er = sum(expansion_ratios) / n if n > 0 else 0.0
        std_er = (
            math.sqrt(sum((x - mean_er) ** 2 for x in expansion_ratios) / n)
            if n > 1
            else 0.0
        )

        smoothnesses = [r.smoothness for r in cat_results]
        mean_smooth = sum(smoothnesses) / len(smoothnesses) if smoothnesses else 0.0

        peak_layers = [r.peak_layer for r in cat_results if r.peak_layer >= 0]
        mean_peak = sum(peak_layers) / len(peak_layers) if peak_layers else 0.0

        # Mean trajectories (element-wise average)
        mean_id_traj = mean_trajectory([r.id_trajectory for r in cat_results])
        mean_se_traj = mean_trajectory(
            [r.spectral_entropy_trajectory for r in cat_results]
        )
        mean_norm_traj = mean_trajectory(
            [r.norm_trajectory for r in cat_results]
        )

        return CategorySummary(
            category=category,
            n_probes=n,
            mean_expansion_ratio=mean_er,
            std_expansion_ratio=std_er,
            mean_smoothness=mean_smooth,
            mean_peak_layer=mean_peak,
            expansion_ratios=expansion_ratios,
            mean_id_trajectory=mean_id_traj,
            mean_spectral_entropy_trajectory=mean_se_traj,
            mean_norm_trajectory=mean_norm_traj,
        )

    def _build_model_result(
        self, model_name: str, results: list[ProbeResult]
    ) -> ModelResult:
        """Build full model result with summaries."""
        model_info = MODEL_REGISTRY[model_name]

        category_summaries = {}
        for category in PROBE_CATEGORIES:
            category_summaries[category] = self._build_category_summary(
                results, category
            )

        # Overall expansion_ratio stats
        all_ers = [
            r.expansion_ratio for r in results
            if r.error is None and r.expansion_ratio == r.expansion_ratio
        ]
        n = len(all_ers)
        mean_er = sum(all_ers) / n if n > 0 else 0.0
        std_er = (
            math.sqrt(sum((x - mean_er) ** 2 for x in all_ers) / n)
            if n > 1
            else 0.0
        )
        cv = std_er / mean_er if mean_er > 0 else 0.0

        # Cross-category CV: variability of per-category means
        cat_means = [
            s.mean_expansion_ratio for s in category_summaries.values()
            if s.n_probes > 0
        ]
        if len(cat_means) >= 2:
            cat_mean = sum(cat_means) / len(cat_means)
            cat_std = math.sqrt(
                sum((x - cat_mean) ** 2 for x in cat_means) / len(cat_means)
            )
            cross_cat_cv = cat_std / cat_mean if cat_mean > 0 else 0.0
        else:
            cross_cat_cv = 0.0

        return ModelResult(
            model_name=model_name,
            training_type=model_info["training"],
            architecture=model_info["architecture"],
            num_layers=self.num_layers,
            probe_results=results,
            category_summaries=category_summaries,
            overall_expansion_ratio_mean=mean_er,
            overall_expansion_ratio_std=std_er,
            overall_expansion_ratio_cv=cv,
            cross_category_cv=cross_cat_cv,
        )

    # ----- Cross-model comparison -----

    def _compare_models(
        self, base: ModelResult, rl: ModelResult
    ) -> ComparisonResult:
        """Statistical comparison between base and RL models."""
        rng = self._rng

        # Overall expansion_ratio distributions
        base_ers = [
            r.expansion_ratio for r in base.probe_results
            if r.error is None and r.expansion_ratio == r.expansion_ratio
        ]
        rl_ers = [
            r.expansion_ratio for r in rl.probe_results
            if r.error is None and r.expansion_ratio == r.expansion_ratio
        ]

        overall_d = cohens_d_two_groups(base_ers, rl_ers)
        overall_ci = cohens_d_bootstrap_ci(base_ers, rl_ers, rng=rng)
        overall_p = permutation_test_p_value(base_ers, rl_ers, rng=rng)

        # Per-category comparison
        per_cat_d: dict[str, float] = {}
        per_cat_p: dict[str, float] = {}
        for cat in PROBE_CATEGORIES:
            base_cat = base.category_summaries[cat].expansion_ratios
            rl_cat = rl.category_summaries[cat].expansion_ratios
            per_cat_d[cat] = cohens_d_two_groups(base_cat, rl_cat)
            per_cat_p[cat] = permutation_test_p_value(base_cat, rl_cat, rng=rng)

        # Variance comparison: Levene's test on per-category expansion_ratios
        base_groups = [
            base.category_summaries[cat].expansion_ratios
            for cat in PROBE_CATEGORIES
            if base.category_summaries[cat].n_probes > 0
        ]
        rl_groups = [
            rl.category_summaries[cat].expansion_ratios
            for cat in PROBE_CATEGORIES
            if rl.category_summaries[cat].n_probes > 0
        ]

        # Levene's test: compare variance across ALL groups (base + RL categories)
        # But the core test is whether base has MORE variance than RL
        levene_stat, levene_p = levene_test_statistic(
            [base_ers, rl_ers]
        )

        # Trajectory divergence
        mean_divergence = []
        base_mean_traj = mean_trajectory(
            [s.mean_id_trajectory for s in base.category_summaries.values()]
        )
        rl_mean_traj = mean_trajectory(
            [s.mean_id_trajectory for s in rl.category_summaries.values()]
        )
        max_len = min(len(base_mean_traj), len(rl_mean_traj))
        for i in range(max_len):
            bv = base_mean_traj[i]
            rv = rl_mean_traj[i]
            if bv == bv and rv == rv:  # both not NaN
                mean_divergence.append(abs(bv - rv))
            else:
                mean_divergence.append(float("nan"))

        max_div_val = 0.0
        max_div_layer = 0
        for i, d in enumerate(mean_divergence):
            if d == d and d > max_div_val:
                max_div_val = d
                max_div_layer = i

        # Attractor detection
        base_cat_means = [
            s.mean_expansion_ratio for s in base.category_summaries.values()
            if s.n_probes > 0
        ]
        rl_cat_means = [
            s.mean_expansion_ratio for s in rl.category_summaries.values()
            if s.n_probes > 0
        ]

        return ComparisonResult(
            overall_cohens_d=overall_d,
            overall_cohens_d_ci=overall_ci,
            overall_permutation_p=overall_p,
            per_category_cohens_d=per_cat_d,
            per_category_permutation_p=per_cat_p,
            base_cross_category_cv=base.cross_category_cv,
            rl_cross_category_cv=rl.cross_category_cv,
            levene_statistic=levene_stat,
            levene_p_value=levene_p,
            mean_trajectory_divergence=mean_divergence,
            max_divergence_layer=max_div_layer,
            max_divergence_value=max_div_val,
            rl_expansion_ratio_band_width=(
                max(rl_cat_means) - min(rl_cat_means) if rl_cat_means else 0.0
            ),
            rl_mean_expansion_ratio=(
                sum(rl_cat_means) / len(rl_cat_means) if rl_cat_means else 0.0
            ),
            base_expansion_ratio_band_width=(
                max(base_cat_means) - min(base_cat_means) if base_cat_means else 0.0
            ),
            base_mean_expansion_ratio=(
                sum(base_cat_means) / len(base_cat_means) if base_cat_means else 0.0
            ),
        )

    # ----- Main experiment loop -----

    def run(self) -> None:
        """Run full RL attractor experiment."""
        logger.info("=" * 70)
        logger.info("RL GEOMETRIC ATTRACTOR EXPERIMENT")
        logger.info("Hypothesis: RL training creates stable geometric attractors")
        logger.info(f"Models: {list(MODEL_REGISTRY.keys())}")
        logger.info(f"Categories: {list(PROBE_CATEGORIES.keys())}")
        prompts_per_cat = 2 if self.smoke else 6
        total_prompts = len(PROBE_CATEGORIES) * prompts_per_cat
        logger.info(f"Prompts: {total_prompts} ({'smoke' if self.smoke else 'full'})")
        logger.info(f"Seed: {self.seed}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 70)

        # Verify volume is accessible
        if not os.path.exists(MODELS_BASE):
            logger.error(f"Model volume not accessible: {MODELS_BASE}")
            logger.error("Is the external drive mounted?")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        model_results: dict[str, ModelResult] = {}

        for model_name in MODEL_REGISTRY:
            logger.info(f"\n{'='*60}")
            logger.info(f"MODEL: {model_name}")
            logger.info(f"{'='*60}")

            t0 = time.time()

            try:
                self._load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            try:
                probe_results = self._collect_model_geometry(model_name)
            except Exception as e:
                logger.error(f"Failed to collect geometry for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                self._unload_model()
                continue

            # Save intermediate results
            self._save_intermediate(model_name, probe_results)

            # Build model result
            result = self._build_model_result(model_name, probe_results)
            model_results[model_name] = result

            elapsed = time.time() - t0
            logger.info(
                f"\n{model_name} complete: {elapsed:.1f}s, "
                f"expansion_ratio={result.overall_expansion_ratio_mean:.3f} "
                f"(CV={result.overall_expansion_ratio_cv:.3f}), "
                f"cross-category CV={result.cross_category_cv:.3f}"
            )

            self._unload_model()

        # Cross-model comparison
        comparison = None
        if "Qwen3-8B" in model_results and "DeepSeek-R1-8B" in model_results:
            logger.info("\n" + "=" * 60)
            logger.info("CROSS-MODEL COMPARISON")
            logger.info("=" * 60)
            comparison = self._compare_models(
                model_results["Qwen3-8B"],
                model_results["DeepSeek-R1-8B"],
            )
            logger.info(f"  Overall Cohen's d: {comparison.overall_cohens_d:+.3f}")
            logger.info(f"  Levene's F: {comparison.levene_statistic:.3f} (p={comparison.levene_p_value:.4f})")
            logger.info(f"  Base cross-category CV: {comparison.base_cross_category_cv:.4f}")
            logger.info(f"  RL cross-category CV: {comparison.rl_cross_category_cv:.4f}")

        # Save results and generate report
        self._save_results(model_results, comparison)
        self._generate_report(model_results, comparison)

        logger.info("\nExperiment complete.")

    # ----- Save / Load -----

    def _save_intermediate(
        self, model_name: str, results: list[ProbeResult]
    ) -> None:
        """Save per-model results for crash recovery."""
        raw_dir = self.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{model_name}_probes.jsonl"

        with open(path, "w") as f:
            for r in results:
                record = {
                    "prompt": r.prompt[:200],
                    "category": r.category,
                    "token_count": r.token_count,
                    "expansion_ratio": r.expansion_ratio if r.expansion_ratio == r.expansion_ratio else None,
                    "peak_layer": r.peak_layer,
                    "peak_dim": r.peak_dim,
                    "final_dim": r.final_dim,
                    "smoothness": r.smoothness,
                    "id_trajectory": [
                        v if v == v else None for v in r.id_trajectory
                    ],
                    "spectral_entropy_trajectory": [
                        v if v == v else None for v in r.spectral_entropy_trajectory
                    ],
                    "norm_trajectory": r.norm_trajectory,
                    "error": r.error,
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"  Saved {len(results)} probes to {path}")

    def _save_results(
        self,
        model_results: dict[str, ModelResult],
        comparison: ComparisonResult | None,
    ) -> None:
        """Save full results as JSON."""
        def _safe_float(v: float) -> float | None:
            return v if v == v else None

        serializable: dict[str, Any] = {
            "experiment": {
                "seed": self.seed,
                "smoke": self.smoke,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_categories": len(PROBE_CATEGORIES),
                "prompts_per_category": 2 if self.smoke else 6,
            },
            "models": {},
        }

        for name, mr in model_results.items():
            model_data: dict[str, Any] = {
                "training_type": mr.training_type,
                "architecture": mr.architecture,
                "num_layers": mr.num_layers,
                "overall_expansion_ratio_mean": _safe_float(mr.overall_expansion_ratio_mean),
                "overall_expansion_ratio_std": _safe_float(mr.overall_expansion_ratio_std),
                "overall_expansion_ratio_cv": _safe_float(mr.overall_expansion_ratio_cv),
                "cross_category_cv": _safe_float(mr.cross_category_cv),
                "categories": {},
            }

            for cat, cs in mr.category_summaries.items():
                model_data["categories"][cat] = {
                    "n_probes": cs.n_probes,
                    "mean_expansion_ratio": _safe_float(cs.mean_expansion_ratio),
                    "std_expansion_ratio": _safe_float(cs.std_expansion_ratio),
                    "mean_smoothness": _safe_float(cs.mean_smoothness),
                    "mean_peak_layer": _safe_float(cs.mean_peak_layer),
                    "expansion_ratios": [_safe_float(v) for v in cs.expansion_ratios],
                    "mean_id_trajectory": [_safe_float(v) for v in cs.mean_id_trajectory],
                    "mean_spectral_entropy_trajectory": [
                        _safe_float(v) for v in cs.mean_spectral_entropy_trajectory
                    ],
                    "mean_norm_trajectory": [_safe_float(v) for v in cs.mean_norm_trajectory],
                }

            serializable["models"][name] = model_data

        if comparison is not None:
            serializable["comparison"] = {
                "overall_cohens_d": _safe_float(comparison.overall_cohens_d),
                "overall_cohens_d_ci": [
                    _safe_float(comparison.overall_cohens_d_ci[0]),
                    _safe_float(comparison.overall_cohens_d_ci[1]),
                ],
                "overall_permutation_p": _safe_float(comparison.overall_permutation_p),
                "per_category_cohens_d": {
                    k: _safe_float(v) for k, v in comparison.per_category_cohens_d.items()
                },
                "per_category_permutation_p": {
                    k: _safe_float(v) for k, v in comparison.per_category_permutation_p.items()
                },
                "base_cross_category_cv": _safe_float(comparison.base_cross_category_cv),
                "rl_cross_category_cv": _safe_float(comparison.rl_cross_category_cv),
                "levene_statistic": _safe_float(comparison.levene_statistic),
                "levene_p_value": _safe_float(comparison.levene_p_value),
                "mean_trajectory_divergence": [
                    _safe_float(v) for v in comparison.mean_trajectory_divergence
                ],
                "max_divergence_layer": comparison.max_divergence_layer,
                "max_divergence_value": _safe_float(comparison.max_divergence_value),
                "rl_expansion_ratio_band_width": _safe_float(
                    comparison.rl_expansion_ratio_band_width
                ),
                "rl_mean_expansion_ratio": _safe_float(
                    comparison.rl_mean_expansion_ratio
                ),
                "base_expansion_ratio_band_width": _safe_float(
                    comparison.base_expansion_ratio_band_width
                ),
                "base_mean_expansion_ratio": _safe_float(
                    comparison.base_mean_expansion_ratio
                ),
            }

        path = self.output_dir / "raw_results.json"
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Results saved to {path}")

    # ----- Report generation -----

    def _generate_report(
        self,
        model_results: dict[str, ModelResult],
        comparison: ComparisonResult | None,
    ) -> None:
        """Generate ATTRACTOR_REPORT.md."""
        lines: list[str] = []
        w = lines.append

        w("# RL Geometric Attractor Report")
        w("")
        w(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
        w(f"**Seed**: {self.seed}")
        w(f"**Mode**: {'smoke test' if self.smoke else 'full experiment'}")
        w(f"**Prompts per category**: {2 if self.smoke else 6}")
        w("")

        # Hypothesis
        w("## Hypothesis")
        w("")
        w("RL training creates stable geometric attractors that freeze processing")
        w("geometry into a single fixed point. Evidence:")
        w("- Base models show task-differentiated expansion_ratio (retrieval ~1.8, reasoning ~1.0)")
        w("- RL/specialist models collapse expansion_ratio to ~1.0 across ALL task types")
        w("- Core test: base cross-category CV >> RL cross-category CV")
        w("")

        # Executive Summary
        w("## Executive Summary")
        w("")
        w("| Model | Training | Mean ER | Std ER | CV | Cross-Cat CV |")
        w("|-------|----------|---------|--------|-----|-------------|")
        for name, mr in model_results.items():
            w(
                f"| {name} | {mr.training_type} | "
                f"{mr.overall_expansion_ratio_mean:.3f} | "
                f"{mr.overall_expansion_ratio_std:.3f} | "
                f"{mr.overall_expansion_ratio_cv:.3f} | "
                f"{mr.cross_category_cv:.4f} |"
            )
        w("")

        # Per-category breakdown
        w("## Per-Category Expansion Ratios")
        w("")
        categories = list(PROBE_CATEGORIES.keys())
        header = "| Model | " + " | ".join(categories) + " |"
        sep = "|-------|" + "|".join(["--------"] * len(categories)) + "|"
        w(header)
        w(sep)
        for name, mr in model_results.items():
            cells = []
            for cat in categories:
                cs = mr.category_summaries[cat]
                if cs.n_probes > 0:
                    cells.append(f"{cs.mean_expansion_ratio:.3f}")
                else:
                    cells.append("N/A")
            w(f"| {name} | " + " | ".join(cells) + " |")
        w("")

        # Comparison statistics
        if comparison is not None:
            w("## Statistical Comparison")
            w("")

            w("### Overall Expansion Ratio")
            w("")
            w(f"- Cohen's d: **{comparison.overall_cohens_d:+.3f}** "
              f"(95% CI: [{comparison.overall_cohens_d_ci[0]:+.3f}, "
              f"{comparison.overall_cohens_d_ci[1]:+.3f}])")
            w(f"- Permutation p: {comparison.overall_permutation_p:.4f}")
            w("")

            w("### Per-Category Cohen's d (base vs RL)")
            w("")
            w("| Category | Cohen's d | Permutation p |")
            w("|----------|-----------|---------------|")
            for cat in categories:
                d = comparison.per_category_cohens_d.get(cat, 0.0)
                p = comparison.per_category_permutation_p.get(cat, 1.0)
                w(f"| {cat} | {d:+.3f} | {p:.4f} |")
            w("")

            # Core hypothesis test
            w("### Variance Analysis (CORE HYPOTHESIS TEST)")
            w("")
            w("**Test**: Is base model geometry more task-differentiated than RL model?")
            w("")
            w(f"- Base cross-category CV: **{comparison.base_cross_category_cv:.4f}**")
            w(f"- RL cross-category CV: **{comparison.rl_cross_category_cv:.4f}**")
            cv_ratio = (
                comparison.base_cross_category_cv / comparison.rl_cross_category_cv
                if comparison.rl_cross_category_cv > 0
                else float("inf")
            )
            w(f"- Ratio (base/RL): **{cv_ratio:.2f}x**")
            w(f"- Levene's F: {comparison.levene_statistic:.3f} (p={comparison.levene_p_value:.4f})")
            w("")

            if comparison.base_cross_category_cv > comparison.rl_cross_category_cv * 2:
                w("**VERDICT: SUPPORTS ATTRACTOR HYPOTHESIS**")
                w("")
                w("Base model shows >2x more task-differentiated geometry than RL model.")
            elif comparison.base_cross_category_cv > comparison.rl_cross_category_cv:
                w("**VERDICT: WEAK SUPPORT** -- base shows more variance, but <2x difference")
            else:
                w("**VERDICT: DOES NOT SUPPORT** -- RL model has equal or more variance")
            w("")

            # Attractor detection
            w("### Attractor Detection")
            w("")
            w(f"- RL expansion_ratio band width: "
              f"**{comparison.rl_expansion_ratio_band_width:.3f}** "
              f"(mean={comparison.rl_mean_expansion_ratio:.3f})")
            w(f"- Base expansion_ratio band width: "
              f"**{comparison.base_expansion_ratio_band_width:.3f}** "
              f"(mean={comparison.base_mean_expansion_ratio:.3f})")
            w("")

            narrow_band = comparison.rl_expansion_ratio_band_width < 0.3
            near_one = abs(comparison.rl_mean_expansion_ratio - 1.0) < 0.3

            if narrow_band and near_one:
                w("RL model shows **narrow attractor basin** near expansion_ratio ~1.0")
            elif narrow_band:
                w(f"RL model shows **narrow attractor basin** "
                  f"(mean={comparison.rl_mean_expansion_ratio:.3f}, not at 1.0)")
            else:
                w("RL model does NOT show a narrow attractor basin")
            w("")

            # Trajectory divergence
            w("### Trajectory Divergence")
            w("")
            w(f"- Maximum divergence at layer **{comparison.max_divergence_layer}**: "
              f"{comparison.max_divergence_value:.3f}")
            w("")

            if comparison.mean_trajectory_divergence:
                w("Layer-by-layer |base_id - rl_id|:")
                w("")
                w("| Layer | Divergence |")
                w("|-------|------------|")
                for i, d in enumerate(comparison.mean_trajectory_divergence):
                    marker = " <-- max" if i == comparison.max_divergence_layer else ""
                    d_str = f"{d:.3f}" if d == d else "NaN"
                    w(f"| {i} | {d_str}{marker} |")
                w("")

        # Per-model details
        for name, mr in model_results.items():
            w(f"## {name} Details")
            w("")
            w(f"- Training: {mr.training_type}")
            w(f"- Architecture: {mr.architecture}")
            w(f"- Layers: {mr.num_layers}")
            w(f"- Overall expansion_ratio: {mr.overall_expansion_ratio_mean:.3f} "
              f"(std={mr.overall_expansion_ratio_std:.3f}, CV={mr.overall_expansion_ratio_cv:.3f})")
            w(f"- Cross-category CV: {mr.cross_category_cv:.4f}")
            w("")

            w("### Per-Category Summary")
            w("")
            w("| Category | N | Mean ER | Std ER | Smoothness | Peak Layer |")
            w("|----------|---|---------|--------|------------|------------|")
            for cat in categories:
                cs = mr.category_summaries[cat]
                w(
                    f"| {cat} | {cs.n_probes} | "
                    f"{cs.mean_expansion_ratio:.3f} | "
                    f"{cs.std_expansion_ratio:.3f} | "
                    f"{cs.mean_smoothness:.3f} | "
                    f"{cs.mean_peak_layer:.1f} |"
                )
            w("")

        # Write report
        report_path = self.output_dir / "ATTRACTOR_REPORT.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Report saved to {report_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RL Geometric Attractor Experiment"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rl_attractor",
        help="Output directory (default: results/rl_attractor)",
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

    experiment = RLAttractorExperiment(
        output_dir=Path(args.output),
        seed=args.seed,
        smoke=args.smoke,
    )
    experiment.run()


if __name__ == "__main__":
    main()
