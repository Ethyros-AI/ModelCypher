#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 17: SOAR Curriculum - Geometric Quality Detection
#
# Tests hypothesis: "Fisher Information and Mode Connectivity can detect
# structural quality of training data without solving problems."
#
# Based on SOAR paper (MIT/Meta, arXiv:2601.18778):
# "Structural quality and well-posedness are more critical for learning
# progress than solution correctness."

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.validation_protocol.shared.config import (
    ExperimentConfig,
    ExperimentResult,
    setup_experiment,
)
from experiments.validation_protocol.shared.lora_utils import (
    train_lora_quick,
    evaluate_perplexity,
)
from experiments.validation_protocol.exp17_soar_curriculum.problem_generator import (
    generate_problems,
    save_problems,
    load_problems,
    ArithmeticChainProblem,
)
from experiments.validation_protocol.exp17_soar_curriculum.structural_metrics import (
    compute_structural_metrics,
    collect_problem_activations,
    StructuralMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = Path("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
OUTPUT_DIR = Path(__file__).parent
PROBLEMS_DIR = OUTPUT_DIR / "problems"


@dataclass
class QualityGroup:
    """A group of problems with similar structural quality."""
    name: str
    problems: list[ArithmeticChainProblem]
    mean_quality_score: float
    mean_fisher: float
    mean_barrier: float

    # Training results (filled after training)
    final_loss: float = 0.0
    final_perplexity: float = 0.0


def compute_metrics_for_problems(
    problems: list[ArithmeticChainProblem],
    model,
    tokenizer,
    reference_activations,
    layer_idx: int,
    model_path: str | None = None,
) -> list[ArithmeticChainProblem]:
    """Compute structural metrics for each problem.

    Args:
        problems: List of problems to measure
        model: Loaded model
        tokenizer: Tokenizer
        reference_activations: Reference for comparison
        layer_idx: Which layer to use
        model_path: Optional model path for architecture detection

    Returns:
        Problems with metrics filled in
    """
    from modelcypher.core.domain._backend import get_default_backend
    backend = get_default_backend()

    for i, problem in enumerate(problems):
        # Collect activations for this problem
        acts = collect_problem_activations(
            model, tokenizer, [problem.prompt], layer_idx, model_path=model_path
        )
        mx.eval(acts)

        # Compute metrics
        metrics = compute_structural_metrics(acts, reference_activations, backend)

        # Update problem
        problem.fisher_compatibility = metrics.cka_similarity
        problem.manifold_curvature = metrics.fisher_mean  # Using as proxy
        problem.curvature_variance = 1.0 / max(metrics.fisher_effective_rank, 1.0)  # Low effective rank = high variance
        problem.barrier_height = metrics.barrier_height

        if (i + 1) % 20 == 0:
            logger.info(f"  Computed metrics for {i + 1}/{len(problems)} problems")

    return problems


def stratify_by_quality(
    problems: list[ArithmeticChainProblem],
    n_groups: int = 3,
) -> list[QualityGroup]:
    """Stratify problems into quality groups.

    INSIGHT from exp17 v1: "High quality" = Goldilocks zone, not maximum similarity!
    Problems that are too similar to reference (CKA~1.0) teach nothing.
    Problems with moderate challenge (CKA~0.9, barrier~0.05) teach best.

    Args:
        problems: Problems with metrics computed
        n_groups: Number of groups (default 3: high, medium, low)

    Returns:
        List of QualityGroup objects
    """
    # Compute combined quality score for each problem
    # NEW: Goldilocks scoring - moderate difficulty is GOOD
    for p in problems:
        # Goldilocks CKA: bell curve centered at 0.9
        cka_goldilocks = 1.0 - abs(p.fisher_compatibility - 0.90) * 5.0
        cka_goldilocks = max(0.0, min(1.0, cka_goldilocks))

        # Productive difficulty: barrier in [0.02, 0.10] is ideal
        if p.barrier_height < 0.02:
            barrier_score = p.barrier_height / 0.02
        elif p.barrier_height <= 0.10:
            barrier_score = 1.0
        else:
            barrier_score = max(0.0, 1.0 - (p.barrier_height - 0.10) * 5)

        # Learning opportunity: lower Fisher = model needs to learn
        fisher_learning = 1.0 - min(p.manifold_curvature * 100, 1.0)

        p.structural_quality = (
            0.4 * cka_goldilocks +      # Moderate similarity
            0.3 * barrier_score +        # Productive difficulty
            0.3 * fisher_learning        # Learning opportunity
        )

    # Sort by quality score
    sorted_problems = sorted(problems, key=lambda p: p.structural_quality, reverse=True)

    # Split into groups
    group_size = len(sorted_problems) // n_groups
    groups = []

    group_names = ["high_quality", "medium_quality", "low_quality"]
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else len(sorted_problems)
        group_problems = sorted_problems[start:end]

        group = QualityGroup(
            name=group_names[i] if i < len(group_names) else f"group_{i}",
            problems=group_problems,
            mean_quality_score=sum(p.structural_quality for p in group_problems) / len(group_problems),
            mean_fisher=sum(p.manifold_curvature for p in group_problems) / len(group_problems),
            mean_barrier=sum(p.barrier_height for p in group_problems) / len(group_problems),
        )
        groups.append(group)

    return groups


def create_training_data(problems: list[ArithmeticChainProblem], output_path: Path) -> Path:
    """Create JSONL training file from problems.

    Args:
        problems: Problems to convert
        output_path: Where to save

    Returns:
        Path to created file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for p in problems:
            json.dump({"text": f"{p.prompt} {p.answer}"}, f)
            f.write("\n")

    return output_path


def train_on_group(
    group: QualityGroup,
    model_path: Path,
    output_dir: Path,
    steps: int = 50,
    rank: int = 8,
) -> QualityGroup:
    """Train a LoRA on a quality group and evaluate.

    Args:
        group: Quality group with problems
        model_path: Base model path
        output_dir: Where to save LoRA
        steps: Training steps
        rank: LoRA rank

    Returns:
        Group with training results filled in
    """
    # Create training data
    train_path = output_dir / f"{group.name}_train.jsonl"
    create_training_data(group.problems, train_path)

    lora_output = output_dir / "loras" / group.name

    try:
        # Train LoRA
        train_result = train_lora_quick(
            model_path=model_path,
            dataset_path=train_path,
            output_path=lora_output,
            rank=rank,
            target_modules=["q_proj", "v_proj"],
            steps=steps,
            lr=1e-4,
        )
        group.final_loss = train_result.final_loss

        # Evaluate perplexity
        eval_result = evaluate_perplexity(
            model_path=model_path,
            dataset_path=train_path,
            lora_path=lora_output,
            max_samples=len(group.problems),
        )
        group.final_perplexity = eval_result.perplexity

        logger.info(
            f"  {group.name}: loss={group.final_loss:.4f}, perplexity={group.final_perplexity:.2f}"
        )

    except Exception as e:
        logger.error(f"  {group.name} training failed: {e}")
        group.final_loss = float("inf")
        group.final_perplexity = float("inf")

    return group


def compute_correlations(groups: list[QualityGroup]) -> dict[str, float]:
    """Compute correlations between structural quality and training outcomes.

    Args:
        groups: Quality groups with training results

    Returns:
        Dictionary of correlation metrics
    """
    import numpy as np

    # Extract data
    quality_scores = [g.mean_quality_score for g in groups]
    fisher_scores = [g.mean_fisher for g in groups]
    barrier_scores = [g.mean_barrier for g in groups]
    perplexities = [g.final_perplexity for g in groups]
    losses = [g.final_loss for g in groups]

    # Filter out infinite values
    valid_mask = [p < float("inf") and l < float("inf") for p, l in zip(perplexities, losses)]
    if sum(valid_mask) < 2:
        return {"error": "Not enough valid data points"}

    quality_valid = [q for q, v in zip(quality_scores, valid_mask) if v]
    perplexity_valid = [p for p, v in zip(perplexities, valid_mask) if v]

    # Compute Pearson correlations
    def pearson_r(x, y):
        if len(x) < 2:
            return 0.0
        x_arr = np.array(x)
        y_arr = np.array(y)
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)
        num = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denom = np.sqrt(np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2))
        return float(num / max(denom, 1e-10))

    # Quality vs perplexity (expect negative - higher quality = lower perplexity)
    quality_perplexity_r = pearson_r(quality_valid, perplexity_valid)

    # Ratio of high vs low quality perplexity
    high_quality_ppl = groups[0].final_perplexity if groups[0].final_perplexity < float("inf") else None
    low_quality_ppl = groups[-1].final_perplexity if groups[-1].final_perplexity < float("inf") else None

    ppl_ratio = None
    if high_quality_ppl and low_quality_ppl:
        ppl_ratio = high_quality_ppl / low_quality_ppl

    return {
        "quality_perplexity_correlation": quality_perplexity_r,
        "high_quality_perplexity": high_quality_ppl,
        "low_quality_perplexity": low_quality_ppl,
        "perplexity_ratio_high_vs_low": ppl_ratio,
    }


def run_experiment():
    """Main experiment runner."""
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("EXPERIMENT 17: SOAR Curriculum - Geometric Quality Detection")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Hypothesis: Structural quality metrics predict training effectiveness")
    logger.info("")

    # Setup
    from modelcypher.backends import initialize_default_backend
    backend = initialize_default_backend()

    config = setup_experiment(
        name="exp17_soar_curriculum",
        source_path=MODEL_PATH,
        target_path=MODEL_PATH,
        backend=backend,
        hyperparameters={
            "hypothesis": "Goldilocks quality (moderate challenge) predicts training effectiveness",
            "test_type": "quality_performance_correlation",
            "version": "v2_goldilocks",  # v1 showed inverted correlation - too similar = bad
            "n_problems": 60,
            "depths": [1, 2, 3],
            "n_groups": 3,
            "lora_rank": 8,
            "training_steps": 50,
            "reference": "SOAR paper (arXiv:2601.18778)",
            "insight_from_v1": "Problems too similar to reference (CKA~1.0) don't teach - need productive difficulty",
        },
    )

    # ===================================================================
    # PART 1: Generate Problems
    # ===================================================================
    logger.info("=" * 70)
    logger.info("PART 1: Generating Arithmetic Chain Problems")
    logger.info("=" * 70)

    problems = generate_problems(
        n_problems=config.hyperparameters["n_problems"],
        depths=config.hyperparameters["depths"],
        seed=config.random_seed,
    )
    save_problems(problems, PROBLEMS_DIR)

    logger.info(f"Generated {len(problems)} problems")
    logger.info("Sample problems:")
    for p in problems[:3]:
        logger.info(f"  [{p.depth}] {p.prompt} → {p.answer}")

    # ===================================================================
    # PART 2: Compute Structural Metrics
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 2: Computing Structural Quality Metrics")
    logger.info("=" * 70)

    # Load model for activation collection
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.adapters.model_architecture import get_model_architecture

    model, tokenizer = load_model_for_training(str(MODEL_PATH))
    arch = get_model_architecture(model, model_path=MODEL_PATH)
    num_layers = arch.num_layers
    layer_idx = num_layers // 2  # Middle layer

    logger.info(f"Using layer {layer_idx} of {num_layers} for metrics")

    # Collect reference activations (from simple prompts)
    reference_prompts = ["What is 2+2?", "What is 5+3?", "What is 10-4?"]
    reference_activations = collect_problem_activations(
        model, tokenizer, reference_prompts, layer_idx, model_path=str(MODEL_PATH)
    )
    mx.eval(reference_activations)
    logger.info(f"Reference activations shape: {reference_activations.shape}")

    # Compute metrics for all problems
    logger.info("Computing metrics for problems...")
    problems = compute_metrics_for_problems(
        problems, model, tokenizer, reference_activations, layer_idx, model_path=str(MODEL_PATH)
    )

    # Clean up model
    del model
    gc.collect()

    # Save problems with metrics
    save_problems(problems, PROBLEMS_DIR)

    # ===================================================================
    # PART 3: Stratify by Quality and Train
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 3: Stratify by Quality and Train LoRAs")
    logger.info("=" * 70)

    groups = stratify_by_quality(problems, n_groups=3)

    for group in groups:
        logger.info(
            f"  {group.name}: n={len(group.problems)}, "
            f"quality={group.mean_quality_score:.4f}, "
            f"barrier={group.mean_barrier:.4f}"
        )

    # Train on each group
    logger.info("")
    logger.info("Training LoRAs on each quality group...")

    for group in groups:
        logger.info(f"\nTraining on {group.name} ({len(group.problems)} problems)...")
        group = train_on_group(
            group,
            MODEL_PATH,
            OUTPUT_DIR,
            steps=config.hyperparameters["training_steps"],
            rank=config.hyperparameters["lora_rank"],
        )

    # ===================================================================
    # PART 4: Analyze Correlations
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 4: Correlation Analysis")
    logger.info("=" * 70)

    correlations = compute_correlations(groups)

    qp_corr = correlations.get('quality_perplexity_correlation', None)
    logger.info(f"Quality-Perplexity correlation: {qp_corr:.3f}" if qp_corr is not None else "Quality-Perplexity correlation: N/A")
    if correlations.get("perplexity_ratio_high_vs_low"):
        logger.info(f"Perplexity ratio (high/low quality): {correlations['perplexity_ratio_high_vs_low']:.3f}")

    # Check success criteria
    quality_r = correlations.get("quality_perplexity_correlation", 0)
    ppl_ratio = correlations.get("perplexity_ratio_high_vs_low")

    success = (
        quality_r < -0.3 and  # Negative correlation (higher quality = lower perplexity)
        (ppl_ratio is not None and ppl_ratio < 0.8)  # High quality has <80% of low quality perplexity
    )

    # ===================================================================
    # PART 5: Save Results
    # ===================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 5: Save Results")
    logger.info("=" * 70)

    duration = time.time() - start_time

    results = ExperimentResult(
        config=config,
        metrics={
            "n_problems": len(problems),
            "n_groups": len(groups),
            **correlations,
            "success": success,
        },
        raw_data={
            "groups": [
                {
                    "name": g.name,
                    "n_problems": len(g.problems),
                    "mean_quality_score": g.mean_quality_score,
                    "mean_fisher": g.mean_fisher,
                    "mean_barrier": g.mean_barrier,
                    "final_loss": g.final_loss,
                    "final_perplexity": g.final_perplexity,
                }
                for g in groups
            ],
        },
        duration_seconds=duration,
        success=success,
    )

    results.save(OUTPUT_DIR / "results.json")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 17 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Success: {success}")
    logger.info(f"Results saved to: {OUTPUT_DIR / 'results.json'}")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    run_experiment()
