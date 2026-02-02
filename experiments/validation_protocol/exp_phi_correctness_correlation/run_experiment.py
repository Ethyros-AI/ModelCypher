#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment: comp/phi Correlation with Reasoning Quality
#
# HYPOTHESIS: comp/phi correlates with reasoning quality
#             (correct answers have different comp/phi distribution than incorrect)
#
# CRITICAL NOTE: This tests CORRELATION, not CAUSATION.
#                We're measuring what emerges, not imposing targets.
#
# PROTOCOL:
# 1. Collect 1000+ prompts with ground-truth answers:
#    - 200 simple factual (What is 2+2?)
#    - 200 CRT problems (bat and ball)
#    - 200 math reasoning (multi-step)
#    - 200 logic puzzles
#    - 200 creative (open-ended - excluded from correctness analysis)
# 2. For each prompt:
#    - Measure comp/phi during generation
#    - Check correctness (substring match + human validation on ambiguous)
# 3. Compute:
#    - Pearson correlation (comp/phi vs correctness)
#    - Mann-Whitney U test (comp/phi distribution: correct vs incorrect)
#    - ROC-AUC for comp/phi as correctness predictor
#
# SUCCESS CRITERIA:
# - Correlation r > 0.3 (weak but significant effect)
# - Mann-Whitney p < 0.05
# - ROC-AUC > 0.6
#
# NULL HYPOTHESIS: comp/phi is independent of correctness (r = 0, AUC = 0.5)
#
# IMPORTANT: If the experiment fails, that's a VALID RESULT.
# It means comp/phi is not predictive of correctness, which is scientifically valuable.

from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PromptWithAnswer:
    """A prompt with its expected answer and category."""
    prompt: str
    expected: str | list[str]  # May have multiple valid answers
    category: str
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class ExpansionMeasurement:
    """Single expansion_ratio measurement with correctness label."""
    prompt: str
    category: str
    expansion_ratio: float
    compression_ratio: float
    peak_layer: int
    total_layers: int
    model_response: str
    expected_answer: str
    is_correct: bool | None  # None for open-ended


@dataclass
class CorrelationResult:
    """Statistical results for expansion_ratio vs correctness correlation."""
    pearson_r: float
    pearson_p_value: float
    mann_whitney_u: float
    mann_whitney_p_value: float
    roc_auc: float
    roc_auc_ci_lower: float
    roc_auc_ci_upper: float
    n_correct: int
    n_incorrect: int
    mean_ratio_correct: float
    mean_ratio_incorrect: float
    std_ratio_correct: float
    std_ratio_incorrect: float
    effect_size_cohens_d: float


# Prompt datasets with ground truth
FACTUAL_PROMPTS = [
    PromptWithAnswer("What is 2 + 2?", ["4", "four"], "factual", "easy"),
    PromptWithAnswer("What is the capital of France?", ["Paris"], "factual", "easy"),
    PromptWithAnswer("How many days are in a week?", ["7", "seven"], "factual", "easy"),
    PromptWithAnswer("What color is the sky on a clear day?", ["blue"], "factual", "easy"),
    PromptWithAnswer("What is 3 times 4?", ["12", "twelve"], "factual", "easy"),
    PromptWithAnswer("What planet do we live on?", ["Earth"], "factual", "easy"),
    PromptWithAnswer("How many legs does a dog have?", ["4", "four"], "factual", "easy"),
    PromptWithAnswer("What is 10 divided by 2?", ["5", "five"], "factual", "easy"),
    PromptWithAnswer("What is the first letter of the alphabet?", ["A", "a"], "factual", "easy"),
    PromptWithAnswer("What is ice made of?", ["water", "frozen water", "H2O"], "factual", "easy"),
]

CRT_PROMPTS = [
    # The classic 3 CRT problems - designed to trigger intuitive traps
    PromptWithAnswer(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        ["$0.05", "0.05", "5 cents", "five cents"],
        "crt",
        "hard"
    ),
    PromptWithAnswer(
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        ["5 minutes", "5", "five minutes"],
        "crt",
        "hard"
    ),
    PromptWithAnswer(
        "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
        ["47 days", "47", "forty-seven days"],
        "crt",
        "hard"
    ),
    # Additional reasoning problems
    PromptWithAnswer(
        "A farmer has 17 sheep. All but 9 die. How many are left?",
        ["9", "nine"],
        "crt",
        "medium"
    ),
    PromptWithAnswer(
        "How many times can you subtract 5 from 25?",
        ["1", "one", "once"],  # After first subtraction it's 20, not 25
        "crt",
        "medium"
    ),
]

MATH_REASONING_PROMPTS = [
    PromptWithAnswer(
        "If Tom has 3 times as many apples as Jane, and Jane has 5 apples, how many does Tom have?",
        ["15", "fifteen"],
        "math_reasoning",
        "medium"
    ),
    PromptWithAnswer(
        "A train travels at 60 mph for 2 hours. How far does it go?",
        ["120 miles", "120"],
        "math_reasoning",
        "easy"
    ),
    PromptWithAnswer(
        "If I have 3 red balls and 2 blue balls, what fraction of the balls are red?",
        ["3/5", "0.6", "60%", "three fifths"],
        "math_reasoning",
        "medium"
    ),
    PromptWithAnswer(
        "A shirt costs $20 and is on sale for 25% off. What is the sale price?",
        ["$15", "15 dollars", "15"],
        "math_reasoning",
        "medium"
    ),
    PromptWithAnswer(
        "If you read 20 pages per hour and the book has 100 pages, how many hours to finish?",
        ["5", "five", "5 hours"],
        "math_reasoning",
        "easy"
    ),
]

LOGIC_PROMPTS = [
    PromptWithAnswer(
        "All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded?",
        ["yes", "true", "correct"],
        "logic",
        "easy"
    ),
    PromptWithAnswer(
        "Some fruits are red. Apples are fruits. Are all apples red?",
        ["no", "not necessarily", "false", "not all"],
        "logic",
        "medium"
    ),
    PromptWithAnswer(
        "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        ["not necessarily", "maybe", "can't tell", "unknown"],
        "logic",
        "hard"  # Affirming the consequent fallacy
    ),
    PromptWithAnswer(
        "All cats are animals. All animals need food. Do cats need food?",
        ["yes", "true", "correct"],
        "logic",
        "easy"
    ),
]


def get_all_prompts() -> list[PromptWithAnswer]:
    """Get all prompts with ground truth answers."""
    all_prompts = []
    all_prompts.extend(FACTUAL_PROMPTS)
    all_prompts.extend(CRT_PROMPTS)
    all_prompts.extend(MATH_REASONING_PROMPTS)
    all_prompts.extend(LOGIC_PROMPTS)
    return all_prompts


def compute_expansion_ratio(
    model,
    tokenizer,
    prompt: str,
) -> tuple[float, float, int, int]:
    """Compute expansion_ratio for a prompt.

    Returns:
        (expansion_ratio, compression_ratio, peak_layer, total_layers)
    """
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get model's layer structure
    base_model = getattr(model, "model", model)

    # Forward through embedding
    hidden = base_model.embed_tokens(input_ids)
    mx.eval(hidden)
    initial_norm = float(mx.sqrt(mx.sum(hidden * hidden)))

    peak_norm = initial_norm
    peak_layer = 0
    norms = [initial_norm]

    # Forward through each layer
    for i, layer in enumerate(base_model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        norms.append(norm)
        if norm > peak_norm:
            peak_norm = norm
            peak_layer = i + 1

    final_norm = norms[-1]
    total_layers = len(base_model.layers)

    # Compute expansion_ratio with dtype-derived epsilon
    eps = math.sqrt(float(mx.finfo(mx.float32).eps))
    compression_ratio = peak_norm / final_norm if final_norm > eps else 1.0
    expansion_ratio = peak_norm / initial_norm if initial_norm > eps else 1.0

    return expansion_ratio, compression_ratio, peak_layer, total_layers


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
) -> str:
    """Generate a response from the model."""
    try:
        from mlx_lm import generate

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        # Extract just the generated part
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
    except Exception as e:
        logger.warning(f"Generation failed: {e}")
        return ""


def check_correctness(
    response: str,
    expected: str | list[str],
) -> bool:
    """Check if response contains expected answer.

    Uses case-insensitive substring matching.
    """
    response_lower = response.lower()
    if isinstance(expected, str):
        expected = [expected]

    for exp in expected:
        if exp.lower() in response_lower:
            return True
    return False


def compute_statistics(
    measurements: list[ExpansionMeasurement],
) -> CorrelationResult:
    """Compute correlation statistics.

    Uses scipy for statistical tests, falls back to simple implementations.
    """
    # Filter to only measurements with correctness labels
    labeled = [m for m in measurements if m.is_correct is not None]

    correct = [m.expansion_ratio for m in labeled if m.is_correct]
    incorrect = [m.expansion_ratio for m in labeled if not m.is_correct]

    if len(correct) < 5 or len(incorrect) < 5:
        logger.warning("Insufficient samples for statistical analysis")
        return CorrelationResult(
            pearson_r=0.0,
            pearson_p_value=1.0,
            mann_whitney_u=0.0,
            mann_whitney_p_value=1.0,
            roc_auc=0.5,
            roc_auc_ci_lower=0.5,
            roc_auc_ci_upper=0.5,
            n_correct=len(correct),
            n_incorrect=len(incorrect),
            mean_ratio_correct=0.0,
            mean_ratio_incorrect=0.0,
            std_ratio_correct=0.0,
            std_ratio_incorrect=0.0,
            effect_size_cohens_d=0.0,
        )

    import statistics

    mean_correct = statistics.mean(correct)
    mean_incorrect = statistics.mean(incorrect)
    std_correct = statistics.stdev(correct) if len(correct) > 1 else 0.01
    std_incorrect = statistics.stdev(incorrect) if len(incorrect) > 1 else 0.01

    # Pooled standard deviation for Cohen's d
    pooled_std = math.sqrt(
        ((len(correct) - 1) * std_correct**2 + (len(incorrect) - 1) * std_incorrect**2) /
        (len(correct) + len(incorrect) - 2)
    )
    cohens_d = (mean_correct - mean_incorrect) / pooled_std if pooled_std > 0 else 0.0

    # Try to use scipy for proper statistical tests
    try:
        from scipy import stats

        # Create binary correctness labels and ratio values
        y_true = [1] * len(correct) + [0] * len(incorrect)
        y_scores = correct + incorrect

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(y_true, y_scores)

        # Mann-Whitney U test
        u_stat, mw_p = stats.mannwhitneyu(correct, incorrect, alternative='two-sided')

        # ROC-AUC
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_scores)

        # Bootstrap CI for AUC
        import random
        bootstrap_aucs = []
        for _ in range(1000):
            indices = random.choices(range(len(y_true)), k=len(y_true))
            y_boot = [y_true[i] for i in indices]
            s_boot = [y_scores[i] for i in indices]
            if len(set(y_boot)) > 1:  # Need both classes
                bootstrap_aucs.append(roc_auc_score(y_boot, s_boot))
        bootstrap_aucs.sort()
        ci_lower = bootstrap_aucs[25] if bootstrap_aucs else 0.5
        ci_upper = bootstrap_aucs[975] if len(bootstrap_aucs) > 975 else 0.5

    except ImportError:
        logger.warning("scipy/sklearn not available, using simple approximations")
        # Simple approximations
        pearson_r = cohens_d / math.sqrt(cohens_d**2 + 4) if abs(cohens_d) > 0 else 0.0
        pearson_p = 0.05 if abs(pearson_r) > 0.2 else 0.5  # Very rough
        u_stat = len(correct) * len(incorrect) / 2
        mw_p = 0.5
        roc_auc = 0.5 + cohens_d / 4  # Rough approximation
        roc_auc = max(0.0, min(1.0, roc_auc))
        ci_lower = roc_auc - 0.1
        ci_upper = roc_auc + 0.1

    return CorrelationResult(
        pearson_r=pearson_r,
        pearson_p_value=pearson_p,
        mann_whitney_u=u_stat,
        mann_whitney_p_value=mw_p,
        roc_auc=roc_auc,
        roc_auc_ci_lower=ci_lower,
        roc_auc_ci_upper=ci_upper,
        n_correct=len(correct),
        n_incorrect=len(incorrect),
        mean_ratio_correct=mean_correct,
        mean_ratio_incorrect=mean_incorrect,
        std_ratio_correct=std_correct,
        std_ratio_incorrect=std_incorrect,
        effect_size_cohens_d=cohens_d,
    )


def run_experiment(
    model_path: str,
    n_prompts: int | None = None,
) -> dict[str, Any]:
    """Run the expansion_ratio-correctness correlation experiment.

    Args:
        model_path: Path to model to test.
        n_prompts: Optional limit on number of prompts.

    Returns:
        Dict with measurements, statistics, and interpretation.
    """
    from mlx_lm import load

    logger.info("=" * 60)
    logger.info("EXPANSION RATIO CORRECTNESS CORRELATION EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load(model_path)

    # Get prompts
    prompts = get_all_prompts()
    if n_prompts:
        prompts = prompts[:n_prompts]

    logger.info(f"Testing {len(prompts)} prompts")

    measurements: list[ExpansionMeasurement] = []

    for i, prompt_data in enumerate(prompts):
        try:
            # Compute expansion_ratio
            expansion_ratio, compression_ratio, peak_layer, total_layers = compute_expansion_ratio(
                model, tokenizer, prompt_data.prompt
            )

            # Generate response
            response = generate_response(model, tokenizer, prompt_data.prompt)

            # Check correctness
            is_correct = check_correctness(response, prompt_data.expected)

            measurement = ExpansionMeasurement(
                prompt=prompt_data.prompt[:50] + "..." if len(prompt_data.prompt) > 50 else prompt_data.prompt,
                category=prompt_data.category,
                expansion_ratio=expansion_ratio,
                compression_ratio=compression_ratio,
                peak_layer=peak_layer,
                total_layers=total_layers,
                model_response=response[:100],
                expected_answer=str(prompt_data.expected),
                is_correct=is_correct,
            )
            measurements.append(measurement)

            status = "CORRECT" if is_correct else "WRONG"
            logger.info(
                f"[{i+1}/{len(prompts)}] {prompt_data.category}: "
                f"expansion_ratio={expansion_ratio:.3f}, {status}"
            )

        except Exception as e:
            logger.warning(f"Failed on prompt {i+1}: {e}")

    # Compute statistics
    logger.info("\nComputing statistics...")
    stats = compute_statistics(measurements)

    # Interpret results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    logger.info(f"\nSample sizes: {stats.n_correct} correct, {stats.n_incorrect} incorrect")
    logger.info(f"Mean expansion_ratio (correct): {stats.mean_ratio_correct:.4f} +/- {stats.std_ratio_correct:.4f}")
    logger.info(f"Mean expansion_ratio (incorrect): {stats.mean_ratio_incorrect:.4f} +/- {stats.std_ratio_incorrect:.4f}")
    logger.info(f"\nPearson r: {stats.pearson_r:.4f} (p = {stats.pearson_p_value:.4f})")
    logger.info(f"Mann-Whitney p: {stats.mann_whitney_p_value:.4f}")
    logger.info(f"ROC-AUC: {stats.roc_auc:.4f} (95% CI: [{stats.roc_auc_ci_lower:.4f}, {stats.roc_auc_ci_upper:.4f}])")
    logger.info(f"Cohen's d: {stats.effect_size_cohens_d:.4f}")

    # Verdict
    logger.info("\n" + "=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)

    hypothesis_supported = (
        abs(stats.pearson_r) > 0.3 and
        stats.mann_whitney_p_value < 0.05 and
        stats.roc_auc > 0.6
    )

    if hypothesis_supported:
        logger.info("HYPOTHESIS SUPPORTED: expansion_ratio correlates with correctness")
        logger.info(f"  - Correlation r = {stats.pearson_r:.3f} (> 0.3 threshold)")
        logger.info(f"  - Mann-Whitney p = {stats.mann_whitney_p_value:.4f} (< 0.05)")
        logger.info(f"  - ROC-AUC = {stats.roc_auc:.3f} (> 0.6 threshold)")
    else:
        logger.info("HYPOTHESIS NOT SUPPORTED: expansion_ratio does not reliably correlate with correctness")
        if abs(stats.pearson_r) <= 0.3:
            logger.info(f"  - Correlation r = {stats.pearson_r:.3f} (failed > 0.3 threshold)")
        if stats.mann_whitney_p_value >= 0.05:
            logger.info(f"  - Mann-Whitney p = {stats.mann_whitney_p_value:.4f} (failed < 0.05)")
        if stats.roc_auc <= 0.6:
            logger.info(f"  - ROC-AUC = {stats.roc_auc:.3f} (failed > 0.6 threshold)")
        logger.info("\nThis is a VALID scientific result. expansion_ratio may measure processing")
        logger.info("characteristics that are not directly predictive of answer correctness.")

    return {
        "model_path": model_path,
        "n_prompts": len(prompts),
        "hypothesis_supported": hypothesis_supported,
        "statistics": asdict(stats),
        "measurements": [asdict(m) for m in measurements],
        "per_category_stats": _compute_per_category_stats(measurements),
    }


def _compute_per_category_stats(measurements: list[ExpansionMeasurement]) -> dict:
    """Compute statistics per category."""
    import statistics

    categories = set(m.category for m in measurements)
    result = {}

    for cat in categories:
        cat_measurements = [m for m in measurements if m.category == cat]
        labeled = [m for m in cat_measurements if m.is_correct is not None]

        if not labeled:
            continue

        correct = [m.expansion_ratio for m in labeled if m.is_correct]
        incorrect = [m.expansion_ratio for m in labeled if not m.is_correct]

        result[cat] = {
            "n_total": len(labeled),
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "accuracy": len(correct) / len(labeled) if labeled else 0.0,
            "mean_ratio_correct": statistics.mean(correct) if correct else None,
            "mean_ratio_incorrect": statistics.mean(incorrect) if incorrect else None,
        }

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test correlation between comp/phi and answer correctness"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model to test",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        help="Limit number of prompts (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file path",
    )

    args = parser.parse_args()

    results = run_experiment(
        model_path=args.model,
        n_prompts=args.n_prompts,
    )

    output_path = Path(__file__).parent / args.output

    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
