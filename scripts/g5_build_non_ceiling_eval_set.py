#!/usr/bin/env python3
"""Build a fixed online-eval problem set with non-ceiling baseline accuracy.

This script evaluates a candidate StarProblem pool on a reference model and
extracts a deterministic subset whose baseline is non-ceiling.

Default targeting is derived from binomial geometry:
- confidence level uses alpha = 1/N (same as regime selection)
- ceiling criterion uses headroom_upper = ci_upper - observed_accuracy
- non-ceiling requires headroom_upper > 1/N
- among non-ceiling counts, choose p=k/N nearest 0.5 because binomial variance
  p(1-p) is maximized at p=0.5 (maximum measurement sensitivity)

Output format is directly consumable by:
  scripts/g5_8b_validation.py --online-eval-problems-json <output>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain.statistics import clopper_pearson_interval
from modelcypher.core.domain.training.online_eval import (
    create_eval_problem_set,
    evaluate_correctness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("g5_build_non_ceiling_eval_set")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed online-eval StarProblem set with target baseline band.",
    )
    parser.add_argument(
        "--model-path",
        default="/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        help="Reference model used to measure baseline correctness.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/g5_8b_validation/non_ceiling_eval_set.json"),
        help="Output JSON file with selected problem records.",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=25,
        help="Size of final online-eval problem set.",
    )
    parser.add_argument(
        "--band-low",
        type=float,
        default=None,
        help=(
            "Optional lower bound of baseline accuracy band. "
            "Only valid when --band-high is also set."
        ),
    )
    parser.add_argument(
        "--band-high",
        type=float,
        default=None,
        help=(
            "Optional upper bound of baseline accuracy band. "
            "Only valid when --band-low is also set."
        ),
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=41,
        help="Seed for candidate StarProblem generation.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=250,
        help=(
            "Candidate pool size to search for feasible correct/incorrect split. "
            "Compute budget control, not a decision boundary."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Generation max_tokens used during correctness evaluation.",
    )
    return parser.parse_args()


def _choose_target_correct_count(
    *,
    n_total: int,
    band_low: float | None,
    band_high: float | None,
    n_correct_available: int,
    n_incorrect_available: int,
) -> tuple[int, dict[str, Any]]:
    if (band_low is None) != (band_high is None):
        raise ValueError("--band-low and --band-high must be provided together")

    alpha = 1.0 / float(n_total)
    low_count = 0
    high_count = n_total
    selection_mode = "auto_non_ceiling_max_variance"
    if band_low is not None and band_high is not None:
        if not (0.0 <= band_low <= band_high <= 1.0):
            raise ValueError(
                "band must satisfy 0 <= low <= high <= 1, "
                f"got low={band_low}, high={band_high}",
            )
        low_count = int(math.ceil(band_low * n_total))
        high_count = int(math.floor(band_high * n_total))
        if low_count > high_count:
            raise ValueError(
                f"Requested band [{band_low}, {band_high}] has no integer support for N={n_total}",
            )
        selection_mode = "user_band"

    feasible: list[int] = []
    feasible_non_ceiling: list[tuple[int, float]] = []
    for k in range(low_count, high_count + 1):
        need_incorrect = n_total - k
        if n_correct_available < k or n_incorrect_available < need_incorrect:
            continue
        feasible.append(k)

        acc = float(k) / float(n_total)
        _ci_lower, ci_upper = clopper_pearson_interval(
            n_correct=k,
            n_total=n_total,
            alpha=alpha,
        )
        headroom_upper = ci_upper - acc
        if headroom_upper > alpha:
            feasible_non_ceiling.append((k, acc))

    if not feasible:
        raise ValueError(
            "No feasible target accuracy in the requested count range with current candidate pool. "
            f"Need counts in [{low_count}, {high_count}] correct out of N={n_total}, "
            f"but only have correct={n_correct_available}, incorrect={n_incorrect_available}.",
        )

    if not feasible_non_ceiling:
        raise ValueError(
            "No feasible non-ceiling target in requested range under "
            "headroom_upper > 1/N criterion.",
        )

    if selection_mode == "user_band":
        midpoint = (low_count + high_count) / 2.0
        target_k = min(
            [k for (k, _acc) in feasible_non_ceiling],
            key=lambda k: (abs(k - midpoint), k),
        )
    else:
        # Binomial variance theorem: Var[X/N] = p(1-p)/N is maximal at p=0.5.
        target_k = min(
            feasible_non_ceiling,
            key=lambda item: (abs(item[1] - 0.5), item[0]),
        )[0]

    target_acc = float(target_k) / float(n_total)
    _target_ci_lower, target_ci_upper = clopper_pearson_interval(
        n_correct=target_k,
        n_total=n_total,
        alpha=alpha,
    )
    return target_k, {
        "mode": selection_mode,
        "alpha": alpha,
        "headroom_resolution": alpha,
        "target_accuracy": target_acc,
        "target_headroom_upper": target_ci_upper - target_acc,
        "feasible_count_range": [low_count, high_count],
        "n_feasible_total": len(feasible),
        "n_feasible_non_ceiling": len(feasible_non_ceiling),
    }


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if args.n_problems <= 1:
        raise ValueError("--n-problems must be > 1")
    if args.candidate_pool_size < args.n_problems:
        raise ValueError("--candidate-pool-size must be >= --n-problems")

    backend = initialize_default_backend()
    model, tokenizer = backend.load_model(str(model_path))

    candidate_problems = create_eval_problem_set(
        n_problems=args.candidate_pool_size,
        seed=args.candidate_seed,
    )
    logger.info(
        "Generated candidate pool: n=%d seed=%d",
        len(candidate_problems),
        args.candidate_seed,
    )

    def _generate(prompt: str, max_tokens: int) -> str:
        return backend.generate(model, tokenizer, prompt, max_tokens=max_tokens)

    candidate_eval = evaluate_correctness(
        problems=candidate_problems,
        generate_fn=_generate,
        epoch=0,
        baseline_correct_ids=None,
        max_tokens=args.max_tokens,
    )
    correct_ids = set(candidate_eval.correct_ids)
    correct_problems: list[Any] = []
    incorrect_problems: list[Any] = []
    for problem in candidate_problems:
        if problem.problem_id in correct_ids:
            correct_problems.append(problem)
        else:
            incorrect_problems.append(problem)

    target_correct, selection_meta = _choose_target_correct_count(
        n_total=args.n_problems,
        band_low=args.band_low,
        band_high=args.band_high,
        n_correct_available=len(correct_problems),
        n_incorrect_available=len(incorrect_problems),
    )
    target_incorrect = args.n_problems - target_correct

    # Deterministic selection: stable ordering by problem_id.
    selected_correct = sorted(correct_problems, key=lambda p: p.problem_id)[:target_correct]
    selected_incorrect = sorted(incorrect_problems, key=lambda p: p.problem_id)[:target_incorrect]
    selected = sorted(
        [*selected_correct, *selected_incorrect],
        key=lambda p: p.problem_id,
    )

    selected_eval = evaluate_correctness(
        problems=selected,
        generate_fn=_generate,
        epoch=0,
        baseline_correct_ids=None,
        max_tokens=args.max_tokens,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "selection": {
            "n_problems": args.n_problems,
            "band_low": args.band_low,
            "band_high": args.band_high,
            "candidate_seed": args.candidate_seed,
            "candidate_pool_size": args.candidate_pool_size,
            "target_correct": target_correct,
            "target_incorrect": target_incorrect,
            "derived": selection_meta,
        },
        "candidate_baseline": candidate_eval.to_dict(),
        "selected_baseline": selected_eval.to_dict(),
        "problems": [problem.to_problem_record() for problem in selected],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    logger.info(
        "Selected baseline: %d/%d (%.1f%%)",
        selected_eval.n_correct,
        selected_eval.n_total,
        selected_eval.accuracy * 100.0,
    )


if __name__ == "__main__":
    main()
