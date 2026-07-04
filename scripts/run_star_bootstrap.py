#!/usr/bin/env python3
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

"""STaR-style self-bootstrap experiment on paired reasoning data.

Architecture:
  - CE for training (Arm A from run_ablation: no active constraints)
  - Generation for exploration (adapter generates deterministic trajectories)
  - Geometry for navigation (reuse eval metrics: geodesic tail, repetition, CKA)

Round loop:
  1. Train adapter on current dataset with Arm A.
  2. Evaluate inference/repetition/geodesic.
  3. Generate answers on bootstrap pool (held-out by default).
  4. Keep only verified-correct generations.
  5. Append kept samples to training set for next round.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from novel_problems import (  # noqa: E402
    NovelProblem,
    generate_novel_problems,
    novel_problem_to_training_sample,
)
from run_ablation import (  # noqa: E402
    ARM_CONFIGS,
    DEFAULT_MODEL,
    DEFAULT_TRAIN,
    DEFAULT_VAL,
    compute_4gram_repetition_rate,
    eval_arm_seed,
    eval_baseline,
    train_arm_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_line(text: str) -> str:
    """Normalize line-level text for exact comparison."""
    text = text.strip().lower()
    # Collapse consecutive whitespace while preserving token order.
    return re.sub(r"\s+", " ", text)


def split_prompt_answer(sample: dict) -> tuple[str, str]:
    """Split paired sample into prompt (without answer) and expected answer line."""
    text = str(sample.get("text", ""))
    answer = str(sample.get("answer_start", "")).strip()

    if answer and answer in text:
        idx = text.index(answer)
        prompt = text[:idx].rstrip()
    else:
        prompt = text.rstrip()
    return prompt, answer


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def verify_generation(
    response: str,
    expected: str,
    mode: str,
    novel_problem: NovelProblem | None = None,
) -> tuple[bool, str]:
    """Deterministically verify generated output against expected answer line.

    Args:
        response: Model-generated text.
        expected: Expected answer string.
        mode: One of exact_first, first_contains_expected, contains_expected,
              or structural (delegates to novel_problem.verify).
        novel_problem: Required when mode is "structural".

    Returns:
        (is_verified, canonical_answer_line)
    """
    expected_line = expected.strip()

    if mode == "structural":
        if novel_problem is None:
            raise ValueError("structural mode requires a NovelProblem instance")
        is_verified = novel_problem.verify(response)
        if not is_verified:
            return False, ""
        return True, expected_line

    expected_norm = _normalize_line(expected_line)
    response_norm = _normalize_line(response)
    first_line_norm = _normalize_line(first_nonempty_line(response))

    if mode == "exact_first":
        is_verified = first_line_norm == expected_norm
    elif mode == "first_contains_expected":
        is_verified = expected_norm in first_line_norm
    elif mode == "contains_expected":
        is_verified = expected_norm in response_norm
    else:
        raise ValueError(
            f"Unknown verify mode: {mode}. "
            "Use one of: exact_first, first_contains_expected, "
            "contains_expected, structural."
        )

    if not is_verified:
        return False, ""
    return True, expected_line


def build_star_sample(
    source_sample: dict,
    prompt: str,
    generated_line: str,
    round_idx: int,
) -> dict:
    """Construct a paired-format training sample from verified generation."""
    prompt = prompt.rstrip()
    text = f"{prompt}\n{generated_line}" if prompt else generated_line
    return {
        "text": text,
        "answer_start": generated_line,
        "logic_id": source_sample.get("logic_id", ""),
        "template_id": source_sample.get("template_id", ""),
        "pair_type": f"star_round_{round_idx}",
        "source_pair_type": source_sample.get("pair_type", ""),
        "source_text": source_sample.get("text", ""),
    }


@dataclass
class RoundSummary:
    round_idx: int
    train_data_path: str
    adapter_path: str
    inference_correct: int
    inference_total: int
    repetition_rate: float
    geodesic_tail_mean_delta: float | None
    prompts_scanned: int
    verified_kept: int
    verified_rate: float
    augmented_train_size: int


def generate_verified_star_samples(
    model_path: str,
    adapter_path: str,
    pool_samples: list[dict],
    round_idx: int,
    max_prompts: int,
    max_tokens: int,
    verify_mode: str,
    max_repetition_rate: float | None = None,
) -> tuple[list[dict], int]:
    """Generate answers (greedy/deterministic) and keep only verified samples."""
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(model_path, adapter_path=adapter_path)

    kept: list[dict] = []
    repetition_rejected = 0
    prompts = pool_samples if max_prompts <= 0 else pool_samples[:max_prompts]

    logger.info("  Generation (greedy): prompts=%d", len(prompts))

    for sample in prompts:
        prompt, expected = split_prompt_answer(sample)
        if not prompt or not expected:
            continue

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        verified, canonical_line = verify_generation(
            response=response,
            expected=expected,
            mode=verify_mode,
        )
        if verified:
            if max_repetition_rate is not None:
                response_rep = compute_4gram_repetition_rate(response)
                if response_rep > max_repetition_rate:
                    repetition_rejected += 1
                    continue
            kept.append(build_star_sample(sample, prompt, canonical_line, round_idx))

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    if max_repetition_rate is not None:
        logger.info(
            "  Repetition gate (fixed pool): rejected=%d threshold=%.4f",
            repetition_rejected,
            max_repetition_rate,
        )

    return kept, len(prompts)


def generate_verified_novel_samples(
    model_path: str,
    adapter_path: str,
    novel_problems: list[NovelProblem],
    round_idx: int,
    max_tokens: int,
    max_repetition_rate: float | None = None,
) -> tuple[list[dict], dict[str, tuple[int, int]], list[NovelProblem]]:
    """Generate answers (greedy/deterministic) on novel problems with structural verification.

    Returns:
        (kept_samples, stats_by_form, failed_problems) where stats_by_form maps
        logic_form -> (verified_count, total_count) and failed_problems are
        problems the model got wrong (candidates for rationalization).
    """
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(model_path, adapter_path=adapter_path)

    kept: list[dict] = []
    repetition_rejected = 0
    failed: list[NovelProblem] = []
    stats: dict[str, tuple[int, int]] = {}

    logger.info("  Novel generation (greedy): problems=%d", len(novel_problems))

    for problem in novel_problems:
        v_count, t_count = stats.get(problem.logic, (0, 0))
        t_count += 1

        response = generate(
            model,
            tokenizer,
            prompt=problem.prompt,
            max_tokens=max_tokens,
        )
        is_verified, canonical = verify_generation(
            response=response,
            expected=problem.expected_answer,
            mode="structural",
            novel_problem=problem,
        )
        if is_verified:
            if max_repetition_rate is not None:
                response_rep = compute_4gram_repetition_rate(response)
                if response_rep > max_repetition_rate:
                    repetition_rejected += 1
                    is_verified = False
            if is_verified:
                kept.append(
                    novel_problem_to_training_sample(problem, canonical, round_idx)
                )
                v_count += 1

        if not is_verified:
            failed.append(problem)
        stats[problem.logic] = (v_count, t_count)

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    if max_repetition_rate is not None:
        logger.info(
            "  Repetition gate (novel): rejected=%d threshold=%.4f",
            repetition_rejected,
            max_repetition_rate,
        )

    return kept, stats, failed


def dedup_by_text_answer(rows: list[dict]) -> list[dict]:
    """Deterministic dedup by (text, answer_start)."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("text", "")), str(row.get("answer_start", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _sample_key(row: dict) -> tuple[str, str]:
    logic_id = str(row.get("logic_id", "")).strip()
    template_id = str(row.get("template_id", "")).strip()
    if logic_id and template_id:
        return logic_id, template_id
    # Fallback for datasets without explicit pairing metadata.
    return str(row.get("text", "")).strip(), str(row.get("answer_start", "")).strip()


def select_bootstrap_pool_for_round(
    pool_samples: list[dict],
    current_train_samples: list[dict],
    max_prompts: int,
    round_idx: int,
) -> list[dict]:
    """Select unsolved bootstrap prompts for the current round.

    Selection is deterministic and rotates through unresolved prompts so each
    round explores a different pool slice when max_prompts is capped.
    """
    seen_keys = {_sample_key(row) for row in current_train_samples}
    unresolved = [row for row in pool_samples if _sample_key(row) not in seen_keys]

    if max_prompts <= 0 or len(unresolved) <= max_prompts:
        return unresolved

    start = ((round_idx - 1) * max_prompts) % len(unresolved)
    end = start + max_prompts
    if end <= len(unresolved):
        return unresolved[start:end]
    return unresolved[start:] + unresolved[: end - len(unresolved)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STaR-style bootstrap rounds")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model path")
    parser.add_argument("--train-data", default=DEFAULT_TRAIN, help="Initial train JSONL")
    parser.add_argument("--val-data", default=DEFAULT_VAL, help="Validation JSONL")
    parser.add_argument(
        "--bootstrap-data",
        default=None,
        help="Held-out pool for generation (default: --val-data)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Training/eval seed")
    parser.add_argument("--rounds", type=int, default=3, help="Number of bootstrap rounds")
    parser.add_argument("--max-prompts", type=int, default=0, help="Pool prompts per round (0=all)")
    parser.add_argument("--max-tokens", type=int, default=96, help="Max generation tokens")
    parser.add_argument(
        "--rep-filter",
        action="store_true",
        help=(
            "Apply repetition acceptance gate to verified generations. "
            "Threshold is data-derived each round as min(baseline_rep, adapted_rep)."
        ),
    )
    parser.add_argument(
        "--verify-mode",
        choices=["exact_first", "first_contains_expected", "contains_expected"],
        default="first_contains_expected",
        help=(
            "Verification rule for keeping generated samples. "
            "Default accepts expected answer when present on first non-empty line."
        ),
    )
    parser.add_argument("--novel", action="store_true", help="Use novel problem generator instead of fixed pool")
    parser.add_argument("--novel-count", type=int, default=100, help="Novel problems per round (default 100)")
    parser.add_argument("--novel-domains", nargs="*", default=None, help="Restrict novel problems to these domains")
    parser.add_argument(
        "--rationalize",
        nargs="*",
        default=None,
        help=(
            "Teach failed logic forms by injecting correct answers as training samples. "
            "List forms to rationalize (e.g. --rationalize modus_tollens chain_contrapositive), "
            "or use --rationalize with no args to rationalize all failed forms."
        ),
    )
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip lm-eval benchmarks")
    parser.add_argument(
        "--output",
        default=f"/Volumes/CodeCypher/experiments/star-350m-{time.strftime('%Y%m%d-%H%M%S')}",
        help="Output directory",
    )
    args = parser.parse_args()

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    current_train_path = Path(args.train_data).expanduser().resolve()
    val_path = Path(args.val_data).expanduser().resolve()
    bootstrap_path = (
        Path(args.bootstrap_data).expanduser().resolve()
        if args.bootstrap_data
        else val_path
    )

    base_train_samples = load_jsonl(current_train_path)

    use_novel = args.novel
    pool_samples: list[dict] = []
    if not use_novel:
        pool_samples = load_jsonl(bootstrap_path)
        if not pool_samples:
            raise ValueError(f"Empty bootstrap dataset: {bootstrap_path}")

    if not base_train_samples:
        raise ValueError(f"Empty train dataset: {current_train_path}")

    config = {
        "model": args.model,
        "seed": args.seed,
        "rounds": args.rounds,
        "initial_train_data": str(current_train_path),
        "val_data": str(val_path),
        "bootstrap_data": str(bootstrap_path) if not use_novel else "novel_generator",
        "novel": use_novel,
        "novel_count": args.novel_count if use_novel else 0,
        "novel_domains": args.novel_domains,
        "rationalize": args.rationalize,
        "decoding": "greedy_deterministic",
        "max_prompts": args.max_prompts,
        "max_tokens": args.max_tokens,
        "rep_filter": args.rep_filter,
        "verify_mode": args.verify_mode if not use_novel else "structural",
        "skip_benchmarks": args.skip_benchmarks,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }
    with open(output_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    round_summaries: list[RoundSummary] = []
    current_samples = list(base_train_samples)

    for round_idx in range(1, args.rounds + 1):
        logger.info("=" * 70)
        logger.info("ROUND %d/%d", round_idx, args.rounds)
        logger.info("=" * 70)

        round_dir = output_root / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        train_path = round_dir / "train_input.jsonl"
        save_jsonl(train_path, current_samples)

        # Train Arm A (answer-only CE, no active constraints)
        train_arm_seed(
            arm_config=ARM_CONFIGS["A"],
            seed=args.seed,
            model_path=args.model,
            train_path=str(train_path),
            val_path=str(val_path),
            output_root=str(round_dir),
        )

        # Evaluate baseline + adapter for this round
        baseline = eval_baseline(
            model_path=args.model,
            output_root=str(round_dir),
            skip_benchmarks=args.skip_benchmarks,
        )
        eval_result = eval_arm_seed(
            arm="A",
            seed=args.seed,
            model_path=args.model,
            output_root=str(round_dir),
            skip_benchmarks=args.skip_benchmarks,
        )

        adapter_path = round_dir / f"arm_A_seed_{args.seed}" / "adapter"

        rep_threshold: float | None = None
        if args.rep_filter:
            rep_candidates: list[float] = []
            for src in (baseline, eval_result):
                rep_raw = src.get("repetition_rate")
                if isinstance(rep_raw, (int, float)) and math.isfinite(float(rep_raw)):
                    rep_candidates.append(float(rep_raw))
            if not rep_candidates:
                raise ValueError(
                    "Repetition filter enabled but no repetition_rate available from eval."
                )
            rep_threshold = min(rep_candidates)
            logger.info(
                "  Repetition gate active: threshold=%.4f (from %s)",
                rep_threshold,
                ", ".join(f"{v:.4f}" for v in rep_candidates),
            )

        # STaR generation: novel problems or held-out pool
        novel_stats: dict[str, tuple[int, int]] | None = None
        rationalized_count = 0

        if use_novel:
            # Fresh novel problems each round (different seed per round)
            round_seed = args.seed + round_idx * 1000
            novel_problems = generate_novel_problems(
                n=args.novel_count,
                seed=round_seed,
                domains=args.novel_domains,
            )
            logger.info(
                "  Novel pool for round %d: %d problems (seed=%d)",
                round_idx,
                len(novel_problems),
                round_seed,
            )

            star_samples, novel_stats, failed_problems = generate_verified_novel_samples(
                model_path=args.model,
                adapter_path=str(adapter_path),
                novel_problems=novel_problems,
                round_idx=round_idx,
                max_tokens=args.max_tokens,
                max_repetition_rate=rep_threshold,
            )
            prompts_scanned = len(novel_problems)

            # Log per-form verification rates
            logger.info("  Novel verification by form:")
            for form, (v, t) in sorted(novel_stats.items()):
                logger.info("    %s: %d/%d (%.0f%%)", form, v, t, 100.0 * v / max(1, t))

            # Rationalization: teach failed forms by injecting correct answers
            rationalized_count = 0
            if args.rationalize is not None and failed_problems:
                # Empty list = rationalize all; non-empty = only listed forms
                rationalize_forms = set(args.rationalize) if args.rationalize else None
                for problem in failed_problems:
                    if rationalize_forms is None or problem.logic in rationalize_forms:
                        sample = novel_problem_to_training_sample(
                            problem, problem.expected_answer, round_idx,
                        )
                        sample["pair_type"] = f"star_rationalized_round_{round_idx}"
                        star_samples.append(sample)
                        rationalized_count += 1
                logger.info(
                    "  Rationalized %d failed problems (forms: %s)",
                    rationalized_count,
                    "all" if rationalize_forms is None else ", ".join(sorted(rationalize_forms)),
                )
        else:
            current_keys = {_sample_key(row) for row in current_samples}
            unresolved_count = sum(
                1 for sample in pool_samples if _sample_key(sample) not in current_keys
            )
            round_pool = select_bootstrap_pool_for_round(
                pool_samples=pool_samples,
                current_train_samples=current_samples,
                max_prompts=args.max_prompts,
                round_idx=round_idx,
            )
            logger.info(
                "  Bootstrap pool for round %d: unresolved=%d selected=%d",
                round_idx,
                unresolved_count,
                len(round_pool),
            )

            star_samples, prompts_scanned = generate_verified_star_samples(
                model_path=args.model,
                adapter_path=str(adapter_path),
                pool_samples=round_pool,
                round_idx=round_idx,
                max_prompts=0,
                max_tokens=args.max_tokens,
                verify_mode=args.verify_mode,
                max_repetition_rate=rep_threshold,
            )

        star_path = round_dir / "star_verified.jsonl"
        save_jsonl(star_path, star_samples)

        # Augment dataset for next round
        augmented = dedup_by_text_answer(current_samples + star_samples)
        next_train_path = round_dir / "train_next.jsonl"
        save_jsonl(next_train_path, augmented)

        verified_rate = len(star_samples) / max(1, prompts_scanned)
        summary = RoundSummary(
            round_idx=round_idx,
            train_data_path=str(train_path),
            adapter_path=str(adapter_path),
            inference_correct=int(eval_result.get("inference_correct", 0)),
            inference_total=int(eval_result.get("inference_total", 0)),
            repetition_rate=float(eval_result.get("repetition_rate", 0.0)),
            geodesic_tail_mean_delta=(
                float(eval_result["geodesic_tail_mean_delta"])
                if eval_result.get("geodesic_tail_mean_delta") is not None
                else None
            ),
            prompts_scanned=prompts_scanned,
            verified_kept=len(star_samples),
            verified_rate=verified_rate,
            augmented_train_size=len(augmented),
        )
        round_summaries.append(summary)

        round_data: dict = {
            "round": asdict(summary),
            "baseline": baseline,
            "eval": eval_result,
        }
        if novel_stats is not None:
            round_data["novel_stats"] = {
                form: {"verified": v, "total": t}
                for form, (v, t) in novel_stats.items()
            }
            round_data["rationalized_count"] = rationalized_count
        with open(round_dir / "round_summary.json", "w") as f:
            json.dump(round_data, f, indent=2, default=str)

        logger.info(
            "  Round %d summary: correct=%d/%d rep=%.3f verified=%d/%d (%.1f%%) "
            "train_size=%d",
            round_idx,
            summary.inference_correct,
            summary.inference_total,
            summary.repetition_rate,
            summary.verified_kept,
            summary.prompts_scanned,
            100.0 * summary.verified_rate,
            summary.augmented_train_size,
        )

        # Stop if we failed to discover any verified samples.
        if summary.verified_kept == 0:
            logger.warning(
                "No verified samples found in round %d. Stopping early.",
                round_idx,
            )
            break

        current_samples = augmented

    result = {
        "config": config,
        "rounds": [asdict(r) for r in round_summaries],
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("STaR BOOTSTRAP SUMMARY")
    print("=" * 70)
    for r in round_summaries:
        print(
            f"Round {r.round_idx}: "
            f"inference={r.inference_correct}/{r.inference_total}, "
            f"rep={r.repetition_rate:.3f}, "
            f"verified={r.verified_kept}/{r.prompts_scanned} ({100.0*r.verified_rate:.1f}%), "
            f"train_size={r.augmented_train_size}"
        )
    print("=" * 70)
    logger.info("Results in: %s", output_root)


if __name__ == "__main__":
    main()
