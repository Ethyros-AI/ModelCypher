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

"""350M ablation matrix for constrained geometric training.

4 arms x 3 seeds = 12 runs. Each arm isolates a constraint group:
  A: Answer-only CE (no constraints, all multipliers frozen at 0)
  B: Answer + invariance + separation (mu_geo frozen at 0)
  C: Answer + geodesic tail only (mu_inv, mu_sep frozen at 0)
  D: Full constrained (no multipliers frozen)

Decision by paired t-test CIs across seeds.

Usage:
    # Full ablation (train + eval + decision)
    poetry run python scripts/run_ablation.py

    # Quick smoke test
    poetry run python scripts/run_ablation.py --arms D --seeds 42 --skip-benchmarks

    # Eval only (skip training, use existing adapters)
    poetry run python scripts/run_ablation.py --eval-only

    # Decision only (compute from existing results)
    poetry run python scripts/run_ablation.py --decision-only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DEFAULT_TRAIN = str(REPO_ROOT / "data/training/paired_reasoning_train.jsonl")
DEFAULT_VAL = str(REPO_ROOT / "data/training/paired_reasoning_val.jsonl")
DEFAULT_OUTPUT = "/Volumes/CodeCypher/experiments/ablation-350m"
INFERENCE_PROMPTS = REPO_ROOT / "data" / "eval_prompts" / "nblora_inference_tests.jsonl"

BENCHMARK_TASKS = [
    "arc_easy", "arc_challenge", "hellaswag",
    "boolq", "piqa", "winogrande", "openbookqa",
]

ALL_ARMS = ["A", "B", "C", "D"]
ALL_SEEDS = [42, 123, 456]

# 95% two-tailed critical values for Student's t (df -> t_crit).
# Source: standard t-table. Used when SciPy is unavailable.
T_CRIT_95_TABLE = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


# =============================================================================
# Arm definitions
# =============================================================================

@dataclass
class ArmConfig:
    """Configuration for one ablation arm."""
    name: str
    description: str
    mu_inv: float
    mu_sep: float
    mu_geo: float
    frozen: frozenset[str]


ARM_CONFIGS: dict[str, ArmConfig] = {
    "A": ArmConfig(
        name="A", description="Answer-only CE (no constraints)",
        mu_inv=0.0, mu_sep=0.0, mu_geo=0.0,
        frozen=frozenset({"mu_inv", "mu_sep", "mu_geo"}),
    ),
    "B": ArmConfig(
        name="B", description="Answer + invariance + separation",
        mu_inv=1.0, mu_sep=1.0, mu_geo=0.0,
        frozen=frozenset({"mu_geo"}),
    ),
    "C": ArmConfig(
        name="C", description="Answer + geodesic tail only",
        mu_inv=0.0, mu_sep=0.0, mu_geo=1.0,
        frozen=frozenset({"mu_inv", "mu_sep"}),
    ),
    "D": ArmConfig(
        name="D", description="Full constrained",
        mu_inv=1.0, mu_sep=1.0, mu_geo=1.0,
        frozen=frozenset(),
    ),
}


# =============================================================================
# Metrics helpers
# =============================================================================

def compute_4gram_repetition_rate(text: str) -> float:
    """Fraction of 4-grams that are repeated in the text."""
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def _t_crit_95_two_tailed(df: int) -> float:
    """Return 95% two-tailed t critical value for given degrees of freedom."""
    if df <= 0:
        return float("inf")

    # Prefer exact computation when SciPy is available.
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(student_t.ppf(0.975, df))
    except Exception:
        pass

    if df in T_CRIT_95_TABLE:
        return T_CRIT_95_TABLE[df]

    # Normal approximation for large df.
    return 1.96


def paired_ci(values_a: list[float], values_b: list[float], t_crit: float | None = None) -> dict:
    """Compute paired t-test CI for B - A across seeds.

    Returns dict with mean_diff, se, ci_low, ci_high, significant.
    """
    n = len(values_a)
    assert n == len(values_b), f"Mismatched lengths: {n} vs {len(values_b)}"
    if n < 2:
        return {
            "mean_diff": values_b[0] - values_a[0] if n == 1 else 0.0,
            "se": float("inf"), "ci_low": float("-inf"),
            "ci_high": float("inf"), "significant": False,
            "n": n,
        }

    diffs = [b - a for a, b in zip(values_a, values_b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n) if var_d > 0 else 0.0

    if t_crit is None:
        t_crit = _t_crit_95_two_tailed(n - 1)
    margin = t_crit * se
    ci_low = mean_d - margin
    ci_high = mean_d + margin

    return {
        "mean_diff": mean_d,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": ci_low > 0 or ci_high < 0,
        "n": n,
    }


def delta_vs_baseline_ci(values: list[float], baseline: float) -> dict:
    """Compute CI for (arm - baseline) over seeds."""
    baseline_vec = [baseline] * len(values)
    return paired_ci(baseline_vec, values)


def run_dir(output_root: str, arm: str, seed: int) -> Path:
    """Get the directory for a specific (arm, seed) run."""
    return Path(output_root) / f"arm_{arm}_seed_{seed}"


# =============================================================================
# Phase 1: Training
# =============================================================================

def train_arm_seed(
    arm_config: ArmConfig,
    seed: int,
    model_path: str,
    train_path: str,
    val_path: str,
    output_root: str,
) -> dict:
    """Train one (arm, seed) run. Returns the train result as dict."""
    import mlx.core as mx

    from modelcypher.cli.composition import get_dataset_training_service
    from modelcypher.core.domain.training.constraint_config import ConstraintState

    out_dir = run_dir(output_root, arm_config.name, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    result_file = out_dir / "train_result.json"
    if result_file.exists():
        logger.info("  [%s/seed=%d] Already trained, skipping", arm_config.name, seed)
        with open(result_file) as f:
            return json.load(f)

    logger.info(
        "  [%s/seed=%d] Training: %s",
        arm_config.name, seed, arm_config.description,
    )

    # Create constraint state with arm's configuration
    constraint_state = ConstraintState(
        mu_inv=arm_config.mu_inv,
        mu_sep=arm_config.mu_sep,
        mu_geo=arm_config.mu_geo,
        frozen=arm_config.frozen,
    )

    service = get_dataset_training_service()

    result = service.train_from_dataset(
        model_path=model_path,
        dataset_path=train_path,
        eval_dataset_path=val_path,
        output_path=str(out_dir / "adapter"),
        seed=seed,
        paired=True,
        constraint_state_override=constraint_state,
    )

    result_dict = asdict(result)
    result_dict["arm"] = arm_config.name
    result_dict["seed"] = seed
    result_dict["frozen"] = sorted(arm_config.frozen)

    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    logger.info(
        "  [%s/seed=%d] Done: iters=%d, PPL=%.2f→%.2f, CKA=%.3f",
        arm_config.name, seed,
        result.train_iters,
        result.baseline_perplexity,
        result.post_perplexity,
        result.mean_cka or 0.0,
    )

    # Free GPU memory
    del service
    mx.clear_cache()
    gc.collect()

    return result_dict


# =============================================================================
# Phase 2: Evaluation
# =============================================================================

def eval_arm_seed(
    arm: str,
    seed: int,
    model_path: str,
    output_root: str,
    skip_benchmarks: bool = False,
) -> dict:
    """Evaluate one (arm, seed) adapter. Returns eval result dict."""
    import mlx.core as mx

    out_dir = run_dir(output_root, arm, seed)
    adapter_path = str(out_dir / "adapter")

    eval_file = out_dir / "eval_result.json"
    cached: dict | None = None
    if eval_file.exists():
        with open(eval_file) as f:
            cached = json.load(f)
        has_benchmarks = (
            cached.get("benchmark_accuracy") is not None
            and cached.get("benchmark_mean") is not None
        )
        if skip_benchmarks or has_benchmarks:
            logger.info("  [%s/seed=%d] Already evaluated, skipping", arm, seed)
            return cached
        logger.info(
            "  [%s/seed=%d] Cached eval missing benchmarks, refreshing",
            arm,
            seed,
        )

    result: dict = dict(cached) if cached is not None else {"arm": arm, "seed": seed}
    result["arm"] = arm
    result["seed"] = seed

    # 1. Benchmarks (lm-eval)
    if not skip_benchmarks:
        from eval_adapter import extract_accuracy, run_lm_eval

        logger.info("  [%s/seed=%d] Running lm-eval benchmarks", arm, seed)
        scores = run_lm_eval(model_path, BENCHMARK_TASKS, adapter_path=adapter_path)
        acc = extract_accuracy(scores)
        result["benchmark_scores"] = scores
        result["benchmark_accuracy"] = acc
        result["benchmark_mean"] = (
            sum(acc.values()) / len(acc) if acc else 0.0
        )

    # 2. Inference tests
    inference_cached = (
        result.get("inference") is not None
        and result.get("inference_total") is not None
        and result.get("inference_correct") is not None
        and result.get("repetition_rate") is not None
    )
    if inference_cached:
        logger.info("  [%s/seed=%d] Reusing cached inference tests", arm, seed)
    else:
        logger.info("  [%s/seed=%d] Running inference tests", arm, seed)
        from eval_adapter import run_inference_tests

        inf_results = run_inference_tests(model_path, adapter_path=adapter_path)
        result["inference"] = inf_results

        # Score inference: count correct and compute repetition
        n_correct = 0
        rep_rates = []
        for ir in inf_results:
            resp = ir.get("response", "")
            expected = ir.get("expected", "")
            if expected.lower() in resp.lower():
                n_correct += 1
            rep_rates.append(compute_4gram_repetition_rate(resp))

        result["inference_correct"] = n_correct
        result["inference_total"] = len(inf_results)
        result["repetition_rate"] = (
            sum(rep_rates) / len(rep_rates) if rep_rates else 0.0
        )

    # 3. Geodesic layer comparison
    geodesic_cached = (
        result.get("geodesic_tail_mean_delta") is not None
        and result.get("geodesic_max_divergence") is not None
        and result.get("geodesic_max_layer") is not None
        and result.get("geodesic_all_deltas") is not None
    )
    if geodesic_cached:
        logger.info("  [%s/seed=%d] Reusing cached geodesic comparison", arm, seed)
    else:
        logger.info("  [%s/seed=%d] Running geodesic comparison", arm, seed)
        try:
            from mlx_lm import load as mlx_load

            from modelcypher.cli.composition import get_geodesic_trajectory_service

            geo_service = get_geodesic_trajectory_service()

            baseline_model, tokenizer = mlx_load(model_path)
            adapted_model, _ = mlx_load(model_path, adapter_path=adapter_path)

            # Use a few prompts per category for geodesic comparison
            prompts_by_cat: dict[str, list[str]] = {}
            with open(INFERENCE_PROMPTS) as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        cat = p.get("category", "unknown")
                        prompts_by_cat.setdefault(cat, []).append(p["prompt"])

            batch_result = geo_service.compare_layer_profiles_batch(
                baseline_model=baseline_model,
                adapted_model=adapted_model,
                tokenizer=tokenizer,
                categorized_prompts=prompts_by_cat,
                model_path=model_path,
                adapter_path=adapter_path,
            )

            # Extract layers 11-15 delta
            tail_deltas = [
                delta for layer_idx, delta in batch_result.mean_delta_by_layer
                if 11 <= layer_idx <= 15
            ]
            result["geodesic_tail_mean_delta"] = (
                sum(tail_deltas) / len(tail_deltas) if tail_deltas else 0.0
            )
            result["geodesic_max_divergence"] = batch_result.max_divergence_magnitude
            result["geodesic_max_layer"] = batch_result.max_divergence_layer
            result["geodesic_all_deltas"] = [
                {"layer": idx, "delta": d}
                for idx, d in batch_result.mean_delta_by_layer
            ]

            del baseline_model, adapted_model, tokenizer, geo_service
            mx.clear_cache()
            gc.collect()

        except Exception as e:
            logger.warning("  [%s/seed=%d] Geodesic comparison failed: %s", arm, seed, e)
            result["geodesic_error"] = str(e)

    with open(eval_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        "  [%s/seed=%d] Eval done: correct=%d/%d, rep=%.3f",
        arm, seed,
        result.get("inference_correct", 0),
        result.get("inference_total", 0),
        result.get("repetition_rate", 0),
    )

    return result


def eval_baseline(model_path: str, output_root: str, skip_benchmarks: bool = False) -> dict:
    """Evaluate the baseline model (no adapter). Cached."""
    import mlx.core as mx

    baseline_dir = Path(output_root) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    eval_file = baseline_dir / "eval_result.json"

    cached: dict | None = None
    if eval_file.exists():
        with open(eval_file) as f:
            cached = json.load(f)
        has_benchmarks = (
            cached.get("benchmark_accuracy") is not None
            and cached.get("benchmark_mean") is not None
        )
        if skip_benchmarks or has_benchmarks:
            logger.info("  [baseline] Already evaluated, skipping")
            return cached
        logger.info("  [baseline] Cached eval missing benchmarks, refreshing")

    result: dict = dict(cached) if cached is not None else {"arm": "baseline", "seed": None}
    result["arm"] = "baseline"
    result["seed"] = None

    if not skip_benchmarks:
        from eval_adapter import extract_accuracy, run_lm_eval

        logger.info("  [baseline] Running lm-eval benchmarks")
        scores = run_lm_eval(model_path, BENCHMARK_TASKS)
        acc = extract_accuracy(scores)
        result["benchmark_scores"] = scores
        result["benchmark_accuracy"] = acc
        result["benchmark_mean"] = (
            sum(acc.values()) / len(acc) if acc else 0.0
        )

    inference_cached = (
        result.get("inference") is not None
        and result.get("inference_total") is not None
        and result.get("inference_correct") is not None
        and result.get("repetition_rate") is not None
    )
    if inference_cached:
        logger.info("  [baseline] Reusing cached inference tests")
    else:
        logger.info("  [baseline] Running inference tests")
        from eval_adapter import run_inference_tests

        inf_results = run_inference_tests(model_path)
        result["inference"] = inf_results

        n_correct = 0
        rep_rates = []
        for ir in inf_results:
            resp = ir.get("response", "")
            expected = ir.get("expected", "")
            if expected.lower() in resp.lower():
                n_correct += 1
            rep_rates.append(compute_4gram_repetition_rate(resp))

        result["inference_correct"] = n_correct
        result["inference_total"] = len(inf_results)
        result["repetition_rate"] = (
            sum(rep_rates) / len(rep_rates) if rep_rates else 0.0
        )

    mx.clear_cache()
    gc.collect()

    with open(eval_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# =============================================================================
# Phase 3: Decision
# =============================================================================

def compute_decision(output_root: str, arms: list[str], seeds: list[int],
                     skip_benchmarks: bool = False) -> dict:
    """Compute decision from all eval results."""
    output_path = Path(output_root)

    # Load all eval results
    eval_data: dict[str, dict[int, dict]] = {}  # arm -> seed -> eval_result
    for arm in arms:
        eval_data[arm] = {}
        for seed in seeds:
            ef = run_dir(output_root, arm, seed) / "eval_result.json"
            if ef.exists():
                with open(ef) as f:
                    eval_data[arm][seed] = json.load(f)

    # Load baseline
    baseline_file = output_path / "baseline" / "eval_result.json"
    baseline = {}
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)

    # Compute pairwise comparisons
    comparisons = []
    comparisons_vs_baseline = []
    pairs_to_test = [
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "D"), ("C", "D"),
    ]

    for arm_a, arm_b in pairs_to_test:
        if arm_a not in eval_data or arm_b not in eval_data:
            continue

        common_seeds = sorted(
            set(eval_data[arm_a].keys()) & set(eval_data[arm_b].keys())
        )
        if len(common_seeds) < 2:
            logger.warning(
                "  Skipping %s vs %s: only %d common seeds",
                arm_a, arm_b, len(common_seeds),
            )
            continue

        comp: dict = {"arms": f"{arm_b} vs {arm_a}", "seeds": common_seeds}

        # Accuracy delta (lm-eval)
        if not skip_benchmarks:
            acc_a = [eval_data[arm_a][s].get("benchmark_mean", 0) for s in common_seeds]
            acc_b = [eval_data[arm_b][s].get("benchmark_mean", 0) for s in common_seeds]
            comp["accuracy"] = paired_ci(acc_a, acc_b)

        # Inference correct delta
        correct_a = [eval_data[arm_a][s].get("inference_correct", 0) for s in common_seeds]
        correct_b = [eval_data[arm_b][s].get("inference_correct", 0) for s in common_seeds]
        comp["inference_correct"] = paired_ci(correct_a, correct_b)

        # Repetition rate delta (negative = better)
        rep_a = [eval_data[arm_a][s].get("repetition_rate", 0) for s in common_seeds]
        rep_b = [eval_data[arm_b][s].get("repetition_rate", 0) for s in common_seeds]
        comp["repetition"] = paired_ci(rep_a, rep_b)

        # Geodesic tail delta
        geo_a = [eval_data[arm_a][s].get("geodesic_tail_mean_delta", 0) for s in common_seeds]
        geo_b = [eval_data[arm_b][s].get("geodesic_tail_mean_delta", 0) for s in common_seeds]
        comp["geodesic_tail"] = paired_ci(geo_a, geo_b)

        comparisons.append(comp)

    # Arm vs baseline comparisons
    baseline_present = (
        baseline.get("inference_correct") is not None
        and baseline.get("repetition_rate") is not None
    )
    if baseline_present:
        for arm in arms:
            arm_seed_data = eval_data.get(arm, {})
            available_seeds = sorted(set(seeds) & set(arm_seed_data.keys()))
            if len(available_seeds) < 2:
                logger.warning(
                    "  Skipping %s vs baseline: only %d seeds",
                    arm, len(available_seeds),
                )
                continue

            comp_b: dict = {"arms": f"{arm} vs baseline", "seeds": available_seeds}

            if not skip_benchmarks and baseline.get("benchmark_mean") is not None:
                arm_acc = [
                    arm_seed_data[s].get("benchmark_mean", 0.0)
                    for s in available_seeds
                ]
                comp_b["accuracy"] = delta_vs_baseline_ci(
                    arm_acc,
                    float(baseline["benchmark_mean"]),
                )

            arm_inf = [
                arm_seed_data[s].get("inference_correct", 0.0)
                for s in available_seeds
            ]
            comp_b["inference_correct"] = delta_vs_baseline_ci(
                arm_inf,
                float(baseline["inference_correct"]),
            )

            arm_rep = [
                arm_seed_data[s].get("repetition_rate", 0.0)
                for s in available_seeds
            ]
            comp_b["repetition"] = delta_vs_baseline_ci(
                arm_rep,
                float(baseline["repetition_rate"]),
            )

            geo_vals = [
                arm_seed_data[s].get("geodesic_tail_mean_delta")
                for s in available_seeds
                if arm_seed_data[s].get("geodesic_tail_mean_delta") is not None
            ]
            if len(geo_vals) >= 2:
                comp_b["geodesic_tail"] = delta_vs_baseline_ci(geo_vals, 0.0)

            comparisons_vs_baseline.append(comp_b)

    # Verdicts
    verdicts: dict = {}

    def _get_comp(arm_a: str, arm_b: str) -> dict | None:
        label = f"{arm_b} vs {arm_a}"
        for c in comparisons:
            if c["arms"] == label:
                return c
        return None

    # D vs A: full constrained beats answer-only
    d_vs_a = _get_comp("A", "D")
    if d_vs_a:
        acc_better = (
            d_vs_a.get("accuracy", {}).get("ci_low", float("-inf")) > 0
            if not skip_benchmarks else None
        )
        inf_better = d_vs_a.get("inference_correct", {}).get("ci_low", float("-inf")) > 0
        rep_better = d_vs_a.get("repetition", {}).get("ci_high", float("inf")) < 0
        verdicts["D_beats_A_accuracy"] = acc_better
        verdicts["D_beats_A_inference"] = inf_better
        verdicts["D_beats_A_repetition"] = rep_better

    # D vs B: full beats inv+sep (geodesic adds value)
    d_vs_b = _get_comp("B", "D")
    if d_vs_b:
        verdicts["D_beats_B_accuracy"] = (
            d_vs_b.get("accuracy", {}).get("ci_low", float("-inf")) > 0
            if not skip_benchmarks else None
        )
        verdicts["D_beats_B_inference"] = (
            d_vs_b.get("inference_correct", {}).get("ci_low", float("-inf")) > 0
        )

    # D vs C: full beats geo-only (inv+sep add value)
    d_vs_c = _get_comp("C", "D")
    if d_vs_c:
        verdicts["D_beats_C_accuracy"] = (
            d_vs_c.get("accuracy", {}).get("ci_low", float("-inf")) > 0
            if not skip_benchmarks else None
        )
        verdicts["D_beats_C_inference"] = (
            d_vs_c.get("inference_correct", {}).get("ci_low", float("-inf")) > 0
        )

    # Promotion rule: D beats A AND (D beats B OR D beats C) on inference
    promote = False
    d_a_win = verdicts.get("D_beats_A_inference", False)
    d_b_win = verdicts.get("D_beats_B_inference", False)
    d_c_win = verdicts.get("D_beats_C_inference", False)
    if d_a_win and (d_b_win or d_c_win):
        promote = True
    verdicts["promote_to_700M_d_framework"] = promote

    # Baseline-centric promotion rule (primary): promote any arm that
    # strictly improves inference without increasing repetition.
    arm_vs_baseline: dict[str, dict] = {}
    for comp in comparisons_vs_baseline:
        label = comp["arms"]
        arm = label.split(" vs ")[0]
        arm_vs_baseline[arm] = comp
        inf_low = comp.get("inference_correct", {}).get("ci_low", float("-inf"))
        rep_high = comp.get("repetition", {}).get("ci_high", float("inf"))
        verdicts[f"{arm}_beats_baseline_inference"] = inf_low > 0.0
        verdicts[f"{arm}_no_worse_repetition"] = rep_high <= 0.0
        if not skip_benchmarks and "accuracy" in comp:
            acc_low = comp["accuracy"].get("ci_low", float("-inf"))
            verdicts[f"{arm}_no_worse_benchmark"] = acc_low >= 0.0

    promote_baseline_rule = False
    for arm in arms:
        inf_ok = verdicts.get(f"{arm}_beats_baseline_inference", False)
        rep_ok = verdicts.get(f"{arm}_no_worse_repetition", False)
        if not skip_benchmarks:
            bench_ok = verdicts.get(f"{arm}_no_worse_benchmark", False)
            if inf_ok and rep_ok and bench_ok:
                promote_baseline_rule = True
                verdicts["promote_arm"] = arm
                break
        else:
            if inf_ok and rep_ok:
                promote_baseline_rule = True
                verdicts["promote_arm"] = arm
                break

    verdicts["promote_to_700M_baseline_rule"] = promote_baseline_rule
    verdicts["promote_to_700M"] = promote_baseline_rule

    decision = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "arms": arms,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "t_crit_table_source": "Student t (95% two-tailed), computed per df",
        "comparisons": comparisons,
        "comparisons_vs_baseline": comparisons_vs_baseline,
        "verdicts": verdicts,
        "baseline": {
            "inference_correct": baseline.get("inference_correct"),
            "repetition_rate": baseline.get("repetition_rate"),
            "benchmark_mean": baseline.get("benchmark_mean"),
        },
    }

    # Save decision
    with open(output_path / "decision.json", "w") as f:
        json.dump(decision, f, indent=2, default=str)

    # Also save aggregated summary
    summary: dict = {"arms": {}}
    for arm in arms:
        arm_summary: dict = {"seeds": {}}
        for seed in seeds:
            if seed in eval_data.get(arm, {}):
                ed = eval_data[arm][seed]
                arm_summary["seeds"][seed] = {
                    "benchmark_mean": ed.get("benchmark_mean"),
                    "inference_correct": ed.get("inference_correct"),
                    "repetition_rate": ed.get("repetition_rate"),
                    "geodesic_tail_mean_delta": ed.get("geodesic_tail_mean_delta"),
                }
        # Compute arm means
        seed_data = list(arm_summary["seeds"].values())
        if seed_data:
            for metric in ["benchmark_mean", "inference_correct",
                           "repetition_rate", "geodesic_tail_mean_delta"]:
                vals = [s[metric] for s in seed_data if s.get(metric) is not None]
                arm_summary[f"mean_{metric}"] = (
                    sum(vals) / len(vals) if vals else None
                )
        summary["arms"][arm] = arm_summary

    summary["verdicts"] = verdicts

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print decision summary
    print("\n" + "=" * 60)
    print("ABLATION DECISION")
    print("=" * 60)

    for comp in comparisons:
        print(f"\n{comp['arms']}:")
        for metric in ["accuracy", "inference_correct", "repetition", "geodesic_tail"]:
            if metric in comp:
                ci = comp[metric]
                sig = "*" if ci["significant"] else " "
                print(
                    f"  {metric:25s}: Δ={ci['mean_diff']:+.4f} "
                    f"CI=[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] {sig}"
                )

    if comparisons_vs_baseline:
        print("\nArm vs baseline:")
        for comp in comparisons_vs_baseline:
            print(f"\n{comp['arms']}:")
            for metric in ["accuracy", "inference_correct", "repetition", "geodesic_tail"]:
                if metric in comp:
                    ci = comp[metric]
                    sig = "*" if ci["significant"] else " "
                    print(
                        f"  {metric:25s}: Δ={ci['mean_diff']:+.4f} "
                        f"CI=[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] {sig}"
                    )

    print(f"\nVerdicts:")
    for k, v in verdicts.items():
        emoji = "PASS" if v else ("FAIL" if v is False else "N/A")
        print(f"  {k:30s}: {emoji}")

    print(f"\n{'>>> PROMOTE TO 700M <<<' if promote else '>>> DO NOT PROMOTE <<<'}")
    print("=" * 60)

    return decision


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="350M ablation matrix")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model path")
    parser.add_argument("--train-data", default=DEFAULT_TRAIN, help="Training JSONL")
    parser.add_argument("--val-data", default=DEFAULT_VAL, help="Validation JSONL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument(
        "--arms", default=",".join(ALL_ARMS),
        help="Comma-separated arm names (default: A,B,C,D)",
    )
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in ALL_SEEDS),
        help="Comma-separated seeds (default: 42,123,456)",
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, eval existing adapters")
    parser.add_argument("--decision-only", action="store_true",
                        help="Skip training+eval, compute decision from results")
    parser.add_argument("--skip-benchmarks", action="store_true",
                        help="Skip lm-eval benchmarks (faster)")
    args = parser.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_root = args.output

    for arm in arms:
        if arm not in ARM_CONFIGS:
            parser.error(f"Unknown arm: {arm}. Must be one of {ALL_ARMS}")

    Path(output_root).mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "model": args.model,
        "train_data": args.train_data,
        "val_data": args.val_data,
        "arms": arms,
        "seeds": seeds,
        "arm_configs": {
            a: {"description": ARM_CONFIGS[a].description,
                "mu_inv": ARM_CONFIGS[a].mu_inv,
                "mu_sep": ARM_CONFIGS[a].mu_sep,
                "mu_geo": ARM_CONFIGS[a].mu_geo,
                "frozen": sorted(ARM_CONFIGS[a].frozen)}
            for a in arms
        },
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }
    with open(Path(output_root) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Phase 1: Training
    if not args.eval_only and not args.decision_only:
        logger.info("=" * 60)
        logger.info("PHASE 1: TRAINING (%d arms x %d seeds = %d runs)",
                     len(arms), len(seeds), len(arms) * len(seeds))
        logger.info("=" * 60)

        for arm in arms:
            for seed in seeds:
                train_arm_seed(
                    ARM_CONFIGS[arm], seed,
                    args.model, args.train_data, args.val_data,
                    output_root,
                )

    # Phase 2: Evaluation
    if not args.decision_only:
        logger.info("=" * 60)
        logger.info("PHASE 2: EVALUATION")
        logger.info("=" * 60)

        # Baseline first (run once)
        eval_baseline(args.model, output_root, args.skip_benchmarks)

        for arm in arms:
            for seed in seeds:
                eval_arm_seed(
                    arm, seed, args.model, output_root,
                    skip_benchmarks=args.skip_benchmarks,
                )

    # Phase 3: Decision
    logger.info("=" * 60)
    logger.info("PHASE 3: DECISION")
    logger.info("=" * 60)

    compute_decision(output_root, arms, seeds, args.skip_benchmarks)

    logger.info("Results in: %s", output_root)


if __name__ == "__main__":
    main()
