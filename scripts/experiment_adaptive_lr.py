#!/usr/bin/env python3
"""Adaptive LR Controlled Experiment — ablation matrix x 3 seeds.

Isolates the effect of adaptive LR improvements (H1: non-monotonic, H2: robust L)
on NB-LoRA by running a factorial experiment on a single model (700M) with multiple
seeds for statistical rigor.

Conditions:
  A:      NB-LoRA + constant LR          (baseline — geometry-derived, fixed eta)
  B_old:  NB-LoRA + adaptive (monotonic, 1-batch)  (previous behavior)
  B_h1:   NB-LoRA + H1 only             (non-monotonic LR, 1-batch L)
  B_h2:   NB-LoRA + H2 only             (monotonic, robust 3-batch L)
  B_h1h2: NB-LoRA + H1+H2              (non-monotonic + robust L — new default)
  C:      Std LoRA + matched rank        (isolate parameterization)
  D:      Std LoRA + default rank        (control — rank=8, Adam, lr=1e-5)

All conditions share: same data, same stopping, same eval tasks, same seeds.

Usage:
  # Quick smoke test (1 seed, limited eval)
  poetry run python scripts/experiment_adaptive_lr.py --quick --seeds 42 --conditions B_h1h2

  # Full NB-LoRA ablation (skip std LoRA controls — already have those)
  poetry run python scripts/experiment_adaptive_lr.py --conditions A B_old B_h1 B_h2 B_h1h2

  # Specific conditions
  poetry run python scripts/experiment_adaptive_lr.py --conditions B_old B_h1h2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Fix lm-eval / transformers 5.x incompatibility
# ---------------------------------------------------------------------------
import transformers
from transformers.utils.import_utils import _LazyModule

_original_lazy_getattr = _LazyModule.__getattr__


def _patched_lazy_getattr(self, name):
    if name == "AutoModelForVision2Seq":
        return type("_DummyAutoModel", (), {})
    return _original_lazy_getattr(self, name)


_LazyModule.__getattr__ = _patched_lazy_getattr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOLUME = Path("/Volumes/CodeCypher")
MODELS_DIR = VOLUME / "models" / "mlx-community"
EXPERIMENTS_DIR = VOLUME / "experiments"

MODEL_PATH = MODELS_DIR / "LFM2-700M-bf16"
MODEL_NAME = "LFM2-700M"
NUM_LAYERS = 32

TRAIN_DATA = REPO_ROOT / "data" / "training" / "benchmark_train.jsonl"
VAL_DATA = REPO_ROOT / "data" / "training" / "benchmark_val.jsonl"

BENCHMARK_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "boolq",
    "piqa",
    "winogrande",
    "openbookqa",
]

SEEDS = [42, 123, 456]
CONDITIONS = ["A", "B_old", "B_h1", "B_h2", "B_h1h2", "C", "D"]
NB_CONDITIONS = {"A", "B_old", "B_h1", "B_h2", "B_h1h2"}
STD_CONDITIONS = {"C", "D"}

STANDARD_LORA_CONFIG = {
    "lr": 1e-5,
    "batch_size": 4,
    "iters": 1000,
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
    "optimizer": "adam",
    "max_seq_length": 2048,
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_standard_lora_data_dir(train_path: Path, val_path: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_path, tmp_dir / "train.jsonl")
    shutil.copy2(val_path, tmp_dir / "valid.jsonl")
    return tmp_dir


# ---------------------------------------------------------------------------
# lm-eval wrapper
# ---------------------------------------------------------------------------
def run_lm_eval(
    model_path: str,
    tasks: list[str],
    adapter_path: str | None = None,
    limit: int | None = None,
) -> dict:
    import mlx.core as mx
    from lm_eval import simple_evaluate
    from mlx_lm.evaluate import MLXLM

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info(f"  lm-eval: {', '.join(tasks)} (adapter={adapter_path or 'none'})")

    lm = MLXLM(path_or_hf_repo=model_path, batch_size=1)

    if adapter_path:
        from mlx_lm.lora import load_adapters
        load_adapters(lm._model, adapter_path)

    results = simple_evaluate(
        model=lm, tasks=tasks, limit=limit, random_seed=42, numpy_random_seed=42,
    )

    scores = {}
    for task_name, task_results in results.get("results", {}).items():
        scores[task_name] = {
            k: v for k, v in task_results.items() if not k.startswith("alias")
        }

    del lm
    mx.clear_cache()
    gc.collect()
    return scores


# ---------------------------------------------------------------------------
# Standard LoRA training
# ---------------------------------------------------------------------------
def train_standard_lora(
    model_path: str,
    data_dir: Path,
    output_dir: Path,
    rank: int = 8,
    seed: int = 42,
    quick: bool = False,
) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load
    from mlx_lm.lora import (
        TrainingArgs,
        load_dataset,
        linear_to_lora_layers,
        save_config,
        train,
    )
    from mlx_lm.tuner.datasets import CacheDataset

    logger.info(f"  Standard LoRA (rank={rank}, seed={seed}): loading model...")
    model, tokenizer = mlx_load(str(model_path))

    iters = 100 if quick else STANDARD_LORA_CONFIG["iters"]
    lora_params = {
        "rank": rank,
        "dropout": STANDARD_LORA_CONFIG["dropout"],
        "scale": STANDARD_LORA_CONFIG["scale"],
    }

    args = SimpleNamespace(
        model=str(model_path),
        data=str(data_dir),
        train=True,
        test=False,
        fine_tune_type="lora",
        optimizer=STANDARD_LORA_CONFIG["optimizer"],
        optimizer_config={"adam": {}, "adamw": {}, "sgd": {}},
        seed=seed,
        num_layers=NUM_LAYERS,
        batch_size=STANDARD_LORA_CONFIG["batch_size"],
        iters=iters,
        val_batches=25,
        learning_rate=STANDARD_LORA_CONFIG["lr"],
        steps_per_report=10,
        steps_per_eval=200,
        resume_adapter_file=None,
        adapter_path=str(output_dir),
        save_every=iters,
        max_seq_length=STANDARD_LORA_CONFIG["max_seq_length"],
        config=None,
        grad_checkpoint=False,
        grad_accumulation_steps=1,
        lr_schedule=None,
        lora_parameters=lora_params,
        mask_prompt=False,
        report_to=None,
        project_name=None,
        hf_dataset=False,
    )

    train_set, valid_set, _ = load_dataset(args, tokenizer)

    model.freeze()
    linear_to_lora_layers(model, NUM_LAYERS, args.lora_parameters)

    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    save_config(vars(args), output_dir / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )

    opt = optim.Adam(learning_rate=args.learning_rate)

    t0 = time.time()
    train(
        model=model,
        args=training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
    )
    training_time = time.time() - t0

    # Spectral norms post-hoc
    spectral_info = measure_standard_lora_spectral_norms(model)

    result = {
        "method": "standard_lora",
        "rank": rank,
        "seed": seed,
        "iters": iters,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "adapter_path": str(output_dir),
        "hyperparameters": {**STANDARD_LORA_CONFIG, "rank": rank},
        "spectral_info": spectral_info,
    }

    del model, tokenizer
    mx.clear_cache()
    gc.collect()
    return result


def measure_standard_lora_spectral_norms(model) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat_params = tree_flatten(model.trainable_parameters())
    lora_pairs = {}
    for name, param in flat_params:
        if "lora_a" in name:
            base = name.replace(".lora_a.weight", "").replace(".lora_a", "")
            lora_pairs.setdefault(base, {})["a"] = param
        elif "lora_b" in name:
            base = name.replace(".lora_b.weight", "").replace(".lora_b", "")
            lora_pairs.setdefault(base, {})["b"] = param

    spectral_norms = []
    for base_key, pair in lora_pairs.items():
        if "a" in pair and "b" in pair:
            delta = pair["a"] @ pair["b"]
            s = mx.linalg.svd(delta.astype(mx.float32), stream=mx.cpu)[1]
            mx.eval(s)
            s_max = float(s[0])
            spectral_norms.append({
                "layer": base_key,
                "spectral_norm": s_max,
                "scaled_spectral_norm": s_max * STANDARD_LORA_CONFIG["scale"],
            })

    max_norm = max(n["scaled_spectral_norm"] for n in spectral_norms) if spectral_norms else 0
    return {"max_spectral_norm": max_norm}


# ---------------------------------------------------------------------------
# NB-LoRA training
# ---------------------------------------------------------------------------
def train_nb_lora(
    model_path: str,
    output_dir: Path,
    adaptive_lr: bool = True,
    lr_monotonic: bool = False,
    seed: int = 42,
    quick: bool = False,
) -> dict:
    import mlx.core as mx
    from modelcypher.backends import initialize_default_backend
    from modelcypher.cli.composition import get_dataset_training_service

    initialize_default_backend()
    service = get_dataset_training_service()

    max_iters = 200 if quick else 10000
    parts = []
    if not adaptive_lr:
        parts.append("constant")
    else:
        parts.append("monotonic" if lr_monotonic else "non-monotonic")
    label = "-".join(parts)

    logger.info(f"  NB-LoRA ({label} LR, seed={seed}): training...")
    t0 = time.time()
    result = service.train_from_dataset(
        model_path=str(model_path),
        dataset_path=str(TRAIN_DATA),
        output_path=str(output_dir),
        eval_dataset_path=str(VAL_DATA),
        max_iters=max_iters,
        seq_length=256,
        deep=False,
        safety_margin=0.9,
        seed=seed,
        adaptive_lr=adaptive_lr,
        lr_monotonic=lr_monotonic,
    )
    training_time = time.time() - t0

    result_dict = result.to_dict()
    result_dict["method"] = "nb_lora"
    result_dict["adaptive_lr"] = adaptive_lr
    result_dict["lr_monotonic"] = lr_monotonic
    result_dict["seed"] = seed
    result_dict["training_time_seconds"] = training_time
    result_dict["adapter_path"] = str(output_dir)

    mx.clear_cache()
    gc.collect()
    return result_dict


# ---------------------------------------------------------------------------
# Single condition runner
# ---------------------------------------------------------------------------
def run_condition(
    condition: str,
    seed: int,
    output_dir: Path,
    nb_param_count: int | None,
    quick: bool = False,
) -> dict:
    """Run a single condition with a single seed. Returns result dict."""
    cond_seed_dir = output_dir / f"condition_{condition}" / f"seed_{seed}"
    cond_seed_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = cond_seed_dir / "adapter"

    logger.info(f"\n--- Condition {condition}, seed {seed} ---")

    # NB-LoRA conditions: map condition name to (adaptive_lr, lr_monotonic)
    # NOTE: lipschitz_batches removed — MASS uses spectral ceiling now
    nb_condition_map = {
        "A":      (False, False),   # Constant LR
        "B_old":  (True,  True),    # Old adaptive: monotonic
        "B_h1":   (True,  False),   # H1 only: non-monotonic
        "B_h2":   (True,  True),    # H2 only: monotonic
        "B_h1h2": (True,  False),   # Both H1+H2 (new default)
    }

    if condition in nb_condition_map:
        adaptive, monotonic = nb_condition_map[condition]
        train_result = train_nb_lora(
            model_path=str(MODEL_PATH),
            output_dir=adapter_dir,
            adaptive_lr=adaptive,
            lr_monotonic=monotonic,
            seed=seed,
            quick=quick,
        )

    elif condition == "C":
        # Standard LoRA + matched rank
        matched_rank = compute_matched_rank(nb_param_count) if nb_param_count else 43
        logger.info(f"  Matched rank: {matched_rank} (from {nb_param_count} NB-LoRA params)")

        std_data_dir = cond_seed_dir / "_std_data"
        prepare_standard_lora_data_dir(TRAIN_DATA, VAL_DATA, std_data_dir)

        train_result = train_standard_lora(
            model_path=str(MODEL_PATH),
            data_dir=std_data_dir,
            output_dir=adapter_dir,
            rank=matched_rank,
            seed=seed,
            quick=quick,
        )
        shutil.rmtree(std_data_dir, ignore_errors=True)

    elif condition == "D":
        # Standard LoRA + default rank
        std_data_dir = cond_seed_dir / "_std_data"
        prepare_standard_lora_data_dir(TRAIN_DATA, VAL_DATA, std_data_dir)

        train_result = train_standard_lora(
            model_path=str(MODEL_PATH),
            data_dir=std_data_dir,
            output_dir=adapter_dir,
            rank=8,
            seed=seed,
            quick=quick,
        )
        shutil.rmtree(std_data_dir, ignore_errors=True)

    else:
        raise ValueError(f"Unknown condition: {condition}")

    train_result["condition"] = condition
    train_result["seed"] = seed

    # Save training log
    with open(cond_seed_dir / "training_log.json", "w") as f:
        json.dump(train_result, f, indent=2, default=str)

    return train_result


def compute_matched_rank(nb_param_count: int) -> int:
    """Compute standard LoRA rank that matches NB-LoRA total param count.

    For LFM2-700M with 32 layers targeting attention + MLP:
    Standard LoRA params per layer ≈ (in_dim + out_dim) * rank per projection.
    We solve for rank to match total NB-LoRA params.
    """
    # LFM2-700M: hidden_dim=1536, intermediate=4096
    # Attention: q,k,v,o each (1536+1536)*rank = 3072*rank, 4 projections
    # MLP: up,down,gate: (1536+4096)*rank=5632*rank, (4096+1536)*rank=5632*rank, (1536+4096)*rank=5632*rank
    # Per layer: 4*3072*rank + 3*5632*rank = (12288 + 16896)*rank = 29184*rank
    # 32 layers: 32 * 29184 * rank = 933888 * rank
    # Actually simpler: just divide total params by estimated per-rank cost

    # Empirical: default rank=8 gives ~4.5M params on 700M
    # So params_per_rank ≈ 4_500_000 / 8 ≈ 562_500
    # For ~24.4M NB-LoRA params: rank ≈ 24_400_000 / 562_500 ≈ 43.4
    params_per_rank = 562_500  # empirical from default rank=8 on 700M
    matched_rank = max(1, round(nb_param_count / params_per_rank))
    return matched_rank


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_condition(
    condition: str,
    seed: int,
    output_dir: Path,
    eval_limit: int | None = None,
) -> dict:
    """Run lm-eval for a single condition+seed. Returns scores dict."""
    adapter_dir = output_dir / f"condition_{condition}" / f"seed_{seed}" / "adapter"

    if not adapter_dir.exists():
        logger.warning(f"No adapter found for condition {condition} seed {seed}")
        return {}

    scores = run_lm_eval(
        model_path=str(MODEL_PATH),
        tasks=BENCHMARK_TASKS,
        adapter_path=str(adapter_dir),
        limit=eval_limit,
    )

    eval_path = output_dir / f"condition_{condition}" / f"seed_{seed}" / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(scores, f, indent=2)

    return scores


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def extract_primary_metric(scores: dict, task: str) -> float | None:
    """Extract the primary metric value for a task."""
    task_scores = scores.get(task, {})
    for m in ["acc_norm,none", "acc,none", "exact_match,strict-match"]:
        if m in task_scores:
            return task_scores[m]
    for k in task_scores:
        if "acc" in k or "exact" in k:
            return task_scores[k]
    return None


def build_analysis(
    all_results: dict,
    baseline_scores: dict,
) -> dict:
    """Build statistical analysis across conditions and seeds."""
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "conditions": {},
        "pairwise": {},
        "mechanism_metrics": {},
    }

    # Per-condition per-task: mean +/- std across seeds
    for cond in CONDITIONS:
        cond_data = all_results.get(cond, {})
        if not cond_data:
            continue

        cond_analysis = {"per_task": {}, "training": {}}

        for task in BENCHMARK_TASKS:
            base_val = extract_primary_metric(baseline_scores, task)
            task_values = []
            for seed_data in cond_data.values():
                eval_scores = seed_data.get("eval", {})
                val = extract_primary_metric(eval_scores, task)
                if val is not None:
                    task_values.append(val)

            if task_values:
                mean = sum(task_values) / len(task_values)
                std = math.sqrt(sum((v - mean) ** 2 for v in task_values) / max(len(task_values) - 1, 1))
                cond_analysis["per_task"][task] = {
                    "baseline": base_val,
                    "values": task_values,
                    "mean": mean,
                    "std": std,
                    "n": len(task_values),
                    "delta_vs_base": mean - base_val if base_val is not None else None,
                }

        # Training summary
        train_times = []
        train_iters = []
        for seed_data in cond_data.values():
            tr = seed_data.get("train", {})
            if "training_time_seconds" in tr:
                train_times.append(tr["training_time_seconds"])
            iters = tr.get("train_iters", tr.get("iters", 0))
            train_iters.append(iters)

        cond_analysis["training"] = {
            "mean_time": sum(train_times) / len(train_times) if train_times else 0,
            "mean_iters": sum(train_iters) / len(train_iters) if train_iters else 0,
        }

        # Epoch metrics (NB-LoRA conditions only)
        if cond in NB_CONDITIONS:
            all_epoch_metrics = []
            for seed_data in cond_data.values():
                tr = seed_data.get("train", {})
                em = tr.get("epoch_metrics", [])
                if em:
                    all_epoch_metrics.append(em)
            if all_epoch_metrics:
                cond_analysis["mechanism"] = summarize_mechanism_metrics(all_epoch_metrics)

        analysis["conditions"][cond] = cond_analysis

    # Pairwise comparisons — test each H fix against old behavior
    pairwise_pairs = [
        ("B_old_vs_B_h1", ("B_old", "B_h1")),
        ("B_old_vs_B_h2", ("B_old", "B_h2")),
        ("B_old_vs_B_h1h2", ("B_old", "B_h1h2")),
        ("A_vs_B_h1h2", ("A", "B_h1h2")),
    ]
    for pair_name, (c1, c2) in pairwise_pairs:
        cond1 = analysis["conditions"].get(c1, {})
        cond2 = analysis["conditions"].get(c2, {})
        if not cond1 or not cond2:
            continue

        pair_results = {}
        c2_wins = 0
        c1_wins = 0
        for task in BENCHMARK_TASKS:
            t1 = cond1.get("per_task", {}).get(task, {})
            t2 = cond2.get("per_task", {}).get(task, {})
            if "mean" in t1 and "mean" in t2:
                delta = t2["mean"] - t1["mean"]
                pair_results[task] = {
                    f"{c1}_mean": t1["mean"],
                    f"{c2}_mean": t2["mean"],
                    "delta": delta,
                    "winner": c2 if delta > 0 else c1 if delta < 0 else "tie",
                }
                if delta > 0:
                    c2_wins += 1
                elif delta < 0:
                    c1_wins += 1

        analysis["pairwise"][pair_name] = {
            "tasks": pair_results,
            f"{c1}_wins": c1_wins,
            f"{c2}_wins": c2_wins,
            "total": c1_wins + c2_wins,
        }

    return analysis


def summarize_mechanism_metrics(all_epoch_metrics: list[list[dict]]) -> dict:
    """Summarize epoch metrics across seeds."""
    summary = {}

    # Aggregate by epoch number
    by_epoch: dict[int, list[dict]] = {}
    for seed_metrics in all_epoch_metrics:
        for m in seed_metrics:
            epoch = m.get("epoch", 0)
            by_epoch.setdefault(epoch, []).append(m)

    epochs = []
    for epoch_num in sorted(by_epoch.keys()):
        entries = by_epoch[epoch_num]
        epoch_summary = {"epoch": epoch_num}

        for key in ["eta", "eta_ceiling", "lipschitz_L", "update_norm",
                     "max_spectral_ratio", "mean_token_entropy", "repetition_rate",
                     "val_loss", "train_loss"]:
            values = [e[key] for e in entries if e.get(key) is not None]
            if values:
                mean = sum(values) / len(values)
                epoch_summary[key] = {
                    "mean": mean,
                    "values": values,
                }

        epochs.append(epoch_summary)

    summary["per_epoch"] = epochs

    # Final epoch summary
    if epochs:
        last = epochs[-1]
        summary["final_eta_mean"] = last.get("eta", {}).get("mean")
        summary["final_rep_rate_mean"] = last.get("repetition_rate", {}).get("mean")
        summary["final_entropy_mean"] = last.get("mean_token_entropy", {}).get("mean")

    return summary


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_analysis(analysis: dict):
    print("\n" + "=" * 80)
    print("EXPERIMENT: Adaptive LR Controlled Comparison")
    print("=" * 80)

    for cond, cond_data in analysis.get("conditions", {}).items():
        cond_labels = {
            "A": "NB-LoRA + Constant LR",
            "B_old": "NB-LoRA + Adaptive (monotonic, 1-batch)",
            "B_h1": "NB-LoRA + H1 (non-monotonic, 1-batch)",
            "B_h2": "NB-LoRA + H2 (monotonic, 3-batch)",
            "B_h1h2": "NB-LoRA + H1+H2 (non-monotonic, 3-batch)",
            "C": "Std LoRA + Matched Rank",
            "D": "Std LoRA + Default (rank=8)",
        }
        print(f"\n  Condition {cond}: {cond_labels.get(cond, cond)}")
        print(f"    {'Task':<20} {'Mean':>8} {'Std':>8} {'Δ Base':>8}")
        print(f"    {'-'*48}")

        for task, data in cond_data.get("per_task", {}).items():
            mean = data.get("mean", 0)
            std = data.get("std", 0)
            delta = data.get("delta_vs_base")
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
            print(f"    {task:<20} {mean:>8.4f} {std:>8.4f} {delta_str:>8}")

        tr = cond_data.get("training", {})
        print(f"    Time: {tr.get('mean_time', 0):.1f}s, Iters: {tr.get('mean_iters', 0):.0f}")

        mech = cond_data.get("mechanism")
        if mech:
            final_eta = mech.get("final_eta_mean")
            final_rep = mech.get("final_rep_rate_mean")
            final_ent = mech.get("final_entropy_mean")
            parts = []
            if final_eta is not None:
                parts.append(f"eta={final_eta:.2e}")
            if final_rep is not None:
                parts.append(f"rep={final_rep:.3f}")
            if final_ent is not None:
                parts.append(f"entropy={final_ent:.2f}")
            if parts:
                print(f"    Mechanism: {', '.join(parts)}")

    for pair_name, pair_data in analysis.get("pairwise", {}).items():
        c1, c2 = pair_name.split("_vs_")
        print(f"\n  {pair_name}:")
        for task, data in pair_data.get("tasks", {}).items():
            winner = data.get("winner", "?")
            delta = data.get("delta", 0)
            print(f"    {task:<20} Δ={delta:+.4f}  → {winner}")
        print(f"    Score: {c1}={pair_data.get(f'{c1}_wins', 0)}, {c2}={pair_data.get(f'{c2}_wins', 0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive LR ablation experiment (H1/H2 x 3 seeds)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
        help="Which conditions to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds (default: 42 123 456)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iters, limited eval",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip lm-eval benchmarks",
    )
    args = parser.parse_args()

    # Validate
    if not VOLUME.exists():
        logger.error(f"Volume not mounted: {VOLUME}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    for path in [TRAIN_DATA, VAL_DATA]:
        if not path.exists():
            logger.error(f"Data file not found: {path}")
            sys.exit(1)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        EXPERIMENTS_DIR / f"adaptive-lr-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Adaptive LR Controlled Experiment")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_NAME} ({MODEL_PATH})")
    logger.info(f"Conditions: {args.conditions}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Output: {output_dir}")

    eval_limit = 50 if args.quick else None
    total_t0 = time.time()

    # Save experiment config
    config = {
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "conditions": args.conditions,
        "seeds": args.seeds,
        "quick": args.quick,
        "train_data": str(TRAIN_DATA),
        "val_data": str(VAL_DATA),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # 1. BASELINE EVAL (once, no adapter)
    # ------------------------------------------------------------------
    baseline_scores = {}
    if not args.skip_eval:
        logger.info(f"\n{'='*60}")
        logger.info("BASELINE EVALUATION")
        logger.info(f"{'='*60}")
        baseline_scores = run_lm_eval(
            model_path=str(MODEL_PATH),
            tasks=BENCHMARK_TASKS,
            limit=eval_limit,
        )
        with open(output_dir / "baseline_eval.json", "w") as f:
            json.dump(baseline_scores, f, indent=2)

    # ------------------------------------------------------------------
    # 2. TRAINING: all conditions x seeds
    # ------------------------------------------------------------------
    all_results: dict[str, dict[int, dict]] = {}
    nb_param_count = None  # Will be set after first NB-LoRA run

    # Run NB-LoRA conditions first to get param count for matched rank
    nb_conditions = [c for c in args.conditions if c in NB_CONDITIONS]
    std_conditions = [c for c in args.conditions if c in STD_CONDITIONS]

    for condition in nb_conditions + std_conditions:
        all_results.setdefault(condition, {})

        for seed in args.seeds:
            logger.info(f"\n{'='*60}")
            logger.info(f"CONDITION {condition}, SEED {seed}")
            logger.info(f"{'='*60}")

            try:
                train_result = run_condition(
                    condition=condition,
                    seed=seed,
                    output_dir=output_dir,
                    nb_param_count=nb_param_count,
                    quick=args.quick,
                )

                # Capture NB-LoRA param count from first run
                if condition in NB_CONDITIONS and nb_param_count is None:
                    nb_param_count = train_result.get("n_trainable_params")
                    if nb_param_count:
                        logger.info(f"NB-LoRA param count: {nb_param_count} (will use for matched rank)")

                all_results[condition][seed] = {"train": train_result}

            except Exception:
                logger.exception(f"FAILED: condition {condition} seed {seed}")
                all_results[condition][seed] = {"train": {"error": True}}

    # ------------------------------------------------------------------
    # 3. EVALUATION: all conditions x seeds
    # ------------------------------------------------------------------
    if not args.skip_eval:
        for condition in args.conditions:
            for seed in args.seeds:
                if condition not in all_results or seed not in all_results[condition]:
                    continue

                logger.info(f"\n--- Evaluating condition {condition}, seed {seed} ---")
                try:
                    eval_scores = evaluate_condition(
                        condition=condition,
                        seed=seed,
                        output_dir=output_dir,
                        eval_limit=eval_limit,
                    )
                    all_results[condition][seed]["eval"] = eval_scores
                except Exception:
                    logger.exception(f"Eval failed: condition {condition} seed {seed}")
                    all_results[condition][seed]["eval"] = {}

    total_time = time.time() - total_t0

    # ------------------------------------------------------------------
    # 4. ANALYSIS
    # ------------------------------------------------------------------
    analysis = build_analysis(all_results, baseline_scores)
    analysis["total_time_seconds"] = total_time
    analysis["nb_param_count"] = nb_param_count

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Save full results
    with open(output_dir / "full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_analysis(analysis)
    logger.info(f"\nTotal experiment time: {total_time:.1f}s")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
