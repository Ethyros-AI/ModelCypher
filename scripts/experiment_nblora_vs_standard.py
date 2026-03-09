#!/usr/bin/env python3
"""NB-LoRA vs Standard LoRA — Controlled Head-to-Head Comparison (R1).

Three-arm comparison:
  Arm 1: NB-LoRA (zero HP, geometry-derived)
  Arm 2: Standard LoRA (community defaults: AdamW, cosine LR, rank=8, alpha=16)
  Arm 3: Tuned Standard LoRA (best of rank x lr grid search)

Each arm x N seeds (default 5) for statistical validity.

Measurements per arm per seed:
  - lm-eval-harness benchmark scores (7 tasks)
  - Training loss convergence (captured via callback)
  - Spectral safety (post-hoc for standard, by construction for NB-LoRA)
  - Inference test responses

Usage:
  # Quick smoke test (350M, 1 seed, limited evals)
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M --quick

  # Full comparison (350M, 5 seeds)
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M

  # Custom seeds
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M --seeds 42 123 456

  # Skip eval for pipeline debugging
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M --skip-eval --skip-inference
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import statistics
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

MODEL_SPECS = {
    "350M": {
        "path": MODELS_DIR / "LFM2-350M-MLX-bf16",
        "name": "LFM2-350M",
        "num_layers": 16,
    },
    "700M": {
        "path": MODELS_DIR / "LFM2-700M-bf16",
        "name": "LFM2-700M",
        "num_layers": 16,
    },
    "0.8B": {
        "path": MODELS_DIR / "Qwen3.5-0.8B-bf16",
        "name": "Qwen3.5-0.8B",
        "num_layers": 24,
    },
    "1.2B": {
        "path": MODELS_DIR / "LFM2.5-1.2B-Base-bf16",
        "name": "LFM2.5-1.2B",
        "num_layers": 16,
    },
}

TRAIN_DATA = REPO_ROOT / "data" / "training" / "benchmark_train.jsonl"
VAL_DATA = REPO_ROOT / "data" / "training" / "benchmark_val.jsonl"
INFERENCE_PROMPTS = REPO_ROOT / "data" / "eval_prompts" / "nblora_inference_tests.jsonl"

BENCHMARK_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "boolq",
    "piqa",
    "winogrande",
    "openbookqa",
]

# LoRA alpha — standard community value
LORA_ALPHA = 16

# Community-standard LoRA defaults (what a competent practitioner starts with)
STANDARD_LORA_CONFIG = {
    "lr": 2e-4,
    "batch_size": 4,
    "iters": 1000,
    "rank": 8,
    "scale": LORA_ALPHA / 8,  # alpha / rank = 2.0
    "dropout": 0.0,
    "optimizer": "adamw",
    "lr_schedule": "cosine",
    "max_seq_length": 2048,
}

# Grid search for tuned arm: rank x lr, scale derived from alpha
TUNED_GRID = [
    {"rank": r, "lr": lr, "scale": LORA_ALPHA / r}
    for r in [4, 8, 16]
    for lr in [1e-5, 5e-5, 2e-4]
]

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_standard_lora_data_dir(train_path: Path, val_path: Path, tmp_dir: Path):
    """mlx_lm.lora.load_dataset expects a directory with train.jsonl/valid.jsonl."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_path, tmp_dir / "train.jsonl")
    shutil.copy2(val_path, tmp_dir / "valid.jsonl")
    return tmp_dir


# ---------------------------------------------------------------------------
# Training callback for loss capture
# ---------------------------------------------------------------------------
class LossCapture:
    """Captures train/val losses from mlx-lm training."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_loss_report(self, info: dict):
        self.train_losses.append({
            "iteration": info["iteration"],
            "train_loss": float(info["train_loss"]),
            "learning_rate": float(info.get("learning_rate", 0)),
        })

    def on_val_loss_report(self, info: dict):
        self.val_losses.append({
            "iteration": info["iteration"],
            "val_loss": float(info["val_loss"]),
        })

    @property
    def final_val_loss(self) -> float | None:
        return self.val_losses[-1]["val_loss"] if self.val_losses else None

    @property
    def final_train_loss(self) -> float | None:
        return self.train_losses[-1]["train_loss"] if self.train_losses else None


# ---------------------------------------------------------------------------
# lm-eval wrapper
# ---------------------------------------------------------------------------
def run_lm_eval(
    model_path: str,
    tasks: list[str],
    adapter_path: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run lm-eval benchmarks. Returns dict mapping task_name -> {metric: value}."""
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
        model=lm,
        tasks=tasks,
        limit=limit,
        random_seed=42,
        numpy_random_seed=42,
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
    num_layers: int,
    data_dir: Path,
    output_dir: Path,
    config: dict | None = None,
    seed: int = 42,
    iters_override: int | None = None,
) -> dict:
    """Train standard LoRA with given config and seed. Returns training metadata."""
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

    cfg = config or STANDARD_LORA_CONFIG
    iters = iters_override or cfg["iters"]

    mx.random.seed(seed)
    random.seed(seed)

    logger.info(f"  Standard LoRA: rank={cfg['rank']}, lr={cfg['lr']}, "
                f"scale={cfg['scale']:.2f}, opt={cfg['optimizer']}, seed={seed}")

    model, tokenizer = mlx_load(str(model_path))

    args = SimpleNamespace(
        model=str(model_path),
        data=str(data_dir),
        train=True,
        test=False,
        fine_tune_type="lora",
        optimizer=cfg["optimizer"],
        optimizer_config={"adam": {}, "adamw": {}, "sgd": {}},
        seed=seed,
        num_layers=num_layers,
        batch_size=cfg["batch_size"],
        iters=iters,
        val_batches=25,
        learning_rate=cfg["lr"],
        steps_per_report=10,
        steps_per_eval=200,
        resume_adapter_file=None,
        adapter_path=str(output_dir),
        save_every=iters,
        max_seq_length=cfg["max_seq_length"],
        config=None,
        grad_checkpoint=False,
        grad_accumulation_steps=1,
        lr_schedule=None,
        lora_parameters={
            "rank": cfg["rank"],
            "dropout": cfg.get("dropout", 0.0),
            "scale": cfg["scale"],
        },
        mask_prompt=False,
        report_to=None,
        project_name=None,
        hf_dataset=False,
    )

    train_set, valid_set, _ = load_dataset(args, tokenizer)

    model.freeze()
    linear_to_lora_layers(model, num_layers, args.lora_parameters)

    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    save_config(vars(args), output_dir / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )

    # Build optimizer with optional cosine schedule
    lr_value = cfg["lr"]
    if cfg.get("lr_schedule") == "cosine":
        lr_value = optim.cosine_decay(init=cfg["lr"], decay_steps=iters)

    if cfg["optimizer"] == "adamw":
        opt = optim.AdamW(learning_rate=lr_value, weight_decay=0.0)
    else:
        opt = optim.Adam(learning_rate=lr_value)

    # Train with loss capture
    loss_capture = LossCapture()
    t0 = time.time()
    train(
        model=model,
        args=training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=loss_capture,
    )
    training_time = time.time() - t0

    # Force sync: ensure adapter weights are flushed to disk before we
    # delete the model.  mlx-lm train() uses mx.save_safetensors which
    # is synchronous in current MLX, but the print messages may appear
    # out of order.
    adapter_file_check = output_dir / "adapters.safetensors"
    if not adapter_file_check.exists():
        # If train() hasn't saved yet (shouldn't happen with sync saves,
        # but guard against it), save manually.
        from mlx.utils import tree_flatten as tf
        adapter_weights = dict(tf(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file_check), adapter_weights)
        logger.warning("  Manually saved adapter weights (sync guard)")

    spectral_info = measure_standard_lora_spectral_norms(model, cfg["scale"])

    result = {
        "method": "standard_lora",
        "seed": seed,
        "iters": iters,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "adapter_path": str(output_dir),
        "config": {k: v for k, v in cfg.items()},
        "hyperparameter_count": len([
            k for k in cfg if k not in ("max_seq_length", "batch_size", "iters")
        ]),
        "final_val_loss": loss_capture.final_val_loss,
        "final_train_loss": loss_capture.final_train_loss,
        "loss_history": {
            "train": loss_capture.train_losses,
            "val": loss_capture.val_losses,
        },
        "spectral_info": spectral_info,
    }

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return result


def measure_standard_lora_spectral_norms(model, scale: float) -> dict:
    """Measure spectral norms of standard LoRA adapter weights post-hoc."""
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
            a = pair["a"]
            b = pair["b"]
            delta = a @ b
            try:
                s = mx.linalg.svd(delta.astype(mx.float32), stream=mx.cpu)[1]
                mx.eval(s)
                s_max = float(s[0])
            except Exception:
                s_max = float(mx.linalg.norm(delta.astype(mx.float32)))
                logger.warning(f"  SVD failed for {base_key}, using Frobenius norm")
            spectral_norms.append({
                "layer": base_key,
                "spectral_norm": s_max,
                "scaled_spectral_norm": s_max * scale,
            })

    max_norm = max(n["scaled_spectral_norm"] for n in spectral_norms) if spectral_norms else 0
    return {
        "max_spectral_norm": max_norm,
        "per_layer": spectral_norms,
    }


# ---------------------------------------------------------------------------
# NB-LoRA training
# ---------------------------------------------------------------------------
def train_nb_lora(
    model_path: str,
    output_dir: Path,
    seed: int = 42,
    weight_decay: float = 0.0,
) -> dict:
    """Train NB-LoRA using geometry-derived everything. Returns training metadata."""
    import mlx.core as mx
    from modelcypher.backends import initialize_default_backend
    from modelcypher.cli.composition import get_dataset_training_service

    initialize_default_backend()
    service = get_dataset_training_service()

    logger.info(f"  NB-LoRA: training (geometry-derived, seed={seed}, wd={weight_decay})...")
    t0 = time.time()
    result = service.train_from_dataset(
        model_path=str(model_path),
        dataset_path=str(TRAIN_DATA),
        output_path=str(output_dir),
        eval_dataset_path=str(VAL_DATA),
        seed=seed,
        weight_decay=weight_decay,
    )
    training_time = time.time() - t0

    result_dict = result.to_dict()
    result_dict["method"] = "nb_lora"
    result_dict["seed"] = seed
    result_dict["training_time_seconds"] = training_time
    result_dict["hyperparameter_count"] = 0
    result_dict["adapter_path"] = str(output_dir)
    # NOTE: final_val_loss populated by commensurable post-hoc eval, not
    # training callback.  post_loss kept for reference only.
    result_dict["post_loss_training"] = result_dict.get("post_loss")

    mx.clear_cache()
    gc.collect()

    return result_dict


def derive_wd_from_training_result(
    training_result: dict, adapter_path: str,
) -> tuple[float, dict]:
    """Derive the adversarial WD test value from a WD=0 training run.

    λ = median(d_norm) / ||θ||

    This is the WD that makes shrinkage comparable to the gradient step,
    i.e., the strongest adversary for the "WD is redundant" prediction.

    Returns (wd_derived, derivation_details).
    """
    import mlx.core as mx

    # d_norm per epoch (last step's gradient direction norm in each epoch)
    epoch_metrics = training_result.get("epoch_metrics", [])
    d_norms = [
        em["d_norm"] for em in (epoch_metrics or [])
        if em.get("d_norm") is not None and em["d_norm"] > 0
    ]
    if not d_norms:
        raise ValueError("No d_norm values found in training result epoch_metrics")
    median_d_norm = statistics.median(d_norms)

    # ||θ|| from saved adapter weights
    adapter_file = Path(adapter_path) / "adapters.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")
    weights = mx.load(str(adapter_file))
    param_norm_sq = 0.0
    for w in weights.values():
        w_f32 = w.astype(mx.float32)
        val = mx.sum(w_f32 * w_f32)
        mx.eval(val)
        param_norm_sq += float(val.item())
    param_norm = math.sqrt(param_norm_sq)

    if param_norm < 1e-12:
        raise ValueError(f"Parameter norm too small: {param_norm}")

    wd_derived = median_d_norm / param_norm

    derivation = {
        "median_d_norm": median_d_norm,
        "param_norm": param_norm,
        "n_epochs": len(d_norms),
        "d_norms_per_epoch": d_norms,
        "formula": "median(d_norm) / ||θ||",
    }

    return wd_derived, derivation


# ---------------------------------------------------------------------------
# Commensurable validation loss — same operator for all arms
# ---------------------------------------------------------------------------
def evaluate_val_loss(
    model_path: str,
    adapter_path: str | None,
    val_data_path: Path,
    max_seq_length: int = 2048,
) -> float:
    """Evaluate cross-entropy loss over the FULL validation set with batch_size=1.

    This is the single measurement operator used for all arms, ensuring
    commensurable val_loss comparisons.  Every sample in val_data_path is
    evaluated; there is no val_batches truncation.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load

    if adapter_path is not None:
        model, tokenizer = mlx_load(str(model_path), adapter_path=adapter_path)
    else:
        model, tokenizer = mlx_load(str(model_path))
    model.eval()

    val_samples = load_jsonl(val_data_path)
    total_loss = 0.0
    total_tokens = 0

    for sample in val_samples:
        text = sample.get("text", "")
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:max_seq_length]
        x = mx.array(tokens[:-1])[None, :]  # (1, T-1)
        y = mx.array(tokens[1:])             # (T-1,)
        logits = model(x)                    # (1, T-1, vocab)
        loss = nn.losses.cross_entropy(logits[0], y, reduction="sum")
        mx.eval(loss)
        total_loss += float(loss)
        total_tokens += len(tokens) - 1

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Inference test generation
# ---------------------------------------------------------------------------
def generate_inference_responses(
    model_path: str,
    adapter_path: str | None,
    prompts: list[dict],
    label: str,
) -> list[dict]:
    """Generate greedy responses for inference test prompts."""
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    logger.info(f"  Inference: generating {len(prompts)} responses ({label})...")

    model, tokenizer = mlx_load(
        str(model_path),
        adapter_path=adapter_path,
    )

    greedy_sampler = make_sampler(temp=0.0)

    responses = []
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            sampler=greedy_sampler,
        )
        responses.append({
            "id": prompt_data["id"],
            "response": response,
        })

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return responses


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def run_grid_search(
    model_spec: dict,
    data_dir: Path,
    output_dir: Path,
    grid: list[dict],
    grid_iters: int,
) -> dict:
    """Run grid search over standard LoRA configs. Returns best config."""
    model_path = str(model_spec["path"])
    num_layers = model_spec["num_layers"]

    grid_dir = output_dir / "grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, grid_config in enumerate(grid):
        config = STANDARD_LORA_CONFIG.copy()
        config["rank"] = grid_config["rank"]
        config["lr"] = grid_config["lr"]
        config["scale"] = grid_config["scale"]

        label = f"rank{config['rank']}_lr{config['lr']:.0e}"
        run_dir = grid_dir / label

        logger.info(f"  Grid [{i+1}/{len(grid)}]: {label}")

        try:
            result = train_standard_lora(
                model_path=model_path,
                num_layers=num_layers,
                data_dir=data_dir,
                output_dir=run_dir,
                config=config,
                seed=42,
                iters_override=grid_iters,
            )
            results.append({
                "label": label,
                "config": grid_config,
                "final_val_loss": result["final_val_loss"],
                "final_train_loss": result["final_train_loss"],
                "training_time": result["training_time_seconds"],
            })
        except Exception as e:
            logger.warning(f"  Grid {label} failed: {e}")
            results.append({
                "label": label,
                "config": grid_config,
                "final_val_loss": None,
                "error": str(e),
            })

        # Delete adapter to save space (grid search only needs val loss)
        adapter_file = run_dir / "adapters.safetensors"
        if adapter_file.exists():
            adapter_file.unlink()

    # Pick best by val loss
    valid_results = [
        r for r in results
        if r.get("final_val_loss") is not None
    ]
    if not valid_results:
        logger.error("  All grid search configs failed! Falling back to rank=8, lr=5e-5")
        best = {"rank": 8, "lr": 5e-5, "scale": LORA_ALPHA / 8}
    else:
        best_result = min(valid_results, key=lambda r: r["final_val_loss"])
        best = best_result["config"]

    grid_summary = {
        "grid_configs": results,
        "best_config": best,
        "best_val_loss": (
            min(r["final_val_loss"] for r in valid_results) if valid_results else None
        ),
    }

    with open(grid_dir / "grid_results.json", "w") as f:
        json.dump(grid_summary, f, indent=2)

    logger.info(f"  Grid search best: rank={best['rank']}, lr={best['lr']}, "
                f"scale={best['scale']:.2f}")

    return grid_summary


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------
def _stderr_key(metric_key: str) -> str:
    """Derive the lm-eval stderr key from a metric key.

    'acc_norm,none' -> 'acc_norm_stderr,none'
    'exact_match,strict-match' -> 'exact_match_stderr,strict-match'
    """
    parts = metric_key.split(",", 1)
    if len(parts) == 2:
        return f"{parts[0]}_stderr,{parts[1]}"
    return f"{metric_key}_stderr"


def extract_primary_metric(task_scores: dict) -> tuple[str, float, float] | None:
    """Find the primary metric for a task's eval scores.

    Returns (metric_name, value, stderr).
    """
    for m in ["acc_norm,none", "acc,none", "exact_match,strict-match"]:
        if m in task_scores:
            se = task_scores.get(_stderr_key(m), 0.0)
            return m, task_scores[m], se
    for k, v in task_scores.items():
        if ("acc" in k or "exact" in k) and "stderr" not in k:
            se = task_scores.get(_stderr_key(k), 0.0)
            return k, v, se
    return None


def aggregate_arm_evals(
    per_seed_evals: list[dict],
    baseline_scores: dict,
) -> dict:
    """Aggregate eval scores across seeds. Returns mean +/- std per task."""
    agg = {}
    for task in BENCHMARK_TASKS:
        base_result = extract_primary_metric(baseline_scores.get(task, {}))
        if base_result is None:
            continue
        metric_name, base_val, base_stderr = base_result
        se_key = _stderr_key(metric_name)

        seed_vals = []
        seed_stderrs = []
        for eval_scores in per_seed_evals:
            task_scores = eval_scores.get(task, {})
            if metric_name in task_scores:
                seed_vals.append(task_scores[metric_name])
                seed_stderrs.append(task_scores.get(se_key, 0.0))

        if seed_vals:
            n = len(seed_vals)
            mean_val = statistics.mean(seed_vals)
            std_val = statistics.stdev(seed_vals) if n > 1 else 0.0
            # SE of the arm's mean estimate:
            # - n > 1: std/sqrt(n) captures both training and eval variance
            # - n == 1: fall back to lm-eval stderr (eval sampling noise)
            if n > 1:
                se_mean = std_val / math.sqrt(n)
            else:
                se_mean = seed_stderrs[0] if seed_stderrs else 0.0
            agg[task] = {
                "metric": metric_name,
                "baseline": base_val,
                "baseline_stderr": base_stderr,
                "mean": mean_val,
                "std": std_val,
                "se_mean": se_mean,
                "n_seeds": n,
                "per_seed": seed_vals,
                "per_seed_stderr": seed_stderrs,
                "delta_vs_baseline": mean_val - base_val,
            }

    return agg


# ---------------------------------------------------------------------------
# Single model experiment (multi-seed, three-arm)
# ---------------------------------------------------------------------------
def run_model_experiment(
    model_key: str,
    model_spec: dict,
    output_dir: Path,
    seeds: list[int],
    quick: bool = False,
    skip_eval: bool = False,
    skip_inference: bool = False,
    skip_grid: bool = False,
    weight_decay: float = 0.0,
    derive_wd_test: bool = False,
) -> dict:
    """Run full three-arm comparison for one model across multiple seeds."""
    model_path = str(model_spec["path"])
    model_name = model_spec["name"]
    num_layers = model_spec["num_layers"]

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL: {model_name} ({model_key})")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Quick: {quick}, Skip eval: {skip_eval}, Skip grid: {skip_grid}")
    logger.info(f"{'='*70}")

    results = {
        "model": model_name,
        "model_path": model_path,
        "seeds": seeds,
        "quick": quick,
        "weight_decay": weight_decay,
        "timestamp": datetime.now().isoformat(),
    }
    eval_limit = 50 if quick else None
    iters = 100 if quick else STANDARD_LORA_CONFIG["iters"]
    grid_iters = 50 if quick else 200

    # Prepare data dir for mlx-lm
    std_data_dir = model_dir / "_std_data"
    prepare_standard_lora_data_dir(TRAIN_DATA, VAL_DATA, std_data_dir)

    # ------------------------------------------------------------------
    # a. BASELINE EVAL (once)
    # ------------------------------------------------------------------
    if not skip_eval:
        logger.info(f"\n--- [{model_name}] Baseline Evaluation ---")
        baseline_scores = run_lm_eval(
            model_path=model_path,
            tasks=BENCHMARK_TASKS,
            limit=eval_limit,
        )
    else:
        baseline_scores = {}
    results["baseline"] = baseline_scores
    with open(model_dir / "baseline_eval.json", "w") as f:
        json.dump(baseline_scores, f, indent=2)

    # ------------------------------------------------------------------
    # b. GRID SEARCH for tuned arm (once, seed=42)
    # ------------------------------------------------------------------
    if not skip_grid:
        logger.info(f"\n--- [{model_name}] Grid Search for Tuned Arm ---")
        grid_summary = run_grid_search(
            model_spec=model_spec,
            data_dir=std_data_dir,
            output_dir=model_dir,
            grid=TUNED_GRID,
            grid_iters=grid_iters,
        )
        results["grid_search"] = grid_summary
        best_tuned_config = STANDARD_LORA_CONFIG.copy()
        best_tuned_config.update(grid_summary["best_config"])
    else:
        grid_summary = None
        results["grid_search"] = "skipped"
        best_tuned_config = None

    # ------------------------------------------------------------------
    # c. PER-SEED TRAINING + EVAL
    # ------------------------------------------------------------------
    # Collect per-arm, per-seed results for aggregation
    # When --derive-wd-test, control arm is always WD=0
    nb_weight_decay = 0.0 if derive_wd_test else weight_decay

    std_evals = []
    tuned_evals = []
    nb_evals = []
    std_trainings = []
    tuned_trainings = []
    nb_trainings = []
    nb_wd_evals = []       # WD=derived arm (only when derive_wd_test)
    nb_wd_trainings = []   # WD=derived arm (only when derive_wd_test)
    wd_derived_value = None
    wd_derivation_info = None
    per_seed_data = {}

    for seed in seeds:
        logger.info(f"\n{'='*50}")
        logger.info(f"[{model_name}] Seed {seed}")
        logger.info(f"{'='*50}")

        seed_dir = model_dir / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_result = {"seed": seed}

        # ---- Standard LoRA ----
        logger.info(f"\n--- [{model_name}] Standard LoRA (seed={seed}) ---")
        std_out = seed_dir / "standard_lora"
        try:
            std_train = train_standard_lora(
                model_path=model_path,
                num_layers=num_layers,
                data_dir=std_data_dir,
                output_dir=std_out,
                config=STANDARD_LORA_CONFIG,
                seed=seed,
                iters_override=iters,
            )
            # Commensurable val_loss: full val set, batch_size=1, same operator
            std_train["callback_val_loss"] = std_train.get("final_val_loss")
            logger.info("  Commensurable val_loss eval (standard)...")
            std_train["final_val_loss"] = evaluate_val_loss(
                model_path, str(std_out), VAL_DATA,
            )
            logger.info(f"  val_loss={std_train['final_val_loss']:.4f} "
                        f"(callback={std_train['callback_val_loss']})")
            with open(std_out / "training_log.json", "w") as f:
                json.dump(std_train, f, indent=2, default=str)
            std_trainings.append(std_train)

            if not skip_eval:
                std_eval = run_lm_eval(
                    model_path=model_path,
                    tasks=BENCHMARK_TASKS,
                    adapter_path=str(std_out),
                    limit=eval_limit,
                )
            else:
                std_eval = {}
            with open(std_out / "eval_results.json", "w") as f:
                json.dump(std_eval, f, indent=2)
            std_evals.append(std_eval)
            seed_result["standard_lora"] = {"training": std_train, "eval": std_eval}
        except Exception as e:
            logger.exception(f"  Standard LoRA seed={seed} failed")
            seed_result["standard_lora"] = {"error": str(e)}

        # ---- Tuned LoRA ----
        if best_tuned_config is not None:
            logger.info(f"\n--- [{model_name}] Tuned LoRA (seed={seed}) ---")
            tuned_out = seed_dir / "tuned_lora"
            try:
                tuned_train = train_standard_lora(
                    model_path=model_path,
                    num_layers=num_layers,
                    data_dir=std_data_dir,
                    output_dir=tuned_out,
                    config=best_tuned_config,
                    seed=seed,
                    iters_override=iters,
                )
                # Commensurable val_loss
                tuned_train["callback_val_loss"] = tuned_train.get("final_val_loss")
                logger.info("  Commensurable val_loss eval (tuned)...")
                tuned_train["final_val_loss"] = evaluate_val_loss(
                    model_path, str(tuned_out), VAL_DATA,
                )
                logger.info(f"  val_loss={tuned_train['final_val_loss']:.4f} "
                            f"(callback={tuned_train['callback_val_loss']})")
                with open(tuned_out / "training_log.json", "w") as f:
                    json.dump(tuned_train, f, indent=2, default=str)
                tuned_trainings.append(tuned_train)

                if not skip_eval:
                    tuned_eval = run_lm_eval(
                        model_path=model_path,
                        tasks=BENCHMARK_TASKS,
                        adapter_path=str(tuned_out),
                        limit=eval_limit,
                    )
                else:
                    tuned_eval = {}
                with open(tuned_out / "eval_results.json", "w") as f:
                    json.dump(tuned_eval, f, indent=2)
                tuned_evals.append(tuned_eval)
                seed_result["tuned_lora"] = {"training": tuned_train, "eval": tuned_eval}
            except Exception as e:
                logger.exception(f"  Tuned LoRA seed={seed} failed")
                seed_result["tuned_lora"] = {"error": str(e)}

        # ---- NB-LoRA ----
        # NB-LoRA always runs to convergence certificate (no iteration cap).
        # Budget-matching by iteration count is not meaningful: NB-LoRA and
        # standard LoRA use different batch sizes, gradient accumulation,
        # and stopping criteria.  The valid comparison is convergence-to-
        # convergence: both arms run until their natural stopping point.
        logger.info(f"\n--- [{model_name}] NB-LoRA (seed={seed}) ---")
        nb_out = seed_dir / "nb_lora"
        try:
            nb_train = train_nb_lora(
                model_path=model_path,
                output_dir=nb_out,
                seed=seed,
                weight_decay=nb_weight_decay,
            )
            # Commensurable val_loss: same operator as standard arm
            logger.info("  Commensurable val_loss eval (NB-LoRA)...")
            nb_train["final_val_loss"] = evaluate_val_loss(
                model_path, str(nb_out), VAL_DATA,
            )
            logger.info(f"  val_loss={nb_train['final_val_loss']:.4f} "
                        f"(training post_loss={nb_train.get('post_loss_training')})")
            with open(nb_out / "training_log.json", "w") as f:
                json.dump(nb_train, f, indent=2, default=str)
            nb_trainings.append(nb_train)

            if not skip_eval:
                nb_eval = run_lm_eval(
                    model_path=model_path,
                    tasks=BENCHMARK_TASKS,
                    adapter_path=str(nb_out),
                    limit=eval_limit,
                )
            else:
                nb_eval = {}
            with open(nb_out / "eval_results.json", "w") as f:
                json.dump(nb_eval, f, indent=2)
            nb_evals.append(nb_eval)
            seed_result["nb_lora"] = {"training": nb_train, "eval": nb_eval}
        except Exception as e:
            logger.exception(f"  NB-LoRA seed={seed} failed")
            seed_result["nb_lora"] = {"error": str(e)}

        # ---- NB-LoRA WD=derived (if --derive-wd-test) ----
        if derive_wd_test and "error" not in seed_result.get("nb_lora", {}):
            nb_train_wd0 = seed_result["nb_lora"]["training"]
            try:
                wd_derived, wd_deriv = derive_wd_from_training_result(
                    nb_train_wd0, str(nb_out),
                )
                # Store derived value (same across seeds by construction,
                # but may vary slightly — keep per-seed for transparency)
                wd_derived_value = wd_derived
                wd_derivation_info = wd_deriv
                logger.info(
                    f"  WD derived: λ={wd_derived:.6f} "
                    f"(median_d_norm={wd_deriv['median_d_norm']:.6f}, "
                    f"||θ||={wd_deriv['param_norm']:.4f})"
                )

                nb_out_wd = seed_dir / "nb_lora_wd_derived"
                logger.info(f"\n--- [{model_name}] NB-LoRA WD={wd_derived:.6f} (seed={seed}) ---")
                nb_train_wd = train_nb_lora(
                    model_path=model_path,
                    output_dir=nb_out_wd,
                    seed=seed,
                    weight_decay=wd_derived,
                )
                # Commensurable val_loss: same operator as all other arms
                logger.info("  Commensurable val_loss eval (NB-LoRA WD=derived)...")
                nb_train_wd["final_val_loss"] = evaluate_val_loss(
                    model_path, str(nb_out_wd), VAL_DATA,
                )
                nb_train_wd["wd_derived"] = wd_derived
                nb_train_wd["wd_derivation"] = wd_deriv
                logger.info(
                    f"  val_loss={nb_train_wd['final_val_loss']:.4f} "
                    f"(WD={wd_derived:.6f})"
                )
                with open(nb_out_wd / "training_log.json", "w") as f:
                    json.dump(nb_train_wd, f, indent=2, default=str)
                nb_wd_trainings.append(nb_train_wd)

                if not skip_eval:
                    nb_wd_eval = run_lm_eval(
                        model_path=model_path,
                        tasks=BENCHMARK_TASKS,
                        adapter_path=str(nb_out_wd),
                        limit=eval_limit,
                    )
                else:
                    nb_wd_eval = {}
                with open(nb_out_wd / "eval_results.json", "w") as f:
                    json.dump(nb_wd_eval, f, indent=2)
                nb_wd_evals.append(nb_wd_eval)
                seed_result["nb_lora_wd_derived"] = {
                    "training": nb_train_wd, "eval": nb_wd_eval,
                }
            except Exception as e:
                logger.exception(f"  NB-LoRA WD=derived seed={seed} failed")
                seed_result["nb_lora_wd_derived"] = {"error": str(e)}

        # ---- Inference (non-fatal) ----
        if not skip_inference:
            try:
                logger.info(f"\n--- [{model_name}] Inference Tests (seed={seed}) ---")
                prompts = load_jsonl(INFERENCE_PROMPTS)

                base_responses = generate_inference_responses(
                    model_path=model_path,
                    adapter_path=None,
                    prompts=prompts,
                    label="base",
                )

                std_adapter = str(std_out) if "error" not in seed_result.get("standard_lora", {}) else None
                std_responses = (
                    generate_inference_responses(model_path, std_adapter, prompts, "standard_lora")
                    if std_adapter else []
                )

                tuned_adapter = (
                    str(seed_dir / "tuned_lora")
                    if best_tuned_config and "error" not in seed_result.get("tuned_lora", {})
                    else None
                )
                tuned_responses = (
                    generate_inference_responses(model_path, tuned_adapter, prompts, "tuned_lora")
                    if tuned_adapter else []
                )

                nb_adapter = str(nb_out) if "error" not in seed_result.get("nb_lora", {}) else None
                nb_responses = (
                    generate_inference_responses(model_path, nb_adapter, prompts, "nb_lora")
                    if nb_adapter else []
                )

                inference_data = {"prompts": []}
                for prompt_data in prompts:
                    pid = prompt_data["id"]
                    entry = {
                        "id": pid,
                        "category": prompt_data.get("category", ""),
                        "distribution": prompt_data.get("distribution", ""),
                        "prompt": prompt_data["prompt"],
                        "expected": prompt_data.get("expected", ""),
                        "responses": {
                            "base": next((r["response"] for r in base_responses if r["id"] == pid), ""),
                            "standard_lora": next((r["response"] for r in std_responses if r["id"] == pid), ""),
                            "tuned_lora": next((r["response"] for r in tuned_responses if r["id"] == pid), ""),
                            "nb_lora": next((r["response"] for r in nb_responses if r["id"] == pid), ""),
                        },
                    }
                    inference_data["prompts"].append(entry)

                seed_result["inference"] = inference_data
                with open(seed_dir / "inference_responses.json", "w") as f:
                    json.dump(inference_data, f, indent=2)
            except Exception as e:
                logger.exception(f"  Inference seed={seed} failed (non-fatal)")
                seed_result["inference"] = {"error": str(e)}

        per_seed_data[str(seed)] = seed_result

    results["per_seed"] = per_seed_data

    # ------------------------------------------------------------------
    # d. AGGREGATE + COMPARE
    # ------------------------------------------------------------------
    logger.info(f"\n--- [{model_name}] Aggregation ---")

    comparison = build_comparison(
        baseline_scores=baseline_scores,
        std_evals=std_evals,
        tuned_evals=tuned_evals,
        nb_evals=nb_evals,
        std_trainings=std_trainings,
        tuned_trainings=tuned_trainings,
        nb_trainings=nb_trainings,
        grid_summary=grid_summary,
        quick=quick,
    )
    results["comparison"] = comparison

    # WD falsification (opt-in)
    if derive_wd_test and nb_wd_trainings and wd_derived_value is not None:
        wd_falsification = build_wd_falsification(
            baseline_scores=baseline_scores,
            nb_wd0_evals=nb_evals,
            nb_wdD_evals=nb_wd_evals,
            nb_wd0_trainings=nb_trainings,
            nb_wdD_trainings=nb_wd_trainings,
            wd_derived=wd_derived_value,
            wd_derivation=wd_derivation_info,
        )
        comparison["weight_decay_falsification"] = wd_falsification
        results["comparison"] = comparison

        logger.info(f"\n  WD Falsification: {wd_falsification['verdict']}")
        logger.info(f"    λ_derived={wd_derived_value:.6f}, "
                     f"tasks_where_wd_wins={wd_falsification['tasks_where_wd_wins']}")

    with open(model_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_comparison(model_name, comparison)

    # Cleanup temp data dir
    shutil.rmtree(std_data_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Comparison builder
# ---------------------------------------------------------------------------
def _safe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _safe_stdev(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def build_wd_falsification(
    baseline_scores: dict,
    nb_wd0_evals: list[dict],
    nb_wdD_evals: list[dict],
    nb_wd0_trainings: list[dict],
    nb_wdD_trainings: list[dict],
    wd_derived: float,
    wd_derivation: dict,
) -> dict:
    """Build WD falsification verdict: WD=0 vs WD=derived.

    Prediction: WD=0 is within tie band of WD=derived on all tasks.
    Falsifier: WD=derived beats WD=0 by more than tie band on >=2/7 tasks.
    """
    wd0_agg = aggregate_arm_evals(nb_wd0_evals, baseline_scores)
    wdD_agg = aggregate_arm_evals(nb_wdD_evals, baseline_scores)

    per_task = {}
    tasks_where_wd_wins = 0

    for task in BENCHMARK_TASKS:
        if task not in wd0_agg or task not in wdD_agg:
            continue
        wd0_mean = wd0_agg[task]["mean"]
        wdD_mean = wdD_agg[task]["mean"]
        wd0_se = wd0_agg[task]["se_mean"]
        wdD_se = wdD_agg[task]["se_mean"]
        tie_band = 2.0 * math.sqrt(wd0_se**2 + wdD_se**2)
        delta = wdD_mean - wd0_mean  # positive = WD=derived wins

        if delta > tie_band:
            verdict = "wd_derived_wins"
            tasks_where_wd_wins += 1
        elif delta < -tie_band:
            verdict = "wd0_wins"
        else:
            verdict = "tie"

        per_task[task] = {
            "wd0_mean": wd0_mean,
            "wdD_mean": wdD_mean,
            "wd0_se": wd0_se,
            "wdD_se": wdD_se,
            "tie_band": tie_band,
            "delta": delta,
            "verdict": verdict,
        }

    falsified = tasks_where_wd_wins >= 2
    if falsified:
        overall_verdict = (
            f"FALSIFIED — WD=derived wins on {tasks_where_wd_wins}/7 tasks "
            f"(>= 2 required)"
        )
    else:
        overall_verdict = "WD=0 optimal — prediction confirmed"

    # Training stats comparison
    wd0_val_losses = [
        t["final_val_loss"] for t in nb_wd0_trainings
        if t.get("final_val_loss") is not None
    ]
    wdD_val_losses = [
        t["final_val_loss"] for t in nb_wdD_trainings
        if t.get("final_val_loss") is not None
    ]

    return {
        "prediction": "WD=0 optimal (MASS+stopping bounds displacement)",
        "wd_derived": wd_derived,
        "wd_derivation": wd_derivation,
        "per_task": per_task,
        "tasks_where_wd_wins": tasks_where_wd_wins,
        "falsified": falsified,
        "verdict": overall_verdict,
        "training_val_loss": {
            "wd0_mean": _safe_mean(wd0_val_losses),
            "wdD_mean": _safe_mean(wdD_val_losses),
        },
    }


def build_comparison(
    baseline_scores: dict,
    std_evals: list[dict],
    tuned_evals: list[dict],
    nb_evals: list[dict],
    std_trainings: list[dict],
    tuned_trainings: list[dict],
    nb_trainings: list[dict],
    grid_summary: dict | None,
    quick: bool = False,
) -> dict:
    """Build multi-seed comparison across three arms."""
    comparison = {
        "benchmarks": {},
        "training": {},
        "spectral": {},
        "head_to_head": {},
    }

    # Aggregate benchmarks per arm
    std_agg = aggregate_arm_evals(std_evals, baseline_scores)
    tuned_agg = aggregate_arm_evals(tuned_evals, baseline_scores)
    nb_agg = aggregate_arm_evals(nb_evals, baseline_scores)

    for task in BENCHMARK_TASKS:
        entry = {"task": task}
        if task in std_agg:
            entry["baseline"] = std_agg[task]["baseline"]
            entry["metric"] = std_agg[task]["metric"]
        elif task in nb_agg:
            entry["baseline"] = nb_agg[task]["baseline"]
            entry["metric"] = nb_agg[task]["metric"]
        else:
            continue

        for arm_name, arm_agg in [
            ("standard_lora", std_agg),
            ("tuned_lora", tuned_agg),
            ("nb_lora", nb_agg),
        ]:
            if task in arm_agg:
                entry[arm_name] = {
                    "mean": arm_agg[task]["mean"],
                    "std": arm_agg[task]["std"],
                    "n_seeds": arm_agg[task]["n_seeds"],
                    "delta_vs_baseline": arm_agg[task]["delta_vs_baseline"],
                }

        comparison["benchmarks"][task] = entry

    # Training stats
    for arm_name, trainings in [
        ("standard_lora", std_trainings),
        ("tuned_lora", tuned_trainings),
        ("nb_lora", nb_trainings),
    ]:
        times = [t.get("training_time_seconds", 0) for t in trainings]
        val_losses = [t.get("final_val_loss", 0) for t in trainings if t.get("final_val_loss") is not None]
        comparison["training"][arm_name] = {
            "n_runs": len(trainings),
            "training_time_mean": _safe_mean(times),
            "training_time_std": _safe_stdev(times),
            "final_val_loss_mean": _safe_mean(val_losses),
            "final_val_loss_std": _safe_stdev(val_losses),
            "hyperparameter_count": trainings[0].get("hyperparameter_count", 0) if trainings else 0,
        }

    # Spectral safety
    std_max_norms = [t.get("spectral_info", {}).get("max_spectral_norm", 0) for t in std_trainings]
    tuned_max_norms = [t.get("spectral_info", {}).get("max_spectral_norm", 0) for t in tuned_trainings]
    comparison["spectral"] = {
        "standard_lora": {
            "max_norm_mean": _safe_mean(std_max_norms),
            "bounded_by_construction": False,
        },
        "tuned_lora": {
            "max_norm_mean": _safe_mean(tuned_max_norms),
            "bounded_by_construction": False,
        },
        "nb_lora": {
            "bounded_by_construction": True,
            "max_spectral_ratio_mean": _safe_mean([
                t.get("max_spectral_ratio", 0) for t in nb_trainings
            ]),
            "spectral_bounds_ok": all(
                t.get("spectral_bounds_ok", False) for t in nb_trainings
            ),
        },
    }

    # Head-to-head: NB vs Standard, NB vs Tuned
    # Per-task tie band from lm-eval stderr.  Each arm's mean has SE
    # (se_mean) from aggregate_arm_evals.  The difference of two
    # independent estimates has SE_diff = sqrt(se1² + se2²).  The 95%
    # CI for the difference is 2 × SE_diff.
    for opponent_name, opponent_agg in [
        ("standard_lora", std_agg),
        ("tuned_lora", tuned_agg),
    ]:
        nb_wins = 0
        opponent_wins = 0
        ties = 0
        deltas = []
        per_task_bands = {}

        for task in BENCHMARK_TASKS:
            if task in nb_agg and task in opponent_agg:
                nb_mean = nb_agg[task]["mean"]
                opp_mean = opponent_agg[task]["mean"]
                nb_se = nb_agg[task]["se_mean"]
                opp_se = opponent_agg[task]["se_mean"]
                tie_band = 2.0 * math.sqrt(nb_se**2 + opp_se**2)
                delta = nb_mean - opp_mean
                deltas.append(delta)
                per_task_bands[task] = {
                    "nb_se": nb_se,
                    "opp_se": opp_se,
                    "tie_band": tie_band,
                }
                if abs(delta) < tie_band:
                    ties += 1
                elif delta > 0:
                    nb_wins += 1
                else:
                    opponent_wins += 1

        key = f"nb_vs_{opponent_name}"
        comparison["head_to_head"][key] = {
            "nb_wins": nb_wins,
            "opponent_wins": opponent_wins,
            "ties": ties,
            "mean_delta": _safe_mean(deltas),
            "total_tasks": len(deltas),
            "tie_band_derivation": "per-task: 2*sqrt(se_nb² + se_opp²)",
            "per_task_tie_bands": per_task_bands,
        }

    # Grid search info
    if grid_summary:
        comparison["grid_search"] = {
            "best_config": grid_summary.get("best_config"),
            "best_val_loss": grid_summary.get("best_val_loss"),
            "n_configs_tested": len(grid_summary.get("grid_configs", [])),
        }

    return comparison


# ---------------------------------------------------------------------------
# Print comparison
# ---------------------------------------------------------------------------
def print_comparison(model_name: str, comparison: dict):
    """Print human-readable multi-seed comparison."""
    benchmarks = comparison.get("benchmarks", {})
    training = comparison.get("training", {})
    h2h = comparison.get("head_to_head", {})

    print(f"\n{'='*80}")
    print(f"  COMPARISON: {model_name}")
    print(f"{'='*80}")

    # Header
    has_tuned = any("tuned_lora" in v for v in benchmarks.values())
    if has_tuned:
        print(f"  {'Task':<18} {'Base':>8} {'Std LoRA':>14} {'Tuned LoRA':>14} {'NB-LoRA':>14}")
    else:
        print(f"  {'Task':<18} {'Base':>8} {'Std LoRA':>14} {'NB-LoRA':>14}")
    print(f"  {'-'*72}")

    for task, data in benchmarks.items():
        base = data.get("baseline", 0)
        std = data.get("standard_lora", {})
        tuned = data.get("tuned_lora", {})
        nb = data.get("nb_lora", {})

        std_str = f"{std.get('mean', 0):.4f}+/-{std.get('std', 0):.3f}" if std else "---"
        nb_str = f"{nb.get('mean', 0):.4f}+/-{nb.get('std', 0):.3f}" if nb else "---"

        if has_tuned:
            tuned_str = f"{tuned.get('mean', 0):.4f}+/-{tuned.get('std', 0):.3f}" if tuned else "---"
            print(f"  {task:<18} {base:>8.4f} {std_str:>14} {tuned_str:>14} {nb_str:>14}")
        else:
            print(f"  {task:<18} {base:>8.4f} {std_str:>14} {nb_str:>14}")

    # Training stats
    print(f"\n  Training:")
    for arm_name in ["standard_lora", "tuned_lora", "nb_lora"]:
        t = training.get(arm_name, {})
        if t.get("n_runs", 0) == 0:
            continue
        hp = t.get("hyperparameter_count", "?")
        time_str = f"{t['training_time_mean']:.1f}+/-{t['training_time_std']:.1f}s"
        vloss_str = f"{t['final_val_loss_mean']:.4f}+/-{t['final_val_loss_std']:.4f}"
        label = {"standard_lora": "Standard", "tuned_lora": "Tuned", "nb_lora": "NB-LoRA"}[arm_name]
        print(f"    {label:<12} time={time_str:<18} val_loss={vloss_str:<20} HPs={hp}")

    # Head-to-head
    print(f"\n  Head-to-Head:")
    for key, h in h2h.items():
        label = key.replace("nb_vs_", "NB vs ")
        print(f"    {label}: NB wins {h['nb_wins']}, "
              f"opponent wins {h['opponent_wins']}, "
              f"ties {h['ties']}, "
              f"mean delta {h['mean_delta']:+.4f}")

    # Grid search
    gs = comparison.get("grid_search", {})
    if gs:
        bc = gs.get("best_config", {})
        print(f"\n  Grid Search ({gs.get('n_configs_tested', '?')} configs):")
        print(f"    Best: rank={bc.get('rank')}, lr={bc.get('lr')}, scale={bc.get('scale', 0):.2f}")
        print(f"    Best val loss: {gs.get('best_val_loss', '?')}")

    # WD falsification
    wdf = comparison.get("weight_decay_falsification")
    if wdf:
        print(f"\n  WD Falsification:")
        print(f"    Prediction: {wdf['prediction']}")
        print(f"    WD derived: {wdf['wd_derived']:.6f} ({wdf['wd_derivation']['formula']})")
        print(f"    Val loss: WD=0 {wdf['training_val_loss']['wd0_mean']:.4f}, "
              f"WD=derived {wdf['training_val_loss']['wdD_mean']:.4f}")
        for task, td in wdf.get("per_task", {}).items():
            print(f"    {task:<18} WD=0={td['wd0_mean']:.4f}  WD=D={td['wdD_mean']:.4f}  "
                  f"band={td['tie_band']:.4f}  {td['verdict']}")
        print(f"    Tasks where WD wins: {wdf['tasks_where_wd_wins']}")
        print(f"    Verdict: {wdf['verdict']}")

    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def build_summary(all_results: dict) -> dict:
    """Build cross-model summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "overall": {},
    }

    all_nb_vs_std_deltas = []
    all_nb_vs_tuned_deltas = []

    for model_key, model_results in all_results.items():
        comparison = model_results.get("comparison", {})
        h2h = comparison.get("head_to_head", {})

        summary["models"][model_key] = {
            "benchmarks": comparison.get("benchmarks", {}),
            "head_to_head": h2h,
            "training": comparison.get("training", {}),
        }

        nb_vs_std = h2h.get("nb_vs_standard_lora", {})
        nb_vs_tuned = h2h.get("nb_vs_tuned_lora", {})

        if nb_vs_std.get("mean_delta") is not None:
            all_nb_vs_std_deltas.append(nb_vs_std["mean_delta"])
        if nb_vs_tuned.get("mean_delta") is not None:
            all_nb_vs_tuned_deltas.append(nb_vs_tuned["mean_delta"])

    if all_nb_vs_std_deltas:
        summary["overall"]["nb_vs_standard_mean_delta"] = _safe_mean(all_nb_vs_std_deltas)
    if all_nb_vs_tuned_deltas:
        summary["overall"]["nb_vs_tuned_mean_delta"] = _safe_mean(all_nb_vs_tuned_deltas)

    # Verdict: aggregate per-task measured SEs from all models.
    # For each (model, task) pair, se_delta = sqrt(nb_se² + opp_se²).
    # Overall mean delta SE = sqrt(Σ se_delta²) / N.
    # Tie band = 2 × SE_overall (95% CI for the difference).
    se_delta_sq_all: list[float] = []
    for model_key, model_results in all_results.items():
        h2h = model_results.get("comparison", {}).get("head_to_head", {})
        nb_vs_std = h2h.get("nb_vs_standard_lora", {})
        for task, band in nb_vs_std.get("per_task_tie_bands", {}).items():
            se_d = math.sqrt(band["nb_se"] ** 2 + band["opp_se"] ** 2)
            se_delta_sq_all.append(se_d ** 2)

    if all_nb_vs_std_deltas and se_delta_sq_all:
        n_pairs = len(se_delta_sq_all)
        agg_se = math.sqrt(sum(se_delta_sq_all)) / n_pairs
        agg_tie_band = 2.0 * agg_se
        mean_delta = _safe_mean(all_nb_vs_std_deltas)
        if mean_delta > agg_tie_band:
            summary["overall"]["verdict"] = "NB-LoRA > Standard"
        elif mean_delta < -agg_tie_band:
            summary["overall"]["verdict"] = "Standard LoRA > NB-LoRA"
        else:
            summary["overall"]["verdict"] = "Within measurement noise"
        summary["overall"]["verdict_threshold"] = agg_tie_band
        summary["overall"]["verdict_derivation"] = (
            f"2*sqrt(Σse²)/{n_pairs} from {n_pairs} measured (model,task) pairs"
        )
    elif all_nb_vs_std_deltas:
        summary["overall"]["verdict"] = "no stderr data for threshold"
    else:
        summary["overall"]["verdict"] = "no data"

    # WD falsification summary (include if any model has it)
    wd_falsifications = {}
    for model_key, model_results in all_results.items():
        wdf = model_results.get("comparison", {}).get("weight_decay_falsification")
        if wdf:
            wd_falsifications[model_key] = {
                "wd_derived": wdf["wd_derived"],
                "tasks_where_wd_wins": wdf["tasks_where_wd_wins"],
                "falsified": wdf["falsified"],
                "verdict": wdf["verdict"],
            }
    if wd_falsifications:
        summary["weight_decay_falsification"] = wd_falsifications

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NB-LoRA vs Standard LoRA — three-arm head-to-head comparison",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS.keys()),
        default=["350M"],
        help="Model scales to test (default: 350M)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Seeds for multi-seed runs (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 seed, fewer iters, limited eval samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: results/nblora_vs_standard/)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip lm-eval benchmarks (for debugging training pipeline)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference response generation",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip grid search (omit tuned arm)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW-decoupled weight decay for NB-LoRA arm (default: 0.0)",
    )
    parser.add_argument(
        "--derive-wd-test",
        action="store_true",
        help="Run WD redundancy falsification: WD=0 vs WD=derived",
    )
    args = parser.parse_args()

    # Quick mode: 1 seed
    if args.quick and args.seeds == DEFAULT_SEEDS:
        args.seeds = [42]

    # Validate volume
    if not VOLUME.exists():
        logger.error(f"Volume not mounted: {VOLUME}")
        sys.exit(1)

    # Validate models
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        if not spec["path"].exists():
            logger.error(f"Model not found: {spec['path']}")
            sys.exit(1)

    # Validate data
    for path in [TRAIN_DATA, VAL_DATA, INFERENCE_PROMPTS]:
        if not path.exists():
            logger.error(f"Data file not found: {path}")
            sys.exit(1)

    # Validate training data format
    with open(TRAIN_DATA) as f:
        first_line = f.readline().strip()
        if first_line:
            record = json.loads(first_line)
            if "text" not in record:
                logger.error(f"Training data missing 'text' field. Keys: {list(record.keys())}")
                sys.exit(1)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "results" / "nblora_vs_standard"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("NB-LoRA vs Standard LoRA — Head-to-Head Comparison (R1)")
    logger.info("=" * 70)
    logger.info(f"Models: {args.models}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Quick: {args.quick}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Standard LoRA config: {STANDARD_LORA_CONFIG}")
    logger.info(f"Train: {TRAIN_DATA} ({sum(1 for _ in open(TRAIN_DATA))} samples)")
    logger.info(f"Val: {VAL_DATA} ({sum(1 for _ in open(VAL_DATA))} samples)")

    # Run experiments
    all_results = {}
    total_t0 = time.time()

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        try:
            result = run_model_experiment(
                model_key=model_key,
                model_spec=spec,
                output_dir=output_dir,
                seeds=args.seeds,
                quick=args.quick,
                skip_eval=args.skip_eval,
                skip_inference=args.skip_inference,
                skip_grid=args.skip_grid,
                weight_decay=args.weight_decay,
                derive_wd_test=args.derive_wd_test,
            )
            all_results[spec["name"]] = result
        except Exception:
            logger.exception(f"FAILED on model {model_key}")
            all_results[spec["name"]] = {"error": True}

    total_time = time.time() - total_t0

    # Build and save summary
    summary = build_summary(all_results)
    summary["total_time_seconds"] = total_time
    summary["config"] = {
        "standard_lora": STANDARD_LORA_CONFIG,
        "lora_alpha": LORA_ALPHA,
        "tuned_grid": TUNED_GRID,
        "seeds": args.seeds,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nTotal experiment time: {total_time:.1f}s")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  summary.json — cross-model summary")
    logger.info(f"  full_results.json — all raw data")
    logger.info(f"  <model>/comparison.json — per-model comparison")


if __name__ == "__main__":
    main()
