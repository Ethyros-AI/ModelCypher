#!/usr/bin/env python3
"""NB-LoRA vs Standard LoRA — Controlled Experiment.

Compares geometry-derived NB-LoRA (zero hyperparameter decisions) against
standard LoRA with mlx-lm defaults across 3 model scales.

Measurements:
  - lm-eval-harness benchmark scores (8 tasks)
  - Training loss convergence
  - Spectral safety (post-hoc for standard, by construction for NB-LoRA)
  - Perplexity (pre/post training)
  - Inference test responses for Claude evaluation

Usage:
  # Quick smoke test (350M only, limited evals)
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M --quick

  # Full run (all 3 scales)
  poetry run python scripts/experiment_nblora_vs_standard.py

  # Specific models
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M 700M
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Fix lm-eval / transformers 5.x incompatibility
#   transformers 5.0 removed AutoModelForVision2Seq; lm-eval 0.4.9 references it.
#   Patch the LazyModule __getattr__ before any lm_eval import.
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

# Silence noisy sub-loggers
for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOLUME = Path("/Volumes/CodeCypher")
MODELS_DIR = VOLUME / "models" / "mlx-community"
EXPERIMENTS_DIR = VOLUME / "experiments"

MODEL_SPECS = {
    "350M": {
        "path": MODELS_DIR / "LFM2-350M-MLX-bf16",
        "name": "LFM2-350M",
        "num_layers": 24,
    },
    "700M": {
        "path": MODELS_DIR / "LFM2-700M-bf16",
        "name": "LFM2-700M",
        "num_layers": 32,
    },
    "1.2B": {
        "path": MODELS_DIR / "LFM2-1.2B-bf16",
        "name": "LFM2-1.2B",
        "num_layers": 32,
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

# Standard LoRA defaults from mlx-lm CONFIG_DEFAULTS
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


def convert_to_text_format(samples: list[dict]) -> list[dict]:
    """Ensure all samples have 'text' key (convert prompt/completion if needed)."""
    converted = []
    for s in samples:
        if "text" in s:
            converted.append(s)
        elif "prompt" in s and "completion" in s:
            converted.append({"text": s["prompt"] + "\n\n" + s["completion"]})
    return converted


def prepare_standard_lora_data_dir(train_path: Path, val_path: Path, tmp_dir: Path):
    """mlx_lm.lora.load_dataset expects a directory with train.jsonl/valid.jsonl."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_path, tmp_dir / "train.jsonl")
    shutil.copy2(val_path, tmp_dir / "valid.jsonl")
    return tmp_dir


# ---------------------------------------------------------------------------
# lm-eval wrapper (uses mlx_lm's MLXLM which registers with lm_eval)
# ---------------------------------------------------------------------------
def run_lm_eval(
    model_path: str,
    tasks: list[str],
    adapter_path: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run lm-eval benchmarks via mlx_lm's MLXLM integration.

    Returns dict mapping task_name -> {metric_name: value}.
    """
    import mlx.core as mx
    from lm_eval import simple_evaluate
    from mlx_lm.evaluate import MLXLM

    # Silence tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info(f"  lm-eval: {', '.join(tasks)} (adapter={adapter_path or 'none'})")

    lm = MLXLM(path_or_hf_repo=model_path, batch_size=1)

    # Load adapter if provided
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

    # Extract scores
    scores = {}
    for task_name, task_results in results.get("results", {}).items():
        scores[task_name] = {
            k: v for k, v in task_results.items() if not k.startswith("alias")
        }

    # Cleanup
    del lm
    mx.clear_cache()
    gc.collect()

    return scores


# ---------------------------------------------------------------------------
# Standard LoRA training (mlx-lm)
# ---------------------------------------------------------------------------
def train_standard_lora(
    model_path: str,
    num_layers: int,
    data_dir: Path,
    output_dir: Path,
    quick: bool = False,
) -> dict:
    """Train standard LoRA using mlx-lm defaults. Returns training metadata."""
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm.lora import (
        TrainingArgs,
        load_dataset,
        linear_to_lora_layers,
        print_trainable_parameters,
        save_config,
        train,
    )
    from mlx_lm.tuner.datasets import CacheDataset

    logger.info("  Standard LoRA: loading model...")
    model, tokenizer = mlx_load(str(model_path))

    # Build args namespace matching what mlx_lm.lora expects
    iters = 100 if quick else STANDARD_LORA_CONFIG["iters"]
    args = SimpleNamespace(
        model=str(model_path),
        data=str(data_dir),
        train=True,
        test=False,
        fine_tune_type="lora",
        optimizer=STANDARD_LORA_CONFIG["optimizer"],
        optimizer_config={"adam": {}, "adamw": {}, "sgd": {}},
        seed=42,
        num_layers=num_layers,
        batch_size=STANDARD_LORA_CONFIG["batch_size"],
        iters=iters,
        val_batches=25,
        learning_rate=STANDARD_LORA_CONFIG["lr"],
        steps_per_report=10,
        steps_per_eval=200,
        resume_adapter_file=None,
        adapter_path=str(output_dir),
        save_every=iters,  # Only save at end
        max_seq_length=STANDARD_LORA_CONFIG["max_seq_length"],
        config=None,
        grad_checkpoint=False,
        grad_accumulation_steps=1,
        lr_schedule=None,
        lora_parameters={
            "rank": STANDARD_LORA_CONFIG["rank"],
            "dropout": STANDARD_LORA_CONFIG["dropout"],
            "scale": STANDARD_LORA_CONFIG["scale"],
        },
        mask_prompt=False,
        report_to=None,
        project_name=None,
        hf_dataset=False,
    )

    # Load dataset
    train_set, valid_set, _ = load_dataset(args, tokenizer)

    # Freeze model, add LoRA layers
    model.freeze()
    linear_to_lora_layers(model, num_layers, args.lora_parameters)
    print_trainable_parameters(model)

    # Count trainable params (trainable_parameters() returns nested dict)
    from mlx.utils import tree_flatten

    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    save_config(vars(args), output_dir / "adapter_config.json")

    # Build training args
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

    # Build optimizer
    import mlx.optimizers as optim

    opt = optim.Adam(learning_rate=args.learning_rate)

    # Train
    t0 = time.time()
    train(
        model=model,
        args=training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
    )
    training_time = time.time() - t0

    # Measure spectral norms post-hoc
    spectral_info = measure_standard_lora_spectral_norms(model)

    result = {
        "method": "standard_lora",
        "iters": iters,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "adapter_path": str(output_dir),
        "hyperparameters": STANDARD_LORA_CONFIG.copy(),
        "hyperparameter_count": len(STANDARD_LORA_CONFIG),
        "spectral_info": spectral_info,
    }

    # Cleanup
    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return result


def measure_standard_lora_spectral_norms(model) -> dict:
    """Measure spectral norms of standard LoRA adapter weights post-hoc."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    # trainable_parameters() returns nested dict — flatten to (key, array) pairs
    flat_params = tree_flatten(model.trainable_parameters())

    # Collect LoRA pairs and compute spectral norms
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
            # mlx-lm LoRA: lora_a=[in, rank], lora_b=[rank, out]
            # delta_W = lora_a @ lora_b → [in, out]
            a = pair["a"]
            b = pair["b"]
            delta = a @ b
            # Spectral norm via SVD
            s = mx.linalg.svd(delta.astype(mx.float32), stream=mx.cpu)[1]
            mx.eval(s)
            s_max = float(s[0])
            spectral_norms.append({
                "layer": base_key,
                "spectral_norm": s_max,
                "scaled_spectral_norm": s_max * STANDARD_LORA_CONFIG["scale"],
            })

    max_norm = max(n["scaled_spectral_norm"] for n in spectral_norms) if spectral_norms else 0
    return {
        "max_spectral_norm": max_norm,
        "per_layer": spectral_norms,
    }


# ---------------------------------------------------------------------------
# NB-LoRA training (ModelCypher)
# ---------------------------------------------------------------------------
def train_nb_lora(
    model_path: str,
    output_dir: Path,
    quick: bool = False,
) -> dict:
    """Train NB-LoRA using geometry-derived everything. Returns training metadata."""
    import mlx.core as mx
    from modelcypher.backends import initialize_default_backend
    from modelcypher.cli.composition import get_dataset_training_service

    initialize_default_backend()
    service = get_dataset_training_service()

    max_iters = 200 if quick else 10000

    logger.info("  NB-LoRA: training (geometry derives all hyperparameters)...")
    t0 = time.time()
    result = service.train_from_dataset(
        model_path=str(model_path),
        dataset_path=str(TRAIN_DATA),
        output_path=str(output_dir),
        eval_dataset_path=str(VAL_DATA),
        max_iters=max_iters,
        seq_length=None,
        deep=False,
        safety_margin=None,
        seed=42,
    )
    training_time = time.time() - t0

    result_dict = result.to_dict()
    result_dict["method"] = "nb_lora"
    result_dict["training_time_seconds"] = training_time
    result_dict["hyperparameter_count"] = 0
    result_dict["adapter_path"] = str(output_dir)

    mx.clear_cache()
    gc.collect()

    return result_dict


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

    # Greedy decoding: temp=0
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
# Single model experiment
# ---------------------------------------------------------------------------
def run_model_experiment(
    model_key: str,
    model_spec: dict,
    output_dir: Path,
    quick: bool = False,
) -> dict:
    """Run full experiment for one model scale."""
    model_path = str(model_spec["path"])
    model_name = model_spec["name"]
    num_layers = model_spec["num_layers"]

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL: {model_name} ({model_key})")
    logger.info(f"{'='*70}")

    results = {"model": model_name, "model_path": model_path}
    eval_limit = 50 if quick else None

    # ------------------------------------------------------------------
    # a. BASELINE EVAL
    # ------------------------------------------------------------------
    logger.info(f"\n--- [{model_name}] Baseline Evaluation ---")
    baseline_scores = run_lm_eval(
        model_path=model_path,
        tasks=BENCHMARK_TASKS,
        limit=eval_limit,
    )
    results["baseline"] = baseline_scores
    with open(model_dir / "baseline_eval.json", "w") as f:
        json.dump(baseline_scores, f, indent=2)
    logger.info(f"  Baseline scores saved.")

    # ------------------------------------------------------------------
    # b. STANDARD LORA (control)
    # ------------------------------------------------------------------
    logger.info(f"\n--- [{model_name}] Standard LoRA Training ---")
    std_output = model_dir / "standard_lora"

    # Prepare data directory for mlx_lm
    std_data_dir = model_dir / "_std_data"
    prepare_standard_lora_data_dir(TRAIN_DATA, VAL_DATA, std_data_dir)

    std_train_result = train_standard_lora(
        model_path=model_path,
        num_layers=num_layers,
        data_dir=std_data_dir,
        output_dir=std_output,
        quick=quick,
    )
    results["standard_lora_training"] = std_train_result

    with open(std_output / "training_log.json", "w") as f:
        json.dump(std_train_result, f, indent=2)

    # Eval standard LoRA
    logger.info(f"  Evaluating standard LoRA adapter...")
    std_eval_scores = run_lm_eval(
        model_path=model_path,
        tasks=BENCHMARK_TASKS,
        adapter_path=str(std_output),
        limit=eval_limit,
    )
    results["standard_lora_eval"] = std_eval_scores
    with open(std_output / "eval_results.json", "w") as f:
        json.dump(std_eval_scores, f, indent=2)

    # ------------------------------------------------------------------
    # c. NB-LORA (treatment)
    # ------------------------------------------------------------------
    logger.info(f"\n--- [{model_name}] NB-LoRA Training ---")
    nb_output = model_dir / "nb_lora"

    nb_train_result = train_nb_lora(
        model_path=model_path,
        output_dir=nb_output,
        quick=quick,
    )
    results["nb_lora_training"] = nb_train_result

    with open(nb_output / "training_log.json", "w") as f:
        json.dump(nb_train_result, f, indent=2)

    # Eval NB-LoRA
    logger.info(f"  Evaluating NB-LoRA adapter...")
    nb_eval_scores = run_lm_eval(
        model_path=model_path,
        tasks=BENCHMARK_TASKS,
        adapter_path=str(nb_output),
        limit=eval_limit,
    )
    results["nb_lora_eval"] = nb_eval_scores
    with open(nb_output / "eval_results.json", "w") as f:
        json.dump(nb_eval_scores, f, indent=2)

    # ------------------------------------------------------------------
    # d. INFERENCE TESTS
    # ------------------------------------------------------------------
    logger.info(f"\n--- [{model_name}] Inference Tests ---")
    prompts = load_jsonl(INFERENCE_PROMPTS)

    base_responses = generate_inference_responses(
        model_path=model_path,
        adapter_path=None,
        prompts=prompts,
        label="base",
    )
    std_responses = generate_inference_responses(
        model_path=model_path,
        adapter_path=str(std_output),
        prompts=prompts,
        label="standard_lora",
    )
    nb_responses = generate_inference_responses(
        model_path=model_path,
        adapter_path=str(nb_output),
        prompts=prompts,
        label="nb_lora",
    )

    # Merge responses into prompt records
    inference_results = {"prompts": []}
    for prompt_data in prompts:
        pid = prompt_data["id"]
        base_resp = next((r["response"] for r in base_responses if r["id"] == pid), "")
        std_resp = next((r["response"] for r in std_responses if r["id"] == pid), "")
        nb_resp = next((r["response"] for r in nb_responses if r["id"] == pid), "")

        inference_results["prompts"].append({
            "id": pid,
            "category": prompt_data["category"],
            "distribution": prompt_data["distribution"],
            "prompt": prompt_data["prompt"],
            "expected": prompt_data["expected"],
            "responses": {
                "base": base_resp,
                "standard_lora": std_resp,
                "nb_lora": nb_resp,
            },
        })

    results["inference_responses"] = inference_results
    with open(model_dir / "inference_responses.json", "w") as f:
        json.dump(inference_results, f, indent=2)

    # ------------------------------------------------------------------
    # e. COMPARISON
    # ------------------------------------------------------------------
    logger.info(f"\n--- [{model_name}] Comparison ---")
    comparison = build_comparison(
        baseline=baseline_scores,
        standard=std_eval_scores,
        nb_lora=nb_eval_scores,
        std_training=std_train_result,
        nb_training=nb_train_result,
    )
    results["comparison"] = comparison
    with open(model_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print_comparison(model_name, comparison)

    # Cleanup temp data dir
    shutil.rmtree(std_data_dir, ignore_errors=True)

    return results


def build_comparison(
    baseline: dict,
    standard: dict,
    nb_lora: dict,
    std_training: dict,
    nb_training: dict,
) -> dict:
    """Build comparison metrics between methods."""
    comparison = {"benchmarks": {}, "training": {}, "spectral": {}}

    # Benchmark deltas
    for task in BENCHMARK_TASKS:
        base_scores = baseline.get(task, {})
        std_scores = standard.get(task, {})
        nb_scores = nb_lora.get(task, {})

        # Find the primary metric (acc_norm preferred, then acc, then exact_match)
        metric = None
        for m in ["acc_norm,none", "acc,none", "exact_match,strict-match"]:
            if m in base_scores:
                metric = m
                break
        if metric is None:
            # Try any metric that looks like accuracy
            for k in base_scores:
                if "acc" in k or "exact" in k:
                    metric = k
                    break

        if metric:
            base_val = base_scores.get(metric, 0)
            std_val = std_scores.get(metric, 0)
            nb_val = nb_scores.get(metric, 0)

            comparison["benchmarks"][task] = {
                "metric": metric,
                "baseline": base_val,
                "standard_lora": std_val,
                "nb_lora": nb_val,
                "delta_std_vs_base": std_val - base_val,
                "delta_nb_vs_base": nb_val - base_val,
                "delta_nb_vs_std": nb_val - std_val,
            }

    # Training comparison
    comparison["training"] = {
        "standard_lora": {
            "iters": std_training.get("iters", 0),
            "training_time_seconds": std_training.get("training_time_seconds", 0),
            "n_trainable_params": std_training.get("n_trainable_params", 0),
            "hyperparameter_count": std_training.get("hyperparameter_count", 0),
        },
        "nb_lora": {
            "iters": nb_training.get("train_iters", 0),
            "training_time_seconds": nb_training.get("training_time_seconds", 0),
            "n_trainable_params": nb_training.get("n_trainable_params", 0),
            "hyperparameter_count": nb_training.get("hyperparameter_count", 0),
            "stop_reason": nb_training.get("stop_reason", ""),
            "initial_loss": nb_training.get("initial_loss", 0),
            "final_loss": nb_training.get("final_loss", 0),
            "baseline_perplexity": nb_training.get("baseline_perplexity", 0),
            "post_perplexity": nb_training.get("post_perplexity", 0),
        },
    }

    # Spectral safety
    std_spectral = std_training.get("spectral_info", {})
    comparison["spectral"] = {
        "standard_lora": {
            "max_spectral_norm": std_spectral.get("max_spectral_norm", 0),
            "bounded_by_construction": False,
        },
        "nb_lora": {
            "max_spectral_ratio": nb_training.get("max_spectral_ratio", 0),
            "spectral_bounds_ok": nb_training.get("spectral_bounds_ok", False),
            "bounded_by_construction": True,
        },
    }

    return comparison


def print_comparison(model_name: str, comparison: dict):
    """Print a human-readable comparison table."""
    logger.info(f"\n  {'Task':<20} {'Baseline':>10} {'Std LoRA':>10} {'NB-LoRA':>10} {'Δ(NB-Std)':>10}")
    logger.info(f"  {'-'*60}")

    for task, data in comparison.get("benchmarks", {}).items():
        base = data["baseline"]
        std = data["standard_lora"]
        nb = data["nb_lora"]
        delta = data["delta_nb_vs_std"]
        sign = "+" if delta > 0 else ""
        logger.info(
            f"  {task:<20} {base:>10.4f} {std:>10.4f} {nb:>10.4f} {sign}{delta:>9.4f}"
        )

    tc = comparison.get("training", {})
    std_t = tc.get("standard_lora", {})
    nb_t = tc.get("nb_lora", {})
    logger.info(f"\n  Training:")
    logger.info(f"    Standard LoRA: {std_t.get('iters', '?')} iters, "
                f"{std_t.get('training_time_seconds', 0):.1f}s, "
                f"{std_t.get('hyperparameter_count', '?')} hyperparameters")
    logger.info(f"    NB-LoRA:       {nb_t.get('iters', '?')} iters, "
                f"{nb_t.get('training_time_seconds', 0):.1f}s, "
                f"{nb_t.get('hyperparameter_count', '?')} hyperparameters")

    sc = comparison.get("spectral", {})
    logger.info(f"\n  Spectral Safety:")
    logger.info(f"    Standard LoRA max norm: {sc.get('standard_lora', {}).get('max_spectral_norm', '?')}")
    logger.info(f"    NB-LoRA bounded by construction: {sc.get('nb_lora', {}).get('spectral_bounds_ok', '?')}")
    logger.info(f"    NB-LoRA max ratio: {sc.get('nb_lora', {}).get('max_spectral_ratio', '?')}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def build_summary(all_results: dict) -> dict:
    """Build cross-model summary table."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "overall": {},
    }

    all_deltas = []
    for model_key, model_results in all_results.items():
        comparison = model_results.get("comparison", {})
        benchmarks = comparison.get("benchmarks", {})

        model_summary = {}
        for task, data in benchmarks.items():
            model_summary[task] = {
                "baseline": data["baseline"],
                "standard_lora": data["standard_lora"],
                "nb_lora": data["nb_lora"],
                "nb_wins": data["delta_nb_vs_std"] > 0,
                "delta": data["delta_nb_vs_std"],
            }
            all_deltas.append(data["delta_nb_vs_std"])

        summary["models"][model_key] = model_summary

    if all_deltas:
        nb_wins = sum(1 for d in all_deltas if d > 0)
        std_wins = sum(1 for d in all_deltas if d < 0)
        ties = sum(1 for d in all_deltas if d == 0)
        summary["overall"] = {
            "nb_lora_wins": nb_wins,
            "standard_lora_wins": std_wins,
            "ties": ties,
            "total_comparisons": len(all_deltas),
            "mean_delta": sum(all_deltas) / len(all_deltas),
            "verdict": (
                "NB-LoRA better" if nb_wins > std_wins
                else "Standard LoRA better" if std_wins > nb_wins
                else "Tie"
            ),
        }

    return summary


def print_summary(summary: dict):
    """Print final summary to stdout."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY: NB-LoRA vs Standard LoRA")
    print("=" * 70)

    for model_name, tasks in summary.get("models", {}).items():
        print(f"\n  {model_name}:")
        print(f"    {'Task':<20} {'Baseline':>10} {'Std LoRA':>10} {'NB-LoRA':>10} {'Winner':>10}")
        print(f"    {'-'*60}")
        for task, data in tasks.items():
            winner = "NB-LoRA" if data["nb_wins"] else "Std LoRA" if data["delta"] < 0 else "Tie"
            print(
                f"    {task:<20} {data['baseline']:>10.4f} "
                f"{data['standard_lora']:>10.4f} {data['nb_lora']:>10.4f} "
                f"{winner:>10}"
            )

    overall = summary.get("overall", {})
    if overall:
        print(f"\n  Overall:")
        print(f"    NB-LoRA wins:        {overall['nb_lora_wins']}")
        print(f"    Standard LoRA wins:  {overall['standard_lora_wins']}")
        print(f"    Ties:                {overall['ties']}")
        print(f"    Mean Δ(NB - Std):    {overall['mean_delta']:+.4f}")
        print(f"    Verdict:             {overall['verdict']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NB-LoRA vs Standard LoRA controlled experiment",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS.keys()),
        default=list(MODEL_SPECS.keys()),
        help="Which model scales to test (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iters (100 std, 200 nb), limited eval (50 samples per task)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: /Volumes/CodeCypher/experiments/nblora-vs-standard-{timestamp})",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip lm-eval benchmarks (for faster debugging of training pipeline)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference response generation",
    )
    args = parser.parse_args()

    # Validate volume
    if not VOLUME.exists():
        logger.error(f"Volume not mounted: {VOLUME}")
        logger.error("Mount the external drive first.")
        sys.exit(1)

    # Validate models exist
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

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        EXPERIMENTS_DIR / f"nblora-vs-standard-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("NB-LoRA vs Standard LoRA — Controlled Experiment")
    logger.info("=" * 70)
    logger.info(f"Models: {args.models}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Train data: {TRAIN_DATA} ({sum(1 for _ in open(TRAIN_DATA))} samples)")
    logger.info(f"Val data: {VAL_DATA} ({sum(1 for _ in open(VAL_DATA))} samples)")

    # Override eval/inference functions if skipped
    global run_lm_eval, generate_inference_responses
    original_run_lm_eval = run_lm_eval
    original_generate_inference = generate_inference_responses

    if args.skip_eval:
        logger.info("SKIPPING lm-eval benchmarks (--skip-eval)")
        run_lm_eval = lambda **kwargs: {}

    if args.skip_inference:
        logger.info("SKIPPING inference tests (--skip-inference)")
        generate_inference_responses = lambda **kwargs: []

    # Run experiment for each model
    all_results = {}
    total_t0 = time.time()

    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        try:
            result = run_model_experiment(
                model_key=model_key,
                model_spec=spec,
                output_dir=output_dir,
                quick=args.quick,
            )
            all_results[spec["name"]] = result
        except Exception:
            logger.exception(f"FAILED on model {model_key}")
            all_results[spec["name"]] = {"error": True}

    total_time = time.time() - total_t0

    # Restore originals
    run_lm_eval = original_run_lm_eval
    generate_inference_responses = original_generate_inference

    # Build and save summary
    summary = build_summary(all_results)
    summary["total_time_seconds"] = total_time

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results
    with open(output_dir / "full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_summary(summary)
    logger.info(f"\nTotal experiment time: {total_time:.1f}s")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
