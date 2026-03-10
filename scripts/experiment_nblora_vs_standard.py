#!/usr/bin/env python3
"""NB-LoRA vs PEFT Baselines — Controlled Head-to-Head Comparison (R1).

N-arm comparison. Available methods:
  - nb_lora:       NB-LoRA (zero HP, geometry-derived)
  - standard_lora: Standard LoRA (community defaults: AdamW, cosine LR, rank=8, alpha=16)
  - tuned_lora:    Tuned Standard LoRA (best of rank x lr grid search)
  - dora:          DoRA (weight decomposition — magnitude + direction)
  - rslora:        rsLoRA (rank-stabilized: scale = alpha / sqrt(rank))
  - pissa:         PiSSA (SVD-initialized: A,B from principal components of W)
  - eva:           EVA (Explained Variance Adaptation: data-driven rank + init)

Each method x N seeds (default 5) for statistical validity.

Measurements per method per seed:
  - lm-eval-harness benchmark scores (7 tasks)
  - Training loss convergence (captured via callback)
  - Spectral safety (post-hoc for standard methods, by construction for NB-LoRA)
  - Commensurable val_loss (full val set, batch_size=1, same CE operator)
  - Inference test responses

Usage:
  # Quick smoke test (350M, 1 seed, default 3 arms)
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M --quick

  # Full R1 comparison (all methods, 5 seeds)
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M \\
      --methods standard_lora tuned_lora dora rslora pissa eva nb_lora

  # Subset of methods
  poetry run python scripts/experiment_nblora_vs_standard.py --models 350M \\
      --methods standard_lora dora nb_lora --seeds 42 123 456

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

# rsLoRA: scale = alpha / sqrt(rank) instead of alpha / rank
RSLORA_CONFIGS = {
    r: {**STANDARD_LORA_CONFIG, "rank": r, "scale": LORA_ALPHA / math.sqrt(r)}
    for r in [4, 8, 16]
}
RSLORA_CONFIG = RSLORA_CONFIGS[8]  # default rank=8

# DoRA: same hyperparameters as standard LoRA, different layer type
DORA_CONFIG = {**STANDARD_LORA_CONFIG}

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# All available methods (populated after training functions are defined)
ALL_METHOD_NAMES = [
    "standard_lora", "tuned_lora", "dora", "rslora", "pissa", "eva", "nb_lora",
    "geometric_pissa",
    "standard_nb_surface", "pissa_nb_surface", "dora_nb_surface",
    "geometric_pissa_nb_surface",
]

# Methods that require an NBTargetSurface (derived once per model)
NB_SURFACE_METHODS = {
    "standard_nb_surface", "pissa_nb_surface", "dora_nb_surface",
    "geometric_pissa_nb_surface",
}


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
# LoRA-family layer conversion
# ---------------------------------------------------------------------------
def _convert_lora_family_layers(
    model,
    num_layers: int,
    lora_parameters: dict,
    fine_tune_type: str,
    converter=None,
) -> None:
    """Convert layers to the requested PEFT family.

    mlx-lm uses the same converter entry point for LoRA and DoRA, with DoRA
    enabled via ``use_dora=True``. Keeping that switch in one helper makes the
    experiment harness easier to test and prevents nominal DoRA arms from
    silently training plain LoRA.
    """
    if converter is None:
        from mlx_lm.lora import linear_to_lora_layers as converter

    converter(
        model,
        num_layers,
        lora_parameters,
        use_dora=(fine_tune_type == "dora"),
    )


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
    fine_tune_type: str = "lora",
) -> dict:
    """Train LoRA-family adapter with given config and seed.

    Supports standard LoRA (fine_tune_type="lora") and DoRA
    (fine_tune_type="dora") via mlx-lm's built-in layer types.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load
    from mlx_lm.lora import (
        TrainingArgs,
        load_dataset,
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
        fine_tune_type=fine_tune_type,
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
    _convert_lora_family_layers(
        model,
        num_layers,
        args.lora_parameters,
        fine_tune_type=fine_tune_type,
    )

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
        "method": fine_tune_type,
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
# PiSSA training (SVD-initialized LoRA)
# ---------------------------------------------------------------------------
def train_pissa(
    model_path: str,
    num_layers: int,
    data_dir: Path,
    output_dir: Path,
    config: dict | None = None,
    seed: int = 42,
    iters_override: int | None = None,
) -> dict:
    """Train PiSSA: LoRA initialized with principal singular vectors of W.

    1. SVD(W) → U, S, V^T for each target layer
    2. A_init = V[:, :r] (top-r right singular vectors)
    3. B_init = (S[:r, None] * U[:, :r]^T) (scaled left singular vectors)
    4. W_residual stored as base weight (W - U_r @ S_r @ V_r^T)
    5. Train with standard LoRA machinery on the residual
    """
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
    rank = cfg["rank"]

    mx.random.seed(seed)
    random.seed(seed)

    logger.info(f"  PiSSA: rank={rank}, lr={cfg['lr']}, seed={seed}")

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
            "rank": rank,
            "dropout": cfg.get("dropout", 0.0),
            "scale": 1.0,  # PiSSA: SVD provides the right magnitudes
        },
        mask_prompt=False,
        report_to=None,
        project_name=None,
        hf_dataset=False,
    )

    train_set, valid_set, _ = load_dataset(args, tokenizer)

    model.freeze()
    linear_to_lora_layers(model, num_layers, args.lora_parameters)

    # PiSSA initialization: replace random A,B with principal SVD components
    pissa_init_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            # Get original weight (before LoRA was applied)
            W = module.linear.weight if hasattr(module, "linear") else None
            if W is None:
                continue
            W_f32 = W.astype(mx.float32)
            try:
                U, S, Vt = mx.linalg.svd(W_f32, stream=mx.cpu)
                mx.eval(U, S, Vt)
            except Exception:
                logger.warning(f"  PiSSA SVD failed for {name}, keeping random init")
                continue

            r = min(rank, S.shape[0])
            # A_init: [input_dims, r] = V[:, :r] = Vt[:r, :]^T
            A_init = Vt[:r, :].T.astype(W.dtype)
            # B_init: [r, output_dims] = diag(S[:r]) @ U[:, :r]^T
            B_init = (S[:r, None] * U[:, :r].T).astype(W.dtype)

            # Update residual base weight: W - U_r @ S_r @ V_r^T
            W_principal = (U[:, :r] * S[:r]) @ Vt[:r, :]
            mx.eval(W_principal)
            W_residual = (W_f32 - W_principal).astype(W.dtype)
            module.linear.weight = W_residual

            # Set LoRA weights
            module.lora_a = A_init
            module.lora_b = B_init
            mx.eval(module.lora_a, module.lora_b, module.linear.weight)
            pissa_init_count += 1

    logger.info(f"  PiSSA: initialized {pissa_init_count} layers with SVD principal components")

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

    lr_value = cfg["lr"]
    if cfg.get("lr_schedule") == "cosine":
        lr_value = optim.cosine_decay(init=cfg["lr"], decay_steps=iters)
    if cfg["optimizer"] == "adamw":
        opt = optim.AdamW(learning_rate=lr_value, weight_decay=0.0)
    else:
        opt = optim.Adam(learning_rate=lr_value)

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

    adapter_file_check = output_dir / "adapters.safetensors"
    if not adapter_file_check.exists():
        from mlx.utils import tree_flatten as tf
        adapter_weights = dict(tf(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file_check), adapter_weights)

    spectral_info = measure_standard_lora_spectral_norms(model, 1.0)  # scale=1.0 for PiSSA

    # Fuse LoRA into residual base weights for correct post-hoc evaluation.
    # PiSSA modifies base weights (W_residual = W - U_r S_r V_r^T), so the
    # standard "load original model + adapter" path double-counts the principal
    # components and produces incorrect results.
    fused_dir = output_dir / "fused_model"
    fused_dir.mkdir(parents=True, exist_ok=True)

    for name, module in model.named_modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            # LoRALinear forward: y = linear(x) + scale * (x @ lora_a) @ lora_b
            # lora_a: [in, r], lora_b: [r, out], linear.weight: [out, in]
            # Delta weight = (lora_a @ lora_b).T  shape [out, in]
            scale = getattr(module, "scale", 1.0)
            delta = (module.lora_a @ module.lora_b) * scale
            module.linear.weight = module.linear.weight + delta.T

    # Save fused model weights, excluding LoRA-specific parameters.
    # After fusion the delta is folded into linear.weight, so LoRA keys
    # (lora_a, lora_b, scale, etc.) are no longer needed.  LoRALinear
    # also nests weight as module.linear.weight — remap to module.weight
    # so the base model can load them.
    from mlx.utils import tree_flatten as _tf
    weights = {}
    for k, v in _tf(model.parameters()):
        if any(t in k for t in ("lora_a", "lora_b", ".scale")):
            continue
        # LoRALinear: proj.linear.weight → proj.weight
        k = k.replace(".linear.weight", ".weight")
        k = k.replace(".linear.bias", ".bias")
        weights[k] = v
    mx.save_safetensors(str(fused_dir / "model.safetensors"), weights)
    # Copy config/tokenizer files from original model
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json"]:
        src = Path(model_path) / fname
        if src.exists():
            shutil.copy2(src, fused_dir / fname)
    logger.info(f"  PiSSA: saved fused model to {fused_dir}")

    result = {
        "method": "pissa",
        "seed": seed,
        "iters": iters,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "adapter_path": str(output_dir),
        "fused_model_path": str(fused_dir),
        "config": {k: v for k, v in cfg.items()},
        "hyperparameter_count": len([
            k for k in cfg if k not in ("max_seq_length", "batch_size", "iters")
        ]),
        "pissa_init_layers": pissa_init_count,
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


# ---------------------------------------------------------------------------
# Targeted LoRA conversion (exact NB-LoRA surface)
# ---------------------------------------------------------------------------
def targeted_lora_conversion(
    model,
    rank_overrides: dict[str, int],
    scale: float,
    dropout: float = 0.0,
    use_dora: bool = False,
) -> int:
    """Inject LoRA/DoRA on exact module keys with per-module ranks.

    Navigates the model tree using weight key paths from NB-LoRA geometry
    analysis.  Each module gets its own rank from rank_overrides.
    Returns count of converted modules.
    """
    import mlx.nn as nn
    from mlx_lm.tuner.lora import LoRALinear

    if use_dora:
        from mlx_lm.tuner.dora import DoRALinear
        LayerClass = DoRALinear
    else:
        LayerClass = LoRALinear

    converted = 0
    for weight_key, rank in rank_overrides.items():
        # weight_key is like "base.layers.2.mixer.q_proj.weight"
        # Strip ".weight" suffix to get module path
        module_path = weight_key.replace(".weight", "")
        parts = module_path.split(".")

        # Navigate model tree to find the parent and attribute name
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        attr_name = parts[-1]
        linear = getattr(parent, attr_name)

        if not isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
            logger.warning(
                "targeted_lora_conversion: %s is %s, not Linear — skipping",
                weight_key, type(linear).__name__,
            )
            continue

        lora_layer = LayerClass.from_base(
            linear, r=rank, scale=scale, dropout=dropout,
        )
        setattr(parent, attr_name, lora_layer)
        converted += 1

    logger.info(
        "targeted_lora_conversion: %d/%d modules converted (dora=%s)",
        converted, len(rank_overrides), use_dora,
    )
    return converted


# ---------------------------------------------------------------------------
# Surface-matched training (same surface as NB-LoRA, different method)
# ---------------------------------------------------------------------------
def train_surface_matched(
    model_path: str,
    data_dir: Path,
    output_dir: Path,
    rank_overrides: dict[str, int],
    config: dict | None = None,
    seed: int = 42,
    iters_override: int | None = None,
    fine_tune_type: str = "lora",
    pissa_init: bool = False,
) -> dict:
    """Train LoRA-family adapter on NB-LoRA's exact module surface.

    Uses targeted_lora_conversion with per-module ranks from
    derive_nb_target_surface().  Supports standard LoRA, DoRA, and PiSSA
    init on the matched surface.
    """
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load
    from mlx_lm.lora import (
        TrainingArgs,
        load_dataset,
        save_config,
        train,
    )
    from mlx_lm.tuner.datasets import CacheDataset

    cfg = config or STANDARD_LORA_CONFIG
    iters = iters_override or cfg["iters"]
    use_dora = fine_tune_type == "dora"
    avg_rank = sum(rank_overrides.values()) / max(len(rank_overrides), 1)
    scale = cfg.get("scale", LORA_ALPHA / avg_rank)

    mx.random.seed(seed)
    random.seed(seed)

    logger.info(
        f"  Surface-matched {fine_tune_type}: {len(rank_overrides)} modules, "
        f"ranks={sorted(set(rank_overrides.values()))}, "
        f"pissa={pissa_init}, seed={seed}"
    )

    model, tokenizer = mlx_load(str(model_path))

    args = SimpleNamespace(
        model=str(model_path), data=str(data_dir), train=True, test=False,
        seed=seed, batch_size=cfg["batch_size"],
        max_seq_length=cfg["max_seq_length"], hf_dataset=False,
    )
    train_set, valid_set, _ = load_dataset(args, tokenizer)

    model.freeze()
    n_converted = targeted_lora_conversion(
        model, rank_overrides, scale=scale,
        dropout=cfg.get("dropout", 0.0), use_dora=use_dora,
    )

    fused_model_path = None

    if pissa_init:
        # SVD-initialize each LoRA layer from base weight principal components
        pissa_count = 0
        for weight_key, rank in rank_overrides.items():
            module_path = weight_key.replace(".weight", "")
            parts = module_path.split(".")
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            module = getattr(parent, parts[-1])

            if not hasattr(module, "lora_a"):
                continue

            W = module.linear.weight if hasattr(module, "linear") else None
            if W is None:
                continue

            W_f32 = W.astype(mx.float32)
            try:
                U, S, Vt = mx.linalg.svd(W_f32, stream=mx.cpu)
                mx.eval(U, S, Vt)
            except Exception:
                logger.warning(f"  PiSSA SVD failed for {weight_key}, keeping random init")
                continue

            r = min(rank, S.shape[0])
            A_init = Vt[:r, :].T.astype(W.dtype)
            B_init = (S[:r, None] * U[:, :r].T).astype(W.dtype)

            W_principal = (U[:, :r] * S[:r]) @ Vt[:r, :]
            mx.eval(W_principal)
            W_residual = (W_f32 - W_principal).astype(W.dtype)
            module.linear.weight = W_residual
            module.lora_a = A_init
            module.lora_b = B_init
            mx.eval(module.lora_a, module.lora_b, module.linear.weight)
            pissa_count += 1

        logger.info(f"  PiSSA SVD init: {pissa_count}/{n_converted} modules")

        # Save fused model (base+residual) for eval
        fused_dir = output_dir / "fused_model"
        fused_dir.mkdir(parents=True, exist_ok=True)
        # Copy tokenizer + config
        import shutil as _shutil
        src_model = Path(model_path)
        for f in src_model.iterdir():
            if f.suffix in (".json", ".txt", ".model") or f.name.startswith("tokenizer"):
                _shutil.copy2(f, fused_dir / f.name)
        # Save modified base weights
        from mlx.utils import tree_flatten as tf
        base_weights = dict(tf(model.parameters()))
        # Remove LoRA params from base weights
        base_only = {k: v for k, v in base_weights.items()
                     if "lora_a" not in k and "lora_b" not in k
                     and "lora_m" not in k}
        mx.save_safetensors(str(fused_dir / "model.safetensors"), base_only)
        fused_model_path = str(fused_dir)

    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    save_config({"rank_overrides": rank_overrides, "scale": scale}, output_dir / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=cfg["batch_size"],
        iters=iters,
        val_batches=25,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=iters,
        adapter_file=adapter_file,
        max_seq_length=cfg["max_seq_length"],
        grad_checkpoint=False,
        grad_accumulation_steps=1,
    )

    lr_value = cfg["lr"]
    if cfg.get("lr_schedule") == "cosine":
        lr_value = optim.cosine_decay(init=cfg["lr"], decay_steps=iters)

    if cfg.get("optimizer", "adamw") == "adamw":
        opt = optim.AdamW(learning_rate=lr_value, weight_decay=0.0)
    else:
        opt = optim.Adam(learning_rate=lr_value)

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

    # Ensure adapter saved
    if not adapter_file.exists():
        from mlx.utils import tree_flatten as tf2
        adapter_weights = dict(tf2(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file), adapter_weights)

    spectral_info = measure_standard_lora_spectral_norms(model, scale)

    result = {
        "method": fine_tune_type,
        "seed": seed,
        "iters": iters,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "adapter_path": str(output_dir),
        "config": {
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "optimizer": cfg.get("optimizer", "adamw"),
            "scale": scale,
            "n_modules": len(rank_overrides),
            "ranks": sorted(set(rank_overrides.values())),
            "pissa_init": pissa_init,
            "fine_tune_type": fine_tune_type,
        },
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

    if fused_model_path:
        result["fused_model_path"] = fused_model_path

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Geometric PiSSA: PiSSA init + MASS-derived LR + Fisher preconditioning
# ---------------------------------------------------------------------------
def train_geometric_pissa(
    model_path: str,
    num_layers: int,
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    rank_overrides: dict[str, int] | None = None,
) -> dict:
    """PiSSA SVD initialization + MASS automation.  Zero heuristics.

    1. SVD of W → A, B from principal singular vectors (PiSSA init)
    2. W_residual = W - U_r S_r V_r^T (PiSSA decomposition)
    3. MASS-derived learning rate per step (replaces heuristic lr=2e-4)
    4. Fisher preconditioning (curvature-aware direction)
    5. Geometric stopping certificate (replaces heuristic iters=1000)
    6. Spectral budget = (σ_r - σ_{r+1})/2 tracks change from PiSSA init

    When rank_overrides is provided, injects LoRA on those exact modules
    with per-module ranks (NB surface mode) instead of all linears.
    """
    import math

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten as mlx_flatten
    from mlx_lm import load as mlx_load
    from mlx_lm.lora import linear_to_lora_layers, load_dataset
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import default_loss, iterate_batches

    from modelcypher.core.domain.training.diagonal_fisher_preconditioner import (
        init_fisher_state,
        precondition_gradient,
        update_fisher_state,
    )
    from modelcypher.core.domain.training.geometric_early_stopping import (
        check_stopping_certificate,
        check_val_loss_converged,
        should_certificate_stop,
    )
    from modelcypher.core.domain.training.mass_step_size import (
        apply_sqrt_n_epoch_correction,
        compute_per_step_rates,
        derive_spectral_ceiling,
    )

    rank = 8
    batch_size = 4
    max_seq_length = 2048
    max_epochs = 100  # safety cap, NOT a tuning knob

    mx.random.seed(seed)
    random.seed(seed)

    if rank_overrides:
        logger.info(
            f"  Geometric PiSSA (NB surface): {len(rank_overrides)} modules, "
            f"ranks={sorted(set(rank_overrides.values()))}, seed={seed}"
        )
    else:
        logger.info(f"  Geometric PiSSA: rank={rank}, seed={seed} (zero heuristics)")

    model, tokenizer = mlx_load(str(model_path))

    # Load data via mlx-lm's standard pipeline
    args = SimpleNamespace(
        model=str(model_path), data=str(data_dir), train=True, test=False,
        seed=seed, batch_size=batch_size, max_seq_length=max_seq_length,
        hf_dataset=False,
    )
    train_set, valid_set, _ = load_dataset(args, tokenizer)

    # Apply LoRA layers (random init, will be overwritten by SVD)
    model.freeze()
    if rank_overrides:
        targeted_lora_conversion(
            model, rank_overrides, scale=1.0, dropout=0.0,
        )
    else:
        linear_to_lora_layers(
            model, num_layers,
            {"rank": rank, "dropout": 0.0, "scale": 1.0},
        )

    # ── PiSSA SVD Initialization + geometry extraction ──
    pissa_init_count = 0
    layer_geometry = {}  # name → {sigma_r, sigma_r_plus_1, sigma_max, gap, budget}
    ab_inits = {}        # name → AB_init snapshot for budget tracking

    # Build lookup from module path → rank when using per-module overrides
    _module_rank_lookup = {}
    if rank_overrides:
        for wk, rk in rank_overrides.items():
            _module_rank_lookup[wk.replace(".weight", "")] = rk

    for name, module in model.named_modules():
        if not (hasattr(module, "lora_a") and hasattr(module, "lora_b")):
            continue
        W = module.linear.weight if hasattr(module, "linear") else None
        if W is None:
            continue

        W_f32 = W.astype(mx.float32)
        try:
            U, S, Vt = mx.linalg.svd(W_f32, stream=mx.cpu)
            mx.eval(U, S, Vt)
        except Exception:
            logger.warning(f"  Geometric PiSSA SVD failed for {name}, keeping random init")
            continue

        module_rank = _module_rank_lookup.get(name, rank) if rank_overrides else rank
        r = min(module_rank, S.shape[0])
        # PiSSA initialization
        A_init = Vt[:r, :].T.astype(W.dtype)       # [in, r]
        B_init = (S[:r, None] * U[:, :r].T).astype(W.dtype)  # [r, out]

        # Residual base weight
        W_principal = (U[:, :r] * S[:r]) @ Vt[:r, :]
        mx.eval(W_principal)
        W_residual = (W_f32 - W_principal).astype(W.dtype)
        module.linear.weight = W_residual

        # Set LoRA weights
        module.lora_a = A_init
        module.lora_b = B_init
        mx.eval(module.lora_a, module.lora_b, module.linear.weight)

        # Geometry for MASS: spectral gap at rank r
        sigma_r = float(S[r - 1])
        sigma_r_plus_1 = float(S[r]) if r < S.shape[0] else 0.0
        sigma_max = float(S[0])
        gap = sigma_r - sigma_r_plus_1
        budget = gap / 2.0  # Weyl crossing threshold

        layer_geometry[name] = {
            "sigma_r": sigma_r, "sigma_r_plus_1": sigma_r_plus_1,
            "sigma_max": sigma_max, "gap": gap, "budget": budget,
        }

        # Snapshot AB_init for budget tracking
        ab_init = (module.lora_a @ module.lora_b)
        mx.eval(ab_init)
        ab_inits[name] = ab_init

        pissa_init_count += 1

    logger.info(f"  Geometric PiSSA: {pissa_init_count} layers SVD-initialized")

    if not layer_geometry:
        raise RuntimeError("No layers successfully initialized with SVD")

    # Global spectral bounds for MASS
    gap_min = min(g["gap"] for g in layer_geometry.values())
    sigma_max_global = max(g["sigma_max"] for g in layer_geometry.values())
    budget_min = min(g["budget"] for g in layer_geometry.values())

    logger.info(
        f"  Spectral geometry: gap_min={gap_min:.4e}, σ_max={sigma_max_global:.4e}, "
        f"budget_min={budget_min:.4e}"
    )

    n_trainable = sum(v.size for _, v in mlx_flatten(model.trainable_parameters()))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data batching ──
    train_ds = CacheDataset(train_set)
    valid_ds = CacheDataset(valid_set)
    batch_iter = iterate_batches(train_ds, batch_size, max_seq_length, loop=True, seed=seed)
    n_batches_per_epoch = len(
        list(iterate_batches(train_ds, batch_size, max_seq_length, loop=False, seed=seed))
    )
    if n_batches_per_epoch <= 0:
        raise ValueError("Training dataset produced zero batches")
    max_iters = n_batches_per_epoch * max_epochs

    logger.info(
        f"  Training: {n_batches_per_epoch} batches/epoch, max_epochs={max_epochs}, "
        f"max_iters={max_iters}"
    )

    # ── MASS Layer 1: Spectral ceiling ──
    eta_ceiling = derive_spectral_ceiling(
        sigma_k_min=gap_min, sigma_max_global=sigma_max_global,
    )
    eta_ceiling = apply_sqrt_n_epoch_correction(eta_ceiling, n_batches_per_epoch)
    logger.info(f"  MASS ceiling: η_ceiling={eta_ceiling:.6e}")

    # ── Fisher state ──
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    initialize_default_backend()
    backend = get_default_backend()

    trainable_flat = dict(mlx_flatten(model.trainable_parameters()))
    fisher_state = init_fisher_state(
        trainable_flat, backend, n_batches_per_epoch=n_batches_per_epoch,
    )
    del trainable_flat
    logger.info(f"  Fisher: β₁={fisher_state.beta1:.4f}, β₂={fisher_state.beta2:.4f}")

    # ── Loss function ──
    loss_fn = default_loss
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    # ── Training state ──
    remaining_budgets = {name: g["budget"] for name, g in layer_geometry.items()}
    val_losses_history = []
    grad_norm_history = []
    train_losses = []
    val_losses_log = []
    stop_reason = "max_iters"

    t0 = time.time()

    for it in range(max_iters):
        # Forward + backward
        batch, lengths = next(batch_iter)
        (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)

        # Flatten gradient + Fisher preconditioning
        grad_flat = dict(mlx_flatten(grad))
        fisher_state = update_fisher_state(fisher_state, grad_flat, backend)
        update_direction = precondition_gradient(grad_flat, fisher_state, backend)

        # Direction norm
        d_flat = [p.reshape(-1) for p in update_direction.values() if p.size > 0]
        d_norm_sq = sum(mx.sum(p * p) for p in d_flat)
        mx.eval(d_norm_sq, loss)
        d_norm_val = float(mx.sqrt(d_norm_sq).item())
        loss_float = float(loss)

        # MASS per-step rates
        min_remaining = min(remaining_budgets.values()) if remaining_budgets else None
        eta_step, eta_sps, eta_weyl, displacement, eta_margin = compute_per_step_rates(
            loss_float, d_norm_val, gap_min, eta_ceiling,
            remaining_budget=min_remaining,
        )

        # Parameter update: θ -= η_step × d
        eta_arr = mx.array(eta_step)
        current_params = dict(mlx_flatten(model.trainable_parameters()))
        updated_params = [
            (k, current_params[k] - eta_arr * update_direction[k])
            for k in current_params
            if k in update_direction
        ]
        model.load_weights(updated_params, strict=False)
        mx.eval(*[v for _, v in mlx_flatten(model.trainable_parameters())])

        # Per-step logging
        if (it + 1) % 10 == 0:
            logger.info(
                f"  Step {it+1}/{max_iters} | loss={loss_float:.4f} | "
                f"η={eta_step:.2e} (sps={eta_sps:.2e} weyl={eta_weyl:.2e} "
                f"ceil={eta_ceiling:.2e}"
                f"{f' margin={eta_margin:.2e}' if eta_margin is not None else ''}) | "
                f"‖d‖={d_norm_val:.4f}"
            )

        # ── Epoch boundary ──
        if (it + 1) % n_batches_per_epoch == 0:
            epoch = (it + 1) // n_batches_per_epoch

            # Measure spectral budget consumption (change from PiSSA init)
            max_ratio = 0.0
            for name, module in model.named_modules():
                if name not in ab_inits:
                    continue
                ab_current = module.lora_a @ module.lora_b
                delta = ab_current - ab_inits[name]
                # Power iteration for spectral norm of dense delta
                v = mx.random.normal((delta.shape[1],))
                mx.eval(v)
                for _ in range(10):
                    u = delta @ v
                    u_norm = mx.linalg.norm(u)
                    u = u / mx.maximum(u_norm, mx.array(1e-12))
                    v = delta.T @ u
                    v_norm = mx.linalg.norm(v)
                    v = v / mx.maximum(v_norm, mx.array(1e-12))
                mx.eval(u, v)
                s_est = float(mx.abs(u @ delta @ v).item())
                budget = layer_geometry[name]["budget"]
                remaining_budgets[name] = max(0.0, budget - s_est)
                if budget > 0:
                    max_ratio = max(max_ratio, s_est / budget)

            # Validation loss
            val_loss = 0.0
            val_ntoks = 0
            for vb, vl in iterate_batches(valid_ds, batch_size, max_seq_length, loop=False):
                vl_val, vn = loss_fn(model, vb, vl)
                mx.eval(vl_val)
                val_loss += float(vl_val) * int(vn)
                val_ntoks += int(vn)
            val_loss = val_loss / max(val_ntoks, 1)
            val_losses_history.append(val_loss)
            val_losses_log.append({"epoch": epoch, "val_loss": val_loss})

            # Gradient norm for stationarity
            raw_flat = [p.reshape(-1) for p in grad_flat.values() if p.size > 0]
            if raw_flat:
                gnorm_sq = sum(mx.sum(p * p) for p in raw_flat)
                mx.eval(gnorm_sq)
                grad_norm = float(mx.sqrt(gnorm_sq).item())
            else:
                grad_norm = 0.0
            grad_norm_history.append(grad_norm)

            logger.info(
                f"  Epoch {epoch} | val_loss={val_loss:.4f} | ‖g‖={grad_norm:.4e} | "
                f"budget_ratio={max_ratio:.4f} | remaining_min={min(remaining_budgets.values()):.4e}"
            )

            # Check stopping: val loss convergence
            converged, reason, _ = check_val_loss_converged(val_losses_history)
            if converged and reason:
                stop_reason = reason
                logger.info(f"  Stopping: {reason} at epoch {epoch}")
                break

            # Check stopping: budget exhausted
            if any(r <= 0 for r in remaining_budgets.values()):
                stop_reason = "budget_exhausted"
                logger.info(f"  Stopping: spectral budget exhausted at epoch {epoch}")
                break

            # Check stopping: gradient stationarity certificate
            cert = check_stopping_certificate(
                grad_norm=grad_norm,
                grad_norm_history=grad_norm_history,
                val_loss_baseline=val_losses_history[0] if val_losses_history else None,
                val_loss_current=val_loss,
                val_ci_half_width=0.0,  # no per-batch CI available
            )
            if should_certificate_stop(cert.all_conditions_met, val_losses_history):
                stop_reason = "certificate"
                logger.info(f"  Stopping: geometric certificate at epoch {epoch}")
                break

        train_losses.append({"iteration": it + 1, "train_loss": loss_float})

    training_time = time.time() - t0
    actual_iters = it + 1

    logger.info(
        f"  Geometric PiSSA: {actual_iters} iters in {training_time:.1f}s, "
        f"stop_reason={stop_reason}"
    )

    # ── Post-hoc spectral measurement ──
    spectral_info = measure_standard_lora_spectral_norms(model, 1.0)

    # ── Fuse LoRA into W_residual (identical to PiSSA) ──
    fused_dir = output_dir / "fused_model"
    fused_dir.mkdir(parents=True, exist_ok=True)

    for name, module in model.named_modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            scale = getattr(module, "scale", 1.0)
            delta = (module.lora_a @ module.lora_b) * scale
            module.linear.weight = module.linear.weight + delta.T

    from mlx.utils import tree_flatten as _tf
    weights = {}
    for k, v in _tf(model.parameters()):
        if any(t in k for t in ("lora_a", "lora_b", ".scale")):
            continue
        k = k.replace(".linear.weight", ".weight")
        k = k.replace(".linear.bias", ".bias")
        weights[k] = v
    mx.save_safetensors(str(fused_dir / "model.safetensors"), weights)
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json"]:
        src = Path(model_path) / fname
        if src.exists():
            shutil.copy2(src, fused_dir / fname)
    logger.info(f"  Geometric PiSSA: saved fused model to {fused_dir}")

    # Save adapters for reference
    adapter_weights = dict(mlx_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), adapter_weights)

    result = {
        "method": "geometric_pissa",
        "seed": seed,
        "iters": actual_iters,
        "max_iters": max_iters,
        "stop_reason": stop_reason,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "adapter_path": str(output_dir),
        "fused_model_path": str(fused_dir),
        "config": {
            "rank": rank,
            "scale": 1.0,
            "optimizer": "MASS+Fisher (derived)",
            "lr": "MASS-derived",
            "schedule": "SPS natural deceleration",
            "stopping": "geometric certificate",
        },
        "hyperparameter_count": 0,
        "pissa_init_layers": pissa_init_count,
        "spectral_geometry": {
            "gap_min": gap_min,
            "sigma_max_global": sigma_max_global,
            "eta_ceiling": eta_ceiling,
            "per_layer_gaps": {n: g["gap"] for n, g in layer_geometry.items()},
            "per_layer_budgets": {n: g["budget"] for n, g in layer_geometry.items()},
        },
        "mass_telemetry": {
            "final_eta_step": eta_step,
            "final_eta_sps": eta_sps,
            "final_remaining_budget": min(remaining_budgets.values()),
        },
        "final_val_loss": val_losses_history[-1] if val_losses_history else None,
        "final_train_loss": loss_float,
        "loss_history": {
            "train": train_losses[-10:],  # last 10 for summary
            "val": val_losses_log,
        },
        "spectral_info": spectral_info,
    }

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# EVA training (Explained Variance Adaptation)
# ---------------------------------------------------------------------------
def train_eva(
    model_path: str,
    num_layers: int,
    data_dir: Path,
    output_dir: Path,
    config: dict | None = None,
    seed: int = 42,
    iters_override: int | None = None,
    n_calib_samples: int = 256,
) -> dict:
    """Train EVA: rank allocation + initialization from input activation variance.

    1. Collect calibration activations X for each target layer
    2. Compute input covariance C = X^T @ X / N
    3. SVD(C) → explained variance per component
    4. Allocate rank per layer proportional to total explained variance
       (within total_rank budget = n_layers * base_rank)
    5. Initialize A with top eigenvectors of C (data-driven)
    6. B initialized to zero
    7. Train with standard LoRA machinery
    """
    import mlx.core as mx
    import mlx.nn as nn
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
    base_rank = cfg["rank"]

    mx.random.seed(seed)
    random.seed(seed)

    logger.info(f"  EVA: base_rank={base_rank}, lr={cfg['lr']}, seed={seed}, "
                f"calib_samples={n_calib_samples}")

    model, tokenizer = mlx_load(str(model_path))

    # --- Phase 1: Collect calibration activations ---
    train_data = load_jsonl(TRAIN_DATA)
    calib_samples = train_data[:n_calib_samples]

    # Identify target modules (same as what linear_to_lora_layers would target)
    # We need to collect input activations for each target linear layer
    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            k in name for k in ["self_attn", "mlp", "attention"]
        ):
            target_modules[name] = module

    # Activation collection via module-tree replacement.
    # Python's data model dispatches obj(x) through type(obj).__call__,
    # so instance-level __call__ monkey-patching does not work.  Instead,
    # temporarily replace each target Linear in the model tree with a thin
    # wrapper whose class-level __call__ records inputs, then restore.
    layer_activations = {name: [] for name in target_modules}

    class _RecordingLinear(nn.Module):
        """Wrapper that records input activations then delegates to inner."""

        def __init__(self, inner, activations_list):
            super().__init__()
            self.inner = inner
            self._activations = activations_list

        def __call__(self, x):
            x_flat = x.reshape(-1, x.shape[-1])
            if x_flat.shape[0] > 512:
                indices = mx.random.randint(0, x_flat.shape[0], shape=(512,))
                mx.eval(indices)
                x_flat = x_flat[indices]
            self._activations.append(x_flat)
            return self.inner(x)

    def _set_nested_attr(obj, dotted_name, value):
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        last = parts[-1]
        if last.isdigit():
            obj[int(last)] = value
        else:
            setattr(obj, last, value)

    # Install recording wrappers
    for name, module in target_modules.items():
        _set_nested_attr(
            model, name,
            _RecordingLinear(module, layer_activations[name]),
        )

    model.eval()
    for sample in calib_samples[:64]:  # limit to 64 samples for memory
        text = sample.get("text", "")
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:512]  # limit sequence length for calibration
        x = mx.array(tokens)[None, :]
        try:
            model(x)
            mx.eval(model.parameters())
        except Exception:
            pass

    # Restore original modules
    for name, module in target_modules.items():
        _set_nested_attr(model, name, module)

    # --- Phase 2: Compute explained variance and allocate ranks ---
    layer_variance = {}
    layer_eigenvectors = {}

    for name in list(layer_activations.keys()):
        acts = layer_activations[name]
        if not acts:
            continue
        X = mx.concatenate(acts, axis=0).astype(mx.float32)
        if X.shape[0] < 2:
            continue
        # Covariance: C = X^T @ X / N
        N = X.shape[0]
        C = (X.T @ X) / N
        mx.eval(C)

        try:
            # Eigendecomposition of covariance (symmetric → use eigvalsh-like)
            U_c, S_c, _ = mx.linalg.svd(C, stream=mx.cpu)
            mx.eval(U_c, S_c)
            total_var = float(mx.sum(S_c))
            if total_var > 0:
                layer_variance[name] = total_var
                layer_eigenvectors[name] = U_c  # columns are eigenvectors
        except Exception:
            logger.warning(f"  EVA: SVD failed for {name}, skipping")

        # Free activations
        layer_activations[name] = []

    if not layer_variance:
        logger.warning("  EVA: no variance data, falling back to standard LoRA")
        del model, tokenizer
        mx.clear_cache()
        gc.collect()
        return train_standard_lora(
            model_path, num_layers, data_dir, output_dir,
            config=cfg, seed=seed, iters_override=iters_override,
        )

    # Allocate ranks proportional to explained variance (informational only).
    # EVA-init for R1: we use data-driven eigenvector initialization (EVA's
    # primary contribution) with uniform rank. Per-layer rank allocation would
    # require reimplementing linear_to_lora_layers logic for marginal benefit —
    # the comparison is NB-LoRA vs baselines, not EVA-paper-faithful vs EVA-lite.
    total_budget = len(layer_variance) * base_rank
    total_var_sum = sum(layer_variance.values())
    layer_ranks = {}
    for name, var in layer_variance.items():
        # Proportional allocation, minimum rank 1
        raw_rank = max(1, round(total_budget * var / total_var_sum))
        layer_ranks[name] = min(raw_rank, base_rank * 2)  # cap at 2x base

    # Redistribute to hit budget exactly
    current_total = sum(layer_ranks.values())
    if current_total != total_budget:
        # Sort by variance (highest first) and adjust
        sorted_layers = sorted(layer_ranks.keys(), key=lambda n: layer_variance[n], reverse=True)
        diff = total_budget - current_total
        for layer_name in sorted_layers:
            if diff == 0:
                break
            adjustment = 1 if diff > 0 else -1
            new_rank = layer_ranks[layer_name] + adjustment
            if new_rank >= 1:
                layer_ranks[layer_name] = new_rank
                diff -= adjustment

    rank_info = {name: layer_ranks[name] for name in sorted(layer_ranks.keys())}
    logger.info(f"  EVA rank allocation (informational — uniform rank={base_rank} used): "
                f"{json.dumps(rank_info, indent=2)}")

    # --- Phase 3: Standard LoRA training with EVA initialization ---
    # Reload model fresh (calibration may have polluted state)
    del model, tokenizer
    mx.clear_cache()
    gc.collect()
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
            "rank": base_rank,
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

    # Apply EVA initialization: set A to top eigenvectors of input covariance
    eva_init_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_a") and name in layer_eigenvectors:
            eigvecs = layer_eigenvectors[name]
            r = module.lora_a.shape[1] if module.lora_a.ndim == 2 else base_rank
            r = min(r, eigvecs.shape[1])
            A_init = eigvecs[:, :r].astype(module.lora_a.dtype)
            if A_init.shape == module.lora_a.shape:
                module.lora_a = A_init
                mx.eval(module.lora_a)
                eva_init_count += 1

    logger.info(f"  EVA: initialized {eva_init_count} layers with eigenvector A")

    # Diagnostic: compare calibration targets vs actual LoRA targets
    lora_names = {name for name, m in model.named_modules() if hasattr(m, "lora_a")}
    calib_names = set(layer_eigenvectors.keys())
    matched = calib_names & lora_names
    calib_only = calib_names - lora_names
    lora_only = lora_names - calib_names
    logger.info(f"  EVA target match: {len(matched)}/{len(lora_names)} LoRA layers have "
                f"eigenvectors ({len(calib_only)} calib-only, {len(lora_only)} lora-only)")
    if eva_init_count == 0:
        logger.error("  EVA: 0 layers initialized — all LoRA layers missed calibration. "
                     "EVA arm is effectively standard LoRA with random init.")
    if calib_only:
        logger.info(f"  EVA calib-only (wasted): {sorted(calib_only)[:5]}{'...' if len(calib_only) > 5 else ''}")
    if lora_only:
        logger.info(f"  EVA lora-only (no eigvecs): {sorted(lora_only)[:5]}{'...' if len(lora_only) > 5 else ''}")

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

    lr_value = cfg["lr"]
    if cfg.get("lr_schedule") == "cosine":
        lr_value = optim.cosine_decay(init=cfg["lr"], decay_steps=iters)
    if cfg["optimizer"] == "adamw":
        opt = optim.AdamW(learning_rate=lr_value, weight_decay=0.0)
    else:
        opt = optim.Adam(learning_rate=lr_value)

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

    adapter_file_check = output_dir / "adapters.safetensors"
    if not adapter_file_check.exists():
        from mlx.utils import tree_flatten as tf
        adapter_weights = dict(tf(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_file_check), adapter_weights)

    spectral_info = measure_standard_lora_spectral_norms(model, cfg["scale"])

    result = {
        "method": "eva",
        "seed": seed,
        "iters": iters,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "adapter_path": str(output_dir),
        "config": {k: v for k, v in cfg.items()},
        "hyperparameter_count": len([
            k for k in cfg if k not in ("max_seq_length", "batch_size", "iters")
        ]),
        "eva_init_layers": eva_init_count,
        "eva_rank_allocation": rank_info,
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
# Method dispatch table
# ---------------------------------------------------------------------------
def _train_method(
    method_name: str,
    model_path: str,
    num_layers: int,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    iters: int,
    best_tuned_config: dict | None = None,
    weight_decay: float = 0.0,
    nb_surface: "NBTargetSurface | None" = None,
) -> dict:
    """Dispatch training to the right function based on method name."""
    if method_name == "standard_lora":
        result = train_standard_lora(
            model_path, num_layers, data_dir, output_dir,
            config=STANDARD_LORA_CONFIG, seed=seed, iters_override=iters,
        )
    elif method_name == "tuned_lora":
        if best_tuned_config is None:
            raise ValueError("tuned_lora requires grid search results")
        result = train_standard_lora(
            model_path, num_layers, data_dir, output_dir,
            config=best_tuned_config, seed=seed, iters_override=iters,
        )
    elif method_name == "dora":
        result = train_standard_lora(
            model_path, num_layers, data_dir, output_dir,
            config=DORA_CONFIG, seed=seed, iters_override=iters,
            fine_tune_type="dora",
        )
    elif method_name == "rslora":
        result = train_standard_lora(
            model_path, num_layers, data_dir, output_dir,
            config=RSLORA_CONFIG, seed=seed, iters_override=iters,
        )
    elif method_name == "pissa":
        result = train_pissa(
            model_path, num_layers, data_dir, output_dir,
            seed=seed, iters_override=iters,
        )
    elif method_name == "eva":
        result = train_eva(
            model_path, num_layers, data_dir, output_dir,
            seed=seed, iters_override=iters,
        )
    elif method_name == "nb_lora":
        result = train_nb_lora(
            model_path, output_dir, seed=seed,
            weight_decay=weight_decay,
        )
    elif method_name == "geometric_pissa":
        result = train_geometric_pissa(
            model_path, num_layers, data_dir, output_dir,
            seed=seed,
        )
    elif method_name == "standard_nb_surface":
        if nb_surface is None:
            raise ValueError("standard_nb_surface requires nb_surface")
        result = train_surface_matched(
            model_path, data_dir, output_dir,
            rank_overrides=nb_surface.rank_overrides,
            config=STANDARD_LORA_CONFIG, seed=seed, iters_override=iters,
            fine_tune_type="lora",
        )
    elif method_name == "pissa_nb_surface":
        if nb_surface is None:
            raise ValueError("pissa_nb_surface requires nb_surface")
        result = train_surface_matched(
            model_path, data_dir, output_dir,
            rank_overrides=nb_surface.rank_overrides,
            config=STANDARD_LORA_CONFIG, seed=seed, iters_override=iters,
            fine_tune_type="lora", pissa_init=True,
        )
    elif method_name == "dora_nb_surface":
        if nb_surface is None:
            raise ValueError("dora_nb_surface requires nb_surface")
        result = train_surface_matched(
            model_path, data_dir, output_dir,
            rank_overrides=nb_surface.rank_overrides,
            config=DORA_CONFIG, seed=seed, iters_override=iters,
            fine_tune_type="dora",
        )
    elif method_name == "geometric_pissa_nb_surface":
        if nb_surface is None:
            raise ValueError("geometric_pissa_nb_surface requires nb_surface")
        result = train_geometric_pissa(
            model_path, num_layers, data_dir, output_dir,
            seed=seed, rank_overrides=nb_surface.rank_overrides,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    result["method"] = method_name
    return result


# Method display names
METHOD_DISPLAY = {
    "standard_lora": "Standard LoRA",
    "tuned_lora": "Tuned LoRA",
    "dora": "DoRA",
    "rslora": "rsLoRA",
    "pissa": "PiSSA",
    "eva": "EVA",
    "nb_lora": "NB-LoRA",
    "geometric_pissa": "Geometric PiSSA",
    "standard_nb_surface": "Standard LoRA (NB surface)",
    "pissa_nb_surface": "PiSSA (NB surface)",
    "dora_nb_surface": "DoRA (NB surface)",
    "geometric_pissa_nb_surface": "Geometric PiSSA (NB surface)",
}

# Methods with post-hoc spectral measurement (not bounded by construction)
SPECTRAL_POSTHOC_METHODS = {
    "standard_lora", "tuned_lora", "dora", "rslora", "pissa", "eva",
    "geometric_pissa",
    "standard_nb_surface", "pissa_nb_surface", "dora_nb_surface",
    "geometric_pissa_nb_surface",
}

# Methods that use NB-LoRA's geometry-derived pipeline
NBLORA_METHODS = {"nb_lora"}


# ---------------------------------------------------------------------------
# Single model experiment (multi-seed, N-arm)
# ---------------------------------------------------------------------------
def run_model_experiment(
    model_key: str,
    model_spec: dict,
    output_dir: Path,
    seeds: list[int],
    methods: list[str],
    quick: bool = False,
    skip_eval: bool = False,
    skip_inference: bool = False,
    skip_grid: bool = False,
    weight_decay: float = 0.0,
    derive_wd_test: bool = False,
) -> dict:
    """Run N-arm comparison for one model across multiple seeds."""
    model_path = str(model_spec["path"])
    model_name = model_spec["name"]
    num_layers = model_spec["num_layers"]

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL: {model_name} ({model_key})")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Quick: {quick}, Skip eval: {skip_eval}, Skip grid: {skip_grid}")
    logger.info(f"{'='*70}")

    results = {
        "model": model_name,
        "model_path": model_path,
        "seeds": seeds,
        "methods": methods,
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
    best_tuned_config = None
    grid_summary = None
    if "tuned_lora" in methods and not skip_grid:
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
    elif "tuned_lora" in methods and skip_grid:
        # tuned_lora requested but grid skipped — remove from active methods
        methods = [m for m in methods if m != "tuned_lora"]
        results["grid_search"] = "skipped"
    else:
        results["grid_search"] = "not_applicable"
    results["methods"] = methods

    # ------------------------------------------------------------------
    # b2. DERIVE NB TARGET SURFACE (once, if any surface method requested)
    # ------------------------------------------------------------------
    nb_surface = None
    if any(m in NB_SURFACE_METHODS for m in methods):
        logger.info(f"\n--- [{model_name}] Deriving NB-LoRA target surface ---")
        from modelcypher.backends import initialize_default_backend
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.dataset_training_service import (
            DatasetTrainingService,
            NBTargetSurface,
        )
        initialize_default_backend()
        backend = get_default_backend()
        adapter = backend.create_adapter()
        svc = DatasetTrainingService(adapter, backend)
        nb_surface = svc.derive_nb_target_surface(
            model_path=model_path,
            dataset_path=str(TRAIN_DATA),
            eval_dataset_path=str(VAL_DATA),
            seed=42,
        )
        results["nb_surface"] = nb_surface.to_dict()
        logger.info(
            f"  NB surface: {len(nb_surface.target_keys)} modules, "
            f"ranks={sorted(set(nb_surface.rank_overrides.values()))}, "
            f"ceiling={nb_surface.rank_ceiling_source}"
        )
        # Clean up
        del svc, adapter
        import mlx.core as mx
        mx.clear_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # c. PER-SEED TRAINING + EVAL (generic N-arm loop)
    # ------------------------------------------------------------------
    nb_weight_decay = 0.0 if derive_wd_test else weight_decay

    # Per-method accumulators
    method_evals: dict[str, list[dict]] = {m: [] for m in methods}
    method_trainings: dict[str, list[dict]] = {m: [] for m in methods}
    # WD falsification (opt-in, NB-LoRA only)
    nb_wd_evals: list[dict] = []
    nb_wd_trainings: list[dict] = []
    wd_derived_value = None
    wd_derivation_info = None
    per_seed_data = {}

    for seed in seeds:
        logger.info(f"\n{'='*50}")
        logger.info(f"[{model_name}] Seed {seed}")
        logger.info(f"{'='*50}")

        seed_dir = model_dir / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_result: dict = {"seed": seed}

        for method_name in methods:
            display = METHOD_DISPLAY.get(method_name, method_name)
            logger.info(f"\n--- [{model_name}] {display} (seed={seed}) ---")
            method_out = seed_dir / method_name

            try:
                train_result = _train_method(
                    method_name=method_name,
                    model_path=model_path,
                    num_layers=num_layers,
                    data_dir=std_data_dir,
                    output_dir=method_out,
                    seed=seed,
                    iters=iters,
                    best_tuned_config=best_tuned_config,
                    weight_decay=nb_weight_decay,
                    nb_surface=nb_surface,
                )

                # Commensurable val_loss: same operator for all arms
                train_result["callback_val_loss"] = train_result.get("final_val_loss")
                logger.info(f"  Commensurable val_loss eval ({method_name})...")

                # PiSSA modifies base weights, so eval must use the fused model
                # (original model + adapter would double-count principal components)
                fused_path = train_result.get("fused_model_path")
                eval_model = fused_path if fused_path else model_path
                eval_adapter = None if fused_path else str(method_out)

                train_result["final_val_loss"] = evaluate_val_loss(
                    eval_model, eval_adapter, VAL_DATA,
                )
                logger.info(f"  val_loss={train_result['final_val_loss']:.4f} "
                            f"(callback={train_result['callback_val_loss']})")

                with open(method_out / "training_log.json", "w") as f:
                    json.dump(train_result, f, indent=2, default=str)
                method_trainings[method_name].append(train_result)

                if not skip_eval:
                    eval_result = run_lm_eval(
                        model_path=eval_model,
                        tasks=BENCHMARK_TASKS,
                        adapter_path=eval_adapter,
                        limit=eval_limit,
                    )
                else:
                    eval_result = {}
                with open(method_out / "eval_results.json", "w") as f:
                    json.dump(eval_result, f, indent=2)
                method_evals[method_name].append(eval_result)

                seed_result[method_name] = {
                    "training": train_result, "eval": eval_result,
                }
            except Exception as e:
                logger.exception(f"  {display} seed={seed} failed")
                seed_result[method_name] = {"error": str(e)}

        # ---- NB-LoRA WD=derived (if --derive-wd-test) ----
        if (derive_wd_test
                and "nb_lora" in methods
                and "error" not in seed_result.get("nb_lora", {})):
            nb_out = seed_dir / "nb_lora"
            nb_train_wd0 = seed_result["nb_lora"]["training"]
            try:
                wd_derived, wd_deriv = derive_wd_from_training_result(
                    nb_train_wd0, str(nb_out),
                )
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

                method_responses: dict[str, list] = {"base": base_responses}
                for method_name in methods:
                    if "error" not in seed_result.get(method_name, {}):
                        adapter = str(seed_dir / method_name)
                        method_responses[method_name] = generate_inference_responses(
                            model_path, adapter, prompts, method_name,
                        )
                    else:
                        method_responses[method_name] = []

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
                            name: next((r["response"] for r in resps if r["id"] == pid), "")
                            for name, resps in method_responses.items()
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
        method_evals=method_evals,
        method_trainings=method_trainings,
        methods=methods,
        grid_summary=grid_summary,
    )
    results["comparison"] = comparison

    # WD falsification (opt-in)
    if derive_wd_test and nb_wd_trainings and wd_derived_value is not None:
        wd_falsification = build_wd_falsification(
            baseline_scores=baseline_scores,
            nb_wd0_evals=method_evals.get("nb_lora", []),
            nb_wdD_evals=nb_wd_evals,
            nb_wd0_trainings=method_trainings.get("nb_lora", []),
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

    print_comparison(model_name, comparison, methods)

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
    method_evals: dict[str, list[dict]],
    method_trainings: dict[str, list[dict]],
    methods: list[str],
    grid_summary: dict | None = None,
) -> dict:
    """Build multi-seed comparison across N arms."""
    comparison: dict = {
        "benchmarks": {},
        "training": {},
        "spectral": {},
        "head_to_head": {},
    }

    # Aggregate benchmarks per arm
    method_agg = {
        m: aggregate_arm_evals(method_evals.get(m, []), baseline_scores)
        for m in methods
    }

    for task in BENCHMARK_TASKS:
        entry: dict = {"task": task}
        # Find baseline from first method that has it
        for m in methods:
            if task in method_agg[m]:
                entry["baseline"] = method_agg[m][task]["baseline"]
                entry["metric"] = method_agg[m][task]["metric"]
                break
        if "baseline" not in entry:
            continue

        for m in methods:
            if task in method_agg[m]:
                entry[m] = {
                    "mean": method_agg[m][task]["mean"],
                    "std": method_agg[m][task]["std"],
                    "n_seeds": method_agg[m][task]["n_seeds"],
                    "delta_vs_baseline": method_agg[m][task]["delta_vs_baseline"],
                }

        comparison["benchmarks"][task] = entry

    # Training stats
    for m in methods:
        trainings = method_trainings.get(m, [])
        times = [t.get("training_time_seconds", 0) for t in trainings]
        val_losses = [
            t.get("final_val_loss", 0) for t in trainings
            if t.get("final_val_loss") is not None
        ]
        comparison["training"][m] = {
            "n_runs": len(trainings),
            "training_time_mean": _safe_mean(times),
            "training_time_std": _safe_stdev(times),
            "final_val_loss_mean": _safe_mean(val_losses),
            "final_val_loss_std": _safe_stdev(val_losses),
            "hyperparameter_count": (
                trainings[0].get("hyperparameter_count", 0) if trainings else 0
            ),
        }

    # Spectral safety
    for m in methods:
        trainings = method_trainings.get(m, [])
        if m in NBLORA_METHODS:
            comparison["spectral"][m] = {
                "bounded_by_construction": True,
                "max_spectral_ratio_mean": _safe_mean([
                    t.get("max_spectral_ratio", 0) for t in trainings
                ]),
                "spectral_bounds_ok": all(
                    t.get("spectral_bounds_ok", False) for t in trainings
                ),
            }
        else:
            max_norms = [
                t.get("spectral_info", {}).get("max_spectral_norm", 0)
                for t in trainings
            ]
            comparison["spectral"][m] = {
                "max_norm_mean": _safe_mean(max_norms),
                "bounded_by_construction": False,
            }

    # Head-to-head: NB-LoRA vs every other method
    if "nb_lora" in methods:
        nb_agg = method_agg["nb_lora"]
        for opponent in methods:
            if opponent == "nb_lora":
                continue
            opp_agg = method_agg[opponent]

            nb_wins = 0
            opponent_wins = 0
            ties = 0
            deltas = []
            per_task_bands = {}

            for task in BENCHMARK_TASKS:
                if task in nb_agg and task in opp_agg:
                    nb_mean = nb_agg[task]["mean"]
                    opp_mean = opp_agg[task]["mean"]
                    nb_se = nb_agg[task]["se_mean"]
                    opp_se = opp_agg[task]["se_mean"]
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

            key = f"nb_vs_{opponent}"
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
def print_comparison(
    model_name: str,
    comparison: dict,
    methods: list[str] | None = None,
):
    """Print human-readable multi-seed comparison for N arms."""
    benchmarks = comparison.get("benchmarks", {})
    training = comparison.get("training", {})
    h2h = comparison.get("head_to_head", {})

    # Determine which methods are present in the data
    if methods is None:
        methods = list(training.keys())

    print(f"\n{'='*80}")
    print(f"  COMPARISON: {model_name}")
    print(f"{'='*80}")

    # Header: Task | Base | method1 | method2 | ...
    col_width = 16
    header = f"  {'Task':<18} {'Base':>8}"
    for m in methods:
        label = METHOD_DISPLAY.get(m, m)[:col_width]
        header += f" {label:>{col_width}}"
    print(header)
    print(f"  {'-'*(20 + 8 + (col_width + 1) * len(methods))}")

    for task, data in benchmarks.items():
        base = data.get("baseline", 0)
        row = f"  {task:<18} {base:>8.4f}"
        for m in methods:
            arm = data.get(m, {})
            if arm:
                row += f" {arm.get('mean', 0):>7.4f}+/-{arm.get('std', 0):.3f}"
            else:
                row += f" {'---':>{col_width}}"
        print(row)

    # Training stats
    print(f"\n  Training:")
    for m in methods:
        t = training.get(m, {})
        if t.get("n_runs", 0) == 0:
            continue
        hp = t.get("hyperparameter_count", "?")
        time_str = f"{t['training_time_mean']:.1f}+/-{t['training_time_std']:.1f}s"
        vloss_str = f"{t['final_val_loss_mean']:.4f}+/-{t['final_val_loss_std']:.4f}"
        label = METHOD_DISPLAY.get(m, m)
        print(f"    {label:<14} time={time_str:<18} val_loss={vloss_str:<20} HPs={hp}")

    # Head-to-head
    if h2h:
        print(f"\n  Head-to-Head (NB-LoRA vs each):")
        for key, h in h2h.items():
            opponent = key.replace("nb_vs_", "")
            label = METHOD_DISPLAY.get(opponent, opponent)
            print(f"    vs {label}: NB wins {h['nb_wins']}, "
                  f"opponent wins {h['opponent_wins']}, "
                  f"ties {h['ties']}, "
                  f"mean delta {h['mean_delta']:+.4f}")

    # Grid search
    gs = comparison.get("grid_search", {})
    if gs:
        bc = gs.get("best_config", {})
        print(f"\n  Grid Search ({gs.get('n_configs_tested', '?')} configs):")
        print(f"    Best: rank={bc.get('rank')}, lr={bc.get('lr')}, "
              f"scale={bc.get('scale', 0):.2f}")
        print(f"    Best val loss: {gs.get('best_val_loss', '?')}")

    # WD falsification
    wdf = comparison.get("weight_decay_falsification")
    if wdf:
        print(f"\n  WD Falsification:")
        print(f"    Prediction: {wdf['prediction']}")
        print(f"    WD derived: {wdf['wd_derived']:.6f} "
              f"({wdf['wd_derivation']['formula']})")
        print(f"    Val loss: WD=0 {wdf['training_val_loss']['wd0_mean']:.4f}, "
              f"WD=derived {wdf['training_val_loss']['wdD_mean']:.4f}")
        for task, td in wdf.get("per_task", {}).items():
            print(f"    {task:<18} WD=0={td['wd0_mean']:.4f}  "
                  f"WD=D={td['wdD_mean']:.4f}  "
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
        description="NB-LoRA vs PEFT baselines — N-arm head-to-head comparison (R1)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS.keys()),
        default=["350M"],
        help="Model scales to test (default: 350M)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHOD_NAMES,
        default=["standard_lora", "tuned_lora", "nb_lora"],
        help=(
            f"Methods to include (default: standard_lora tuned_lora nb_lora). "
            f"Available: {', '.join(ALL_METHOD_NAMES)}"
        ),
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
    logger.info("NB-LoRA vs PEFT Baselines — Head-to-Head Comparison (R1)")
    logger.info("=" * 70)
    logger.info(f"Models: {args.models}")
    logger.info(f"Methods: {args.methods}")
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
                methods=args.methods,
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
        "methods": args.methods,
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
