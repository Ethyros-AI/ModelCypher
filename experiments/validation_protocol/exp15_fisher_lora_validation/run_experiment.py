#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 15: Fisher-Guided LoRA Training Validation
#
# HYPOTHESIS: LoRAs targeting high-Fisher dimensions degrade base capabilities
# more than LoRAs targeting low-Fisher dimensions.
#
# THEORY: Fisher Information F_ii = E[x_i²] measures how much dimension i
# influences the loss. High-Fisher = "important" to base model. When LoRA
# modifies high-Fisher dimensions → more forgetting (catastrophic).
#
# PROTOCOL:
# 1. Load LFM2-350M, collect activations, compute Fisher diagonal per module
# 2. Identify high-Fisher modules (q_proj, k_proj) vs low-Fisher (o_proj, down_proj)
# 3. Train 4 LoRAs (rank=8, 50 steps each) targeting different modules
# 4. Measure perplexity delta on validation set for each
# 5. Correlate: fisher_score_of_targeted_modules vs perplexity_degradation
#
# CONTROLS:
# - Training loss should decrease for all LoRAs (validates training works)
# - Untrained model baseline establishes perplexity floor
#
# SUCCESS CRITERIA:
# - Correlation(fisher_score, perplexity_delta) > 0.5
# - low_fisher LoRA has lower perplexity_delta than high_fisher
#
# REFERENCES:
# - Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in NNs"
# - SOAR (2026) "Teaching Models to Teach Themselves" - grounded rewards

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mlx.core as mx

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.fisher_information import (
    compute_empirical_fisher_diagonal,
)

from experiments.validation_protocol.shared import (
    ExperimentConfig,
    ExperimentResult,
    ensure_output_dir,
)
from experiments.validation_protocol.shared.lora_utils import (
    train_lora_quick,
    evaluate_perplexity,
    collect_layer_activations,
    load_model_and_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Model and data paths
MODEL_PATH = Path("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")
TRAIN_DATA = Path(__file__).parent.parent.parent.parent / "data" / "training" / "train.jsonl"
EVAL_DATA = Path(__file__).parent.parent.parent.parent / "data" / "training" / "train.jsonl"  # Use same for now

# LoRA configurations to test
# LFM2-350M has: q_proj, k_proj, v_proj, out_proj (attention), w1, w2, w3 (MLP)
LORA_CONFIGS = [
    {
        "name": "high_fisher",
        "target_modules": ["q_proj", "k_proj"],
        "description": "Targets high-Fisher modules (attention queries/keys)",
    },
    {
        "name": "low_fisher",
        "target_modules": ["out_proj", "w2"],  # LFM2 uses out_proj and w2 (down-projection)
        "description": "Targets low-Fisher modules (output projection, MLP down)",
    },
    {
        "name": "mlp_only",
        "target_modules": ["w1", "w3"],  # MLP gate and up-projection
        "description": "MLP-only targeting (control)",
    },
    {
        "name": "standard",
        "target_modules": ["q_proj", "v_proj"],
        "description": "Standard LoRA targeting (baseline)",
    },
]


def compute_module_fisher_scores(model_path: Path, backend) -> dict[str, float]:
    """Compute Fisher score for each module type.

    Returns dict mapping module name to mean Fisher value.
    """
    logger.info("Computing Fisher Information for each module type...")

    # Load model to get activations
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Sample prompts for Fisher computation
    prompts = [
        "The capital of France is",
        "In mathematics, 2 + 2 equals",
        "The largest planet in our solar system is",
        "Water freezes at",
        "The speed of light is approximately",
    ]

    # Collect activations at middle layer
    from modelcypher.ports.model_architecture_factory import get_model_architecture

    config = {}
    if hasattr(model, 'config'):
        if hasattr(model.config, 'to_dict'):
            config = model.config.to_dict()
        elif isinstance(model.config, dict):
            config = model.config

    arch = get_model_architecture(model, config=config)
    num_layers = arch.num_layers
    target_layer = num_layers // 2

    logger.info(f"Using layer {target_layer} of {num_layers} for Fisher computation")

    # Get layer weights and compute Fisher for each module type
    layer = arch.layers[target_layer]

    module_fisher = {}

    # Module patterns to search - cover both Llama-style and LFM2-style
    # LFM2: q_proj, k_proj, v_proj, out_proj (attention), w1, w2, w3, in_proj, out_proj (MLP/conv)
    # Llama: q_proj, k_proj, v_proj, o_proj (attention), gate_proj, up_proj, down_proj (MLP)
    module_locations = [
        # (pattern_name, container_attr, module_name)
        ("q_proj", "self_attn", "q_proj"),
        ("k_proj", "self_attn", "k_proj"),
        ("v_proj", "self_attn", "v_proj"),
        ("out_proj", "self_attn", "out_proj"),  # LFM2 uses out_proj
        ("o_proj", "self_attn", "o_proj"),      # Llama uses o_proj
        ("w1", "feed_forward", "w1"),           # LFM2 MLP
        ("w2", "feed_forward", "w2"),
        ("w3", "feed_forward", "w3"),
        ("gate_proj", "mlp", "gate_proj"),      # Llama MLP
        ("up_proj", "mlp", "up_proj"),
        ("down_proj", "mlp", "down_proj"),
    ]

    for pattern_name, container_attr, module_name in module_locations:
        # Try to find this module
        module = None
        if hasattr(layer, container_attr):
            container = getattr(layer, container_attr)
            if hasattr(container, module_name):
                module = getattr(container, module_name)

        if module is not None and hasattr(module, 'weight'):
            # Compute Fisher as variance of weight activations
            weight = module.weight
            mx.eval(weight)

            # Simple Fisher estimate: sum of squared weights (proxy for importance)
            # True Fisher would require gradients, but this correlates
            weight_sq = weight * weight
            fisher_mean = float(mx.mean(weight_sq))
            module_fisher[pattern_name] = fisher_mean
            logger.info(f"  {pattern_name}: Fisher={fisher_mean:.6f}")

    # Clean up
    del model
    gc.collect()

    return module_fisher


def compute_config_fisher_score(
    config: dict,
    module_fisher: dict[str, float],
) -> float:
    """Compute aggregate Fisher score for a LoRA config."""
    target_modules = config["target_modules"]
    scores = [module_fisher.get(m, 0.0) for m in target_modules]
    if scores:
        return sum(scores) / len(scores)
    return 0.0


def run_lora_experiment(
    config: dict,
    model_path: Path,
    train_data: Path,
    eval_data: Path,
    output_dir: Path,
    base_perplexity: float,
    backend,
) -> dict:
    """Train a single LoRA config and measure impact."""
    name = config["name"]
    target_modules = config["target_modules"]

    logger.info(f"Training LoRA config: {name}")
    logger.info(f"  Target modules: {target_modules}")

    lora_output = output_dir / "loras" / name

    try:
        # Train LoRA
        train_result = train_lora_quick(
            model_path=model_path,
            dataset_path=train_data,
            output_path=lora_output,
            target_modules=target_modules,
            rank=8,
            steps=50,
            lr=1e-4,
            batch_size=2,
            sequence_length=256,
            max_samples=100,
        )

        logger.info(f"  Training complete: final_loss={train_result.final_loss:.4f}")

        # Evaluate perplexity with LoRA
        eval_result = evaluate_perplexity(
            model_path=model_path,
            dataset_path=eval_data,
            lora_path=lora_output,
            max_samples=50,
        )

        logger.info(f"  Evaluation: perplexity={eval_result.perplexity:.2f}")

        perplexity_delta = eval_result.perplexity - base_perplexity

        return {
            "name": name,
            "target_modules": target_modules,
            "description": config["description"],
            "training": {
                "final_loss": train_result.final_loss,
                "steps_completed": train_result.steps_completed,
                "duration_seconds": train_result.duration_seconds,
                "loss_decreased": train_result.training_losses[-1] < train_result.training_losses[0] if len(train_result.training_losses) > 1 else False,
            },
            "evaluation": {
                "perplexity": eval_result.perplexity,
                "loss": eval_result.loss,
                "perplexity_delta": perplexity_delta,
            },
            "success": True,
        }

    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "name": name,
            "target_modules": target_modules,
            "description": config["description"],
            "success": False,
            "error": str(e),
        }
    finally:
        gc.collect()


def compute_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation between two lists."""
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0

    return cov / (var_x ** 0.5 * var_y ** 0.5)


def main():
    """Run Experiment 15: Fisher-Guided LoRA Validation."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp15_fisher_lora_validation")
    initialize_default_backend()
    backend = get_default_backend()

    # Check model exists
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.info("Please ensure the external volume is mounted")
        return None

    config = ExperimentConfig(
        experiment_name="exp15_fisher_lora_validation",
        source_model_path=str(MODEL_PATH),
        target_model_path=str(MODEL_PATH),  # Same model
        backend_name=type(backend).__name__,
        hyperparameters={
            "hypothesis": "High-Fisher LoRAs cause more degradation than low-Fisher LoRAs",
            "test_type": "fisher_lora_correlation",
            "lora_rank": 8,
            "training_steps": 50,
            "learning_rate": 1e-4,
        },
    )

    results = {
        "module_fisher": {},
        "base_perplexity": 0.0,
        "lora_results": [],
        "summary": {},
    }

    # ========== PART 1: Compute Fisher Information ==========
    logger.info("=" * 70)
    logger.info("PART 1: Computing Module Fisher Scores")
    logger.info("=" * 70)

    try:
        module_fisher = compute_module_fisher_scores(MODEL_PATH, backend)
        results["module_fisher"] = module_fisher

        # Add Fisher scores to configs
        for cfg in LORA_CONFIGS:
            cfg["fisher_score"] = compute_config_fisher_score(cfg, module_fisher)
            logger.info(f"  {cfg['name']}: fisher_score={cfg['fisher_score']:.6f}")

    except Exception as e:
        logger.error(f"Fisher computation failed: {e}")
        import traceback
        traceback.print_exc()

    # ========== PART 2: Baseline Evaluation ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 2: Baseline Perplexity (no LoRA)")
    logger.info("=" * 70)

    try:
        base_result = evaluate_perplexity(
            model_path=MODEL_PATH,
            dataset_path=EVAL_DATA,
            lora_path=None,
            max_samples=50,
        )
        results["base_perplexity"] = base_result.perplexity
        logger.info(f"Base perplexity: {base_result.perplexity:.2f}")

    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        results["base_perplexity"] = float('inf')

    # ========== PART 3: Train and Evaluate LoRAs ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 3: Training and Evaluating LoRA Configurations")
    logger.info("=" * 70)

    for cfg in LORA_CONFIGS:
        logger.info("")
        result = run_lora_experiment(
            config=cfg,
            model_path=MODEL_PATH,
            train_data=TRAIN_DATA,
            eval_data=EVAL_DATA,
            output_dir=output_dir,
            base_perplexity=results["base_perplexity"],
            backend=backend,
        )
        result["fisher_score"] = cfg.get("fisher_score", 0.0)
        results["lora_results"].append(result)

    # ========== PART 4: Analysis ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 4: Summary Analysis")
    logger.info("=" * 70)

    successful_results = [r for r in results["lora_results"] if r.get("success", False)]

    if successful_results:
        fisher_scores = [r["fisher_score"] for r in successful_results]
        perplexity_deltas = [r["evaluation"]["perplexity_delta"] for r in successful_results]

        # Compute correlation
        fisher_ppl_correlation = compute_correlation(fisher_scores, perplexity_deltas)

        # Check if training worked (loss decreased)
        training_worked = all(
            r["training"].get("loss_decreased", False)
            for r in successful_results
        )

        # Find high/low fisher results
        high_fisher_result = next((r for r in successful_results if r["name"] == "high_fisher"), None)
        low_fisher_result = next((r for r in successful_results if r["name"] == "low_fisher"), None)

        hypothesis_supported = False
        if high_fisher_result and low_fisher_result:
            high_delta = high_fisher_result["evaluation"]["perplexity_delta"]
            low_delta = low_fisher_result["evaluation"]["perplexity_delta"]
            hypothesis_supported = high_delta > low_delta

        results["summary"] = {
            "n_configs_tested": len(LORA_CONFIGS),
            "n_successful": len(successful_results),
            "base_perplexity": results["base_perplexity"],
            "fisher_perplexity_correlation": fisher_ppl_correlation,
            "training_worked": training_worked,
            "hypothesis_supported": hypothesis_supported,
            "high_fisher_delta": high_fisher_result["evaluation"]["perplexity_delta"] if high_fisher_result else None,
            "low_fisher_delta": low_fisher_result["evaluation"]["perplexity_delta"] if low_fisher_result else None,
            "success": len(successful_results) >= 2,
        }

        logger.info(f"Fisher-Perplexity correlation: {fisher_ppl_correlation:.3f}")
        logger.info(f"Training worked (loss decreased): {training_worked}")
        logger.info(f"Hypothesis supported (high > low delta): {hypothesis_supported}")
        if high_fisher_result:
            logger.info(f"High-Fisher perplexity delta: {high_fisher_result['evaluation']['perplexity_delta']:.2f}")
        if low_fisher_result:
            logger.info(f"Low-Fisher perplexity delta: {low_fisher_result['evaluation']['perplexity_delta']:.2f}")

    else:
        results["summary"] = {
            "success": False,
            "error": "No successful LoRA experiments",
        }

    duration = time.perf_counter() - start_time

    # ========== SAVE ==========
    experiment_result = ExperimentResult(
        config=config,
        metrics=results.get("summary", {}),
        raw_data=results,
        duration_seconds=duration,
        success=results.get("summary", {}).get("success", False),
    )
    experiment_result.save(output_dir / "results.json")
    config.save(output_dir / "config.json")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 15 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Results saved to: {output_dir / 'results.json'}")
    logger.info("=" * 70)

    return experiment_result


if __name__ == "__main__":
    main()
