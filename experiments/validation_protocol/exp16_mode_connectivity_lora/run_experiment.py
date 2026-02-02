#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 16: Mode Connectivity LoRA Barrier Validation
#
# HYPOTHESIS: Mode connectivity barrier (base → base+LoRA) correlates with
# downstream task degradation. Higher barrier = worse preservation.
#
# THEORY: Models in the same loss basin can be smoothly interpolated without
# crossing high-loss regions. When a LoRA pushes the model into a different
# basin (high barrier), the merged model loses coherence.
#
# PROTOCOL:
# 1. Train LoRAs with varying "aggressiveness" (rank and training steps)
# 2. For each merged model, compute CKA barrier along interpolation path
# 3. Measure perplexity delta on validation set
# 4. Correlate: barrier_height vs perplexity_delta
#
# CONTROLS:
# - Untrained LoRA: barrier ~ 0 (by construction)
# - High rank + many steps: high barrier expected
#
# SUCCESS CRITERIA:
# - barrier_height ~ 0 for untrained model
# - barrier_height increases with rank and training steps
# - Correlation(barrier_height, perplexity_delta) > 0.6
#
# REFERENCES:
# - Draxler et al. (2018) "Essentially No Barriers in Neural Network Energy Landscape"
# - Garipov et al. (2018) "Loss Surfaces, Mode Connectivity, and Fast Ensembling"
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
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations
from modelcypher.core.domain.geometry.mode_connectivity import (
    analyze_mode_connectivity,
    InterpolationMethod,
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
EVAL_DATA = Path(__file__).parent.parent.parent.parent / "data" / "training" / "train.jsonl"

# LoRA configurations to test - varying aggressiveness
# Rank sweep: how much capacity for change
RANK_SWEEP = [
    {"name": "rank_2", "rank": 2, "steps": 50},
    {"name": "rank_4", "rank": 4, "steps": 50},
    {"name": "rank_8", "rank": 8, "steps": 50},
    {"name": "rank_16", "rank": 16, "steps": 50},
    {"name": "rank_32", "rank": 32, "steps": 50},
]

# Step sweep: how much training
STEP_SWEEP = [
    {"name": "steps_10", "rank": 8, "steps": 10},
    {"name": "steps_50", "rank": 8, "steps": 50},
    {"name": "steps_100", "rank": 8, "steps": 100},
    {"name": "steps_200", "rank": 8, "steps": 200},
]

# All configs
ALL_CONFIGS = RANK_SWEEP + STEP_SWEEP

# Prompts for CKA computation
CKA_PROMPTS = [
    "The capital of France is",
    "In mathematics, 2 + 2 equals",
    "The largest planet in our solar system is",
    "Water freezes at",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "Shakespeare wrote the play",
    "The first president of the United States was",
    "Photosynthesis converts sunlight into",
    "The human heart has how many chambers",
]


def compute_cka_barrier(
    model_path: Path,
    lora_path: Path | None,
    prompts: list[str],
    layer_idx: int,
    backend,
) -> dict:
    """Compute CKA-based barrier between base and base+LoRA.

    Returns barrier analysis result with:
    - barrier_height: max CKA divergence along interpolation
    - normalized_barrier: barrier relative to endpoints
    - barrier_location: where barrier peaks (0-1)
    """
    # Collect base activations
    logger.info("  Collecting base activations...")
    base_acts = collect_layer_activations(
        model_path=model_path,
        prompts=prompts,
        layer_idx=layer_idx,
        lora_path=None,
    )
    mx.eval(base_acts)

    if lora_path is None:
        # No LoRA = no barrier
        return {
            "barrier_height": 0.0,
            "normalized_barrier": 0.0,
            "barrier_location": 0.5,
            "source_loss": 0.0,
            "target_loss": 0.0,
            "cka_at_target": 1.0,
        }

    # Collect LoRA activations
    logger.info("  Collecting LoRA activations...")
    lora_acts = collect_layer_activations(
        model_path=model_path,
        prompts=prompts,
        layer_idx=layer_idx,
        lora_path=lora_path,
    )
    mx.eval(lora_acts)

    # Compute CKA at endpoints
    cka_at_target = compute_linear_cka_from_activations(base_acts, lora_acts, backend)

    # For mode connectivity, we treat activations as "weights" and interpolate
    # CKA loss = 1 - CKA(base, interpolated)

    # Center activations for CKA
    base_centered = base_acts - backend.mean(base_acts, axis=0, keepdims=True)
    lora_centered = lora_acts - backend.mean(lora_acts, axis=0, keepdims=True)
    backend.eval(base_centered, lora_centered)

    def cka_loss_fn(interpolated_acts):
        """CKA-based loss at interpolated activations."""
        interp_centered = interpolated_acts - backend.mean(interpolated_acts, axis=0, keepdims=True)
        backend.eval(interp_centered)
        cka = compute_linear_cka_from_activations(base_centered, interp_centered, backend)
        return 1.0 - cka

    # Analyze mode connectivity
    result = analyze_mode_connectivity(
        base_centered,  # "source weights"
        lora_centered,  # "target weights"
        cka_loss_fn,
        n_steps=11,
        method=InterpolationMethod.LINEAR,
        backend=backend,
    )

    return {
        "barrier_height": result.barrier_height,
        "normalized_barrier": result.normalized_barrier,
        "barrier_location": result.barrier_location,
        "source_loss": result.source_loss,
        "target_loss": result.target_loss,
        "cka_at_target": cka_at_target,
    }


def run_lora_experiment(
    config: dict,
    model_path: Path,
    train_data: Path,
    eval_data: Path,
    output_dir: Path,
    base_perplexity: float,
    layer_idx: int,
    backend,
) -> dict:
    """Train a single LoRA config, compute barrier, and measure impact."""
    name = config["name"]
    rank = config["rank"]
    steps = config["steps"]

    logger.info(f"Training LoRA config: {name} (rank={rank}, steps={steps})")

    lora_output = output_dir / "loras" / name

    try:
        # Train LoRA
        train_result = train_lora_quick(
            model_path=model_path,
            dataset_path=train_data,
            output_path=lora_output,
            target_modules=["q_proj", "v_proj"],  # Standard targeting
            rank=rank,
            steps=steps,
            lr=1e-4,
            batch_size=2,
            sequence_length=256,
            max_samples=100,
        )

        logger.info(f"  Training complete: final_loss={train_result.final_loss:.4f}")

        # Compute barrier
        logger.info("  Computing mode connectivity barrier...")
        barrier_result = compute_cka_barrier(
            model_path=model_path,
            lora_path=lora_output,
            prompts=CKA_PROMPTS,
            layer_idx=layer_idx,
            backend=backend,
        )
        logger.info(f"  Barrier: height={barrier_result['barrier_height']:.4f}")

        # Evaluate perplexity with LoRA
        logger.info("  Evaluating perplexity...")
        eval_result = evaluate_perplexity(
            model_path=model_path,
            dataset_path=eval_data,
            lora_path=lora_output,
            max_samples=50,
        )
        logger.info(f"  Perplexity: {eval_result.perplexity:.2f}")

        perplexity_delta = eval_result.perplexity - base_perplexity

        return {
            "name": name,
            "rank": rank,
            "steps": steps,
            "training": {
                "final_loss": train_result.final_loss,
                "steps_completed": train_result.steps_completed,
                "duration_seconds": train_result.duration_seconds,
            },
            "barrier": barrier_result,
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
            "rank": rank,
            "steps": steps,
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
    """Run Experiment 16: Mode Connectivity LoRA Barrier Validation."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp16_mode_connectivity_lora")
    initialize_default_backend()
    backend = get_default_backend()

    # Check model exists
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        logger.info("Please ensure the external volume is mounted")
        return None

    config = ExperimentConfig(
        experiment_name="exp16_mode_connectivity_lora",
        source_model_path=str(MODEL_PATH),
        target_model_path=str(MODEL_PATH),
        backend_name=type(backend).__name__,
        hyperparameters={
            "hypothesis": "Mode connectivity barrier correlates with LoRA degradation",
            "test_type": "barrier_perplexity_correlation",
            "target_modules": ["q_proj", "v_proj"],
            "rank_sweep": [c["rank"] for c in RANK_SWEEP],
            "step_sweep": [c["steps"] for c in STEP_SWEEP],
        },
    )

    results = {
        "base_perplexity": 0.0,
        "control_barrier": {},
        "lora_results": [],
        "summary": {},
    }

    # ========== PART 1: Baseline Evaluation ==========
    logger.info("=" * 70)
    logger.info("PART 1: Baseline Perplexity (no LoRA)")
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

    # ========== PART 2: Control - Barrier for Untrained ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 2: Control - Barrier for Untrained Model")
    logger.info("=" * 70)

    # Get model info for layer selection
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    from modelcypher.ports.model_architecture_factory import get_model_architecture
    model_config = {}
    if hasattr(model, 'config'):
        if hasattr(model.config, 'to_dict'):
            model_config = model.config.to_dict()
        elif isinstance(model.config, dict):
            model_config = model.config
    arch = get_model_architecture(model, config=model_config)
    num_layers = arch.num_layers
    layer_idx = num_layers // 2
    del model
    gc.collect()

    logger.info(f"Using layer {layer_idx} of {num_layers} for barrier computation")

    try:
        control_barrier = compute_cka_barrier(
            model_path=MODEL_PATH,
            lora_path=None,  # No LoRA = control
            prompts=CKA_PROMPTS,
            layer_idx=layer_idx,
            backend=backend,
        )
        results["control_barrier"] = control_barrier
        logger.info(f"Control barrier (no LoRA): {control_barrier['barrier_height']:.6f}")

    except Exception as e:
        logger.error(f"Control barrier computation failed: {e}")
        import traceback
        traceback.print_exc()

    # ========== PART 3: Train and Evaluate LoRAs ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 3: Training and Evaluating LoRA Configurations")
    logger.info("=" * 70)

    for cfg in ALL_CONFIGS:
        logger.info("")
        result = run_lora_experiment(
            config=cfg,
            model_path=MODEL_PATH,
            train_data=TRAIN_DATA,
            eval_data=EVAL_DATA,
            output_dir=output_dir,
            base_perplexity=results["base_perplexity"],
            layer_idx=layer_idx,
            backend=backend,
        )
        results["lora_results"].append(result)

    # ========== PART 4: Analysis ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 4: Summary Analysis")
    logger.info("=" * 70)

    successful_results = [r for r in results["lora_results"] if r.get("success", False)]

    if successful_results:
        barriers = [r["barrier"]["barrier_height"] for r in successful_results]
        perplexity_deltas = [r["evaluation"]["perplexity_delta"] for r in successful_results]
        ranks = [r["rank"] for r in successful_results]
        steps = [r["steps"] for r in successful_results]

        # Correlations
        barrier_ppl_correlation = compute_correlation(barriers, perplexity_deltas)
        barrier_rank_correlation = compute_correlation(
            [r["rank"] for r in successful_results if r["steps"] == 50],  # Rank sweep only
            [r["barrier"]["barrier_height"] for r in successful_results if r["steps"] == 50],
        )
        barrier_steps_correlation = compute_correlation(
            [r["steps"] for r in successful_results if r["rank"] == 8],  # Step sweep only
            [r["barrier"]["barrier_height"] for r in successful_results if r["rank"] == 8],
        )

        # Control validation
        control_near_zero = results.get("control_barrier", {}).get("barrier_height", 1.0) < 0.01

        results["summary"] = {
            "n_configs_tested": len(ALL_CONFIGS),
            "n_successful": len(successful_results),
            "base_perplexity": results["base_perplexity"],
            "control_barrier_near_zero": control_near_zero,
            "barrier_perplexity_correlation": barrier_ppl_correlation,
            "barrier_rank_correlation": barrier_rank_correlation,
            "barrier_steps_correlation": barrier_steps_correlation,
            "mean_barrier": sum(barriers) / len(barriers) if barriers else 0,
            "mean_perplexity_delta": sum(perplexity_deltas) / len(perplexity_deltas) if perplexity_deltas else 0,
            "success": len(successful_results) >= 4,
        }

        logger.info(f"Barrier-Perplexity correlation: {barrier_ppl_correlation:.3f}")
        logger.info(f"Barrier-Rank correlation: {barrier_rank_correlation:.3f}")
        logger.info(f"Barrier-Steps correlation: {barrier_steps_correlation:.3f}")
        logger.info(f"Control barrier near zero: {control_near_zero}")
        logger.info(f"Mean barrier: {results['summary']['mean_barrier']:.4f}")
        logger.info(f"Mean perplexity delta: {results['summary']['mean_perplexity_delta']:.2f}")

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
    logger.info("EXPERIMENT 16 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Results saved to: {output_dir / 'results.json'}")
    logger.info("=" * 70)

    return experiment_result


if __name__ == "__main__":
    main()
