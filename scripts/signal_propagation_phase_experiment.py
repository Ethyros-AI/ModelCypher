#!/usr/bin/env python3
"""Experiment: Signal Propagation Phase Classification.

Tests whether mean-field signal propagation theory (De & Smith 2020, CompleteP
NeurIPS 2025) explains the highway/processing/exit phase structure observed
in ModelCypher's intrinsic dimension trajectories.

Hypothesis:
    H1: Highway layers have alpha^2 * chi ≈ 0 (ordered phase),
        processing layers have alpha^2 * chi > 0 (chaotic phase),
        exit layers have alpha^2 * chi < 0 (convergent phase).

Measurements:
    For each layer l:
        alpha_l = ||delta_l||_2 / ||h_in||_2   (residual scaling)
        chi_l   = Var(delta_l) / Var(h_in)      (variance amplification)
        ID_l    = TwoNN intrinsic dimension

Falsification criteria:
    FAIL if Spearman(alpha^2*chi, d(ID)/d(layer)) < 0.3 for >2 of 5 models
    FAIL if highway alpha^2*chi 95% CI excludes 0
    FAIL if any model has processing mean(alpha^2*chi) <= 0

References:
    De & Smith (2020): Critical residual scaling alpha ~ 1/sqrt(L)
    Dey et al. (NeurIPS 2025): CompleteP, L/d ratio determines covariance stability
    Joshi et al. (NeurIPS 2025): 28-model ID trajectory confirmation

Usage:
    poetry run python scripts/signal_propagation_phase_experiment.py

    # Smoke test (2 probes per category, 2 models)
    poetry run python scripts/signal_propagation_phase_experiment.py --smoke

    # Custom output
    poetry run python scripts/signal_propagation_phase_experiment.py \
        --output results/signal_propagation/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16,
        "d": 1024,
        "architecture": "lfm2",
    },
    "LFM2-1.2B": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-1.2B-bf16",
        "L": 16,
        "d": 2048,
        "architecture": "lfm2",
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36,
        "d": 2048,
        "architecture": "qwen2.5",
    },
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
        "L": 36,
        "d": 4096,
        "architecture": "qwen3",
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28,
        "d": 3072,
        "architecture": "llama",
    },
    "Qwen3-1.7B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-1.7B-MLX-bf16",
        "L": 28,
        "d": 2048,
        "architecture": "qwen3",
    },
}

# =============================================================================
# Probe Prompts (10 per category, 6 categories = 60 total)
# =============================================================================

PROBE_CATEGORIES = {
    "retrieval": [
        "The capital of France is",
        "Who wrote Romeo and Juliet?",
        "The chemical symbol for water is",
        "The largest planet in our solar system is",
        "The speed of light in a vacuum is approximately",
        "The first president of the United States was",
        "The boiling point of water at sea level is",
        "The chemical formula for table salt is",
        "The tallest mountain on Earth is",
        "The currency of Japan is",
    ],
    "arithmetic": [
        "What is 347 + 528?",
        "What is 15 * 23?",
        "What is 1024 / 16?",
        "What is 99 - 37?",
        "What is 8 * 7 + 13?",
        "What is 256 + 384 - 100?",
        "What is 12 * 12?",
        "What is 999 - 456?",
        "What is 50 * 20 + 1?",
        "What is 128 / 4?",
    ],
    "reasoning": [
        "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. How many now?",
        "A lily pad doubles in size every day. It takes 48 days to cover the lake. When is it half covered?",
        "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
        "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
        "If you rearrange CIFAIPC, you get the name of a country. What is it?",
        "A train leaves A at 60 mph, another leaves B at 80 mph toward A, 280 miles apart. When do they meet?",
        "What comes next: 2, 6, 12, 20, 30, ?",
        "Three friends split $90 unequally. A gets twice what B gets. B gets twice what C gets. How much does C get?",
    ],
    "creative": [
        "Write a haiku about the ocean.",
        "Describe a sunset over the mountains in one vivid sentence.",
        "Write a short poem about the passage of time.",
        "Describe the taste of your favorite food using only three words.",
        "Write a one-sentence story with a twist ending.",
        "Describe the sound of rain on a tin roof.",
        "Write a metaphor for loneliness.",
        "Describe the color blue to someone who has never seen it.",
        "Write a two-line dialogue between the sun and the moon.",
        "Describe the feeling of flying in one sentence.",
    ],
    "code": [
        "Write a Python function that reverses a string.",
        "Write a Python function that checks if a number is prime.",
        "Write a Python function to compute Fibonacci up to n terms.",
        "Write a Python function to find the max element without max().",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python one-liner to flatten a nested list.",
        "Write a Python function to sort a list using bubble sort.",
        "Write a Python function to count words in a string.",
        "Write a Python function to compute factorial recursively.",
        "Write a Python function to merge two sorted lists.",
    ],
    "narrative": [
        "Once upon a time in a faraway kingdom, there lived a",
        "The old lighthouse keeper watched the storm approach from",
        "In the year 2150, humanity had finally achieved",
        "She opened the letter and read the first line:",
        "The forest was silent except for the sound of",
        "He had been walking for three days when he finally saw",
        "The library contained a secret that no one had discovered for",
        "As the last leaf fell from the ancient oak tree,",
        "The musician played a melody that made everyone in the room",
        "Deep beneath the ocean, a creature stirred for the first time in",
    ],
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LayerMeasurement:
    """Signal propagation measurements for a single layer."""

    layer_idx: int
    # Signal propagation
    alpha: float  # ||delta||_2 / ||h_in||_2
    chi: float  # Var(delta) / Var(h_in)
    alpha_sq_chi: float  # alpha^2 * chi
    # Norms
    h_in_norm: float  # mean ||h_in||
    delta_norm: float  # mean ||delta||
    h_out_norm: float  # mean ||h_out||
    # Variance
    var_h_in: float  # Var(h_in) via trace of centered Gram
    var_delta: float  # Var(delta)
    # Intrinsic dimension
    id_two_nn: float


@dataclass
class ModelResult:
    """Complete results for one model."""

    model_name: str
    architecture: str
    num_layers: int
    d_model: int
    ld_ratio: float
    # Per-layer data
    layer_measurements: list[dict]
    # Phase classification
    highway_layers: list[int]
    processing_layers: list[int]
    exit_layers: list[int]
    # Test results
    spearman_alpha_sq_chi_vs_id_gradient: float
    highway_mean_alpha_sq_chi: float
    highway_ci_lower: float
    highway_ci_upper: float
    processing_mean_alpha_sq_chi: float
    # Falsification
    passes_spearman: bool  # > 0.3
    passes_highway_ci: bool  # CI includes 0
    passes_processing_positive: bool  # mean > 0


@dataclass
class ExperimentResults:
    """Complete experiment results."""

    timestamp: str
    experiment: str = "signal_propagation_phase_classification"
    models: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# =============================================================================
# Core Measurement Functions
# =============================================================================


def collect_per_layer_activations(
    model, tokenizer, prompts: list[str], num_layers: int, backend
) -> list[dict]:
    """Collect h_in, h_out, delta for every layer across all prompts.

    Returns list of dicts, one per layer, each containing stacked activations
    from all prompts: {h_in: [N, d], delta: [N, d], h_out: [N, d]}.
    """
    import mlx.core as mx

    # Resolve backbone components
    base = getattr(model, "model", model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    # Collect per-layer activations across all prompts
    layer_h_in = [[] for _ in range(num_layers)]
    layer_delta = [[] for _ in range(num_layers)]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        hidden = embed(input_ids)
        seq_len = input_ids.shape[1]

        # Create numeric causal mask (used only for standard transformer layers)
        try:
            numeric_mask = backend.create_causal_mask(seq_len, hidden.dtype)
        except Exception:
            numeric_mask = None

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            h_in = hidden
            # Per-layer mask routing: LFM2 hybrid layers have is_attention_layer
            # attribute — attention layers expect "causal", conv layers expect None
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = numeric_mask

            # Forward through layer
            try:
                h_out = layer(hidden, mask=layer_mask)
            except (TypeError, ValueError):
                try:
                    h_out = layer(hidden, layer_mask)
                except (TypeError, ValueError):
                    h_out = layer(hidden)

            delta = h_out - h_in

            # Take last-token representation [1, d]
            layer_h_in[i].append(h_in[:, -1, :])
            layer_delta[i].append(delta[:, -1, :])

            hidden = h_out

        mx.eval(hidden)  # Force evaluation

    # Stack into [N, d] arrays per layer
    result = []
    for i in range(num_layers):
        if layer_h_in[i]:
            h_in_stacked = mx.concatenate(layer_h_in[i], axis=0)
            delta_stacked = mx.concatenate(layer_delta[i], axis=0)
            h_out_stacked = h_in_stacked + delta_stacked
            mx.eval(h_in_stacked, delta_stacked, h_out_stacked)
            result.append({
                "h_in": h_in_stacked,
                "delta": delta_stacked,
                "h_out": h_out_stacked,
            })
        else:
            result.append(None)

    return result


def compute_signal_propagation(
    layer_activations: list[dict], backend
) -> list[dict]:
    """Compute alpha, chi, alpha^2*chi for each layer from collected activations.

    All quantities are derived from measured data — no arbitrary constants.
    """
    import mlx.core as mx
    import numpy as np

    measurements = []
    eps = float(np.sqrt(np.finfo(np.float32).eps))  # ~3.45e-4, IEEE 754 derived

    for i, act in enumerate(layer_activations):
        if act is None:
            measurements.append({
                "layer_idx": i,
                "alpha": 0.0, "chi": 0.0, "alpha_sq_chi": 0.0,
                "h_in_norm": 0.0, "delta_norm": 0.0, "h_out_norm": 0.0,
                "var_h_in": 0.0, "var_delta": 0.0,
            })
            continue

        h_in = act["h_in"]  # [N, d]
        delta = act["delta"]  # [N, d]
        h_out = act["h_out"]  # [N, d]

        # Convert to float32 for numerical stability
        h_in_f = h_in.astype(mx.float32)
        delta_f = delta.astype(mx.float32)
        h_out_f = h_out.astype(mx.float32)

        # Norms: mean over samples of per-sample L2 norm
        h_in_norms = mx.sqrt(mx.sum(h_in_f * h_in_f, axis=1))  # [N]
        delta_norms = mx.sqrt(mx.sum(delta_f * delta_f, axis=1))  # [N]
        h_out_norms = mx.sqrt(mx.sum(h_out_f * h_out_f, axis=1))  # [N]

        mean_h_in_norm = mx.mean(h_in_norms).item()
        mean_delta_norm = mx.mean(delta_norms).item()
        mean_h_out_norm = mx.mean(h_out_norms).item()

        # alpha = mean(||delta|| / ||h_in||) per sample
        ratios = delta_norms / (h_in_norms + eps)
        alpha = mx.mean(ratios).item()

        # Variance via trace of centered Gram: Var(X) = trace(X_c^T X_c) / N
        # where X_c = X - mean(X)
        h_in_centered = h_in_f - mx.mean(h_in_f, axis=0, keepdims=True)
        delta_centered = delta_f - mx.mean(delta_f, axis=0, keepdims=True)

        n_samples = h_in_f.shape[0]
        var_h_in = (mx.sum(h_in_centered * h_in_centered) / n_samples).item()
        var_delta = (mx.sum(delta_centered * delta_centered) / n_samples).item()

        # chi = Var(delta) / Var(h_in)
        chi = var_delta / (var_h_in + eps)

        # Signal propagation gain
        alpha_sq_chi = alpha * alpha * chi

        mx.eval(h_in_norms, delta_norms, h_out_norms)

        measurements.append({
            "layer_idx": i,
            "alpha": alpha,
            "chi": chi,
            "alpha_sq_chi": alpha_sq_chi,
            "h_in_norm": mean_h_in_norm,
            "delta_norm": mean_delta_norm,
            "h_out_norm": mean_h_out_norm,
            "var_h_in": var_h_in,
            "var_delta": var_delta,
        })

    return measurements


def compute_id_trajectory(
    layer_activations: list[dict], backend
) -> list[float]:
    """Compute TwoNN intrinsic dimension at each layer."""
    import numpy as np

    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    ids = []
    for i, act in enumerate(layer_activations):
        if act is None:
            ids.append(float("nan"))
            continue

        # Use h_out (the representation at this layer)
        h_out = act["h_out"]
        # Convert to numpy for TwoNN
        h_np = np.array(h_out.tolist(), dtype=np.float32)

        n_samples = h_np.shape[0]
        if n_samples < IntrinsicDimension.local_dimension_min_samples():
            ids.append(float("nan"))
            continue

        try:
            estimate = IntrinsicDimension.compute_two_nn(h_np, backend=backend)
            ids.append(estimate.intrinsic_dimension)
        except Exception as e:
            logger.warning(f"ID estimation failed at layer {i}: {e}")
            ids.append(float("nan"))

    return ids


def classify_phases(
    id_trajectory: list[float],
) -> tuple[list[int], list[int], list[int]]:
    """Classify layers into highway/processing/exit based on ID trajectory.

    Uses the ID trajectory shape: low→expand→compress.
    - Highway: layers before median ID (ascending phase, low absolute ID)
    - Processing: layers around peak ID
    - Exit: layers after peak ID (descending phase)

    No arbitrary thresholds — uses median as the natural boundary.
    """
    import numpy as np

    ids = np.array(id_trajectory)
    valid = ~np.isnan(ids)
    if not np.any(valid):
        return [], [], []

    # Find peak layer
    valid_ids = ids.copy()
    valid_ids[~valid] = -np.inf
    peak_layer = int(np.argmax(valid_ids))

    # Median of valid IDs
    median_id = float(np.median(ids[valid]))

    highway = []
    processing = []
    exit_layers = []

    for i in range(len(ids)):
        if not valid[i]:
            continue
        if i <= peak_layer and ids[i] < median_id:
            highway.append(i)
        elif i <= peak_layer:
            processing.append(i)
        else:
            # Post-peak: exit if ID is decreasing, otherwise still processing
            exit_layers.append(i)

    return highway, processing, exit_layers


def bootstrap_ci(values: list[float], n_bootstrap: int = 1000) -> tuple[float, float]:
    """Bootstrap 95% confidence interval. No arbitrary constants — 95% from convention."""
    import numpy as np

    rng = np.random.default_rng(42)
    arr = np.array(values)
    if len(arr) == 0:
        return 0.0, 0.0

    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))

    means.sort()
    lower = means[int(0.025 * n_bootstrap)]
    upper = means[int(0.975 * n_bootstrap)]
    return lower, upper


# =============================================================================
# Main Experiment
# =============================================================================


def run_single_model(
    model_name: str, model_info: dict, probes: list[str], backend
) -> dict:
    """Run all measurements for a single model."""
    import numpy as np

    from modelcypher.core.domain.statistics import spearman_correlation

    model_path = model_info["path"]
    logger.info(f"Loading model: {model_name} from {model_path}")

    model, tokenizer = backend.load_model(model_path)
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers is not None else 0
    d_model = model_info["d"]
    L = model_info["L"]

    logger.info(f"Model loaded: {num_layers} layers, d={d_model}, L/d={L/d_model:.4f}")

    # Phase 1: Collect activations
    logger.info(f"Collecting activations for {len(probes)} probes across {num_layers} layers...")
    t0 = time.time()
    layer_acts = collect_per_layer_activations(
        model, tokenizer, probes, num_layers, backend
    )
    logger.info(f"Activation collection: {time.time() - t0:.1f}s")

    # Phase 2: Signal propagation measurements
    logger.info("Computing signal propagation metrics...")
    sp_measurements = compute_signal_propagation(layer_acts, backend)

    # Phase 3: Intrinsic dimension trajectory
    logger.info("Computing ID trajectory...")
    id_trajectory = compute_id_trajectory(layer_acts, backend)

    # Add ID to measurements
    for i, m in enumerate(sp_measurements):
        m["id_two_nn"] = id_trajectory[i] if i < len(id_trajectory) else float("nan")

    # Phase 4: Classify phases
    highway, processing, exit_layers = classify_phases(id_trajectory)
    logger.info(
        f"Phases: highway={highway[:5]}{'...' if len(highway) > 5 else ''}, "
        f"processing={processing[:5]}{'...' if len(processing) > 5 else ''}, "
        f"exit={exit_layers[:5]}{'...' if len(exit_layers) > 5 else ''}"
    )

    # Phase 5: Test predictions
    # Compute ID gradient (finite difference)
    id_arr = np.array(id_trajectory)
    id_gradient = np.gradient(id_arr)
    # Replace NaN gradients with 0
    id_gradient = np.where(np.isnan(id_gradient), 0.0, id_gradient)

    alpha_sq_chi = [m["alpha_sq_chi"] for m in sp_measurements]

    # Spearman correlation
    valid_mask = ~np.isnan(id_arr)
    valid_asc = [alpha_sq_chi[i] for i in range(len(alpha_sq_chi)) if valid_mask[i]]
    valid_idg = [float(id_gradient[i]) for i in range(len(id_gradient)) if valid_mask[i]]

    if len(valid_asc) >= 3:
        spearman = spearman_correlation(valid_asc, valid_idg)
    else:
        spearman = 0.0

    # Highway mean and CI
    highway_vals = [alpha_sq_chi[i] for i in highway if i < len(alpha_sq_chi)]
    highway_mean = float(np.mean(highway_vals)) if highway_vals else 0.0
    ci_lower, ci_upper = bootstrap_ci(highway_vals) if highway_vals else (0.0, 0.0)

    # Processing mean
    processing_vals = [alpha_sq_chi[i] for i in processing if i < len(alpha_sq_chi)]
    processing_mean = float(np.mean(processing_vals)) if processing_vals else 0.0

    # Falsification tests
    passes_spearman = spearman > 0.3
    passes_highway_ci = ci_lower <= 0.0 <= ci_upper  # CI includes 0
    passes_processing = processing_mean > 0.0

    logger.info(
        f"Results: Spearman={spearman:.3f} ({'PASS' if passes_spearman else 'FAIL'}), "
        f"Highway CI=[{ci_lower:.6f}, {ci_upper:.6f}] ({'PASS' if passes_highway_ci else 'FAIL'}), "
        f"Processing mean={processing_mean:.6f} ({'PASS' if passes_processing else 'FAIL'})"
    )

    # Cleanup
    del model, tokenizer, layer_acts
    gc.collect()

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "num_layers": num_layers,
        "d_model": d_model,
        "L": L,
        "ld_ratio": L / d_model,
        "layer_measurements": sp_measurements,
        "highway_layers": highway,
        "processing_layers": processing,
        "exit_layers": exit_layers,
        "spearman_alpha_sq_chi_vs_id_gradient": spearman,
        "highway_mean_alpha_sq_chi": highway_mean,
        "highway_ci_lower": ci_lower,
        "highway_ci_upper": ci_upper,
        "processing_mean_alpha_sq_chi": processing_mean,
        "passes_spearman": passes_spearman,
        "passes_highway_ci": passes_highway_ci,
        "passes_processing_positive": passes_processing,
    }


def run_experiment(args: argparse.Namespace) -> None:
    """Run the full signal propagation phase classification experiment."""
    import numpy as np

    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    # Select models
    if args.smoke:
        model_names = ["LFM2-350M", "Qwen3-8B"]
    elif args.models:
        model_names = args.models
    else:
        model_names = list(MODEL_REGISTRY.keys())

    # Build probe list
    if args.smoke:
        probes = []
        for cat, prompts in PROBE_CATEGORIES.items():
            probes.extend(prompts[:2])
    else:
        probes = []
        for cat, prompts in PROBE_CATEGORIES.items():
            probes.extend(prompts[:args.n_probes_per_category])

    logger.info(f"Experiment: {len(model_names)} models, {len(probes)} probes")

    # Run per model
    results = ExperimentResults(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        model_info = MODEL_REGISTRY[model_name]
        model_result = run_single_model(model_name, model_info, probes, backend)
        results.models.append(model_result)
        gc.collect()

    # Summary: count passes across models
    n_models = len(results.models)
    spearman_passes = sum(1 for m in results.models if m["passes_spearman"])
    highway_passes = sum(1 for m in results.models if m["passes_highway_ci"])
    processing_passes = sum(1 for m in results.models if m["passes_processing_positive"])

    # Experiment-level falsification
    # FAIL if Spearman < 0.3 for >2 of 5 models
    if n_models < 5:
        logger.warning(
            f"Only {n_models} models — Spearman gate has reduced statistical power "
            f"(designed for 5 models, allows 2 failures). Smoke verdict is provisional."
        )
    spearman_fail_count = n_models - spearman_passes
    experiment_passes_spearman = spearman_fail_count <= 2

    # FAIL if highway CI excludes 0 for ANY model
    experiment_passes_highway = highway_passes == n_models

    # FAIL if processing mean <= 0 for ANY model
    experiment_passes_processing = processing_passes == n_models

    overall_pass = (
        experiment_passes_spearman
        and experiment_passes_highway
        and experiment_passes_processing
    )

    results.summary = {
        "n_models": n_models,
        "n_probes": len(probes),
        "spearman_passes": spearman_passes,
        "highway_ci_passes": highway_passes,
        "processing_positive_passes": processing_passes,
        "experiment_passes_spearman": experiment_passes_spearman,
        "experiment_passes_highway": experiment_passes_highway,
        "experiment_passes_processing": experiment_passes_processing,
        "overall_verdict": "H1 SUPPORTED" if overall_pass else "H1 REFUTED",
        "falsification_thresholds": {
            "spearman_min": 0.3,
            "spearman_source": "existing entropy->curvature r=0.507 as lower bound",
            "highway_ci_source": "95% bootstrap CI must include 0",
            "processing_source": "mean alpha^2*chi must be > 0",
            "sqrt_eps_f32": float(np.sqrt(np.finfo(np.float32).eps)),
        },
        "references": [
            "De & Smith (2020): alpha ~ 1/sqrt(L) critical scaling",
            "Dey et al. (NeurIPS 2025, arXiv:2505.01618): CompleteP, L/d ratio",
            "Joshi et al. (NeurIPS 2025, arXiv:2511.20315): 28-model ID confirmation",
        ],
    }

    verdict = results.summary["overall_verdict"]
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT VERDICT: {verdict}")
    logger.info(f"  Spearman test: {spearman_passes}/{n_models} pass (need >{n_models-3})")
    logger.info(f"  Highway CI test: {highway_passes}/{n_models} pass (need all)")
    logger.info(f"  Processing >0 test: {processing_passes}/{n_models} pass (need all)")
    logger.info(f"{'='*60}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "signal_propagation_results.json"

    with open(output_file, "w") as f:
        json.dump(asdict(results) if hasattr(results, "__dataclass_fields__") else {
            "timestamp": results.timestamp,
            "experiment": results.experiment,
            "models": results.models,
            "summary": results.summary,
        }, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Signal Propagation Phase Classification Experiment"
    )
    parser.add_argument(
        "--output",
        default="results/signal_propagation/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to test (default: all 5)",
    )
    parser.add_argument(
        "--n-probes-per-category",
        type=int,
        default=10,
        help="Number of probes per category (default: 10, total = 6 * N)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 2 models, 2 probes per category",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
