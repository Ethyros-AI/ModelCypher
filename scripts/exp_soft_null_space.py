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

"""Soft Null-Space Merge Experiment.

Tests whether partial null-space projection can transfer capability
while preserving target behavior.

Hypothesis: Current null-space merge is too conservative (0% transfer, 100% preserve).
By allowing controlled partial geometry change via blend_alpha, we can transfer
capability while accepting minimal degradation.

The key insight is that standard null-space projection:
    N = I - V_r @ V_r.T
    delta_proj = delta_W @ N  # Only null-space component

Can be softened to:
    N_soft = I - alpha * (V_r @ V_r.T)
    delta_proj = delta_W @ N_soft  # Partially preserve used directions

Where alpha=1.0 gives standard null-space (current behavior) and alpha=0.0
gives full delta (no projection).

Usage:
    python scripts/exp_soft_null_space.py --alpha 0.5
    python scripts/exp_soft_null_space.py --sweep  # Run all alphas
    python scripts/exp_soft_null_space.py --analyze  # Analyze existing results
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default model paths (on CodeCypher volume)
# For same-architecture testing, use LFM2 variants or Qwen variants
DEFAULT_SOURCE = "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16"  # Larger LFM2 as "expert"
DEFAULT_TARGET = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"  # Smaller LFM2 as target
DEFAULT_OUTPUT_BASE = "/Volumes/CodeCypher/models/merged/soft_null_space"
RESULTS_DIR = Path("data/experiments/soft_null_space")

# Sweep alphas: from full null-space (1.0) to no projection (0.0)
SWEEP_ALPHAS = [1.0, 0.7, 0.5, 0.3, 0.0]


@dataclass
class BenchmarkResult:
    """Result of benchmarking a model."""

    model_path: str
    model_name: str
    code_score: float
    reasoning_score: float
    n_code_samples: int
    n_reasoning_samples: int
    errors: list[str]


@dataclass
class GeometryResult:
    """Geometric fingerprint metrics."""

    comp_phi_mean: float
    comp_phi_variance: float
    comp_phi_std: float
    per_task: dict[str, float]


@dataclass
class ExperimentCondition:
    """Result for a single alpha condition."""

    alpha: float
    merged_model_path: str
    benchmark: BenchmarkResult | None
    geometry: GeometryResult | None
    merge_metrics: dict[str, Any]
    merge_success: bool
    error: str | None = None


@dataclass
class ExperimentResult:
    """Full experiment result."""

    timestamp: str
    source_model: str
    target_model: str
    source_baseline: BenchmarkResult
    target_baseline: BenchmarkResult
    conditions: list[ExperimentCondition]
    analysis: dict[str, Any]


# Benchmark prompts (from exp_capability_transfer)
CODE_PROMPTS = [
    ("Write a Python function that returns the sum of two numbers:\ndef add(a, b):", "return a + b"),
    ("Write a Python function to check if a number is even:\ndef is_even(n):", "return n % 2 == 0"),
    ("Complete this Python list comprehension to double all numbers:\nnumbers = [1, 2, 3, 4, 5]\ndoubled = [x", "* 2 for x in numbers"),
    ("Write code to print 'Hello, World!':\n", "print"),
    ("Write a function to find the maximum of a list:\ndef find_max(lst):", "return max(lst)"),
    ("Write a function to reverse a string:\ndef reverse(s):", "return s[::-1]"),
    ("Write code to check if a string is a palindrome:\ndef is_palindrome(s):", "return s == s[::-1]"),
    ("Write a function to compute factorial:\ndef factorial(n):", "if n <= 1"),
    ("Write a function to find the length of a list:\ndef list_length(lst):", "return len(lst)"),
    ("Write code to join a list of strings with commas:\ndef join_strings(lst):", "return"),
]

REASONING_PROMPTS = [
    ("What is the capital of France?\nAnswer:", "paris"),
    ("All mammals are warm-blooded. Dogs are mammals. Are dogs warm-blooded? Answer:", "yes"),
    ("What is 7 + 5?\nAnswer:", "12"),
    ("If it rains, the ground gets wet. It rained today. Is the ground wet? Answer:", "yes"),
    ("How many days are in a week?\nAnswer:", "7"),
    ("What planet is closest to the Sun?\nAnswer:", "mercury"),
    ("If A > B and B > C, is A > C? Answer:", "yes"),
    ("What is the opposite of 'hot'?\nAnswer:", "cold"),
    ("How many legs does a dog have?\nAnswer:", "4"),
    ("What comes after Tuesday?\nAnswer:", "wednesday"),
]

# Geometric fingerprint probes
FINGERPRINT_PROBES = {
    "retrieval": "What is the capital of France?",
    "arithmetic": "What is 7 + 5?",
    "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "logic": "If all cats are animals, and all animals need water, do cats need water?",
    "creative": "Write the first line of a story about a dragon.",
    "code": "Write a Python function that returns the sum of two numbers.",
    "cot": "Let me think step by step about how to solve this problem: What is 15% of 80?",
}


def soft_null_space_project(
    delta_W: "mx.array",
    activations: "mx.array",
    alpha: float = 1.0,
) -> tuple["mx.array", dict[str, Any]]:
    """Project delta with controllable null-space strength.

    This is the core mathematical contribution of this experiment.

    Standard null-space projection preserves target behavior exactly by removing
    all delta components that overlap with the used activation directions:
        N = I - V_r @ V_r.T
        delta_proj = delta_W @ N

    Soft null-space allows partial overlap:
        N_soft = I - alpha * (V_r @ V_r.T)
        delta_proj = delta_W @ N_soft

    Args:
        delta_W: Weight delta [out_dim, in_dim]
        activations: Target activations [n_samples, in_dim]
        alpha: Blend factor (0=no projection, 1=full null-space)

    Returns:
        Tuple of (projected_delta, metrics_dict)

    Example:
        alpha=1.0: Standard null-space (current behavior, 0% transfer to used space)
        alpha=0.5: Half in null-space, half preserved (50% leak into used space)
        alpha=0.0: No projection (full delta, 100% leak into used space)
    """
    import mlx.core as mx

    # Compute covariance of activations
    # A.T @ A gives [in_dim, in_dim] covariance matrix
    A = activations.astype(mx.float32)
    AtA = mx.matmul(A.T, A)
    mx.eval(AtA)

    # SVD to find the used directions
    U, S, Vt = mx.linalg.svd(AtA, stream=mx.cpu)
    mx.eval(U, S, Vt)

    # Determine rank threshold (dtype-aware)
    eps = float(mx.finfo(S.dtype).eps)
    max_dim = max(activations.shape)
    s_max = float(S[0].item()) if S.shape[0] > 0 else 0.0
    tol = max_dim * s_max * eps

    # Count significant singular values
    r = int((S > tol).sum().item())
    if r == 0:
        # No significant directions - return original delta
        return delta_W, {
            "rank": 0,
            "null_rank": int(delta_W.shape[1]),
            "alpha": alpha,
            "projection_loss": 0.0,
        }

    # V_r contains the top r right singular vectors (used space basis)
    V_r = Vt[:r].T  # [in_dim, r]
    mx.eval(V_r)

    # Compute projector onto used space: P_used = V_r @ V_r.T
    P_used = mx.matmul(V_r, V_r.T)  # [in_dim, in_dim]
    mx.eval(P_used)

    # Soft null-space: blend between identity and null-space
    # N_soft = I - alpha * P_used
    # When alpha=1: N_soft = I - P_used (standard null-space)
    # When alpha=0: N_soft = I (identity, no projection)
    in_dim = int(delta_W.shape[1])
    I = mx.eye(in_dim, dtype=mx.float32)
    N_soft = I - alpha * P_used
    mx.eval(N_soft)

    # Project delta
    delta_W_float = delta_W.astype(mx.float32)
    delta_proj = mx.matmul(delta_W_float, N_soft)
    mx.eval(delta_proj)

    # Compute metrics
    delta_norm = float(mx.sqrt(mx.sum(delta_W_float * delta_W_float)).item())
    proj_norm = float(mx.sqrt(mx.sum(delta_proj * delta_proj)).item())

    if delta_norm > eps:
        preserved_fraction = proj_norm / delta_norm
        projection_loss = 1.0 - preserved_fraction
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    metrics = {
        "rank": r,
        "null_rank": in_dim - r,
        "alpha": alpha,
        "delta_norm": delta_norm,
        "projected_norm": proj_norm,
        "preserved_fraction": preserved_fraction,
        "projection_loss": projection_loss,
    }

    return delta_proj, metrics


def get_model_hidden_dim(model_base) -> int | None:
    """Get hidden dimension from model, handling different architectures."""
    layer0 = model_base.layers[0]

    # Try different architecture patterns
    # LFM2 style: feed_forward.w1
    if hasattr(layer0, "feed_forward"):
        ff = layer0.feed_forward
        if hasattr(ff, "w1"):
            return int(ff.w1.weight.shape[1])
        if hasattr(ff, "gate_proj"):
            return int(ff.gate_proj.weight.shape[1])

    # Qwen/Llama style: mlp.gate_proj
    if hasattr(layer0, "mlp"):
        mlp = layer0.mlp
        if hasattr(mlp, "gate_proj"):
            return int(mlp.gate_proj.weight.shape[1])

    # Try self_attn as fallback
    if hasattr(layer0, "self_attn"):
        attn = layer0.self_attn
        if hasattr(attn, "q_proj"):
            return int(attn.q_proj.weight.shape[0])

    return None


def get_weight_tensors(layer, weight_name: str):
    """Get weight tensor handling different architectures."""
    # LFM2 style: feed_forward.{w1,w2,w3}
    if hasattr(layer, "feed_forward"):
        ff = layer.feed_forward
        if hasattr(ff, weight_name):
            return getattr(ff, weight_name).weight

    # Qwen/Llama style: mlp.{gate_proj, up_proj, down_proj}
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        # Map w1/w2/w3 to gate_proj/down_proj/up_proj
        mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        mapped_name = mapping.get(weight_name, weight_name)
        if hasattr(mlp, mapped_name):
            return getattr(mlp, mapped_name).weight

    return None


def merge_with_alpha(
    source_path: str,
    target_path: str,
    output_path: str,
    alpha: float,
) -> dict[str, Any]:
    """Perform soft null-space merge with given alpha.

    This experiment tests whether partial null-space projection (controlled by alpha)
    can enable capability transfer while preserving target behavior.

    When source and target have the same architecture, we directly compute weight
    deltas and apply soft null-space projection.

    When architectures differ, we apply a synthetic "capability injection" to test
    the null-space math - we scale the target weights by a small factor to simulate
    knowledge that needs to be injected.

    Returns merge metrics including per-layer preserved fractions.
    """
    import mlx.core as mx
    from mlx_lm import load

    logger.info(f"Loading target model: {Path(target_path).name}")
    target_model, target_tokenizer = load(target_path)

    # Get base module
    target_base = getattr(target_model, "model", target_model)
    target_layers = len(target_base.layers)
    target_hidden = get_model_hidden_dim(target_base)

    logger.info(f"Target: {target_layers} layers, hidden={target_hidden}")

    # Try to load source
    source_model = None
    source_base = None
    source_hidden = None
    try:
        logger.info(f"Loading source model: {Path(source_path).name}")
        source_model, source_tokenizer = load(source_path)
        source_base = getattr(source_model, "model", source_model)
        source_hidden = get_model_hidden_dim(source_base)
        logger.info(f"Source: {len(source_base.layers)} layers, hidden={source_hidden}")
    except Exception as e:
        logger.warning(f"Failed to load source model: {e}")
        logger.info("Will use synthetic delta for null-space testing")

    # Check if architectures match
    same_arch = (
        source_base is not None
        and source_hidden == target_hidden
        and len(source_base.layers) == target_layers
    )

    if same_arch:
        logger.info("Same-architecture merge: computing true weight deltas")
    else:
        logger.info("Cross-architecture or synthetic: will use scaled target as proxy delta")

    # Generate probe activations for null-space computation
    probe_texts = [
        "What is 2 + 2?",
        "The quick brown fox jumps over the lazy dog.",
        "def hello(): return 'world'",
        "Paris is the capital of France.",
        "If A implies B and B implies C, then A implies C.",
        "Write a function to reverse a string.",
        "The mitochondria is the powerhouse of the cell.",
        "What is the capital of Japan?",
        "Explain the concept of recursion.",
        "List three prime numbers.",
    ] * 5  # 50 total probes

    logger.info(f"Collecting activations from {len(probe_texts)} probes...")

    # Collect target activations
    target_activations: dict[int, mx.array] = {}
    for layer_idx in range(target_layers):
        layer_acts = []
        for text in probe_texts:
            tokens = target_tokenizer.encode(text)
            input_ids = mx.array([tokens])
            hidden = target_base.embed_tokens(input_ids)
            mx.eval(hidden)

            for i in range(layer_idx + 1):
                hidden = target_base.layers[i](hidden, mask=None, cache=None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
            mx.eval(hidden)

            act = mx.mean(hidden[0], axis=0)
            mx.eval(act)
            layer_acts.append(act)

        target_activations[layer_idx] = mx.stack(layer_acts, axis=0)
        mx.eval(target_activations[layer_idx])

    logger.info(f"Collected target activations for {len(target_activations)} layers")

    # Perform merge layer by layer
    metrics = {
        "alpha": alpha,
        "n_layers": target_layers,
        "same_arch": same_arch,
        "per_layer": [],
        "preserved_fractions": [],
        "projection_losses": [],
    }

    # Load target weights using MLX's native safetensors loader (handles bfloat16)
    # Handle both single-file and sharded models
    target_dir = Path(target_path)
    target_files = sorted(target_dir.glob("model*.safetensors"))
    if not target_files:
        target_files = sorted(target_dir.glob("*.safetensors"))
    if not target_files:
        raise RuntimeError(f"No safetensors files found in {target_path}")

    # Load all shards
    merged_weights = {}
    for shard_file in target_files:
        shard_weights = dict(mx.load(str(shard_file)))
        merged_weights.update(shard_weights)
        logger.info(f"Loaded {len(shard_weights)} weights from {shard_file.name}")

    logger.info(f"Total weights loaded: {len(merged_weights)}")

    # Process each layer
    n_merged = 0
    for layer_idx in range(target_layers):
        target_layer = target_base.layers[layer_idx]
        layer_activations = target_activations.get(layer_idx)
        if layer_activations is None:
            continue

        # Process feed-forward weights
        # Note: We only process w1 and w3 (gate_proj, up_proj) which have input dim = hidden
        # w2 (down_proj) has input dim = intermediate, which requires different activations
        for weight_name in ["w1", "w3"]:
            target_w = get_weight_tensors(target_layer, weight_name)
            if target_w is None:
                continue

            # Verify dimension compatibility
            # w1/w3 should have shape [intermediate, hidden] where hidden matches activation dim
            act_dim = int(layer_activations.shape[1])
            weight_in_dim = int(target_w.shape[1])
            if weight_in_dim != act_dim:
                logger.warning(
                    f"Layer {layer_idx}.{weight_name}: dimension mismatch "
                    f"(weight_in={weight_in_dim}, activation={act_dim}), skipping"
                )
                continue

            # Determine the delta
            if same_arch:
                source_layer = source_base.layers[layer_idx]
                source_w = get_weight_tensors(source_layer, weight_name)
                if source_w is None or source_w.shape != target_w.shape:
                    continue
                delta_W = source_w.astype(mx.float32) - target_w.astype(mx.float32)
            else:
                # Synthetic delta: scale target weights by 10% as proxy for "new knowledge"
                # This tests whether null-space projection preserves behavior under perturbation
                delta_W = target_w.astype(mx.float32) * 0.1

            mx.eval(delta_W)

            # Apply soft null-space projection
            delta_proj, proj_metrics = soft_null_space_project(
                delta_W=delta_W,
                activations=layer_activations,
                alpha=alpha,
            )

            # Merge: target + projected_delta
            merged_w = target_w.astype(mx.float32) + delta_proj
            mx.eval(merged_w)

            # Determine correct key based on model architecture
            # LFM2: feed_forward.{w1,w2,w3}
            # Qwen/Llama: mlp.{gate_proj,up_proj,down_proj}
            if hasattr(target_layer, "feed_forward"):
                key = f"model.layers.{layer_idx}.feed_forward.{weight_name}.weight"
            elif hasattr(target_layer, "mlp"):
                qwen_mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
                mapped_name = qwen_mapping.get(weight_name, weight_name)
                key = f"model.layers.{layer_idx}.mlp.{mapped_name}.weight"
            else:
                logger.warning(f"Unknown architecture for layer {layer_idx}, skipping")
                continue

            merged_weights[key] = merged_w
            n_merged += 1

            metrics["per_layer"].append({
                "layer": layer_idx,
                "weight": weight_name,
                **proj_metrics,
            })
            metrics["preserved_fractions"].append(proj_metrics.get("preserved_fraction", 1.0))
            metrics["projection_losses"].append(proj_metrics.get("projection_loss", 0.0))

    # Compute aggregate metrics
    if metrics["preserved_fractions"]:
        metrics["mean_preserved_fraction"] = sum(metrics["preserved_fractions"]) / len(metrics["preserved_fractions"])
        metrics["mean_projection_loss"] = sum(metrics["projection_losses"]) / len(metrics["projection_losses"])
    else:
        metrics["mean_preserved_fraction"] = 1.0
        metrics["mean_projection_loss"] = 0.0

    metrics["weights_merged"] = n_merged

    logger.info(
        f"Merged {n_merged} weights with alpha={alpha}, "
        f"mean_preserved={metrics['mean_preserved_fraction']:.1%}"
    )

    # Save merged model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save using MLX's native save (handles bfloat16 properly)
    # Ensure all weights are evaluated
    mx.eval(*merged_weights.values())
    mx.save_safetensors(str(output_dir / "model.safetensors"), merged_weights)

    # Copy config and tokenizer from target
    target_dir = Path(target_path)
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src_file = target_dir / file
        if src_file.exists():
            shutil.copy(src_file, output_dir / file)

    logger.info(f"Saved merged model to {output_path}")

    return metrics


def evaluate_model(model_path: str) -> BenchmarkResult:
    """Evaluate a model on code and reasoning benchmarks."""
    from mlx_lm import load
    import mlx.core as mx

    model_name = Path(model_path).name
    logger.info(f"Evaluating {model_name}...")

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return BenchmarkResult(
            model_path=model_path,
            model_name=model_name,
            code_score=0.0,
            reasoning_score=0.0,
            n_code_samples=0,
            n_reasoning_samples=0,
            errors=[str(e)],
        )

    errors = []

    def generate(prompt: str, max_tokens: int = 30) -> str:
        """Generate text from prompt."""
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            current = input_ids

            for _ in range(max_tokens):
                logits = model(current)
                mx.eval(logits)
                next_token = int(mx.argmax(logits[0, -1, :]).item())

                if next_token == tokenizer.eos_token_id:
                    break

                generated.append(next_token)
                current = mx.concatenate([current, mx.array([[next_token]])], axis=1)

            return tokenizer.decode(generated).strip()
        except Exception as e:
            errors.append(f"Generation error: {e}")
            return ""

    # Evaluate code prompts
    code_correct = 0
    for prompt, expected in CODE_PROMPTS:
        output = generate(prompt, max_tokens=50)
        if expected.lower() in output.lower():
            code_correct += 1
    code_score = code_correct / len(CODE_PROMPTS)

    # Evaluate reasoning prompts
    reasoning_correct = 0
    for prompt, expected in REASONING_PROMPTS:
        output = generate(prompt, max_tokens=20)
        if expected.lower() in output.lower():
            reasoning_correct += 1
    reasoning_score = reasoning_correct / len(REASONING_PROMPTS)

    logger.info(f"  Code: {code_score:.1%} ({code_correct}/{len(CODE_PROMPTS)})")
    logger.info(f"  Reasoning: {reasoning_score:.1%} ({reasoning_correct}/{len(REASONING_PROMPTS)})")

    return BenchmarkResult(
        model_path=model_path,
        model_name=model_name,
        code_score=code_score,
        reasoning_score=reasoning_score,
        n_code_samples=len(CODE_PROMPTS),
        n_reasoning_samples=len(REASONING_PROMPTS),
        errors=errors,
    )


def compute_geometric_fingerprint(model_path: str) -> GeometryResult | None:
    """Compute geometric fingerprint (comp/φ variance) for a model."""
    import mlx.core as mx
    from mlx_lm import load
    import statistics

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model for fingerprint: {e}")
        return None

    base = getattr(model, "model", model)

    def trace_norm_trajectory(prompt: str) -> list[float]:
        """Trace L2 norm through all layers."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        hidden = base.embed_tokens(input_ids)
        mx.eval(hidden)

        norms = [float(mx.sqrt(mx.sum(hidden * hidden)).item())]

        for layer in base.layers:
            hidden = layer(hidden, mask=None, cache=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            mx.eval(hidden)
            norms.append(float(mx.sqrt(mx.sum(hidden * hidden)).item()))

        return norms

    def compute_comp_phi(norms: list[float]) -> float:
        """Compute compression ratio φ from norm trajectory."""
        if len(norms) < 2:
            return 1.0
        peak = max(norms)
        final = norms[-1]
        if final < 1e-10:
            return float("inf")
        return peak / final

    per_task = {}
    for task_type, prompt in FINGERPRINT_PROBES.items():
        norms = trace_norm_trajectory(prompt)
        comp_phi = compute_comp_phi(norms)
        per_task[task_type] = comp_phi

    phi_values = list(per_task.values())
    phi_mean = statistics.mean(phi_values)
    phi_variance = statistics.variance(phi_values) if len(phi_values) > 1 else 0.0
    phi_std = statistics.stdev(phi_values) if len(phi_values) > 1 else 0.0

    return GeometryResult(
        comp_phi_mean=phi_mean,
        comp_phi_variance=phi_variance,
        comp_phi_std=phi_std,
        per_task=per_task,
    )


def run_single_condition(
    source_path: str,
    target_path: str,
    output_base: str,
    alpha: float,
) -> ExperimentCondition:
    """Run a single experimental condition (one alpha value)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"CONDITION: alpha={alpha}")
    logger.info(f"{'='*60}")

    alpha_str = f"{alpha:.1f}".replace(".", "p")
    output_path = f"{output_base}/alpha_{alpha_str}"

    # Merge
    try:
        merge_metrics = merge_with_alpha(
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            alpha=alpha,
        )
        merge_success = True
    except Exception as e:
        logger.error(f"Merge failed for alpha={alpha}: {e}")
        return ExperimentCondition(
            alpha=alpha,
            merged_model_path=output_path,
            benchmark=None,
            geometry=None,
            merge_metrics={},
            merge_success=False,
            error=str(e),
        )

    # Benchmark
    benchmark = evaluate_model(output_path)

    # Geometry
    geometry = compute_geometric_fingerprint(output_path)

    return ExperimentCondition(
        alpha=alpha,
        merged_model_path=output_path,
        benchmark=benchmark,
        geometry=geometry,
        merge_metrics=merge_metrics,
        merge_success=merge_success,
    )


def run_sweep(
    source_path: str,
    target_path: str,
    output_base: str,
    alphas: list[float] | None = None,
) -> ExperimentResult:
    """Run full sweep over all alpha values."""
    if alphas is None:
        alphas = SWEEP_ALPHAS

    logger.info("="*70)
    logger.info("SOFT NULL-SPACE MERGE EXPERIMENT")
    logger.info("="*70)
    logger.info(f"Source (coding): {Path(source_path).name}")
    logger.info(f"Target (general): {Path(target_path).name}")
    logger.info(f"Alphas: {alphas}")
    logger.info("="*70)

    timestamp = datetime.now().isoformat()

    # Baseline source
    logger.info("\n[BASELINE] Evaluating source model...")
    source_baseline = evaluate_model(source_path)

    # Baseline target
    logger.info("\n[BASELINE] Evaluating target model...")
    target_baseline = evaluate_model(target_path)

    # Run conditions
    conditions = []
    for alpha in alphas:
        condition = run_single_condition(
            source_path=source_path,
            target_path=target_path,
            output_base=output_base,
            alpha=alpha,
        )
        conditions.append(condition)

        # Save intermediate result
        save_condition_result(condition, alpha)

    # Analyze results
    analysis = analyze_results(source_baseline, target_baseline, conditions)

    result = ExperimentResult(
        timestamp=timestamp,
        source_model=source_path,
        target_model=target_path,
        source_baseline=source_baseline,
        target_baseline=target_baseline,
        conditions=conditions,
        analysis=analysis,
    )

    # Print summary
    print_summary(result)

    return result


def save_condition_result(condition: ExperimentCondition, alpha: float) -> None:
    """Save result for a single condition."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    alpha_str = f"{alpha:.1f}".replace(".", "p")
    result_file = RESULTS_DIR / f"alpha_{alpha_str}.json"

    result_dict = {
        "alpha": condition.alpha,
        "merged_model_path": condition.merged_model_path,
        "merge_success": condition.merge_success,
        "error": condition.error,
        "merge_metrics": condition.merge_metrics,
    }

    if condition.benchmark:
        result_dict["benchmark"] = asdict(condition.benchmark)
    if condition.geometry:
        result_dict["geometry"] = asdict(condition.geometry)

    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Saved condition result to {result_file}")


def analyze_results(
    source_baseline: BenchmarkResult,
    target_baseline: BenchmarkResult,
    conditions: list[ExperimentCondition],
) -> dict[str, Any]:
    """Analyze results to find optimal alpha."""
    analysis = {
        "source_code_baseline": source_baseline.code_score,
        "source_reasoning_baseline": source_baseline.reasoning_score,
        "target_code_baseline": target_baseline.code_score,
        "target_reasoning_baseline": target_baseline.reasoning_score,
        "conditions_summary": [],
        "optimal_alpha": None,
        "pareto_frontier": [],
    }

    # Process each condition
    for cond in conditions:
        if not cond.merge_success or not cond.benchmark:
            analysis["conditions_summary"].append({
                "alpha": cond.alpha,
                "success": False,
                "error": cond.error,
            })
            continue

        code_score = cond.benchmark.code_score
        reasoning_score = cond.benchmark.reasoning_score

        code_delta = code_score - target_baseline.code_score
        reasoning_preservation = (
            reasoning_score / target_baseline.reasoning_score
            if target_baseline.reasoning_score > 0
            else 1.0
        )

        # Success criteria
        code_transferred = code_delta > 0
        reasoning_preserved = reasoning_preservation >= 0.90

        summary = {
            "alpha": cond.alpha,
            "success": True,
            "code_score": code_score,
            "reasoning_score": reasoning_score,
            "code_delta": code_delta,
            "reasoning_preservation": reasoning_preservation,
            "code_transferred": code_transferred,
            "reasoning_preserved": reasoning_preserved,
            "experiment_success": code_transferred and reasoning_preserved,
        }

        if cond.geometry:
            summary["comp_phi_mean"] = cond.geometry.comp_phi_mean
            summary["comp_phi_variance"] = cond.geometry.comp_phi_variance

        if cond.merge_metrics:
            summary["mean_preserved_fraction"] = cond.merge_metrics.get("mean_preserved_fraction")
            summary["weights_merged"] = cond.merge_metrics.get("weights_merged")

        analysis["conditions_summary"].append(summary)

    # Find optimal alpha (code > target AND reasoning >= 90%)
    successful = [
        s for s in analysis["conditions_summary"]
        if s.get("experiment_success", False)
    ]

    if successful:
        # Pick the one with highest code_delta
        best = max(successful, key=lambda x: x.get("code_delta", 0))
        analysis["optimal_alpha"] = best["alpha"]
        analysis["pareto_frontier"] = successful

    return analysis


def print_summary(result: ExperimentResult) -> None:
    """Print experiment summary."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nBaselines:")
    print(f"  Source - Code: {result.source_baseline.code_score:.1%}, Reasoning: {result.source_baseline.reasoning_score:.1%}")
    print(f"  Target - Code: {result.target_baseline.code_score:.1%}, Reasoning: {result.target_baseline.reasoning_score:.1%}")

    print("\nConditions:")
    print(f"{'Alpha':<8} {'Code':>8} {'Reason':>8} {'Δ Code':>8} {'Preserve':>10} {'Status':>12}")
    print("-"*56)

    for cond in result.analysis.get("conditions_summary", []):
        if not cond.get("success"):
            print(f"{cond['alpha']:<8.1f} {'FAILED':>8} {'-':>8} {'-':>8} {'-':>10} {'ERROR':>12}")
            continue

        code = cond.get("code_score", 0)
        reason = cond.get("reasoning_score", 0)
        delta = cond.get("code_delta", 0)
        preserve = cond.get("reasoning_preservation", 0)
        status = "SUCCESS" if cond.get("experiment_success") else "PARTIAL"

        print(f"{cond['alpha']:<8.1f} {code:>8.1%} {reason:>8.1%} {delta:>+8.1%} {preserve:>10.1%} {status:>12}")

    print()

    optimal = result.analysis.get("optimal_alpha")
    if optimal is not None:
        print(f"OPTIMAL ALPHA: {optimal}")
        print("  Found alpha that transfers code capability while preserving reasoning.")
    else:
        print("NO OPTIMAL ALPHA FOUND")
        print("  No alpha achieved both code transfer and reasoning preservation.")

    print("="*70)


def analyze_existing_results() -> None:
    """Analyze existing results from previous runs."""
    if not RESULTS_DIR.exists():
        print(f"No results found at {RESULTS_DIR}")
        return

    results_files = sorted(RESULTS_DIR.glob("alpha_*.json"))
    if not results_files:
        print("No condition results found.")
        return

    print("\nExisting Results:")
    print(f"{'Alpha':<8} {'Code':>8} {'Reason':>8} {'Preserved':>10} {'Status':>12}")
    print("-"*48)

    for result_file in results_files:
        with open(result_file) as f:
            data = json.load(f)

        alpha = data.get("alpha", 0)

        if not data.get("merge_success", False):
            print(f"{alpha:<8.1f} {'FAILED':>8} {'-':>8} {'-':>10} {'ERROR':>12}")
            continue

        benchmark = data.get("benchmark", {})
        code = benchmark.get("code_score", 0)
        reason = benchmark.get("reasoning_score", 0)

        metrics = data.get("merge_metrics", {})
        preserved = metrics.get("mean_preserved_fraction", 0)

        status = "OK" if code > 0 and reason > 0 else "PARTIAL"
        print(f"{alpha:<8.1f} {code:>8.1%} {reason:>8.1%} {preserved:>10.1%} {status:>12}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Soft Null-Space Merge Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Run single condition with this alpha value",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run full sweep over all alphas",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing results",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help="Path to source model (coding)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Path to target model (general)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help="Base path for merged model outputs",
    )

    args = parser.parse_args()

    # Validate paths
    for name, path in [("source", args.source), ("target", args.target)]:
        if not Path(path).exists():
            print(f"ERROR: {name} model not found: {path}")
            sys.exit(1)

    if args.analyze:
        analyze_existing_results()
    elif args.sweep:
        result = run_sweep(
            source_path=args.source,
            target_path=args.target,
            output_base=args.output,
        )
        # Save full result
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_file = RESULTS_DIR / "sweep_result.json"
        with open(result_file, "w") as f:
            # Convert to dict for JSON serialization
            result_dict = {
                "timestamp": result.timestamp,
                "source_model": result.source_model,
                "target_model": result.target_model,
                "source_baseline": asdict(result.source_baseline),
                "target_baseline": asdict(result.target_baseline),
                "analysis": result.analysis,
            }
            json.dump(result_dict, f, indent=2)
        print(f"\nFull results saved to: {result_file}")
    elif args.alpha is not None:
        # Single condition
        source_baseline = evaluate_model(args.source)
        target_baseline = evaluate_model(args.target)

        condition = run_single_condition(
            source_path=args.source,
            target_path=args.target,
            output_base=args.output,
            alpha=args.alpha,
        )

        # Quick analysis
        if condition.merge_success and condition.benchmark:
            code_delta = condition.benchmark.code_score - target_baseline.code_score
            reasoning_pres = (
                condition.benchmark.reasoning_score / target_baseline.reasoning_score
                if target_baseline.reasoning_score > 0
                else 1.0
            )
            print(f"\nRESULT: alpha={args.alpha}")
            print(f"  Code: {condition.benchmark.code_score:.1%} (delta: {code_delta:+.1%})")
            print(f"  Reasoning: {condition.benchmark.reasoning_score:.1%} (preserved: {reasoning_pres:.1%})")
            print(f"  Success: {code_delta > 0 and reasoning_pres >= 0.90}")
        else:
            print(f"\nRESULT: alpha={args.alpha} - FAILED")
            print(f"  Error: {condition.error}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
