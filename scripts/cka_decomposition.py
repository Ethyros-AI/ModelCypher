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

"""CKA Geometric Decomposition: What does min_cka=0.934 mean?

Decomposes per-layer CKA into its geometric constituents to determine
whether CKA < 1.0 is expected null-space learning or signal-space leakage.

Six measurements per layer:
  M1. Per-layer CKA curve (all layers)
  M2. Per-layer activation perturbation norm
  M3. Signal-space vs null-space energy decomposition (LoRA layers)
  M4. Per-layer spectral budget usage (LoRA layers)
  M5. Gram perturbation analysis (all layers)
  M6. Theoretical CKA bound vs actual

Usage:
    poetry run python scripts/cka_decomposition.py
    poetry run python scripts/cka_decomposition.py --seed 4231027559
    poetry run python scripts/cka_decomposition.py --skip-training --adapter-path /path/to/adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
TRAIN_DATA = "data/training/benchmark_train.jsonl"
EVAL_DATA = "data/training/benchmark_val.jsonl"
OUTPUT_DIR = Path("results/cka_decomposition")
DEFAULT_SEED = 4231027559  # Same seed as the validation smoke test

# LFM2-350M: layers with full attention (LoRA targets)
LORA_LAYER_INDICES = [2, 5, 8, 10, 12, 14]
N_LAYERS = 16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerMeasurement:
    """All six measurements for a single layer."""

    layer_idx: int
    is_lora_layer: bool

    # M1: CKA
    cka: float

    # M2: Activation perturbation
    relative_perturbation_norm: float  # ||Δh||_F / ||h_base||_F

    # M3: Signal/null decomposition (LoRA layers only)
    signal_energy_fraction: float | None = None  # ||Δh @ U_signal||² / ||Δh||²
    null_energy_fraction: float | None = None    # ||Δh @ U_null||² / ||Δh||²
    structural_rank: int | None = None           # floor(shannon_effective_rank)

    # M4: Spectral budget usage (LoRA layers only)
    budget_usage: float | None = None            # ||ΔW||_2 / σ_k
    spectral_norm_delta: float | None = None     # ||ΔW||_2
    sigma_k: float | None = None

    # M5: Gram perturbation
    gram_perturbation_ratio: float = 0.0         # ε = ||ΔK_c||_F / ||K_base_c||_F
    gram_perturbation_eff_rank: float | None = None  # participation ratio of ΔK_c

    # M6: Theoretical CKA bound
    cka_lower_bound: float = 1.0                 # (1 - ε) / (1 + ε)
    bound_is_tight: bool = True                  # actual ≥ bound


@dataclass
class DecompositionResult:
    """Full decomposition results."""

    model_path: str
    adapter_path: str | None
    seed: int
    n_probes: int
    n_layers: int

    # Training results (if training was run)
    training_loss_delta: float | None = None
    training_ppl_delta: float | None = None
    training_min_cka: float | None = None
    training_mean_cka: float | None = None
    training_max_spectral_ratio: float | None = None
    training_stop_reason: str | None = None

    per_layer: list[LayerMeasurement] = field(default_factory=list)

    # Aggregate answers to the five key questions
    worst_cka_layer: int | None = None
    worst_cka_value: float | None = None
    perturbation_monotonic_with_depth: bool | None = None
    max_signal_energy_at_worst: float | None = None
    max_null_energy_at_worst: float | None = None
    all_bounds_consistent: bool | None = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CKA geometric decomposition experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Training seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training, use existing adapter.",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Path to existing adapter (required with --skip-training).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-probes", type=int, default=None,
        help="Limit number of probe texts (default: all eval samples).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: Train and save adapter
# ---------------------------------------------------------------------------

def phase1_train(seed: int, output_dir: Path, log: logging.Logger) -> dict[str, Any]:
    """Train 350M model, return DatasetTrainResult as dict."""
    from modelcypher.cli.composition import get_dataset_training_service

    log.info("Phase 1: Training 350M with seed=%d", seed)
    service = get_dataset_training_service()

    adapter_output = output_dir / "adapter"
    adapter_output.mkdir(parents=True, exist_ok=True)

    result = service.train_from_dataset(
        model_path=MODEL_PATH,
        dataset_path=TRAIN_DATA,
        eval_dataset_path=EVAL_DATA,
        output_path=str(adapter_output),
        seed=seed,
    )

    log.info(
        "Training complete: loss %.4f→%.4f, min_cka=%.6f, stop=%s",
        result.baseline_loss,
        result.post_loss,
        result.min_cka if result.min_cka is not None else -1.0,
        result.stop_reason,
    )

    return {
        "adapter_path": result.adapter_path,
        "baseline_loss": result.baseline_loss,
        "post_loss": result.post_loss,
        "baseline_perplexity": result.baseline_perplexity,
        "post_perplexity": result.post_perplexity,
        "min_cka": result.min_cka,
        "mean_cka": result.mean_cka,
        "max_spectral_ratio": result.max_spectral_ratio,
        "stop_reason": result.stop_reason,
    }


# ---------------------------------------------------------------------------
# Phase 2: Collect per-layer activations (base + adapted)
# ---------------------------------------------------------------------------

def phase2_collect_activations(
    adapter_path: str,
    max_probes: int | None,
    log: logging.Logger,
) -> tuple[dict[int, Any], dict[int, Any], int]:
    """Collect per-layer activations for base and adapted models.

    Returns (base_stacks, adapted_stacks, n_probes) where each stack dict
    maps layer_idx -> Array[n_probes, hidden].
    """
    import mlx.core as mx

    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()
    loader = ModelLoader(backend)

    # Load eval samples for probes
    eval_path = Path(EVAL_DATA)
    eval_samples: list[dict[str, Any]] = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_samples.append(json.loads(line))

    probe_texts = [
        s["text"] for s in eval_samples
        if isinstance(s.get("text"), str) and s["text"]
    ]
    if max_probes is not None and max_probes < len(probe_texts):
        probe_texts = probe_texts[:max_probes]

    n_probes = len(probe_texts)
    log.info("Phase 2: Collecting activations with %d probes", n_probes)

    # -- Base model activations --
    log.info("  Loading base model...")
    model_base, tokenizer = loader.load_model(MODEL_PATH)

    log.info("  Collecting base activations...")
    base_acts: dict[int, list] = {}
    for i, text in enumerate(probe_texts):
        acts = backend.collect_hidden_activations(model_base, tokenizer, [text])
        for layer_idx, act in acts.items():
            # act: [1, seq, hidden] -> mean over seq -> [hidden]
            pooled = backend.mean(act, axis=1)
            pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            base_acts.setdefault(layer_idx, []).append(pooled)
        if (i + 1) % 50 == 0:
            log.info("    Base probe %d/%d", i + 1, n_probes)
    mx.eval()  # Ensure all base activations are materialized

    # Unload base model
    del model_base
    mx.eval()  # Force cleanup

    # -- Adapted model activations --
    log.info("  Loading adapted model (adapter=%s)...", adapter_path)
    model_adapted, _ = loader.load_model(MODEL_PATH, adapter_path=adapter_path)

    log.info("  Collecting adapted activations...")
    adapted_acts: dict[int, list] = {}
    for i, text in enumerate(probe_texts):
        acts = backend.collect_hidden_activations(model_adapted, tokenizer, [text])
        for layer_idx, act in acts.items():
            pooled = backend.mean(act, axis=1)
            pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            adapted_acts.setdefault(layer_idx, []).append(pooled)
        if (i + 1) % 50 == 0:
            log.info("    Adapted probe %d/%d", i + 1, n_probes)
    mx.eval()

    # Unload adapted model
    del model_adapted
    mx.eval()

    # Stack into [n_probes, hidden] per layer
    base_stacks: dict[int, Any] = {}
    adapted_stacks: dict[int, Any] = {}
    for layer_idx in sorted(base_acts.keys()):
        if layer_idx in adapted_acts:
            base_stacks[layer_idx] = backend.stack(base_acts[layer_idx])
            adapted_stacks[layer_idx] = backend.stack(adapted_acts[layer_idx])
            backend.eval(base_stacks[layer_idx], adapted_stacks[layer_idx])

    log.info("  Collected %d layers, %d probes each", len(base_stacks), n_probes)
    return base_stacks, adapted_stacks, n_probes


# ---------------------------------------------------------------------------
# Phase 3: Six measurements
# ---------------------------------------------------------------------------

def _frobenius_norm(arr: Any, backend: Any) -> float:
    """||arr||_F = sqrt(sum(arr²))."""
    sq_sum = backend.sum(arr * arr)
    backend.eval(sq_sum)
    return math.sqrt(float(backend.to_scalar(sq_sum)))


def m1_per_layer_cka(
    base_stacks: dict[int, Any],
    adapted_stacks: dict[int, Any],
    backend: Any,
    log: logging.Logger,
) -> dict[int, float]:
    """M1: Per-layer linear CKA between base and adapted activations."""
    from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

    log.info("M1: Computing per-layer CKA...")
    cka_per_layer: dict[int, float] = {}
    for layer_idx in sorted(base_stacks.keys()):
        cka = compute_linear_cka_from_activations(
            base_stacks[layer_idx], adapted_stacks[layer_idx], backend,
        )
        cka_per_layer[layer_idx] = cka
        log.info("  Layer %2d: CKA = %.6f", layer_idx, cka)
    return cka_per_layer


def m2_perturbation_norms(
    base_stacks: dict[int, Any],
    adapted_stacks: dict[int, Any],
    backend: Any,
    log: logging.Logger,
) -> dict[int, float]:
    """M2: Per-layer relative activation perturbation norm."""
    log.info("M2: Computing per-layer perturbation norms...")
    norms: dict[int, float] = {}
    for layer_idx in sorted(base_stacks.keys()):
        delta = adapted_stacks[layer_idx] - base_stacks[layer_idx]
        backend.eval(delta)
        delta_norm = _frobenius_norm(delta, backend)
        base_norm = _frobenius_norm(base_stacks[layer_idx], backend)
        rel = delta_norm / base_norm if base_norm > 0 else 0.0
        norms[layer_idx] = rel
        log.info("  Layer %2d: ||Δh||/||h|| = %.6f", layer_idx, rel)
    return norms


def m3_signal_null_decomposition(
    base_stacks: dict[int, Any],
    adapted_stacks: dict[int, Any],
    backend: Any,
    log: logging.Logger,
) -> dict[int, dict[str, Any]]:
    """M3: Signal-space vs null-space energy decomposition at LoRA layers.

    For each LoRA layer, we decompose the activation perturbation into
    components along the base weight's right singular directions (input space).

    The weight W [out, in] maps input -> output. Its SVD: W = U @ diag(S) @ Vt.
    V = Vt.T columns span the input space. The hidden state h lives in
    the input space (hidden_dim = in_features). Top-k columns of V = signal
    input directions. Remaining columns = tail directions (where LoRA operates).

    We project Δh onto V_signal (signal input space) and V_tail (tail input space).
    The orthogonal complement of V in R^in is the true null space of W.
    """
    import mlx.core as mx

    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.core.domain.training.geometric_lora import compute_layer_geometry

    log.info("M3: Signal/null decomposition at LoRA layers...")

    # Load base weights for SVD decomposition
    weights = ModelLoader(backend).load_weights(MODEL_PATH)

    results: dict[int, dict[str, Any]] = {}

    for layer_idx in LORA_LAYER_INDICES:
        if layer_idx not in base_stacks:
            continue

        # Find the self_attn projection weights for this layer.
        # NB-LoRA targets projections on attention layers.
        # We use the v_proj as representative (LoRA modifies q_proj, k_proj,
        # v_proj, o_proj — v_proj is typically the most impactful).
        weight_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        if weight_key not in weights:
            # Try alternative key patterns
            for proj in ["v_proj", "o_proj", "q_proj", "k_proj"]:
                candidate = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
                if candidate in weights:
                    weight_key = candidate
                    break
            else:
                log.warning("  Layer %d: no attention weight found, skipping", layer_idx)
                continue

        W = backend.astype(weights[weight_key], "float32")
        backend.eval(W)

        # Compute geometry to get structural rank
        geom = compute_layer_geometry(W, weight_key, backend)
        structural_rank = math.floor(geom.shannon_effective_rank)

        # Full SVD for signal/null decomposition
        # W [out, in]: SVD gives U [out, min], S [min], Vt [min, in]
        # V = Vt.T [in, min] — right singular vectors in the INPUT space
        U, S_w, Vt = backend.svd(W)
        backend.eval(S_w, Vt)

        out_dim = int(W.shape[0])
        in_dim = int(W.shape[1])
        min_dim = min(out_dim, in_dim)
        k = min(structural_rank, min_dim)

        # V = Vt.T: columns are right singular vectors in R^in (hidden_dim)
        V = backend.transpose(Vt)  # [in, min_dim]
        backend.eval(V)

        # Activation perturbation at this layer output
        delta_h = adapted_stacks[layer_idx] - base_stacks[layer_idx]
        delta_h = backend.astype(delta_h, "float32")
        backend.eval(delta_h)

        total_energy = _frobenius_norm(delta_h, backend) ** 2
        if total_energy < 1e-30:
            results[layer_idx] = {
                "signal_energy_fraction": 0.0,
                "null_energy_fraction": 0.0,
                "structural_rank": structural_rank,
            }
            continue

        # Decompose delta_h [n_probes, in_dim] into three subspaces:
        #   1. Signal: top-k right singular directions (W uses strongly)
        #   2. Tail: remaining right singular directions (W uses weakly, LoRA target)
        #   3. Null complement: directions orthogonal to all of V (W maps to zero)
        V_signal = V[:, :k]          # [in, k] — signal input directions
        V_tail = V[:, k:]            # [in, min_dim-k] — tail input directions
        backend.eval(V_signal, V_tail)

        # Project: Δh [n_probes, in] @ V_signal [in, k] -> [n_probes, k]
        proj_signal = backend.matmul(delta_h, V_signal)
        proj_tail = backend.matmul(delta_h, V_tail)
        backend.eval(proj_signal, proj_tail)

        signal_energy = _frobenius_norm(proj_signal, backend) ** 2
        tail_energy = _frobenius_norm(proj_tail, backend) ** 2

        # Energy in V's span (signal + tail)
        v_span_energy = signal_energy + tail_energy
        # Energy in V's orthogonal complement (true null space of W)
        # By Parseval: ||x||² = ||P_V x||² + ||P_V⊥ x||²
        null_complement_energy = total_energy - v_span_energy

        signal_frac = signal_energy / total_energy
        # "null" = tail + null_complement (all non-signal directions)
        null_frac = (tail_energy + max(0.0, null_complement_energy)) / total_energy

        # Parseval sanity: fractions should sum to ≈ 1.0
        parseval_check = signal_frac + null_frac
        if abs(parseval_check - 1.0) > 0.02:
            log.warning(
                "  Layer %d: Parseval violation! signal+null=%.4f (expected 1.0)",
                layer_idx, parseval_check,
            )

        results[layer_idx] = {
            "signal_energy_fraction": signal_frac,
            "null_energy_fraction": null_frac,
            "structural_rank": structural_rank,
            "tail_energy_fraction": tail_energy / total_energy,
            "null_complement_energy_fraction": max(0.0, null_complement_energy) / total_energy,
        }
        log.info(
            "  Layer %2d: signal=%.1f%% tail=%.1f%% null_complement=%.1f%% (rank=%d/%d)",
            layer_idx, signal_frac * 100,
            (tail_energy / total_energy) * 100,
            (max(0.0, null_complement_energy) / total_energy) * 100,
            structural_rank, min_dim,
        )

    return results


def m4_spectral_budget_usage(
    adapter_path: str,
    backend: Any,
    log: logging.Logger,
) -> dict[int, dict[str, float]]:
    """M4: Per-layer spectral budget usage via weight diff.

    Loads base weights and adapted model (with fused adapter), computes
    ΔW = W_adapted - W_base per projection, then ||ΔW||_2 / σ_k.
    """
    import mlx.core as mx

    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.core.domain.training.geometric_lora import compute_layer_geometry

    log.info("M4: Computing spectral budget usage via weight diff...")

    # Load base weights from safetensors
    loader = ModelLoader(backend)
    base_weights = loader.load_weights(MODEL_PATH)

    # Load adapted model (adapter fused into weights)
    log.info("  Loading adapted model for weight diff...")
    model_adapted, _ = loader.load_model(MODEL_PATH, adapter_path=adapter_path)

    results: dict[int, dict[str, float]] = {}

    for layer_idx in LORA_LAYER_INDICES:
        for proj in ["v_proj", "q_proj", "k_proj"]:
            w_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"

            if w_key not in base_weights:
                log.warning("  Base weight key not found: %s", w_key)
                continue

            # Compute ΔW from LoRA parameters directly.
            # mlx-lm LoRALinear does NOT fuse — linear.weight is the base weight.
            # Forward: out = linear(x) + (x @ lora_a) @ lora_b * scale
            # So ΔW [out, in] = scale * lora_b.T @ lora_a.T
            try:
                layer = model_adapted.model.layers[layer_idx]
                proj_module = getattr(layer.self_attn, proj)
                lora_a = proj_module.lora_a  # [in, rank]
                lora_b = proj_module.lora_b  # [rank, out]
                scale = getattr(proj_module, "scale", 1.0)
            except (AttributeError, IndexError) as e:
                log.warning("  Cannot access LoRA params for layer %d %s: %s", layer_idx, proj, e)
                continue

            delta_W = scale * backend.matmul(
                backend.transpose(backend.astype(lora_b, "float32")),
                backend.transpose(backend.astype(lora_a, "float32")),
            )
            backend.eval(delta_W)

            # Base geometry for σ_k
            W_base = backend.astype(base_weights[w_key], "float32")
            backend.eval(W_base)
            geom = compute_layer_geometry(W_base, w_key, backend)

            # Spectral norm of delta
            S_delta = backend.svd(delta_W, compute_uv=False)
            backend.eval(S_delta)
            spectral_norm = float(backend.to_scalar(S_delta[0]))
            budget = spectral_norm / geom.sigma_k if geom.sigma_k > 0 else float("inf")

            # Aggregate: take max across projections per layer
            if layer_idx not in results or budget > results[layer_idx]["budget_usage"]:
                results[layer_idx] = {
                    "budget_usage": budget,
                    "spectral_norm_delta": spectral_norm,
                    "sigma_k": geom.sigma_k,
                }
                log.info(
                    "  Layer %2d (%s): ||ΔW||_2=%.6f, σ_k=%.6f, budget=%.4f",
                    layer_idx, proj, spectral_norm, geom.sigma_k, budget,
                )

    del model_adapted
    mx.eval()

    return results


def m5_gram_perturbation(
    base_stacks: dict[int, Any],
    adapted_stacks: dict[int, Any],
    backend: Any,
    log: logging.Logger,
) -> dict[int, dict[str, float]]:
    """M5: Gram perturbation analysis per layer.

    Computes ε = ||ΔK_c||_F / ||K_base_c||_F and the effective rank of ΔK_c.
    """
    log.info("M5: Gram perturbation analysis...")

    results: dict[int, dict[str, float]] = {}

    for layer_idx in sorted(base_stacks.keys()):
        X = backend.astype(base_stacks[layer_idx], "float32")
        Y = backend.astype(adapted_stacks[layer_idx], "float32")
        backend.eval(X, Y)

        # Dot-product Gram matrices
        K_base = backend.matmul(X, backend.transpose(X))
        K_adapted = backend.matmul(Y, backend.transpose(Y))
        backend.eval(K_base, K_adapted)

        # Center: K_c = K - row_mean - col_mean + grand_mean
        def _center(K: Any) -> Any:
            row_mean = backend.mean(K, axis=1, keepdims=True)
            col_mean = backend.mean(K, axis=0, keepdims=True)
            grand_mean = backend.mean(K)
            centered = K - row_mean - col_mean + grand_mean
            backend.eval(centered)
            return centered

        K_base_c = _center(K_base)
        K_adapted_c = _center(K_adapted)
        DeltaK_c = K_adapted_c - K_base_c
        backend.eval(DeltaK_c)

        # Gram perturbation ratio
        delta_norm = _frobenius_norm(DeltaK_c, backend)
        base_norm = _frobenius_norm(K_base_c, backend)
        epsilon = delta_norm / base_norm if base_norm > 0 else 0.0

        # Effective rank of ΔK_c (participation ratio)
        eff_rank = None
        try:
            S_delta = backend.svd(DeltaK_c, compute_uv=False)
            backend.eval(S_delta)
            s_vals = [float(backend.to_scalar(S_delta[i])) for i in range(int(S_delta.shape[0]))]
            s_sum = sum(s_vals)
            s_sq_sum = sum(v * v for v in s_vals)
            if s_sq_sum > 0:
                eff_rank = (s_sum ** 2) / s_sq_sum
        except Exception as e:
            log.warning("  Layer %d: SVD of ΔK_c failed: %s", layer_idx, e)

        results[layer_idx] = {
            "gram_perturbation_ratio": epsilon,
            "gram_perturbation_eff_rank": eff_rank,
        }
        log.info(
            "  Layer %2d: ε=%.6f, eff_rank=%s",
            layer_idx, epsilon,
            f"{eff_rank:.2f}" if eff_rank is not None else "N/A",
        )

    return results


def m6_theoretical_bounds(
    cka_per_layer: dict[int, float],
    gram_perturbation: dict[int, dict[str, float]],
    log: logging.Logger,
) -> dict[int, dict[str, Any]]:
    """M6: Compare theoretical CKA lower bound to actual CKA."""
    log.info("M6: Theoretical CKA bounds vs actual...")
    results: dict[int, dict[str, Any]] = {}

    for layer_idx in sorted(cka_per_layer.keys()):
        actual = cka_per_layer[layer_idx]
        epsilon = gram_perturbation.get(layer_idx, {}).get("gram_perturbation_ratio", 0.0)

        # Theoretical lower bound: CKA ≥ (1 - ε) / (1 + ε)
        bound = (1.0 - epsilon) / (1.0 + epsilon) if (1.0 + epsilon) > 0 else 0.0
        is_tight = actual >= bound - 1e-6  # Small tolerance for floating point

        results[layer_idx] = {
            "cka_lower_bound": bound,
            "bound_is_tight": is_tight,
        }
        marker = "OK" if is_tight else "VIOLATION"
        log.info(
            "  Layer %2d: actual=%.6f bound=%.6f gap=%.6f [%s]",
            layer_idx, actual, bound, actual - bound, marker,
        )

    return results


# ---------------------------------------------------------------------------
# Phase 4: Assemble and output
# ---------------------------------------------------------------------------

def assemble_results(
    training_info: dict[str, Any] | None,
    adapter_path: str,
    seed: int,
    n_probes: int,
    cka_per_layer: dict[int, float],
    perturbation_norms: dict[int, float],
    signal_null: dict[int, dict[str, Any]],
    budget_usage: dict[int, dict[str, float]],
    gram_perturbation: dict[int, dict[str, float]],
    theoretical_bounds: dict[int, dict[str, Any]],
) -> DecompositionResult:
    """Assemble all measurements into a single result object."""
    result = DecompositionResult(
        model_path=MODEL_PATH,
        adapter_path=adapter_path,
        seed=seed,
        n_probes=n_probes,
        n_layers=len(cka_per_layer),
    )

    if training_info:
        result.training_loss_delta = (
            training_info["baseline_loss"] - training_info["post_loss"]
        )
        result.training_ppl_delta = (
            training_info["baseline_perplexity"] - training_info["post_perplexity"]
        )
        result.training_min_cka = training_info.get("min_cka")
        result.training_mean_cka = training_info.get("mean_cka")
        result.training_max_spectral_ratio = training_info.get("max_spectral_ratio")
        result.training_stop_reason = training_info.get("stop_reason")

    for layer_idx in sorted(cka_per_layer.keys()):
        is_lora = layer_idx in LORA_LAYER_INDICES
        sn = signal_null.get(layer_idx, {})
        bu = budget_usage.get(layer_idx, {})
        gp = gram_perturbation.get(layer_idx, {})
        tb = theoretical_bounds.get(layer_idx, {})

        m = LayerMeasurement(
            layer_idx=layer_idx,
            is_lora_layer=is_lora,
            cka=cka_per_layer[layer_idx],
            relative_perturbation_norm=perturbation_norms.get(layer_idx, 0.0),
            signal_energy_fraction=sn.get("signal_energy_fraction"),
            null_energy_fraction=sn.get("null_energy_fraction"),
            structural_rank=sn.get("structural_rank"),
            budget_usage=bu.get("budget_usage"),
            spectral_norm_delta=bu.get("spectral_norm_delta"),
            sigma_k=bu.get("sigma_k"),
            gram_perturbation_ratio=gp.get("gram_perturbation_ratio", 0.0),
            gram_perturbation_eff_rank=gp.get("gram_perturbation_eff_rank"),
            cka_lower_bound=tb.get("cka_lower_bound", 1.0),
            bound_is_tight=tb.get("bound_is_tight", True),
        )
        result.per_layer.append(m)

    # Aggregate answers
    if result.per_layer:
        worst = min(result.per_layer, key=lambda m: m.cka)
        result.worst_cka_layer = worst.layer_idx
        result.worst_cka_value = worst.cka
        result.max_signal_energy_at_worst = worst.signal_energy_fraction
        result.max_null_energy_at_worst = worst.null_energy_fraction
        result.all_bounds_consistent = all(m.bound_is_tight for m in result.per_layer)

        # Check monotonicity: does perturbation increase with depth?
        norms = [m.relative_perturbation_norm for m in result.per_layer]
        result.perturbation_monotonic_with_depth = all(
            norms[i] <= norms[i + 1] + 1e-6 for i in range(len(norms) - 1)
        )

    return result


def build_markdown_report(result: DecompositionResult) -> str:
    """Build human-readable markdown analysis."""
    lines = [
        "# CKA Geometric Decomposition: 350M Analysis",
        "",
        f"**Model:** {result.model_path}",
        f"**Adapter:** {result.adapter_path}",
        f"**Seed:** {result.seed}",
        f"**Probes:** {result.n_probes}",
        "",
    ]

    if result.training_loss_delta is not None:
        lines.extend([
            "## Training Summary",
            "",
            f"- Loss delta: {result.training_loss_delta:+.4f}",
            f"- PPL delta: {result.training_ppl_delta:+.4f}",
            f"- min_cka (training): {result.training_min_cka:.6f}"
            if result.training_min_cka is not None else "- min_cka: N/A",
            f"- mean_cka (training): {result.training_mean_cka:.6f}"
            if result.training_mean_cka is not None else "- mean_cka: N/A",
            f"- max_spectral_ratio: {result.training_max_spectral_ratio:.4f}"
            if result.training_max_spectral_ratio is not None else "- max_spectral_ratio: N/A",
            f"- stop_reason: {result.training_stop_reason}",
            "",
        ])

    # Main table
    lines.extend([
        "## Per-Layer Decomposition",
        "",
        "| Layer | CKA | ||Δh||/||h|| | Signal% | Null% | Budget | ε | CKA_bound | Tight? | LoRA? |",
        "|------:|------:|------------:|--------:|------:|-------:|------:|----------:|-------:|------:|",
    ])

    for m in result.per_layer:
        signal = f"{m.signal_energy_fraction * 100:.1f}" if m.signal_energy_fraction is not None else "-"
        null = f"{m.null_energy_fraction * 100:.1f}" if m.null_energy_fraction is not None else "-"
        budget = f"{m.budget_usage:.4f}" if m.budget_usage is not None else "-"
        tight = "OK" if m.bound_is_tight else "FAIL"
        lora = "yes" if m.is_lora_layer else "no"
        lines.append(
            f"| {m.layer_idx:2d} | {m.cka:.6f} | {m.relative_perturbation_norm:.6f} "
            f"| {signal} | {null} | {budget} | {m.gram_perturbation_ratio:.6f} "
            f"| {m.cka_lower_bound:.6f} | {tight} | {lora} |"
        )

    lines.append("")

    # Gram perturbation effective rank
    has_eff_rank = any(m.gram_perturbation_eff_rank is not None for m in result.per_layer)
    if has_eff_rank:
        lines.extend([
            "## Gram Perturbation Structure",
            "",
            "| Layer | ε | ΔK_c eff_rank | LoRA? |",
            "|------:|------:|--------------:|------:|",
        ])
        for m in result.per_layer:
            er = f"{m.gram_perturbation_eff_rank:.2f}" if m.gram_perturbation_eff_rank is not None else "-"
            lora = "yes" if m.is_lora_layer else "no"
            lines.append(f"| {m.layer_idx:2d} | {m.gram_perturbation_ratio:.6f} | {er} | {lora} |")
        lines.append("")

    # Answers to five key questions
    lines.extend([
        "## Key Questions",
        "",
    ])

    # Q1: Does CKA decrease monotonically with depth?
    if result.per_layer:
        cka_vals = [(m.layer_idx, m.cka) for m in result.per_layer]
        lines.append(f"**Q1. Does CKA decrease monotonically with depth?**")
        monotonic = all(
            cka_vals[i][1] >= cka_vals[i + 1][1] - 1e-6
            for i in range(len(cka_vals) - 1)
        )
        lines.append(f"  {'Yes' if monotonic else 'No'}. Worst layer: {result.worst_cka_layer} (CKA={result.worst_cka_value:.6f})")
        lines.append("")

    # Q2: Does perturbation accumulate through depth?
    lines.append(f"**Q2. Does perturbation accumulate monotonically through depth?**")
    lines.append(f"  {'Yes' if result.perturbation_monotonic_with_depth else 'No'}.")
    lines.append("")

    # Q3: Is the CKA drop from null-space or signal-space?
    lines.append(f"**Q3. Is the CKA drop at worst layer from null-space addition or signal-space leakage?**")
    if result.max_signal_energy_at_worst is not None:
        sig = result.max_signal_energy_at_worst * 100
        nul = result.max_null_energy_at_worst * 100 if result.max_null_energy_at_worst else 0
        if nul > 80:
            lines.append(f"  **Null-space dominant.** Signal={sig:.1f}%, Null={nul:.1f}%.")
            lines.append(f"  The CKA drop is expected null-space learning.")
        elif sig > 50:
            lines.append(f"  **Signal-space leakage.** Signal={sig:.1f}%, Null={nul:.1f}%.")
            lines.append(f"  The adapter is perturbing existing features.")
        else:
            lines.append(f"  **Mixed.** Signal={sig:.1f}%, Null={nul:.1f}%.")
    else:
        lines.append(f"  Worst layer ({result.worst_cka_layer}) is not a LoRA layer — perturbation is propagated from upstream.")
    lines.append("")

    # Q4: Are all perturbation theory bounds consistent?
    lines.append(f"**Q4. Are all theoretical CKA bounds consistent (actual ≥ bound)?**")
    lines.append(f"  {'Yes' if result.all_bounds_consistent else 'No — destructive interference detected'}.")
    lines.append("")

    # Q5: What should the CKA gate threshold be?
    lines.append(f"**Q5. What should the CKA gate threshold be?**")
    if result.all_bounds_consistent and result.worst_cka_value is not None:
        worst_bound = None
        for m in result.per_layer:
            if m.layer_idx == result.worst_cka_layer:
                worst_bound = m.cka_lower_bound
                break
        if worst_bound is not None:
            lines.append(f"  Current threshold: 1.0 - sqrt(eps) ≈ 0.9997")
            lines.append(f"  Theoretical minimum at worst layer: {worst_bound:.6f}")
            lines.append(f"  Actual at worst layer: {result.worst_cka_value:.6f}")
            if result.max_null_energy_at_worst is not None and result.max_null_energy_at_worst > 0.8:
                lines.append(f"  **Recommendation:** The CKA drop is geometrically expected.")
                lines.append(f"  Threshold should be derived from the Gram perturbation bound,")
                lines.append(f"  not from machine epsilon.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", mode="w"),
        ],
    )
    log = logging.getLogger("cka_decomposition")

    # Check volume
    if not Path("/Volumes/CodeCypher/models/mlx-community").exists():
        print("ERROR: Volume not mounted at /Volumes/CodeCypher/models/mlx-community", file=sys.stderr)
        sys.exit(2)

    log.info("=" * 70)
    log.info("CKA Geometric Decomposition Experiment")
    log.info("  Model: %s", MODEL_PATH)
    log.info("  Seed:  %d", args.seed)
    log.info("=" * 70)

    t_start = time.time()

    # Phase 1: Train (or skip)
    training_info = None
    adapter_path = args.adapter_path

    if args.skip_training:
        if not adapter_path:
            print("ERROR: --adapter-path required with --skip-training", file=sys.stderr)
            sys.exit(1)
        log.info("Skipping training, using adapter: %s", adapter_path)
    else:
        training_info = phase1_train(args.seed, output_dir, log)
        adapter_path = training_info["adapter_path"]

    if not adapter_path or not Path(adapter_path).exists():
        print(f"ERROR: Adapter path does not exist: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Collect activations
    base_stacks, adapted_stacks, n_probes = phase2_collect_activations(
        adapter_path, args.max_probes, log,
    )

    # Initialize backend for measurements
    from modelcypher.core.domain._backend import get_default_backend
    backend = get_default_backend()

    # Phase 3: Six measurements
    cka_per_layer = m1_per_layer_cka(base_stacks, adapted_stacks, backend, log)
    perturbation_norms = m2_perturbation_norms(base_stacks, adapted_stacks, backend, log)
    signal_null = m3_signal_null_decomposition(base_stacks, adapted_stacks, backend, log)
    budget_usage = m4_spectral_budget_usage(adapter_path, backend, log)
    gram_perturbation = m5_gram_perturbation(base_stacks, adapted_stacks, backend, log)
    theoretical_bounds = m6_theoretical_bounds(cka_per_layer, gram_perturbation, log)

    # Phase 4: Assemble and output
    result = assemble_results(
        training_info, adapter_path, args.seed, n_probes,
        cka_per_layer, perturbation_norms, signal_null,
        budget_usage, gram_perturbation, theoretical_bounds,
    )

    # Write JSON
    json_path = output_dir / "cka_decomposition_results.json"
    json_data = asdict(result)
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %s", json_path)

    # Write markdown
    report = build_markdown_report(result)
    report_path = output_dir / "ANALYSIS.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("Wrote %s", report_path)

    # Print report to stdout
    print()
    print(report)

    elapsed = time.time() - t_start
    log.info("Completed in %.1fs", elapsed)
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results: {json_path}")
    print(f"Analysis: {report_path}")


if __name__ == "__main__":
    main()
