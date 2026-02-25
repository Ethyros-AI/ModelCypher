#!/usr/bin/env python3
"""K-FAC Validation Experiments — 3 go/no-go tests on real models.

Experiment 1: Gain Ratio — How much larger is Null(G_cap) vs Null(K_cap)?
Experiment 2: K-FAC vs Full Jacobian — Do signal spaces agree?
Experiment 3: Training Curvature Alignment — Does Cayley+MASS already avoid high-curvature?

Usage:
    poetry run python scripts/kfac_validation.py --experiment all
    poetry run python scripts/kfac_validation.py --experiment 1 --models LFM2-350M LFM2-700M
    poetry run python scripts/kfac_validation.py --experiment 2
    poetry run python scripts/kfac_validation.py --experiment 3
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain.geometry.kfac_diagnostic import (
    KFACDiagnosticResult,
    compute_kfac_diagnostic_from_weight_gradients,
)
from modelcypher.core.domain.geometry.kfac_projector import (
    KFACFactors,
    factors_from_diagnostic,
)
from modelcypher.core.domain.geometry.subspace import (
    compute_grassmann_distance,
)
from modelcypher.core.domain.geometry.transplant import (
    compute_behavior_jacobian_projector,
)
from modelcypher.core.domain.training.kfac_curvature_monitor import (
    compute_curvature_alignment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = {
    "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "LFM2-700M": "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16",
    "Qwen3-8B": "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
}

PROBES_PATH = Path("data/training/benchmark_val.jsonl")
RESULTS_DIR = Path("/Volumes/CodeCypher/experiments/kfac-validation")
N_PROBES = 30
N_PROBES_8B_FFN = 15  # Reduced for 8B FFN to halve gradient memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer configs per model
# ---------------------------------------------------------------------------

@dataclass
class WeightSpec:
    """One weight matrix to measure."""
    layer_idx: int
    weight_suffix: str  # e.g. "self_attn.q_proj", "feed_forward.w1"
    act_type: str  # "hidden" or "intermediate"


def _lfm2_weight_name(layer_idx: int, suffix: str) -> str:
    """Weight name for LFM2 models (tree_flatten dot-path)."""
    return f"model.layers.{layer_idx}.{suffix}.weight"


def _qwen_weight_name(layer_idx: int, suffix: str) -> str:
    """Weight name for Qwen models (tree_flatten dot-path)."""
    return f"model.layers.{layer_idx}.{suffix}.weight"


def get_layer_config(model_name: str) -> list[WeightSpec]:
    """Return list of (layer_idx, weight_suffix, act_type) to probe."""
    if model_name.startswith("LFM2"):
        return [
            WeightSpec(2, "self_attn.q_proj", "hidden"),
            WeightSpec(2, "feed_forward.w1", "hidden"),
            WeightSpec(2, "feed_forward.w2", "intermediate"),
            WeightSpec(5, "self_attn.q_proj", "hidden"),
            WeightSpec(5, "feed_forward.w1", "hidden"),
            WeightSpec(5, "feed_forward.w2", "intermediate"),
            WeightSpec(8, "self_attn.q_proj", "hidden"),
            WeightSpec(8, "feed_forward.w1", "hidden"),
            WeightSpec(8, "feed_forward.w2", "intermediate"),
            WeightSpec(12, "self_attn.q_proj", "hidden"),
            WeightSpec(12, "feed_forward.w1", "hidden"),
            WeightSpec(12, "feed_forward.w2", "intermediate"),
        ]
    elif model_name == "Qwen3-8B":
        return [
            WeightSpec(4, "self_attn.q_proj", "hidden"),
            WeightSpec(4, "mlp.up_proj", "hidden"),
            WeightSpec(4, "mlp.down_proj", "intermediate"),
            WeightSpec(12, "self_attn.q_proj", "hidden"),
            WeightSpec(12, "mlp.up_proj", "hidden"),
            WeightSpec(12, "mlp.down_proj", "intermediate"),
            WeightSpec(24, "self_attn.q_proj", "hidden"),
            WeightSpec(24, "mlp.up_proj", "hidden"),
            WeightSpec(24, "mlp.down_proj", "intermediate"),
            WeightSpec(32, "self_attn.q_proj", "hidden"),
            WeightSpec(32, "mlp.up_proj", "hidden"),
            WeightSpec(32, "mlp.down_proj", "intermediate"),
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_weight_name(model_name: str, spec: WeightSpec) -> str:
    """Get full dot-path weight name for compute_per_probe_gradients."""
    if model_name.startswith("LFM2"):
        return _lfm2_weight_name(spec.layer_idx, spec.weight_suffix)
    else:
        return _qwen_weight_name(spec.layer_idx, spec.weight_suffix)


def _is_8b_ffn(model_name: str, spec: WeightSpec) -> bool:
    """Check if this is a Qwen3-8B FFN weight (needs reduced probes)."""
    return model_name == "Qwen3-8B" and "mlp." in spec.weight_suffix


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probes(n: int = N_PROBES) -> list[str]:
    """Load n probe texts from benchmark_val.jsonl."""
    probes = []
    with open(PROBES_PATH) as f:
        for line in f:
            if len(probes) >= n:
                break
            data = json.loads(line.strip())
            probes.append(data["text"])
    if len(probes) < n:
        logger.warning("Only %d probes available (requested %d)", len(probes), n)
    return probes


# ---------------------------------------------------------------------------
# Activation collection helpers
# ---------------------------------------------------------------------------

def collect_input_activations(
    backend: Any,
    model: Any,
    tokenizer: Any,
    probes: list[str],
    spec: WeightSpec,
) -> Any:
    """Collect input activations for a weight matrix, stacked as [N, dim]."""
    import mlx.core as mx

    if spec.act_type == "hidden":
        results = backend.collect_hidden_activations_batch(model, tokenizer, probes)
        acts = [results[i][spec.layer_idx] for i in range(len(probes))
                if spec.layer_idx in results[i]]
    elif spec.act_type == "intermediate":
        results = backend.collect_intermediate_activations_batch(model, tokenizer, probes)
        acts = [results[i][spec.layer_idx] for i in range(len(probes))
                if spec.layer_idx in results[i]]
    else:
        raise ValueError(f"Unknown act_type: {spec.act_type}")

    if not acts:
        raise RuntimeError(
            f"No activations collected for layer {spec.layer_idx} "
            f"({spec.act_type}). Model may not have this layer."
        )

    stacked = mx.stack(acts)  # [N, dim]
    mx.eval(stacked)
    return stacked


def get_weight_shape(model: Any, weight_name: str) -> tuple[int, int]:
    """Get (out_dim, in_dim) from model weight."""
    from mlx.utils import tree_flatten

    for name, tensor in tree_flatten(model.parameters()):
        if name == weight_name:
            shape = tensor.shape
            return (int(shape[0]), int(shape[1]))
    raise ValueError(f"Weight '{weight_name}' not found in model parameters.")


# ---------------------------------------------------------------------------
# Experiment 1: Gain Ratio
# ---------------------------------------------------------------------------

@dataclass
class GainRatioMeasurement:
    model_name: str
    weight_name: str
    layer_idx: int
    weight_suffix: str
    n_probes: int
    activation_rank: int
    act_null_rank: int
    kfac_null_rank: int
    gain_ratio: float


def _measure_gain_ratio(
    backend: Any,
    model: Any,
    tokenizer: Any,
    probes: list[str],
    model_name: str,
    spec: WeightSpec,
) -> GainRatioMeasurement:
    """Measure K-FAC gain ratio for one weight matrix."""
    import mlx.core as mx

    wname = get_weight_name(model_name, spec)
    wshape = get_weight_shape(model, wname)
    label = f"layer.{spec.layer_idx}.{spec.weight_suffix}"

    logger.info("  Computing %s (shape=%s)...", label, wshape)
    t0 = time.time()

    # Collect input activations
    acts = collect_input_activations(backend, model, tokenizer, probes, spec)
    logger.info("    activations: %s", acts.shape)

    # Compute per-probe weight gradients
    grads = backend.compute_per_probe_gradients(model, tokenizer, probes, wname)
    logger.info("    gradients: %s (%.1fs)", grads.shape, time.time() - t0)

    # K-FAC diagnostic
    diagnostic = compute_kfac_diagnostic_from_weight_gradients(
        input_activations=acts,
        per_probe_weight_gradients=grads,
        weight_shape=wshape,
        backend=backend,
    )

    result = GainRatioMeasurement(
        model_name=model_name,
        weight_name=wname,
        layer_idx=spec.layer_idx,
        weight_suffix=spec.weight_suffix,
        n_probes=diagnostic.n_probes,
        activation_rank=diagnostic.activation_rank,
        act_null_rank=diagnostic.activation_null_rank_weight,
        kfac_null_rank=diagnostic.kfac_null_rank,
        gain_ratio=diagnostic.kfac_gain_ratio,
    )

    logger.info(
        "    act_rank=%d, act_null=%d, kfac_null=%d, gain=%.3f (%.1fs)",
        result.activation_rank, result.act_null_rank,
        result.kfac_null_rank, result.gain_ratio,
        time.time() - t0,
    )

    # Free gradient memory
    del grads, acts, diagnostic
    gc.collect()

    return result


def experiment_1_gain_ratio(
    backend: Any,
    model_names: list[str],
) -> dict[str, Any]:
    """Experiment 1: K-FAC Gain Ratio on Real Models.

    GO if median gain > 1.10 on 2+ models.
    NO-GO if < 1.05 everywhere.
    """
    all_probes = load_probes(N_PROBES)
    results_by_model: dict[str, list[GainRatioMeasurement]] = {}
    model_medians: dict[str, float] = {}

    print("\n" + "=" * 60)
    print("=== EXPERIMENT 1: K-FAC Gain Ratio ===")
    print("=" * 60)

    for model_name in model_names:
        model_path = MODELS[model_name]
        print(f"\nModel: {model_name}")
        logger.info("Loading %s from %s", model_name, model_path)

        model, tokenizer = backend.load_model(model_path)
        layer_config = get_layer_config(model_name)

        measurements: list[GainRatioMeasurement] = []
        for spec in layer_config:
            # Use fewer probes for 8B FFN to manage memory
            n = N_PROBES_8B_FFN if _is_8b_ffn(model_name, spec) else N_PROBES
            probes = all_probes[:n]

            m = _measure_gain_ratio(
                backend, model, tokenizer, probes, model_name, spec,
            )
            measurements.append(m)

            label = f"  layer.{spec.layer_idx}.{spec.weight_suffix}"
            print(
                f"{label:40s} act_rank={m.activation_rank:3d}, "
                f"kfac_null={m.kfac_null_rank:>8d}, "
                f"act_null={m.act_null_rank:>8d}, "
                f"gain={m.gain_ratio:.3f}"
            )

        gains = [m.gain_ratio for m in measurements if math.isfinite(m.gain_ratio)]
        if gains:
            gains_sorted = sorted(gains)
            median_gain = gains_sorted[len(gains_sorted) // 2]
        else:
            median_gain = 0.0

        print(f"  MEDIAN GAIN: {median_gain:.3f}")
        results_by_model[model_name] = measurements
        model_medians[model_name] = median_gain

        # Free model memory
        del model, tokenizer
        gc.collect()

    # Go/no-go determination
    models_above_110 = sum(1 for g in model_medians.values() if g > 1.10)
    all_below_105 = all(g < 1.05 for g in model_medians.values())

    if models_above_110 >= 2:
        verdict = "GO"
        reason = f"median > 1.10 on {models_above_110}/{len(model_medians)} models"
    elif all_below_105:
        verdict = "NO-GO"
        reason = "< 1.05 everywhere"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"median > 1.10 on {models_above_110}/{len(model_medians)} models (need 2+)"

    print(f"\nGO/NO-GO: {verdict} ({reason})")

    # Serialize
    serializable = {
        "experiment": "1_gain_ratio",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "reason": reason,
        "model_medians": model_medians,
        "measurements": {
            model_name: [
                {
                    "weight_name": m.weight_name,
                    "layer_idx": m.layer_idx,
                    "weight_suffix": m.weight_suffix,
                    "n_probes": m.n_probes,
                    "activation_rank": m.activation_rank,
                    "act_null_rank": m.act_null_rank,
                    "kfac_null_rank": m.kfac_null_rank,
                    "gain_ratio": m.gain_ratio,
                }
                for m in measurements
            ]
            for model_name, measurements in results_by_model.items()
        },
    }
    return serializable


# ---------------------------------------------------------------------------
# Experiment 2: K-FAC vs Full Jacobian
# ---------------------------------------------------------------------------

@dataclass
class JacobianComparisonMeasurement:
    weight_name: str
    layer_idx: int
    weight_suffix: str
    kfac_signal_rank: int
    jacobian_signal_rank: int
    principal_angles_deg: list[float]
    median_angle_deg: float
    max_angle_deg: float


def _build_kfac_signal_basis(
    diagnostic: KFACDiagnosticResult,
    backend: Any,
    max_signal_dirs: int = 50,
) -> Any:
    """Build K-FAC signal-space basis in flattened weight coordinates.

    Signal directions are where null_mask[i,j] == False. Each such direction
    is vec(S_eigvecs[:,i] @ A_eigvecs[:,j]^T) in flattened weight space.
    We take the top max_signal_dirs by Kronecker eigenvalue.

    Returns:
        Orthonormal basis [k, D] where D = out_dim * in_dim.
    """
    import mlx.core as mx

    b = backend
    out_dim = diagnostic.out_dim
    in_dim = diagnostic.in_dim

    # Build Kronecker eigenvalue grid
    S_eigvals = diagnostic.output_gradient_eigenvalues  # [out_dim] descending
    A_eigvals = diagnostic.activation_eigenvalues  # [in_dim] descending
    S_col = b.reshape(S_eigvals, (out_dim, 1))
    A_row = b.reshape(A_eigvals, (1, in_dim))
    kron_eigvals = b.matmul(S_col, A_row)  # [out_dim, in_dim]
    b.eval(kron_eigvals)

    # Signal mask: NOT null
    null_mask_np = diagnostic.kron_null_mask.tolist()
    kron_np = kron_eigvals.tolist()

    # Collect signal direction indices with their eigenvalues
    signal_dirs: list[tuple[float, int, int]] = []
    for i in range(out_dim):
        for j in range(in_dim):
            if not null_mask_np[i][j]:
                signal_dirs.append((kron_np[i][j], i, j))

    if not signal_dirs:
        logger.warning("No signal directions found — all directions are null.")
        return mx.zeros((0, out_dim * in_dim))

    # Sort by eigenvalue descending, take top-k
    signal_dirs.sort(key=lambda x: -x[0])
    signal_dirs = signal_dirs[:max_signal_dirs]

    # Build basis vectors: vec(S_eigvecs[:,i] @ A_eigvecs[:,j]^T)
    S_eigvecs = diagnostic.output_gradient_eigenvectors  # [out_dim, out_dim]
    A_eigvecs = diagnostic.activation_eigenvectors  # [in_dim, in_dim]

    basis_vectors = []
    for _, i, j in signal_dirs:
        # s_i = S_eigvecs[:, i], a_j = A_eigvecs[:, j]
        s_i = b.take(S_eigvecs, mx.array(i), axis=1)  # [out_dim]
        a_j = b.take(A_eigvecs, mx.array(j), axis=1)  # [in_dim]
        # Outer product → [out_dim, in_dim], then flatten
        outer = b.reshape(s_i, (out_dim, 1)) * b.reshape(a_j, (1, in_dim))
        basis_vectors.append(b.reshape(outer, (-1,)))

    basis = mx.stack(basis_vectors)  # [k, D]
    mx.eval(basis)

    # QR-orthogonalize via SVD: basis = U @ diag(S) @ Vt
    # For [k, D] with k < D: Vt rows are orthonormal in D-space
    svd_result = b.svd(basis, compute_uv=True)
    Vt = svd_result[2]  # [k, D]
    S_vals = svd_result[1]  # [k]
    b.eval(Vt, S_vals)

    eps = float(b.finfo(S_vals.dtype).eps)
    max_s = float(b.to_scalar(b.max(S_vals)))
    rank_thr = max_s * max(eps * max(basis.shape), 1e-10)
    s_list = S_vals.tolist()
    rank = sum(1 for s in s_list if s > rank_thr)

    if rank == 0:
        return mx.zeros((0, out_dim * in_dim))

    # Take top-rank rows of Vt
    ortho_basis = Vt[:rank]  # [rank, D]
    mx.eval(ortho_basis)
    return ortho_basis


def _build_jacobian_signal_basis(
    gradient_matrix: Any,
    backend: Any,
) -> Any:
    """Build Jacobian signal-space basis in flattened weight coordinates.

    From G [N, D], eigendecompose GG^T to get signal eigenvectors in
    probe-space, then convert to weight-space: signal_basis = G^T @ eigvecs[:,:k],
    then QR-orthogonalize.

    Returns:
        Orthonormal basis [k, D].
    """
    import mlx.core as mx

    b = backend
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        svd_rank_threshold,
    )

    G = b.array(gradient_matrix)
    n_probes = int(G.shape[0])
    D = int(G.shape[1])

    # GG^T ∈ [N, N]
    GGt = b.matmul(G, b.transpose(G))
    b.eval(GGt)

    eigvals, eigvecs = b.eigh(GGt)
    b.eval(eigvals, eigvecs)

    # Sort descending
    idx = b.arange(n_probes - 1, -1, -1)
    eigvals = b.take(eigvals, idx, axis=0)
    eigvecs = b.take(eigvecs, idx, axis=1)
    b.eval(eigvals, eigvecs)

    eps = machine_epsilon(b, GGt)
    eigvals_pos = b.maximum(eigvals, eps)
    max_eig = float(b.to_scalar(b.max(eigvals_pos)))
    max_eig_safe = max(max_eig, eps)

    rank_scale = svd_rank_threshold(b, eigvals_pos, n_probes)
    rank_threshold = max_eig_safe * rank_scale
    rank_mask = eigvals_pos > rank_threshold
    rank_count = int(round(float(b.to_scalar(b.sum(
        b.astype(rank_mask, eigvals.dtype)
    )))))
    rank_count = max(0, min(rank_count, n_probes))

    if rank_count == 0:
        return mx.zeros((0, D))

    # Signal eigenvectors in probe-space: eigvecs[:, :rank_count]
    # Convert to weight-space: W_basis = G^T @ eigvecs[:, :rank_count]  → [D, rank]
    signal_eigvecs = eigvecs[:, :rank_count]  # [N, rank]
    W_basis = b.matmul(b.transpose(G), signal_eigvecs)  # [D, rank]
    b.eval(W_basis)

    # QR-orthogonalize via SVD of W_basis^T = [rank, D]
    W_basis_t = b.transpose(W_basis)  # [rank, D]
    svd_result = b.svd(W_basis_t, compute_uv=True)
    Vt = svd_result[2]  # [rank, D]
    S_vals = svd_result[1]
    b.eval(Vt, S_vals)

    eps_f = float(b.finfo(S_vals.dtype).eps)
    max_s = float(b.to_scalar(b.max(S_vals)))
    s_thr = max_s * max(eps_f * max(W_basis_t.shape), 1e-10)
    s_list = S_vals.tolist()
    final_rank = sum(1 for s in s_list if s > s_thr)

    if final_rank == 0:
        return mx.zeros((0, D))

    ortho_basis = Vt[:final_rank]  # [final_rank, D]
    mx.eval(ortho_basis)
    return ortho_basis


def experiment_2_jacobian_comparison(
    backend: Any,
) -> dict[str, Any]:
    """Experiment 2: K-FAC vs Full Jacobian agreement.

    GO if median principal angle < 15°.
    NO-GO if > 30°.

    LFM2-350M only (full Jacobian fits in memory).
    """
    model_name = "LFM2-350M"
    model_path = MODELS[model_name]
    probes = load_probes(N_PROBES)

    # 3 layers: attention, FFN-up, FFN-down
    specs = [
        WeightSpec(5, "self_attn.q_proj", "hidden"),
        WeightSpec(8, "feed_forward.w1", "hidden"),
        WeightSpec(12, "feed_forward.w2", "intermediate"),
    ]

    print("\n" + "=" * 60)
    print("=== EXPERIMENT 2: K-FAC vs Full Jacobian ===")
    print("=" * 60)
    print(f"Model: {model_name}")

    logger.info("Loading %s from %s", model_name, model_path)
    model, tokenizer = backend.load_model(model_path)

    measurements: list[JacobianComparisonMeasurement] = []

    for spec in specs:
        wname = get_weight_name(model_name, spec)
        wshape = get_weight_shape(model, wname)
        label = f"layer.{spec.layer_idx}.{spec.weight_suffix}"

        logger.info("  Computing %s (shape=%s)...", label, wshape)
        t0 = time.time()

        # Collect input activations
        acts = collect_input_activations(backend, model, tokenizer, probes, spec)

        # Compute per-probe weight gradients (this is G: the behavior Jacobian)
        G = backend.compute_per_probe_gradients(model, tokenizer, probes, wname)
        logger.info("    G shape: %s (%.1fs)", G.shape, time.time() - t0)

        # K-FAC diagnostic
        diagnostic = compute_kfac_diagnostic_from_weight_gradients(
            input_activations=acts,
            per_probe_weight_gradients=G,
            weight_shape=wshape,
            backend=backend,
        )

        # Build K-FAC signal basis
        kfac_basis = _build_kfac_signal_basis(diagnostic, backend)
        kfac_rank = int(kfac_basis.shape[0])
        logger.info("    K-FAC signal rank: %d", kfac_rank)

        # Build Jacobian signal basis
        jac_basis = _build_jacobian_signal_basis(G, backend)
        jac_rank = int(jac_basis.shape[0])
        logger.info("    Jacobian signal rank: %d", jac_rank)

        # Compute principal angles between signal subspaces
        if kfac_rank > 0 and jac_rank > 0:
            grassmann = compute_grassmann_distance(kfac_basis, jac_basis, backend)
            angles_rad = grassmann.principal_angles.tolist()
            angles_deg = [a * 180.0 / math.pi for a in angles_rad]
            angles_deg_sorted = sorted(angles_deg)
            median_angle = angles_deg_sorted[len(angles_deg_sorted) // 2]
            max_angle = max(angles_deg)
        else:
            angles_deg = []
            median_angle = 90.0  # Degenerate: no overlap
            max_angle = 90.0

        m = JacobianComparisonMeasurement(
            weight_name=wname,
            layer_idx=spec.layer_idx,
            weight_suffix=spec.weight_suffix,
            kfac_signal_rank=kfac_rank,
            jacobian_signal_rank=jac_rank,
            principal_angles_deg=angles_deg,
            median_angle_deg=median_angle,
            max_angle_deg=max_angle,
        )
        measurements.append(m)

        print(f"  {label}:")
        print(f"    kfac_signal_rank={kfac_rank}, jacobian_signal_rank={jac_rank}")
        if angles_deg:
            top5 = angles_deg_sorted[:5]
            print(f"    principal_angles (deg, first 5): [{', '.join(f'{a:.1f}' for a in top5)}]")
        print(f"    median_angle={median_angle:.1f}°, max_angle={max_angle:.1f}°")

        del acts, G, diagnostic, kfac_basis, jac_basis
        gc.collect()
        logger.info("    done (%.1fs total)", time.time() - t0)

    # Overall median
    all_medians = [m.median_angle_deg for m in measurements]
    overall_median = sorted(all_medians)[len(all_medians) // 2]

    print(f"\nMEDIAN PRINCIPAL ANGLE: {overall_median:.1f}°")

    if overall_median < 15.0:
        verdict = "GO"
        reason = f"median {overall_median:.1f}° < 15°"
    elif overall_median > 30.0:
        verdict = "NO-GO"
        reason = f"median {overall_median:.1f}° > 30°"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"median {overall_median:.1f}° (between 15° and 30°)"

    print(f"GO/NO-GO: {verdict} ({reason})")

    del model, tokenizer
    gc.collect()

    serializable = {
        "experiment": "2_jacobian_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_name,
        "verdict": verdict,
        "reason": reason,
        "overall_median_angle_deg": overall_median,
        "measurements": [
            {
                "weight_name": m.weight_name,
                "layer_idx": m.layer_idx,
                "weight_suffix": m.weight_suffix,
                "kfac_signal_rank": m.kfac_signal_rank,
                "jacobian_signal_rank": m.jacobian_signal_rank,
                "principal_angles_deg": m.principal_angles_deg,
                "median_angle_deg": m.median_angle_deg,
                "max_angle_deg": m.max_angle_deg,
            }
            for m in measurements
        ],
    }
    return serializable


# ---------------------------------------------------------------------------
# Experiment 3: Training Curvature Alignment
# ---------------------------------------------------------------------------

@dataclass
class CurvatureAlignmentMeasurement:
    weight_name: str
    layer_idx: int
    weight_suffix: str
    delta_frobenius: float
    top_10pct_fraction: float
    top_25pct_fraction: float
    null_fraction: float
    kfac_gain_ratio: float


def experiment_3_curvature_alignment(
    backend: Any,
) -> dict[str, Any]:
    """Experiment 3: Training Curvature Alignment.

    Compute K-FAC factors from base model, train LoRA, measure adapter delta
    alignment with curvature directions.

    GO to gradient projection if > 20% in top-10%.
    NO-GO (Cayley+MASS sufficient) if < 5%.
    """
    import subprocess

    model_name = "LFM2-350M"
    model_path = MODELS[model_name]
    probes = load_probes(N_PROBES)

    # Weights to analyze
    specs = [
        WeightSpec(2, "self_attn.q_proj", "hidden"),
        WeightSpec(5, "feed_forward.w1", "hidden"),
        WeightSpec(8, "self_attn.q_proj", "hidden"),
        WeightSpec(10, "feed_forward.w2", "intermediate"),
        WeightSpec(14, "self_attn.q_proj", "hidden"),
    ]

    print("\n" + "=" * 60)
    print("=== EXPERIMENT 3: Training Curvature Alignment ===")
    print("=" * 60)
    print(f"Model: {model_name}, trained on benchmark_train.jsonl")

    # Step 1: Compute K-FAC factors from base model
    print("\nStep 1: Computing K-FAC factors from base model...")
    logger.info("Loading base model for K-FAC factor computation...")
    model, tokenizer = backend.load_model(model_path)

    factors_by_weight: dict[str, KFACFactors] = {}
    weight_shapes: dict[str, tuple[int, int]] = {}

    for spec in specs:
        wname = get_weight_name(model_name, spec)
        wshape = get_weight_shape(model, wname)
        weight_shapes[wname] = wshape
        label = f"layer.{spec.layer_idx}.{spec.weight_suffix}"

        logger.info("  K-FAC factors for %s...", label)
        t0 = time.time()

        acts = collect_input_activations(backend, model, tokenizer, probes, spec)
        grads = backend.compute_per_probe_gradients(model, tokenizer, probes, wname)

        diagnostic = compute_kfac_diagnostic_from_weight_gradients(
            input_activations=acts,
            per_probe_weight_gradients=grads,
            weight_shape=wshape,
            backend=backend,
        )
        factors = factors_from_diagnostic(diagnostic)
        factors_by_weight[wname] = factors

        print(f"  {label}: gain={factors.gain_ratio:.3f} ({time.time()-t0:.1f}s)")

        del acts, grads, diagnostic
        gc.collect()

    del model, tokenizer
    gc.collect()

    # Step 2: Train with NB-LoRA
    print("\nStep 2: Training with NB-LoRA...")
    adapter_dir = RESULTS_DIR / "exp3_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = adapter_dir / "adapters.safetensors"

    if adapter_path.exists():
        print(f"  Found existing adapter at {adapter_path}, skipping training.")
    else:
        train_cmd = [
            "poetry", "run", "mc", "train", "run",
            "--model", model_path,
            "--dataset", "data/training/benchmark_train.jsonl",
            "--output", str(adapter_dir),
        ]
        print(f"  Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  TRAINING FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}")
            return {
                "experiment": "3_curvature_alignment",
                "verdict": "ERROR",
                "reason": f"Training failed: {result.stderr[-200:]}",
            }
        print("  Training complete.")

    # Find adapter file (may be adapters.safetensors or in a subdirectory)
    adapter_file = None
    for candidate in [
        adapter_dir / "adapters.safetensors",
        adapter_dir / "adapter" / "adapters.safetensors",
    ]:
        if candidate.exists():
            adapter_file = candidate
            break

    # Also search recursively
    if adapter_file is None:
        safetensor_files = list(adapter_dir.rglob("adapters.safetensors"))
        if safetensor_files:
            adapter_file = safetensor_files[0]

    if adapter_file is None:
        # Check if the training output went somewhere else
        print(f"  No adapter found in {adapter_dir}. Listing contents:")
        for p in sorted(adapter_dir.rglob("*")):
            print(f"    {p}")
        return {
            "experiment": "3_curvature_alignment",
            "verdict": "ERROR",
            "reason": f"No adapter file found in {adapter_dir}",
        }

    print(f"  Adapter: {adapter_file}")

    # Step 3: Post-hoc curvature analysis
    print("\nStep 3: Post-hoc curvature analysis...")

    # Load trained model to extract adapter deltas
    logger.info("Loading trained model with adapter...")
    trained_model, _ = backend.load_model(model_path, adapter_path=str(adapter_file.parent))
    base_model, _ = backend.load_model(model_path)

    from mlx.utils import tree_flatten
    import mlx.core as mx

    trained_params = dict(tree_flatten(trained_model.parameters()))
    base_params = dict(tree_flatten(base_model.parameters()))

    measurements: list[CurvatureAlignmentMeasurement] = []

    for spec in specs:
        wname = get_weight_name(model_name, spec)
        factors = factors_by_weight[wname]
        label = f"layer.{spec.layer_idx}.{spec.weight_suffix}"

        if wname not in trained_params or wname not in base_params:
            logger.warning("  %s not found in parameters, skipping.", wname)
            continue

        # Compute delta = trained_weight - base_weight
        trained_w = trained_params[wname]
        base_w = base_params[wname]
        delta = trained_w - base_w
        mx.eval(delta)

        delta_norm = float(mx.sqrt(mx.sum(delta * delta)).item())
        if delta_norm < 1e-12:
            logger.info("  %s: zero delta (no LoRA applied here), skipping.", label)
            continue

        # Curvature alignment
        curv = compute_curvature_alignment(
            delta_weight=delta,
            factors=factors,
            layer_name=label,
            backend=backend,
        )

        m = CurvatureAlignmentMeasurement(
            weight_name=wname,
            layer_idx=spec.layer_idx,
            weight_suffix=spec.weight_suffix,
            delta_frobenius=curv.delta_frobenius,
            top_10pct_fraction=curv.top_10pct_fraction,
            top_25pct_fraction=curv.top_25pct_fraction,
            null_fraction=curv.null_fraction,
            kfac_gain_ratio=curv.kfac_gain_ratio,
        )
        measurements.append(m)

        print(
            f"  {label}:\n"
            f"    delta_frob={curv.delta_frobenius:.4f}, "
            f"top10={curv.top_10pct_fraction*100:.1f}%, "
            f"top25={curv.top_25pct_fraction*100:.1f}%, "
            f"null={curv.null_fraction*100:.1f}%"
        )

    del trained_model, base_model, trained_params, base_params
    gc.collect()

    if not measurements:
        print("\n  No LoRA deltas found. The adapter may not have modified these layers.")
        return {
            "experiment": "3_curvature_alignment",
            "verdict": "ERROR",
            "reason": "No non-zero adapter deltas found for measured weights.",
        }

    # Overall statistics
    top10_fracs = sorted(m.top_10pct_fraction for m in measurements)
    median_top10 = top10_fracs[len(top10_fracs) // 2]

    print(f"\nMEDIAN TOP-10%: {median_top10*100:.1f}%")

    if median_top10 > 0.20:
        verdict = "NEEDS PROJECTION"
        reason = f"median {median_top10*100:.1f}% > 20% in top-10% curvature"
    elif median_top10 < 0.05:
        verdict = "CAYLEY+MASS SUFFICIENT"
        reason = f"median {median_top10*100:.1f}% < 5% in top-10% curvature"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"median {median_top10*100:.1f}% (between 5% and 20%)"

    print(f"GO/NO-GO: {verdict} ({reason})")

    serializable = {
        "experiment": "3_curvature_alignment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_name,
        "verdict": verdict,
        "reason": reason,
        "median_top_10pct": median_top10,
        "measurements": [
            {
                "weight_name": m.weight_name,
                "layer_idx": m.layer_idx,
                "weight_suffix": m.weight_suffix,
                "delta_frobenius": m.delta_frobenius,
                "top_10pct_fraction": m.top_10pct_fraction,
                "top_25pct_fraction": m.top_25pct_fraction,
                "null_fraction": m.null_fraction,
                "kfac_gain_ratio": m.kfac_gain_ratio,
            }
            for m in measurements
        ],
    }
    return serializable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_results(results: dict[str, Any], filename: str) -> Path:
    """Save results JSON to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="K-FAC Validation Experiments — 3 go/no-go tests on real models.",
    )
    parser.add_argument(
        "--experiment",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="Models to use for Experiment 1.",
    )
    args = parser.parse_args()

    # Ensure results directory exists before setting up file handler
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(RESULTS_DIR / "kfac_validation.log"),
        ],
    )

    # Initialize backend
    backend = initialize_default_backend()
    logger.info("Backend initialized: %s", type(backend).__name__)

    experiments = (
        ["1", "2", "3"] if args.experiment == "all"
        else [args.experiment]
    )

    for exp in experiments:
        if exp == "1":
            # Validate requested models exist
            valid_models = [m for m in args.models if m in MODELS]
            if not valid_models:
                print(f"ERROR: No valid models in {args.models}. Available: {list(MODELS.keys())}")
                sys.exit(1)
            results = experiment_1_gain_ratio(backend, valid_models)
            _save_results(results, "experiment_1_gain_ratio.json")

        elif exp == "2":
            results = experiment_2_jacobian_comparison(backend)
            _save_results(results, "experiment_2_jacobian_comparison.json")

        elif exp == "3":
            results = experiment_3_curvature_alignment(backend)
            _save_results(results, "experiment_3_curvature_alignment.json")


if __name__ == "__main__":
    main()
