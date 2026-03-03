#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf
#
# Exp 2: Gradient Effective Rank Probe
#
# Measures the minimum adapter rank needed per task by projecting early-step
# gradients onto the null-space basis V_tail and computing the rank of the
# projected gradient.
#
#   C_t = G_A_t @ V_tail   [r, tail_dims]
#   grad_rank_t = rank_eps(SVD(C_t))
#   cumulative_union_rank = rank_eps(SVD(stack([C_1, ..., C_T])))
#
# The cumulative union rank gives the minimum adapter rank that captures all
# gradient information across n_probe_steps training steps. Then:
#
#   max_nights = floor(capacity_total / total_cumulative_union_rank)
#
# is the geometric lower bound on consecutive nightly consolidation sessions.
#
# Default model: Qwen3-1.7B bf16 (per AGENTS.md smallest-first policy; 8B is
# for production confidence, not discovery).

import argparse
import json
import math
from pathlib import Path

from modelcypher.backends import initialize_default_backend
from modelcypher.cli.composition import (
    get_capacity_analysis_service,
    get_dataset_training_service,
    get_model_loader,
)

# Smallest model that demonstrates the property under test.
# Per AGENTS.md: "Do NOT run 8B models for research iteration."
MODEL_PATH_DEFAULT = "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-2B-bf16"
TASK_PATH_DEFAULT = "data/training/shards/S1.jsonl"
OUTPUT_ROOT_DEFAULT = Path("results/continual_learning/exp2")

# TODO: derive minimum probe steps from gradient effective rank convergence rate
N_PROBE_STEPS_DEFAULT = 5


# ---------------------------------------------------------------------------
# Helpers copied verbatim from exp1 (shared logic; no shared module yet)
# ---------------------------------------------------------------------------

def _svd_singular_values(backend, tensor):
    """Return singular values array (float32) for a 2D tensor."""
    tensor_f32 = backend.astype(tensor, "float32")
    S = backend.svd(tensor_f32, compute_uv=False)
    backend.eval(S)
    return S


def _spectral_norm(backend, tensor) -> float:
    """Compute spectral norm (largest singular value) of a tensor."""
    S = _svd_singular_values(backend, tensor)
    return float(backend.to_scalar(backend.max(S)))


def _rank_eps(s_list: list[float], eps: float) -> int:
    """Count singular values above IEEE 754-derived noise threshold.

    Threshold = σ_1 * sqrt(eps). Below this is indistinguishable from
    numerical noise at the working precision.
    """
    if not s_list or s_list[0] <= 0:
        return 0
    threshold = s_list[0] * math.sqrt(eps)
    return sum(1 for s in s_list if s > threshold)


def _read_quant_config(model_path: str) -> dict:
    """Read quantization parameters from model config.json."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open() as fh:
            cfg = json.load(fh)
        return cfg.get("quantization", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_svd_weight_tensor(backend, model_loader, model_path, layer_name, tensor, quant_cfg):
    """Return a float-domain weight tensor suitable for SVD.

    For quantized MLX weights (packed uint), this finds matching
    .scales/.biases and dequantizes so spectral norms are in the same
    [out, in] space as adapter deltas.
    """
    dtype_str = str(getattr(tensor, "dtype", "")).lower()
    if "uint" not in dtype_str or "float" in dtype_str:
        return tensor

    scales_key = layer_name.replace(".weight", ".scales")
    biases_key = layer_name.replace(".weight", ".biases")
    scales = None
    biases = None
    for name, q_tensor in model_loader.iter_weights(model_path):
        if name == biases_key:
            biases = q_tensor
        elif name == scales_key:
            scales = q_tensor

    if scales is None:
        raise RuntimeError(
            f"Quantized reference weight {layer_name!r} has no scales tensor. "
            "Cannot compute tail basis from packed geometry."
        )

    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) != 2:
        raise RuntimeError(
            f"Reference weight {layer_name!r} is not 2D; cannot compute tail basis."
        )

    bits = quant_cfg.get("bits", 4)
    in_full = shape[1] * (32 // bits)
    group_size = quant_cfg.get("group_size") or (in_full // scales.shape[1])
    mode = quant_cfg.get("mode", "affine")
    tensor = backend.dequantize(tensor, scales, biases,
                                group_size=group_size, bits=bits, mode=mode)
    backend.eval(tensor)
    return tensor


def _precompute_tail_bases(backend, model_loader, model_path: str,
                           adapted_keys: set, eps: float) -> dict:
    """Precompute V_tail basis for each adapted layer from base model SVD.

    For each 2D weight matrix that has an adapter:
      W [out, in] = U @ diag(S) @ V^T
      tail_dims = full_rank - rank_eps(S)
      V_tail = last tail_dims columns of V [in, tail_dims]

    For 4-bit quantized models the weight is dequantized before SVD.

    Returns dict mapping layer_name -> (V_tail, tail_dims).
    Layers with tail_dims == 0 are excluded.
    """
    quant_cfg = _read_quant_config(model_path)

    needed = set(adapted_keys)
    needed |= {k.replace(".weight", ".scales") for k in adapted_keys}
    needed |= {k.replace(".weight", ".biases") for k in adapted_keys}

    collected: dict = {}
    for layer_name, tensor in model_loader.iter_weights(model_path):
        if layer_name in needed:
            collected[layer_name] = tensor

    tail_bases = {}

    for layer_name in sorted(adapted_keys):
        tensor = collected.get(layer_name)
        if tensor is None:
            continue

        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue

        tensor = _resolve_svd_weight_tensor(
            backend, model_loader, model_path, layer_name, tensor, quant_cfg
        )

        tensor_f32 = backend.astype(tensor, "float32")
        U, S, Vt = backend.svd(tensor_f32, compute_uv=True)
        backend.eval(S)
        backend.eval(Vt)

        s_list = backend.tolist(S)
        full_rank = len(s_list)
        used_rank = _rank_eps(s_list, eps)
        tail_dims = full_rank - used_rank

        if tail_dims > 0:
            V_tail = backend.transpose(backend.astype(Vt, "float32")[used_rank:])
            backend.eval(V_tail)
            tail_bases[layer_name] = (V_tail, tail_dims)
            print(f"  {layer_name}: rank={used_rank}/{full_rank}, tail_dims={tail_dims}")

    return tail_bases


# ---------------------------------------------------------------------------
# New: gradient rank measurement
# ---------------------------------------------------------------------------

def _get_grad_by_key(grad_tree, target_key):
    """Find a gradient tensor by its dotted-path key via mlx tree_flatten."""
    try:
        from mlx.utils import tree_flatten as mlx_tree_flatten
    except ImportError:
        return None
    for key, value in mlx_tree_flatten(grad_tree):
        if key == target_key:
            return value
    return None


def _build_grad_rank_hook(backend, tail_bases: dict, n_probe_steps: int, eps: float):
    """Build a gradient hook that records per-step projected gradient ranks.

    Returns (hook, step_data) where:
      - hook is a Callable suitable for train_from_dataset(gradient_hook=...)
      - step_data is a list that will be populated in-place: one entry per
        probe step, each containing per-layer grad_rank and the C matrix.

    The hook records for steps 0..(n_probe_steps-1) then becomes a no-op.
    It does NOT modify the gradient tree — measurement only.
    """
    step_data: list[dict] = []
    step_count = [0]

    def hook(grad_tree):
        t = step_count[0]
        step_count[0] += 1

        if t >= n_probe_steps:
            return grad_tree  # recording window closed

        per_layer: dict = {}
        for layer_name, (V_tail, tail_dims) in tail_bases.items():
            # NBLoRALinear gradient is at the A_tilde parameter path
            a_tilde_key = layer_name.replace(".weight", ".A_tilde")
            G_A = _get_grad_by_key(grad_tree, a_tilde_key)

            if G_A is None:
                per_layer[layer_name] = {"tail_dims": tail_dims, "grad_rank": 0, "C": None}
                continue

            # Project gradient onto null space: C = G_A @ V_tail [r, tail_dims]
            G_A_f32 = backend.astype(G_A, "float32")
            C = backend.matmul(G_A_f32, V_tail)
            backend.eval(C)

            S = backend.svd(C, compute_uv=False)
            backend.eval(S)
            s_list = backend.tolist(S)
            grad_rank = _rank_eps(s_list, eps)

            per_layer[layer_name] = {"tail_dims": tail_dims, "grad_rank": grad_rank, "C": C}

        step_data.append({"step": t, "per_layer": per_layer})
        return grad_tree

    return hook, step_data


def _compute_cumulative_union_rank(backend, step_data: list, tail_bases: dict, eps: float) -> dict:
    """Compute the cumulative union rank across all probe steps per layer.

    Stacks C_1, C_2, ..., C_T for each layer and computes
    rank_eps(SVD(stack)) — the number of unique null-space directions
    explored by the gradient across all probe steps.

    Returns dict[layer_name -> cumulative_rank].
    """
    result: dict = {}
    for layer_name in tail_bases:
        c_matrices = [
            s["per_layer"][layer_name]["C"]
            for s in step_data
            if layer_name in s["per_layer"] and s["per_layer"][layer_name]["C"] is not None
        ]
        if not c_matrices:
            result[layer_name] = 0
            continue

        c_f32 = [backend.astype(c, "float32") for c in c_matrices]
        C_stack = backend.concatenate(c_f32, axis=0)
        backend.eval(C_stack)

        S = backend.svd(C_stack, compute_uv=False)
        backend.eval(S)
        s_list = backend.tolist(S)
        result[layer_name] = _rank_eps(s_list, eps)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp 2: Gradient Effective Rank Probe — minimum rank per task"
    )
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT,
                        help="Path to base model (default: Qwen3-1.7B bf16)")
    parser.add_argument("--model-id", default=None,
                        help="Human-readable model identifier (defaults to directory name)")
    parser.add_argument("--task-path", default=TASK_PATH_DEFAULT,
                        help="JSONL task dataset to probe gradients on")
    parser.add_argument("--n-probe-steps", type=int, default=N_PROBE_STEPS_DEFAULT,
                        help="Number of gradient steps to record (default: 5)")
    parser.add_argument("--run-id", default="R_Q1_grad",
                        help="Run identifier for output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_id = args.model_id or Path(args.model_path).name
    run_dir = args.output_root.expanduser().resolve() / args.run_id / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    backend = initialize_default_backend()
    dataset_service = get_dataset_training_service()
    model_loader = get_model_loader()
    capacity_service = get_capacity_analysis_service()
    eps = float(backend.finfo().eps)

    model_path = str(Path(args.model_path).expanduser().resolve())
    print(f"=== Exp 2 Gradient Rank Probe | run={args.run_id} | model={model_id} | "
          f"n_probe_steps={args.n_probe_steps} | seed={args.seed} ===")

    # 1. Capacity analysis — get adapted_keys and total capacity
    print("\nProfiling base model capacity...")
    capacity_report = capacity_service.analyze(
        model_path=model_path,
        checkpoint_path=run_dir / "capacity_checkpoint.json",
        resume=True,
    )
    # Use all layer names as candidates — _precompute_tail_bases is authoritative.
    # For packed 4-bit models the capacity service SVDs the packed uint tensor and
    # reports null_space_dim_f32=0 for all layers, so filtering on that value here
    # would produce an empty adapted_keys set and a capacity_total of 0 (wrong).
    adapted_keys = {r.layer_name for r in capacity_report.layer_reports}
    print(f"Candidate layers: {len(adapted_keys)}")

    # 2. Precompute V_tail bases (dequantizes packed weights, authoritative source)
    print("\nPrecomputing tail bases from base model SVD...")
    tail_bases = _precompute_tail_bases(backend, model_loader, model_path, adapted_keys, eps)
    # Derive capacity_total from dequantized tail_bases, not from capacity report.
    capacity_total = sum(td for _, td in tail_bases.values())
    print(f"Tail bases computed for {len(tail_bases)} layers, capacity_total: {capacity_total} tail dims")

    # 3. Build gradient rank hook
    hook, step_data = _build_grad_rank_hook(backend, tail_bases, args.n_probe_steps, eps)

    # 4. Run training with hook — adapter output not needed
    print(f"\nRunning {args.n_probe_steps} probe steps on {args.task_path}...")
    try:
        dataset_service.train_from_dataset(
            model_path=model_path,
            dataset_path=str(args.task_path),
            seed=args.seed,
            gradient_hook=hook,
            no_save=True,
        )
    except Exception as e:
        print(f"Training error: {e}")
        if not step_data:
            raise

    print(f"Recorded {len(step_data)} gradient steps")

    # 5. Cumulative union rank per layer
    print("\nComputing cumulative union rank...")
    cumulative_ranks = _compute_cumulative_union_rank(backend, step_data, tail_bases, eps)
    total_cumulative_rank = sum(cumulative_ranks.values())
    max_nights: int | float = (
        capacity_total // total_cumulative_rank
        if total_cumulative_rank > 0 else float("inf")
    )

    # 6. Build output
    per_layer_out = {}
    for name, (V_tail, tail_dims) in tail_bases.items():
        per_step_ranks = [
            s["per_layer"].get(name, {}).get("grad_rank", 0)
            for s in step_data
        ]
        per_layer_out[name] = {
            "tail_dims": tail_dims,
            "per_step_grad_rank": per_step_ranks,
            "cumulative_union_rank": cumulative_ranks.get(name, 0),
        }

    output = {
        "run_id": args.run_id,
        "model_id": model_id,
        "model_path": model_path,
        "task_path": str(args.task_path),
        "n_probe_steps": args.n_probe_steps,
        "eps": eps,
        "capacity_total": capacity_total,
        "per_layer": per_layer_out,
        "summary": {
            "total_tail_dims": capacity_total,
            "total_cumulative_union_rank": total_cumulative_rank,
            "max_nights_lower_bound": max_nights,
            "grad_rank_fraction": (
                total_cumulative_rank / capacity_total if capacity_total > 0 else 0.0
            ),
        },
    }

    out_file = run_dir / "exp2_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_file}")

    s = output["summary"]
    print(f"\n=== Summary ===")
    print(f"  Capacity total:           {s['total_tail_dims']} tail dims")
    print(f"  Cumulative grad rank:     {s['total_cumulative_union_rank']} dims")
    print(f"  Grad rank fraction:       {s['grad_rank_fraction']:.2%}")
    print(f"  Max nights (lower bound): {s['max_nights_lower_bound']}")


if __name__ == "__main__":
    main()
