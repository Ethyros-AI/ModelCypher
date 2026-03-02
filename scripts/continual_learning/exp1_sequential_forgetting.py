#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

import argparse
import json
import math
import random
from pathlib import Path

from modelcypher.backends import initialize_default_backend
from modelcypher.cli.composition import (
    get_capacity_analysis_service,
    get_dataset_training_service,
    get_model_loader,
)
from modelcypher.core.domain.continual_learning_metrics import get_continual_learning_metrics
from modelcypher.core.domain.geometry.cka import compute_linear_cka_gram

OUTPUT_ROOT_DEFAULT = Path("results/continual_learning/exp1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 1: Sequential Forgetting")
    parser.add_argument(
        "--model-path",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--task-datasets",
        nargs="+",
        default=[
            "data/training/shards/S1.jsonl",
            "data/training/shards/S2.jsonl",
            "data/training/shards/S3.jsonl",
            "data/training/shards/S4.jsonl",
            "data/training/shards/S5.jsonl",
            "data/training/shards/S6.jsonl",
            "data/training/shards/S7.jsonl",
            "data/training/shards/S8.jsonl",
        ],
    )
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.0,
        help="Fraction of previous-task samples to mix into each task's training data",
    )
    parser.add_argument(
        "--run-id",
        default="R0",
        help="Run label for output directory (e.g. R1, R2)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Human-readable model label for output JSON (defaults to model path basename)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _extract_adapter_deltas_keyed(backend, adapter_path: Path) -> dict:
    """Extract all LoRA delta pairs from a saved adapter, keyed by base weight name.

    Returns dict mapping base weight key (e.g. 'model.layers.10.self_attn.q_proj.weight')
    to delta array [out, in].
    """
    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        print(f"  WARNING: No adapter file at {adapter_file}")
        return {}

    adapter_weights = backend.load_safetensors(str(adapter_file))
    deltas = {}

    for k in sorted(adapter_weights):
        if ".lora_a" in k:
            b_key = k.replace(".lora_a", ".lora_b")
            if b_key in adapter_weights:
                a = adapter_weights[k]      # [in, r]
                b = adapter_weights[b_key]  # [r, out]
                # delta = (a @ b).T = b.T @ a.T = [out, in] (weight convention)
                delta = backend.transpose(backend.matmul(a, b))
                base_key = k.replace(".lora_a", ".weight")
                deltas[base_key] = delta

    return deltas


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


def _pick_representative_delta(backend, deltas: list):
    """Pick the delta with largest spectral norm for cross-task CKA."""
    best = deltas[0]
    best_norm = 0.0
    for d in deltas:
        n = _spectral_norm(backend, d)
        if n > best_norm:
            best_norm = n
            best = d
    return best


def _read_quant_config(model_path: str) -> dict:
    """Read quantization parameters from model config.json.

    Returns dict with 'bits', 'group_size' keys if present, else empty dict.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open() as fh:
            cfg = json.load(fh)
        return cfg.get("quantization", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _precompute_tail_bases(backend, model_loader, model_path: str,
                           adapted_keys: set[str], eps: float) -> dict:
    """Precompute V_tail basis for each adapted layer from base model SVD.

    For each 2D weight matrix that has an adapter:
      W [out, in] = U @ diag(S) @ V^T
      tail_dims = full_rank - rank_eps(S)
      V_tail = last tail_dims columns of V [in, tail_dims]

    For 4-bit quantized models (MLX packed uint32 dtype), the weight is
    dequantized before SVD. The LoRA delta lives in the unpacked [out, in]
    space; V_tail must be in that same space or delta @ V_tail fails
    dimensionally. Quantization parameters are read from config.json.

    Returns dict mapping layer_name -> (V_tail, tail_dims).
    Layers with tail_dims == 0 are excluded (fully saturated, no null space).
    """
    quant_cfg = _read_quant_config(model_path)

    # Single-pass collection: gather adapted weights and their quant tensors.
    # For quantized layers we need .weight (uint32), .scales, and .biases.
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

        # Detect 4-bit quantized weight: MLX packs 8 4-bit values per uint32,
        # giving shape [out, in/8]. The LoRA delta has shape [out, in] (full),
        # so SVD must operate on the dequantized [out, in] matrix.
        dtype_str = str(getattr(tensor, "dtype", "")).lower()
        if "uint" in dtype_str and "float" not in dtype_str:
            scales = collected.get(layer_name.replace(".weight", ".scales"))
            biases = collected.get(layer_name.replace(".weight", ".biases"))
            if scales is None:
                print(f"  SKIP {layer_name}: quantized weight without scales")
                continue
            bits = quant_cfg.get("bits", 4)
            # group_size: derived from packed shape and scales shape.
            # in_full = in_packed * (32 // bits); n_groups = scales.shape[1]
            # group_size = in_full / n_groups
            in_full = shape[1] * (32 // bits)
            group_size = quant_cfg.get("group_size") or (in_full // scales.shape[1])
            # mx.dequantize(w, scales, biases, group_size, bits) — no mode arg
            tensor = backend.mx.dequantize(tensor, scales, biases, group_size, bits)
            backend.eval(tensor)

        tensor_f32 = backend.astype(tensor, "float32")
        U, S, Vt = backend.svd(tensor_f32, compute_uv=True)
        backend.eval(S)
        backend.eval(Vt)

        s_list = backend.tolist(S)
        full_rank = len(s_list)
        used_rank = _rank_eps(s_list, eps)
        tail_dims = full_rank - used_rank

        if tail_dims > 0:
            # V_tail = last tail_dims columns of V = Vt[used_rank:, :].T
            V_tail = backend.transpose(backend.astype(Vt, "float32")[used_rank:])
            backend.eval(V_tail)
            tail_bases[layer_name] = (V_tail, tail_dims)
            print(f"  {layer_name}: rank={used_rank}/{full_rank}, tail_dims={tail_dims}")

    return tail_bases


def _compute_consumed_dims(backend, deltas_keyed: dict, tail_bases: dict,
                           eps: float) -> tuple[int, dict]:
    """Project adapter deltas into each layer's tail basis, count consumed dims.

    For each adapted layer with tail_dims > 0:
      C_l = delta_l @ V_tail_l   [out, tail_dims]
      consumed_l = rank_eps(C_l)
      energy_ratio_l = ||C_l||_F^2 / ||delta_l||_F^2

    Returns (total_consumed, per_layer_dict).
    """
    total_consumed = 0
    per_layer = {}

    for layer_name, (V_tail, tail_dims) in tail_bases.items():
        if layer_name not in deltas_keyed:
            per_layer[layer_name] = {"tail_dims": tail_dims, "consumed": 0, "energy_ratio": 0.0}
            continue

        delta = deltas_keyed[layer_name]
        delta_f32 = backend.astype(delta, "float32")
        C = backend.matmul(delta_f32, V_tail)
        backend.eval(C)

        S_c = backend.svd(C, compute_uv=False)
        backend.eval(S_c)
        s_list = backend.tolist(S_c)
        consumed = _rank_eps(s_list, eps)

        c_frob_sq = sum(s * s for s in s_list)
        S_delta = _svd_singular_values(backend, delta_f32)
        sd_list = backend.tolist(S_delta)
        d_frob_sq = sum(s * s for s in sd_list)
        energy_ratio = c_frob_sq / d_frob_sq if d_frob_sq > 0 else 0.0

        per_layer[layer_name] = {
            "tail_dims": tail_dims, "consumed": consumed, "energy_ratio": energy_ratio,
        }
        total_consumed += consumed

    return total_consumed, per_layer


def _incremental_new_dims(backend, all_projections: list, tail_bases: dict,
                          eps: float) -> list[int]:
    """Compute incremental new null-space dimensions consumed per task.

    incremental_new_dims_t = rank([C_1..C_t]) - rank([C_1..C_{t-1}])
    Summed across layers.
    """
    if not all_projections:
        return []

    incremental = []
    prev_total_rank = 0

    for t in range(len(all_projections)):
        total_rank = 0
        for layer_name, (V_tail, _) in tail_bases.items():
            c_stack = []
            for task_idx in range(t + 1):
                deltas_keyed = all_projections[task_idx]
                if layer_name not in deltas_keyed:
                    continue
                delta = deltas_keyed[layer_name]
                delta_f32 = backend.astype(delta, "float32")
                C = backend.matmul(delta_f32, V_tail)
                backend.eval(C)
                c_stack.append(C)

            if not c_stack:
                continue

            stacked = backend.concatenate(c_stack, axis=0)
            backend.eval(stacked)
            S_stacked = backend.svd(stacked, compute_uv=False)
            backend.eval(S_stacked)
            s_list = backend.tolist(S_stacked)
            total_rank += _rank_eps(s_list, eps)

        incremental.append(total_rank - prev_total_rank)
        prev_total_rank = total_rank

    return incremental


def _cumulative_utilization_by_layer(backend, all_deltas_keyed: list[dict],
                                     tail_bases: dict, eps: float) -> dict:
    """Per-layer cumulative null-space utilization across all tasks.

    For each layer: rank([C_1..C_T]) / tail_dims — the fraction of null-space
    dimensions consumed by the UNION of all task adapters.

    This is the correct end-state capacity metric; per-task consumed values are
    not additive (adapters may reuse the same directions).
    """
    utilization = {}
    for layer_name, (V_tail, tail_dims) in tail_bases.items():
        c_stack = []
        for dk in all_deltas_keyed:
            if layer_name not in dk:
                continue
            delta_f32 = backend.astype(dk[layer_name], "float32")
            C = backend.matmul(delta_f32, V_tail)
            backend.eval(C)
            c_stack.append(C)

        if not c_stack:
            utilization[layer_name] = 0.0
            continue

        stacked = backend.concatenate(c_stack, axis=0)
        backend.eval(stacked)
        S_stacked = backend.svd(stacked, compute_uv=False)
        backend.eval(S_stacked)
        s_list = backend.tolist(S_stacked)
        cumulative_rank = _rank_eps(s_list, eps)
        utilization[layer_name] = round(cumulative_rank / tail_dims, 6) if tail_dims > 0 else 0.0

    return utilization


def _compute_pairwise_overlap(backend, delta_prev: dict, delta_curr: dict,
                               tail_bases: dict, eps: float) -> int:
    """Sum over layers of: consumed[prev] + consumed[curr] - rank([C_prev; C_curr]).

    Measures the null-space dimensions shared between two consecutive tasks.
    Zero when tasks are orthogonal in null space; positive when they overlap.
    """
    total_overlap = 0

    for layer_name, (V_tail, _) in tail_bases.items():
        c_stack = []
        for dk in (delta_prev, delta_curr):
            if layer_name not in dk:
                continue
            delta_f32 = backend.astype(dk[layer_name], "float32")
            C = backend.matmul(delta_f32, V_tail)
            backend.eval(C)
            c_stack.append(C)

        if len(c_stack) < 2:
            continue

        # rank of each individually
        s_prev = backend.tolist(backend.svd(c_stack[0], compute_uv=False))
        s_curr = backend.tolist(backend.svd(c_stack[1], compute_uv=False))
        r_prev = _rank_eps(s_prev, eps)
        r_curr = _rank_eps(s_curr, eps)

        # rank of joint stack
        stacked = backend.concatenate(c_stack, axis=0)
        backend.eval(stacked)
        s_joint = backend.tolist(backend.svd(stacked, compute_uv=False))
        r_joint = _rank_eps(s_joint, eps)

        total_overlap += r_prev + r_curr - r_joint

    return total_overlap


def _percentile(vals: list[float], p: int) -> float:
    """p-th percentile of a list. p in [0, 100]."""
    if not vals:
        return 0.0
    sorted_v = sorted(vals)
    idx = max(0, min(len(sorted_v) - 1, int(len(sorted_v) * p / 100)))
    return sorted_v[idx]


def _energy_ratio_stats(per_layer: dict) -> dict:
    """Compute mean, p10, p50, p90 of energy_ratio across layers."""
    vals = [info["energy_ratio"] for info in per_layer.values() if info["consumed"] > 0]
    if not vals:
        vals = [info["energy_ratio"] for info in per_layer.values()]
    if not vals:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": sum(vals) / len(vals),
        "p10": _percentile(vals, 10),
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
    }


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _write_merged_dataset(samples: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")


def _build_replay_dataset(current_samples: list[dict], prev_samples: list[list[dict]],
                          replay_fraction: float, seed: int) -> list[dict]:
    """Mix current task samples with a random sample of previous-task samples.

    Replay count = floor(replay_fraction * len(current_samples)).
    Samples uniformly from the pool of all previous-task samples.
    """
    n_replay = math.floor(replay_fraction * len(current_samples))
    pool = [s for shard in prev_samples for s in shard]
    rng = random.Random(seed)
    replay = rng.sample(pool, min(n_replay, len(pool)))
    merged = current_samples + replay
    rng.shuffle(merged)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    model_id = args.model_id or Path(args.model_path).name
    run_dir = args.output_root.expanduser().resolve() / args.run_id / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    backend = initialize_default_backend()
    cpu_metrics = get_continual_learning_metrics(None)
    dataset_service = get_dataset_training_service()
    model_loader = get_model_loader()
    capacity_service = get_capacity_analysis_service()
    eps = float(backend.finfo().eps)

    print(f"=== Exp 1 | run={args.run_id} | model={model_id} | seed={args.seed} | replay={args.replay_fraction} ===")

    current_model_path = args.model_path

    # Spectral reference from first weight layer
    weight_items = model_loader.iter_weights(str(current_model_path))
    try:
        first_layer_name, first_tensor = next(weight_items)
        sigma_k_ref = _spectral_norm(backend, first_tensor)
    except StopIteration:
        first_layer_name = "unknown"
        sigma_k_ref = 1.0
    print(f"sigma_k_ref from {first_layer_name}: {sigma_k_ref:.4f}")

    # Base model null-space capacity (once, before training)
    print("Profiling base model capacity...")
    base_capacity = capacity_service.analyze(
        model_path=str(current_model_path),
        checkpoint_path=run_dir / "capacity_checkpoint.json",
    )
    base_mean_null_dim = (
        sum(r.null_space_dim_f32 for r in base_capacity.layer_reports)
        / max(1, len(base_capacity.layer_reports))
    ) if base_capacity.layer_reports else 0.0
    print(f"Base mean null-space dim: {base_mean_null_dim:.2f} ({base_capacity.analyzed_layers} layers)")

    # Per-task accumulators
    task_remaining_null = []
    delta_history_keyed = []   # list[dict[str, Array]]
    delta_history_flat = []    # list[list[Array]]
    representative_deltas = []
    cka_matrix = []
    per_task_cka = []
    cumulative_weyl = 0.0
    tail_bases = None
    capacity_total = 0
    cumulative_new_dims = 0
    all_task_metrics = []      # full per-task metric records
    prev_shard_samples: list[list[dict]] = []  # raw samples per completed task

    task_datasets = args.task_datasets
    for i, dataset_path in enumerate(task_datasets):
        print(f"\n--> Task {i+1}/{len(task_datasets)}: {dataset_path}")
        adapter_path = run_dir / f"adapter_task_{i}"

        # Load raw samples for replay bookkeeping
        current_samples = _load_jsonl(dataset_path)

        # Build replay-merged dataset if needed
        if args.replay_fraction > 0 and prev_shard_samples:
            merged_samples = _build_replay_dataset(
                current_samples, prev_shard_samples, args.replay_fraction, args.seed + i,
            )
            merged_path = run_dir / f"replay_merged_task_{i}.jsonl"
            _write_merged_dataset(merged_samples, merged_path)
            train_path = str(merged_path)
            print(f"  Replay: {len(merged_samples) - len(current_samples)} samples from {len(prev_shard_samples)} prior shards")
        else:
            train_path = dataset_path

        # Train
        result = None
        try:
            result = dataset_service.train_from_dataset(
                model_path=str(current_model_path),
                dataset_path=train_path,
                eval_dataset_path=dataset_path,  # eval always on current task only
                output_path=str(adapter_path),
                seed=args.seed + i,
            )
            print(f"  Train: {result.train_iters} iters, loss {result.initial_loss:.3f} -> {result.final_loss:.3f}")
            print(f"  Eval:  baseline={result.baseline_loss:.3f}, post={result.post_loss:.3f}")
        except Exception as e:
            print(f"  Training FAILED: {e}")

        # Clean up temp replay file
        if args.replay_fraction > 0 and prev_shard_samples:
            merged_path.unlink(missing_ok=True)

        prev_shard_samples.append(current_samples)

        # Extract geometry deltas
        deltas_keyed = _extract_adapter_deltas_keyed(backend, adapter_path)
        task_deltas = list(deltas_keyed.values())
        if not task_deltas:
            task_deltas = [backend.zeros((10, 10))]
        delta_history_keyed.append(deltas_keyed)
        delta_history_flat.append(task_deltas)
        print(f"  Extracted {len(deltas_keyed)} LoRA delta pairs")

        rep_delta = _pick_representative_delta(backend, task_deltas)
        representative_deltas.append(rep_delta)

        # Precompute V_tail bases once (after first adapter reveals target keys)
        if tail_bases is None:
            print("\n  Precomputing tail bases from base model SVD...")
            tail_bases = _precompute_tail_bases(
                backend, model_loader, str(current_model_path),
                set(deltas_keyed.keys()), eps,
            )
            capacity_total = sum(td for _, td in tail_bases.values())
            print(f"  Capacity total: {capacity_total} tail dims across {len(tail_bases)} layers")

        # Null-space depletion
        consumed, per_layer = _compute_consumed_dims(backend, deltas_keyed, tail_bases, eps)
        remaining = capacity_total - consumed
        depletion_fraction = consumed / capacity_total if capacity_total > 0 else 0.0
        task_remaining_null.append(remaining)

        # Incremental new dims (single-task: joint rank - cumulative rank so far)
        # Computed incrementally to avoid re-stacking all tasks each iteration.
        # Re-use _incremental_new_dims on history up to this task.
        incr_list = _incremental_new_dims(backend, delta_history_keyed, tail_bases, eps)
        this_incr = incr_list[-1] if incr_list else 0
        cumulative_new_dims += this_incr

        # Overlap with previous task
        if i > 0:
            overlap = _compute_pairwise_overlap(
                backend, delta_history_keyed[-2], delta_history_keyed[-1], tail_bases, eps,
            )
        else:
            overlap = 0

        # Energy ratio statistics across layers
        er_stats = _energy_ratio_stats(per_layer)

        print(f"  Null-space: consumed={consumed}/{capacity_total}, depletion={depletion_fraction:.4f}, remaining={remaining}")
        print(f"  Incremental new dims: {this_incr}, cumulative: {cumulative_new_dims}")
        print(f"  Energy ratio: mean={er_stats['mean']:.4f}, p10={er_stats['p10']:.4f}, p50={er_stats['p50']:.4f}, p90={er_stats['p90']:.4f}")
        print(f"  Overlap with prev task: {overlap}")

        for ln, info in sorted(per_layer.items()):
            print(f"    {ln}: {info['consumed']}/{info['tail_dims']} consumed, energy_ratio={info['energy_ratio']:.4f}")

        # CKA
        task_min_cka = result.min_cka if result is not None else None
        task_mean_cka = result.mean_cka if result is not None else None
        per_task_cka.append({"task": i, "min_cka": task_min_cka, "mean_cka": task_mean_cka})
        print(f"  CKA: min={task_min_cka}, mean={task_mean_cka}")

        # Cross-task CKA matrix (weight-delta Gram)
        eval_cka_row = []
        for past_rep in representative_deltas:
            gram1 = backend.matmul(rep_delta, backend.transpose(rep_delta))
            gram2 = backend.matmul(past_rep, backend.transpose(past_rep))
            cka_val = float(compute_linear_cka_gram(gram1, gram2, backend))
            eval_cka_row.append(cka_val)
        cka_matrix.append(eval_cka_row)

        # Spectral / Weyl
        task_max_spectral = 0.0
        task_weyl = 0.0
        for d in task_deltas:
            norm_val = _spectral_norm(backend, d)
            task_max_spectral = max(task_max_spectral, norm_val)
            task_weyl += norm_val
        cumulative_weyl += task_weyl
        print(f"  Spectral: max={task_max_spectral:.4f}, task_weyl={task_weyl:.4f}")

        # Accumulate full per-task record
        all_task_metrics.append({
            "task": i,
            "dataset": dataset_path,
            "incremental_new_dims": this_incr,
            "cumulative_new_dims": cumulative_new_dims,
            "consumed": consumed,
            "remaining": remaining,
            "depletion_fraction": depletion_fraction,
            "energy_ratio_mean": er_stats["mean"],
            "energy_ratio_p10": er_stats["p10"],
            "energy_ratio_p50": er_stats["p50"],
            "energy_ratio_p90": er_stats["p90"],
            "overlap_with_previous_task": overlap,
            "train_loss_start_end": [
                result.initial_loss if result else None,
                result.final_loss if result else None,
            ],
            "eval_loss_start_end": [
                result.baseline_loss if result else None,
                result.post_loss if result else None,
            ],
            "min_cka": task_min_cka,
            "mean_cka": task_mean_cka,
            "per_layer": {
                ln: {
                    "consumed": info["consumed"],
                    "tail_dims": info["tail_dims"],
                    "energy_ratio": round(info["energy_ratio"], 6),
                }
                for ln, info in sorted(per_layer.items())
            },
        })

        # Stop rule: full capacity consumed
        if capacity_total > 0 and cumulative_new_dims >= capacity_total:
            print(f"\n  *** Capacity saturated ({cumulative_new_dims}/{capacity_total}) — stopping early ***")
            break

    # ---------------------------------------------------------------------------
    # Post-loop telemetry
    # ---------------------------------------------------------------------------
    print(f"\n=== Final Telemetry Summary ===")
    depletion_rate = cpu_metrics.null_space_depletion_rate(task_remaining_null)
    cka_stability = cpu_metrics.cka_stability(cka_matrix)

    safe_sigma = max(sigma_k_ref, eps)
    trajectory = []
    for td in delta_history_flat:
        task_max = 0.0
        for d in td:
            task_max = max(task_max, _spectral_norm(backend, d))
        trajectory.append(task_max / safe_sigma)

    # Final tail utilization by layer: cumulative rank across ALL tasks / tail_dims.
    # Not the last task's consumed — adapters may reuse the same null-space directions,
    # so per-task consumed values are not additive. We need rank([C_1..C_T]) per layer.
    final_tail_utilization_by_layer = _cumulative_utilization_by_layer(
        backend, delta_history_keyed, tail_bases or {}, eps,
    )

    # Full incremental_new_dims list (may be shorter than task_datasets if early stop)
    incremental_all = [m["incremental_new_dims"] for m in all_task_metrics]

    # Per-task depletion (re-compute consumed/energy from adapter deltas for all tasks in output)
    per_task_depletion = []
    for t_idx, dk in enumerate(delta_history_keyed):
        consumed_t, per_layer_t = _compute_consumed_dims(backend, dk, tail_bases or {}, eps)
        per_task_depletion.append({
            "task": t_idx,
            "consumed": consumed_t,
            "remaining": capacity_total - consumed_t,
            "per_layer": {
                ln: {
                    "consumed": info["consumed"],
                    "tail_dims": info["tail_dims"],
                    "energy_ratio": round(info["energy_ratio"], 6),
                }
                for ln, info in sorted(per_layer_t.items())
            },
        })

    output = {
        "run_id": args.run_id,
        "seed": args.seed,
        "model_id": model_id,
        "model_path": args.model_path,
        "baseline": args.baseline,
        "replay_fraction": args.replay_fraction,
        "tasks": args.task_datasets,
        "sigma_k_ref": sigma_k_ref,
        "base_capacity": {
            "analyzed_layers": base_capacity.analyzed_layers,
            "mean_null_dim": base_mean_null_dim,
            "capacity_total_tail_dims": capacity_total,
        },
        "telemetry": {
            "task_metrics": all_task_metrics,
            "cross_task_cka_matrix": cka_matrix,
            "per_task_cka": per_task_cka,
            "task_remaining_null": task_remaining_null,
            "depletion_rate": depletion_rate,
            "cka_stability": cka_stability,
            "incremental_new_dims": incremental_all,
            "spectral_budget_trajectory": trajectory,
            "weyl_accumulation": cumulative_weyl,
            "deltas_per_task": [len(td) for td in delta_history_flat],
            "final_tail_utilization_by_layer": final_tail_utilization_by_layer,
            "per_task_depletion": per_task_depletion,
        },
    }

    out_file = run_dir / "exp1_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
