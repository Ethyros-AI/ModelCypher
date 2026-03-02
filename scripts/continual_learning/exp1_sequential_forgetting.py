#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

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
from modelcypher.core.domain.continual_learning_metrics import get_continual_learning_metrics
from modelcypher.core.domain.geometry.cka import compute_linear_cka_gram

MODEL_PATH_DEFAULT = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
OUTPUT_ROOT_DEFAULT = Path("results/continual_learning/exp1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 1: Sequential Forgetting")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="Run unconstrained LoRA baseline")
    parser.add_argument(
        "--task-datasets",
        nargs="+",
        default=["data/training/benchmark_train.jsonl", "data/training/retention_replay.jsonl"],
        help="List of dataset paths for sequential training",
    )
    return parser.parse_args()


def _extract_adapter_deltas_keyed(backend, adapter_path: Path) -> dict:
    """Extract ALL LoRA delta pairs from a saved adapter, keyed by base weight name.

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
                # Map to base weight key: strip .lora_a, add .weight
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

    Threshold = σ_1 * sqrt(eps). Singular values below this are
    indistinguishable from numerical noise at the working precision.
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


def _precompute_tail_bases(backend, model_loader, model_path: str,
                           adapted_keys: set[str], eps: float) -> dict:
    """Precompute V_tail basis for each adapted layer from base model SVD.

    For each 2D weight matrix that has an adapter:
      W [out, in] = U @ diag(S) @ V^T
      tail_dims = full_rank - rank_eps(S)
      V_tail = last tail_dims columns of V [in, tail_dims]

    Returns dict mapping layer_name -> (V_tail, tail_dims).
    Layers with tail_dims == 0 are excluded (fully saturated, no null space).
    """
    tail_bases = {}
    sqrt_eps = math.sqrt(eps)

    for layer_name, tensor in model_loader.iter_weights(model_path):
        if layer_name not in adapted_keys:
            continue

        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue

        tensor_f32 = backend.astype(tensor, "float32")
        U, S, Vt = backend.svd(tensor_f32, compute_uv=True)
        backend.eval(S)
        backend.eval(Vt)

        s_list = backend.tolist(S)
        full_rank = len(s_list)
        used_rank = _rank_eps(s_list, eps)
        tail_dims = full_rank - used_rank

        if tail_dims > 0:
            # V = Vt.T, V_tail = V[:, used_rank:] = Vt[used_rank:, :].T
            # Vt shape: [min(out,in), in] — rows are right singular vectors
            Vt_tail = backend.astype(Vt, "float32")
            # Slice last tail_dims rows of Vt, then transpose to get V_tail [in, tail_dims]
            V_tail = backend.transpose(Vt_tail[used_rank:])
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

    Returns (total_consumed, per_layer_dict).
    """
    total_consumed = 0
    per_layer = {}

    for layer_name, (V_tail, tail_dims) in tail_bases.items():
        if layer_name not in deltas_keyed:
            per_layer[layer_name] = {"tail_dims": tail_dims, "consumed": 0}
            continue

        delta = deltas_keyed[layer_name]
        delta_f32 = backend.astype(delta, "float32")
        # C = delta @ V_tail: [out, in] @ [in, tail_dims] = [out, tail_dims]
        C = backend.matmul(delta_f32, V_tail)
        backend.eval(C)

        S_c = backend.svd(C, compute_uv=False)
        backend.eval(S_c)
        s_list = backend.tolist(S_c)
        consumed = _rank_eps(s_list, eps)

        per_layer[layer_name] = {"tail_dims": tail_dims, "consumed": consumed}
        total_consumed += consumed

    return total_consumed, per_layer


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    seed_dir = output_root / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    backend = initialize_default_backend()
    cpu_metrics = get_continual_learning_metrics(None)
    dataset_service = get_dataset_training_service()
    model_loader = get_model_loader()
    capacity_service = get_capacity_analysis_service()
    eps = float(backend.finfo().eps)

    print(f"=== Starting Experiment 1 (Sequential Forgetting) | Seed: {args.seed} ===")

    current_model_path = args.model_path

    # Calculate sigma_k_ref from first weight layer (spectral bound reference)
    weight_items = model_loader.iter_weights(str(current_model_path))
    try:
        first_layer_name, first_tensor = next(weight_items)
        sigma_k_ref = _spectral_norm(backend, first_tensor)
    except StopIteration:
        first_layer_name = "unknown"
        sigma_k_ref = 1.0
    print(f"sigma_k_ref from {first_layer_name}: {sigma_k_ref:.4f}")

    # Profile base model null-space capacity (once, before any training)
    print("Profiling base model capacity...")
    base_capacity = capacity_service.analyze(
        model_path=str(current_model_path),
        checkpoint_path=seed_dir / "capacity_checkpoint.json",
    )
    base_mean_null_dim = (
        sum(r.null_space_dim_f32 for r in base_capacity.layer_reports)
        / max(1, len(base_capacity.layer_reports))
    ) if base_capacity.layer_reports else 0.0
    print(f"Base mean null-space dim: {base_mean_null_dim:.2f} ({base_capacity.analyzed_layers} layers)")

    # Discover which layers the adapter targets by doing a quick probe train
    # (or read from an existing adapter). For now, we precompute V_tail for all
    # adapted layers after the first training pass.
    # We defer tail basis computation until we know which keys the adapter uses.

    # Per-task telemetry accumulators
    task_remaining_null = []
    delta_history_keyed = []  # list[dict[str, Array]]
    delta_history_flat = []   # list[list[Array]] — for CKA/spectral
    representative_deltas = []
    cka_matrix = []
    per_task_cka = []
    cumulative_weyl = 0.0
    tail_bases = None         # Computed once after first adapter is saved
    capacity_total = 0

    for i, dataset_path in enumerate(args.task_datasets):
        print(f"\n--> Task {i+1}/{len(args.task_datasets)}: {dataset_path}")
        adapter_path = seed_dir / f"adapter_task_{i}"

        # 1. Train adapter on current task
        result = None
        try:
            result = dataset_service.train_from_dataset(
                model_path=str(current_model_path),
                dataset_path=str(dataset_path),
                eval_dataset_path=str(dataset_path),
                output_path=str(adapter_path),
                seed=args.seed + i,
            )
            print(f"  Training: {result.train_iters} iters, loss {result.initial_loss:.3f} -> {result.final_loss:.3f}")
        except Exception as e:
            print(f"  Training FAILED: {e}")

        # 2. Extract ALL geometry deltas from adapter, keyed by base weight name
        deltas_keyed = _extract_adapter_deltas_keyed(backend, adapter_path)
        task_deltas = list(deltas_keyed.values())
        if not task_deltas:
            task_deltas = [backend.zeros((10, 10))]
        delta_history_keyed.append(deltas_keyed)
        delta_history_flat.append(task_deltas)
        print(f"  Extracted {len(deltas_keyed)} LoRA delta pairs")

        # Pick representative delta for cross-task CKA
        rep_delta = _pick_representative_delta(backend, task_deltas)
        representative_deltas.append(rep_delta)

        # 3. Precompute V_tail bases once (after first adapter reveals target keys)
        if tail_bases is None:
            print("\n  Precomputing tail bases from base model SVD...")
            tail_bases = _precompute_tail_bases(
                backend, model_loader, str(current_model_path),
                set(deltas_keyed.keys()), eps,
            )
            capacity_total = sum(td for _, td in tail_bases.values())
            print(f"  Capacity total: {capacity_total} tail dims across {len(tail_bases)} layers")

        # 4. Null-space depletion: project delta into V_tail, count consumed dims
        consumed, per_layer = _compute_consumed_dims(
            backend, deltas_keyed, tail_bases, eps,
        )
        remaining = capacity_total - consumed
        depletion = consumed / capacity_total if capacity_total > 0 else 0.0
        task_remaining_null.append(remaining)
        print(f"  Null-space: consumed={consumed}/{capacity_total}, depletion={depletion:.4f}, remaining={remaining}")

        # Per-layer detail
        for ln, info in sorted(per_layer.items()):
            print(f"    {ln}: {info['consumed']}/{info['tail_dims']} consumed")

        # 5. Real CKA from training result (activation-space, not weight-space)
        task_min_cka = result.min_cka if result is not None else None
        task_mean_cka = result.mean_cka if result is not None else None
        per_task_cka.append({"task": i, "min_cka": task_min_cka, "mean_cka": task_mean_cka})
        print(f"  CKA: min={task_min_cka}, mean={task_mean_cka}")

        # 6. Cross-task CKA matrix (weight-delta Gram proxy for inter-adapter similarity)
        eval_cka_row = []
        for past_rep in representative_deltas:
            gram1 = backend.matmul(rep_delta, backend.transpose(rep_delta))
            gram2 = backend.matmul(past_rep, backend.transpose(past_rep))
            cka_val = float(compute_linear_cka_gram(gram1, gram2, backend))
            eval_cka_row.append(cka_val)
        cka_matrix.append(eval_cka_row)

        # 7. Per-task spectral norms (for trajectory and Weyl accumulation)
        task_max_spectral = 0.0
        task_weyl = 0.0
        for d in task_deltas:
            norm_val = _spectral_norm(backend, d)
            task_max_spectral = max(task_max_spectral, norm_val)
            task_weyl += norm_val
        cumulative_weyl += task_weyl
        print(f"  Spectral: max={task_max_spectral:.4f}, task_weyl={task_weyl:.4f}")

    # Final telemetry summary
    print(f"\n=== Final Telemetry Summary ===")
    depletion_rate = cpu_metrics.null_space_depletion_rate(task_remaining_null)
    cka_stability = cpu_metrics.cka_stability(cka_matrix)

    # Spectral budget trajectory: max spectral norm per task / sigma_k_ref
    safe_sigma = max(sigma_k_ref, eps)
    trajectory = []
    for task_deltas in delta_history_flat:
        task_max = 0.0
        for d in task_deltas:
            task_max = max(task_max, _spectral_norm(backend, d))
        trajectory.append(task_max / safe_sigma)

    deltas_per_task = [len(td) for td in delta_history_flat]

    output = {
        "seed": args.seed,
        "model_path": args.model_path,
        "baseline": args.baseline,
        "tasks": args.task_datasets,
        "sigma_k_ref": sigma_k_ref,
        "base_capacity": {
            "analyzed_layers": base_capacity.analyzed_layers,
            "mean_null_dim": base_mean_null_dim,
            "capacity_total_tail_dims": capacity_total,
        },
        "telemetry": {
            "task_remaining_null": task_remaining_null,
            "per_task_cka": per_task_cka,
            "cka_matrix": cka_matrix,
            "depletion_rate": depletion_rate,
            "spectral_budget_trajectory": trajectory,
            "cka_stability": cka_stability,
            "weyl_accumulation": cumulative_weyl,
            "deltas_per_task": deltas_per_task,
        },
    }

    out_file = seed_dir / "exp1_results.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
