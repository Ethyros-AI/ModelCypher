#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf

import argparse
import json
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


def _extract_adapter_deltas(backend, adapter_path: Path) -> list:
    """Extract ALL LoRA delta pairs from a saved adapter.

    Returns list of delta arrays. Each delta = (lora_a @ lora_b).T = lora_b.T @ lora_a.T,
    which is the exact effective weight perturbation (Cayley transform already baked in
    by to_standard_lora() during save).
    """
    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        print(f"  WARNING: No adapter file at {adapter_file}")
        return []

    adapter_weights = backend.load_safetensors(str(adapter_file))
    deltas = []

    for k in sorted(adapter_weights):
        if ".lora_a" in k:
            b_key = k.replace(".lora_a", ".lora_b")
            if b_key in adapter_weights:
                a = adapter_weights[k]      # [in, r]
                b = adapter_weights[b_key]  # [r, out]
                # delta = (a @ b).T = b.T @ a.T = [out, in] (weight convention)
                delta = backend.transpose(backend.matmul(a, b))
                deltas.append(delta)

    return deltas


def _spectral_norm(backend, tensor) -> float:
    """Compute spectral norm (largest singular value) of a tensor."""
    # MLX SVD requires float32; model weights are typically bf16
    tensor_f32 = backend.astype(tensor, "float32")
    S = backend.svd(tensor_f32, compute_uv=False)
    norm_t = backend.max(S)
    backend.eval(norm_t)
    return float(backend.to_scalar(norm_t))


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

    # Per-task telemetry accumulators
    task_ranks = []
    delta_history = []       # list[list[Array]] — all deltas per task
    representative_deltas = []  # One delta per task for cross-task CKA
    cka_matrix = []
    per_task_cka = []
    cumulative_weyl = 0.0
    eps = float(backend.finfo().eps)

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
                dim_monitor=True,
            )
            print(f"  Training: {result.train_iters} iters, loss {result.initial_loss:.3f} -> {result.final_loss:.3f}")
        except Exception as e:
            print(f"  Training FAILED: {e}")

        # 2. Extract ALL geometry deltas from adapter
        task_deltas = _extract_adapter_deltas(backend, adapter_path)
        if not task_deltas:
            task_deltas = [backend.zeros((10, 10))]
        delta_history.append(task_deltas)
        print(f"  Extracted {len(task_deltas)} LoRA delta pairs")

        # Pick representative delta for cross-task CKA
        rep_delta = _pick_representative_delta(backend, task_deltas)
        representative_deltas.append(rep_delta)

        # 3. Null-space tracking via training result
        if result is not None and result.dim_final_null_fraction is not None:
            current_null_rank = base_mean_null_dim * result.dim_final_null_fraction
        else:
            # No measured depletion available — use base capacity (conservative)
            current_null_rank = base_mean_null_dim
            print(f"  WARNING: dim_final_null_fraction not available")
        task_ranks.append(current_null_rank)

        # 4. Real CKA from training result (activation-space, not weight-space)
        task_min_cka = result.min_cka if result is not None else None
        task_mean_cka = result.mean_cka if result is not None else None
        per_task_cka.append({"task": i, "min_cka": task_min_cka, "mean_cka": task_mean_cka})
        print(f"  CKA: min={task_min_cka}, mean={task_mean_cka}")

        # 5. Cross-task CKA matrix (weight-delta Gram proxy for inter-adapter similarity)
        eval_cka_row = []
        for past_rep in representative_deltas:
            gram1 = backend.matmul(rep_delta, backend.transpose(rep_delta))
            gram2 = backend.matmul(past_rep, backend.transpose(past_rep))
            cka_val = float(compute_linear_cka_gram(gram1, gram2, backend))
            eval_cka_row.append(cka_val)
        cka_matrix.append(eval_cka_row)

        # 6. Per-task spectral norms (for trajectory and Weyl accumulation)
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
    depletion_rate = cpu_metrics.null_space_depletion_rate(task_ranks)
    cka_stability = cpu_metrics.cka_stability(cka_matrix)

    # Spectral budget trajectory: max spectral norm per task / sigma_k_ref
    safe_sigma = max(sigma_k_ref, eps)
    trajectory = []
    for task_deltas in delta_history:
        task_max = 0.0
        for d in task_deltas:
            task_max = max(task_max, _spectral_norm(backend, d))
        trajectory.append(task_max / safe_sigma)

    deltas_per_task = [len(td) for td in delta_history]

    output = {
        "seed": args.seed,
        "model_path": args.model_path,
        "baseline": args.baseline,
        "tasks": args.task_datasets,
        "sigma_k_ref": sigma_k_ref,
        "base_capacity": {
            "analyzed_layers": base_capacity.analyzed_layers,
            "mean_null_dim": base_mean_null_dim,
        },
        "telemetry": {
            "task_ranks": task_ranks,
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
