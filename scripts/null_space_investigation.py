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

"""Null-Space Geometry Investigation.

Why does learning in the weight null space perturb inference?

The training pipeline defines three different "null spaces":
  1. Weight null: Shannon effective rank of W (tail_dims) — where NB-LoRA operates
  2. CKA probe null: 181 eval texts — what CKA verification checks
  3. Inference probe null: 10 StarProblem reasoning prompts — what Phase 5 checks

This script investigates whether these three spaces are misaligned, explaining
why structural PASS (CKA verification) co-occurs with inference FAIL.

Four analyses:
  A1. CKA on inference probes vs eval probes (CKA blindness test)
  A2. Per-probe activation delta (localize perturbation to lost/gained/stable probes)
  A3. Subspace overlap (do eval probes span inference probe geometry?)
  A4. LoRA delta projection (are lost probes more exposed to adapter directions?)

Usage:
    poetry run python scripts/null_space_investigation.py --adapter-path /path/to/adapter
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DEFAULT_EVAL_DATA = "data/training/benchmark_val.jsonl"
DEFAULT_OUTPUT_DIR = Path("results/null_space_investigation")

# Phase 5 probe parameters (from pipeline validation run)
PHASE5_PROBE_COUNT = 10
PHASE5_PROBE_SEED = 3475334679

# Seq length used in the training run (from result.json analysis)
SEQ_LENGTH = 320

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("null_space_investigation")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to the adapted trial directory containing adapters.safetensors and geometry_manifest.json.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Base model path used for the comparison.",
    )
    parser.add_argument(
        "--eval-data",
        default=DEFAULT_EVAL_DATA,
        help="JSONL evaluation dataset used for the probe subspace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where investigation_results.json will be written.",
    )
    return parser.parse_args()


def _validate_adapter_path(adapter_path: Path) -> Path:
    required_files = (
        adapter_path / "adapters.safetensors",
        adapter_path / "geometry_manifest.json",
    )
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        msg = (
            "adapter-path must point to a retained trial directory with "
            f"adapters.safetensors and geometry_manifest.json; missing: {missing}"
        )
        raise FileNotFoundError(msg)
    return adapter_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_backend():
    """Load the MLX backend."""
    from modelcypher.backends.mlx_backend import MLXBackend
    return MLXBackend()


def _load_eval_samples(eval_data: str) -> list[dict[str, Any]]:
    """Load the 181 eval dataset samples."""
    from modelcypher.core.domain.dataset_loading import load_jsonl_dataset
    samples = load_jsonl_dataset(eval_data)
    logger.info("Loaded %d eval samples from %s", len(samples), eval_data)
    return samples


def _create_inference_probes():
    """Create the same 10 Phase 5 inference probes used in the validation run."""
    from modelcypher.core.domain.training.online_eval import create_eval_problem_set
    problems = create_eval_problem_set(
        n_problems=PHASE5_PROBE_COUNT,
        seed=PHASE5_PROBE_SEED,
    )
    logger.info(
        "Created %d inference probes (seed=%d): %s",
        len(problems),
        PHASE5_PROBE_SEED,
        [p.problem_type for p in problems],
    )
    return problems


def _get_inference_probe_texts(problems) -> list[str]:
    """Convert StarProblems to the actual prompt text the model sees."""
    from modelcypher.core.domain.star.prompting import (
        build_forward_prompt,
        default_few_shot_examples,
    )
    n_demonstrations = len(default_few_shot_examples())
    return [build_forward_prompt(p, demonstrations=n_demonstrations) for p in problems]


def _collect_activations(
    backend, model, tokenizer, texts: list[str],
) -> dict[int, list]:
    """Collect per-layer mean-pooled activations (same as _collect_probe_activations)."""
    activations: dict[int, list] = {}
    for text in texts:
        acts = backend.collect_hidden_activations(model, tokenizer, [text])
        for layer_idx, act in acts.items():
            pooled = backend.mean(act, axis=1)  # [1, seq, hidden] -> [1, hidden]
            pooled = backend.reshape(pooled, (-1,))  # [hidden]
            backend.eval(pooled)
            activations.setdefault(layer_idx, []).append(pooled)
    return activations


def _run_correctness_eval(backend, model_path: str, adapter_path: str | None, problems):
    """Run online eval to get correct_ids for a model configuration."""
    from modelcypher.core.domain.training.online_eval import evaluate_correctness

    model, tokenizer = backend.load_model(model_path, adapter_path)

    def _generate(prompt: str, max_tokens: int) -> str:
        return backend.generate(model, tokenizer, prompt, max_tokens=max_tokens)

    result = evaluate_correctness(
        problems=problems,
        generate_fn=_generate,
        epoch=0,
        baseline_correct_ids=None,
        max_tokens=SEQ_LENGTH,
    )
    model = None
    tokenizer = None
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Analysis 1: CKA on inference probes vs eval probes
# ---------------------------------------------------------------------------


def analysis_1_cka_comparison(
    backend,
    base_eval_acts: dict[int, list],
    adapted_eval_acts: dict[int, list],
    base_inference_acts: dict[int, list],
    adapted_inference_acts: dict[int, list],
) -> dict:
    """Compare CKA and gram perturbation between eval probes and inference probes."""
    from modelcypher.core.domain.geometry.cka import (
        compute_gram_perturbation_ratio,
        compute_linear_cka_from_activations,
    )

    logger.info("=== Analysis 1: CKA on inference probes vs eval probes ===")
    per_layer = {}

    for layer_idx in sorted(base_eval_acts.keys()):
        # Eval probes (181 samples)
        base_eval_stack = backend.stack(base_eval_acts[layer_idx])
        adapted_eval_stack = backend.stack(adapted_eval_acts[layer_idx])
        cka_eval = compute_linear_cka_from_activations(
            base_eval_stack, adapted_eval_stack, backend,
        )
        eps_eval, bound_eval = compute_gram_perturbation_ratio(
            base_eval_stack, adapted_eval_stack, backend,
        )

        # Inference probes (10 samples)
        if layer_idx not in base_inference_acts:
            continue
        base_inf_stack = backend.stack(base_inference_acts[layer_idx])
        adapted_inf_stack = backend.stack(adapted_inference_acts[layer_idx])
        cka_inf = compute_linear_cka_from_activations(
            base_inf_stack, adapted_inf_stack, backend,
        )
        eps_inf, bound_inf = compute_gram_perturbation_ratio(
            base_inf_stack, adapted_inf_stack, backend,
        )

        per_layer[str(layer_idx)] = {
            "cka_eval": round(cka_eval, 6),
            "cka_inference": round(cka_inf, 6),
            "eps_eval": round(eps_eval, 6),
            "eps_inference": round(eps_inf, 6),
            "bound_eval": round(bound_eval, 6),
            "bound_inference": round(bound_inf, 6),
            "eps_ratio": round(eps_inf / eps_eval, 3) if eps_eval > 1e-10 else None,
        }
        logger.info(
            "  Layer %2d: CKA eval=%.4f inf=%.4f | eps eval=%.4f inf=%.4f (ratio=%.2fx)",
            layer_idx,
            cka_eval,
            cka_inf,
            eps_eval,
            eps_inf,
            eps_inf / eps_eval if eps_eval > 1e-10 else float("inf"),
        )

    return {"per_layer": per_layer}


# ---------------------------------------------------------------------------
# Analysis 2: Per-probe activation delta
# ---------------------------------------------------------------------------


def analysis_2_per_probe_delta(
    backend,
    base_inference_acts: dict[int, list],
    adapted_inference_acts: dict[int, list],
    problems,
    baseline_correct_ids: frozenset[str],
    adapted_correct_ids: frozenset[str],
) -> dict:
    """Per-probe, per-layer activation delta grouped by correctness change."""
    import mlx.core as mx

    logger.info("=== Analysis 2: Per-probe activation delta ===")

    lost = baseline_correct_ids - adapted_correct_ids
    gained = adapted_correct_ids - baseline_correct_ids
    stable_correct = baseline_correct_ids & adapted_correct_ids
    stable_incorrect = (
        {p.problem_id for p in problems} - baseline_correct_ids - adapted_correct_ids
    )

    logger.info(
        "  Lost=%d, Gained=%d, Stable correct=%d, Stable incorrect=%d",
        len(lost), len(gained), len(stable_correct), len(stable_incorrect),
    )

    def _classify(problem_id: str) -> str:
        if problem_id in lost:
            return "lost"
        if problem_id in gained:
            return "gained"
        if problem_id in stable_correct:
            return "stable_correct"
        return "stable_incorrect"

    probes_out = []
    for i, problem in enumerate(problems):
        change = _classify(problem.problem_id)
        per_layer = {}
        for layer_idx in sorted(base_inference_acts.keys()):
            base_vec = base_inference_acts[layer_idx][i]
            adapted_vec = adapted_inference_acts[layer_idx][i]

            # Cosine similarity
            dot = float(mx.sum(base_vec * adapted_vec).item())
            norm_base = float(mx.sqrt(mx.sum(base_vec * base_vec)).item())
            norm_adapted = float(mx.sqrt(mx.sum(adapted_vec * adapted_vec)).item())
            cos_sim = dot / (norm_base * norm_adapted + 1e-12)

            # L2 delta
            delta = adapted_vec - base_vec
            l2 = float(mx.sqrt(mx.sum(delta * delta)).item())

            # Relative L2
            rel_l2 = l2 / (norm_base + 1e-12)

            per_layer[str(layer_idx)] = {
                "cosine_sim": round(cos_sim, 6),
                "l2_delta": round(l2, 6),
                "relative_l2": round(rel_l2, 6),
            }

        probes_out.append({
            "problem_id": problem.problem_id,
            "problem_type": problem.problem_type,
            "correctness_change": change,
            "per_layer": per_layer,
        })

        logger.info(
            "  Probe %s (%s) [%s]: worst cosine=%.4f @ layer %s",
            problem.problem_id[:12],
            problem.problem_type,
            change,
            min(v["cosine_sim"] for v in per_layer.values()),
            min(per_layer, key=lambda k: per_layer[k]["cosine_sim"]),
        )

    return {"probes": probes_out}


# ---------------------------------------------------------------------------
# Analysis 3: Subspace overlap
# ---------------------------------------------------------------------------


def analysis_3_subspace_overlap(
    backend,
    base_eval_acts: dict[int, list],
    base_inference_acts: dict[int, list],
    problems,
) -> dict:
    """Measure how much inference probe variance lies within the eval probe subspace."""
    import mlx.core as mx

    from modelcypher.core.domain.geometry.subspace import _compute_intrinsic_rank

    logger.info("=== Analysis 3: Subspace overlap ===")

    per_layer = {}
    for layer_idx in sorted(base_eval_acts.keys()):
        if layer_idx not in base_inference_acts:
            continue

        eval_stack = backend.stack(base_eval_acts[layer_idx])  # [181, hidden]
        intrinsic_rank = _compute_intrinsic_rank(eval_stack, backend)

        # SVD of eval activations to get basis
        eval_f32 = eval_stack.astype(mx.float32)
        mx.eval(eval_f32)
        U, S, Vt = mx.linalg.svd(eval_f32, stream=mx.cpu)
        mx.eval(U, S, Vt)

        # Take top-k basis vectors (right singular vectors)
        k = min(intrinsic_rank, int(Vt.shape[0]))
        basis = Vt[:k, :]  # [k, hidden]

        # Project each inference probe onto this basis
        per_probe = []
        for i, problem in enumerate(problems):
            inf_vec = base_inference_acts[layer_idx][i]  # [hidden]
            inf_f32 = inf_vec.astype(mx.float32)

            # Projection: proj = (v @ basis^T) @ basis
            coords = inf_f32 @ basis.T  # [k]
            projected = coords @ basis  # [hidden]
            mx.eval(projected)

            # Explained variance ratio = ||projected||^2 / ||original||^2
            proj_norm_sq = float(mx.sum(projected * projected).item())
            orig_norm_sq = float(mx.sum(inf_f32 * inf_f32).item())
            evr = proj_norm_sq / (orig_norm_sq + 1e-12)

            per_probe.append({
                "problem_id": problem.problem_id,
                "problem_type": problem.problem_type,
                "explained_variance_ratio": round(evr, 6),
            })

        mean_evr = sum(p["explained_variance_ratio"] for p in per_probe) / len(per_probe)
        per_layer[str(layer_idx)] = {
            "eval_intrinsic_rank": intrinsic_rank,
            "eval_total_probes": int(eval_stack.shape[0]),
            "hidden_dim": int(eval_stack.shape[1]),
            "basis_dims_used": k,
            "mean_explained_variance_ratio": round(mean_evr, 6),
            "per_inference_probe": per_probe,
        }

        logger.info(
            "  Layer %2d: intrinsic_rank=%d, basis_dims=%d, mean EVR=%.4f",
            layer_idx, intrinsic_rank, k, mean_evr,
        )

    return {"per_layer": per_layer}


# ---------------------------------------------------------------------------
# Analysis 4: LoRA delta projection
# ---------------------------------------------------------------------------


def analysis_4_lora_delta_projection(
    backend,
    base_inference_acts: dict[int, list],
    problems,
    baseline_correct_ids: frozenset[str],
    adapted_correct_ids: frozenset[str],
    adapter_path: Path,
) -> dict:
    """Check whether lost probes are more exposed to LoRA delta directions."""
    import mlx.core as mx

    logger.info("=== Analysis 4: LoRA delta projection ===")

    # Load adapter weights
    weights_path = adapter_path / "adapters.safetensors"
    adapter_weights = backend.load_safetensors(str(weights_path))
    logger.info("Loaded adapter weights: %d keys", len(adapter_weights))

    # Load geometry manifest for sigma_k values
    manifest_path = adapter_path / "geometry_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    sigma_k_by_module = manifest.get("sigma_k_by_module", {})

    lost = baseline_correct_ids - adapted_correct_ids
    gained = adapted_correct_ids - baseline_correct_ids

    def _classify(problem_id: str) -> str:
        if problem_id in lost:
            return "lost"
        if problem_id in gained:
            return "gained"
        if problem_id in (baseline_correct_ids & adapted_correct_ids):
            return "stable_correct"
        return "stable_incorrect"

    # Map layer index to lora modules at that layer
    per_module = {}

    for key in sorted(adapter_weights.keys()):
        if not key.endswith(".lora_a"):
            continue
        key_base = key[:-7]  # e.g., "model.layers.2.self_attn.q_proj"
        key_b = f"{key_base}.lora_b"
        if key_b not in adapter_weights:
            continue

        lora_a = adapter_weights[key]    # LoRA A factor
        lora_b = adapter_weights[key_b]  # LoRA B factor

        # Reconstruct weight delta: delta = lora_b^T @ lora_a^T (for [out, in] layout)
        # But for activation projection we need the delta in input space
        # Forward pass: x @ lora_a @ lora_b → output contribution
        # So lora_a projects input → rank-r, lora_b projects rank-r → output
        # The input-space directions are the rows of lora_a (right singular vectors of delta)

        # Effective delta in weight space: W_delta = scale * lora_b^T @ lora_a^T  [out, in]
        lora_a_f32 = lora_a.astype(mx.float32)
        lora_b_f32 = lora_b.astype(mx.float32)
        delta = 2.0 * mx.matmul(mx.transpose(lora_b_f32), mx.transpose(lora_a_f32))
        mx.eval(delta)

        # SVD of delta
        U_d, S_d, Vt_d = mx.linalg.svd(delta, stream=mx.cpu)
        mx.eval(U_d, S_d, Vt_d)

        # Top singular values and right singular vectors (input-space directions)
        top_k = min(5, int(S_d.shape[0]))
        top_svs = [round(float(S_d[i].item()), 6) for i in range(top_k)]
        # Right singular vectors Vt_d[i, :] are the input-space directions
        top_input_dirs = Vt_d[:top_k, :]  # [top_k, in_features]

        # Extract layer index from key
        parts = key_base.split(".")
        layer_idx = None
        for j, p in enumerate(parts):
            if p == "layers" and j + 1 < len(parts):
                layer_idx = int(parts[j + 1])
                break

        if layer_idx is None:
            continue

        # Get input activations at this layer for inference probes
        # The input to self_attn at layer L is the hidden state at layer L-1
        # But our activations are the OUTPUT of each layer
        # For layer L, the input activations are the output of layer L-1
        # For layer 0, input is the embedding
        input_layer_idx = layer_idx - 1 if layer_idx > 0 else 0
        if input_layer_idx not in base_inference_acts:
            # Use same layer if previous not available
            input_layer_idx = layer_idx
        if input_layer_idx not in base_inference_acts:
            continue

        # Compute projection magnitude for each inference probe
        per_probe = []
        for i, problem in enumerate(problems):
            inp_vec = base_inference_acts[input_layer_idx][i]  # [hidden]
            inp_f32 = inp_vec.astype(mx.float32)

            # Project onto top delta directions
            coords = inp_f32 @ top_input_dirs.T  # [top_k]
            mx.eval(coords)
            proj_magnitude = float(mx.sqrt(mx.sum(coords * coords)).item())
            inp_norm = float(mx.sqrt(mx.sum(inp_f32 * inp_f32)).item())
            relative_proj = proj_magnitude / (inp_norm + 1e-12)

            per_probe.append({
                "problem_id": problem.problem_id,
                "problem_type": problem.problem_type,
                "correctness_change": _classify(problem.problem_id),
                "projection_magnitude": round(proj_magnitude, 6),
                "input_norm": round(inp_norm, 6),
                "relative_projection": round(relative_proj, 6),
            })

        module_name = f"{key_base}.weight"
        sigma_k = sigma_k_by_module.get(module_name)

        per_module[key_base] = {
            "layer_idx": layer_idx,
            "delta_shape": list(delta.shape),
            "delta_top_svs": top_svs,
            "sigma_k": round(sigma_k, 6) if sigma_k else None,
            "per_probe": per_probe,
        }

        # Summary: mean projection by group
        groups: dict[str, list[float]] = {}
        for pp in per_probe:
            groups.setdefault(pp["correctness_change"], []).append(
                pp["relative_projection"]
            )
        summary = {
            g: round(sum(v) / len(v), 6) for g, v in groups.items()
        }
        logger.info(
            "  %s: top_sv=%.4f | mean rel_proj by group: %s",
            key_base, top_svs[0] if top_svs else 0.0, summary,
        )

    return {"per_module": per_module}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    adapter_path = _validate_adapter_path(args.adapter_path.resolve())
    output_dir = args.output_dir
    model_path = args.model_path
    eval_data = args.eval_data

    logger.info("Null-Space Geometry Investigation")
    logger.info("=" * 70)
    logger.info("Model path: %s", model_path)
    logger.info("Adapter path: %s", adapter_path)
    logger.info("Eval data: %s", eval_data)
    logger.info("Output dir: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _load_backend()

    # -----------------------------------------------------------------------
    # Step 1: Generate probe sets
    # -----------------------------------------------------------------------
    logger.info("Step 1: Loading probe sets")
    eval_samples = _load_eval_samples(eval_data)
    eval_texts = [s["text"] for s in eval_samples if isinstance(s.get("text"), str)]

    inference_problems = _create_inference_probes()
    inference_texts = _get_inference_probe_texts(inference_problems)

    logger.info("  Eval probes: %d texts", len(eval_texts))
    logger.info("  Inference probes: %d texts", len(inference_texts))

    # -----------------------------------------------------------------------
    # Step 2: Run correctness eval on base and adapted models
    # -----------------------------------------------------------------------
    logger.info("Step 2: Running correctness eval to identify lost/gained probes")
    baseline_result = _run_correctness_eval(
        backend, model_path, None, inference_problems,
    )
    logger.info(
        "  Baseline: %d/%d correct",
        baseline_result.n_correct, baseline_result.n_total,
    )

    adapted_result = _run_correctness_eval(
        backend, model_path, str(adapter_path), inference_problems,
    )
    # Re-evaluate with baseline knowledge
    adapted_with_baseline = _run_correctness_eval_with_baseline(
        backend, model_path, str(adapter_path), inference_problems,
        baseline_result.correct_ids,
    )
    logger.info(
        "  Adapted: %d/%d correct (lost=%d, gained=%d)",
        adapted_result.n_correct,
        adapted_result.n_total,
        len(baseline_result.correct_ids - adapted_result.correct_ids),
        len(adapted_result.correct_ids - baseline_result.correct_ids),
    )

    # -----------------------------------------------------------------------
    # Step 3: Collect activations (both probe sets, both models)
    # -----------------------------------------------------------------------
    logger.info("Step 3: Collecting activations (base model)")
    base_model, base_tokenizer = backend.load_model(model_path)

    base_eval_acts = _collect_activations(backend, base_model, base_tokenizer, eval_texts)
    logger.info("  Base eval activations: %d layers", len(base_eval_acts))

    base_inference_acts = _collect_activations(
        backend, base_model, base_tokenizer, inference_texts,
    )
    logger.info("  Base inference activations: %d layers", len(base_inference_acts))

    del base_model, base_tokenizer
    gc.collect()

    logger.info("Step 3b: Collecting activations (adapted model)")
    adapted_model, adapted_tokenizer = backend.load_model(model_path, str(adapter_path))

    adapted_eval_acts = _collect_activations(
        backend, adapted_model, adapted_tokenizer, eval_texts,
    )
    logger.info("  Adapted eval activations: %d layers", len(adapted_eval_acts))

    adapted_inference_acts = _collect_activations(
        backend, adapted_model, adapted_tokenizer, inference_texts,
    )
    logger.info("  Adapted inference activations: %d layers", len(adapted_inference_acts))

    del adapted_model, adapted_tokenizer
    gc.collect()

    # -----------------------------------------------------------------------
    # Step 4: Run all analyses
    # -----------------------------------------------------------------------
    logger.info("Step 4: Running analyses")

    a1 = analysis_1_cka_comparison(
        backend,
        base_eval_acts, adapted_eval_acts,
        base_inference_acts, adapted_inference_acts,
    )

    a2 = analysis_2_per_probe_delta(
        backend,
        base_inference_acts, adapted_inference_acts,
        inference_problems,
        baseline_result.correct_ids,
        adapted_result.correct_ids,
        adapter_path,
    )

    a3 = analysis_3_subspace_overlap(
        backend,
        base_eval_acts, base_inference_acts,
        inference_problems,
    )

    a4 = analysis_4_lora_delta_projection(
        backend,
        base_inference_acts,
        inference_problems,
        baseline_result.correct_ids,
        adapted_result.correct_ids,
    )

    # -----------------------------------------------------------------------
    # Step 5: Write results
    # -----------------------------------------------------------------------
    results = {
        "model_path": model_path,
        "adapter_path": str(adapter_path),
        "eval_data": eval_data,
        "phase5_probe_seed": PHASE5_PROBE_SEED,
        "phase5_probe_count": PHASE5_PROBE_COUNT,
        "baseline_correct": sorted(baseline_result.correct_ids),
        "adapted_correct": sorted(adapted_result.correct_ids),
        "n_lost": len(baseline_result.correct_ids - adapted_result.correct_ids),
        "n_gained": len(adapted_result.correct_ids - baseline_result.correct_ids),
        "adapted_with_baseline_correct": sorted(adapted_with_baseline.correct_ids),
        "analysis_1_cka_comparison": a1,
        "analysis_2_per_probe_delta": a2,
        "analysis_3_subspace_overlap": a3,
        "analysis_4_lora_delta_projection": a4,
    }

    output_path = output_dir / "investigation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", output_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # A1 summary: worst eps ratio
    if a1["per_layer"]:
        worst_ratio_layer = max(
            a1["per_layer"].items(),
            key=lambda kv: kv[1].get("eps_ratio") or 0,
        )
        logger.info(
            "A1: Worst eps ratio = %.2fx at layer %s (eps_eval=%.4f, eps_inf=%.4f)",
            worst_ratio_layer[1].get("eps_ratio") or 0,
            worst_ratio_layer[0],
            worst_ratio_layer[1]["eps_eval"],
            worst_ratio_layer[1]["eps_inference"],
        )

    # A2 summary: mean cosine by group at worst layer
    if a2["probes"]:
        all_layers = list(a2["probes"][0]["per_layer"].keys())
        if all_layers:
            worst_layer = min(
                all_layers,
                key=lambda l: min(
                    p["per_layer"][l]["cosine_sim"]
                    for p in a2["probes"]
                ),
            )
            by_group: dict[str, list[float]] = {}
            for p in a2["probes"]:
                by_group.setdefault(p["correctness_change"], []).append(
                    p["per_layer"][worst_layer]["cosine_sim"]
                )
            logger.info("A2: Per-group mean cosine at worst layer (%s):", worst_layer)
            for g, vals in sorted(by_group.items()):
                logger.info("    %s: %.4f (n=%d)", g, sum(vals) / len(vals), len(vals))

    # A3 summary: mean EVR across layers
    if a3["per_layer"]:
        mean_evrs = [
            v["mean_explained_variance_ratio"]
            for v in a3["per_layer"].values()
        ]
        logger.info(
            "A3: Mean subspace overlap EVR = %.4f (min=%.4f, max=%.4f)",
            sum(mean_evrs) / len(mean_evrs),
            min(mean_evrs),
            max(mean_evrs),
        )

    # A4 summary: lost vs stable relative projection
    if a4["per_module"]:
        lost_projs = []
        stable_projs = []
        for mod_data in a4["per_module"].values():
            for pp in mod_data["per_probe"]:
                if pp["correctness_change"] == "lost":
                    lost_projs.append(pp["relative_projection"])
                elif pp["correctness_change"] in ("stable_correct", "stable_incorrect"):
                    stable_projs.append(pp["relative_projection"])
        if lost_projs and stable_projs:
            logger.info(
                "A4: Mean relative projection — lost=%.4f, stable=%.4f (ratio=%.2fx)",
                sum(lost_projs) / len(lost_projs),
                sum(stable_projs) / len(stable_projs),
                (sum(lost_projs) / len(lost_projs))
                / (sum(stable_projs) / len(stable_projs) + 1e-12),
            )

    logger.info("=" * 70)
    logger.info("Done. Full results at %s", output_path)


def _run_correctness_eval_with_baseline(
    backend, model_path, adapter_path, problems, baseline_correct_ids,
):
    """Run adapted eval with baseline comparison."""
    from modelcypher.core.domain.training.online_eval import evaluate_correctness

    model, tokenizer = backend.load_model(model_path, adapter_path)

    def _generate(prompt: str, max_tokens: int) -> str:
        return backend.generate(model, tokenizer, prompt, max_tokens=max_tokens)

    result = evaluate_correctness(
        problems=problems,
        generate_fn=_generate,
        epoch=0,
        baseline_correct_ids=baseline_correct_ids,
        max_tokens=SEQ_LENGTH,
    )
    model = None
    tokenizer = None
    gc.collect()
    return result


if __name__ == "__main__":
    main()
