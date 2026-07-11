# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Per-fact retained fraction and interference matrix for LKM falsifier.

Measures how much of each fact's learning signal (base-weight gradient)
is capturable by the LoRA rank-r subspace, and whether failed facts
interfere with each other in that subspace.

Key observables:
  RF_i(r) = ||P_r(g_i)||_beh / ||g_i||_beh
  I_ij = cos(P_r(g_i), P_r(g_j))

Usage:
    poetry run python scripts/lkm/measure_fact_geometry.py \\
        --model /path/to/model \\
        --adapter /path/to/adapter \\
        --data data/lkm/phonebook_4000tok.jsonl \\
        --output /tmp/fact_geometry.json \\
        --max-facts 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_PHONE_PATTERN = re.compile(r"\d{3}-\d{3}-\d{4}")


def _extract_name(text: str) -> str:
    """Extract person name from training text."""
    try:
        return text.split("phone number of ")[1].split("?")[0].strip()
    except (IndexError, AttributeError):
        return "unknown"


def _extract_phone(text: str) -> str | None:
    """Extract phone number from training text."""
    match = _PHONE_PATTERN.search(text.split("Answer:")[-1])
    return match.group() if match else None


def _get_nested(tree: dict | list, path: str):
    """Navigate a nested dict/list by dot-separated path."""
    current = tree
    for part in path.split("."):
        if isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def find_lora_layers(model):
    """Find all LoRALinear layers and their module paths.

    Returns:
        Dict mapping dotted path string to LoRALinear module instance.
    """
    from mlx_lm.tuner.lora import LoRALinear

    layers = {}

    def _visitor(path, module):
        if isinstance(module, LoRALinear):
            layers[path] = module

    model.apply_to_modules(_visitor)
    return layers


def compute_q_matrices(lora_layers: dict) -> dict:
    """Compute orthonormal basis Q for each LoRA layer's column space.

    For lora_a [in_dim, r], computes Q [in_dim, r] via QR decomposition.
    The projector onto the LoRA subspace is P = Q @ Q.T.

    Returns:
        Dict mapping layer path to Q array [in_dim, r].
    """
    import mlx.core as mx

    q_matrices = {}
    for path, layer in lora_layers.items():
        Q, _ = mx.linalg.qr(layer.lora_a, stream=mx.cpu)
        q_matrices[path] = Q
    mx.eval(*q_matrices.values())
    return q_matrices


def collect_activations(model, input_ids, lora_layers: dict) -> dict:
    """Forward pass collecting input activations at each LoRA layer.

    Temporarily patches LoRALinear.__call__ to store inputs.

    Args:
        model: The loaded model (with adapter).
        input_ids: Token IDs [1, seq_len].
        lora_layers: Dict mapping path to LoRALinear instance.

    Returns:
        Dict mapping layer path to input activations [seq_len, in_dim].
    """
    import mlx.core as mx
    from mlx_lm.tuner.lora import LoRALinear

    activations = {}
    id_to_path = {id(layer): path for path, layer in lora_layers.items()}

    orig_call = LoRALinear.__call__

    def _hooked_call(self, x):
        path = id_to_path.get(id(self))
        if path is not None:
            activations[path] = x
        return orig_call(self, x)

    LoRALinear.__call__ = _hooked_call
    try:
        logits = model(input_ids)
        mx.eval(logits, *activations.values())
    finally:
        LoRALinear.__call__ = orig_call

    # Squeeze batch dim: [1, seq, dim] -> [seq, dim]
    return {p: a.squeeze(0) for p, a in activations.items()}


def compute_base_weight_grads(model, input_ids, target_ids) -> tuple:
    """Compute loss and gradients w.r.t. base weights at LoRA layers.

    Assumes base weights are unfrozen and LoRA params are frozen.

    Args:
        model: Model with appropriate freeze/unfreeze state.
        input_ids: [1, seq_len - 1].
        target_ids: [seq_len - 1].

    Returns:
        (loss_value: float, grads: nested dict matching model parameter tree)
    """
    import mlx.core as mx
    import mlx.nn as nn

    def loss_fn(model, ids, targets):
        logits = model(ids)
        logits = logits.squeeze(0)  # [seq, vocab]
        return mx.mean(nn.losses.cross_entropy(logits, targets))

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_grad_fn(model, input_ids, target_ids)
    mx.eval(loss, grads)
    return float(loss.item()), grads


def compute_fact_metrics(
    grads: dict,
    activations: dict,
    q_matrices: dict,
    lora_paths: list[str],
) -> tuple[float, list]:
    """Compute retained fraction and compressed gradients for one fact.

    Args:
        grads: Gradient tree from compute_base_weight_grads.
        activations: Dict of per-layer input activations [seq, in_dim].
        q_matrices: Dict of per-layer Q matrices [in_dim, r].
        lora_paths: Ordered list of LoRA layer paths.

    Returns:
        (rf, compressed_grads) where:
            rf: retained fraction (scalar in [0, 1])
            compressed_grads: list of arrays [out_dim, r] per layer
    """
    import mlx.core as mx

    beh_full_sum = 0.0
    beh_proj_sum = 0.0
    compressed = []

    for path in lora_paths:
        Q = q_matrices[path]  # [in_dim, r]
        A = activations[path]  # [seq, in_dim]
        grad_path = path + ".linear.weight"
        g = _get_nested(grads, grad_path)  # [out_dim, in_dim]

        # Compressed gradient: g @ Q [out_dim, r]
        c = g @ Q
        compressed.append(c)

        # Behavioral norm of full gradient: ||A @ g.T||_F
        full_output = A @ g.T  # [seq, out_dim]
        beh_full = mx.sqrt(mx.sum(full_output * full_output))

        # Behavioral norm of projected gradient: ||A @ Q @ c.T||_F
        # = ||A @ Q @ Q.T @ g.T||_F = ||(A @ Q) @ c.T||_F
        AQ = A @ Q  # [seq, r]
        proj_output = AQ @ c.T  # [seq, out_dim]
        beh_proj = mx.sqrt(mx.sum(proj_output * proj_output))

        mx.eval(beh_full, beh_proj, c)
        beh_full_sum += beh_full.item()
        beh_proj_sum += beh_proj.item()

    rf = beh_proj_sum / beh_full_sum if beh_full_sum > 0 else 0.0
    return rf, compressed


def compute_interference_matrix(
    all_compressed: list[list],
    lora_paths: list[str],
) -> list[list[float]]:
    """Compute pairwise cosine similarity of projected gradients.

    Uses the compressed representation c_i^l = g_i^l @ Q^l to avoid
    materializing full projected gradients.

    Identity: <P_g_i, P_g_j> = sum_l tr((c_i^l).T @ c_j^l)

    Args:
        all_compressed: List (per fact) of lists (per layer) of arrays [out_dim, r].
        lora_paths: Ordered list of layer paths (for alignment).

    Returns:
        Symmetric matrix [n_facts, n_facts] of cosine similarities.
    """
    import mlx.core as mx

    n = len(all_compressed)
    n_layers = len(lora_paths)

    # Precompute norms
    norms = []
    for i in range(n):
        norm_sq = 0.0
        for l in range(n_layers):
            c = all_compressed[i][l]
            norm_sq += mx.sum(c * c).item()
        norms.append(math.sqrt(norm_sq))

    # Compute pairwise cosines
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            dot = 0.0
            for l in range(n_layers):
                ci = all_compressed[i][l]
                cj = all_compressed[j][l]
                dot += mx.sum(ci * cj).item()
            denom = norms[i] * norms[j]
            cos = dot / denom if denom > 0 else 0.0
            cos = round(cos, 6)
            matrix[i][j] = cos
            matrix[j][i] = cos

    return matrix


def measure_fact_geometry(
    model_path: str,
    adapter_path: str,
    data_path: str,
    output_path: str,
    max_facts: int | None = None,
) -> dict:
    """Compute per-fact RF and interference matrix for a trained adapter.

    Args:
        model_path: Path to base model.
        adapter_path: Path to adapter directory.
        data_path: Path to training JSONL (phonebook format).
        output_path: Path to write fact_geometry.json.
        max_facts: Limit number of facts (for smoke testing).

    Returns:
        The output dict (also written to output_path).
    """
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    # 1. Load model + adapter
    print(f"Loading model: {model_path}")
    print(f"Loading adapter: {adapter_path}")
    model, tokenizer = mlx_load(model_path, adapter_path=adapter_path)
    model.train()

    # 2. Find LoRA layers and compute Q matrices
    lora_layers = find_lora_layers(model)
    lora_paths = sorted(lora_layers.keys())
    print(f"Found {len(lora_paths)} LoRA layers")

    q_matrices = compute_q_matrices(lora_layers)
    rank = lora_layers[lora_paths[0]].lora_a.shape[1]
    print(f"LoRA rank: {rank}")

    # 3. Freeze LoRA, unfreeze base weights for gradient computation
    model.freeze()
    for path, layer in lora_layers.items():
        layer.linear.unfreeze()

    # 4. Load training data
    with open(data_path) as f:
        facts = [json.loads(line) for line in f if line.strip()]
    if max_facts is not None:
        facts = facts[:max_facts]
    print(f"Processing {len(facts)} facts")

    # 5. Load eval scores (if available)
    scores_path = Path(adapter_path) / "raw_scores.jsonl"
    scores_by_name: dict[str, bool] = {}
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    scores_by_name[r["name"]] = r["exact_match"]
        print(f"Loaded {len(scores_by_name)} eval scores")

    # 6. Process each fact
    fact_results = []
    all_compressed = []

    for i, fact in enumerate(facts):
        text = fact["text"]
        name = _extract_name(text)
        phone = _extract_phone(text)

        # Tokenize
        tokens = tokenizer.encode(text)
        input_ids = mx.array(tokens[:-1])[None]
        target_ids = mx.array(tokens[1:])

        # Collect activations (forward pass with hooks)
        activations = collect_activations(model, input_ids, lora_layers)

        # Compute loss and base weight gradients
        loss, grads = compute_base_weight_grads(model, input_ids, target_ids)

        # Compute RF and compressed grads
        rf, compressed = compute_fact_metrics(
            grads, activations, q_matrices, lora_paths
        )

        fact_results.append({
            "name": name,
            "phone": phone,
            "rf": round(rf, 6),
            "exact_match": scores_by_name.get(name),
            "loss": round(loss, 6),
        })
        all_compressed.append(compressed)

        if (i + 1) % 10 == 0 or (i + 1) == len(facts):
            print(
                f"  [{i + 1}/{len(facts)}] {name}: RF={rf:.4f} "
                f"loss={loss:.4f} EM={scores_by_name.get(name, '?')}"
            )

        # Free large intermediates
        del activations, grads
        mx.clear_cache()

    # 7. Compute interference matrix
    print("Computing interference matrix...")
    interference = compute_interference_matrix(all_compressed, lora_paths)

    # 8. Build summary
    passed = [r for r in fact_results if r["exact_match"] is True]
    failed = [r for r in fact_results if r["exact_match"] is False]

    # Off-diagonal interference values
    n = len(fact_results)
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(interference[i][j])

    summary = {
        "mean_rf": round(
            sum(r["rf"] for r in fact_results) / len(fact_results), 6
        ),
        "mean_rf_failed": (
            round(sum(r["rf"] for r in failed) / len(failed), 6)
            if failed
            else None
        ),
        "mean_rf_passed": (
            round(sum(r["rf"] for r in passed) / len(passed), 6)
            if passed
            else None
        ),
        "mean_interference": (
            round(sum(off_diag) / len(off_diag), 6) if off_diag else 0.0
        ),
        "max_interference": round(max(off_diag), 6) if off_diag else 0.0,
        "n_high_interference_pairs": sum(1 for v in off_diag if v > 0.5),
    }

    output = {
        "retained_fractions": fact_results,
        "interference_matrix": interference,
        "summary": summary,
    }

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    print("\nSummary:")
    print(f"  mean RF: {summary['mean_rf']:.4f}")
    print(f"  mean RF (passed): {summary['mean_rf_passed']}")
    print(f"  mean RF (failed): {summary['mean_rf_failed']}")
    print(f"  mean interference: {summary['mean_interference']:.4f}")
    print(f"  max interference: {summary['max_interference']:.4f}")
    print(f"  high-interference pairs: {summary['n_high_interference_pairs']}")
    print(f"Wrote: {output_path}")

    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Measure per-fact retained fraction and interference."
    )
    parser.add_argument(
        "--model", required=True, help="Path to base model."
    )
    parser.add_argument(
        "--adapter", required=True, help="Path to adapter directory."
    )
    parser.add_argument(
        "--data", required=True, help="Path to training JSONL."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write fact_geometry.json."
    )
    parser.add_argument(
        "--max-facts",
        type=int,
        default=None,
        help="Limit number of facts (for smoke testing).",
    )

    args = parser.parse_args()
    measure_fact_geometry(
        model_path=args.model,
        adapter_path=args.adapter,
        data_path=args.data,
        output_path=args.output,
        max_facts=args.max_facts,
    )


if __name__ == "__main__":
    main()
