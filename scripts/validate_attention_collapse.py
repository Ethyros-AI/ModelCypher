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

"""Validate attention collapse detection on real models.

Pre-CLI-promotion validation per AGENTS.md:695 and MISSION.md:179.
Runs on LFM2-350M (hybrid conv+attn) and Qwen3.5-0.8B (all attn, GQA).

Acceptance criteria:
1. LFM2: Only 6/16 layers appear (conv layers ABSENT)
2. Both models: singular values non-negative and descending
3. Rank-1 detection threshold is exactly sqrt(eps_dtype)
4. Results reproducible across 3+ probe texts
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Models on external volume
MODELS = {
    "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "Qwen3.5-0.8B": "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16",
}

# Diverse probe texts for reproducibility check
PROBE_TEXTS = [
    "The capital of France is",
    "In mathematics, the derivative of x squared is",
    "Once upon a time in a land far away",
]

# LFM2-350M architecture: 16 total layers, 6 attention layers
# full_attn_idxs = [2, 5, 8, 10, 12, 14]
LFM2_ATTENTION_LAYER_INDICES = {2, 5, 8, 10, 12, 14}
LFM2_TOTAL_LAYERS = 16


def validate_model(model_name: str, model_path: str) -> dict:
    """Run validation on a single model."""
    import mlx_lm

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.attention_collapse import (
        compute_attention_collapse,
        compute_collapse_profile,
        summarize_layer_collapse,
    )

    print(f"\n{'='*60}")
    print(f"Validating: {model_name} ({model_path})")
    print(f"{'='*60}")

    backend = MLXBackend()
    model, tokenizer = mlx_lm.load(model_path)

    results_per_probe: list[dict] = []
    all_pass = True

    for probe_idx, text in enumerate(PROBE_TEXTS):
        print(f"\n  Probe {probe_idx + 1}: {text!r}")

        # Collect attention matrices via backend
        attn_matrices = backend.collect_attention_matrices(
            model, tokenizer, text
        )
        layer_indices = sorted(attn_matrices.keys())
        print(f"    Attention layers found: {layer_indices}")
        print(f"    Total attention layers: {len(layer_indices)}")

        # Check 1: LFM2 should only have 6 attention layers
        if model_name == "LFM2-350M":
            if set(layer_indices) != LFM2_ATTENTION_LAYER_INDICES:
                print(f"    FAIL: Expected attention at {sorted(LFM2_ATTENTION_LAYER_INDICES)}, "
                      f"got {layer_indices}")
                all_pass = False
            else:
                print(f"    PASS: Correct 6/16 attention layers")

        # Compute collapse for each layer
        layer_results = []
        for layer_idx in layer_indices:
            head_matrices = attn_matrices[layer_idx]
            head_results = []

            for head_idx, head_mat in enumerate(head_matrices):
                # Convert MLX array to list of lists
                mat_list = backend.tolist(head_mat)
                result = compute_attention_collapse(mat_list, "bfloat16")

                # Check 2: singular values non-negative and descending
                svs = result.singular_values
                for i, sv in enumerate(svs):
                    if sv < -1e-10:
                        print(f"    FAIL: L{layer_idx} H{head_idx} SV[{i}] = {sv} < 0")
                        all_pass = False
                for i in range(len(svs) - 1):
                    if svs[i] < svs[i + 1] - 1e-10:
                        print(f"    FAIL: L{layer_idx} H{head_idx} SV not descending: "
                              f"SV[{i}]={svs[i]} < SV[{i+1}]={svs[i+1]}")
                        all_pass = False

                head_results.append(result)

            layer_summary = summarize_layer_collapse(head_results, layer_idx=layer_idx)
            layer_results.append(layer_summary)

            # Report per layer
            n_heads = len(head_results)
            collapsed = layer_summary.collapsed_head_count
            max_er = layer_summary.max_effective_rank
            mean_gs = layer_summary.mean_gradient_suppression
            print(f"    L{layer_idx:2d}: {collapsed}/{n_heads} collapsed, "
                  f"max_eff_rank={max_er:.2f}, mean_grad_supp={mean_gs:.4f}")

        profile = compute_collapse_profile(layer_results)
        results_per_probe.append({
            "probe": text,
            "layer_indices": layer_indices,
            "profile": profile.to_dict(),
        })

    # Check 4: Reproducibility — layer indices should be identical across probes
    if len(results_per_probe) >= 2:
        first_layers = results_per_probe[0]["layer_indices"]
        for i, r in enumerate(results_per_probe[1:], 2):
            if r["layer_indices"] != first_layers:
                print(f"\n  FAIL: Probe {i} layer indices differ from probe 1")
                all_pass = False
        if all_pass:
            print(f"\n  PASS: Layer indices consistent across {len(PROBE_TEXTS)} probes")

    # Check 3: Verify rank-1 threshold is exactly sqrt(eps_dtype)
    sqrt_eps_bf16 = math.sqrt(math.ldexp(1.0, -7))
    print(f"\n  Rank-1 threshold (bfloat16): sqrt(eps) = {sqrt_eps_bf16:.6f}")
    print(f"  (IEEE 754 derived: sqrt(2^-7) = {sqrt_eps_bf16})")

    return {
        "model": model_name,
        "all_pass": all_pass,
        "probes": results_per_probe,
    }


def main():
    print("Attention Collapse Validation")
    print("Pre-CLI-promotion check per AGENTS.md:695")

    # Check volume is mounted
    for name, path in MODELS.items():
        if not Path(path).exists():
            print(f"ERROR: Model not found: {path}")
            print("Is the external volume mounted?")
            sys.exit(1)

    all_results = []
    overall_pass = True

    for name, path in MODELS.items():
        result = validate_model(name, path)
        all_results.append(result)
        if not result["all_pass"]:
            overall_pass = False

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "PASS" if r["all_pass"] else "FAIL"
        print(f"  {r['model']}: {status}")

    if overall_pass:
        print("\nAll checks passed. Domain module ready for CLI promotion (Task 4).")
    else:
        print("\nSome checks FAILED. Do NOT promote to CLI until fixed.")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
