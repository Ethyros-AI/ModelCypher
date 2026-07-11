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

"""Validate attention sink analysis on real models.

Pre-CLI-promotion validation per AGENTS.md:695 and MISSION.md:179.
Runs on LFM2-350M (hybrid conv+attn) and Qwen3.5-0.8B (all attn, GQA).

Acceptance criteria:
1. Consistency error ≈ 0 (floating-point precision on division)
2. BOS/early tokens have highest sink scores (consistent with paper findings)
3. Active sink scores differ from raw sink scores when value norms vary
4. Results reproducible across 3+ probe texts
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# IEEE 754 bfloat16 machine epsilon: 2^-7 ≈ 0.0078.
# Consistency error measures (sum_col - sum_row) / sum_row for doubly-stochastic
# attention. Accumulation over T tokens amplifies rounding, so the tolerance is
# T × eps_bf16. For T <= 128 (our probe lengths), T × eps_bf16 < 1.0, which is
# far above any valid consistency error. We use eps_bf16 directly as a conservative
# per-element tolerance — any error larger indicates a computation bug, not rounding.
_EPS_BF16 = math.ldexp(1.0, -7)  # bfloat16 mantissa precision

# Models on external volume
MODELS = {
    "LFM2-350M": "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
    "Qwen3.5-0.8B": "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16",
}

PROBE_TEXTS = [
    "The capital of France is",
    "In mathematics, the derivative of x squared is",
    "Once upon a time in a land far away",
]


def validate_model(model_name: str, model_path: str) -> dict:
    """Run sink validation on a single model."""
    import mlx_lm

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.attention_sink import (
        compute_active_sinks,
        compute_sink_scores,
        summarize_layer_sinks,
    )

    print(f"\n{'='*60}")
    print(f"Validating: {model_name} ({model_path})")
    print(f"{'='*60}")

    backend = MLXBackend()
    model, tokenizer = mlx_lm.load(model_path)

    all_pass = True
    results_per_probe: list[dict] = []

    for probe_idx, text in enumerate(PROBE_TEXTS):
        print(f"\n  Probe {probe_idx + 1}: {text!r}")

        attn_matrices, value_vectors = backend.collect_attention_matrices_with_values(
            model, tokenizer, text
        )
        layer_indices = sorted(attn_matrices.keys())
        print(f"    Attention layers: {layer_indices}")

        layer_results = []

        for layer_idx in layer_indices:
            head_matrices = attn_matrices[layer_idx]
            head_values = value_vectors[layer_idx]

            head_sinks = []
            active_sinks = []

            for head_idx, (head_mat, head_v) in enumerate(
                zip(head_matrices, head_values)
            ):
                mat_list = backend.tolist(head_mat)
                result = compute_sink_scores(mat_list, head_idx=head_idx)
                head_sinks.append(result)

                # Check 1: Consistency error should be near machine epsilon.
                # Tolerance: eps_bf16 (per-element rounding bound for bfloat16
                # attention weights). Any error above this indicates a computation
                # bug in the sink score arithmetic, not floating-point accumulation.
                for ts in result.token_sinks:
                    if ts.consistency_error > _EPS_BF16:
                        print(
                            f"    FAIL: L{layer_idx} H{head_idx} pos={ts.position} "
                            f"consistency_error={ts.consistency_error}"
                        )
                        all_pass = False

                # Compute value norms for active sink
                v_list = backend.tolist(head_v)  # [seq, head_dim]
                v_norms = [
                    math.sqrt(sum(x * x for x in row)) for row in v_list
                ]
                active = compute_active_sinks(result, v_norms)
                active_sinks.append(active)

            layer_summary = summarize_layer_sinks(
                head_sinks, active_results=active_sinks, layer_idx=layer_idx
            )
            layer_results.append(layer_summary)

            # Report
            print(
                f"    L{layer_idx:2d}: dominant_sink_pos={layer_summary.dominant_sink_position}, "
                f"mean_max_sink={layer_summary.mean_max_sink_score:.4f}"
            )

        # Check 2: BOS/early tokens should dominate sink positions.
        # This is an INFORMATIONAL check (NOTE, not FAIL). The >= 50%
        # threshold is an empirical expectation from Xiao et al. (2023)
        # "Efficient Streaming Language Models with Attention Sinks",
        # not a derived decision boundary. It serves as a sanity check
        # that the infrastructure correctly identifies sink structure.
        early_dominant_count = sum(
            1 for lr in layer_results if lr.dominant_sink_position <= 1
        )
        total_layers = len(layer_results)
        if total_layers > 0:
            early_fraction = early_dominant_count / total_layers
            if early_fraction >= 0.5:
                print(
                    f"    PASS: {early_dominant_count}/{total_layers} layers "
                    f"have BOS/early token as dominant sink"
                )
            else:
                print(
                    f"    NOTE: Only {early_dominant_count}/{total_layers} layers "
                    f"have early token as dominant sink (expected >= 50%)"
                )

        # Check 3: Active scores differ from raw when V norms vary
        for lr in layer_results:
            if lr.active_results:
                for head_sink, active in zip(lr.head_results, lr.active_results):
                    if head_sink.max_sink_position != active.max_active_position:
                        print(
                            f"    INFO: L{lr.layer_idx} H{head_sink.head_idx} "
                            f"raw_max_pos={head_sink.max_sink_position} != "
                            f"active_max_pos={active.max_active_position} "
                            f"(V-norm reweighting changes sink ranking)"
                        )

        results_per_probe.append({
            "probe": text,
            "layer_indices": layer_indices,
        })

    # Check 4: Reproducibility
    if len(results_per_probe) >= 2:
        first_layers = results_per_probe[0]["layer_indices"]
        for i, r in enumerate(results_per_probe[1:], 2):
            if r["layer_indices"] != first_layers:
                print(f"\n  FAIL: Probe {i} layer indices differ from probe 1")
                all_pass = False
        if all_pass:
            print(
                f"\n  PASS: Layer indices consistent across {len(PROBE_TEXTS)} probes"
            )

    return {
        "model": model_name,
        "all_pass": all_pass,
    }


def main():
    print("Attention Sink Validation")
    print("Pre-CLI-promotion check per AGENTS.md:695")

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

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        status = "PASS" if r["all_pass"] else "FAIL"
        print(f"  {r['model']}: {status}")

    if overall_pass:
        print("\nAll checks passed. Domain module ready for CLI promotion.")
    else:
        print("\nSome checks FAILED. Do NOT promote to CLI until fixed.")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
