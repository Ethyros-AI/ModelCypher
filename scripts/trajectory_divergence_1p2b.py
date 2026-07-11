#!/usr/bin/env python3
"""Trajectory Divergence: Per-layer comparison of base vs adapted 1.2B.

For specific inputs (degenerate, regressed, improved), traces hidden states
through both base and adapted models. Finds WHERE the trajectory diverges.

Measures per layer:
1. CKA between base and adapted activations
2. Velocity norm (||h_{l+1} - h_l||) for each model
3. Velocity ratio (adapted/base) — where does adapted speed up or slow down?
4. Cosine similarity between base and adapted hidden states

Also generates text from both models to confirm behavior.
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

BASE_MODEL = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
ADAPTER_PATH = "/Volumes/CodeCypher/experiments/1p2b-cayley-v1"
OUTPUT_DIR = "/Volumes/CodeCypher/experiments/1p2b-trajectory-divergence"

# Test inputs: 2 degenerate, 2 regressed, 2 improved
TEST_INPUTS = [
    # Degenerate outputs from evaluation
    {
        "id": "degen_math",
        "category": "degenerate",
        "prompt": "What is 15% of 80?",
        "note": "Adapted produced infinite .00 loop",
    },
    {
        "id": "degen_gk",
        "category": "degenerate",
        "prompt": "How many continents are there?",
        "note": "Adapted produced semantic wandering",
    },
    # Regressed (baseline correct, adapted wrong)
    {
        "id": "regress_bat",
        "category": "regressed",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "note": "Baseline $0.05, adapted $0.90",
    },
    {
        "id": "regress_mult",
        "category": "regressed",
        "prompt": "What is 7 times 8?",
        "note": "Baseline correct, adapted degenerate",
    },
    # Improved (adapted better than baseline)
    {
        "id": "improve_mp",
        "category": "improved",
        "prompt": "If it is raining, then the ground is wet. It is raining. Is the ground wet?",
        "note": "Modus ponens - adapted improved",
    },
    {
        "id": "improve_mt",
        "category": "improved",
        "prompt": "If it is raining, then the ground is wet. The ground is not wet. Is it raining?",
        "note": "Modus tollens - adapted improved",
    },
]

MAX_TOKENS = 128


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class LayerDivergence:
    """Per-layer divergence between base and adapted hidden states."""

    layer_idx: int
    cka: float
    cosine_sim: float  # mean cosine similarity between base and adapted h
    base_velocity_norm: float  # ||h_{l} - h_{l-1}|| for base
    adapted_velocity_norm: float  # ||h_{l} - h_{l-1}|| for adapted
    velocity_ratio: float  # adapted / base (>1 = adapted takes bigger steps)
    base_state_norm: float  # ||h_l|| for base
    adapted_state_norm: float  # ||h_l|| for adapted


@dataclass
class InputDivergenceReport:
    """Full divergence report for one input."""

    input_id: str
    category: str
    prompt: str
    base_response: str
    adapted_response: str
    base_response_length: int
    adapted_response_length: int
    per_layer: list[LayerDivergence]
    min_cka_layer: int
    min_cka_value: float
    max_velocity_ratio_layer: int
    max_velocity_ratio: float


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    import mlx.core as mx

    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.cka import (
        compute_linear_cka_from_activations,
    )

    initialize_default_backend()

    print("=" * 70)
    print("TRAJECTORY DIVERGENCE: Base vs Adapted LFM2-1.2B")
    print("=" * 70)

    # Verify paths
    if not Path(BASE_MODEL).exists():
        print(f"ERROR: Base model not found at {BASE_MODEL}")
        sys.exit(1)

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    backend = get_default_backend()

    # ── Load base model ──
    print("\nLoading base model...")
    t0 = time.time()
    loader = ModelLoader(backend)
    model_base, tokenizer = loader.load_model(BASE_MODEL)
    print(f"  Base model loaded in {time.time()-t0:.1f}s")

    # ── Load adapted model ──
    print("Loading adapted model...")
    t0 = time.time()
    model_adapted, _ = loader.load_model(BASE_MODEL, adapter_path=ADAPTER_PATH)
    print(f"  Adapted model loaded in {time.time()-t0:.1f}s")

    # Count layers
    n_layers = len(model_base.model.layers)
    print(f"  Model has {n_layers} layers")

    reports: list[InputDivergenceReport] = []

    for test_input in TEST_INPUTS:
        input_id = test_input["id"]
        prompt = test_input["prompt"]
        category = test_input["category"]

        print(f"\n{'─' * 60}")
        print(f"Input: {input_id} ({category})")
        print(f"Prompt: {prompt[:80]}")

        # ── Generate responses ──
        print("  Generating base response...", end=" ", flush=True)
        base_response = backend.generate(
            model_base, tokenizer, prompt, max_tokens=MAX_TOKENS
        )
        print(f"({len(base_response)} chars)")

        print("  Generating adapted response...", end=" ", flush=True)
        adapted_response = backend.generate(
            model_adapted, tokenizer, prompt, max_tokens=MAX_TOKENS
        )
        print(f"({len(adapted_response)} chars)")

        print(f"  Base: {base_response[:100]}...")
        print(f"  Adapted: {adapted_response[:100]}...")

        # ── Collect per-layer activations on the PROMPT ──
        # We use the prompt only (not the generated text) to compare
        # how the models process the SAME input differently.
        print("  Collecting base activations...", end=" ", flush=True)
        base_acts = backend.collect_hidden_activations(
            model_base, tokenizer, [prompt]
        )
        print(f"done ({len(base_acts)} layers)")

        print("  Collecting adapted activations...", end=" ", flush=True)
        adapted_acts = backend.collect_hidden_activations(
            model_adapted, tokenizer, [prompt]
        )
        print(f"done ({len(adapted_acts)} layers)")

        # ── Per-layer divergence ──
        layer_divergences: list[LayerDivergence] = []
        prev_base_h = None
        prev_adapted_h = None

        for layer_idx in sorted(base_acts.keys()):
            if layer_idx not in adapted_acts:
                continue

            # Get activations: [batch=1, seq, hidden]
            base_h = base_acts[layer_idx]
            adapted_h = adapted_acts[layer_idx]

            # Mean-pool over sequence → [batch=1, hidden]
            base_pooled = backend.mean(base_h, axis=1)
            adapted_pooled = backend.mean(adapted_h, axis=1)
            backend.eval(base_pooled, adapted_pooled)

            # CKA (needs at least 2 samples, use token-level instead of batch-level)
            # Reshape to [seq, hidden] for CKA
            base_tokens = backend.reshape(base_h, (-1, base_h.shape[-1]))
            adapted_tokens = backend.reshape(adapted_h, (-1, adapted_h.shape[-1]))
            backend.eval(base_tokens, adapted_tokens)

            try:
                cka = compute_linear_cka_from_activations(
                    base_tokens, adapted_tokens, backend
                )
            except Exception:
                cka = -1.0

            # Cosine similarity (on mean-pooled)
            base_flat = backend.reshape(base_pooled, (-1,))
            adapted_flat = backend.reshape(adapted_pooled, (-1,))
            backend.eval(base_flat, adapted_flat)

            dot = float(mx.sum(base_flat * adapted_flat))
            norm_b = float(mx.sqrt(mx.sum(base_flat * base_flat)))
            norm_a = float(mx.sqrt(mx.sum(adapted_flat * adapted_flat)))
            cosine = dot / (norm_b * norm_a) if (norm_b > 0 and norm_a > 0) else 0.0

            # Velocity: ||h_l - h_{l-1}||
            base_vel = 0.0
            adapted_vel = 0.0
            vel_ratio = 1.0
            if prev_base_h is not None:
                base_diff = base_pooled - prev_base_h
                adapted_diff = adapted_pooled - prev_adapted_h
                backend.eval(base_diff, adapted_diff)
                base_vel = float(mx.sqrt(mx.sum(base_diff * base_diff)))
                adapted_vel = float(mx.sqrt(mx.sum(adapted_diff * adapted_diff)))
                vel_ratio = adapted_vel / base_vel if base_vel > 1e-10 else 1.0

            prev_base_h = base_pooled
            prev_adapted_h = adapted_pooled

            layer_divergences.append(
                LayerDivergence(
                    layer_idx=layer_idx,
                    cka=cka,
                    cosine_sim=cosine,
                    base_velocity_norm=base_vel,
                    adapted_velocity_norm=adapted_vel,
                    velocity_ratio=vel_ratio,
                    base_state_norm=norm_b,
                    adapted_state_norm=norm_a,
                )
            )

        # Find most divergent layer
        cka_layers = [(ld.cka, ld.layer_idx) for ld in layer_divergences if ld.cka >= 0]
        min_cka_val, min_cka_layer = min(cka_layers) if cka_layers else (0.0, -1)

        vel_layers = [
            (ld.velocity_ratio, ld.layer_idx)
            for ld in layer_divergences
            if ld.velocity_ratio > 0
        ]
        max_vel_ratio, max_vel_layer = max(vel_layers) if vel_layers else (1.0, -1)

        report = InputDivergenceReport(
            input_id=input_id,
            category=category,
            prompt=prompt,
            base_response=base_response,
            adapted_response=adapted_response,
            base_response_length=len(base_response),
            adapted_response_length=len(adapted_response),
            per_layer=layer_divergences,
            min_cka_layer=min_cka_layer,
            min_cka_value=min_cka_val,
            max_velocity_ratio_layer=max_vel_layer,
            max_velocity_ratio=max_vel_ratio,
        )
        reports.append(report)

        # Print per-layer table
        print(f"\n  {'Layer':>5} {'CKA':>8} {'Cosine':>8} {'BaseVel':>8} {'AdaptVel':>8} {'VelRatio':>8} {'BaseNorm':>8} {'AdaptNorm':>8}")
        for ld in layer_divergences:
            marker = ""
            if ld.layer_idx == min_cka_layer:
                marker = " ← MIN CKA"
            if ld.layer_idx == max_vel_layer:
                marker += " ← MAX VEL"
            print(
                f"  {ld.layer_idx:>5} "
                f"{ld.cka:>8.4f} "
                f"{ld.cosine_sim:>8.4f} "
                f"{ld.base_velocity_norm:>8.2f} "
                f"{ld.adapted_velocity_norm:>8.2f} "
                f"{ld.velocity_ratio:>8.3f} "
                f"{ld.base_state_norm:>8.2f} "
                f"{ld.adapted_state_norm:>8.2f}"
                f"{marker}"
            )

        print(f"\n  Min CKA: layer {min_cka_layer} = {min_cka_val:.4f}")
        print(f"  Max velocity ratio: layer {max_vel_layer} = {max_vel_ratio:.3f}")

    # ── Cross-input Summary ──
    print("\n" + "=" * 70)
    print("CROSS-INPUT SUMMARY")
    print("=" * 70)

    print(f"\n{'Input ID':<20} {'Category':<12} {'MinCKA':>8} {'@Layer':>7} {'MaxVelR':>8} {'@Layer':>7}")
    for r in reports:
        print(
            f"{r.input_id:<20} "
            f"{r.category:<12} "
            f"{r.min_cka_value:>8.4f} "
            f"{r.min_cka_layer:>7} "
            f"{r.max_velocity_ratio:>8.3f} "
            f"{r.max_velocity_ratio_layer:>7}"
        )

    # Per-layer CKA averaged by category
    categories = sorted(set(r.category for r in reports))
    print("\nPer-layer CKA by category:")
    layer_indices = [ld.layer_idx for ld in reports[0].per_layer] if reports else []
    print(f"  {'Layer':>5}", end="")
    for cat in categories:
        print(f"  {cat:>12}", end="")
    print()

    for li, layer_idx in enumerate(layer_indices):
        print(f"  {layer_idx:>5}", end="")
        for cat in categories:
            cat_reports = [r for r in reports if r.category == cat]
            avg_cka = sum(r.per_layer[li].cka for r in cat_reports) / len(cat_reports)
            print(f"  {avg_cka:>12.4f}", end="")
        print()

    # Save full report
    full_report = {
        "base_model": BASE_MODEL,
        "adapter_path": ADAPTER_PATH,
        "n_inputs": len(reports),
        "inputs": [],
    }
    for r in reports:
        entry = {
            "input_id": r.input_id,
            "category": r.category,
            "prompt": r.prompt,
            "base_response": r.base_response,
            "adapted_response": r.adapted_response,
            "min_cka_layer": r.min_cka_layer,
            "min_cka_value": r.min_cka_value,
            "max_velocity_ratio_layer": r.max_velocity_ratio_layer,
            "max_velocity_ratio": r.max_velocity_ratio,
            "per_layer": [asdict(ld) for ld in r.per_layer],
        }
        full_report["inputs"].append(entry)

    report_path = out / "trajectory_divergence.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
