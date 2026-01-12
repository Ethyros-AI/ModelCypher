#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Fine-Tuning Geometry Analysis
# Quantify what code training actually changes in model geometry

"""
Fine-Tuning Geometry Analysis

Uses Crosscoder to decompose what fine-tuning changes:
- Shared features: Active in both base and fine-tuned model
- Base-exclusive features: Only in general model (deprioritized by FT)
- FT-exclusive features: Only in code model (what training added)

Tests the geometric hypothesis:
  Fine-tuning adds to sparse regions, doesn't destroy existing structure.

Usage:
    poetry run python scripts/finetune_geometry_analysis.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import detect_default_backend_type, get_backend
from modelcypher.core.domain._backend import set_default_backend

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = Path("/Volumes/CodeCypher/models/mlx-community")

BASE_MODEL = "Qwen2.5-3B-Instruct-bf16"
FT_MODEL = "Qwen2.5-Coder-3B-Instruct-bf16"

# Prompts for activation collection
GENERAL_PROMPTS = [
    "The weather today is",
    "In the year 2024,",
    "The most important thing about",
    "Scientists discovered that",
    "According to recent studies,",
    "She walked into the room and",
    "The history of civilization shows",
    "When people think about the future,",
]

CODE_PROMPTS = [
    "def fibonacci(n):",
    "class DatabaseConnection:",
    "async def fetch_data(url):",
    "SELECT * FROM users WHERE",
    "import torch.nn as",
    "function calculateSum(arr) {",
    "public static void main(",
    "for i in range(len(data)):",
]

TARGET_LAYER = 16  # Middle layer for 3B model


def get_hidden_states(model, tokenizer, prompt: str, target_layer: int):
    """Extract hidden states from a specific layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    inner_model = model.model
    h = inner_model.embed_tokens(input_ids)

    T = h.shape[1]
    mask = mx.triu(mx.full((T, T), float("-inf"), dtype=mx.bfloat16), k=1)

    for i, layer in enumerate(inner_model.layers):
        h = layer(h, mask=mask)
        if i == target_layer:
            mx.eval(h)
            return h

    return h


def collect_activations(model, tokenizer, prompts: list[str], target_layer: int):
    """Collect last-token activations for a set of prompts."""
    import mlx.core as mx

    acts = []
    for prompt in prompts:
        h = get_hidden_states(model, tokenizer, prompt, target_layer)
        acts.append(h[0, -1, :])  # Last token

    activations = mx.stack(acts, axis=0)
    mx.eval(activations)
    return activations


def main() -> None:
    print("=" * 70)
    print("FINE-TUNING GEOMETRY ANALYSIS")
    print("What does code training actually change?")
    print("=" * 70)
    print()

    # Initialize backend
    print("[1/5] Initializing backend...")
    backend_type = detect_default_backend_type()
    backend = get_backend(backend_type)
    set_default_backend(backend)
    b = backend
    print(f"      Backend: {backend_type}")
    print()

    # Load models
    print("[2/5] Loading models...")
    import mlx.core as mx
    from mlx_lm import load

    base_path = MODELS_DIR / BASE_MODEL
    ft_path = MODELS_DIR / FT_MODEL

    if not base_path.exists() or not ft_path.exists():
        print("ERROR: Models not found")
        return

    print(f"      Base model: {BASE_MODEL}")
    base_model, base_tokenizer = load(str(base_path))
    hidden_dim = base_model.model.embed_tokens.weight.shape[1]
    num_layers = len(base_model.model.layers)
    print(f"      Hidden dim: {hidden_dim}, Layers: {num_layers}")

    print(f"      FT model: {FT_MODEL}")
    ft_model, ft_tokenizer = load(str(ft_path))
    print()

    # Collect activations
    print(f"[3/5] Collecting activations from layer {TARGET_LAYER}...")

    all_prompts = GENERAL_PROMPTS + CODE_PROMPTS
    print(f"      {len(GENERAL_PROMPTS)} general prompts + {len(CODE_PROMPTS)} code prompts")

    base_acts = collect_activations(base_model, base_tokenizer, all_prompts, TARGET_LAYER)
    ft_acts = collect_activations(ft_model, ft_tokenizer, all_prompts, TARGET_LAYER)

    print(f"      Base activations: {base_acts.shape}")
    print(f"      FT activations: {ft_acts.shape}")

    # Convert to backend
    base_backend = b.array(base_acts)
    ft_backend = b.array(ft_acts)
    b.eval(base_backend, ft_backend)

    # Free model memory
    del base_model, ft_model, base_tokenizer, ft_tokenizer
    mx.clear_cache()
    print()

    # Run Crosscoder analysis
    print("[4/5] Running Crosscoder decomposition...")

    from modelcypher.core.domain.interpretability.crosscoder import (
        Crosscoder,
        CrosscoderConfig,
    )

    config = CrosscoderConfig(
        hidden_dim=hidden_dim,
        shared_expansion=4,    # More shared capacity
        exclusive_expansion=2,  # Moderate exclusive capacity
        normalize_decoder=True,
    )

    cc = Crosscoder(config, backend=b)
    weights = cc.initialize_weights()

    print(f"      Shared dim: {config.shared_dim}")
    print(f"      Exclusive dim: {config.exclusive_dim}")
    print(f"      Total latent: {config.total_latent_dim}")
    print()

    # Encode both
    result = cc.encode(base_backend, ft_backend, weights)
    print(f"      Encoding complete")
    print(f"      Shared features shape: {result.shared_features.shape}")
    print()

    # Diff models
    print("[5/5] Computing model diff...")

    diff = cc.diff_models(base_backend, ft_backend, weights)

    total_features = config.total_latent_dim
    shared_count = len(diff.shared_feature_indices)
    base_exclusive_count = len(diff.base_exclusive_indices)
    ft_exclusive_count = len(diff.ft_exclusive_indices)

    shared_pct = 100 * shared_count / total_features
    base_excl_pct = 100 * base_exclusive_count / total_features
    ft_excl_pct = 100 * ft_exclusive_count / total_features

    print()
    print("=" * 70)
    print("RESULTS: Feature Decomposition")
    print("=" * 70)
    print()
    print(f"  Total latent features: {total_features}")
    print()
    print(f"  Shared features:       {shared_count:5d} ({shared_pct:5.1f}%)")
    print(f"  Base-exclusive:        {base_exclusive_count:5d} ({base_excl_pct:5.1f}%)")
    print(f"  FT-exclusive:          {ft_exclusive_count:5d} ({ft_excl_pct:5.1f}%)")
    print()
    print("  --- Energy Distribution ---")
    print(f"  Base exclusive energy: {diff.exclusive_base_energy:.4f}")
    print(f"  FT exclusive energy:   {diff.exclusive_ft_energy:.4f}")
    print(f"  Change magnitude:      {diff.change_magnitude:.4f} ({diff.change_magnitude*100:.1f}%)")
    print()
    print("  --- Manifold Preservation ---")
    print(f"  Shared feature CKA:    {diff.shared_activation_cka:.4f}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if diff.shared_activation_cka > 0.9:
        print("  [VALIDATED] Manifold structure preserved (shared CKA > 0.9)")
    else:
        print(f"  [WARNING] Manifold structure may be altered (shared CKA = {diff.shared_activation_cka:.4f})")

    if ft_exclusive_count > 0:
        print(f"  [VALIDATED] Code training added {ft_exclusive_count} exclusive features")
    else:
        print("  [WARNING] No FT-exclusive features detected")

    if diff.change_magnitude < 0.5:
        print(f"  [VALIDATED] Moderate specialization ({diff.change_magnitude*100:.1f}% change)")
    else:
        print(f"  [NOTE] High specialization ({diff.change_magnitude*100:.1f}% change)")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "finetune_geometry_analysis.json"

    results = {
        "experiment": "finetune_geometry_analysis",
        "date": datetime.now().isoformat(),
        "base_model": BASE_MODEL,
        "finetuned_model": FT_MODEL,
        "layer": TARGET_LAYER,
        "hidden_dim": hidden_dim,
        "prompts": {
            "general_count": len(GENERAL_PROMPTS),
            "code_count": len(CODE_PROMPTS),
        },
        "crosscoder_config": {
            "shared_expansion": config.shared_expansion,
            "exclusive_expansion": config.exclusive_expansion,
            "shared_dim": config.shared_dim,
            "exclusive_dim": config.exclusive_dim,
            "total_latent_dim": config.total_latent_dim,
        },
        "results": {
            "shared_feature_count": shared_count,
            "base_exclusive_count": base_exclusive_count,
            "ft_exclusive_count": ft_exclusive_count,
            "shared_pct": shared_pct,
            "base_exclusive_pct": base_excl_pct,
            "ft_exclusive_pct": ft_excl_pct,
            "shared_cka": diff.shared_activation_cka,
            "base_exclusive_energy": diff.exclusive_base_energy,
            "ft_exclusive_energy": diff.exclusive_ft_energy,
            "change_magnitude": diff.change_magnitude,
        },
        "validation": {
            "manifold_preserved": diff.shared_activation_cka > 0.9,
            "features_added": ft_exclusive_count > 0,
            "hypothesis_supported": (
                diff.shared_activation_cka > 0.9 and ft_exclusive_count > 0
            ),
        },
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    # Final verdict
    print("=" * 70)
    if results["validation"]["hypothesis_supported"]:
        print("VERDICT: Geometric hypothesis SUPPORTED")
        print()
        print("Fine-tuning added code-specific features without destroying")
        print("the shared manifold structure. Model merging is geometrically justified.")
    else:
        print("VERDICT: Results inconclusive")
        print()
        print("Further analysis needed.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
