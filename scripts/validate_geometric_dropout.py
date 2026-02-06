#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Validates geometry-derived LoRA configuration on real model weights.
# Tests the full pipeline: load weights → compute geometry → derive configs
# (rank, scale, target modules, dropout) with zero user hyperparameters.
#
# Usage:
#   poetry run python scripts/validate_geometric_dropout.py
#   poetry run python scripts/validate_geometric_dropout.py --deep   # 8 layers + 8B models

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Initialize backend before any domain imports
from modelcypher.backends import initialize_default_backend

initialize_default_backend()

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.geometric_lora import (
    LayerGeometry,
    analyze_weight_geometries,
    compute_geometric_dropout,
    compute_layer_geometry,
    derive_lora_configs,
    select_target_modules,
)


MODELS_DIR = Path("/Volumes/CodeCypher/models/mlx-community")

# Core models (fast validation)
MODELS_CORE = [
    ("LFM2-350M-MLX-bf16", "LFM2 350M"),
    ("LFM2-700M-bf16", "LFM2 700M"),
    ("LFM2-1.2B-bf16", "LFM2 1.2B"),
    ("Qwen2.5-Coder-0.5B-Instruct-bf16", "Qwen2.5 0.5B"),
    ("Qwen2.5-3B-Instruct-bf16", "Qwen2.5 3B"),
    ("Llama-3.2-3B-Instruct-bf16", "Llama 3.2 3B"),
    ("SmolLM3-3B-bf16", "SmolLM3 3B"),
]

# Additional models for deep validation
MODELS_DEEP = [
    ("Qwen3-8B-bf16", "Qwen3 8B"),
    ("DeepSeek-R1-0528-Qwen3-8B-bf16", "DS-R1 8B"),
]


def load_model_weights(
    model_dir: Path,
    backend,
    layer_filter: str = "self_attn",
    max_layers: int = 4,
) -> dict[str, any]:
    """Load weight matrices from a model directory.

    Loads only attention/MLP projection weights to keep memory manageable.
    Handles both single-file and sharded safetensors.
    """
    weights = {}

    # Check for index file (sharded model)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Find relevant weight keys
        relevant_keys = {}
        for key, shard in weight_map.items():
            if "weight" not in key:
                continue
            # Filter to projection weights in first N layers
            parts = key.split(".")
            layer_idx = None
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break
            if layer_idx is not None and layer_idx >= max_layers:
                continue
            if any(t in key for t in ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "up_proj", "down_proj", "gate_proj"]):
                if shard not in relevant_keys:
                    relevant_keys[shard] = []
                relevant_keys[shard].append(key)

        # Load needed shards
        for shard, keys in relevant_keys.items():
            shard_path = model_dir / shard
            if not shard_path.exists():
                continue
            shard_weights = backend.load_safetensors(str(shard_path))
            for k in keys:
                if k in shard_weights:
                    w = backend.astype(shard_weights[k], "float32")
                    backend.eval(w)
                    weights[k] = w
    else:
        # Single safetensors file
        sf_files = sorted(model_dir.glob("*.safetensors"))
        for sf in sf_files[:3]:  # Limit to first few files
            file_weights = backend.load_safetensors(str(sf))
            for k, v in file_weights.items():
                if "weight" not in k:
                    continue
                parts = k.split(".")
                layer_idx = None
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                if layer_idx is not None and layer_idx >= max_layers:
                    continue
                if any(t in k for t in ["q_proj", "k_proj", "v_proj", "o_proj",
                                         "up_proj", "down_proj", "gate_proj"]):
                    w = backend.astype(v, "float32")
                    backend.eval(w)
                    weights[k] = w

    return weights


def analyze_model(model_name: str, label: str, backend, max_layers: int = 4) -> dict:
    """Run full geometry analysis on a model and return results."""
    model_dir = MODELS_DIR / model_name

    if not model_dir.exists():
        return {"error": f"Model not found: {model_dir}"}

    print(f"\n{'='*70}")
    print(f"  {label} ({model_name})")
    print(f"{'='*70}")

    t0 = time.time()
    weights = load_model_weights(model_dir, backend, max_layers=max_layers)
    load_time = time.time() - t0

    if not weights:
        return {"error": "No weights loaded"}

    print(f"  Loaded {len(weights)} weight matrices in {load_time:.1f}s")

    # Compute geometries
    t0 = time.time()
    geometries = analyze_weight_geometries(weights, backend)
    geom_time = time.time() - t0
    print(f"  Computed {len(geometries)} geometries in {geom_time:.1f}s")

    # Select targets and derive configs
    targets = select_target_modules(geometries)
    print(f"  Targetable layers: {len(targets)} / {len(geometries)}")

    if not targets:
        print("  WARNING: No targetable layers found (all full-rank)")
        for key, geom in sorted(geometries.items()):
            print(f"    {key}:")
            print(f"      shape={geom.shape}, eff_rank={geom.effective_rank}/{geom.full_rank}")
            print(f"      shannon_eff_rank={geom.shannon_effective_rank:.2f}")
            print(f"      σ_max={geom.sigma_max:.4f}, σ_k={geom.sigma_k:.6f}")
        return {
            "model": label,
            "layers": len(geometries),
            "targetable": 0,
            "error": "No targetable layers",
        }

    configs = derive_lora_configs(geometries, targets)

    # Report per-layer results
    print(f"\n  {'Layer':<45} {'Shape':<14} {'Rank':>5} {'σ_k':>10} {'Shannon':>8} {'Util%':>6} {'Dropout':>8}")
    print(f"  {'-'*45} {'-'*14} {'-'*5} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")

    dropout_values = []
    rank_values = []
    scale_values = []
    for cfg in configs:
        geom = geometries[cfg.layer_key]
        util = geom.shannon_effective_rank / geom.full_rank if geom.full_rank > 0 else 0
        short_key = cfg.layer_key.replace("model.layers.", "L").replace(".self_attn.", ".attn.").replace(".mlp.", ".mlp.")
        print(
            f"  {short_key:<45} {str(geom.shape):<14} {cfg.rank:>5} "
            f"{cfg.sigma_k:>10.6f} {geom.shannon_effective_rank:>8.2f} "
            f"{util*100:>5.1f}% {cfg.dropout:>8.4f}"
        )
        dropout_values.append(cfg.dropout)
        rank_values.append(cfg.rank)
        scale_values.append(cfg.sigma_k)

    # Summary statistics
    if dropout_values:
        print(f"\n  Dropout summary:")
        print(f"    min={min(dropout_values):.4f}  max={max(dropout_values):.4f}  "
              f"mean={sum(dropout_values)/len(dropout_values):.4f}  "
              f"median={sorted(dropout_values)[len(dropout_values)//2]:.4f}")

    # Consistency checks
    checks_passed = 0
    checks_total = 0

    # Check 1: dropout in valid range [0, 1)
    checks_total += 1
    if all(0.0 <= d < 1.0 for d in dropout_values):
        checks_passed += 1
    else:
        print("  FAIL: dropout outside [0, 1)")

    # Check 2: dropout monotonic with redundancy × adapter_fraction
    checks_total += 1
    monotonic = True
    for cfg in configs:
        geom = geometries[cfg.layer_key]
        util = geom.shannon_effective_rank / geom.full_rank if geom.full_rank > 0 else 0
        expected = (1 - util) * (cfg.rank / geom.full_rank) if geom.full_rank > 0 else 0
        if cfg.rank > 1 and abs(cfg.dropout - round(expected, 4)) > 0.0001:
            print(f"  FAIL: {cfg.layer_key} dropout={cfg.dropout} != expected {round(expected, 4)}")
            monotonic = False
    if monotonic:
        checks_passed += 1

    # Check 3: rank-1 layers have 0.0 dropout
    checks_total += 1
    rank1_ok = True
    for cfg in configs:
        if cfg.rank <= 1 and cfg.dropout != 0.0:
            print(f"  FAIL: rank-1 layer {cfg.layer_key} has dropout={cfg.dropout}")
            rank1_ok = False
    if rank1_ok:
        checks_passed += 1

    # Check 4: rank > 0 for all configs
    checks_total += 1
    if all(cfg.rank > 0 for cfg in configs):
        checks_passed += 1
    else:
        print("  FAIL: some configs have rank=0")

    # Check 5: sigma_k > 0 for all configs
    checks_total += 1
    if all(cfg.sigma_k > 0 for cfg in configs):
        checks_passed += 1
    else:
        print("  FAIL: some configs have sigma_k=0")

    print(f"\n  Consistency checks: {checks_passed}/{checks_total} passed")

    return {
        "model": label,
        "layers": len(geometries),
        "targetable": len(targets),
        "configs": len(configs),
        "dropout_min": min(dropout_values),
        "dropout_max": max(dropout_values),
        "dropout_mean": sum(dropout_values) / len(dropout_values),
        "rank_min": min(rank_values),
        "rank_max": max(rank_values),
        "scale_min": min(scale_values),
        "scale_max": max(scale_values),
        "checks_passed": checks_passed,
        "checks_total": checks_total,
    }


def main():
    deep = "--deep" in sys.argv
    backend = get_default_backend()
    print("Backend:", type(backend).__name__)

    models = MODELS_CORE + (MODELS_DEEP if deep else [])
    max_layers = 8 if deep else 4
    print(f"Testing {len(models)} models, {max_layers} layers each")
    print(f"Models dir: {MODELS_DIR}")

    results = []
    total_checks_passed = 0
    total_checks = 0

    for model_name, label in models:
        try:
            result = analyze_model(model_name, label, backend, max_layers=max_layers)
            results.append(result)
            total_checks_passed += result.get("checks_passed", 0)
            total_checks += result.get("checks_total", 0)
        except Exception as e:
            print(f"\n  ERROR analyzing {label}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"model": label, "error": str(e)})

    # Final summary table
    print(f"\n\n{'='*90}")
    print("  VALIDATION SUMMARY: Geometry-Derived LoRA Configuration")
    print(f"{'='*90}")
    print(f"  {'Model':<20} {'Layers':>7} {'Target':>7} {'Dropout Range':>15} {'Rank Range':>12} {'Scale Range':>20} {'Checks':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*15} {'-'*12} {'-'*20} {'-'*7}")

    for r in results:
        if "error" in r and "dropout_min" not in r:
            print(f"  {r['model']:<20} {'ERROR':>7}")
            continue
        dropout_range = f"{r.get('dropout_min', 0):.4f}-{r.get('dropout_max', 0):.4f}"
        rank_range = f"{r.get('rank_min', 0)}-{r.get('rank_max', 0)}"
        scale_range = f"{r.get('scale_min', 0):.6f}-{r.get('scale_max', 0):.6f}"
        checks = f"{r.get('checks_passed', 0)}/{r.get('checks_total', 0)}"
        print(
            f"  {r['model']:<20} {r.get('layers', 0):>7} {r.get('targetable', 0):>7} "
            f"{dropout_range:>15} {rank_range:>12} {scale_range:>20} {checks:>7}"
        )

    print(f"\n  Total consistency checks: {total_checks_passed}/{total_checks}")
    print(f"\n  Formula: dropout = redundancy x adapter_fraction")
    print(f"           redundancy = 1 - shannon_eff_rank / full_rank")
    print(f"           adapter_fraction = rank / full_rank")
    print(f"  No arbitrary constants. Both factors are spectral ratios.")

    if total_checks_passed == total_checks:
        print(f"\n  ALL CHECKS PASSED.")
    else:
        print(f"\n  {total_checks - total_checks_passed} CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
