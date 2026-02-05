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

"""Adapter analysis CLI commands.

Commands:
    mc adapter analyze  - Compute geometry metrics for a LoRA adapter
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer

from modelcypher.cli.output import write_error, write_output


app = typer.Typer(no_args_is_help=True)


@app.command()
def analyze(
    adapter_path: str = typer.Argument(
        ...,
        help="Path to adapter weights (safetensors file or directory containing adapters.safetensors)",
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model", "-b",
        help="Path to base model (auto-detected from adapter_config.json if not specified)",
    ),
    output_format: str = typer.Option(
        "text",
        "--output", "-o",
        help="Output format: text, json",
    ),
) -> None:
    """Compute geometry metrics for a LoRA adapter.

    Analyzes the spectral properties of adapter weight deltas to measure:
    - Amplification CV (spectral selectivity): How selectively the adapter
      targets specific directions. Higher = more selective.
    - Weyl utilization: What fraction of the theoretical singular value
      shift bound is used. Lower = more conservative.

    These metrics can distinguish trained adapters from random perturbations.
    Trained adapters typically show:
    - Higher amplification CV (~0.34) vs random (~0.26)
    - Much lower Weyl utilization (~0.002-0.009) vs random (~0.054)

    Example:
        mc adapter analyze /path/to/adapter
        mc adapter analyze /path/to/adapter.safetensors -b /path/to/base_model
    """
    try:
        result = _run_analysis(adapter_path, base_model)

        if output_format == "json":
            write_output(result, output_format="json", pretty=True)
        else:
            _print_text_output(result)

    except Exception as e:
        write_error({"code": "ADAPTER_ANALYSIS_ERROR", "message": str(e)})
        raise typer.Exit(1)


def _run_analysis(adapter_path: str, base_model: Optional[str]) -> dict[str, Any]:
    """Run adapter geometry analysis."""
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.experimental.lora_geometry.measurements import collect_layer_measurements

    backend = get_default_backend()

    # Resolve adapter path
    adapter_file = Path(adapter_path)
    if adapter_file.is_dir():
        adapter_file = adapter_file / "adapters.safetensors"

    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_file}")

    adapter_dir = adapter_file.parent

    # Get base model from config if not specified
    if base_model is None:
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            base_model = config.get("model")

        if base_model is None:
            raise ValueError(
                "Base model not specified and not found in adapter_config.json. "
                "Use --base-model to specify."
            )

    base_model_path = Path(base_model)
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    # Load adapter weights with mlx for bfloat16 support
    import mlx.core as mx

    raw_weights = mx.load(str(adapter_file))
    adapter_weights = {k: v.astype(mx.float32) for k, v in raw_weights.items()}
    mx.eval(*adapter_weights.values())

    # Get adapter config
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        lora_params = config.get("lora_parameters", {})
        lora_rank = lora_params.get("rank", config.get("r", "unknown"))
        lora_scale = lora_params.get("scale", config.get("lora_alpha", "unknown"))
    else:
        lora_rank = "unknown"
        lora_scale = "unknown"

    # Load base model weights
    base_weights = _load_base_weights(base_model_path, adapter_weights, backend)

    # Analyze each layer
    measurements = []
    layer_details = []

    for adapter_key in sorted(adapter_weights.keys()):
        if "lora_a" not in adapter_key.lower():
            continue

        is_lower = "lora_a" in adapter_key
        lora_a_suffix = ".lora_a" if is_lower else ".lora_A"
        lora_b_suffix = ".lora_b" if is_lower else ".lora_B"
        lora_b_key = adapter_key.replace(lora_a_suffix, lora_b_suffix)

        if lora_b_key not in adapter_weights:
            continue

        base_key = adapter_key.replace(lora_a_suffix, ".weight")
        if base_key not in base_weights:
            continue

        A = adapter_weights[adapter_key]
        B = adapter_weights[lora_b_key]

        # MLX format: delta = (A @ B).T
        if A.shape[1] == B.shape[0]:
            delta_raw = backend.matmul(A, B)
            delta_w = backend.transpose(delta_raw)
        else:
            continue

        base_w = base_weights[base_key]
        mx.eval(delta_w, base_w)

        # Parse layer info
        parts = adapter_key.split(".")
        layer_idx = -1
        proj_name = "unknown"
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
            if part in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                proj_name = part

        try:
            m = collect_layer_measurements(
                weight_original=base_w,
                delta_w=delta_w,
                layer_idx=layer_idx,
                projection_name=proj_name,
                backend=backend,
            )
            measurements.append(m)
            layer_details.append({
                "layer": layer_idx,
                "projection": proj_name,
                "amplification_cv": float(m.amplification_cv),
                "weyl_utilization": float(m.weyl_utilization),
                "delta_frobenius_norm": float(m.delta_frobenius_norm),
                "delta_spectral_norm": float(m.delta_spectral_norm),
            })
        except Exception:
            pass

    if not measurements:
        raise ValueError("No measurements could be collected. Check adapter/base model compatibility.")

    # Aggregate
    cvs = [m.amplification_cv for m in measurements]
    weyls = [m.weyl_utilization for m in measurements]
    frobs = [m.delta_frobenius_norm for m in measurements]
    specs = [m.delta_spectral_norm for m in measurements]

    return {
        "adapter_name": adapter_dir.name,
        "base_model": base_model_path.name,
        "lora_rank": lora_rank,
        "lora_scale": lora_scale,
        "n_layers": len(measurements),
        "metrics": {
            "amplification_cv": {
                "mean": float(sum(cvs) / len(cvs)),
                "min": float(min(cvs)),
                "max": float(max(cvs)),
            },
            "weyl_utilization": {
                "mean": float(sum(weyls) / len(weyls)),
                "min": float(min(weyls)),
                "max": float(max(weyls)),
            },
            "delta_frobenius_norm": {
                "total": float(sum(frobs)),
                "mean": float(sum(frobs) / len(frobs)),
            },
            "delta_spectral_norm": {
                "mean": float(sum(specs) / len(specs)),
                "max": float(max(specs)),
            },
        },
        "interpretation": _interpret_metrics(sum(cvs) / len(cvs), sum(weyls) / len(weyls)),
        "layers": layer_details,
    }


def _load_base_weights(base_model_path: Path, adapter_weights: dict, backend) -> dict:
    """Load base model weights needed for adapter analysis."""
    import mlx.core as mx

    # Get target projection keys from adapter
    target_keys = set()
    for k in adapter_weights.keys():
        if "lora_a" in k.lower():
            is_lower = "lora_a" in k
            lora_a_suffix = ".lora_a" if is_lower else ".lora_A"
            base_key = k.replace(lora_a_suffix, ".weight")
            target_keys.add(base_key)

    # Try to load from index
    index_file = base_model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Find which shards we need
        shards_needed = {}
        for k in target_keys:
            if k in weight_map:
                shard = weight_map[k]
                if shard not in shards_needed:
                    shards_needed[shard] = []
                shards_needed[shard].append(k)

        # Load shards
        base_weights = {}
        for shard, keys in shards_needed.items():
            shard_path = base_model_path / shard
            if shard_path.exists():
                shard_weights = mx.load(str(shard_path))
                for k in keys:
                    if k in shard_weights:
                        base_weights[k] = shard_weights[k].astype(mx.float32)
                del shard_weights

        mx.eval(*base_weights.values()) if base_weights else None
        return base_weights

    # Single file fallback
    safetensors_file = base_model_path / "model.safetensors"
    if safetensors_file.exists():
        all_weights = mx.load(str(safetensors_file))
        base_weights = {}
        for k in target_keys:
            if k in all_weights:
                base_weights[k] = all_weights[k].astype(mx.float32)
        mx.eval(*base_weights.values()) if base_weights else None
        return base_weights

    # Try all safetensors files
    base_weights = {}
    for f in base_model_path.glob("*.safetensors"):
        shard = mx.load(str(f))
        for k in target_keys:
            if k in shard and k not in base_weights:
                base_weights[k] = shard[k].astype(mx.float32)
        del shard

    mx.eval(*base_weights.values()) if base_weights else None
    return base_weights


def _interpret_metrics(cv: float, weyl: float) -> str:
    """Interpret geometry metrics relative to baselines."""
    # Baselines from validation:
    # Random: CV ~0.26, Weyl ~0.054
    # Trained: CV ~0.34, Weyl ~0.002-0.009

    cv_vs_random = cv / 0.26
    weyl_vs_random = weyl / 0.054

    if weyl < 0.015 and cv > 0.30:
        return (
            f"Typical trained adapter profile. "
            f"Selectivity {cv_vs_random:.1f}x random, "
            f"Weyl utilization {1/weyl_vs_random:.0f}x lower than random."
        )
    elif weyl > 0.04:
        return (
            f"High Weyl utilization ({weyl:.3f}) similar to random perturbation. "
            f"This adapter may have unusual training dynamics."
        )
    elif cv < 0.25:
        return (
            f"Low selectivity ({cv:.3f}), below random baseline. "
            f"Adapter effects are spread uniformly across directions."
        )
    else:
        return (
            f"Intermediate profile. "
            f"CV={cv:.3f} ({cv_vs_random:.1f}x random), "
            f"Weyl={weyl:.4f} ({weyl_vs_random:.2f}x random)."
        )


def _print_text_output(result: dict) -> None:
    """Print human-readable output."""
    print(f"\n{'='*60}")
    print(f"Adapter: {result['adapter_name']}")
    print(f"Base Model: {result['base_model']}")
    print(f"Rank: {result['lora_rank']}, Scale: {result['lora_scale']}")
    print(f"{'='*60}")

    metrics = result["metrics"]

    print(f"\nGeometry Metrics ({result['n_layers']} layers):")
    print(f"  Amplification CV:    {metrics['amplification_cv']['mean']:.4f} "
          f"(range: {metrics['amplification_cv']['min']:.4f} - {metrics['amplification_cv']['max']:.4f})")
    print(f"  Weyl Utilization:    {metrics['weyl_utilization']['mean']:.4f} "
          f"(range: {metrics['weyl_utilization']['min']:.4f} - {metrics['weyl_utilization']['max']:.4f})")
    print(f"  Total Frobenius:     {metrics['delta_frobenius_norm']['total']:.4f}")
    print(f"  Max Spectral Norm:   {metrics['delta_spectral_norm']['max']:.4f}")

    print(f"\nInterpretation:")
    print(f"  {result['interpretation']}")

    print(f"\nReference Baselines:")
    print(f"  Random:  CV=0.26, Weyl=0.054")
    print(f"  Trained: CV=0.34, Weyl=0.002-0.009")
