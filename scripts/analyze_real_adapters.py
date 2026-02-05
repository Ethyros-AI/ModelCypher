#!/usr/bin/env python3
"""Analyze real LoRA adapters with geometry metrics.

This script measures spectral selectivity, Weyl utilization, and other
geometric properties of real trained adapters to understand what distinguishes
good adapters from random perturbations.
"""

import json
import sys
from pathlib import Path

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def log(msg):
    print(msg, flush=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.backends import initialize_default_backend

initialize_default_backend()

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.adapters.adapter_weights_loader import AutoAdapterWeightsLoader
from modelcypher.experimental.lora_geometry.measurements import (
    collect_layer_measurements,
    LayerMeasurement,
)
from modelcypher.experimental.lora_isometry import (
    compute_spectral_selectivity,
    compute_weyl_utilization,
    compute_isometry_metrics,
)


def load_base_weights(model_path: Path, backend) -> dict:
    """Load base model weights using mlx for bfloat16 support."""
    import mlx.core as mx

    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        # Single file model
        weight_map = None

    weights = {}
    loaded_files = set()

    # Load only the weights we need (attention projections)
    target_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if weight_map:
        for key, filename in weight_map.items():
            if any(t in key for t in target_keys) and "weight" in key:
                if filename not in loaded_files:
                    shard_path = model_path / filename
                    if shard_path.exists():
                        # Use mlx.core.load for native bfloat16 support
                        shard = mx.load(str(shard_path))
                        for k, v in shard.items():
                            if any(t in k for t in target_keys) and "weight" in k:
                                # Convert to float32 for computation
                                weights[k] = v.astype(mx.float32)
                        loaded_files.add(filename)
    else:
        # Single file
        safetensors_file = model_path / "model.safetensors"
        if safetensors_file.exists():
            shard = mx.load(str(safetensors_file))
            for k, v in shard.items():
                if any(t in k for t in target_keys) and "weight" in k:
                    weights[k] = v.astype(mx.float32)

    return weights


def analyze_adapter(adapter_path: Path, base_model_path: Path, backend, cached_base_weights=None):
    """Analyze a single adapter."""
    import mlx.core as mx

    log(f"\n{'='*60}")
    log(f"Adapter: {adapter_path.parent.name}")
    log(f"{'='*60}")

    # Load adapter weights directly with mlx for bfloat16 support
    try:
        raw_weights = mx.load(str(adapter_path))
        adapter_weights = {k: v.astype(mx.float32) for k, v in raw_weights.items()}
        mx.eval(*adapter_weights.values())
    except Exception as e:
        log(f"  ERROR loading adapter: {e}")
        return None

    # Get adapter config
    config_path = adapter_path.parent / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        lora_params = config.get("lora_parameters", {})
        lora_rank = lora_params.get("rank", config.get("r", "unknown"))
        lora_alpha = lora_params.get("scale", config.get("lora_alpha", config.get("alpha", "unknown")))
        log(f"  Rank: {lora_rank}, Alpha: {lora_alpha}")
        log(f"  Base: {base_model_path.name}")
    else:
        lora_rank = "unknown"
        lora_alpha = "unknown"

    # Use cached base weights or load them
    if cached_base_weights:
        base_weights = cached_base_weights
    else:
        log(f"  Loading base weights from {base_model_path.name}...")
        base_weights = load_base_weights(base_model_path, backend)

    if not base_weights:
        log(f"  ERROR: Could not load base weights")
        return None

    # Analyze each layer
    measurements = []

    for adapter_key in sorted(adapter_weights.keys()):
        # Map adapter key to base weight key
        # Adapter keys: "model.layers.0.self_attn.q_proj.lora_a" (lowercase, no .weight)
        # Base keys: "model.layers.0.self_attn.q_proj.weight"

        # Handle both formats: lora_a/lora_b (lowercase) or lora_A/lora_B (uppercase)
        if "lora_a" in adapter_key.lower():
            is_lowercase = "lora_a" in adapter_key
            lora_a_suffix = ".lora_a" if is_lowercase else ".lora_A"
            lora_b_suffix = ".lora_b" if is_lowercase else ".lora_B"
        else:
            continue

        # Find corresponding lora_B
        lora_b_key = adapter_key.replace(lora_a_suffix, lora_b_suffix)
        if lora_b_key not in adapter_weights:
            # Try with .weight suffix
            lora_b_key = adapter_key.replace(lora_a_suffix + ".weight", lora_b_suffix + ".weight")
            if lora_b_key not in adapter_weights:
                continue

        # Find base weight key
        if ".weight" in adapter_key:
            base_key = adapter_key.replace(lora_a_suffix + ".weight", ".weight")
        else:
            base_key = adapter_key.replace(lora_a_suffix, ".weight")

        if base_key not in base_weights:
            # Try without "model." prefix
            alt_base_key = base_key.replace("model.", "", 1)
            if alt_base_key in base_weights:
                base_key = alt_base_key
            else:
                continue

        # Compute delta weight
        # MLX LoRA format:
        #   lora_a: (in_features, rank)
        #   lora_b: (rank, out_features)
        #   delta = lora_a @ lora_b = (in, out)
        # Base weights are (out, in), so delta needs transpose
        A = adapter_weights[adapter_key]  # lora_a: (in, rank)
        B = adapter_weights[lora_b_key]   # lora_b: (rank, out)

        a_shape = A.shape
        b_shape = B.shape

        # Check if A @ B is compatible: A[1] == B[0]
        if len(a_shape) == 2 and len(b_shape) == 2 and a_shape[1] == b_shape[0]:
            # MLX format: delta = (A @ B).T to match base weight shape (out, in)
            delta_raw = backend.matmul(A, B)  # (in, out)
            delta_w = backend.transpose(delta_raw)  # (out, in)
        else:
            log(f"  WARNING: Incompatible shapes A={a_shape} B={b_shape} for {adapter_key}")
            continue

        base_w = base_weights[base_key]
        backend.eval(delta_w, base_w)

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

        # Compute metrics
        try:
            m = collect_layer_measurements(
                weight_original=base_w,
                delta_w=delta_w,
                layer_idx=layer_idx,
                projection_name=proj_name,
                backend=backend,
            )
            measurements.append(m)
        except Exception as e:
            log(f"  WARNING: Failed to measure {adapter_key}: {e}")

    if not measurements:
        log(f"  ERROR: No measurements collected")
        return None

    # Aggregate statistics
    cvs = [m.amplification_cv for m in measurements]
    weyls = [m.weyl_utilization for m in measurements]
    frob_norms = [m.delta_frobenius_norm for m in measurements]
    spectral_norms = [m.delta_spectral_norm for m in measurements]

    result = {
        "adapter_name": adapter_path.parent.name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "n_layers": len(measurements),
        "mean_amplification_cv": sum(cvs) / len(cvs),
        "std_amplification_cv": (sum((x - sum(cvs)/len(cvs))**2 for x in cvs) / len(cvs)) ** 0.5,
        "mean_weyl_utilization": sum(weyls) / len(weyls),
        "std_weyl_utilization": (sum((x - sum(weyls)/len(weyls))**2 for x in weyls) / len(weyls)) ** 0.5,
        "total_frobenius_norm": sum(frob_norms),
        "mean_spectral_norm": sum(spectral_norms) / len(spectral_norms),
    }

    log(f"\n  METRICS ({len(measurements)} layers):")
    log(f"    Amplification CV:  {result['mean_amplification_cv']:.4f} ± {result['std_amplification_cv']:.4f}")
    log(f"    Weyl Utilization:  {result['mean_weyl_utilization']:.4f} ± {result['std_weyl_utilization']:.4f}")
    log(f"    Total Frob Norm:   {result['total_frobenius_norm']:.2f}")
    log(f"    Mean Spectral Norm: {result['mean_spectral_norm']:.4f}")

    return result


def get_base_model_path(adapter_dir: Path) -> Path | None:
    """Get the base model path from adapter config."""
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_path = config.get("model")
        if model_path:
            return Path(model_path)
    return None


def main():
    backend = get_default_backend()

    # Find all adapters
    adapter_dir = Path("/Volumes/CodeCypher/archive/modelcypher-legacy/adapters")
    adapters = list(adapter_dir.glob("*/adapters.safetensors"))

    log(f"Found {len(adapters)} adapters")

    # Cache loaded base weights by model path
    base_weight_cache = {}

    results = []

    # Analyze adapters
    for adapter_path in adapters[:15]:  # Analyze 15 adapters
        try:
            # Get the correct base model for this adapter
            adapter_parent = adapter_path.parent
            base_model = get_base_model_path(adapter_parent)

            if base_model is None:
                log(f"\n  SKIP {adapter_parent.name}: No model path in config")
                continue

            if not base_model.exists():
                log(f"\n  SKIP {adapter_parent.name}: Base model not found: {base_model}")
                continue

            # Load base weights (cached)
            if str(base_model) not in base_weight_cache:
                log(f"\n  Loading base weights from {base_model.name}...")
                base_weight_cache[str(base_model)] = load_base_weights(base_model, backend)

            result = analyze_adapter(adapter_path, base_model, backend, base_weight_cache.get(str(base_model)))
            if result:
                results.append(result)
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        log(f"\n{'='*60}")
        log("SUMMARY ACROSS ALL ADAPTERS")
        log(f"{'='*60}")

        all_cvs = [r["mean_amplification_cv"] for r in results]
        all_weyls = [r["mean_weyl_utilization"] for r in results]
        all_norms = [r["total_frobenius_norm"] for r in results]

        log(f"\nAmplification CV:")
        log(f"  Mean: {sum(all_cvs)/len(all_cvs):.4f}")
        log(f"  Range: {min(all_cvs):.4f} - {max(all_cvs):.4f}")

        log(f"\nWeyl Utilization:")
        log(f"  Mean: {sum(all_weyls)/len(all_weyls):.4f}")
        log(f"  Range: {min(all_weyls):.4f} - {max(all_weyls):.4f}")

        log(f"\nTotal Frobenius Norm:")
        log(f"  Mean: {sum(all_norms)/len(all_norms):.2f}")
        log(f"  Range: {min(all_norms):.2f} - {max(all_norms):.2f}")

        # Compare to synthetic random baseline
        log(f"\n{'='*60}")
        log("COMPARISON TO SYNTHETIC RANDOM BASELINE")
        log(f"{'='*60}")
        log(f"\nSynthetic Random (from Exp 2):")
        log(f"  Amplification CV: 0.2599")
        log(f"  Weyl Utilization: 0.0541")
        log(f"\nReal Adapters:")
        log(f"  Amplification CV: {sum(all_cvs)/len(all_cvs):.4f}")
        log(f"  Weyl Utilization: {sum(all_weyls)/len(all_weyls):.4f}")

        if sum(all_cvs)/len(all_cvs) > 0.2599:
            log(f"\n✓ Real adapters show HIGHER selectivity than random")
            log(f"  This suggests trained adapters are more selective in which")
            log(f"  directions they amplify - a sign of meaningful structure.")
        else:
            log(f"\n✗ Real adapters show LOWER selectivity than random")
            log(f"  This is unexpected and warrants investigation.")

        # Save results
        output_path = Path("results/real_adapter_analysis.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
