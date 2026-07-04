#!/usr/bin/env python3
"""Validate isometry metrics on real LoRA adapters across 3 base models.

Replaces test_isometry_real.py: uses Backend protocol exclusively (no raw mlx imports).

Computes IsometryMetrics, WeylMetrics, and SpectralSelectivity per adapter per layer,
then correlates each metric against the ground-truth geometric scale ratio.

Usage:
    poetry run python scripts/validate_isometry_real.py
    poetry run python scripts/validate_isometry_real.py --phase 350m
    poetry run python scripts/validate_isometry_real.py --phase 8b --output results_8b.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Adapter inventory ────────────────────────────────────────────────────────

ADAPTER_ROOT = Path("/Volumes/CodeCypher/archive/modelcypher-legacy/adapters")
MODEL_ROOT = Path("/Volumes/CodeCypher/models/mlx-community")

BASE_MODELS = {
    "350m": MODEL_ROOT / "LFM2-350M-MLX-bf16",
    "8b": MODEL_ROOT / "Qwen3-8B-bf16",
    "1.2b": MODEL_ROOT / "LFM2.5-1.2B-Instruct-bf16",
}

ADAPTER_INVENTORY: dict[str, list[str]] = {
    "350m": [
        "algebra_lora",
        "cot_preserve_geometry_lora",
        "fix_transform_lora",
        "fix_transform_lora_v2",
        "fix_word_problems_lora",
        "geometric_alignment_lora",
        "unified_math_lora",
    ],
    "8b": [
        "early_layer_expansion_lora",
        "qwen3_gsm8k_lora",
        "qwen3_math_lora",
        "qwen3_surgical_lora",
        "universal_reasoning_lora",
    ],
    "1.2b": [
        "self_improve_lora",
        "self_improve_lora_v2",
        "transfer_math_lora",
    ],
}


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class LayerMetrics:
    layer_key: str
    # IsometryMetrics
    spectral_preservation_ratio: float
    spectral_angle: float
    condition_ratio: float
    grassmann_distance: float
    relative_frobenius_deviation: float
    rank_original: int
    rank_modified: int
    # WeylMetrics
    weyl_utilization: float
    delta_spectral_norm: float
    max_singular_value_change: float
    delta_effective_rank: int
    projection_depth: int
    # SpectralSelectivity
    amplification_cv: float
    # Ground truth
    scale_ratio: float  # configured_scale / geometric_bound
    sigma_k: float  # smallest significant SV of base weight


@dataclass
class AdapterResult:
    adapter_name: str
    base_model: str
    configured_scale: float
    rank: int
    num_layers: int
    layer_metrics: list[LayerMetrics]


@dataclass
class PhaseResult:
    phase: str
    base_model_path: str
    num_adapters: int
    adapter_results: list[AdapterResult]
    correlations: dict[str, float]  # metric_name -> Pearson r vs scale_ratio
    elapsed_seconds: float


# ── Helpers ──────────────────────────────────────────────────────────────────

def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return cov / (sx * sy)


def discover_adapters(base_key: str) -> list[Path]:
    """Find adapter directories that exist on disk."""
    adapters = []
    known = ADAPTER_INVENTORY.get(base_key, [])
    for name in known:
        path = ADAPTER_ROOT / name
        if path.exists() and (path / "adapter_config.json").exists():
            adapters.append(path)
    # Also scan for unlisted adapters matching this base model
    if ADAPTER_ROOT.exists():
        for candidate in sorted(ADAPTER_ROOT.iterdir()):
            if not candidate.is_dir():
                continue
            config_path = candidate / "adapter_config.json"
            if not config_path.exists():
                continue
            if candidate.name in known:
                continue
            # Check if adapter's base model matches
            try:
                config = json.loads(config_path.read_text())
                base_model_name = config.get("base_model", "")
                model_path_str = str(BASE_MODELS.get(base_key, ""))
                if base_model_name and model_path_str and (
                    base_model_name in model_path_str or model_path_str in base_model_name
                ):
                    adapters.append(candidate)
            except (json.JSONDecodeError, OSError):
                continue
    return adapters


def load_sharded_weight(backend, model_path: Path, weight_key: str, weight_map: dict[str, str]):
    """Load a single weight from a sharded safetensors model via Backend."""
    shard_name = weight_map.get(weight_key)
    if not shard_name:
        return None
    shard_path = model_path / shard_name
    if not shard_path.exists():
        return None
    weights = backend.load_safetensors(str(shard_path))
    return weights.get(weight_key)


def load_weight_map(model_path: Path) -> dict[str, str]:
    """Load the model's weight_map from index file, or build from single file."""
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        return index.get("weight_map", {})
    # Single-file model
    single = model_path / "model.safetensors"
    if single.exists():
        return {"__single_file__": str(single)}
    return {}


# ── Core validation logic ────────────────────────────────────────────────────

def validate_adapter(
    backend,
    adapter_path: Path,
    model_path: Path,
    weight_map: dict[str, str],
) -> AdapterResult | None:
    """Validate isometry metrics for a single adapter."""
    from modelcypher.adapters.adapter_weights_loader import AutoAdapterWeightsLoader
    from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
    from modelcypher.experimental.lora_isometry import (
        compute_isometry_metrics,
        compute_spectral_selectivity,
        compute_weyl_utilization,
    )

    config_path = adapter_path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping %s: config error: %s", adapter_path.name, exc)
        return None

    # Extract LoRA parameters
    lora_params = config.get("lora_parameters", config)
    rank = lora_params.get("rank", lora_params.get("r"))
    if rank is None:
        logger.warning("Skipping %s: no rank in config", adapter_path.name)
        return None
    rank = int(rank)
    alpha = float(lora_params.get("alpha", lora_params.get("lora_alpha", rank)))
    configured_scale = alpha / rank

    # Load adapter weights
    loader = AutoAdapterWeightsLoader()
    adapter_file = None
    for candidate in ["adapters.safetensors", "lora_weights.safetensors", "adapter_model.safetensors"]:
        if (adapter_path / candidate).exists():
            adapter_file = adapter_path / candidate
            break
    if adapter_file is None:
        logger.warning("Skipping %s: no weights file found", adapter_path.name)
        return None

    try:
        adapter_weights = loader.load(adapter_file, backend)
        backend.eval(*adapter_weights.values())
    except Exception as exc:
        logger.warning("Skipping %s: load error: %s", adapter_path.name, exc)
        return None

    # Organize LoRA pairs
    lora_pairs: dict[str, dict] = {}
    for key, value in adapter_weights.items():
        if key.endswith(".lora_a"):
            base_key = key[:-7]
            lora_pairs.setdefault(base_key, {})["a"] = value
        elif key.endswith(".lora_b"):
            base_key = key[:-7]
            lora_pairs.setdefault(base_key, {})["b"] = value

    sqrt_eps = sqrt_scalar(backend.finfo().eps, backend)
    layer_metrics: list[LayerMetrics] = []

    for layer_key, pair in sorted(lora_pairs.items()):
        if "a" not in pair or "b" not in pair:
            continue

        lora_a = pair["a"]
        lora_b = pair["b"]

        # Compute delta_w = lora_a @ lora_b (convention varies by format)
        delta_w = backend.matmul(lora_a, lora_b)

        # Find corresponding base weight
        base_weight_key = layer_key + ".weight"
        base_w = load_sharded_weight(backend, model_path, base_weight_key, weight_map)
        if base_w is None:
            logger.debug("Base weight not found: %s", base_weight_key)
            continue

        # Handle shape mismatches (transposition)
        dw_shape = backend.shape(delta_w)
        bw_shape = backend.shape(base_w)
        if dw_shape != bw_shape:
            if len(dw_shape) == 2 and len(bw_shape) == 2 and dw_shape == (bw_shape[1], bw_shape[0]):
                delta_w = backend.transpose(delta_w)
            else:
                logger.debug(
                    "Shape mismatch for %s: delta=%s base=%s",
                    layer_key, dw_shape, bw_shape,
                )
                continue

        # Cast to float32 for SVD
        base_f32 = backend.astype(base_w, "float32")
        delta_f32 = backend.astype(delta_w, "float32")
        w_mod_f32 = base_f32 + delta_f32
        backend.eval(base_f32, delta_f32, w_mod_f32)

        # Compute geometric scale bound for ground truth
        _, S_base, _ = backend.svd(base_f32)
        backend.eval(S_base)
        sigma_max = float(backend.to_scalar(S_base[0]))
        threshold = sqrt_eps * sigma_max

        s_list = backend.tolist(S_base)
        sig_vals = [float(s) for s in s_list if float(s) > threshold]
        if not sig_vals:
            continue
        sigma_k = sig_vals[-1]

        # Compute all three metric suites (need weyl first for spectral norm)
        try:
            iso = compute_isometry_metrics(base_f32, w_mod_f32, backend)
            weyl = compute_weyl_utilization(base_f32, delta_f32, backend)
            sel = compute_spectral_selectivity(base_f32, delta_f32, backend=backend)

            # Use spectral norm (||ΔW||_2) for geometric bound, not Frobenius.
            # Production code (lora_safety_service) uses spectral norm.
            # weyl.delta_spectral_norm = S_delta[0] = ||ΔW||_2.
            delta_spectral = weyl.delta_spectral_norm
            if delta_spectral > 0:
                geometric_bound = sigma_k / delta_spectral
            else:
                geometric_bound = float("inf")

            scale_ratio = configured_scale / geometric_bound if geometric_bound > 0 else float("inf")

            layer_metrics.append(LayerMetrics(
                layer_key=layer_key,
                spectral_preservation_ratio=iso.spectral_preservation_ratio,
                spectral_angle=iso.spectral_angle,
                condition_ratio=iso.condition_ratio,
                grassmann_distance=iso.grassmann_distance,
                relative_frobenius_deviation=iso.relative_frobenius_deviation,
                rank_original=iso.rank_original,
                rank_modified=iso.rank_modified,
                weyl_utilization=weyl.weyl_utilization,
                delta_spectral_norm=weyl.delta_spectral_norm,
                max_singular_value_change=weyl.max_singular_value_change,
                delta_effective_rank=weyl.delta_effective_rank,
                projection_depth=weyl.projection_depth,
                amplification_cv=sel.amplification_cv,
                scale_ratio=scale_ratio,
                sigma_k=sigma_k,
            ))
        except Exception as exc:
            logger.warning("Metrics failed for %s/%s: %s", adapter_path.name, layer_key, exc)
            continue

    if not layer_metrics:
        logger.warning("No valid layers for %s", adapter_path.name)
        return None

    return AdapterResult(
        adapter_name=adapter_path.name,
        base_model=model_path.name,
        configured_scale=configured_scale,
        rank=rank,
        num_layers=len(layer_metrics),
        layer_metrics=layer_metrics,
    )


def run_phase(phase: str, backend) -> PhaseResult | None:
    """Run validation for a single phase (base model)."""
    model_path = BASE_MODELS.get(phase)
    if model_path is None or not model_path.exists():
        logger.error("Model not found for phase %s: %s", phase, model_path)
        return None

    logger.info("=" * 70)
    logger.info("Phase: %s (%s)", phase, model_path.name)
    logger.info("=" * 70)

    weight_map = load_weight_map(model_path)
    if not weight_map:
        logger.error("No weight map for %s", model_path)
        return None

    adapters = discover_adapters(phase)
    if not adapters:
        logger.warning("No adapters found for %s", phase)
        return None

    logger.info("Found %d adapters", len(adapters))

    t0 = time.time()
    adapter_results: list[AdapterResult] = []

    for adapter_path in adapters:
        logger.info("  Processing: %s", adapter_path.name)
        result = validate_adapter(backend, adapter_path, model_path, weight_map)
        if result is not None:
            adapter_results.append(result)
            logger.info(
                "    -> %d layers, scale=%.4f, rank=%d",
                result.num_layers, result.configured_scale, result.rank,
            )

    if not adapter_results:
        logger.warning("No successful adapter validations for %s", phase)
        return PhaseResult(
            phase=phase,
            base_model_path=str(model_path),
            num_adapters=0,
            adapter_results=[],
            correlations={},
            elapsed_seconds=time.time() - t0,
        )

    # Aggregate all layer metrics for correlation analysis
    all_scale_ratios: list[float] = []
    metric_vectors: dict[str, list[float]] = {
        "spectral_preservation_ratio": [],
        "spectral_angle": [],
        "condition_ratio": [],
        "grassmann_distance": [],
        "relative_frobenius_deviation": [],
        "weyl_utilization": [],
        "amplification_cv": [],
        "projection_depth": [],
    }

    for ar in adapter_results:
        for lm in ar.layer_metrics:
            if lm.scale_ratio == float("inf") or lm.scale_ratio != lm.scale_ratio:
                continue
            all_scale_ratios.append(lm.scale_ratio)
            metric_vectors["spectral_preservation_ratio"].append(lm.spectral_preservation_ratio)
            metric_vectors["spectral_angle"].append(lm.spectral_angle)
            metric_vectors["condition_ratio"].append(lm.condition_ratio)
            metric_vectors["grassmann_distance"].append(lm.grassmann_distance)
            metric_vectors["relative_frobenius_deviation"].append(lm.relative_frobenius_deviation)
            metric_vectors["weyl_utilization"].append(lm.weyl_utilization)
            metric_vectors["amplification_cv"].append(lm.amplification_cv)
            metric_vectors["projection_depth"].append(float(lm.projection_depth))

    # Compute correlations
    correlations = {}
    for name, values in metric_vectors.items():
        r = pearson_r(all_scale_ratios, values)
        correlations[name] = round(r, 4)

    elapsed = time.time() - t0

    # Print summary
    logger.info("")
    logger.info("Phase %s Summary:", phase)
    logger.info("  Adapters: %d / %d successful", len(adapter_results), len(adapters))
    logger.info("  Total layers: %d", len(all_scale_ratios))
    logger.info("  Elapsed: %.1fs", elapsed)
    logger.info("")
    logger.info("  Correlations with scale_ratio:")
    strong_count = 0
    for name, r in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        marker = " ***" if abs(r) > 0.7 else ""
        logger.info("    %-35s r = %+.4f%s", name, r, marker)
        if abs(r) > 0.7:
            strong_count += 1

    logger.info("")
    if strong_count >= 3:
        logger.info("  PASS: %d metrics with |r| > 0.7 (need >= 3)", strong_count)
    else:
        logger.info("  FAIL: Only %d metrics with |r| > 0.7 (need >= 3)", strong_count)

    return PhaseResult(
        phase=phase,
        base_model_path=str(model_path),
        num_adapters=len(adapter_results),
        adapter_results=adapter_results,
        correlations=correlations,
        elapsed_seconds=elapsed,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate isometry metrics on real LoRA adapters")
    parser.add_argument(
        "--phase", choices=["350m", "8b", "1.2b", "all"], default="all",
        help="Which base model phase to run (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Check volume mount
    if not Path("/Volumes/CodeCypher").exists():
        logger.error("Volume not mounted: /Volumes/CodeCypher")
        sys.exit(1)

    # Initialize backend via protocol
    from modelcypher.backends import initialize_default_backend
    backend = initialize_default_backend()
    logger.info("Backend: %s", type(backend).__name__)

    phases = ["350m", "8b", "1.2b"] if args.phase == "all" else [args.phase]
    all_results: list[PhaseResult] = []

    for phase in phases:
        result = run_phase(phase, backend)
        if result is not None:
            all_results.append(result)

    # Write output if requested
    if args.output and all_results:
        output_path = Path(args.output)
        output_data = []
        for pr in all_results:
            phase_dict = {
                "phase": pr.phase,
                "base_model_path": pr.base_model_path,
                "num_adapters": pr.num_adapters,
                "correlations": pr.correlations,
                "elapsed_seconds": pr.elapsed_seconds,
                "adapters": [],
            }
            for ar in pr.adapter_results:
                adapter_dict = {
                    "adapter_name": ar.adapter_name,
                    "configured_scale": ar.configured_scale,
                    "rank": ar.rank,
                    "num_layers": ar.num_layers,
                    "layer_metrics": [asdict(lm) for lm in ar.layer_metrics],
                }
                phase_dict["adapters"].append(adapter_dict)
            output_data.append(phase_dict)
        output_path.write_text(json.dumps(output_data, indent=2))
        logger.info("Results written to %s", output_path)

    # Overall summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 70)
    for pr in all_results:
        strong = sum(1 for r in pr.correlations.values() if abs(r) > 0.7)
        status = "PASS" if strong >= 3 else "FAIL"
        logger.info(
            "  %s: %d adapters, %d strong correlations -> %s",
            pr.phase, pr.num_adapters, strong, status,
        )


if __name__ == "__main__":
    main()
