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

"""Build a Qwen-anchored feasibility map for local memory-constrained runs."""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.cli.composition import get_system_service
from modelcypher.core.use_cases.feasibility_projection import (
    RuntimeOverhead,
    mean_runtime_overhead,
    project_memory_gib,
    static_weight_memory_gib,
)


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _guess_model_hint_billions(model_id: str) -> float | None:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*b", model_id.lower())
    if not match:
        return None
    return float(match.group(1))


def _magnitude_decade(value: float) -> int:
    if value <= 0.0:
        return 0
    return int(math.floor(math.log10(value)))


def _is_mislabeled_order_of_magnitude(model_id: str, param_count: int) -> bool:
    hinted_b = _guess_model_hint_billions(model_id)
    if hinted_b is None:
        return False
    measured_b = float(param_count) / 1e9
    return _magnitude_decade(hinted_b) != _magnitude_decade(measured_b)


def _discover_default_models() -> list[Path]:
    roots = [
        Path("/Volumes/CodeCypher/models"),
        Path("/Volumes/CodeCypher/models/llm"),
        Path.home() / "models",
    ]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            model_dir = config.parent
            name = model_dir.name.lower()
            if "qwen" in name or "gemma-3-27b" in name or "gemma3-27b" in name:
                found.append(model_dir)
    deduped = sorted({p.resolve() for p in found})
    return deduped


def _memory_stage(profile: dict[str, Any], stage: str) -> dict[str, Any] | None:
    for row in profile.get("memory_stages", []):
        if row.get("stage") == stage:
            return row
    return None


def _profile_model(
    system_service: Any,
    model_path: Path,
    prompt: str,
    decode_tokens: int,
    train_probe: bool,
) -> dict[str, Any]:
    payload = system_service.memory_profile(
        model=str(model_path),
        prompt=prompt,
        train_probe=train_probe,
        decode_tokens=decode_tokens,
    )
    payload["model_path"] = str(model_path)
    payload["profiled_at"] = _now_utc_iso()
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _system_memory_gib() -> float:
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / float(1024**3)
    except Exception:
        return 0.0


def _build_projection_table(
    profiles: list[dict[str, Any]],
    target_params: list[int],
    decode_tokens: int,
    system_memory_gib: float,
) -> dict[str, Any]:
    by_bits: dict[int, list[RuntimeOverhead]] = {}
    for profile in profiles:
        if profile.get("excluded"):
            continue
        load = _memory_stage(profile, "load")
        forward = _memory_stage(profile, "forward")
        if load is None or forward is None:
            continue

        param_count = int(profile.get("param_count") or 0)
        bits = int(profile.get("precision_bits") or 16)
        static_gib = static_weight_memory_gib(param_count, bits)
        load_active = float(load.get("active_gb", 0.0))
        forward_active = float(forward.get("active_gb", 0.0))
        decode_slope = float(profile.get("decode_slope", {}).get("gb_per_token", 0.0))
        sample = RuntimeOverhead(
            load_overhead_gib=load_active - static_gib,
            forward_delta_gib=forward_active - load_active,
            decode_slope_gib_per_token=decode_slope,
        )
        by_bits.setdefault(bits, []).append(sample)

    tier_overheads: dict[int, RuntimeOverhead] = {}
    for bits, samples in by_bits.items():
        tier_overheads[bits] = mean_runtime_overhead(samples)

    projections: dict[str, Any] = {
        "tiers": {},
        "targets": {},
    }
    for bits, overhead in sorted(tier_overheads.items()):
        projections["tiers"][str(bits)] = {
            "sample_count": len(by_bits[bits]),
            "load_overhead_gib": overhead.load_overhead_gib,
            "forward_delta_gib": overhead.forward_delta_gib,
            "decode_slope_gib_per_token": overhead.decode_slope_gib_per_token,
        }

    for target in target_params:
        rows: list[dict[str, Any]] = []
        for bits, overhead in sorted(tier_overheads.items()):
            projected = project_memory_gib(
                param_count=target,
                precision_bits=bits,
                overhead=overhead,
                decode_tokens=decode_tokens,
            )
            rows.append({
                "precision_bits": bits,
                "param_count": target,
                "fits_system_memory": projected["decode_active_gib"] <= system_memory_gib,
                **projected,
            })
        projections["targets"][str(target)] = rows
    return projections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model directory path (repeat for multiple models)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why deterministic geometric trajectories can still appear probabilistic at readout.",
        help="Prompt used for tokenization/forward/decode measurements",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=32,
        help="Decode cap used for bounded decode windows",
    )
    parser.add_argument(
        "--train-probe",
        action="store_true",
        help="Run optional train-probe surrogate stage",
    )
    parser.add_argument(
        "--target-params",
        action="append",
        type=int,
        default=[],
        help="Projection target parameter count (repeatable). Defaults to 70B and 120B.",
    )
    parser.add_argument(
        "--output-root",
        default="results/feasibility_map",
        help="Output root directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from modelcypher.backends import initialize_default_backend

    initialize_default_backend()
    system_service = get_system_service()
    run_id = _now_utc_compact()
    output_dir = Path(args.output_root).expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model_inputs = [Path(p).expanduser().resolve() for p in args.model]
    if not model_inputs:
        model_inputs = _discover_default_models()
    if not model_inputs:
        raise RuntimeError(
            "No models found. Pass --model paths explicitly or place Qwen/Gemma-27B models in local roots."
        )

    target_params = args.target_params or [70_000_000_000, 120_000_000_000]
    profiled: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for idx, model_path in enumerate(model_inputs, start=1):
        try:
            profile = _profile_model(
                system_service=system_service,
                model_path=model_path,
                prompt=args.prompt,
                decode_tokens=args.decode_tokens,
                train_probe=args.train_probe,
            )
            model_id = str(profile.get("model_id") or model_path.name)
            param_count = int(profile.get("param_count") or 0)
            profile["excluded"] = _is_mislabeled_order_of_magnitude(model_id, param_count)
            if profile["excluded"]:
                profile["excluded_reason"] = (
                    "model_id parameter hint and measured parameters differ by order of magnitude"
                )
            bits = int(profile.get("precision_bits") or 16)
            model_tag = f"{idx:02d}_{model_id}_{bits}bit".replace("/", "_")
            _write_json(output_dir / f"{model_tag}.json", profile)
            profiled.append(profile)
        except Exception as exc:
            errors.append({"model_path": str(model_path), "error": str(exc)})

    projection = _build_projection_table(
        profiles=profiled,
        target_params=target_params,
        decode_tokens=args.decode_tokens,
        system_memory_gib=_system_memory_gib(),
    )
    summary = {
        "generated_at": _now_utc_iso(),
        "run_id": run_id,
        "host": platform.node(),
        "system_memory_gib": _system_memory_gib(),
        "decode_tokens": args.decode_tokens,
        "models_profiled": len(profiled),
        "models_failed": len(errors),
        "profiles": profiled,
        "errors": errors,
        "projection": projection,
        "assumptions": [
            "Static weight memory uses params * bits / 8.",
            "Runtime overhead terms are empirical means by precision tier.",
            "Projections reuse measured load overhead, forward delta, and decode slope.",
            "Mislabeled checkpoints are excluded only when model-name hint and measured params differ by order of magnitude.",
        ],
    }
    _write_json(output_dir / "feasibility_map.json", summary)

    print(f"wrote feasibility map to: {output_dir}")
    for target, rows in projection["targets"].items():
        print(f"target={target}")
        for row in rows:
            print(
                f"  {row['precision_bits']:>2}b"
                f" load={row['load_active_gib']:.2f}GiB"
                f" forward={row['forward_active_gib']:.2f}GiB"
                f" decode={row['decode_active_gib']:.2f}GiB"
                f" fits={row['fits_system_memory']}"
            )


if __name__ == "__main__":
    main()
