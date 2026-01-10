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

"""
Safety Audit Example

This example demonstrates how to run safety probes and entropy diagnostics
against adapter metadata and entropy baselines.

Usage:
    poetry run python examples/02_safety_audit.py --name "adapter-name"
    MC_ALLOW_STUB_EMBEDDINGS=1 poetry run python examples/02_safety_audit.py --name "adapter-name"
    poetry run python examples/02_safety_audit.py --name "adapter-name" --baseline /path/to/baseline.json \
        --observed "[0.1, 0.12, 0.09]"
    poetry run python examples/02_safety_audit.py --name "adapter-name" --samples /path/to/samples.json

You can also pass an adapter path as the first argument; its filename will be
used as the adapter name.

Notes:
    - This example requires a working GPU backend (MLX on macOS/Apple Silicon).
      If MLX fails to initialize inside VSCode/Claude Code, run from Terminal.app.
    - To run embedding-backed metadata probes without MLX, set `MC_ALLOW_STUB_EMBEDDINGS=1`.

Observed deltas format:
    [0.12, 0.08, 0.15]

Samples format:
    [
      [entropy, variance],
      [entropy, variance]
    ]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelcypher.backends import initialize_default_backend
from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService
from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService


def _load_json_payload(value: str, description: str) -> object:
    path = Path(value)
    raw = path.read_text(encoding="utf-8") if path.exists() else value
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{description} must be JSON or a path to JSON: {exc}"
        ) from exc


def _resolve_adapter_name(adapter_arg: str | None, name_override: str | None) -> str | None:
    if name_override:
        return name_override
    if not adapter_arg:
        return None
    adapter_path = Path(adapter_arg)
    return adapter_path.name if adapter_path.exists() else adapter_arg


def _parse_observed_deltas(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    payload = _load_json_payload(raw, "Observed deltas")
    if not isinstance(payload, list):
        raise ValueError("Observed deltas must be a JSON array of numbers")
    return [float(value) for value in payload]


def _parse_samples(raw: str | None) -> list[tuple[float, float]] | None:
    if raw is None:
        return None
    payload = _load_json_payload(raw, "Entropy samples")
    if not isinstance(payload, list):
        raise ValueError("Entropy samples must be a JSON array of [entropy, variance] pairs")
    samples: list[tuple[float, float]] = []
    for idx, entry in enumerate(payload):
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
        ):
            raise ValueError(f"Entropy sample {idx} must be [entropy, variance]")
        samples.append((float(entry[0]), float(entry[1])))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run safety probes and entropy diagnostics on adapter metadata",
    )
    parser.add_argument(
        "adapter",
        nargs="?",
        help="Adapter name or path (optional)",
    )
    parser.add_argument(
        "--name",
        help="Adapter name for metadata probes",
    )
    parser.add_argument(
        "--description",
        help="Adapter description",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Skill tag (repeatable)",
    )
    parser.add_argument(
        "--creator",
        help="Creator identifier",
    )
    parser.add_argument(
        "--base-model",
        dest="base_model",
        help="Base model ID",
    )
    parser.add_argument(
        "--target-module",
        action="append",
        dest="target_modules",
        help="Target module name (repeatable)",
    )
    parser.add_argument(
        "--training-dataset",
        action="append",
        dest="training_datasets",
        help="Training dataset identifier (repeatable)",
    )
    parser.add_argument(
        "--baseline",
        help="Path to baseline JSON (from mc entropy calibrate)",
    )
    parser.add_argument(
        "--observed",
        help="JSON array or path to JSON array of observed deltas",
    )
    parser.add_argument(
        "--samples",
        help="JSON array or path to JSON array of [entropy, variance] pairs",
    )
    args = parser.parse_args()

    backend_error: str | None = None
    exit_code = 0

    def _ensure_backend() -> bool:
        nonlocal backend_error
        if backend_error is not None:
            return False
        try:
            initialize_default_backend()
        except RuntimeError as exc:
            backend_error = str(exc)
            return False
        return True

    adapter_name = _resolve_adapter_name(args.adapter, args.name)

    try:
        observed_deltas = _parse_observed_deltas(args.observed)
        samples = _parse_samples(args.samples)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 1

    print("Safety Audit (Raw Metrics)")
    print("=" * 60)

    embedder = EmbeddingDefaults.make_default_embedder()

    # 1. Static metadata scan
    print("\n[1/4] Static Metadata Scan")
    print("-" * 40)
    if adapter_name is None:
        print("  Skipped: provide --name or adapter argument")
    else:
        if embedder is None:
            print(
                "  Skipped: embedder unavailable "
                "(set MC_ALLOW_STUB_EMBEDDINGS=1 for stub embeddings)"
            )
        else:
            safety_service = SafetyProbeService(embedder=embedder)
            indicators = safety_service.scan_adapter_metadata(
                name=adapter_name,
                description=args.description,
                skill_tags=args.tags,
                creator=args.creator,
                base_model_id=args.base_model,
                target_modules=args.target_modules,
                training_datasets=args.training_datasets,
            )
            payload = SafetyProbeService.threat_indicators_payload(indicators)
            print(f"  Adapter: {adapter_name}")
            print(f"  Indicator count: {payload['count']}")
            print(f"  Max mean distance: {payload['maxMeanDistance']:.4f}")
            if indicators:
                print("  Indicators:")
                for ind in indicators[:5]:
                    print(f"    - [{ind.mean_distance:.4f}] {ind.field}: {ind.text}")

    # 2. Behavioral probes
    print("\n[2/4] Behavioral Probes")
    print("-" * 40)
    if adapter_name is None:
        print("  Skipped: provide --name or adapter argument")
    else:
        if embedder is None:
            print(
                "  Skipped: embedder unavailable "
                "(set MC_ALLOW_STUB_EMBEDDINGS=1 for stub embeddings)"
            )
        else:
            safety_service = SafetyProbeService(embedder=embedder)
            result = safety_service.run_behavioral_probes(
                adapter_name=adapter_name,
                adapter_description=args.description,
                skill_tags=args.tags,
                creator=args.creator,
                base_model_id=args.base_model,
            )
            payload = SafetyProbeService.composite_result_payload(result)
            print(f"  Adapter: {adapter_name}")
            print(f"  Probes run: {payload['probeCount']}")
            print(f"  Any findings: {payload['anyFindings']}")
            if payload["aggregateFindingCounts"]:
                counts = ", ".join(
                    f"{key}: {value}" for key, value in payload["aggregateFindingCounts"].items()
                )
                print(f"  Aggregate finding counts: {counts}")
            if payload["allFindings"]:
                print("  Findings:")
                for finding in payload["allFindings"][:5]:
                    print(f"    - {finding}")

    # 3. Entropy baseline verification
    print("\n[3/4] Entropy Baseline Verification")
    print("-" * 40)
    if args.baseline and observed_deltas is not None:
        if not _ensure_backend():
            print(f"  Error: backend unavailable ({backend_error})")
            exit_code = 1
        else:
            entropy_service = EntropyProbeService()
            try:
                result = entropy_service.verify_baseline(
                    baseline_path=args.baseline,
                    observed_deltas=observed_deltas,
                    adapter_path=adapter_name or "unknown",
                )
            except ValueError as exc:
                print(f"  Baseline verification failed: {exc}")
            else:
                declared = result.declared_baseline
                observed = result.observed_baseline
                comparison = result.comparison
                print(
                    f"  Declared mean/std: {declared.delta_mean:.4f} / {declared.delta_std_dev:.4f}"
                )
                print(
                    f"  Observed mean/std: {observed.delta_mean:.4f} / {observed.delta_std_dev:.4f}"
                )
                print(f"  Mean Z-score: {comparison.mean_z_score:.4f}")
                print(f"  StdDev ratio: {comparison.std_dev_ratio:.4f}")
                print(f"  Max deviation: {comparison.max_deviation:.4f}")
                print(f"  Min deviation: {comparison.min_deviation:.4f}")
                print(f"  Declared range: {comparison.declared_range:.4f}")
                print(f"  Observed range: {comparison.observed_range:.4f}")
    else:
        print("  Skipped: provide --baseline and --observed")

    # 4. Entropy pattern analysis
    print("\n[4/4] Entropy Pattern Analysis")
    print("-" * 40)
    if samples is None:
        print("  Skipped: provide --samples")
    else:
        if not _ensure_backend():
            print(f"  Error: backend unavailable ({backend_error})")
            exit_code = 1
        else:
            entropy_service = EntropyProbeService()
            pattern = entropy_service.analyze_pattern(samples)
            distress = entropy_service.detect_distress(samples)
            print(f"  Sample count: {pattern.sample_count}")
            print(f"  Trend slope: {pattern.trend_slope:.6f}")
            print(f"  Volatility: {pattern.volatility:.6f}")
            print(
                f"  Entropy mean/std: {pattern.entropy_mean:.6f} / {pattern.entropy_std_dev:.6f}"
            )
            print(
                f"  Variance mean/std: {pattern.variance_mean:.6f} / {pattern.variance_std_dev:.6f}"
            )
            print(
                f"  Entropy-variance correlation: {pattern.entropy_variance_correlation:.6f}"
            )
            print(f"  Sustained high count: {pattern.sustained_high_count}")
            print(f"  Sustained significance: {pattern.sustained_significance:.6f}")
            print(f"  Peak entropy: {pattern.peak_entropy:.6f}")
            print(f"  Min entropy: {pattern.min_entropy:.6f}")
            if pattern.anomaly_indices:
                indices = ", ".join(str(idx) for idx in pattern.anomaly_indices)
                print(f"  Anomaly indices: {indices}")
            if distress is not None:
                print("  Distress metrics:")
                print(f"    Sustained high count: {distress.sustained_high_count}")
                print(f"    Sustained significance: {distress.sustained_significance:.6f}")
                print(f"    Entropy mean: {distress.entropy_mean:.6f}")
                print(f"    Variance mean: {distress.variance_mean:.6f}")
                print(
                    f"    Entropy-variance correlation: {distress.entropy_variance_correlation:.6f}"
                )

    print("\n" + "=" * 60)
    print("Audit complete.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
