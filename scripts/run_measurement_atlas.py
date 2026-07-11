#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelcypher.adapters.live_generation_trace import build_live_generation_trace_runner
from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.cli.composition import get_activation_provider, get_backend
from modelcypher.core.use_cases.generation_trace_service import GenerationTraceService
from modelcypher.core.use_cases.measurement_atlas_report_service import (
    MeasurementAtlasExecution,
    MeasurementAtlasReportService,
)
from modelcypher.core.use_cases.observation_service import (
    PromptFamilyManifest,
)

RUN_MANIFEST_VERSION = "mc.measurement_atlas.run_manifest.v2"
DEFAULT_OUTPUT_ROOT = Path("results/measurement_atlas")
LINKED_BLOCKER = "A1"
DEFAULT_MAX_TOKENS = 128
ATLAS_REQUESTED_REPLAY_SPACES = ("hidden", "embedding")
ATLAS_REQUESTED_LIVE_SPACES = ("hidden",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a research-only measurement atlas over explicit prompt families.",
    )
    parser.add_argument("--model", required=True, help="Path to the base model directory.")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Prompt family manifest to execute. Pass multiple times for multiple studies.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for measurement atlas runs.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum greedy decode tokens per variant.",
    )
    return parser


def run_measurement_atlas(
    *,
    model_path: str,
    manifest_paths: list[str],
    output_root: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    backend: Any | None = None,
    activation_provider: Any | None = None,
    model_loader: Any | None = None,
    trace_service: GenerationTraceService | None = None,
    report_service: MeasurementAtlasReportService | None = None,
    timestamp_utc: str | None = None,
    commit: str | None = None,
) -> Path:
    resolved_backend = backend or get_backend()
    resolved_activation_provider = activation_provider or get_activation_provider()
    resolved_model_loader = model_loader or ModelLoader(resolved_backend)
    resolved_trace_service = trace_service or GenerationTraceService(
        backend=resolved_backend,
        model_loader=resolved_model_loader,
        activation_provider=resolved_activation_provider,
        live_trace_runner=build_live_generation_trace_runner(resolved_backend),
    )
    resolved_report_service = report_service or MeasurementAtlasReportService(
        backend=resolved_backend,
    )

    manifests = [PromptFamilyManifest.from_json_path(path) for path in manifest_paths]
    study_space_set = ATLAS_REQUESTED_REPLAY_SPACES

    timestamp = timestamp_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = _run_id_from_timestamp(timestamp)
    output_dir = Path(output_root).expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_commit = commit or _resolve_git_commit()
    model, tokenizer = resolved_model_loader.load_model(model_path)

    executions: list[MeasurementAtlasExecution] = []
    for manifest in manifests:
        for variant in manifest.variants:
            trace = resolved_trace_service.trace_variant(
                model=model,
                tokenizer=tokenizer,
                prompt=variant.text,
                spaces=study_space_set,
                max_tokens=max_tokens,
            )
            executions.append(
                MeasurementAtlasExecution(
                    study_id=manifest.name,
                    case_id=variant.case_id,
                    variant_id=variant.variant_id,
                    prompt_text=variant.text,
                    comparison_to=variant.comparison_to,
                    tags=variant.tags,
                    annotations=variant.annotations,
                    trace=trace,
                    tokenizer=tokenizer,
                )
            )

    command = _format_command(model_path, manifest_paths, output_root, max_tokens)
    observed_replay_spaces = _observed_spaces(executions, mode="replay")
    observed_live_spaces = _observed_spaces(executions, mode="live")
    run_manifest = _build_run_manifest(
        run_id=run_id,
        timestamp_utc=timestamp,
        model_path=model_path,
        output_dir=output_dir,
        manifest_paths=manifest_paths,
        manifests=manifests,
        command=command,
        max_tokens=max_tokens,
        requested_replay_spaces=ATLAS_REQUESTED_REPLAY_SPACES,
        observed_replay_spaces=observed_replay_spaces,
        requested_live_spaces=ATLAS_REQUESTED_LIVE_SPACES,
        observed_live_spaces=observed_live_spaces,
    )
    build_result = resolved_report_service.build(
        run_id=run_id,
        timestamp_utc=timestamp,
        commit=resolved_commit,
        linked_blocker=LINKED_BLOCKER,
        claim=(
            "Prompt perturbations and grounded hallucination conditions move model "
            "trajectories in measurable ways across prompt, response, and generated "
            "representation spaces."
        ),
        mutable_surface="scripts/run_measurement_atlas.py + measurement_atlas_v2_artifact_schema",
        frozen_surfaces=(
            f"model={Path(model_path).expanduser().resolve()};"
            f"manifests={','.join(str(Path(path).expanduser().resolve()) for path in manifest_paths)};"
            "decode=greedy;requested_live_spaces="
            + ",".join(ATLAS_REQUESTED_LIVE_SPACES)
            + ";observed_live_spaces="
            + ",".join(observed_live_spaces)
            + ";requested_replay_spaces="
            + ",".join(ATLAS_REQUESTED_REPLAY_SPACES)
            + ";observed_replay_spaces="
            + ",".join(observed_replay_spaces)
            + f";max_tokens={max_tokens}"
        ),
        command=command,
        primary_observable=(
            "Per-study region and space deltas in meanGeodesicDeviation, "
            "plus earliest generated-token divergence and grounded label onset."
        ),
        artifact_dir=output_dir,
        next_falsifier=(
            "Run the same frozen study pack on the smallest local model that is free, "
            "then check whether live generated-token divergence and replay response "
            "divergence agree on the earliest shift."
        ),
        executions=executions,
    )

    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "summary.json", build_result.summary)
    (output_dir / "REPORT.md").write_text(build_result.report_markdown, encoding="utf-8")
    (output_dir / "ledger.tsv").write_text(
        build_result.ledger_header + "\n" + build_result.ledger_row + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "variants.jsonl", build_result.variant_rows)
    _write_jsonl(output_dir / "sequence_metrics.jsonl", build_result.sequence_metrics)
    _write_jsonl(output_dir / "step_metrics.jsonl", build_result.step_metrics)
    _write_jsonl(output_dir / "space_step_metrics.jsonl", build_result.space_step_metrics)
    _write_jsonl(output_dir / "comparisons.jsonl", build_result.comparisons)
    _write_jsonl(output_dir / "onset_events.jsonl", build_result.onset_events)
    return output_dir


def _build_run_manifest(
    *,
    run_id: str,
    timestamp_utc: str,
    model_path: str,
    output_dir: Path,
    manifest_paths: list[str],
    manifests: list[PromptFamilyManifest],
    command: str,
    max_tokens: int,
    requested_replay_spaces: tuple[str, ...],
    observed_replay_spaces: list[str],
    requested_live_spaces: tuple[str, ...],
    observed_live_spaces: list[str],
) -> dict[str, Any]:
    return {
        "schema": RUN_MANIFEST_VERSION,
        "runId": run_id,
        "requestedAt": timestamp_utc,
        "linkedBlocker": LINKED_BLOCKER,
        "claimContract": (
            "observable = f(geometry_state, architecture_state, scale_state, "
            "precision_state, measurement_operator)"
        ),
        "primaryObservable": (
            "Prompt-region, response-region, and generated-region geometry shifts "
            "measured by trajectory deviation, curvature, and grounded onset events."
        ),
        "explicitFalsifier": (
            "If the frozen study pack does not produce stable divergence or onset "
            "signals across replay and live traces, the atlas schema is not yet "
            "promotable to a CLI workflow."
        ),
        "mutableSurface": "scripts/run_measurement_atlas.py",
        "frozenSurfaces": {
            "model": str(Path(model_path).expanduser().resolve()),
            "decodePolicy": "greedy",
            "maxTokens": max_tokens,
            "requestedReplaySpaces": list(requested_replay_spaces),
            "observedReplaySpaces": observed_replay_spaces,
            "requestedLiveSpaces": list(requested_live_spaces),
            "observedLiveSpaces": observed_live_spaces,
            "artifactDir": str(output_dir),
        },
        "baselineCommand": command,
        "comparisonBudget": {
            "variants": sum(len(manifest.variants) for manifest in manifests),
            "studies": len(manifests),
        },
        "artifactDirectory": str(output_dir),
        "ledgerPath": str(output_dir / "ledger.tsv"),
        "model": str(Path(model_path).expanduser().resolve()),
        "manifests": [
            {
                "path": str(Path(path).expanduser().resolve()),
                "name": manifest.name,
                "schema": manifest.schema_version,
                "variantCount": len(manifest.variants),
            }
            for path, manifest in zip(manifest_paths, manifests, strict=False)
        ],
        "command": command,
    }


def _format_command(
    model_path: str,
    manifest_paths: list[str],
    output_root: str,
    max_tokens: int,
) -> str:
    manifest_flags = " ".join(
        f"--manifest {Path(path).expanduser().resolve()}"
        for path in manifest_paths
    )
    return (
        "poetry run python scripts/run_measurement_atlas.py "
        f"--model {Path(model_path).expanduser().resolve()} "
        f"{manifest_flags} "
        f"--output-root {Path(output_root).expanduser().resolve()} "
        f"--max-tokens {max_tokens}"
    ).strip()


def _observed_spaces(
    executions: list[MeasurementAtlasExecution],
    *,
    mode: str,
) -> list[str]:
    observed: set[str] = set()
    decode_key = f"{mode}Spaces"
    for execution in executions:
        decode_spaces = execution.trace.decode.get(decode_key, [])
        if isinstance(decode_spaces, list):
            observed.update(
                str(space).strip()
                for space in decode_spaces
                if str(space).strip()
            )
        for row in execution.trace.sequence_metrics:
            if str(row.get("mode", "")).strip() != mode:
                continue
            space = str(row.get("space", "")).strip()
            if space:
                observed.add(space)
    ordered = [space for space in ATLAS_REQUESTED_REPLAY_SPACES if space in observed]
    if mode == "live":
        ordered = [space for space in ATLAS_REQUESTED_LIVE_SPACES if space in observed]
    return ordered


def _run_id_from_timestamp(timestamp_utc: str) -> str:
    clean = timestamp_utc.replace(":", "").replace("-", "").replace("Z", "Z")
    return f"{clean}-measurement-atlas"


def _resolve_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = run_measurement_atlas(
        model_path=args.model,
        manifest_paths=list(args.manifest),
        output_root=args.output_root,
        max_tokens=args.max_tokens,
    )
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
