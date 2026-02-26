"""Artifact writers for Baranov replication tracks.

EXPERIMENTAL: Not validated for production use.

Writes manifest JSON, metrics CSV, and markdown summary stubs.
All writers include a collision guard: they raise ``FileExistsError``
if the output path already exists unless ``overwrite=True`` is passed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from modelcypher.experimental.baranov.manifest import ReplicationManifest


def _check_collision(output_path: Path, overwrite: bool) -> None:
    """Raise ``FileExistsError`` if *output_path* exists and *overwrite* is False."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_path}. "
            "Pass overwrite=True to replace.",
        )


def write_manifest_json(
    manifest: ReplicationManifest,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a replication manifest to a JSON file.

    Returns the written path.
    """
    output_path = Path(output_path)
    _check_collision(output_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_metrics_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    fieldnames: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write metrics rows to a CSV file (one row per run/seed/split).

    If *fieldnames* is ``None``, derives column order from the union
    of all row keys (sorted).

    Returns the written path.
    """
    output_path = Path(output_path)
    _check_collision(output_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def write_summary_stub(
    manifest: ReplicationManifest,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a markdown summary stub with pre-filled metadata.

    Returns the written path.
    """
    output_path = Path(output_path)
    _check_collision(output_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = manifest.metrics_dict
    metrics_lines = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(metrics.items())
    )

    content = f"""\
# Replication Summary: Track {manifest.track}

- **Run ID**: {manifest.run_id}
- **Timestamp**: {manifest.timestamp_utc}
- **Model**: {manifest.model.id} ({manifest.model.quantization}, {manifest.model.backend})
- **ModelCypher Commit**: {manifest.code.modelcypher_commit}

## Pre-registered Decision

- **Criteria Version**: {manifest.pre_registered_decision.criteria_version}
- **Outcome**: {manifest.pre_registered_decision.outcome}
- **Reason**: {manifest.pre_registered_decision.reason}

## Metrics

| Metric | Value |
|--------|-------|
{metrics_lines}

## Controls

- Base control: {manifest.controls.base_control}
- LoRA-only control: {manifest.controls.lora_only_control}
- Edit-only control: {manifest.controls.edit_only_control}

## Notes

(Fill in after analysis.)
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


__all__ = [
    "write_manifest_json",
    "write_metrics_csv",
    "write_summary_stub",
]
