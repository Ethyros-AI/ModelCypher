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

"""Emit an exploratory `R5 / Q8` merge portability falsifier scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelcypher.experimental.merge.falsifier_contract import (
    build_merge_portability_manifest,
    build_merge_portability_summary,
    emit_merge_portability_bundle,
    validate_merge_portability_bundle,
)


DEFAULT_RESULTS_ROOT = Path("results") / "merge_portability_falsifier"


def emit_scaffold(
    *,
    output_dir: str | Path,
    run_id: str,
) -> Path:
    manifest = build_merge_portability_manifest(
        run_id=run_id,
        output_dir=output_dir,
    )
    summary = build_merge_portability_summary(run_id=run_id)
    out_dir = emit_merge_portability_bundle(
        output_dir,
        manifest=manifest,
        summary=summary,
    )

    validation = validate_merge_portability_bundle(out_dir)
    if not validation["ok"]:
        raise RuntimeError(
            "Merge portability falsifier bundle failed validation: "
            + "; ".join(validation["errors"])
        )

    return out_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the research-only artifact scaffold for merge portability "
            "falsifier work (`R5 / Q8`)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "scaffold",
        help="Directory where REPORT.md, summary.json, manifest.json, and ledger.jsonl are written.",
    )
    parser.add_argument(
        "--run-id",
        default="merge_portability_scaffold",
        help="Stable run identifier stored in the scaffold artifacts.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    out_dir = emit_scaffold(output_dir=args.output_dir, run_id=args.run_id)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

