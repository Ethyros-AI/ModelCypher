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

"""Auto-generate D-6 evidence tables from falsification artifacts.

Reads trajectory_falsification_200_multiseed_summary.json and per-model
20-step results, then generates markdown tables between sentinel comments
in docs/research/INDUSTRY-DIVERGENCES.md.

Usage:
    python scripts/generate_d6_tables.py [--check]

    --check  Verify tables are up-to-date (exit 1 if stale). Does not write.

Research script — not a CLI command (per promotion policy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTISEED_SUMMARY = REPO_ROOT / "results" / "weight_geometry" / "trajectory_falsification_200_multiseed_summary.json"
DOC_PATH = REPO_ROOT / "docs" / "research" / "INDUSTRY-DIVERGENCES.md"

# 20-step per-model results (single seed=42)
TWENTY_STEP_RESULTS = {
    "LFM2-350M": REPO_ROOT / "results" / "weight_geometry" / "trajectory_falsification" / "LFM2-350M" / "results.json",
    "Qwen2.5-Coder-0.5B": REPO_ROOT / "results" / "weight_geometry" / "trajectory_falsification" / "Qwen2.5-Coder-0.5B" / "results.json",
}

BEGIN_SENTINEL = "<!-- BEGIN D6-TABLES -->"
END_SENTINEL = "<!-- END D6-TABLES -->"


def _fmt(val: float, sig: int = 4) -> str:
    """Format a float in scientific notation with backticks."""
    if abs(val) == 0:
        return "`0`"
    return f"`{val:.{sig}e}`"


def _fmt_fixed(val: float, decimals: int = 7) -> str:
    """Format a float in fixed notation with backticks."""
    return f"`{val:.{decimals}f}`"


def _load_20step_metrics(path: Path) -> dict | None:
    """Load per-model 20-step results and extract F1-F6 metrics."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    verdicts = data.get("verdicts", [])
    metrics = {}
    for v in verdicts:
        name = v.get("test_name", "")
        details = v.get("details", {})
        if "F1" in name:
            metrics["f1_median"] = details.get("median", 0)
        elif "F2" in name:
            metrics["f2_cohens_d"] = details.get("cohens_d", 0)
        elif "F3" in name:
            metrics["f3_median"] = details.get("median", 0)
        elif "F4" in name:
            metrics["f4_max_drift"] = details.get("max_final_drift", 0)
        elif "F5" in name:
            metrics["f5_cond_p10"] = details.get("condition_number_p10", 0)
        elif "F6" in name:
            metrics["f6_max_dev"] = details.get("max_deviation_over_norm_sq", 0)
    return metrics


def generate_tables() -> str:
    """Generate markdown tables from artifacts."""
    lines = []

    # ── 20-step table ──
    lines.append("")
    lines.append("Cross-family trajectory falsification (20-step protocol):")
    lines.append("")
    lines.append("| Model | F1 median `\\|\\|P_hat-I\\|\\|_F/sqrt(r)` | F3 median `cos(Pg,g)` | F4 max drift | F2 Cohen's d | F5 cond(p10) | F6 max `\\|\\|dev\\|\\|/\\|\\|Ω\\|\\|^2` |")
    lines.append("|-------|------------------------------------|------------------------|--------------|--------------|----------------|---------------------------|")

    for model_name, path in TWENTY_STEP_RESULTS.items():
        m = _load_20step_metrics(path)
        if m is None:
            lines.append(f"| {model_name} | *artifact missing* | | | | | |")
            continue
        lines.append(
            f"| {model_name} "
            f"| {_fmt(m.get('f1_median', 0))} "
            f"| {_fmt_fixed(m.get('f3_median', 0))} "
            f"| {_fmt(m.get('f4_max_drift', 0))} "
            f"| {_fmt(m.get('f2_cohens_d', 0))} "
            f"| {_fmt(m.get('f5_cond_p10', 0))} "
            f"| {_fmt(m.get('f6_max_dev', 0))} |"
        )

    # ── 200-step multiseed table ──
    if MULTISEED_SUMMARY.exists():
        data = json.loads(MULTISEED_SUMMARY.read_text())
        models = data.get("models", {})
        n_seeds = data.get("n_seeds_per_model", "?")

        lines.append("")
        lines.append(f"Cross-family trajectory falsification (200-step, {n_seeds} seeds, 95% t-interval):")
        lines.append("")
        lines.append("| Model | F1 median (mean ± CI) | F3 median (mean ± CI) | F4 max drift (mean ± CI) | F2 Cohen's d (mean ± CI) | F5 cond_p10 (mean ± CI) | F6 max dev (mean ± CI) |")
        lines.append("|-------|----------------------|----------------------|-------------------------|--------------------------|------------------------|------------------------|")

        for model_name, stats in models.items():
            def _cell(key: str) -> str:
                s = stats.get(key, {})
                m = s.get("mean", 0)
                hw = s.get("ci95_halfwidth", 0)
                return f"`{m:.4e} ± {hw:.2e}`"

            lines.append(
                f"| {model_name} "
                f"| {_cell('f1_median')} "
                f"| {_cell('f3_median')} "
                f"| {_cell('f4_max_drift')} "
                f"| {_cell('f2_cohens_d')} "
                f"| {_cell('f5_cond_p10')} "
                f"| {_cell('f6_max_dev_over_norm_sq')} |"
            )

        # Per-seed detail table
        per_run = data.get("per_run", [])
        if per_run:
            lines.append("")
            lines.append("Per-seed results:")
            lines.append("")
            lines.append("| Model | Seed | F1 median | F2 d | F2 result | F4 max | F5 cond_p10 |")
            lines.append("|-------|------|-----------|------|-----------|--------|-------------|")
            for run in per_run:
                lines.append(
                    f"| {run['model']} "
                    f"| {run['seed']} "
                    f"| {_fmt(run.get('f1_median', 0))} "
                    f"| {run.get('f2_cohens_d', 0):.4f} "
                    f"| {run.get('f2_result', '?')} "
                    f"| {_fmt(run.get('f4_max_drift', 0))} "
                    f"| {_fmt(run.get('f5_cond_p10', 0))} |"
                )

    lines.append("")
    return "\n".join(lines)


def update_doc(check_only: bool = False) -> bool:
    """Update or check D-6 tables in INDUSTRY-DIVERGENCES.md.

    Returns True if tables are up-to-date, False if stale (check mode) or
    if update was performed (write mode).
    """
    if not DOC_PATH.exists():
        print(f"ERROR: {DOC_PATH} not found", file=sys.stderr)
        return False

    content = DOC_PATH.read_text()

    if BEGIN_SENTINEL not in content or END_SENTINEL not in content:
        print(f"ERROR: Sentinel comments not found in {DOC_PATH}", file=sys.stderr)
        print(f"  Add '{BEGIN_SENTINEL}' and '{END_SENTINEL}' around the D-6 evidence tables.", file=sys.stderr)
        return False

    before = content[:content.index(BEGIN_SENTINEL) + len(BEGIN_SENTINEL)]
    after = content[content.index(END_SENTINEL):]

    new_tables = generate_tables()
    new_content = before + new_tables + "\n" + after

    if check_only:
        if new_content == content:
            print("D-6 tables are up-to-date.")
            return True
        else:
            print("D-6 tables are STALE. Run: python scripts/generate_d6_tables.py")
            return False
    else:
        DOC_PATH.write_text(new_content)
        print(f"Updated D-6 tables in {DOC_PATH}")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate D-6 evidence tables from artifacts.")
    parser.add_argument("--check", action="store_true", help="Check if tables are up-to-date (exit 1 if stale)")
    args = parser.parse_args()

    ok = update_doc(check_only=args.check)
    if args.check and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
