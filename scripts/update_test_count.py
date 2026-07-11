#!/usr/bin/env python3
"""Update the README test-count block from pytest collection output."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
START_MARKER = "<!-- TEST-COUNT:START -->"
END_MARKER = "<!-- TEST-COUNT:END -->"
COUNT_RE = re.compile(r"^(?P<count>\d+) tests collected in ", re.MULTILINE)


def collect_test_count() -> int:
    """Return pytest's collected-test count without running tests."""
    env = os.environ.copy()
    env.pop("MC_DISABLE_MLX", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout
    match = COUNT_RE.search(output)
    if result.returncode != 0 or match is None:
        raise RuntimeError(
            "Could not collect tests with pytest --collect-only -q.\n\n"
            f"Exit code: {result.returncode}\n{output}"
        )
    return int(match.group("count"))


def render_block(count: int) -> str:
    formatted = f"{count:,}"
    return (
        f"{START_MARKER}\n"
        f"{formatted} collected tests. This count is generated from "
        "`pytest --collect-only`; refresh it with "
        "`poetry run python scripts/update_test_count.py --write`.\n"
        f"{END_MARKER}"
    )


def replace_block(text: str, count: int) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("README.md is missing the test-count marker block.")
    end += len(END_MARKER)
    return text[:start] + render_block(count) + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Update README.md.")
    parser.add_argument("--check", action="store_true", help="Fail if README.md is stale.")
    args = parser.parse_args()

    count = collect_test_count()
    current = README_PATH.read_text()
    updated = replace_block(current, count)

    if args.write:
        README_PATH.write_text(updated)
        print(f"README test count updated to {count:,}.")
        return 0

    if args.check and current != updated:
        print(
            "README test count is stale. Run "
            "`poetry run python scripts/update_test_count.py --write`.",
            file=sys.stderr,
        )
        return 1

    print(f"{count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
