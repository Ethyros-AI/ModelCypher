#!/usr/bin/env python3
"""Report tracked text files that exceed a token budget."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tiktoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List tracked UTF-8 files above a token threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20_000,
        help="Token threshold (default: 20000).",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name (default: cl100k_base).",
    )
    parser.add_argument(
        "--path-prefix",
        default=None,
        help="Optional prefix filter (for example: src/ or docs/).",
    )
    return parser.parse_args()


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    result: list[Path] = []
    for rel in output.splitlines():
        path = Path(rel)
        if path.is_file():
            result.append(path)
    return result


def main() -> int:
    args = parse_args()
    encoder = tiktoken.get_encoding(args.encoding)

    rows: list[tuple[int, str]] = []
    for path in tracked_files():
        rel = path.as_posix()
        if args.path_prefix and not rel.startswith(args.path_prefix):
            continue

        raw = path.read_bytes()
        if b"\x00" in raw:
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        tokens = len(encoder.encode(text, disallowed_special=()))
        if tokens > args.threshold:
            rows.append((tokens, rel))

    rows.sort(reverse=True)
    for tokens, rel in rows:
        print(f"{tokens:8d} {rel}")
    print(f"COUNT {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
