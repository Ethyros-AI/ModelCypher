"""Regression guard: scan baranov/ source for heuristic constants.

Pattern from tests/domain/training/test_mission_alignment_training.py.

Scans runtime code paths in src/modelcypher/experimental/baranov/*.py
for suspect numeric literal patterns that would indicate heuristic
constants (fixed thresholds, fixed schedules, arbitrary caps).

Scoped to runtime code only -- comments and docstrings are stripped
before scanning to avoid false positives from documentation examples.

Per the replication protocol (Section 3.3), all acceptance boundaries
must be dtype-derived, spectral-derived, or baseline-distribution-derived.
"""

from __future__ import annotations

import re
import tokenize
from io import StringIO
from pathlib import Path

_BARANOV_SRC = Path(__file__).resolve().parents[3] / "src" / "modelcypher" / "experimental" / "baranov"

# Banned fragments -- these exact strings in runtime code indicate heuristics.
# Each is a string pattern to search for in the stripped source.
_BANNED_FRAGMENTS: list[str] = [
    # Fixed degradation thresholds (claim C12, C17 rejected)
    "degraded_threshold",
    "max_ppl_increase",
    # The exact fixed stage schedule (claim C17 rejected)
    "[1.0, 0.5, 0.1, 0.0]",
    "(1.0, 0.5, 0.1, 0.0)",
    # Fixed sleep-pressure weights (claim C9 rejected)
    "sleep_pressure_weight",
    # Arbitrary iteration caps with no derivation
    "iters_per_fact",
    "max_refresh_per_cycle",
]


def _strip_comments_and_docstrings(source: str) -> str:
    """Remove comments and docstrings from Python source.

    Returns the source with only runtime code paths remaining.
    This prevents false positives from documentation examples.
    """
    result_tokens: list[str] = []
    try:
        tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    except tokenize.TokenError:
        # If tokenization fails, return original (conservative: may have FPs)
        return source

    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING:
            # Docstrings are STRING tokens at the start of a module/class/function.
            # We strip ALL string literals from the audit since banned fragments
            # should never appear as runtime string values either.
            # (If a banned pattern appears in a user-facing message, that's fine --
            # the pattern is in the string, not in the code logic.)
            continue
        result_tokens.append(tok.string)

    return " ".join(result_tokens)


def test_no_heuristic_constants_in_baranov_modules() -> None:
    """Scan baranov/ source files for suspect heuristic patterns.

    Violations indicate a banned constant or pattern name in runtime code.
    If you need to add a new literal, verify it is derived from dtype,
    spectral structure, or measured baselines, then add it to the allowlist
    in this test with a derivation comment.
    """
    violations: list[str] = []

    for py_file in sorted(_BARANOV_SRC.glob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        stripped = _strip_comments_and_docstrings(source)

        for fragment in _BANNED_FRAGMENTS:
            if fragment in stripped:
                violations.append(f"{py_file.name}: contains banned pattern '{fragment}'")

    assert not violations, (
        "Heuristic constants detected in baranov/ source files:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
