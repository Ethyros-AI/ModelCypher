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

"""Utility functions for alignment experiments."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

# Package root for datasets
DATASETS_DIR = Path(__file__).parent / "datasets"


def load_harmful_prompts() -> list[str]:
    """Load harmful prompts dataset.

    Returns:
        List of harmful prompt strings
    """
    path = DATASETS_DIR / "harmful_prompts.json"
    if not path.exists():
        logger.warning("Harmful prompts dataset not found at %s", path)
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("prompts", [])


def load_jailbreak_prompts() -> list[str]:
    """Load jailbreak prompts dataset.

    Returns:
        List of jailbreak prompt strings
    """
    path = DATASETS_DIR / "jailbreak_prompts.json"
    if not path.exists():
        logger.warning("Jailbreak prompts dataset not found at %s", path)
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("prompts", [])


def load_harmless_prompts() -> list[str]:
    """Load harmless prompts dataset.

    Returns:
        List of harmless prompt strings
    """
    path = DATASETS_DIR / "harmless_prompts.json"
    if not path.exists():
        logger.warning("Harmless prompts dataset not found at %s", path)
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("prompts", [])


def load_contrastive_pairs() -> list[tuple[str, str]]:
    """Load contrastive pairs (harmful, harmless).

    Returns pairs of prompts where the harmful prompt should trigger
    refusal and the harmless prompt should not.

    Returns:
        List of (harmful, harmless) prompt tuples
    """
    harmful = load_harmful_prompts()
    harmless = load_harmless_prompts()

    # Pair them up
    n = min(len(harmful), len(harmless))
    return [(harmful[i], harmless[i]) for i in range(n)]


def ensure_output_dir(output_path: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        output_path: Path to output file or directory

    Returns:
        Path object with parent directory created
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def derive_separation_threshold(
    harmless: list[float],
    harmful: list[float],
) -> tuple[float, float]:
    """Derive a separation threshold from the largest cross-label gap.

    Finds the point between two distributions that maximizes separation.

    Args:
        harmless: Values for harmless class
        harmful: Values for harmful class

    Returns:
        Tuple of (threshold, gap_size)
    """
    if not harmless or not harmful:
        return 0.0, 0.0
    pairs = [(float(v), 0) for v in harmless] + [(float(v), 1) for v in harmful]
    pairs.sort(key=lambda item: item[0])
    best_gap = float("-inf")
    threshold = pairs[0][0]
    for i in range(len(pairs) - 1):
        if pairs[i][1] == pairs[i + 1][1]:
            continue
        gap = pairs[i + 1][0] - pairs[i][0]
        if gap > best_gap:
            best_gap = gap
            threshold = 0.5 * (pairs[i + 1][0] + pairs[i][0])
    if best_gap == float("-inf"):
        min_val = min(v for v, _ in pairs)
        max_val = max(v for v, _ in pairs)
        threshold = 0.5 * (min_val + max_val)
        best_gap = max_val - min_val
    if not math.isfinite(threshold):
        threshold = 0.0
    if not math.isfinite(best_gap):
        best_gap = 0.0
    return threshold, best_gap


__all__ = [
    "DATASETS_DIR",
    "derive_separation_threshold",
    "ensure_output_dir",
    "load_contrastive_pairs",
    "load_harmful_prompts",
    "load_harmless_prompts",
    "load_jailbreak_prompts",
]
