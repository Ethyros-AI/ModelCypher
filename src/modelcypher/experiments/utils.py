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


__all__ = [
    "DATASETS_DIR",
    "ensure_output_dir",
    "load_contrastive_pairs",
    "load_harmful_prompts",
    "load_harmless_prompts",
]
