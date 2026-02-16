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

"""Dataset loading for LoRA training.

Reads JSONL format where each line contains at least ``{"text": "..."}``.
Tokenization happens in adapter code; this module is pure Python.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset from disk.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of parsed sample dictionaries containing at least a ``text`` key.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If no valid ``{"text": ...}`` samples are found.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict) and "text" in payload:
                samples.append(payload)

    if not samples:
        raise ValueError(f"No valid samples found in {dataset_path}")

    logger.info("Loaded %d samples from %s", len(samples), dataset_path)
    return samples


_PAIRED_FIELDS = {"logic_id", "answer_start", "template_id"}


def is_paired_dataset(samples: list[dict[str, Any]]) -> bool:
    """Check if ALL samples have paired training fields.

    Returns True only when every sample contains logic_id, answer_start,
    and template_id. Logs a warning if some (but not all) samples are missing
    required fields.
    """
    if not samples:
        return False
    missing_count = 0
    for s in samples:
        if not _PAIRED_FIELDS.issubset(s):
            missing_count += 1
    if missing_count == 0:
        return True
    if missing_count < len(samples):
        logger.warning(
            "Paired-data check: %d/%d samples missing required fields %s",
            missing_count, len(samples), _PAIRED_FIELDS,
        )
    return False


def build_pair_groups(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Group sample indices by logic_id and template_id for pair construction.

    Raises:
        ValueError: If any sample is missing logic_id or template_id.

    Returns:
        (logic_groups, template_groups) where each maps ID -> list of sample indices.
        Samples sharing a logic_id are invariance partners.
        Samples sharing a template_id are counterfactual partners.
    """
    logic_groups: dict[str, list[int]] = {}
    template_groups: dict[str, list[int]] = {}

    for i, s in enumerate(samples):
        lid = s.get("logic_id")
        tid = s.get("template_id")
        if lid is None or tid is None:
            raise ValueError(
                f"Sample {i} missing required field(s): "
                f"logic_id={'present' if lid is not None else 'MISSING'}, "
                f"template_id={'present' if tid is not None else 'MISSING'}"
            )
        logic_groups.setdefault(lid, []).append(i)
        template_groups.setdefault(tid, []).append(i)

    return logic_groups, template_groups


__all__ = ["load_jsonl_dataset", "is_paired_dataset", "build_pair_groups"]
